"""HTTP client for arXiv + Semantic Scholar (no MCP daemon required).

Implements the MCP-style agent tools ``search_papers(query, domain)`` and
``fetch_paper_content(paper_id)`` as plain functions. All network calls are
guarded and time-boxed; when offline we fall back to the curated offline bank,
so the research agent is always able to produce citation-backed suggestions.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import httpx

from backend.config import settings

ARXIV_API = "http://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"


def _parse_arxiv_feed(xml_text: str) -> list[dict[str, Any]]:
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall(f"{_ATOM}entry"):
        paper_id = ""
        for link in entry.findall(f"{_ATOM}link"):
            if link.get("rel") == "alternate":
                paper_id = link.get("href", "").split("/abs/")[-1]
        title = " ".join((entry.findtext(f"{_ATOM}title") or "").split())
        summary = " ".join((entry.findtext(f"{_ATOM}summary") or "").split())
        authors = [a.findtext(f"{_ATOM}name") for a in entry.findall(f"{_ATOM}author")]
        papers.append(
            {
                "paper_id": paper_id or title[:32],
                "title": title,
                "abstract": summary,
                "authors": [a for a in authors if a],
                "source": "arxiv",
                "url": f"https://arxiv.org/abs/{paper_id}" if paper_id else None,
            }
        )
    return papers


def search_papers(query: str, domain: str = "", max_results: int = 5) -> list[dict[str, Any]]:
    """MCP tool equivalent: search arXiv & Semantic Scholar for papers."""
    papers: list[dict[str, Any]] = []
    try:
        q = f"{query} {domain}".strip()
        resp = httpx.get(
            ARXIV_API,
            params={"search_query": f"all:{q}", "max_results": max_results, "sortBy": "relevance"},
            timeout=settings.research_request_timeout,
            follow_redirects=True,
        )
        if resp.status_code == 200:
            papers.extend(_parse_arxiv_feed(resp.text))
    except Exception:
        pass
    try:
        s2 = search_semantic_scholar(
            query, api_key=settings.semantic_scholar_api_key, limit=max_results
        )
        if s2:
            papers.extend(s2)
    except Exception:
        pass
    # De-duplicate by paper_id preserving order.
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for p in papers:
        key = p["paper_id"]
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique[:max_results]


def search_semantic_scholar(query: str, api_key: str = "", limit: int = 5) -> list[dict[str, Any]]:
    headers = {"x-api-key": api_key} if api_key else {}
    try:
        resp = httpx.get(
            SEMANTIC_SCHOLAR_API,
            params={
                "query": query,
                "limit": limit,
                "fields": "title,abstract,authors,url,externalIds,corpusId",
            },
            headers=headers,
            timeout=settings.research_request_timeout,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        out = []
        for item in data.get("data", []):
            out.append(
                {
                    "paper_id": item.get("corpusId")
                    or item.get("externalIds", {}).get("ArXiv")
                    or item.get("paperId")
                    or item.get("title", "")[:32],
                    "title": item.get("title", ""),
                    "abstract": item.get("abstract") or "",
                    "authors": [a.get("name", "") for a in item.get("authors", [])],
                    "source": "semantic_scholar",
                    "url": item.get("url"),
                }
            )
        return out
    except Exception:
        return []


def fetch_paper_content(paper_id: str) -> dict[str, Any]:
    """MCP tool equivalent: fetch abstract + metadata for one paper."""
    if paper_id.startswith("http"):
        paper_id = paper_id.split("/abs/")[-1]
    try:
        resp = httpx.get(
            ARXIV_API,
            params={"id_list": paper_id, "max_results": 1},
            timeout=settings.research_request_timeout,
            follow_redirects=True,
        )
        if resp.status_code == 200:
            we: list[dict[str, Any]] = _parse_arxiv_feed(resp.text)
            if we:
                return {"content": we[0]["abstract"], **we[0]}
    except Exception:
        pass
    s2 = search_semantic_scholar(paper_id)
    if s2:
        return {"content": s2[0].get("abstract", ""), **s2[0]}
    return {"paper_id": paper_id, "title": "", "abstract": "", "content": ""}


_TECHNIQUE_KEYWORDS: dict[str, list[str]] = {
    "ratio": ["ratio", "proportion", "utilization", "per capita", "debt-to"],
    "interaction": ["interaction", "cross feature", "feature crossing", "multiplic"],
    "log1p": ["log transform", "logarithm", "log-space", "log space"],
    "quantile_bin": ["quantile", "binning", "bucketing", "discreti"],
    "freq_encode": ["frequency encoding", "count encoding", "inverse frequency"],
    "year": ["calendar", "year"],
    "month": ["calendar", "month"],
    "dayofweek": ["weekday", "day of week"],
}


def extract_technique_hints(papers: list[dict[str, Any]]) -> list[str]:
    """Map paper titles/abstracts to concrete feature 'kind' hints."""
    hints: list[str] = []
    for p in papers:
        haystack = f"{p.get('title', '')} {p.get('abstract', '')}".lower()
        for kind, keywords in _TECHNIQUE_KEYWORDS.items():
            if kind in hints:
                continue
            if any(kw in haystack for kw in keywords):
                hints.append(kind)
    return hints


class ResearchClient:
    """Research agent's client: live papers + offline fallback recommendations."""

    def __init__(self) -> None:
        self.enabled_offline = settings.enable_offline_paper_bank

    def get_recommendations(
        self,
        columns: list[str],
        dtypes: dict[str, str],
        task_type: str,
        goal: str = "",
        max_results: int = 5,
    ) -> dict[str, Any]:
        from .paper_bank import build_offline_features, select_offline_papers

        offline_cards = select_offline_papers(columns, dtypes, task_type, goal, limit=max_results)
        offline_features = build_offline_features(columns, dtypes, goal, max_features=max_results)

        live_papers: list[dict[str, Any]] = []
        hints: list[str] = []
        query = goal or "feature engineering"
        try:
            live_papers = search_papers(f"{query} {task_type}", max_results=max_results)
        except Exception:
            live_papers = []
        if live_papers:
            hints = extract_technique_hints(live_papers)

        suggestions: list[dict[str, Any]] = []
        suggested_kinds: set[str] = set()

        # Apply live hint kinds first when a compatible offline feature exists.
        for kind in hints:
            for feat in offline_features:
                if feat["kind"] == kind and kind not in suggested_kinds:
                    suggestions.append(feat)
                    suggested_kinds.add(kind)
                    break
        # Fill the rest from the offline bank.
        for feat in offline_features:
            if len(suggestions) >= max_results:
                break
            if feat["kind"] not in suggested_kinds:
                suggestions.append(feat)
                suggested_kinds.add(feat["kind"])

        return {
            "live_papers": live_papers,
            "offline_cards": offline_cards,
            "technique_hints": hints,
            "features": suggestions,
        }
