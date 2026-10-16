"""Agent 3: MCP Research & Feature Engineering Agent.

Constructs paper search queries from the dataset columns and user topic, calls the
arXiv / Semantic Scholar adapter (or the offline paper bank), extracts concrete
techniques, and safely engineers new features directly into the single central
store table. Every applied feature is citation-backed.
"""

from __future__ import annotations

from typing import Any, Callable

from backend.agents.common import Requirements
from backend.central_store import CentralStore
from backend.research import ResearchClient, search_papers


class ResearchAgent:
    def __init__(self, client: ResearchClient | None = None):
        self.client = client or ResearchClient()

    def run(
        self,
        store: CentralStore,
        requirements: Requirements,
        emit: Callable[[dict[str, Any]], None],
        max_features: int = 6,
    ) -> list[dict[str, Any]]:
        emit({"type": "agent_message", "agent": "research", "text": "Searching papers for feature-engineering techniques…"})

        columns = store.columns()
        dtypes = store.dtypes()
        recommendations = self.client.get_recommendations(
            columns=columns,
            dtypes=dtypes,
            task_type=requirements.task_type,
            goal=requirements.goal,
            max_results=max_features,
        )

        live_papers = recommendations.get("live_papers", [])
        offline_cards = recommendations.get("offline_cards", [])
        cards = []
        if live_papers:
            cards.extend(
                [
                    {
                        "paper_id": p.get("paper_id", ""),
                        "title": p.get("title", ""),
                        "abstract": p.get("abstract", "")[:300],
                        "source": p.get("source", "arxiv"),
                        "url": p.get("url"),
                    }
                    for p in live_papers
                ]
            )
        else:
            cards.extend(
                [
                    {
                        "paper_id": c.get("paper_id", ""),
                        "title": c.get("title", ""),
                        "abstract": c.get("summary", ""),
                        "source": "offline-bank",
                        "url": f"https://arxiv.org/abs/{c.get('paper_id', '')}",
                    }
                    for c in offline_cards
                ]
            )

        applied: list[dict[str, Any]] = []
        for feat in recommendations.get("features", []):
            if store.shape()[1] >= 120:
                break
            try:
                applied_feat = store.add_feature(feat)
                applied.append(applied_feat)
                emit(
                    {
                        "type": "research_card",
                        "card": {
                            "paper_id": feat.get("paper_id", ""),
                            "title": feat.get("paper_title", ""),
                            "feature": applied_feat["name"],
                            "technique": feat.get("technique", ""),
                            "description": feat.get("description", ""),
                        },
                    }
                )
                emit(
                    {
                        "type": "agent_message",
                        "agent": "research",
                        "text": f"Applied technique from paper {feat.get('paper_id', '?')}: added feature '{applied_feat['name']}'.",
                    }
                )
            except Exception as exc:  # keep the loop alive for any single bad spec
                emit({"type": "agent_message", "agent": "research", "text": f"Skipped feature '{feat.get('name')}': {exc}"})

        if not applied:
            emit(
                {
                    "type": "agent_message",
                    "agent": "research",
                    "text": "No new numeric features could be engineered from the current columns.",
                }
            )
        emit({"type": "dataset_preview", "event": "dataset_preview", "data": store.preview(limit=25)})
        return applied


def mcp_search_papers(query: str, domain: str = "") -> list[dict[str, Any]]:
    """Small MCP-style adapter so agent tools behave like MCP tools."""
    return search_papers(query, domain=domain, max_results=5)