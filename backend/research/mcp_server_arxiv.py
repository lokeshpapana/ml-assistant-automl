"""Minimal MCP-compatible paper-researcher server (stdio JSON-RPC).

Runs as ``python -m backend.research.mcp_server_arxiv`` and exposes two tools:

* ``search_papers(query, domain, max_results)``
* ``fetch_paper_content(paper_id)``

Uses the same HTTP adapters as the in-process agent -- no extra dependencies and
no cloud requirement (offline paper bank fills gaps when the network is absent).
"""

from __future__ import annotations

import json
import sys
from typing import Any

from .arxiv_client import fetch_paper_content, search_papers

TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_papers",
        "description": "Search arXiv & Semantic Scholar for research papers on a query/domain.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "domain": {"type": "string", "description": "Optional domain hint"},
                "max_results": {"type": "integer", "default": 5},
            },
        },
    },
    {
        "name": "fetch_paper_content",
        "description": "Fetch abstract + metadata for a single paper by arxiv id.",
        "inputSchema": {
            "type": "object",
            "properties": {"paper_id": {"type": "string"}},
            "required": ["paper_id"],
        },
    },
]


def _tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "search_papers":
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        search_papers(
                            arguments.get("query", ""),
                            domain=arguments.get("domain", ""),
                            max_results=int(arguments.get("max_results", 5)),
                        ),
                        default=str,
                    ),
                }
            ]
        }
    if name == "fetch_paper_content":
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(fetch_paper_content(arguments.get("paper_id", "")), default=str),
                }
            ]
        }
    return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}], "isError": True}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        req_id = request.get("id")
        method = request.get("method", "")

        if method == "initialize":
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "paper-researcher", "version": "0.1.0"},
                },
            }
        elif method == "tools/list":
            response = {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}
        elif method == "tools/call":
            params = request.get("params", {})
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": _tool_call(params.get("name", ""), params.get("arguments", {})),
            }
        elif method in ("notifications/initialized", "shutdown"):
            response = {"jsonrpc": "2.0", "id": req_id, "result": {}}
        else:
            response = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unsupported method: {method}"}}

        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()