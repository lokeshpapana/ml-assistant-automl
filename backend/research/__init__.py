"""Paper research & knowledge retrieval for the AutoML engine.

Provides an always-available offline "paper bank" of vetted, citable ML
techniques plus live arXiv / Semantic Scholar lookup over HTTP. The MCP-style
tools (``search_papers``, ``fetch_paper_content``) are exposed by
:mod:`backend.api.automl` so agents can call them with plain function calls
rather than requiring an external MCP daemon.
"""

from .arxiv_client import (
    ResearchClient,
    fetch_paper_content,
    search_papers,
    search_semantic_scholar,
)
from .paper_bank import OFFLINE_PAPER_BANK, select_offline_papers

__all__ = [
    "ResearchClient",
    "search_papers",
    "search_semantic_scholar",
    "fetch_paper_content",
    "OFFLINE_PAPER_BANK",
    "select_offline_papers",
]