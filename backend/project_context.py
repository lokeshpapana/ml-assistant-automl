"""Phase B bridge: project/run context validation shared by REST and WebSocket entries.

Minimum link between Track 1 (persistent ``Project``/``Session`` rows) and Track 2
(in-memory orchestrator runs). This module VALIDATES ONLY — no writes, no
persistence, no state changes. A run never starts when validation fails.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.models import Project, Session


def extract_run_context(message: dict[str, Any] | None) -> dict[str, Any]:
    """Pull ``project_id``/``session_id`` out of a start payload.

    Accepts both shapes (top-level wins, config-nested is the fallback) so old
    flat-config bodies keep working byte-for-byte:

    * ``{"project_id": "<uuid>", "config": {...}}``
    * ``{"project_id": "<uuid>", "goal": "...", ...}``  (the body IS the config)

    Returns ``{"project_id": str | None, "session_id": str | None, "config": dict}``.
    """
    body = message or {}
    config = body.get("config")
    if not isinstance(config, dict):
        config = {k: v for k, v in body.items() if k not in ("project_id", "session_id")}
    project_id = body.get("project_id", config.get("project_id"))
    session_id = body.get("session_id", config.get("session_id"))
    return {
        "project_id": str(project_id) if project_id else None,
        "session_id": str(session_id) if session_id else None,
        "config": config,
    }


async def validate_run_context(
    project_id: str | None, session_id: str | None, db: AsyncSession
) -> dict[str, str | None]:
    """Validate ids against Track 1 rows. Raises ``HTTPException``, never starts runs.

    - malformed ``project_id`` → 400; unknown project → 404 (no orphaned runs).
    - ``session_id`` given: malformed → 400; unknown → 404; belonging to a
      different project → 400 (no cross-project confusion).

    Returns normalized ``{"project_id": ..., "session_id": ...}`` (either None).
    ``project_id`` is optional: ad-hoc runs without a project (e.g. classic
    ``/ui``) keep working with ``None`` context.
    """
    pid = _parse_uuid(project_id, "project_id") if project_id else None
    sid = _parse_uuid(session_id, "session_id") if session_id else None
    if pid is not None:
        project = await db.get(Project, pid)
        if project is None:
            raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    if sid is not None:
        session = await db.get(Session, sid)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        if pid is not None and session.project_id != pid:
            raise HTTPException(
                status_code=400,
                detail="Session does not belong to the given project",
            )
    return {
        "project_id": str(pid) if pid is not None else None,
        "session_id": str(sid) if sid is not None else None,
    }


def _parse_uuid(value: str, field: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=400, detail=f"Invalid {field}: {value!r} (expected UUID)") from None
