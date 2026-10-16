"""Synchronous persistence for Track 2 write-through.

Why sync instead of the async engine? The orchestrator worker thread has no
asyncio loop, and chat handling also runs outside request handlers. These are
tiny single-row writes; a short blocking call is acceptable and keeps ONE code
path for worker thread, WS loop, and tests.

Same SQLite file as the async engine (WAL mode + 30s busy timeout, NullPool so
no connection is ever shared across threads). When ``DATABASE_URL`` is not
SQLite there is no bundled sync driver by design (no psycopg dependency) — the
module disables itself with a warning and every helper becomes a documented
no-op returning ``None``/``[]``/``{}``. Local-first, honest degradation.
"""

from __future__ import annotations

import copy
import time
import warnings
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from backend.config import settings

try:
    from sqlalchemy import create_engine, desc, select
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import NullPool

    _SA = True
except ImportError:  # pragma: no cover
    _SA = False

from backend.database.models import Base, ChatMessage, ProjectImport, ProjectMemory, Run, RunEvent

_engine = None
_Session = None
_enabled = False


def _init() -> None:
    global _engine, _Session, _enabled
    if _engine is not None or not _SA:
        return
    url = settings.database_url
    if not url.startswith("sqlite"):
        warnings.warn("runlog disabled: DATABASE_URL is not SQLite (no sync driver bundled)")
        return
    sync_url = url.replace("+aiosqlite", "")
    _engine = create_engine(
        sync_url,
        connect_args={"check_same_thread": False, "timeout": 30},
        poolclass=NullPool,
    )
    try:
        with _engine.begin() as conn:
            conn.exec_driver_sql("PRAGMA journal_mode=WAL")
            conn.exec_driver_sql("PRAGMA busy_timeout=30000")
        Base.metadata.create_all(_engine)  # idempotent; covers fresh + existing DBs
    except Exception as exc:
        warnings.warn(f"runlog disabled: {exc}")
        _engine = None
        return
    _Session = sessionmaker(bind=_engine, expire_on_commit=False)
    _enabled = True


def enabled() -> bool:
    _init()
    return _enabled and _Session is not None


def _uuid(value: Any) -> UUID | None:
    if value is None or value == "":
        return None
    return value if isinstance(value, UUID) else UUID(str(value))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


_RETRIES = 4
_BACKOFF = (0.02, 0.1, 0.35, 0.8)


def _is_contention(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "locked" in text or "busy" in text or "timeout" in text


def _with_retry(operation: str, fn):
    """Run a small write with bounded retry.

    The sync engine (this module) and the async app engine share one SQLite
    file, so a write can transiently hit "database is locked". Persisted events,
    chat turns and provenance must never be silently dropped, so contention is
    retried; any other error keeps the previous warn-and-continue behaviour.
    """
    last: BaseException | None = None
    for attempt in range(_RETRIES):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if _is_contention(exc) and attempt < _RETRIES - 1:
                time.sleep(_BACKOFF[attempt])
                continue
            warnings.warn(f"runlog.{operation} failed: {exc}")
            return None
    warnings.warn(f"runlog.{operation} failed: {last}")
    return None


# ------------------------------------------------------------------ runs

def save_run(run_id: str, project_id: Any, session_id: Any, config: dict) -> dict | None:
    """Insert the run record at campaign start. Best-effort: warns, never raises."""
    if not enabled():
        return None

    def _op():
        with _Session() as db:
            row = Run(
                id=run_id,
                project_id=_uuid(project_id),
                session_id=_uuid(session_id),
                status="running",
                stage="STARTING",
                progress=0.0,
                config=dict(config or {}),
                started_at=_utcnow(),
            )
            db.add(row)
            db.commit()
            return _run_to_dict(row)

    return _with_retry("save_run", _op)


def update_run(run_id: str, **fields: Any) -> dict | None:
    if not enabled() or not run_id:
        return None
    allowed = {
        "status", "stage", "progress", "eta_seconds", "config", "dataset",
        "best", "feature_history", "research", "error", "completed_at",
    }
    try:
        with _Session() as db:
            row = db.get(Run, run_id)
            if row is None:
                return None
            for key, value in fields.items():
                if key in allowed:
                    setattr(row, key, value)
            db.commit()
            return _run_to_dict(row)
    except Exception as exc:
        warnings.warn(f"runlog.update_run failed: {exc}")
        return None


def get_run(run_id: str) -> dict | None:
    if not enabled() or not run_id:
        return None
    try:
        with _Session() as db:
            row = db.get(Run, run_id)
            return _run_to_dict(row) if row else None
    except Exception as exc:
        warnings.warn(f"runlog.get_run failed: {exc}")
        return None


def list_runs(
    project_id: Any = None, status: str | None = None, limit: int = 50, offset: int = 0
) -> list[dict]:
    if not enabled():
        return []
    try:
        with _Session() as db:
            q = select(Run).order_by(desc(Run.created_at)).offset(offset).limit(limit)
            if project_id:
                q = q.where(Run.project_id == _uuid(project_id))
            if status:
                q = q.where(Run.status == status)
            return [_run_to_dict(r) for r in db.execute(q).scalars().all()]
    except Exception as exc:
        warnings.warn(f"runlog.list_runs failed: {exc}")
        return []


def _run_to_dict(row: Run) -> dict:
    return {
        "run_id": row.id,
        "project_id": str(row.project_id) if row.project_id else None,
        "session_id": str(row.session_id) if row.session_id else None,
        "status": row.status,
        "stage": row.stage,
        "progress": row.progress,
        "eta_seconds": row.eta_seconds,
        "config": row.config or {},
        "dataset": row.dataset,
        "best": row.best,
        "feature_history": row.feature_history or [],
        "research": row.research or [],
        "error": row.error,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "completed_at": row.completed_at.isoformat() if row.completed_at else None,
    }


# ------------------------------------------------------------------ events

def log_event(
    run_id: str | None,
    event_type: str,
    message: str | None = None,
    status: str | None = None,
    stage: str | None = None,
    progress: float | None = None,
    duration_ms: float | None = None,
    metadata: dict | None = None,
) -> None:
    """Append one operational event. Events without a run_id cannot be
    attributed and are skipped (never faked)."""
    if not enabled() or not run_id:
        return

    def _op():
        with _Session() as db:
            db.add(
                RunEvent(
                    run_id=run_id,
                    ts=_utcnow(),
                    event_type=event_type,
                    status=status,
                    message=(message or "")[:2000],
                    progress=progress,
                    stage=stage,
                    duration_ms=duration_ms,
                    metadata_=dict(metadata or {}),
                )
            )
            db.commit()

    _with_retry("log_event", _op)


def list_events(run_id: str, limit: int = 1000) -> list[dict]:
    if not enabled() or not run_id:
        return []
    try:
        with _Session() as db:
            q = (
                select(RunEvent)
                .where(RunEvent.run_id == run_id)
                .order_by(RunEvent.ts)
                .limit(limit)
            )
            out = []
            for r in db.execute(q).scalars().all():
                out.append(
                    {
                        "ts": r.ts.isoformat() if r.ts else None,
                        "event_type": r.event_type,
                        "status": r.status,
                        "message": r.message,
                        "progress": r.progress,
                        "stage": r.stage,
                        "duration_ms": r.duration_ms,
                        "metadata": r.metadata_ or {},
                    }
                )
            return out
    except Exception as exc:
        warnings.warn(f"runlog.list_events failed: {exc}")
        return []


# ------------------------------------------------------------------ chat

def save_chat(
    project_id: Any, session_id: Any, role: str, text: str, agent: str | None = None
) -> None:
    if not enabled() or not text:
        return

    def _op():
        with _Session() as db:
            db.add(
                ChatMessage(
                    project_id=_uuid(project_id),
                    session_id=_uuid(session_id),
                    role=role,
                    agent=agent,
                    text=text[:20000],
                )
            )
            db.commit()

    _with_retry("save_chat", _op)


def list_chat(
    project_id: Any = None, session_id: Any = None, limit: int = 200
) -> list[dict]:
    if not enabled():
        return []
    try:
        with _Session() as db:
            q = select(ChatMessage).order_by(ChatMessage.created_at).limit(limit)
            if session_id:
                q = q.where(ChatMessage.session_id == _uuid(session_id))
            elif project_id:
                q = q.where(ChatMessage.project_id == _uuid(project_id))
            return [
                {
                    "role": r.role,
                    "agent": r.agent,
                    "text": r.text,
                    "ts": r.created_at.isoformat() if r.created_at else None,
                }
                for r in db.execute(q).scalars().all()
            ]
    except Exception as exc:
        warnings.warn(f"runlog.list_chat failed: {exc}")
        return []


def clear_chat(session_id: Any) -> int:
    """Delete one session's messages. Requires a session (never wipes blindly)."""
    if not enabled() or not session_id:
        return 0
    try:
        with _Session() as db:
            q = select(ChatMessage).where(ChatMessage.session_id == _uuid(session_id))
            rows = db.execute(q).scalars().all()
            n = len(rows)
            for r in rows:
                db.delete(r)
            db.commit()
            return n
    except Exception as exc:
        warnings.warn(f"runlog.clear_chat failed: {exc}")
        return 0


# ------------------------------------------------------------------ memory

_MEMORY_DEFAULT: dict[str, Any] = {
    "goal": "",
    "dataset": {},
    "target": None,
    "task_type": "auto",
    "constraints": [],
    "preferences": {},
    "decisions": [],
    "rejected": [],
    "feature_decisions": [],
    "model_decisions": [],
    "research": [],
    "best": None,
    "warnings": [],
    "excluded_features": [],
    "pending_features": [],
    "pending_model": None,
    "pending_config": None,
    "chat_digest": {},
}


def _memory_base() -> dict[str, Any]:
    """Deep copy of the default document.

    A shallow ``dict(...)`` copy would share the list values with the module
    default, so writing one project's memory would mutate the template and leak
    that project's decisions into every other project. Always deep-copy.
    """
    return copy.deepcopy(_MEMORY_DEFAULT)


def get_memory(project_id: Any) -> dict:
    if not enabled() or not project_id:
        return _memory_base()
    try:
        with _Session() as db:
            row = db.get(ProjectMemory, _uuid(project_id))
            if row is None or not row.data:
                return _memory_base()
            merged = _memory_base()
            merged.update(copy.deepcopy(dict(row.data)))
            return merged
    except Exception as exc:
        warnings.warn(f"runlog.get_memory failed: {exc}")
        return _memory_base()


def remember(project_id: Any, section: str, payload: Any) -> dict:
    """Upsert one memory section. List sections append (deduped); scalars set.

    ``copy.deepcopy`` on read is mandatory, not cosmetic: mutating a nested list
    that is still referenced by the ORM instance makes SQLAlchemy's JSON change
    detection see no delta, so the UPDATE is silently skipped and the write is
    lost. The regression test ``test_memory_appends_persist`` guards this.
    """
    if not enabled() or not project_id:
        return _memory_base()

    def _op():
        with _Session() as db:
            pid = _uuid(project_id)
            row = db.get(ProjectMemory, pid)
            data = copy.deepcopy(dict(row.data)) if row and row.data else _memory_base()
            current = data.get(section)
            if isinstance(current, list):
                items = payload if isinstance(payload, list) else [payload]
                for item in items:
                    if item not in current:
                        current.append(item)
                data[section] = current[-100:]
            else:
                data[section] = payload
            if row is None:
                db.add(ProjectMemory(project_id=pid, data=data))
            else:
                row.data = data
            db.commit()
            return data

    return _with_retry("remember", _op) or _memory_base()


# ------------------------------------------------------------------ imports

def record_import(
    target_project_id: Any,
    resource_type: str,
    resource_ref: dict,
    mode: str,
    verdict: str,
    verdict_reasons: list | None = None,
    source_project_id: Any = None,
    source_run_id: str | None = None,
    source_experiment_id: Any = None,
    adapted: bool = False,
) -> dict | None:
    if not enabled():
        return None

    def _op():
        with _Session() as db:
            row = ProjectImport(
                target_project_id=_uuid(target_project_id),
                source_project_id=_uuid(source_project_id),
                source_run_id=source_run_id,
                source_experiment_id=_uuid(source_experiment_id),
                resource_type=resource_type,
                resource_ref=dict(resource_ref or {}),
                mode=mode,
                verdict=verdict,
                verdict_reasons=list(verdict_reasons or []),
                adapted=adapted,
            )
            db.add(row)
            db.commit()
            return {
                "id": str(row.id),
                "target_project_id": str(row.target_project_id),
                "resource_type": row.resource_type,
                "verdict": row.verdict,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }

    return _with_retry("record_import", _op)


def list_imports(target_project_id: Any) -> list[dict]:
    if not enabled() or not target_project_id:
        return []
    try:
        with _Session() as db:
            q = (
                select(ProjectImport)
                .where(ProjectImport.target_project_id == _uuid(target_project_id))
                .order_by(desc(ProjectImport.created_at))
            )
            out = []
            for r in db.execute(q).scalars().all():
                out.append(
                    {
                        "id": str(r.id),
                        "source_project_id": str(r.source_project_id) if r.source_project_id else None,
                        "source_run_id": r.source_run_id,
                        "resource_type": r.resource_type,
                        "resource_ref": r.resource_ref or {},
                        "mode": r.mode,
                        "verdict": r.verdict,
                        "verdict_reasons": r.verdict_reasons or [],
                        "adapted": r.adapted,
                        "created_at": r.created_at.isoformat() if r.created_at else None,
                    }
                )
            return out
    except Exception as exc:
        warnings.warn(f"runlog.list_imports failed: {exc}")
        return []


# ------------------------------------------------------------------ compaction

_COMPACT_AT = 300
_COMPACT_KEEP = 100


def count_chat(session_id: Any) -> int:
    if not enabled() or not session_id:
        return 0
    try:
        from sqlalchemy import func as _func
        from sqlalchemy import select as _select

        with _Session() as db:
            return int(
                db.scalar(
                    _select(_func.count())
                    .select_from(ChatMessage)
                    .where(ChatMessage.session_id == _uuid(session_id))
                )
                or 0
            )
    except Exception as exc:
        warnings.warn(f"runlog.count_chat failed: {exc}")
        return 0


def compact_session(session_id: Any, project_id: Any, keep: int = _COMPACT_KEEP) -> dict | None:
    """Rule-based compaction (no LLM): when a session exceeds 300 turns, digest
    the oldest into project memory (counts, span, user-topic samples, referenced
    decisions) and delete them, keeping the latest ``keep`` turns verbatim.

    The digest preserves continuity without resending history: future context =
    digest + recent turns only.
    """
    if not enabled() or not session_id:
        return None
    try:
        with _Session() as db:
            from sqlalchemy import select as _select

            q = (
                _select(ChatMessage)
                .where(ChatMessage.session_id == _uuid(session_id))
                .order_by(ChatMessage.created_at)
            )
            rows = db.execute(q).scalars().all()
            if len(rows) <= _COMPACT_AT:
                return None
            drop, kept = rows[: len(rows) - keep], rows[len(rows) - keep :]
            roles: dict[str, int] = {}
            users: list[str] = []
            for r in drop:
                roles[r.role] = roles.get(r.role, 0) + 1
                if r.role == "user" and len(users) < 6:
                    users.append(r.text[:200])
            digest = {
                "dropped_turns": len(drop),
                "kept_turns": len(kept),
                "roles": roles,
                "span": [
                    drop[0].created_at.isoformat() if drop[0].created_at else None,
                    drop[-1].created_at.isoformat() if drop[-1].created_at else None,
                ],
                "user_samples": users,
            }
            for r in drop:
                db.delete(r)
            db.commit()
        if project_id:
            remember(project_id, "chat_digest", digest)
        return digest
    except Exception as exc:
        warnings.warn(f"runlog.compact_session failed: {exc}")
        return None
