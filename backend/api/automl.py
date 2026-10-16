"""REST + WebSocket API for the Interactive AutoML Multi-Agent engine.

Endpoints
---------
* ``POST /automl/start``      start a campaign (or reuse form config;
  accepts optional ``project_id`` / ``session_id``, validated, no orphans)
* ``POST /automl/stop``       set the interrupt event
* ``GET  /automl/state``      full snapshot for the UI (includes run_context)
* ``GET  /workspace``         Phase B project workspace shell (old /ui untouched)
* ``GET  /automl/leaderboard``  MLflow runs + best model
* ``GET  /automl/dataset-preview``  live single-dataset preview
* ``POST /automl/upload``     stage a CSV for the data-fetcher agent
* ``GET  /automl/export-best-model``  download .pkl artifact
* ``GET  /automl/papers``     MCP-style research tool proxy
* ``WS   /ws/automl``         bidirectional real-time channel (events carry
  project_id / session_id / run_id for per-project association)
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.database import async_session_maker, get_async_session
from backend.orchestrator import get_orchestrator
from backend.project_context import extract_run_context, validate_run_context
from backend.research import fetch_paper_content, search_papers
from backend.ws_bus import get_event_bus

router = APIRouter(tags=["automl"])

OUTPUT_DIR = Path(settings.automl_output_dir)
UPLOAD_DIR = OUTPUT_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ REST

@router.post("/automl/start")
async def start_automl(body: dict | None = None, db: AsyncSession = Depends(get_async_session)) -> dict:
    """Start a campaign. ``body`` is the run config (unchanged shape); optional
    top-level (or config-nested) ``project_id`` / ``session_id`` are validated
    against Track 1 rows first — invalid ids return 404/400 and start nothing.
    """
    ctx = extract_run_context(body)
    validated = await validate_run_context(ctx["project_id"], ctx["session_id"], db)
    config = dict(ctx["config"])
    if validated["project_id"]:
        config["project_id"] = validated["project_id"]
    if validated["session_id"]:
        config["session_id"] = validated["session_id"]
    return get_orchestrator().start(config)


@router.post("/automl/stop")
async def stop_automl() -> dict:
    get_orchestrator().request_stop()
    return {"status": "stop_signal_sent", "state": get_orchestrator().state}


@router.get("/automl/state")
async def automl_state() -> dict:
    return get_orchestrator().snapshot()


@router.get("/automl/leaderboard")
async def automl_leaderboard() -> dict:
    return get_orchestrator().leaderboard()


@router.get("/automl/dataset-preview")
async def automl_dataset_preview(limit: int = Query(25, ge=1, le=100)):
    return get_orchestrator()._safe_preview(limit=limit)


@router.get("/automl/dataset-details")
async def automl_dataset_details(project_id: str | None = None):
    """Live dataset + per-column details computed from the real working table.

    Includes the preview, declared dtypes, real column statistics (nulls,
    uniques, min/mean/max or top values), and the dataset provenance the agent
    recorded for this project (source / path / imported_snapshot_of). Every
    number comes from the actual single-source table — nothing is estimated.
    """
    orch = get_orchestrator()
    try:
        preview = orch._safe_preview(limit=50)
        stats = orch.store.column_stats()
    except Exception as exc:  # noqa: BLE001
        return {"preview": {"rows": 0, "columns": [], "dtypes": {}, "preview": [], "unavailable": str(exc)},
                "stats": {}, "provenance": None}
    provenance = None
    if project_id:
        try:
            from backend import runlog

            memory = runlog.get_memory(str(project_id))
            provenance = memory.get("dataset") or memory.get("provenance")
        except Exception:
            provenance = None
    target = orch.requirements.target_column
    if not target and project_id:
        try:
            from backend import runlog

            target = runlog.get_memory(str(project_id)).get("target")
        except Exception:
            pass
    return {
        "preview": preview,
        "stats": stats["columns"],
        "provenance": provenance,
        "target": target,
    }


@router.post("/automl/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict:
    from backend.filecheck import check_filename, check_size, dataset_info, read_tabular

    check_filename(file.filename)
    check_size(file.size, context=file.filename or "upload")
    suffix = Path(file.filename or "data.csv").suffix.lower() or ".csv"
    dest = UPLOAD_DIR / f"upload_{len(list(UPLOAD_DIR.glob('*'))) + 1}{suffix}"
    with open(dest, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        check_size(dest.stat().st_size, context=file.filename or "upload")
        info = dataset_info(read_tabular(dest))
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    orch = get_orchestrator()
    orch.upload_path = str(dest)
    return {
        "status": "uploaded",
        "path": str(dest),
        "size_bytes": dest.stat().st_size,
        "rows": info["rows"],
        "columns": info["columns"],
    }


@router.get("/automl/export-best-model")
async def export_best_model():
    from fastapi.responses import FileResponse

    orch = get_orchestrator()
    if not orch.trainer or not orch.trainer.best:
        raise HTTPException(status_code=404, detail="No trained model available yet")
    path = orch.trainer.export_best_model()
    if not path:
        raise HTTPException(status_code=404, detail="No best model artifact found")
    return FileResponse(path, filename="best_model.pkl", media_type="application/octet-stream")


@router.get("/automl/papers")
async def papers_proxy(
    query: str = Query(..., description="Search query"),
    domain: str = Query("", description="Optional domain hint"),
    max_results: int = Query(5, ge=1, le=12),
) -> dict:
    """MCP-style research tool exposed over plain HTTP for local use."""
    results = search_papers(query, domain=domain, max_results=max_results)
    return {"query": query, "domain": domain, "results": results}


@router.get("/automl/papers/{paper_id:path}")
async def paper_content_proxy(paper_id: str) -> dict:
    """MCP-style fetch_paper_content exposed over HTTP."""
    return fetch_paper_content(paper_id)


@router.get("/ui")
async def automl_ui() -> HTMLResponse:
    content = (Path(__file__).parent.parent / "static" / "automl_ui.html").read_text()
    return HTMLResponse(content)


@router.post("/automl/resume")
async def resume_automl(
    body: dict | None = None,
    db: AsyncSession = Depends(get_async_session),
) -> dict:
    """Continue a stopped run from its latest persisted checkpoint.

    ``project_id`` is validated first (404/400, nothing starts on failure); the
    orchestrator additionally refuses a run that belongs to another project.
    """
    payload = body or {}
    project_id = payload.get("project_id")
    if not project_id:
        return get_orchestrator().resume_from(
            run_id=payload.get("run_id"), session_id=payload.get("session_id")
        )
    from uuid import UUID as _UUID

    try:
        pid = str(_UUID(str(project_id)))
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=400, detail=f"Invalid project_id: {project_id!r}")
    validated = await validate_run_context(pid, None, db)
    result = get_orchestrator().resume_from(
        run_id=payload.get("run_id"),
        project_id=validated["project_id"],
        session_id=payload.get("session_id"),
    )
    if result.get("status") in ("not_found", "wrong_project"):
        raise HTTPException(status_code=404 if result["status"] == "not_found" else 400,
                            detail=result.get("detail"))
    return result


@router.get("/workspace")
async def workspace_ui() -> HTMLResponse:
    """Phase B project workspace shell. The classic /ui above is untouched."""
    content = (Path(__file__).parent.parent / "static" / "workspace.html").read_text()
    return HTMLResponse(content)


@router.get("/automl/features")
async def feature_state(run_id: str | None = None) -> dict:
    """Feature state: live working table + persisted per-run feature history.

    The live columns/ops describe the table in memory right now. The
    ``run_history`` block is read from the stored run record, so the Features
    page still shows a run's pipeline after the working table was replaced by a
    later run or after a refresh.
    """
    orch = get_orchestrator()
    live: dict = {}
    try:
        live = {
            "columns": orch.store.columns(),
            "dtypes": orch.store.dtypes(),
            "excluded": orch.excluded_features(),
            "added": orch.store.feature_log[-50:],
            "ops": orch.store.op_log[-50:],
            "target": orch.requirements.target_column,
        }
    except Exception as exc:
        live = {"columns": [], "dtypes": {}, "excluded": [], "added": [], "ops": [],
                "target": orch.requirements.target_column, "unavailable": str(exc)}

    history: list[dict] = []
    try:
        from backend import runlog as _runlog

        rows = _runlog.list_runs(project_id=orch.project_id, limit=10) if orch.project_id else []
        if run_id:
            rows = [r for r in _runlog.list_runs(project_id=orch.project_id, limit=50) if r["run_id"] == run_id] or rows
        history = [
            {
                "run_id": r["run_id"],
                "status": r["status"],
                "features": r.get("feature_history") or [],
                "research": r.get("research") or [],
                "n_features": len(r.get("feature_history") or []),
            }
            for r in rows
        ]
    except Exception:
        history = []
    return {**live, "run_history": history}


@router.post("/automl/features/disable")
async def feature_disable(body: dict) -> dict:
    orch = get_orchestrator()
    return {"note": orch.disable_feature(body.get("column", ""), body.get("project_id")),
            "excluded": orch.excluded_features()}


@router.post("/automl/features/enable")
async def feature_enable(body: dict) -> dict:
    orch = get_orchestrator()
    return {"note": orch.enable_feature(body.get("column", ""), body.get("project_id")),
            "excluded": orch.excluded_features()}


@router.post("/automl/features/remove")
async def feature_remove(body: dict) -> dict:
    orch = get_orchestrator()
    return {"note": orch.remove_feature(body.get("column", ""), body.get("project_id"))}


@router.post("/automl/features/undo")
async def feature_undo(body: dict | None = None) -> dict:
    orch = get_orchestrator()
    pid = (body or {}).get("project_id")
    return {"note": orch.undo_last_feature(pid)}


@router.post("/automl/target")
async def set_target(body: dict) -> dict:
    """Set the prediction target for the next run (validated against the table)."""
    orch = get_orchestrator()
    col = (body or {}).get("column", "")
    try:
        cols = orch.store.columns()
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"No working dataset loaded: {exc}")
    if col not in cols:
        raise HTTPException(status_code=400, detail=f"Column '{col}' not in working dataset.")
    orch.requirements.target_column = col
    pid = (body or {}).get("project_id") or orch.project_id
    if pid:
        from backend import runlog as _runlog

        _runlog.remember(pid, "target", col)
    return {"target": col}


# ------------------------------------------------------------------ WebSocket

@router.websocket("/ws/automl")
async def automl_websocket(ws: WebSocket) -> None:
    await ws.accept()
    orch = get_orchestrator()
    bus = get_event_bus()
    loop = _running_loop()
    # Optional per-project filter: ?project_id=<uuid> (or "none"). Without it the
    # connection keeps the classic whole-buffer behaviour.
    raw_project = ws.query_params.get("project_id")
    if raw_project is None:
        filter_project, all_projects = None, True
    else:
        filter_project = None if raw_project in ("none", "null", "") else raw_project
        all_projects = False
    client_id, queue = bus.subscribe(loop)

    try:
        # Initial state + replay so late-joining clients sync instantly.
        await ws.send_json({"type": "snapshot", "data": orch.snapshot()})
        for event in bus.replay(client_id, project_id=filter_project, all_projects=all_projects):
            await ws.send_json(event)

        async def pump() -> None:
            while True:
                event = await queue.get()
                if not all_projects and event.get("project_id") != filter_project:
                    continue  # live events of other projects never reach this client
                await ws.send_json(event)

        import asyncio

        tasks = [asyncio.create_task(pump())]
        try:
            while True:
                message = await ws.receive_json()
                if (message.get("event") or message.get("type")) == "start":
                    # Same project/session validation as REST. On failure send
                    # an error ack and start nothing (never kill the socket).
                    message = await _validate_ws_start(message, ws)
                    if message is None:
                        continue
                result = orch.handle_message(message)
                if result:
                    await ws.send_json({"type": "ack", **result})
        except WebSocketDisconnect:
            pass
        finally:
            for task in tasks:
                task.cancel()
    finally:
        bus.unsubscribe(client_id)


def _running_loop():
    import asyncio

    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


async def _validate_ws_start(message: dict, ws: WebSocket) -> dict | None:
    """Validate a WS ``start`` message's project/session context.

    Returns the message with validated context attached, or ``None`` when
    validation failed (an error ``ack`` has been sent; the run was NOT started).
    """
    ctx = extract_run_context(message)
    try:
        async with async_session_maker() as db:
            validated = await validate_run_context(ctx["project_id"], ctx["session_id"], db)
    except HTTPException as exc:
        await ws.send_json({"type": "ack", "status": "error", "detail": exc.detail, "code": exc.status_code})
        return None
    message = dict(message)
    if validated["project_id"]:
        message["project_id"] = validated["project_id"]
    if validated["session_id"]:
        message["session_id"] = validated["session_id"]
    return message