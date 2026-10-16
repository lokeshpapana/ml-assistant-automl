"""Project reuse: resource listing, compatibility analysis, import with provenance.

Modes: quick (compatible only), smart (compatible + safe auto-adaptations),
review (dry-run diff, applies nothing). Every APPLIED artifact writes a
``project_imports`` provenance row. Nothing is ever silently copied: verdicts
carry explicit reasons, and incompatible resources are refused.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend import runlog
from backend.central_store import get_central_store
from backend.config import settings
from backend.database import get_async_session
from backend.database.models import Dataset, Project
from backend.orchestrator import get_orchestrator

router = APIRouter(tags=["imports"])

RESOURCE_TYPES = {
    "dataset", "dataset_snapshot", "feature_pipeline",
    "model_artifact", "experiment_config", "research",
}
MODES = {"quick", "smart", "review"}

try:
    from ml.models.registry import get_available_models

    _KNOWN_MODELS = set(get_available_models("classification")) | set(get_available_models("regression"))
except Exception:
    _KNOWN_MODELS = set()


async def _projects_or_404(db: AsyncSession, target_id: UUID, source_id: UUID) -> tuple[Any, Any]:
    target = await db.get(Project, target_id)
    if not target:
        raise HTTPException(status_code=404, detail="Target project not found")
    source = await db.get(Project, source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source project not found")
    if target_id == source_id:
        raise HTTPException(status_code=400, detail="Cannot import a project into itself")
    return target, source


def _source_run_or_404(source_project_id: str, source_run_id: str | None) -> dict:
    run = runlog.get_run(source_run_id) if source_run_id else None
    if source_run_id and run is None:
        raise HTTPException(status_code=404, detail=f"Source run {source_run_id} not found")
    if run and (run.get("project_id") or "") != source_project_id:
        raise HTTPException(status_code=400, detail="Source run does not belong to the source project")
    if run is None:
        runs = runlog.list_runs(project_id=source_project_id, limit=1)
        if not runs:
            raise HTTPException(status_code=404, detail="Source project has no runs to import from")
        run = runs[0]
    return run


@router.get("/projects/{project_id}/resources")
async def project_resources(project_id: UUID, db: AsyncSession = Depends(get_async_session)) -> dict:
    """Everything importable, assembled from real rows/artifacts (no mocks)."""
    from sqlalchemy import select

    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    ds_rows = (await db.execute(select(Dataset).where(Dataset.project_id == project_id))).scalars().all()
    runs = runlog.list_runs(project_id=str(project_id), limit=100)
    return {
        "project_id": str(project_id),
        "datasets": [
            {"id": str(d.id), "name": d.name, "rows": d.rows, "columns": d.columns,
             "status": str(d.status), "file_path": d.file_path}
            for d in ds_rows
        ],
        "runs": [
            {"run_id": r["run_id"], "status": r["status"], "stage": r["stage"],
             "started_at": r["started_at"],
             "best_model": (r.get("best") or {}).get("model"),
             "n_features": len(r.get("feature_history") or [])}
            for r in runs
        ],
        "memory": runlog.get_memory(str(project_id)),
        "imports": runlog.list_imports(str(project_id)),
    }


@router.get("/projects/{project_id}/imports")
async def list_project_imports(project_id: UUID, db: AsyncSession = Depends(get_async_session)) -> dict:
    if not await db.get(Project, project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    return {"imports": runlog.list_imports(str(project_id))}


@router.post("/projects/{project_id}/imports")
async def import_resources(
    project_id: UUID,
    body: dict,
    db: AsyncSession = Depends(get_async_session),
) -> dict:
    source_id = body.get("source_project_id")
    if not source_id:
        raise HTTPException(status_code=400, detail="source_project_id is required")
    mode = (body.get("mode") or "quick").lower()
    if mode not in MODES:
        raise HTTPException(status_code=400, detail=f"mode must be one of {sorted(MODES)}")
    wanted = body.get("resources") or []
    if not wanted:
        raise HTTPException(status_code=400, detail="resources list is required")

    _, source = await _projects_or_404(db, project_id, UUID(str(source_id)))
    run = _source_run_or_404(str(source.id), body.get("source_run_id"))
    store = get_central_store()
    # The central store is a single global working table. Only trust it as the
    # TARGET project's data when the live run actually belongs to that project;
    # otherwise the leftover table belongs to somebody else and comparing against
    # it would silently judge compatibility on the wrong columns.
    orch = get_orchestrator()
    live_owned = (orch.project_id or "") == str(project_id)
    target_cols = set(store.columns()) if (live_owned and _store_usable(store)) else set()
    target_mem = runlog.get_memory(str(project_id))
    if not live_owned:
        target_cols = set()

    results = []
    for item in wanted:
        rtype = (item or {}).get("type", "")
        ref = (item or {}).get("ref") or {}
        if rtype not in RESOURCE_TYPES:
            results.append({"type": rtype, "verdict": "incompatible",
                            "reasons": [f"Unknown resource type {rtype!r}"], "applied": False,
                            "adapted": False, "detail": None,
                            "would_apply": {"items": [], "note": "unknown resource type"}})
            continue
        verdict, reasons, plan = await _check(rtype, ref, run, db, target_cols, target_mem, store)
        applied, adapted, note = False, False, None
        if mode == "review":
            # Dry run: also return exactly what WOULD be applied, so the UI can
            # show a real diff instead of only a verdict.
            results.append({"type": rtype, "ref": ref, "verdict": verdict,
                            "reasons": reasons, "applied": False, "adapted": False,
                            "detail": None, "would_apply": _diff_preview(rtype, plan)})
            continue
        if verdict == "compatible":
            applied, adapted, note = _apply(rtype, plan, str(project_id), run)
        elif mode == "smart" and verdict == "partially_compatible" and plan:
            # Smart mode adapts anything the checker could stage/trim safely.
            applied, adapted, note = _apply(rtype, plan, str(project_id), run, adapted=True)
        else:
            reasons = list(reasons) + [
                "QUICK mode only imports fully compatible resources; use SMART to adapt this one."
            ]
        if applied:
            runlog.record_import(
                str(project_id), rtype, {"ref": ref, "applied": note},
                mode, verdict, reasons,
                source_project_id=str(source.id), source_run_id=run["run_id"], adapted=adapted,
            )
        results.append({"type": rtype, "ref": ref, "verdict": verdict,
                        "reasons": reasons, "applied": applied,
                        "adapted": adapted, "detail": note,
                        # Uniform contract: callers can always read would_apply.
                        "would_apply": _diff_preview(rtype, plan)})
    return {"mode": mode, "source_run_id": run["run_id"], "results": results}


def _store_usable(store: Any) -> bool:
    try:
        store.columns()
        return True
    except Exception:
        return False


def _diff_preview(rtype: str, plan: dict) -> dict:
    """Human-readable summary of what an import would change (REVIEW mode).

    Lists the concrete items (feature names/kinds, files, config keys) instead
    of a bare verdict, so the user can judge the change before applying it.
    """
    if not plan:
        return {"items": [], "note": "nothing to apply"}
    if rtype == "feature_pipeline":
        specs = plan.get("adapted_specs") or plan.get("specs") or []
        skipped = len((plan.get("specs") or [])) - len(specs)
        return {
            "items": [
                {"add": f.get("name") or f"f_{f.get('kind')}_{f.get('a')}",
                 "operation": f.get("kind"), "inputs": [x for x in (f.get("a"), f.get("b")) if x]}
                for f in specs
            ],
            "skipped": skipped,
            "note": "staged (applied after the target loads data)" if plan.get("stage_only") else "applied in place",
        }
    if rtype in ("dataset", "dataset_snapshot"):
        return {"items": [{"copy": plan.get("path"), "name": plan.get("name")}], "note": "copied into this project"}
    if rtype == "model_artifact":
        return {"items": [{"copy": plan.get("path")}],
                "note": "staged copy; never auto-activated as the project model"}
    if rtype == "experiment_config":
        return {"items": [{"add": f"config.{k}"} for k in sorted((plan.get("config") or {}).keys())],
                "note": "stored as pending_config for the next run"}
    if rtype == "research":
        return {"items": [{"add": (c or {}).get("title") or (c or {}).get("paper_id")} for c in plan.get("cards") or []],
                "note": "informational findings copied into project memory"}
    return {"items": [], "note": "no preview available"}


async def _check(rtype: str, ref: dict, run: dict, db: AsyncSession, target_cols: set,
           target_mem: dict, store: Any) -> tuple[str, list[str], dict]:
    """Return (verdict, reasons, plan). Verdicts: compatible | partially_compatible | incompatible."""
    if rtype == "dataset":
        return await _check_dataset(ref, db)
    if rtype == "dataset_snapshot":
        snap = ((run.get("best") or {}).get("dataset_snapshot")) or ""
        if snap and Path(snap).exists():
            return "compatible", [f"Snapshot CSV present ({Path(snap).stat().st_size // 1024}KB)."], {"path": snap}
        return "incompatible", ["Source run has no dataset snapshot artifact."], {}
    if rtype == "feature_pipeline":
        feats = run.get("feature_history") or []
        if not feats:
            return "incompatible", ["Source run engineered no features."], {}
        if not target_cols:
            # Target has no working table of its own yet: SMART stages the specs
            # (applied when the target loads data), QUICK refuses. Never blind-copy.
            return ("partially_compatible",
                    [f"Target project has no working table yet, so {len(feats)} feature spec(s) are staged rather than applied.",
                     "They will be applied when this project loads a dataset."],
                    {"specs": feats, "adapted_specs": feats, "stage_only": True})
        missing = sorted({c for c in (f.get("a"), f.get("b")) for f in feats if c and c not in target_cols})
        if not missing:
            return "compatible", [f"All {len(feats)} feature specs reference existing columns."], {"specs": feats}
        ok = [f for f in feats if (f.get("a") in target_cols) and (not f.get("b") or f.get("b") in target_cols)]
        return ("partially_compatible",
                [f"{len(missing)} column(s) missing in target: {', '.join(missing[:8])}.",
                 f"{len(ok)} of {len(feats)} specs can be adapted (the rest are skipped)."],
                {"specs": feats, "adapted_specs": ok, "missing": missing})
    if rtype == "model_artifact":
        best = run.get("best") or {}
        pkl = best.get("model_path", "")
        if not pkl or not Path(pkl).exists():
            return "incompatible", ["Source run has no stored model artifact."], {}
        src_task = (run.get("dataset") or {}).get("task_type")
        tgt_task = (target_mem.get("task_type") or "auto")
        if tgt_task not in ("auto", "") and src_task and tgt_task != src_task:
            return ("incompatible",
                    [f"Task mismatch: source is {src_task}, target expects {tgt_task}. Reusing the model would be invalid."],
                    {})
        reasons = [f"Artifact present ({best.get('model')})."]
        if not src_task or tgt_task in ("auto", ""):
            return ("partially_compatible",
                    reasons + ["Target task type is not confirmed yet — SMART stages the file only; it is never set as the project's model."],
                    {"path": pkl, "stage_only": True})
        return "compatible", reasons + [f"Task types match ({src_task})."], {"path": pkl}
    if rtype == "experiment_config":
        cfg = run.get("config") or {}
        if not cfg:
            return "incompatible", ["Source run stored no configuration."], {}
        reasons, partial = [], False
        src = cfg.get("dataset_source")
        if src == "upload":
            p = cfg.get("dataset_path", "")
            if p and Path(p).exists():
                reasons.append("Upload file still present.")
            else:
                partial, reasons = True, reasons + ["Original upload file is gone; config needs a new dataset."]
        elif src == "kaggle":
            partial, reasons = True, reasons + ["Kaggle needs credentials in this environment."]
        else:
            reasons.append(f"Dataset source {src or 'synthetic'} is reproducible here.")
        return ("partially_compatible" if partial else "compatible"), reasons, {"config": cfg}
    if rtype == "research":
        cards = run.get("research") or []
        if not cards:
            return "incompatible", ["Source run recorded no research findings."], {}
        return "compatible", [f"{len(cards)} finding(s), informational — always safe to import."], {"cards": cards}
    return "incompatible", [f"Unhandled type {rtype}"], {}


async def _check_dataset(ref: dict, db: AsyncSession) -> tuple[str, list[str], dict]:
    from sqlalchemy import select as _select

    ds_id = ref.get("dataset_id")
    if not ds_id:
        return "incompatible", ["ref.dataset_id is required for type 'dataset'."], {}
    try:
        row = (await db.execute(_select(Dataset).where(Dataset.id == UUID(str(ds_id))))).scalars().first()
    except (ValueError, AttributeError):
        return "incompatible", [f"Malformed dataset_id {ds_id!r}."], {}
    if row is None:
        return "incompatible", [f"Dataset {ds_id} not found."], {}
    if not row.file_path or not Path(row.file_path).exists():
        return "incompatible", [f"Dataset file missing: {row.file_path}."], {}
    return "compatible", [f"{row.rows} rows × {row.columns} cols, file present."], {
        "dataset_id": str(row.id), "path": row.file_path, "name": row.name,
        "rows": row.rows, "columns": row.columns,
    }


def _apply(rtype: str, plan: dict, target_project: str, run: dict, adapted: bool = False) -> tuple[bool, bool, Any]:
    """Execute one import. Returns (applied, adapted, detail). Real file/row/store ops only."""
    out_dir = Path(settings.upload_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(settings.automl_output_dir) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    if rtype == "dataset":
        src = Path(plan["path"])
        dest = out_dir / f"imported_{uuid.uuid4().hex[:8]}{src.suffix.lower()}"
        shutil.copyfile(src, dest)
        dataset = {"source": "upload", "path": str(dest), "rows": plan.get("rows"),
                   "columns": plan.get("columns"), "imported_from": plan.get("name")}
        runlog.remember(target_project, "dataset", dataset)
        return True, False, {"path": str(dest), "dataset": dataset}

    if rtype == "dataset_snapshot":
        src = Path(plan["path"])
        dest = out_dir / f"imported_{uuid.uuid4().hex[:8]}.csv"
        shutil.copyfile(src, dest)
        dataset = {"source": "upload", "path": str(dest), "imported_snapshot_of": run["run_id"]}
        runlog.remember(target_project, "dataset", dataset)
        return True, False, {"path": str(dest)}

    if rtype == "feature_pipeline":
        store = get_central_store()
        specs = plan["adapted_specs"] if adapted else plan["specs"]
        if plan.get("stage_only") or not _store_usable(store):
            runlog.remember(target_project, "pending_features", specs)
            runlog.remember(target_project, "feature_decisions",
                            f"Staged {len(specs)} feature spec(s) from run {run['run_id']} (apply after data load).")
            return True, adapted, {"staged": len(specs), "note": "staged; not applied yet"}
        applied, skipped = [], []
        for spec in specs:
            try:
                entry = store.add_feature(dict(spec))
                applied.append(entry.get("name"))
            except Exception as exc:
                skipped.append(f"{spec.get('name')}: {exc}")
        runlog.remember(target_project, "feature_decisions",
                        f"Imported {len(applied)} feature(s) from run {run['run_id']}" + (" (adapted)" if adapted else ""))
        return True, adapted, {"applied": applied, "skipped": skipped}

    if rtype == "model_artifact":
        src = Path(plan["path"])
        dest = model_dir / f"imported_{run['run_id']}_{src.name}"
        shutil.copyfile(src, dest)
        runlog.remember(target_project, "model_decisions",
                        f"Imported model artifact {dest.name} (inference-ready copy; not set as best).")
        if plan.get("stage_only"):
            runlog.remember(target_project, "pending_model", {"path": str(dest), "run_id": run["run_id"]})
            return True, True, {"path": str(dest), "note": "staged copy; not activated"}
        return True, False, {"path": str(dest)}

    if rtype == "experiment_config":
        runlog.remember(target_project, "pending_config", plan["config"])
        return True, False, {"pending_config": plan["config"]}

    if rtype == "research":
        runlog.remember(target_project, "research", plan["cards"])
        return True, False, {"cards_imported": len(plan["cards"])}

    return False, False, "unknown type"
