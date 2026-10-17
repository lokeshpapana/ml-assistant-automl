"""Run records, diagnostics, and reports — all from persisted data, never invented.

Sources, in order: ``runs``/``run_events`` rows (operational truth) → best-model
``.pkl`` + dataset snapshot CSV artifacts (recomputed evidence) → MLflow
(params/metrics lineage) → project memory (context). Anything missing yields
404/empty with an honest reason, never a fabricated value.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any
from uuid import UUID

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend import runlog
from backend.database import get_async_session
from backend.database.models import Project

router = APIRouter(tags=["runs"])

_RUN_ID_RE = re.compile(r"^run-[0-9a-f]{12}$")

GLOSSARY = {
    "accuracy": "Share of correct predictions. Misleading on imbalanced targets.",
    "precision": "Of predicted positives, share actually positive.",
    "recall": "Of actual positives, share found. Same as sensitivity.",
    "f1": "Harmonic mean of precision and recall.",
    "roc_auc": "Ranking quality across thresholds (binary only). 0.5 = chance.",
    "pr_auc": "Precision-recall trade-off area (binary only); sensitive to imbalance.",
    "mae": "Mean absolute error, same units as the target.",
    "mse": "Mean squared error; punishes large errors.",
    "rmse": "Root mean squared error, same units as the target.",
    "r2": "Share of target variance explained. 1.0 = perfect.",
}


def _check_run_id(run_id: str) -> str:
    if not _RUN_ID_RE.match(run_id or ""):
        raise HTTPException(status_code=400, detail=f"Invalid run_id: {run_id!r}")
    return run_id


def _get_run_or_404(run_id: str) -> dict:
    run = runlog.get_run(_check_run_id(run_id))
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return run


@router.get("/runs")
async def list_runs(
    project_id: UUID | None = None,
    status: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict:
    """List persisted runs, optionally filtered. Newest first."""
    return {
        "runs": runlog.list_runs(
            project_id=str(project_id) if project_id else None,
            status=status,
            limit=limit,
            offset=offset,
        ),
        "limit": limit,
        "offset": offset,
    }


@router.get("/runs/{run_id}")
async def get_run(run_id: str, events_limit: int = Query(1000, ge=0, le=5000)) -> dict:
    """Complete run record: metadata, status, progress, stages, timestamps,
    metrics, model, dataset, events, errors, artifacts. Refresh-safe."""
    run = _get_run_or_404(run_id)
    events = runlog.list_events(run_id, limit=events_limit)
    stages = sorted({e.get("stage") for e in events if e.get("stage")})
    durations = [
        (events[i + 1]["ts"], events[i]["ts"])
        for i in range(len(events) - 1)
        if events[i].get("ts") and events[i + 1].get("ts")
    ]
    artifacts = []
    best = run.get("best") or {}
    if best.get("model_path"):
        artifacts.append({"kind": "model", "path": best["model_path"]})
    if best.get("dataset_snapshot"):
        artifacts.append({"kind": "dataset_snapshot", "path": best["dataset_snapshot"]})
    return {
        "run": run,
        "events": events,
        "event_count": len(events),
        "stages_seen": stages,
        "artifacts": artifacts,
        "mlflow_run_ids": sorted(
            {
                e.get("metadata", {}).get("mlflow_run_id")
                for e in events
                if e.get("metadata", {}).get("mlflow_run_id")
            }
        ),
    }


@router.get("/runs/{run_id}/diagnostics")
async def run_diagnostics(run_id: str) -> dict:
    """Recomputed evaluation evidence for the run's best model.

    Loads the stored ``.pkl`` bundle + dataset snapshot CSV and recomputes -
    nothing is taken on trust from stored metrics. Omits (with reasons) anything
    mathematically invalid for the task (e.g. ROC for multiclass/regression).
    """
    run = _get_run_or_404(run_id)
    best = run.get("best") or {}
    bundle, df, target = _load_best_bundle(run, best)
    model = bundle.get("model")
    engineer = bundle.get("engineer")
    task = (run.get("dataset") or {}).get("task_type") or _infer_task(df[target])

    X, y = _transformed(engineer, df, target)
    y_pred = model.predict(X)
    try:
        y_proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    except Exception:
        y_proba = None

    out: dict[str, Any] = {
        "run_id": run_id,
        "model": best.get("model"),
        "iteration": best.get("iteration"),
        "task_type": task,
        "n_rows": int(len(df)),
        "glossary": GLOSSARY,
        "stored_metrics": best.get("metrics"),
        "feature_importance": best.get("feature_importance"),
        "omitted": [],
    }
    if task == "classification":
        out.update(_classification_diagnostics(y, y_pred, y_proba))
    else:
        out.update(_regression_diagnostics(y, y_pred))
    out["permutation_importance"] = _permutation_importance(model, X, y, task, out["omitted"])
    out["prediction_distribution"] = _distribution(y_pred, task)
    out["error_examples"] = _error_examples(df, y, y_pred, task)
    return out


# ------------------------------------------------------------------ internals


def _infer_task(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > 20:
        return "regression"
    return "classification"


def _load_best_bundle(run: dict, best: dict) -> tuple[dict, pd.DataFrame, str]:
    if not best or not best.get("model_path"):
        raise HTTPException(status_code=404, detail="This run has no evaluated best model yet.")
    pkl, snap = Path(best["model_path"]), Path(best.get("dataset_snapshot") or "")
    if not pkl.exists():
        raise HTTPException(status_code=404, detail=f"Best-model artifact missing: {pkl}")
    if not snap.exists():
        raise HTTPException(
            status_code=404, detail="Dataset snapshot for this run is missing; diagnostics need it."
        )
    try:
        with open(pkl, "rb") as fh:
            bundle = pickle.load(fh)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Could not load model artifact: {exc}"
        ) from exc
    try:
        df = pd.read_csv(snap)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Could not read dataset snapshot: {exc}"
        ) from exc
    target = (run.get("dataset") or {}).get("target_column") or (run.get("config") or {}).get(
        "target_column"
    )
    engineer = bundle.get("engineer")
    if not target or target not in df.columns:
        eng_target = getattr(engineer, "target_column", None)
        if eng_target and eng_target in df.columns:
            target = eng_target
    if not target or target not in df.columns:
        raise HTTPException(status_code=404, detail="Target column for this run is unknown.")
    return bundle, df, target


def _transformed(engineer: Any, df: pd.DataFrame, target: str) -> tuple[Any, Any]:
    if engineer is None or not hasattr(engineer, "transform"):
        raise HTTPException(status_code=500, detail="Stored bundle has no usable feature pipeline.")
    try:
        X, y = engineer.transform(df)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Feature pipeline failed on the snapshot: {exc}"
        ) from exc
    if y is None:
        y = df[target].to_numpy()
    return X, y


def _sample_idx(n: int, cap: int) -> Any:
    import numpy as np

    if n <= cap:
        return slice(None)
    return np.linspace(0, n - 1, cap).astype(int)


def _classification_diagnostics(y: Any, y_pred: Any, y_proba: Any) -> dict:
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    y = np.asarray(y)
    labels = sorted(pd.unique(np.concatenate([y, np.asarray(y_pred)])).tolist())
    out: dict[str, Any] = {
        "metrics_recomputed": {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
        },
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y, y_pred, labels=labels).tolist(),
        },
        "classification_report": classification_report(
            y, y_pred, zero_division=0, output_dict=True
        ),
        "roc": None,
        "pr": None,
    }
    if len(labels) == 2 and y_proba is not None:
        from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

        try:
            proba = np.asarray(y_proba)
            pos = proba[:, 1] if proba.ndim > 1 else proba
            fpr, tpr, _ = roc_curve(y, pos, pos_label=labels[1])
            prec, rec, _ = precision_recall_curve(y, pos, pos_label=labels[1])
            idx = _sample_idx(len(fpr), 200)
            jdx = _sample_idx(len(prec), 200)
            out["roc"] = {
                "fpr": np.asarray(fpr)[idx].tolist(),
                "tpr": np.asarray(tpr)[idx].tolist(),
                "auc": float(roc_auc_score(y, pos)),
            }
            out["pr"] = {
                "precision": np.asarray(prec)[jdx].tolist(),
                "recall": np.asarray(rec)[jdx].tolist(),
                "auc": float(auc(np.asarray(rec)[jdx], np.asarray(prec)[jdx])),
            }
        except Exception as exc:
            out["omitted"].append(f"ROC/PR curves unavailable: {exc}")
    else:
        out["omitted"].append(
            "ROC/PR curves need a binary target with predicted probabilities; "
            "confusion matrix and report above are the valid evidence."
        )
    return out


def _regression_diagnostics(y: Any, y_pred: Any) -> dict:
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y = np.asarray(y, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    resid = y - yp
    order = np.argsort(yp)
    idx = _sample_idx(len(yp), 500)
    deciles = np.array_split(np.sort(yp), 10)
    out: dict[str, Any] = {
        "metrics_recomputed": {
            "mae": float(mean_absolute_error(y, yp)),
            "mse": float(mean_squared_error(y, yp)),
            "rmse": float(mean_squared_error(y, yp) ** 0.5),
            "r2": float(r2_score(y, yp)),
        },
        "residuals": {
            "mean": float(np.mean(resid)),
            "std": float(np.std(resid)),
            "min": float(np.min(resid)),
            "max": float(np.max(resid)),
        },
        "pred_vs_actual": {
            "predicted": yp[order][idx].tolist(),
            "actual": y[order][idx].tolist(),
        },
        "decile_means": [
            {
                "predicted_mean": float(np.mean(d)),
                "actual_mean": float(
                    np.mean(y[np.argsort(yp)][i * len(y) // 10 : (i + 1) * len(y) // 10 or None])
                ),
            }
            for i, d in enumerate(deciles)
        ],
        "roc": None,
        "pr": None,
        "confusion_matrix": None,
    }
    out["omitted"].append(
        "ROC/PR/confusion matrix are classification-only and correctly omitted for regression."
    )
    return out


def _distribution(y_pred: Any, task: str) -> dict:
    import numpy as np

    yp = np.asarray(y_pred)
    if task == "classification":
        vals, counts = np.unique(yp, return_counts=True)
        return {"classes": vals.tolist(), "counts": counts.tolist()}
    hist, edges = np.histogram(yp, bins=20)
    return {"hist_counts": hist.tolist(), "hist_edges": edges.tolist()}


def _error_examples(df: pd.DataFrame, y: Any, y_pred: Any, task: str) -> list[dict]:
    import numpy as np

    y = np.asarray(y)
    yp = np.asarray(y_pred)
    if task == "classification":
        bad = np.where(y != yp)[0][:5]
        return [
            {
                "row": int(i),
                "actual": str(y[i]),
                "predicted": str(yp[i]),
                "features": {c: str(df.iloc[int(i)][c]) for c in list(df.columns)[:8]},
            }
            for i in bad
        ]
    resid = np.abs(np.asarray(y, dtype=float) - np.asarray(yp, dtype=float))
    bad = np.argsort(resid)[-5:][::-1]
    return [
        {
            "row": int(i),
            "actual": float(y[i]),
            "predicted": float(yp[i]),
            "abs_residual": float(resid[i]),
            "features": {c: str(df.iloc[int(i)][c]) for c in list(df.columns)[:8]},
        }
        for i in bad
    ]


def _permutation_importance(model: Any, X: Any, y: Any, task: str, omitted: list) -> dict | None:
    import numpy as np

    try:
        from sklearn.inspection import permutation_importance

        n = len(y)
        idx = np.random.RandomState(42).choice(n, size=min(n, 300), replace=False)
        Xs = X[idx] if hasattr(X, "__getitem__") else X
        ys = np.asarray(y)[idx]
        scoring = "f1_weighted" if task == "classification" else "r2"
        r = permutation_importance(model, Xs, ys, n_repeats=3, random_state=42, scoring=scoring)
        names = [f"feature_{i}" for i in range(len(r.importances_mean))]
        order = np.argsort(r.importances_mean)[::-1][:15]
        return {
            "scoring": scoring,
            "importance": {names[i]: float(r.importances_mean[i]) for i in order},
            "note": "Computed on a 300-row sample, 3 repeats.",
        }
    except Exception as exc:
        omitted.append(f"Permutation importance unavailable: {exc}")
        return None


# ------------------------------------------------------------------ reports


@router.get("/projects/{project_id}/report")
async def project_report(
    project_id: UUID,
    run_id: str | None = None,
    format: str = Query("json", pattern="^(json|markdown|html)$"),
    db: AsyncSession = Depends(get_async_session),
) -> Any:
    """Assemble the project report from stored data only.

    Sections with no backing data say so explicitly instead of inventing values.
    """
    from backend.database.models import Project as ProjectRow

    project = await db.get(ProjectRow, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    memory = runlog.get_memory(str(project_id))
    run = runlog.get_run(run_id) if run_id else _latest_done_run(str(project_id))
    if run_id and run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    events = runlog.list_events(run["run_id"]) if run else []
    report = _build_report(project, memory, run, events)
    if run:
        runlog.log_event(
            run["run_id"],
            "report_generated",
            message="Project report generated.",
            stage=run.get("stage"),
        )
    if format == "json":
        from fastapi.responses import JSONResponse

        return JSONResponse(report)
    if format == "markdown":
        from fastapi.responses import PlainTextResponse

        return PlainTextResponse(_report_markdown(report), media_type="text/markdown")
    from fastapi.responses import HTMLResponse

    return HTMLResponse(_report_html(report))


def _latest_done_run(project_id: str) -> dict | None:
    for status in ("done", "paused", "failed", "running"):
        runs = runlog.list_runs(project_id=project_id, status=status, limit=1)
        if runs:
            return runs[0]
    return None


def _build_report(project: Any, memory: dict, run: dict | None, events: list) -> dict:
    import sklearn
    import pandas as pdmod

    dataset = (memory.get("dataset") or {}) if memory else {}
    if run and run.get("dataset"):
        dataset = {**(dataset or {}), **run["dataset"]}
    best = (run or {}).get("best") if run else (memory.get("best") if memory else None)
    iters = [
        {
            "iteration": e.get("metadata", {}).get("iteration"),
            "model": e.get("metadata", {}).get("model"),
            "params": e.get("metadata", {}).get("params"),
            "metrics": e.get("metadata", {}).get("metrics"),
            "target_met": e.get("metadata", {}).get("target_met"),
        }
        for e in events
        if e.get("event_type") == "model_training"
    ]
    target_metric = ((run or {}).get("config") or {}).get("metric", "target_metric")
    ranked = sorted(
        iters,
        key=lambda r: (r.get("metrics") or {}).get(target_metric, float("-inf")),
        reverse=True,
    )
    limitations = list((memory.get("warnings") or []))
    limitations.append(
        "Evaluation uses a single holdout split (no cross-validation in AutoML track)."
    )
    if not run:
        limitations.append("No completed run is stored for this project yet — run AutoML first.")
    recommendations = []
    if best and best.get("target_met"):
        recommendations.append(
            f"Target met by {best.get('model')} — export the artifact for deployment."
        )
    elif run:
        recommendations.append(
            "Target not met — try more iterations, additional features, or a looser target."
        )
    if memory.get("warnings"):
        recommendations.append(
            "Address the warnings listed in Limitations before trusting the model."
        )
    return {
        "project": {
            "id": str(project.id),
            "name": project.name,
            "description": project.description,
            "status": str(project.status),
        },
        "dataset": dataset or {"note": "No dataset recorded for this project yet."},
        "data_quality": {
            "warnings": memory.get("warnings") or [],
            "note": "Extended profiling appears on the Analysis page once a dataset is profiled.",
        },
        "target": {
            "column": dataset.get("target_column"),
            "task_type": dataset.get("task_type"),
            "metric": ((run or {}).get("config") or {}).get("metric"),
        },
        "research": (run or {}).get("research") or memory.get("research") or [],
        "features": (run or {}).get("feature_history") or [],
        "experiments": ranked,
        "best_model": best,
        "evaluation": {
            "note": "Open the Evaluation page for recomputed diagnostics of the best model."
        },
        "limitations": limitations,
        "reproducibility": {
            "trainer_seed": 42,
            "run_id": (run or {}).get("run_id"),
            "config": (run or {}).get("config"),
            "package_versions": {
                "scikit-learn": getattr(sklearn, "__version__", "?"),
                "pandas": getattr(pdmod, "__version__", "?"),
            },
        },
        "run": (
            {
                k: run.get(k)
                for k in ("run_id", "status", "stage", "started_at", "completed_at", "error")
            }
            if run
            else None
        ),
        "recommendations": recommendations,
        "memory_decisions": (memory.get("decisions") or [])[-20:],
    }


def _report_markdown(r: dict) -> str:
    L = [f"# AutoML Report — {r['project']['name']}", ""]
    L.append(f"Status: {r['project']['status']}")
    if r.get("run"):
        L.append(f"Run: {r['run']['run_id']} ({r['run']['status']})")
    L += ["", "## Dataset", _kv(r["dataset"]), "", "## Target", _kv(r["target"])]
    L += ["", "## Experiments", _table(r["experiments"])]
    L += ["", "## Best model", _kv(r["best_model"] or {"note": "none yet"})]
    L += ["", "## Limitations"] + [f"- {x}" for x in r["limitations"]]
    L += ["", "## Recommendations"] + [f"- {x}" for x in r["recommendations"]]
    return "\n".join(L)


def _kv(d: dict) -> str:
    import json as _json

    if not d:
        return "_none_"
    return "\n".join(f"- **{k}**: `{_json.dumps(v)[:300]}`" for k, v in d.items())


def _table(rows: list) -> str:
    if not rows:
        return "_no experiments yet_"
    head = "| iter | model | target | met |\n|---|---|---|---|"
    lines = [head] + [
        f"| {r.get('iteration')} | {r.get('model')} | "
        f"{((r.get('metrics') or {}).get('target_metric') if isinstance(r.get('metrics'), dict) else r.get('metrics'))} | {r.get('target_met')} |"
        for r in rows[:30]
    ]
    return "\n".join(lines)


def _report_html(r: dict) -> str:
    import html as _html
    import json as _json

    body = f"<h1>AutoML Report — {_html.escape(str(r['project']['name']))}</h1>"
    body += (
        "<h2>Dataset</h2><pre>"
        + _html.escape(_json.dumps(r["dataset"], indent=2, default=str))
        + "</pre>"
    )
    body += (
        "<h2>Best model</h2><pre>"
        + _html.escape(_json.dumps(r["best_model"], indent=2, default=str))
        + "</pre>"
    )
    body += (
        "<h2>Limitations</h2><ul>"
        + "".join(f"<li>{_html.escape(str(x))}</li>" for x in r["limitations"])
        + "</ul>"
    )
    body += (
        "<h2>Recommendations</h2><ul>"
        + "".join(f"<li>{_html.escape(str(x))}</li>" for x in r["recommendations"])
        + "</ul>"
    )
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Report</title></head><body>{body}</body></html>"


@router.get("/runs/{run_id}/research")
async def run_research(run_id: str) -> dict:
    """Research findings behind a run, with honest provenance.

    Each card is labeled OFFLINE KNOWLEDGE (curated paper bank, no network
    needed) or LIVE RESEARCH (fetched from arXiv/Semantic Scholar during the
    run). The label is derived from the bank — never asserted blindly.
    """
    from backend.research.paper_bank import OFFLINE_PAPER_BANK

    run = _get_run_or_404(run_id)
    offline_ids = {c.get("paper_id") for c in OFFLINE_PAPER_BANK}
    cards = []
    for f in run.get("research") or []:
        pid = f.get("paper_id") or ""
        cards.append(
            {
                "paper_id": pid,
                "title": f.get("title"),
                "technique": f.get("technique"),
                "feature": f.get("feature"),
                "origin": "OFFLINE KNOWLEDGE"
                if pid in offline_ids
                else ("LIVE RESEARCH" if pid else "HEURISTIC"),
                "url": f"https://arxiv.org/abs/{pid}" if pid else None,
            }
        )
    return {"run_id": run_id, "cards": cards}
