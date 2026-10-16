from __future__ import annotations

import uuid
import warnings
from pathlib import Path
from typing import Any

import numpy as np

import pandas as pd

from ml.dataset_acquisition.evaluate import (
    analyze_features,
    identify_target,
    inspect_dataframe,
    rank_candidates,
    score_candidate,
)
from ml.dataset_acquisition.optimize import optimize_dataset
from ml.dataset_acquisition.report import build_report
from ml.dataset_acquisition.sources import (
    CandidateDataset,
    ingest_local_file,
    load_candidate,
    search_all_sources,
)
from ml.dataset_acquisition.spec import DatasetSpecification

warnings.filterwarnings("ignore")

DEFAULT_OUTPUT_DIR = Path("./acquired_datasets")


def _infer_target_from_request(request_text: str | None, columns: list[str]) -> str | None:
    """Find a target column implied by the user's request.

    Matches when a column name appears verbatim in the request, or when a
    meaningful request word (>=4 chars) is contained in a column name.
    """
    if not request_text:
        return None
    rt = request_text.lower()
    # 1. exact column name present in request
    for col in columns:
        if col.lower() in rt:
            return col
    # 2. shared meaningful token (word-boundary; avoids "loan" matching "loans")
    req_tokens = {w for w in rt.replace("_", " ").split() if len(w) >= 4}
    for col in columns:
        col_tokens = set(col.lower().replace("_", " ").split())
        if req_tokens & col_tokens:
            return col
    return None


def _make_spec(
    request: str | None, spec: DatasetSpecification | None, target: str | None
) -> DatasetSpecification:
    if spec is not None:
        return spec
    if request:
        return DatasetSpecification.from_user_request(request, target=target)
    raise ValueError("Either 'request' text or a 'spec' must be provided")


def _evaluate_candidate(
    candidate: CandidateDataset, spec: DatasetSpecification, max_eval: int = 5
) -> dict[str, Any] | None:
    try:
        df, meta = load_candidate(candidate)
    except Exception as e:
        warnings.warn(f"Failed to load candidate '{candidate.name}': {e}")
        return None

    if len(df) > max_eval * 1000:
        df = df.sample(max_eval * 1000, random_state=42)

    inspection = inspect_dataframe(df)
    target_hint = (
        candidate.target_suggestion
        or spec.target
        or _infer_target_from_request(spec.problem_description, list(df.columns))
    )
    target_info = identify_target(df, target_hint, spec.task.value)
    best_target = target_info.get("best_target")
    if not best_target:
        return None
    target_col = best_target["column"]

    assessments = analyze_features(df, target_col, spec.task.value)
    score = score_candidate(inspection, target_info, assessments, spec, candidate.description, candidate.license)
    return {
        "candidate": candidate,
        "df": df,
        "inspection": inspection,
        "target_info": target_info,
        "target_col": target_col,
        "assessments": assessments,
        "score": score,
    }


def acquire_dataset(
    request: str | None = None,
    spec: DatasetSpecification | None = None,
    local_file_path: str | Path | None = None,
    query: str | None = None,
    output_dir: str | Path | None = None,
    save_csv: bool = True,
    output_format: str = "csv",
    max_candidates: int = 5,
    requested_input_features: int | None = None,
) -> dict[str, Any]:
    """End-to-end Dataset Acquisition & Optimization Agent (spec steps 1-27)."""
    specification = _make_spec(request, spec, target=None)
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 2-4: gather & inspect candidates
    evaluated: list[dict[str, Any]] = []
    if local_file_path:
        candidate = ingest_local_file(local_file_path)
        ev = _evaluate_candidate(candidate, specification, max_eval=max_candidates)
        if ev:
            evaluated.append(ev)
    else:
        q = query or specification.problem_description
        candidates = search_all_sources(q, spec_task=specification.task.value, max_per_source=max_candidates)
        if not candidates:
            return {
                "status": "no_candidates",
                "message": "No datasets found from available sources. Provide a local file or relax constraints.",
                "spec": specification.model_dump(),
            }
        for c in candidates[:max_candidates]:
            ev = _evaluate_candidate(c, specification, max_eval=max_candidates)
            if ev:
                evaluated.append(ev)

    if not evaluated:
        return {
            "status": "evaluation_failed",
            "message": "No candidate could be evaluated successfully.",
            "spec": specification.model_dump(),
        }

    # Step 20-21: rank by dataset score (NOT model performance)
    scored = []
    for ev in evaluated:
        scored.append({"score": ev["score"], **ev})
    ranked = rank_candidates(scored)
    best = ranked[0]

    # Step 22-25: construct final optimized dataset
    req_feats = requested_input_features if requested_input_features is not None else specification.resolve_final_feature_count()
    optimization = optimize_dataset(
        best["df"],
        best["target_col"],
        best["assessments"],
        requested_input_features=req_feats,
    )

    output_path = None
    if save_csv:
        base = f"{best['candidate'].source}_{best['candidate'].name}_{uuid.uuid4().hex[:8]}"
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in base)
        output_path = _save_dataset(optimization.df, out_dir / safe, output_format)

    report = build_report(
        best["candidate"].to_summary(),
        specification,
        best["inspection"],
        best["target_info"],
        best["assessments"],
        best["score"],
        optimization,
        output_path=output_path,
    )
    report["status"] = "success"
    report["output_format"] = output_format
    report["spec"] = specification.model_dump()
    report["n_candidates_evaluated"] = len(evaluated)
    return report


def _save_dataset(df: pd.DataFrame, base_path: Path, output_format: str) -> str:
    """Save the optimized dataset in the requested format. Returns the file path."""
    fmt = (output_format or "csv").lower()
    if fmt == "csv":
        path = base_path.with_suffix(".csv")
        df.to_csv(path, index=False)
    elif fmt in ("xlsx", "excel"):
        path = base_path.with_suffix(".xlsx")
        df.to_excel(path, index=False, engine="openpyxl")
    elif fmt == "parquet":
        path = base_path.with_suffix(".parquet")
        df.to_parquet(path, index=False)
    elif fmt == "json":
        path = base_path.with_suffix(".json")
        df.to_json(path, orient="records", indent=2)
    elif fmt == "sql":
        path = base_path.with_suffix(".sql")
        _write_sql(df, path, table_name=base_path.stem)
    else:
        path = base_path.with_suffix(".csv")
        df.to_csv(path, index=False)
    return str(path)


def _write_sql(df: pd.DataFrame, path: Path, table_name: str) -> None:
    """Generate a portable SQL dump (CREATE TABLE + INSERTs) for the dataset."""
    import datetime as _dt

    sql_types: dict[str, str] = {}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            sql_types[col] = "BOOLEAN"
        elif pd.api.types.is_integer_dtype(series):
            sql_types[col] = "INTEGER"
        elif pd.api.types.is_float_dtype(series):
            sql_types[col] = "DOUBLE PRECISION"
        elif pd.api.types.is_datetime64_any_dtype(series):
            sql_types[col] = "TIMESTAMP"
        else:
            sql_types[col] = "TEXT"

    def sql_val(v) -> str:
        if pd.isna(v):
            return "NULL"
        if isinstance(v, (bool, np.bool_)):
            return "TRUE" if v else "FALSE"
        if isinstance(v, (int, float, np.integer, np.floating)):
            return str(v)
        s = str(v).replace("'", "''")
        return f"'{s}'"

    lines = [f'CREATE TABLE "{table_name}" (']
    lines.append(", ".join(f'"{c}" {sql_types[c]}' for c in df.columns))
    lines.append(");")
    lines.append("")
    n = len(df)
    chunk = 500
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        lines.append(f"INSERT INTO \"{table_name}\" ({', '.join(f'\"{c}\"' for c in df.columns)}) VALUES")
        rows = []
        for _, row in df.iloc[start:end].iterrows():
            vals = ", ".join(sql_val(v) for v in row.tolist())
            rows.append(f"  ({vals})")
        lines.append(",\n".join(rows) + ";")
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
