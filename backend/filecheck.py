"""Shared upload validation for dataset intake endpoints.

Used by both ``POST /datasets`` (Track 1) and ``POST /automl/upload`` (Track 2
staging). Every failure raises ``HTTPException`` with a user-facing message —
never a traceback, never a silent accept.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import HTTPException

from backend.config import settings

ALLOWED_SUFFIXES = {".csv", ".parquet", ".pq", ".xlsx", ".xls", ".json"}


def check_filename(filename: str | None) -> str:
    """Validate the extension. Returns the lowercased suffix."""
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type {suffix or '(none)'} for {filename or 'upload'!r}. "
                f"Supported: {', '.join(sorted(ALLOWED_SUFFIXES))}."
            ),
        )
    # Path traversal: only ever use the bare name downstream; reject separators.
    if (filename or "").replace("\\", "/").split("/")[-1] != (filename or ""):
        raise HTTPException(status_code=400, detail="File name must not contain path separators.")
    return suffix


def check_size(size_bytes: int | None, *, context: str = "upload") -> None:
    cap = settings.max_file_size_mb * 1024 * 1024
    if size_bytes is not None and size_bytes > cap:
        raise HTTPException(
            status_code=413,
            detail=f"{context} is {size_bytes / 1024 / 1024:.1f}MB; limit is {settings.max_file_size_mb}MB.",
        )


def read_tabular(path: str | Path) -> pd.DataFrame:
    """Parse + validate a staged file. Returns the dataframe or raises 400."""
    p = Path(path)
    try:
        if p.suffix.lower() in (".parquet", ".pq"):
            df = pd.read_parquet(p)
        elif p.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(p)
        elif p.suffix.lower() == ".json":
            df = pd.read_json(p)
        else:
            df = pd.read_csv(p)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse {p.name}: {exc}") from exc
    if df.empty or len(df.columns) == 0:
        raise HTTPException(status_code=400, detail=f"{p.name} contains no data rows or columns.")
    return df


def dataset_info(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": [str(c) for c in df.columns],
    }
