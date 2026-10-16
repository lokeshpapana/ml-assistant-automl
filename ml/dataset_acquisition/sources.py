from __future__ import annotations

import json
import shutil
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

warnings.filterwarnings("ignore")

MAX_DOWNLOAD_ROWS = 200_000


@dataclass
class CandidateDataset:
    name: str
    source: str
    url: str | None = None
    author: str | None = None
    description: str | None = None
    n_rows: int | None = None
    n_columns: int | None = None
    target_suggestion: str | None = None
    license: str | None = None
    format: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    # loader returns (DataFrame, metadata_dict)
    loader: Callable[[], tuple[pd.DataFrame, dict[str, Any]]] | None = None

    def to_summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "url": self.url,
            "author": self.author,
            "description": self.description,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "target_suggestion": self.target_suggestion,
            "license": self.license,
            "format": self.format,
            "extra": self.extra,
        }


def _read_any(path: Path, nrows: int | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, nrows=nrows)
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, nrows=nrows)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported file format: {suffix}")


def ingest_local_file(path: str | Path, name: str | None = None) -> CandidateDataset:
    """Wrap a user-provided local file as a candidate dataset."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Local dataset not found: {path}")

    def loader() -> tuple[pd.DataFrame, dict[str, Any]]:
        df = _read_any(p)
        meta = {
            "file_path": str(p),
            "format": p.suffix.lstrip("."),
            "n_rows_actual": int(df.shape[0]),
            "n_columns_actual": int(df.shape[1]),
        }
        return df, meta

    df_preview, _ = loader()
    return CandidateDataset(
        name=name or p.stem,
        source="user_provided_local",
        url=str(p),
        description="User-provided dataset file",
        n_rows=int(df_preview.shape[0]),
        n_columns=int(df_preview.shape[1]),
        license="user_provided",
        format=p.suffix.lstrip("."),
        loader=loader,
    )


def search_openml(query: str, max_results: int = 10, task_hint: str | None = None) -> list[CandidateDataset]:
    """Search OpenML datasets (no auth required for listing/download).

    Filters by name substring and (when available) by instance/feature counts so
    that results are close to typical tabular ML needs.
    """
    try:
        import openml
    except ImportError:
        return []

    results: list[CandidateDataset] = []
    try:
        df_meta = openml.datasets.list_datasets(
            limit=max(100, max_results * 10), output_format="dataframe"
        )
    except Exception as e:  # network / API issues
        warnings.warn(f"OpenML search failed: {e}")
        return results

    if query:
        q = query.lower()
        sub = df_meta[df_meta["name"].str.lower().str.contains(q, na=False)]
        if len(sub) == 0:
            sub = df_meta
    else:
        sub = df_meta

    # Prefer datasets with a reasonable number of features/rows when columns exist
    if "NumberOfFeatures" in sub.columns and "NumberOfInstances" in sub.columns:
        sub = sub[
            (sub["NumberOfFeatures"].between(5, 40))
            & (sub["NumberOfInstances"].between(100, 3000))
        ]
    sub = sub.head(max_results * 3)

    for _, row in sub.iterrows():
        did = int(row["did"])

        def make_loader(ds_id=did):
            def _load() -> tuple[pd.DataFrame, dict[str, Any]]:
                d = openml.datasets.get_dataset(ds_id)
                data, target, _, _ = d.get_data(target=None)
                if target is not None:
                    # OpenML separates the default target; re-attach it so the
                    # acquisition agent can evaluate any column as the target.
                    data = data.copy()
                    data[target.name] = target.values
                df = data
                if len(df) > MAX_DOWNLOAD_ROWS:
                    df = df.sample(MAX_DOWNLOAD_ROWS, random_state=42)
                meta = {
                    "openml_id": ds_id,
                    "format": "arff/csv",
                    "n_rows_actual": int(df.shape[0]),
                    "n_columns_actual": int(df.shape[1]),
                }
                return df, meta

            return _load

        target = getattr(row, "default_target_attribute", None)
        results.append(
            CandidateDataset(
                name=str(row["name"]),
                source="openml",
                url=f"https://www.openml.org/d/{did}",
                author=None,
                description=None,
                n_rows=int(row["NumberOfInstances"]) if "NumberOfInstances" in row else None,
                n_columns=int(row["NumberOfFeatures"]) if "NumberOfFeatures" in row else None,
                target_suggestion=target,
                license=str(row["License"]) if "License" in row else None,
                format="arff",
                loader=make_loader(),
            )
        )
    return results[:max_results]


def search_huggingface(query: str, max_results: int = 10) -> list[CandidateDataset]:
    """Search HuggingFace datasets (tabular ones). Requires `datasets` lib."""
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError:
        return []

    try:
        api = HfApi()
        items = api.list_datasets(search=query, limit=max_results)
    except Exception as e:
        warnings.warn(f"HuggingFace search failed: {e}")
        return []

    results: list[CandidateDataset] = []
    for item in list(items)[:max_results]:
        ds_id = getattr(item, "id", None)
        if not ds_id:
            continue

        def make_loader(ds_id=ds_id):
            def _load() -> tuple[pd.DataFrame, dict[str, Any]]:
                ds = load_dataset(ds_id, split="train")
                df = ds.to_pandas()
                if len(df) > MAX_DOWNLOAD_ROWS:
                    df = df.sample(MAX_DOWNLOAD_ROWS, random_state=42)
                meta = {
                    "hf_id": ds_id,
                    "format": "hf_dataset",
                    "n_rows_actual": int(df.shape[0]),
                    "n_columns_actual": int(df.shape[1]),
                }
                return df, meta

            return _load

        results.append(
            CandidateDataset(
                name=ds_id,
                source="huggingface",
                url=f"https://huggingface.co/datasets/{ds_id}",
                description=getattr(item, "description", None),
                license=getattr(item, "license", None),
                format="hf_dataset",
                loader=make_loader(),
            )
        )
    return results


def search_all_sources(
    query: str, spec_task: str | None = None, max_per_source: int = 8
) -> list[CandidateDataset]:
    """Aggregate candidates from every available source."""
    candidates: list[CandidateDataset] = []
    candidates += search_openml(query, max_results=max_per_source, task_hint=spec_task)
    candidates += search_huggingface(query, max_results=max_per_source)
    return candidates


def load_candidate(candidate: CandidateDataset) -> tuple[pd.DataFrame, dict[str, Any]]:
    if candidate.loader is None:
        raise ValueError(f"Candidate '{candidate.name}' has no loader")
    return candidate.loader()
