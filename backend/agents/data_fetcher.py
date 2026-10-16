"""Agent 2: Data Fetcher & Central Store Agent.

Ingests data from an upload / local path / OpenML / Kaggle / synthetic demo,
applies the max-row limit (random/stratified sampling), and loads it into the
single unified table via :class:`backend.central_store.CentralStore`. Every
subsequent mutation happens on that store — no duplicated CSV versions on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from backend.agents.common import Requirements, detect_task
from backend.central_store import get_central_store


def build_synthetic_dataset(goal: str = "", rows: int = 2_000, task_type: str = "auto") -> pd.DataFrame:
    """Generate a deterministic demo dataset with numeric, categorical & datetime cols."""
    goal = goal.lower()
    rng = np.random.RandomState(42)

    if "regress" in task_type or any(k in goal for k in ("price", "salary", "sales", "score", "temperature")):
        base = [
            rng.normal(6_000, 1_200, rows),
            rng.lognormal(3.2, 0.55, rows),
            rng.randint(0, 6, rows),
            rng.randint(18, 85, rows),
            rng.uniform(40, 120, rows),
        ]
        target = 20_000 + 2.1 * base[0] - 180 * base[2] + 0.9 * base[3] + 25 * base[4] + rng.normal(0, 1_500, rows)
        task = "regression"
        names = ["size_sqft", "income", "bedrooms", "age", "quality"]
    else:
        base = [
            rng.normal(0, 1, rows),
            rng.normal(0, 1, rows),
            rng.randint(0, 4, rows),
            rng.randint(18, 90, rows),
            rng.uniform(0, 100, rows),
        ]
        prob = 1 / (1 + np.exp(-(0.4 * base[0] - 0.3 * base[1] + 0.05 * (base[1] * base[2]) - 0.6)))
        target = rng.binomial(1, prob, rows)
        task = "classification"
        names = ["feature_a", "feature_b", "segment", "age", "spend"]

    df = pd.DataFrame(
        {
            names[0]: base[0],
            names[1]: base[1],
            names[2]: base[2].astype(int),
            names[3]: base[3],
            names[4]: base[4],
            "category": [["low", "med", "high", "premium"][i] for i in rng.randint(0, 4, rows)],
            "target": target if task == "classification" else target.round(2),
        }
    )
    return df


class DataFetcherAgent:
    def __init__(self) -> None:
        self.store = get_central_store()
        self.loaded_from: str = ""

    def fetch(
        self,
        requirements: Requirements,
        emit: Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]:
        """Load the dataset into the single central store and detect the ML task."""
        emit({"type": "agent_message", "agent": "data-fetcher", "text": "Ingesting dataset into the single central store…"})

        df, source = self._obtain(requirements)
        n_before = len(df)

        if df is None or len(df.columns) == 0 or len(df) == 0:
            raise ValueError(
                f"Dataset from {source} is empty (0 rows or 0 columns) — nothing to train on."
            )
        explicit = (requirements.target_column or "").strip() if requirements.target_column else ""
        if explicit and explicit not in df.columns:
            raise ValueError(
                f"Target column '{explicit}' not found in the dataset. "
                f"Available columns: {', '.join(str(c) for c in df.columns)}."
            )

        if requirements.max_rows and n_before > requirements.max_rows:
            df = df.sample(n=requirements.max_rows, random_state=42)
            emit(
                {
                    "type": "agent_message",
                    "agent": "data-fetcher",
                    "text": f"Applied max-row limit: sampled {len(df):,} of {n_before:,} rows (random).",
                }
            )

        loaded = self.store.load_dataframe(df, max_rows=requirements.max_rows)
        self.loaded_from = source

        target, task_type, profile = detect_task(df, explicit_target=requirements.target_column)
        if df[target].isna().all():
            raise ValueError(
                f"Target column '{target}' contains only missing values — pick a different target."
            )
        requirements.target_column = target
        if requirements.task_type in ("auto", ""):
            requirements.task_type = task_type

        emit(
            {
                "type": "dataset_preview",
                "event": "dataset_preview",
                "data": self.store.preview(limit=25),
                "loaded_from": source,
            }
        )
        emit(
            {
                "type": "agent_message",
                "agent": "data-fetcher",
                "text": (
                    f"Dataset ready in {self.store.backend} table '{self.store.table_name}': "
                    f"{loaded['rows']:,} rows x {len(loaded['columns'])} columns from {source}. "
                    f"Detected target='{target}', task={task_type}. "
                    f"Feature-engineering will mutate this same table in place."
                ),
            }
        )
        return {
            "source": source,
            "rows": loaded["rows"],
            "columns": loaded["columns"],
            "target_column": target,
            "task_type": task_type,
            "profile": profile,
        }

    def _obtain(self, requirements: Requirements) -> tuple[pd.DataFrame, str]:
        src = requirements.dataset_source
        path = requirements.dataset_path or ""
        if src in ("upload", "local") and path:
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"Dataset file not found: {path}")
            df = self._read_file(p)
            return df, f"file {p.name}"
        if src == "kaggle":
            df = self._from_kaggle(requirements.dataset_query)
            if df is not None:
                return df, "kaggle"
        if src == "openml":
            df = self._from_openml(requirements.dataset_query, requirements.goal)
            if df is not None:
                return df, "openml"
        if src == "existing":
            return self.store.to_dataframe(), "existing central store"
        return build_synthetic_dataset(requirements.goal, max(requirements.max_rows, 2_000), requirements.task_type), "synthetic demo"

    @staticmethod
    def _read_file(p: Path) -> pd.DataFrame:
        suffix = p.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(p)
        if suffix in (".parquet", ".pq"):
            return pd.read_parquet(p)
        if suffix in (".xlsx", ".xls"):
            return pd.read_excel(p)
        if suffix in (".json", ".jsonl"):
            return pd.read_json(p) if suffix == ".json" else pd.read_json(p, lines=True)
        raise ValueError(f"Unsupported dataset format: {suffix}")

    def _from_kaggle(self, query: str) -> pd.DataFrame | None:
        from backend.config import settings

        if not (settings.kaggle_username and settings.kaggle_key):
            return None
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            return None
        # opendatasets would be the smoother path, but we keep it dependency-light:
        # require a 'kaggle://owner/dataset/file.csv' style path or a local download dir.
        if not query or not query.startswith("kaggle://"):
            return None
        # e.g. kaggle://titanic/train.csv → dataset 'titanic', file 'train.csv'
        _, rel = query.split("://", 1)
        parts = rel.split("/")
        if not parts:
            return None
        api = KaggleApi()
        import os

        os.environ["KAGGLE_USERNAME"] = settings.kaggle_username
        os.environ["KAGGLE_KEY"] = settings.kaggle_key
        api.authenticate()
        base = Path("./automl_outputs/kaggle")
        api.dataset_download_files(parts[0], path=base, unzip=True)
        return self._read_file(base / (parts[1] if len(parts) > 1 else f"{parts[0].split('/')[-1]}.csv"))

    def _from_openml(self, query: str, goal: str) -> pd.DataFrame | None:
        try:
            from ml.dataset_acquisition.sources import search_openml, load_candidate
        except ImportError:
            return None
        try:
            candidates = search_openml(query or goal, max_results=5)
            if not candidates:
                return None
            df, _ = load_candidate(candidates[0])
            return df
        except Exception:
            return None

    def ensure_columns(self) -> list[str]:
        return self.store.columns()