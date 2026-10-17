"""Shared helpers for the four AutoML agent roles."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

_TARGET_RE = re.compile(
    r"^\s*(?P<metric>[a-zA-Z_][a-zA-Z0-9_]*)\s*(?P<op>>=|<=|>|<|=)?\s*(?P<value>[0-9]+(?:\.[0-9]+)?)\s*$"
)

METRIC_SYNONYMS = {
    "acc": "accuracy",
    "f1": "f1",
    "auc": "roc_auc",
    "roc": "roc_auc",
    "roc_auc": "roc_auc",
    "precision": "precision",
    "recall": "recall",
    "rmse": "rmse",
    "mae": "mae",
    "mse": "mse",
    "r2": "r2",
    "r2_score": "r2",
    "logloss": "logloss",
}

MINIMIZE_METRICS = {"rmse", "mae", "mse", "logloss"}


@dataclass
class TargetSpec:
    metric: str = "accuracy"
    op: str = ">="
    value: float = 0.85
    direction: str = "maximize"

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "op": self.op,
            "value": self.value,
            "direction": self.direction,
        }


def parse_target_spec(
    raw: str | TargetSpec | dict[str, Any] | None, default_metric: str = "accuracy"
) -> TargetSpec:
    """Parse a user target like ``accuracy >= 0.90`` / ``RMSE <= 1.2`` into a spec."""
    if isinstance(raw, TargetSpec):
        raw = raw.to_dict()
    if isinstance(raw, dict):
        metric = METRIC_SYNONYMS.get(
            str(raw.get("metric", default_metric)).lower(),
            str(raw.get("metric", default_metric)).lower(),
        )
        value = float(raw.get("value", 0.0))
        op = str(raw.get("op", ">=" if metric not in MINIMIZE_METRICS else "<="))
        direction = str(
            raw.get("direction", "minimize" if metric in MINIMIZE_METRICS else "maximize")
        )
        return TargetSpec(metric=metric, op=op, value=value, direction=direction)

    text = (raw or "").strip()
    m = _TARGET_RE.match(text)
    if m:
        metric = METRIC_SYNONYMS.get(m.group("metric").lower(), m.group("metric").lower())
        op = m.group("op") or (">=" if metric not in MINIMIZE_METRICS else "<=")
        value = float(m.group("value"))
        return TargetSpec(
            metric=metric,
            op=op,
            value=value,
            direction="minimize" if metric in MINIMIZE_METRICS else "maximize",
        )

    metric = METRIC_SYNONYMS.get(text.lower(), text.lower() or default_metric)
    return TargetSpec(metric=metric)


def met_target(record_metrics: dict[str, Any], spec: TargetSpec) -> bool:
    """True when the recorded metrics satisfy the target spec."""
    value = record_metrics.get(spec.metric)
    if value is None or value != value:  # not present or NaN
        return False
    value = float(value)
    if spec.op == ">=":
        return value >= spec.value
    if spec.op == "<=":
        return value <= spec.value
    if spec.op == ">":
        return value > spec.value
    if spec.op == "<":
        return value < spec.value
    return value == spec.value


@dataclass
class Requirements:
    goal: str = ""
    target_column: str | None = None
    task_type: str = "auto"
    metric_expr: str = ""
    metric: str = "accuracy"
    target_value: float = 0.85
    direction: str = "maximize"
    max_rows: int = 50_000
    dataset_source: str = "synthetic"  # upload | local | openml | kaggle | synthetic | existing
    dataset_path: str = ""
    dataset_query: str = ""
    confirm: bool = False
    max_iterations: int = 12
    feature_engineering_every_n: int = 2
    meta: dict[str, Any] = field(default_factory=dict)

    def target_spec(self) -> TargetSpec:
        return TargetSpec(metric=self.metric, value=self.target_value, direction=self.direction)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "target_column": self.target_column,
            "task_type": self.task_type,
            "metric_expr": self.metric_expr,
            "metric": self.metric,
            "target_value": self.target_value,
            "direction": self.direction,
            "max_rows": self.max_rows,
            "dataset_source": self.dataset_source,
            "dataset_path": self.dataset_path,
            "dataset_query": self.dataset_query,
            "confirm": self.confirm,
            "max_iterations": self.max_iterations,
            "feature_engineering_every_n": self.feature_engineering_every_n,
        }


_TARGET_COLUMN_CANDIDATES = (
    "target",
    "label",
    "class",
    "y",
    "outcome",
    "result",
    "fraud",
    "price",
    "churn",
    "score",
)


def detect_task(
    df: pd.DataFrame, explicit_target: str | None = None
) -> tuple[str | None, str, dict[str, Any]]:
    """Detect the target column and task type from the single central dataset."""
    from ml.preprocessing.profiler import profile_dataset

    target = None
    if explicit_target and explicit_target in df.columns:
        target = explicit_target
    else:
        lower = [str(c).strip().lower() for c in df.columns]
        for cand in _TARGET_COLUMN_CANDIDATES:
            if cand in lower:
                target = df.columns[lower.index(cand)]
                break
        if target is None:
            # Last column as a pragmatic default.
            target = df.columns[-1]

    profile = profile_dataset(df, target_column=target)
    task_type = profile.get("target_analysis", {}).get("type", "classification")
    task_type = "classification" if task_type == "classification" else "regression"
    return target, task_type, profile


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        value = float(x)
        return default if np.isnan(value) else value
    except (TypeError, ValueError):
        return default


def json_ready(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_ready(v) for v in obj]
    return obj


def json_safe(obj: Any) -> Any:
    """Recursively coerce numpy scalars/arrays to plain Python types.

    FastAPI/pydantic cannot serialize ``numpy.float32`` etc., so every payload
    that crosses the API or the WebSocket boundary must pass through here.
    """
    converted = json_ready(obj)
    if isinstance(converted, dict):
        return {k: json_safe(v) for k, v in converted.items()}
    if isinstance(converted, list):
        return [json_safe(v) for v in converted]
    if isinstance(converted, float) and not math.isfinite(converted):
        return None
    return converted
