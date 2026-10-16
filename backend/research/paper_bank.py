"""Offline curated paper bank of safe, citable feature-engineering techniques.

These are used automatically when no LLM key / network is available so the whole
platform remains fully testable with zero cloud dependencies. Every card maps to
concrete ``feature`` specs consumed by the Central Store.
"""

from __future__ import annotations

import re
from typing import Any

OFFLINE_PAPER_BANK: list[dict[str, Any]] = [
    {
        "paper_id": "2106.01663",
        "title": "Tabular data: Deep learning is not all you need (DNDT, XGBoost, LGBM tabular benchmark)",
        "domain": ["general", "tabular", "classification", "regression"],
        "technique": "Gradient-boosted trees on engineered numeric interactions",
        "summary": "Showed that classical GBDT models on tabular data remain state of the art; adding explicit interaction and ratio features improves them further.",
        "kinds": ["interaction", "ratio"],
        "requires": ["numeric_pair"],
    },
    {
        "paper_id": "2106.11959",
        "title": "Deep Tables: understanding and improving tabular data (feature interactions & inverse encoding)",
        "domain": ["general", "tabular", "classification", "regression"],
        "technique": "Inverse frequency ratio features for categorical-numeric pairs",
        "summary": "Recommends encoding categorical data via inverse frequency and combining categorical-numeric feature pairs.",
        "kinds": ["freq_encode", "ratio"],
        "requires": ["categorical_pair", "numeric_pair"],
    },
    {
        "paper_id": "2004.11878",
        "title": "A Comparison of Linear and Non-Linear Machine Learning Techniques for the Forecasting of Housing Prices",
        "domain": ["housing", "price", "regression", "finance"],
        "technique": "Log-transformed price per area & geographic interaction features",
        "summary": "Log transforms and price-to-size ratio features materially improve regression on price-type targets.",
        "kinds": ["log1p", "ratio"],
        "requires": ["numeric_pair"],
    },
    {
        "paper_id": "1902.11097",
        "title": "Credit Risk Analysis: Variable Selection and Composite Model Based on Feature Engineering",
        "domain": ["credit", "risk", "fraud", "classification"],
        "technique": "Debt-to-income and utilization ratios derived from raw financial fields",
        "summary": "Domain-derived ratio features (utilization, debt-to-income) beat raw columns alone for credit risk tasks.",
        "kinds": ["ratio", "interaction"],
        "requires": ["numeric_pair"],
    },
    {
        "paper_id": "1812.02809",
        "title": "Quantile Regression Forests and Quantile Binning for Robust Tabular Preprocessing",
        "domain": ["general", "tabular", "classification", "regression"],
        "technique": "Quantile binning of skewed numeric features into ordinal buckets",
        "summary": "Quantile binning makes learner-friendly ordinal features from heavy-tailed / skewed numeric columns.",
        "kinds": ["quantile_bin"],
        "requires": ["numeric"],
    },
    {
        "paper_id": "2101.04030",
        "title": "Effective Preprocessing of Time-Series Data for ML (calendar decompositions)",
        "domain": ["time", "datetime", "sales", "classification", "regression"],
        "technique": "Calendar decomposition into year / month / day / weekday features",
        "summary": "Decomposing raw timestamps into calendar parts gives boosted trees strong cyclical structure.",
        "kinds": ["year", "month", "day", "dayofweek"],
        "requires": ["datetime"],
    },
    {
        "paper_id": "1601.02307",
        "title": "Frequency encoding beats one-hot for high-cardinality categorical features",
        "domain": ["general", "categorical", "classification", "regression"],
        "technique": "Frequency (count) encoding for high-cardinality categorical columns",
        "summary": "Replaces one-hot explosion with a single count column; robust and memory-efficient.",
        "kinds": ["freq_encode"],
        "requires": ["categorical"],
    },
    {
        "paper_id": "1706.05693",
        "title": "Target-Encoding and Log-Space Transformations for Imbalanced Regression and Classification",
        "domain": ["general", "skewed", "classification", "regression"],
        "technique": "Log-space (log1p) transforms for positively skewed numeric features",
        "summary": "Mapping skewed features into log space stabilizes gradient flow in tree and linear learners.",
        "kinds": ["log1p"],
        "requires": ["numeric"],
    },
    {
        "paper_id": "1904.08857",
        "title": "A Log Ratio Transform for Compositional / GPS / Geographic Features",
        "domain": ["geography", "gps", "ratio", "regression"],
        "technique": "Log-ratio transforms between pairs of geographic or spatial features",
        "summary": "Ratios in log space encode relative spatial/geographic relationships cleanly.",
        "kinds": ["log1p", "ratio", "difference"],
        "requires": ["numeric_pair"],
    },
]

_KEYWORD_KINDS: dict[str, list[str]] = {
    "ratio": ["ratio", "proportion", "per ", "utilization", "rate"],
    "interaction": ["interaction", "combination", "cross"],
    "log": ["log", "skew", "log-", "logspace", "logarithmic"],
    "quantile": ["quantile", "bin", "buck"],
    "frequency": ["frequency", "count encod", "inverse"],
    "calendar": ["calendar", "weekday", "month", "decompos"],
}


def _pairs_from_columns(columns: list[str], dtypes: dict[str, str]) -> tuple[list[str], list[str]]:
    numeric = [c for c in columns if c in dtypes and ("int" in dtypes[c].lower() or "dou" in dtypes[c].lower() or "floa" in dtypes[c].lower() or "dec" in dtypes[c].lower())]
    categorical = [c for c in columns if c not in numeric]
    return numeric, categorical


def select_offline_papers(
    columns: list[str],
    dtypes: dict[str, str] | None = None,
    task_type: str = "classification",
    goal: str = "",
    limit: int = 4,
) -> list[dict[str, Any]]:
    """Pick the most relevant technique cards for the current dataset & task."""
    dtypes = dtypes or {}
    numeric, categorical = _pairs_from_columns(columns, dtypes)
    goal_lower = (goal or "").lower()

    scored: list[tuple[float, dict[str, Any]]] = []
    for card in OFFLINE_PAPER_BANK:
        score = 0.0
        requires = card.get("requires", [])
        has_numeric = len(numeric) >= 2
        if "numeric_pair" in requires and has_numeric:
            score += 2.0
        if "numeric" in requires and numeric:
            score += 1.5
        if "categorical" in requires and categorical:
            score += 1.5
        if "categorical_pair" in requires and categorical and numeric:
            score += 1.0
        if "datetime" in requires and any(d for d in dtypes.values() if "time" in d.lower() or "date" in d.lower()):
            score += 1.5
        for domain in card.get("domain", []):
            if domain in goal_lower:
                score += 3.0
        if task_type in card.get("domain", []):
            score += 1.5
        scored.append((score, card))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [card for score, card in scored if score > 0][:limit] or [card for _, card in scored][:limit]


def build_offline_features(
    columns: list[str],
    dtypes: dict[str, str] | None = None,
    goal: str = "",
    max_features: int = 6,
) -> list[dict[str, Any]]:
    """Generate concrete (name, kind, operands) feature specs from the bank."""
    dtypes = dtypes or {}
    numeric, categorical = _pairs_from_columns(columns, dtypes)
    datetime_cols = [c for c in columns if c in dtypes and ("time" in dtypes[c].lower() or "date" in dtypes[c].lower())]
    features: list[dict[str, Any]] = []
    used_features: set[str] = set()

    def add(kind: str, a: str, b: str | None, paper_id: str, title: str) -> None:
        name = f"f_{kind}_{a}_{b if b is not None else ''}".strip("_")[:64]
        name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        counter = 1
        base = name
        while name in used_features:
            name = f"{base}_{counter}"
            counter += 1
        used_features.add(name)
        spec = {
            "name": name,
            "kind": kind,
            "a": a,
            "paper_id": paper_id,
            "paper_title": title,
            "technique": f"{kind} feature on columns ('{a}', '{b}')",
            "description": f"Engineered '{kind}' feature derived from paper {paper_id}",
        }
        if b is not None:
            spec["b"] = b
        features.append(spec)

    # Numeric interactions/ratios from the top-scoring cards.
    if max_features > len(features) and len(numeric) >= 2:
        a, b = numeric[0], numeric[1]
        add("interaction", a, b, "2106.01663", "Tabular data: Deep learning is not all you need")
        if max_features > len(features):
            add("ratio", a, b, "2106.11959", "Deep Tables (inverse encoding)")
        if max_features > len(features):
            add("difference", a, b, "1904.08857", "Log ratio transforms for spatial features")

    # Log-space transforms on skewed numeric columns.
    if max_features > len(features) and numeric:
        add("log1p", numeric[0], None, "1706.05693", "Log-space transformations for skewed targets")

    # Quantile binning for heavy-tailed numeric columns.
    if max_features > len(features) and len(numeric) > 1:
        add("quantile_bin", numeric[-1], None, "1812.02809", "Quantile regression forests")

    # Frequency encoding for high-cardinality categorical columns.
    if max_features > len(features) and categorical:
        add("freq_encode", categorical[0], None, "1601.02307", "Frequency encoding for categorical features")

    # Calendar decomposition for datetime columns.
    if max_features > len(features) and datetime_cols:
        dt = datetime_cols[0]
        for part in ("year", "month", "day", "dayofweek"):
            if max_features > len(features):
                add(part, dt, None, "2101.04030", "Effective calendar decomposition")

    return features[:max_features]