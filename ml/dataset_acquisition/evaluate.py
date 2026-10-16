from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class FeatureClass(StrEnum):
    ESSENTIAL = "essential"
    HIGHLY_USEFUL = "highly_useful"
    USEFUL = "useful"
    POTENTIALLY_USEFUL = "potentially_useful"
    REDUNDANT = "redundant"
    WEAK = "weak"
    IRRELEVANT = "irrelevant"
    DANGEROUS = "dangerous"
    LEAKAGE = "leakage"
    IDENTIFIER = "identifier"
    CONSTANT = "constant"
    INVALID = "invalid"


@dataclass
class FeatureAssessment:
    name: str
    dtype: str
    missing_pct: float = 0.0
    unique_count: int = 0
    cardinality: float = 0.0
    variance: float = 0.0
    outlier_pct: float = 0.0
    correlation_with_target: float | None = None
    importance_score: float = 0.0
    classification: FeatureClass = FeatureClass.USEFUL
    reason: str = ""


def inspect_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Verify actual data (never trust metadata)."""
    n_rows, n_cols = df.shape
    dtypes = df.dtypes.astype(str).to_dict()
    missing = {c: round(df[c].isnull().mean() * 100, 2) for c in df.columns}
    dup_rows = int(df.duplicated().sum())
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    near_constant = [
        c for c in df.columns
        if 0 < df[c].nunique(dropna=True) <= max(2, int(0.01 * n_rows)) and c not in constant_cols
    ]
    # duplicate columns (same values)
    dup_cols = []
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            if df.iloc[:, i].equals(df.iloc[:, j]):
                dup_cols.append((df.columns[i], df.columns[j]))
    return {
        "n_rows": int(n_rows),
        "n_columns": int(n_cols),
        "dtypes": dtypes,
        "missing_pct": missing,
        "duplicate_rows": dup_rows,
        "constant_columns": constant_cols,
        "near_constant_columns": near_constant,
        "duplicate_column_pairs": dup_cols,
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
    }


def identify_target(
    df: pd.DataFrame, target_hint: str | None, task: str
) -> dict[str, Any]:
    """Select the most appropriate target variable (step 5)."""
    candidates: list[str] = []
    if target_hint and target_hint in df.columns:
        candidates = [target_hint]
    else:
        # heuristic ranking: prefer columns matching common target names (word-boundary)
        import re

        name_priority = ["target", "label", "class", "output", "result", "status", "outcome", "y"]
        for c in df.columns:
            cl = c.lower()
            if any(re.search(rf"\b{re.escape(p)}\b", cl) for p in name_priority):
                candidates.append(c)
        if not candidates:
            candidates = list(df.columns)

    ranked: list[dict[str, Any]] = []
    for col in candidates:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        n_unique = series.nunique()
        n_samples = len(series)
        identifier_like = (n_unique / max(1, n_samples)) > 0.95

        if task == "regression":
            ttype = "continuous"
        elif is_numeric and (n_unique > 20 or n_unique / n_samples >= 0.05):
            ttype = "continuous"  # numeric w/ many values -> treat as regression target
        else:
            ttype = "binary" if n_unique == 2 else "multiclass"

        if ttype == "continuous":
            dist = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
            }
        else:
            vc = series.value_counts()
            dist = {
                "n_classes": int(n_unique),
                "class_distribution": vc.to_dict(),
                "min_class_pct": round(vc.min() / len(series) * 100, 2),
                "max_class_pct": round(vc.max() / len(series) * 100, 2),
            }
        quality = _target_quality_score(series, ttype)
        # Penalize identifier-like targets (unique per row) and, for a
        # classification task, discourage continuous targets.
        if identifier_like:
            quality -= 0.6
        if task == "classification" and ttype == "continuous":
            quality -= 0.5
        ranked.append(
            {
                "column": col,
                "type": ttype,
                "distribution": dist,
                "quality_score": round(quality, 3),
                "missing_pct": round(df[col].isnull().mean() * 100, 2),
            }
        )

    ranked.sort(key=lambda x: x["quality_score"], reverse=True)
    best = ranked[0] if ranked else None
    return {"best_target": best, "ranked_targets": ranked}


def _target_quality_score(series: pd.Series, ttype: str) -> float:
    score = 1.0
    n = len(series)
    missing = series.isnull().mean()
    score -= missing * 0.5
    if ttype in ("binary", "multiclass"):
        vc = series.value_counts(normalize=True)
        # penalize severe imbalance
        score -= max(0, 0.5 - vc.min()) * 0.5
    if n < 100:
        score -= 0.3
    return float(max(0.0, score))


def analyze_features(
    df: pd.DataFrame, target: str | None, task: str
) -> list[FeatureAssessment]:
    """Per-feature analysis (steps 6, 7, 8)."""
    assessments: list[FeatureAssessment] = []
    n = len(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)

    # correlation with target
    target_corr: dict[str, float] = {}
    if target and target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
        for c in numeric_cols:
            try:
                corr = df[c].corr(df[target])
                if pd.notna(corr):
                    target_corr[c] = float(corr)
            except Exception:
                pass

    for col in df.columns:
        if col == target:
            continue
        series = df[col]
        missing_pct = round(series.isnull().mean() * 100, 2)
        unique = int(series.nunique(dropna=True))
        card = round(unique / n, 4) if n else 0.0
        is_numeric = pd.api.types.is_numeric_dtype(series)
        variance = float(series.var()) if is_numeric else 0.0
        outlier_pct = 0.0
        if is_numeric:
            clean = series.dropna()
            if len(clean) > 0:
                q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
                iqr = q3 - q1
                out = clean[(clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)]
                outlier_pct = round(len(out) / len(clean) * 100, 2)

        cls, reason = _classify_feature(
            col, series, missing_pct, unique, n, is_numeric, target_corr.get(col), task
        )
        imp = abs(target_corr[col]) if target_corr.get(col) is not None else (
            1.0 if cls == FeatureClass.ESSENTIAL else 0.6 if cls == FeatureClass.HIGHLY_USEFUL
            else 0.4 if cls == FeatureClass.USEFUL else 0.2 if cls == FeatureClass.POTENTIALLY_USEFUL else 0.0
        )
        assessments.append(
            FeatureAssessment(
                name=col,
                dtype=str(series.dtype),
                missing_pct=missing_pct,
                unique_count=unique,
                cardinality=card,
                variance=variance,
                outlier_pct=outlier_pct,
                correlation_with_target=target_corr.get(col),
                importance_score=round(imp, 4),
                classification=cls,
                reason=reason,
            )
        )
    return assessments


def _classify_feature(
    name: str, series: pd.Series, missing_pct: float, unique: int, n: int,
    is_numeric: bool, corr: float | None, task: str,
) -> tuple[FeatureClass, str]:
    col = name.lower()
    id_patterns = ["id", "uuid", "user_id", "customer_id", "transaction_id", "row_number", "index_col"]
    leakage_patterns = [
        "target", "label", "probability", "prediction", "future", "next",
        "ground_truth", "y_", "_y", "leak", "is_default", "will_default",
    ]
    # constant
    if unique <= 1:
        return FeatureClass.CONSTANT, "Column has a single unique value"
    # identifier: only non-numeric high-cardinality OR explicit id-like names
    name_is_id = any(re.search(rf"\b{p}\b", col) for p in id_patterns)
    nonnumeric_high_card = (not is_numeric) and (n > 0 and unique / n > 0.95)
    if name_is_id or nonnumeric_high_card:
        return FeatureClass.IDENTIFIER, "Appears to be a unique identifier (high cardinality)"
    # leakage by name
    if any(p in col for p in leakage_patterns) and col not in ("id",):
        return FeatureClass.LEAKAGE, "Name suggests it reveals the target / outcome"
    # leakage by correlation
    if corr is not None and abs(corr) > 0.97:
        return FeatureClass.LEAKAGE, f"Near-perfect correlation with target ({corr:.2f})"
    # missingness
    if missing_pct > 60:
        return FeatureClass.WEAK, f"High missingness ({missing_pct}%)"
    # importance roughly by |corr|
    if corr is not None:
        if abs(corr) >= 0.3:
            return FeatureClass.ESSENTIAL, f"Strong relationship with target (corr={corr:.2f})"
        if abs(corr) >= 0.1:
            return FeatureClass.USEFUL, f"Moderate relationship with target (corr={corr:.2f})"
        return FeatureClass.WEAK, f"Weak relationship with target (corr={corr:.2f})"
    # categorical high/low cardinality heuristic
    if not is_numeric:
        if unique <= 2:
            return FeatureClass.USEFUL, "Low-cardinality categorical feature"
        if unique > 0.5 * n:
            return FeatureClass.POTENTIALLY_USEFUL, "High-cardinality categorical; may need encoding"
        return FeatureClass.USEFUL, "Categorical feature"
    return FeatureClass.POTENTIALLY_USEFUL, "Numeric feature with unclear predictive value"


def detect_leakage(assessments: list[FeatureAssessment]) -> list[str]:
    return [a.name for a in assessments if a.classification == FeatureClass.LEAKAGE]


def score_candidate(
    inspection: dict[str, Any],
    target_info: dict[str, Any],
    assessments: list[FeatureAssessment],
    spec: Any,
    source_docs: str | None = None,
    license: str | None = None,
) -> dict[str, Any]:
    """Weighted dataset score (step 20)."""
    n_rows = inspection["n_rows"]
    n_cols = inspection["n_columns"]

    # Relevance — based on target quality & presence
    relevance = 0.8 if target_info.get("best_target") else 0.3

    # Feature quality — fraction of useful/essential features
    useful = [a for a in assessments if a.classification in (
        FeatureClass.ESSENTIAL, FeatureClass.HIGHLY_USEFUL, FeatureClass.USEFUL,
        FeatureClass.POTENTIALLY_USEFUL,
    )]
    feature_quality = len(useful) / max(1, len(assessments))

    # Completeness — low missingness
    avg_missing = np.mean(list(inspection["missing_pct"].values())) if inspection["missing_pct"] else 0
    completeness = max(0.0, 1.0 - avg_missing / 100)

    # Target quality
    target_quality = (target_info.get("best_target") or {}).get("quality_score", 0.0)

    # Sample size adequacy
    req_rows = getattr(spec, "requested_rows", None) or getattr(spec, "min_acceptable_rows", None)
    if req_rows:
        sample_quality = min(1.0, n_rows / req_rows)
    else:
        sample_quality = min(1.0, n_rows / 1000)

    # Leakage risk (inverse of leakage fraction)
    leakage_count = sum(1 for a in assessments if a.classification == FeatureClass.LEAKAGE)
    leakage_risk = max(0.0, 1.0 - leakage_count / max(1, len(assessments)))

    # Documentation / license
    doc_quality = 0.7 if source_docs else 0.4
    license_quality = 0.8 if license and license.lower() not in ("unknown", "none", "") else 0.4

    weights = {
        "relevance": 0.30,
        "feature_quality": 0.15,
        "completeness": 0.15,
        "target_quality": 0.10,
        "sample_quality": 0.10,
        "leakage_risk": 0.05,
        "representativeness": 0.05,
        "documentation": 0.05,
        "license": 0.05,
    }
    scores = {
        "relevance": relevance,
        "feature_quality": feature_quality,
        "completeness": completeness,
        "target_quality": target_quality,
        "sample_quality": sample_quality,
        "leakage_risk": leakage_risk,
        "representativeness": 0.6,
        "documentation": doc_quality,
        "license": license_quality,
    }
    overall = sum(scores[k] * weights[k] for k in weights)
    return {
        "overall_score": round(overall, 4),
        "component_scores": {k: round(v, 4) for k, v in scores.items()},
        "weights": weights,
    }


def rank_candidates(scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(scored, key=lambda x: x["score"]["overall_score"], reverse=True)
