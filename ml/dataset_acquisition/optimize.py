from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ml.dataset_acquisition.evaluate import FeatureClass

warnings.filterwarnings("ignore")

KEEP_CLASSES = {
    FeatureClass.ESSENTIAL,
    FeatureClass.HIGHLY_USEFUL,
    FeatureClass.USEFUL,
    FeatureClass.POTENTIALLY_USEFUL,
}
DROP_CLASSES = {
    FeatureClass.LEAKAGE,
    FeatureClass.IDENTIFIER,
    FeatureClass.CONSTANT,
    FeatureClass.INVALID,
}


@dataclass
class OptimizationResult:
    df: pd.DataFrame
    target: str
    selected_features: list[str] = field(default_factory=list)
    removed_features: list[dict[str, str]] = field(default_factory=list)
    engineered_features: list[dict[str, str]] = field(default_factory=list)
    transformation_log: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    requested_features: int | None = None
    final_input_feature_count: int = 0


def _importance_key(a) -> float:
    if a.correlation_with_target is not None:
        return abs(a.correlation_with_target)
    order = {
        FeatureClass.ESSENTIAL: 3, FeatureClass.HIGHLY_USEFUL: 2.5,
        FeatureClass.USEFUL: 2, FeatureClass.POTENTIALLY_USEFUL: 1,
    }
    return order.get(a.classification, 0)


def optimize_dataset(
    df: pd.DataFrame,
    target: str,
    assessments: list[Any],
    requested_input_features: int | None = None,
    encode_categorical: bool = True,
    clip_outliers: bool = True,
    scale: bool = False,
) -> OptimizationResult:
    log: list[str] = []
    warns: list[str] = []
    df = df.copy()
    original_n = len(df)

    # 1. drop constant / identifier / leakage features
    assoc = {a.name: a for a in assessments}
    removed: list[dict[str, str]] = []
    keep_cols: list[str] = []
    for a in assessments:
        if a.classification in DROP_CLASSES:
            removed.append({"feature": a.name, "reason": a.reason, "action": "dropped"})
            log.append(f"Dropped feature '{a.name}': {a.reason}")
        else:
            keep_cols.append(a.name)

    df = df[[c for c in keep_cols if c in df.columns] + ([target] if target in df.columns else [])]

    # 2. select top-N requested features among keepable
    selected = [c for c in keep_cols if c != target]
    if requested_input_features is not None:
        ranked = sorted(
            [a for a in assessments if a.name in selected],
            key=_importance_key,
            reverse=True,
        )
        if len(ranked) >= requested_input_features:
            selected = [a.name for a in ranked[:requested_input_features]]
        else:
            warns.append(
                f"Only {len(ranked)} high-quality features available; "
                f"requested {requested_input_features}. Returning all available."
            )
            selected = [a.name for a in ranked]

    # 3. handle missing values (median/mode) on selected
    for col in selected:
        series = df[col]
        if series.isnull().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(series):
            df[col] = series.fillna(series.median())
            log.append(f"Imputed '{col}' with median")
        else:
            mode = series.mode()
            df[col] = series.fillna(mode.iloc[0] if len(mode) else "missing")
            log.append(f"Imputed '{col}' with mode")

    # 4. handle exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        log.append(f"Removed {before - len(df)} duplicate rows")

    # 5. handle outliers (clip) for numeric selected features
    if clip_outliers:
        for col in selected:
            series = df[col]
            if not pd.api.types.is_numeric_dtype(series):
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = int(((series < lb) | (series > ub)).sum())
            if n_out / max(1, len(series)) > 0.05:
                df[col] = series.clip(lb, ub)
                log.append(f"Clipped {n_out} outliers in '{col}' (IQR bounds)")

    # 6. categorical encoding
    engineered: list[dict[str, str]] = []
    if encode_categorical:
        encoded_cols = []
        for col in selected:
            series = df[col]
            if not pd.api.types.is_numeric_dtype(series):
                n_unique = series.nunique()
                if n_unique <= 10:
                    dummies = pd.get_dummies(series, prefix=col, drop_first=True)
                    df = pd.concat([df, dummies], axis=1)
                    encoded_cols.extend(dummies.columns.tolist())
                    log.append(f"One-hot encoded '{col}' ({n_unique} categories)")
                else:
                    mp = {v: i for i, v in enumerate(series.astype(str).unique())}
                    df[col + "_ord"] = series.astype(str).map(mp)
                    encoded_cols.append(col + "_ord")
                    log.append(f"Ordinal encoded '{col}' (high cardinality)")
                df = df.drop(columns=[col])
        selected = [c for c in selected if c not in df.columns or pd.api.types.is_numeric_dtype(df[c])]
        selected = [c for c in df.columns if c != target and (c in selected or c in encoded_cols)] + encoded_cols
        selected = list(dict.fromkeys(selected))

    # 7. optional scaling
    if scale:
        from sklearn.preprocessing import StandardScaler

        num = [c for c in selected if pd.api.types.is_numeric_dtype(df[c])]
        if num:
            df[num] = StandardScaler().fit_transform(df[num])
            log.append("Applied StandardScaler to numeric features")

    final_feats = [c for c in selected if c in df.columns and c != target]
    result = OptimizationResult(
        df=df,
        target=target,
        selected_features=final_feats,
        removed_features=removed,
        engineered_features=engineered,
        transformation_log=log,
        warnings=warns,
        requested_features=requested_input_features,
        final_input_feature_count=len(final_feats),
    )
    return result
