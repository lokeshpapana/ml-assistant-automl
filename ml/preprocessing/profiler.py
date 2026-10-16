from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


class DatasetProfiler:
    def __init__(self, df: pd.DataFrame, target_column: str | None = None):
        self.df = df.copy()
        self.target_column = target_column
        self.n_rows, self.n_cols = df.shape
        self.profile: dict[str, Any] = {}

    def run(self) -> dict[str, Any]:
        self.profile = {
            "overview": self._get_overview(),
            "columns": self._profile_columns(),
            "correlations": self._get_correlations(),
            "data_quality": self._get_data_quality(),
            "target_analysis": self._analyze_target() if self.target_column else None,
            "recommendations": self._get_recommendations(),
        }
        return self.profile

    def _get_overview(self) -> dict[str, Any]:
        return {
            "n_rows": int(self.n_rows),
            "n_columns": int(self.n_cols),
            "memory_usage_mb": round(self.df.memory_usage(deep=True).sum() / 1024**2, 2),
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "duplicate_rows": int(self.df.duplicated().sum()),
        }

    def _profile_columns(self) -> dict[str, dict[str, Any]]:
        columns = {}
        for col in self.df.columns:
            series = self.df[col]
            col_profile: dict[str, Any] = {
                "dtype": str(series.dtype),
                "missing_count": int(series.isnull().sum()),
                "missing_pct": round(series.isnull().mean() * 100, 2),
                "unique_count": int(series.nunique()),
                "unique_pct": round(series.nunique() / len(series) * 100, 2),
            }

            if pd.api.types.is_numeric_dtype(series):
                col_profile.update(self._profile_numeric(series))
            elif pd.api.types.is_datetime64_any_dtype(series):
                col_profile.update(self._profile_datetime(series))
            else:
                col_profile.update(self._profile_categorical(series))

            columns[col] = col_profile
        return columns

    def _profile_numeric(self, series: pd.Series) -> dict[str, Any]:
        clean = series.dropna()
        if len(clean) == 0:
            return {"stats": None, "outliers": None, "distribution": None}

        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = clean[(clean < lower) | (clean > upper)]

        return {
            "stats": {
                "mean": float(clean.mean()),
                "std": float(clean.std()),
                "min": float(clean.min()),
                "max": float(clean.max()),
                "median": float(clean.median()),
                "q1": float(q1),
                "q3": float(q3),
                "skewness": float(stats.skew(clean)),
                "kurtosis": float(stats.kurtosis(clean)),
            },
            "outliers": {
                "count": int(len(outliers)),
                "pct": round(len(outliers) / len(clean) * 100, 2),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
            },
            "distribution": self._get_distribution_type(clean),
        }

    def _profile_datetime(self, series: pd.Series) -> dict[str, Any]:
        clean = series.dropna()
        if len(clean) == 0:
            return {"range": None}
        return {
            "min": clean.min().isoformat(),
            "max": clean.max().isoformat(),
            "range_days": (clean.max() - clean.min()).days,
        }

    def _profile_categorical(self, series: pd.Series) -> dict[str, Any]:
        clean = series.dropna()
        if len(clean) == 0:
            return {"top_values": None, "entropy": None}

        value_counts = clean.value_counts()
        top_10 = value_counts.head(10)
        entropy = stats.entropy(value_counts.values) if len(value_counts) > 1 else 0

        return {
            "top_values": top_10.to_dict(),
            "entropy": float(entropy),
            "is_high_cardinality": len(value_counts) > 50,
            "is_constant": len(value_counts) == 1,
            "is_id_like": len(value_counts) / len(clean) > 0.9,
        }

    def _get_distribution_type(self, series: pd.Series) -> str:
        if len(series) < 20:
            return "insufficient_data"
        _, p_norm = stats.normaltest(series)
        if p_norm > 0.05:
            return "normal"
        _, p_exp = stats.kstest(series, "expon", args=(series.min(), series.mean() - series.min()))
        if p_exp > 0.05:
            return "exponential"
        return "unknown"

    def _get_correlations(self) -> dict[str, Any]:
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return {"matrix": None, "high_correlation_pairs": None}

        corr_matrix = self.df[numeric_cols].corr(method="pearson")
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                val = corr_matrix.iloc[i, j]
                if abs(val) > 0.8:
                    high_corr.append(
                        {
                            "col1": corr_matrix.columns[i],
                            "col2": corr_matrix.columns[j],
                            "correlation": round(float(val), 4),
                        }
                    )

        return {
            "matrix": corr_matrix.round(4).to_dict(),
            "high_correlation_pairs": high_corr,
        }

    def _get_data_quality(self) -> dict[str, Any]:
        issues = []

        for col in self.df.columns:
            series = self.df[col]
            missing_pct = series.isnull().mean() * 100
            if missing_pct > 50:
                issues.append(
                    {
                        "type": "high_missing",
                        "column": col,
                        "severity": "high",
                        "message": f"Column '{col}' has {missing_pct:.1f}% missing values",
                    }
                )
            elif missing_pct > 0:
                issues.append(
                    {
                        "type": "missing_values",
                        "column": col,
                        "severity": "medium",
                        "message": f"Column '{col}' has {missing_pct:.1f}% missing values",
                    }
                )

            if pd.api.types.is_numeric_dtype(series):
                clean = series.dropna()
                if len(clean) > 0:
                    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
                    iqr = q3 - q1
                    outliers = clean[(clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)]
                    if len(outliers) / len(clean) > 0.05:
                        issues.append(
                            {
                                "type": "many_outliers",
                                "column": col,
                                "severity": "medium",
                                "message": f"Column '{col}' has {len(outliers)} outliers ({len(outliers)/len(clean)*100:.1f}%)",
                            }
                        )

        duplicate_pct = self.df.duplicated().mean() * 100
        if duplicate_pct > 1:
            issues.append(
                {
                    "type": "duplicates",
                    "column": "all",
                    "severity": "medium",
                    "message": f"Dataset has {duplicate_pct:.1f}% duplicate rows",
                }
            )

        return {
            "issues": issues,
            "issue_count": len(issues),
            "high_severity_count": len([i for i in issues if i["severity"] == "high"]),
        }

    def _analyze_target(self) -> dict[str, Any] | None:
        if self.target_column not in self.df.columns:
            return None

        target = self.df[self.target_column]
        clean = target.dropna()

        is_numeric = pd.api.types.is_numeric_dtype(target)
        n_unique = clean.nunique()
        n_samples = len(clean)

        is_classification = not is_numeric or (is_numeric and n_unique <= 20 and n_unique / n_samples < 0.05)

        if is_classification:
            value_counts = clean.value_counts()
            return {
                "type": "classification",
                "n_classes": len(value_counts),
                "class_distribution": value_counts.to_dict(),
                "class_balance": {
                    "min_class_pct": round(value_counts.min() / len(clean) * 100, 2),
                    "max_class_pct": round(value_counts.max() / len(clean) * 100, 2),
                    "is_imbalanced": value_counts.min() / value_counts.max() < 0.1,
                },
                "missing_pct": round(target.isnull().mean() * 100, 2),
            }
        else:
            return {
                "type": "regression",
                "stats": {
                    "mean": float(clean.mean()),
                    "std": float(clean.std()),
                    "min": float(clean.min()),
                    "max": float(clean.max()),
                    "median": float(clean.median()),
                    "skewness": float(stats.skew(clean)),
                },
                "missing_pct": round(target.isnull().mean() * 100, 2),
            }

    def _get_recommendations(self) -> list[dict[str, Any]]:
        recs = []

        for col in self.df.columns:
            series = self.df[col]
            missing_pct = series.isnull().mean() * 100

            if missing_pct > 50:
                recs.append(
                    {
                        "type": "drop_column",
                        "column": col,
                        "reason": f"High missing rate ({missing_pct:.1f}%)",
                        "priority": "high",
                    }
                )
            elif missing_pct > 0:
                if pd.api.types.is_numeric_dtype(series):
                    recs.append(
                        {
                            "type": "impute_numeric",
                            "column": col,
                            "method": "median" if missing_pct > 10 else "mean",
                            "reason": f"Missing values ({missing_pct:.1f}%)",
                            "priority": "medium",
                        }
                    )
                else:
                    recs.append(
                        {
                            "type": "impute_categorical",
                            "column": col,
                            "method": "mode",
                            "reason": f"Missing values ({missing_pct:.1f}%)",
                            "priority": "medium",
                        }
                    )

            if pd.api.types.is_numeric_dtype(series):
                clean = series.dropna()
                if len(clean) > 0:
                    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
                    iqr = q3 - q1
                    outliers = clean[(clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)]
                    if len(outliers) / len(clean) > 0.05:
                        recs.append(
                            {
                                "type": "handle_outliers",
                                "column": col,
                                "method": "clip_iqr",
                                "reason": f"Many outliers ({len(outliers)/len(clean)*100:.1f}%)",
                                "priority": "medium",
                            }
                        )

        if self.target_column and self.target_column in self.df.columns:
            target = self.df[self.target_column]
            if not pd.api.types.is_numeric_dtype(target):
                clean = target.dropna()
                if len(clean) > 0:
                    value_counts = clean.value_counts()
                    if value_counts.min() / value_counts.max() < 0.1:
                        recs.append(
                            {
                                "type": "handle_imbalance",
                                "column": self.target_column,
                                "method": "smote_or_class_weight",
                                "reason": "Severe class imbalance detected",
                                "priority": "high",
                            }
                        )

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr()
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.95:
                        recs.append(
                            {
                                "type": "remove_multicollinearity",
                                "columns": [corr_matrix.columns[i], corr_matrix.columns[j]],
                                "reason": f"Very high correlation ({corr_matrix.iloc[i, j]:.3f})",
                                "priority": "medium",
                            }
                        )

        return recs


def profile_dataset(
    data: str | Path | pd.DataFrame,
    target_column: str | None = None,
    sample_size: int | None = None,
) -> dict[str, Any]:
    if isinstance(data, pd.DataFrame):
        df = data
    else:
        path = Path(data)
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, nrows=sample_size)
        elif path.suffix.lower() in [".parquet", ".pq"]:
            df = pd.read_parquet(path)
        elif path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(path, nrows=sample_size)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    profiler = DatasetProfiler(df, target_column)
    return profiler.run()


def save_profile(profile: dict[str, Any], output_path: str | Path) -> None:
    with open(output_path, "w") as f:
        json.dump(profile, f, indent=2, default=str)