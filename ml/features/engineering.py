from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

warnings.filterwarnings("ignore")


class FeatureEngineer:
    def __init__(
        self,
        target_column: str | None = None,
        max_cardinality: int = 50,
        handle_outliers: bool = True,
        outlier_method: str = "clip_iqr",
        scaling: str = "standard",
    ):
        self.target_column = target_column
        self.max_cardinality = max_cardinality
        self.handle_outliers = handle_outliers
        self.outlier_method = outlier_method
        self.scaling = scaling

        self.numeric_features: list[str] = []
        self.categorical_features: list[str] = []
        self.datetime_features: list[str] = []
        self.high_cardinality_features: list[str] = []
        self.constant_features: list[str] = []
        self.id_like_features: list[str] = []

        self.preprocessor: ColumnTransformer | None = None
        self.label_encoder: LabelEncoder | None = None
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> FeatureEngineer:
        if self.target_column and self.target_column in df.columns:
            self.df = df.drop(columns=[self.target_column])
        else:
            self.df = df.copy()

        if y is not None and self.target_column is not None:
            if not pd.api.types.is_numeric_dtype(y):
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(y)

        self._identify_feature_types()
        self._build_preprocessor()
        self.preprocessor.fit(self.df)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, pd.Series | None]:
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")

        if self.target_column and self.target_column in df.columns:
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]
            if self.label_encoder is not None:
                y = pd.Series(self.label_encoder.transform(y), index=y.index)
        else:
            X = df.copy()
            y = None

        X_transformed = self.preprocessor.transform(X)
        return X_transformed, y

    def fit_transform(self, df: pd.DataFrame, y: pd.Series | None = None) -> tuple[np.ndarray, pd.Series | None]:
        return self.fit(df, y).transform(df)

    def _identify_feature_types(self) -> None:
        for col in self.df.columns:
            series = self.df[col]
            nunique = series.nunique()

            if nunique <= 1:
                self.constant_features.append(col)
            elif pd.api.types.is_numeric_dtype(series):
                self.numeric_features.append(col)
            elif pd.api.types.is_datetime64_any_dtype(series):
                self.datetime_features.append(col)
            elif nunique / len(series) > 0.9:
                self.id_like_features.append(col)
            elif nunique > self.max_cardinality:
                self.high_cardinality_features.append(col)
            else:
                self.categorical_features.append(col)

    def _build_preprocessor(self) -> None:
        transformers = []

        if self.numeric_features:
            numeric_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", self._get_scaler()),
                ]
            )
            if self.handle_outliers:
                numeric_pipeline.steps.insert(1, ("outlier_handler", OutlierHandler(method=self.outlier_method)))
            transformers.append(("numeric", numeric_pipeline, self.numeric_features))

        if self.categorical_features:
            categorical_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            )
            transformers.append(("categorical", categorical_pipeline, self.categorical_features))

        if self.high_cardinality_features:
            high_card_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ]
            )
            transformers.append(("high_cardinality", high_card_pipeline, self.high_cardinality_features))

        if self.datetime_features:
            datetime_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="constant", fill_value=pd.Timestamp("1970-01-01"))),
                    ("extractor", DateTimeFeatureExtractor()),
                ]
            )
            transformers.append(("datetime", datetime_pipeline, self.datetime_features))

        drop_features = self.constant_features + self.id_like_features
        if drop_features:
            transformers.append(("drop", "drop", drop_features))

        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            sparse_threshold=0,
        )

    def _get_scaler(self):
        scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
        }
        return scalers.get(self.scaling, StandardScaler())

    def get_feature_names(self) -> list[str]:
        if not self.is_fitted or self.preprocessor is None:
            return []
        return self.preprocessor.get_feature_names_out().tolist()

    def get_feature_importance_info(self) -> dict[str, Any]:
        return {
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "datetime_features": self.datetime_features,
            "high_cardinality_features": self.high_cardinality_features,
            "dropped_features": self.constant_features + self.id_like_features,
        }


class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, method: str = "clip_iqr", factor: float = 1.5):
        self.method = method
        self.factor = factor
        self.bounds_: dict[int, tuple[float, float]] = {}

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> OutlierHandler:
        for i in range(X.shape[1]):
            col = X[:, i]
            if self.method == "clip_iqr":
                q1, q3 = np.nanpercentile(col, [25, 75])
                iqr = q3 - q1
                self.bounds_[i] = (q1 - self.factor * iqr, q3 + self.factor * iqr)
            elif self.method == "clip_zscore":
                mean, std = np.nanmean(col), np.nanstd(col)
                self.bounds_[i] = (mean - self.factor * std, mean + self.factor * std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        for i, (lower, upper) in self.bounds_.items():
            X[:, i] = np.clip(X[:, i], lower, upper)
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features


class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> DateTimeFeatureExtractor:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        features = []
        for i in range(X.shape[1]):
            col = pd.Series(X[:, i])
            dt_col = pd.to_datetime(col, errors="coerce")
            extracted = np.column_stack(
                [
                    dt_col.dt.year.fillna(0).values,
                    dt_col.dt.month.fillna(0).values,
                    dt_col.dt.day.fillna(0).values,
                    dt_col.dt.dayofweek.fillna(0).values,
                    dt_col.dt.hour.fillna(0).values,
                    dt_col.dt.quarter.fillna(0).values,
                    dt_col.dt.is_weekend.astype(int).fillna(0).values,
                ]
            )
            features.append(extracted)
        return np.hstack(features) if features else np.empty((X.shape[0], 0))


def create_preprocessing_pipeline(
    df: pd.DataFrame,
    target_column: str | None = None,
    recommendations: list[dict[str, Any]] | None = None,
) -> FeatureEngineer:
    engineer = FeatureEngineer(target_column=target_column)

    if recommendations:
        for rec in recommendations:
            if rec["type"] == "drop_column" and rec["column"] in df.columns:
                df = df.drop(columns=[rec["column"]])
            elif rec["type"] == "impute_numeric" and rec["column"] in df.columns:
                pass
            elif rec["type"] == "impute_categorical" and rec["column"] in df.columns:
                pass

    return engineer