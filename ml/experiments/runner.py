from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.features.engineering import FeatureEngineer
from ml.models.registry import (
    instantiate_model,
    train_model,
    get_available_models,
    get_model_config,
    cross_validate_model,
    get_feature_importance,
)
from ml.preprocessing.profiler import profile_dataset
from ml.evaluation.ranking import rank_models, generate_model_report

warnings.filterwarnings("ignore")


class ExperimentRunner:
    def __init__(
        self,
        target_column: str,
        task_type: str,
        models_to_run: list[str] | None = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        cv_folds: int = 5,
        max_training_time: int = 300,
    ):
        self.target_column = target_column
        self.task_type = task_type
        self.models_to_run = models_to_run or get_available_models(task_type)
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.max_training_time = max_training_time

        self.feature_engineer: FeatureEngineer | None = None
        self.results: list[dict[str, Any]] = []
        self.best_model: Any = None
        self.X_test = None
        self.y_test = None

    def run(self, df: pd.DataFrame) -> dict[str, Any]:
        start_time = time.time()

        X_train, X_val, X_test, y_train, y_val, y_test = self._prepare_data(df)

        self.X_test, self.y_test = X_test, y_test

        for model_name in self.models_to_run:
            try:
                result = self._run_single_model(model_name, X_train, y_train, X_val, y_val)
                if result:
                    self.results.append(result)
            except Exception as e:
                self.results.append(
                    {
                        "name": model_name,
                        "model_type": model_name,
                        "status": "failed",
                        "error": str(e),
                        "training_time_seconds": 0,
                    }
                )

        ranked = rank_models(
            [r for r in self.results if r.get("status") != "failed"],
            primary_metric="f1" if self.task_type == "classification" else "r2",
        )

        self.best_model = ranked[0] if ranked else None

        report = generate_model_report(ranked)
        report["experiment_time_seconds"] = round(time.time() - start_time, 2)
        report["task_type"] = self.task_type
        report["target_column"] = self.target_column

        return report

    def _prepare_data(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train_df, test_df = train_test_split(
            df, test_size=self.test_size, random_state=self.random_state, stratify=None
        )
        train_df, val_df = train_test_split(
            train_df, test_size=self.val_size / (1 - self.test_size), random_state=self.random_state
        )

        self.feature_engineer = FeatureEngineer(target_column=self.target_column)
        X_train, y_train = self.feature_engineer.fit_transform(train_df)
        X_val, y_val = self.feature_engineer.transform(val_df)
        X_test, y_test = self.feature_engineer.transform(test_df)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _run_single_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, Any] | None:
        config = get_model_config(self.task_type, model_name)
        if not config:
            return None

        model = instantiate_model(self.task_type, model_name)

        trained_model, train_result = train_model(model, X_train, y_train, X_val, y_val)

        feature_names = self.feature_engineer.get_feature_names() if self.feature_engineer else []
        feature_importance = get_feature_importance(trained_model, feature_names)

        cv_result = cross_validate_model(
            instantiate_model(self.task_type, model_name),
            X_train,
            y_train,
            cv=self.cv_folds,
        )

        return {
            "name": model_name,
            "model_type": model_name,
            "hyperparameters": config["default_params"],
            "status": "success",
            "training_time_seconds": train_result["training_time_seconds"],
            "metrics": train_result.get("val_metrics", {}),
            "cv_metrics": cv_result,
            "feature_importance": feature_importance,
        }

    def get_best_model(self) -> dict[str, Any] | None:
        return self.best_model

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.best_model or not self.feature_engineer:
            raise ValueError("Experiment must be run first")
        X, _ = self.feature_engineer.transform(df)
        model = self.best_model.get("model")
        if model is None:
            raise ValueError("Best model not stored")
        return model.predict(X)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray | None:
        if not self.best_model or not self.feature_engineer:
            raise ValueError("Experiment must be run first")
        X, _ = self.feature_engineer.transform(df)
        model = self.best_model.get("model")
        if model is None or not hasattr(model, "predict_proba"):
            return None
        return model.predict_proba(X)


def run_quick_experiment(
    file_path: str | Path,
    target_column: str,
    task_type: str = "classification",
    models: list[str] | None = None,
    sample_size: int | None = None,
) -> dict[str, Any]:
    path = Path(file_path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, nrows=sample_size)
    elif path.suffix.lower() in [".parquet", ".pq"]:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    runner = ExperimentRunner(
        target_column=target_column,
        task_type=task_type,
        models_to_run=models,
    )

    return runner.run(df)


def run_experiment_from_profile(
    df: pd.DataFrame,
    profile: dict[str, Any],
    target_column: str,
    models: list[str] | None = None,
) -> dict[str, Any]:
    task_type = "classification"
    if profile.get("target_analysis"):
        task_type = profile["target_analysis"].get("type", "classification")

    runner = ExperimentRunner(
        target_column=target_column,
        task_type=task_type,
        models_to_run=models,
    )

    return runner.run(df)