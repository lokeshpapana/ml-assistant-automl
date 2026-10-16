from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

warnings.filterwarnings("ignore")


MODEL_REGISTRY = {
    "classification": {
        "logistic_regression": {
            "model": LogisticRegression,
            "default_params": {"max_iter": 1000, "random_state": 42, "n_jobs": -1},
            "param_grid": {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"], "solver": ["lbfgs"]},
        },
        "random_forest": {
            "model": RandomForestClassifier,
            "default_params": {"n_estimators": 200, "random_state": 42, "n_jobs": -1},
            "param_grid": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
            },
        },
        "xgboost": {
            "model": XGBClassifier,
            "default_params": {
                "n_estimators": 200,
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": 0,
                "eval_metric": "logloss",
            },
            "param_grid": {
                "n_estimators": [100, 200],
                "max_depth": [3, 6, 10],
                "learning_rate": [0.01, 0.1, 0.3],
            },
        },
        "lightgbm": {
            "model": LGBMClassifier,
            "default_params": {
                "n_estimators": 200,
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": -1,
            },
            "param_grid": {
                "n_estimators": [100, 200],
                "max_depth": [3, 6, 10],
                "learning_rate": [0.01, 0.1, 0.3],
            },
        },
        "gradient_boosting": {
            "model": GradientBoostingClassifier,
            "default_params": {"n_estimators": 200, "random_state": 42},
            "param_grid": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.01, 0.1],
            },
        },
        "svm": {
            "model": SVC,
            "default_params": {"random_state": 42, "probability": True},
            "param_grid": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
        },
        "neural_network": {
            "model": MLPClassifier,
            "default_params": {"random_state": 42, "max_iter": 500, "early_stopping": True},
            "param_grid": {
                "hidden_layer_sizes": [(64,), (128,), (64, 32)],
                "alpha": [0.0001, 0.001],
            },
        },
    },
    "regression": {
        "linear_regression": {
            "model": LinearRegression,
            "default_params": {"n_jobs": -1},
            "param_grid": {},
        },
        "ridge": {
            "model": Ridge,
            "default_params": {"random_state": 42},
            "param_grid": {"alpha": [0.01, 0.1, 1, 10, 100]},
        },
        "random_forest": {
            "model": RandomForestRegressor,
            "default_params": {"n_estimators": 200, "random_state": 42, "n_jobs": -1},
            "param_grid": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
            },
        },
        "xgboost": {
            "model": XGBRegressor,
            "default_params": {
                "n_estimators": 200,
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": 0,
            },
            "param_grid": {
                "n_estimators": [100, 200],
                "max_depth": [3, 6, 10],
                "learning_rate": [0.01, 0.1, 0.3],
            },
        },
        "lightgbm": {
            "model": LGBMRegressor,
            "default_params": {
                "n_estimators": 200,
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": -1,
            },
            "param_grid": {
                "n_estimators": [100, 200],
                "max_depth": [3, 6, 10],
                "learning_rate": [0.01, 0.1, 0.3],
            },
        },
        "gradient_boosting": {
            "model": GradientBoostingRegressor,
            "default_params": {"n_estimators": 200, "random_state": 42},
            "param_grid": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.01, 0.1],
            },
        },
        "svm": {
            "model": SVR,
            "default_params": {},
            "param_grid": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
        },
        "neural_network": {
            "model": MLPRegressor,
            "default_params": {"random_state": 42, "max_iter": 500, "early_stopping": True},
            "param_grid": {
                "hidden_layer_sizes": [(64,), (128,), (64, 32)],
                "alpha": [0.0001, 0.001],
            },
        },
    },
}


def get_available_models(task_type: str) -> list[str]:
    return list(MODEL_REGISTRY.get(task_type, {}).keys())


def get_model_config(task_type: str, model_name: str) -> dict[str, Any] | None:
    return MODEL_REGISTRY.get(task_type, {}).get(model_name)


def instantiate_model(task_type: str, model_name: str, params: dict[str, Any] | None = None) -> BaseEstimator:
    config = get_model_config(task_type, model_name)
    if not config:
        raise ValueError(f"Unknown model: {model_name} for task type: {task_type}")

    model_params = config["default_params"].copy()
    if params:
        model_params.update(params)

    return config["model"](**model_params)


METRICS = {
    "classification": {
        "accuracy": accuracy_score,
        "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted"),
        "f1_macro": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"),
        "roc_auc": lambda y_true, y_proba: (
            roc_auc_score(y_true, y_proba[:, 1])  # binary: positive-class column
            if y_proba is not None and y_proba.ndim > 1 and y_proba.shape[1] == 2
            else (
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
                if y_proba is not None and y_proba.ndim > 1
                else roc_auc_score(y_true, y_proba)  # 1d binary scores
            )
        ),
    },
    "regression": {
        "mae": mean_absolute_error,
        "mse": mean_squared_error,
        "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score,
    },
}


def compute_metrics(
    task_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    metrics = {}
    for name, func in METRICS.get(task_type, {}).items():
        try:
            if name == "roc_auc" and y_proba is not None:
                metrics[name] = float(func(y_true, y_proba))
            else:
                metrics[name] = float(func(y_true, y_pred))
        except Exception:
            metrics[name] = 0.0
    return metrics


def train_model(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> tuple[BaseEstimator, dict[str, Any]]:
    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    result = {
        "model": model,
        "training_time_seconds": round(training_time, 3),
    }

    if X_val is not None and y_val is not None:
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
        result["val_metrics"] = compute_metrics(
            "classification" if hasattr(model, "predict_proba") else "regression",
            y_val,
            y_pred,
            y_proba,
        )

    return model, result


def cross_validate_model(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str | None = None,
) -> dict[str, float]:
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        "cv_mean": float(np.mean(scores)),
        "cv_std": float(np.std(scores)),
        "cv_scores": scores.tolist(),
    }


def get_feature_importance(model: BaseEstimator, feature_names: list[str]) -> dict[str, float] | None:
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        return dict(sorted(zip(feature_names, importances, strict=False), key=lambda x: x[1], reverse=True))
    elif hasattr(model, "coef_"):
        coef = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
        return dict(sorted(zip(feature_names, coef, strict=False), key=lambda x: x[1], reverse=True))
    return None