"""Agent 4: AutoML & Model Trainer Agent.

Runs iteratively over a candidate ladder of model/hyperparameter configurations,
evaluates on a held-out test set, logs parameters / metrics / code / artifacts /
citation tags to MLflow, and keeps looping until the user's target metric is hit or
the interrupt event is set.
"""

from __future__ import annotations

import pickle
import time
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from backend.agents.common import TargetSpec, json_safe, met_target, safe_float
from backend.central_store import CentralStore
from backend.mlflow_tracker import get_tracker
from backend.config import settings
from ml.features.engineering import FeatureEngineer
from ml.models.registry import (
    compute_metrics,
    get_feature_importance,
    get_model_config,
    instantiate_model,
)

warnings.filterwarnings("ignore")


def build_candidate_ladder(task_type: str) -> list[dict[str, Any]]:
    """Deterministic list of (model_name, params) rungs run one per iteration."""
    defaults = [
        "logistic_regression" if task_type == "classification" else "linear_regression",
        "random_forest",
        "lightgbm",
        "xgboost",
        "gradient_boosting",
        "svm",
        "neural_network",
    ]
    tuned: list[dict[str, Any]] = [
        {"model": "random_forest", "params": {"n_estimators": 300, "max_depth": None, "min_samples_split": 2}},
        {"model": "lightgbm", "params": {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 127}},
        {"model": "xgboost", "params": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 5, "subsample": 0.8}},
    ]
    ladder: list[dict[str, Any]] = [{"model": m, "params": {}} for m in defaults]
    ladder.extend(tuned)
    # Hunt phase: keep alternating tuned gbdt configs so long campaigns keep
    # improving. Parameters MUST match the estimator they are passed to --
    # e.g. RandomForest has no `learning_rate`; mixing them raised TypeError and
    # killed any run that reached these rungs.
    hunt = [
        {"model": "lightgbm", "params": {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 63}},
        {"model": "xgboost", "params": {"n_estimators": 400, "learning_rate": 0.04, "max_depth": 6, "subsample": 0.8}},
        {"model": "random_forest", "params": {"n_estimators": 400, "max_depth": None, "min_samples_split": 2}},
    ]
    for i in range(3):
        base = dict(hunt[i % 3])
        if i >= 3:  # second sweep: tighten the learning rate
            if "learning_rate" in base["params"]:
                base["params"]["learning_rate"] = round(base["params"]["learning_rate"] * 0.8, 3)
            base["params"]["n_estimators"] = 600
        ladder.append(base)
    return ladder


class TrainerAgent:
    def __init__(self, task_type: str = "classification", test_size: float = 0.2, seed: int = 42, target_column: str | None = None, model_hint: str | None = None, start_index: int = 0):
        self.task_type = task_type
        self.test_size = test_size
        self.seed = seed
        self.target_column = target_column
        # start_index > 0 means "resume": continue the ladder where the previous
        # run stopped instead of retraining models it already tried.
        self.start_index = max(0, int(start_index or 0))
        self.ladder = build_candidate_ladder(task_type)
        if model_hint:
            # Chat-directed starting point ("use xgboost"): rotate the ladder so
            # the requested model trains first; everything else keeps its order.
            names = [c["model"] for c in self.ladder]
            if model_hint in names:
                i = names.index(model_hint)
                self.ladder = self.ladder[i:] + self.ladder[:i]
        self.tracker = get_tracker()
        self.best: dict[str, Any] | None = None
        self.best_cache: dict[str, Any] = {}
        self.model_dir = Path(settings.automl_output_dir) / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def model_for(self, iteration: int) -> dict[str, Any]:
        """Rung for a given iteration, offset by ``start_index`` when resuming."""
        return self.ladder[(self.start_index + iteration) % len(self.ladder)]

    def train_iteration(
        self,
        store: CentralStore,
        target_spec: TargetSpec,
        iteration: int,
        emit: Callable[[dict[str, Any]], None],
        time_budget: int | None = None,
        excluded_features: list[str] | None = None,
        feature_provenance: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        candidate = self.model_for(iteration)
        model_name = candidate["model"]
        params = candidate["params"]
        time_budget = time_budget or settings.automl_model_time_budget

        emit(
            {
                "type": "agent_message",
                "agent": "trainer",
                "text": f"Iteration {iteration + 1}: training '{model_name}' with params {params or 'default'}…",
            }
        )

        df = store.to_dataframe()
        target = self.target_column if self.target_column and self.target_column in df.columns else "target"
        if target not in df.columns:
            target = store.columns()[-1]
        df = df.dropna(subset=[target]).copy()
        if len(df) < 50:
            raise ValueError("Dataset too small after dropping missing targets")

        X = df.drop(columns=[target])
        y = df[target]

        # Soft-disabled features (Features page / chat "remove"): drop the base
        # columns plus any materialized derived columns whose provenance (a/b)
        # depends on them. The store table itself is untouched (reversible).
        excluded = [c for c in (excluded_features or []) if c != target]
        if excluded:
            derived = {
                str(p.get("name"))
                for p in (feature_provenance or [])
                if p.get("a") in excluded or p.get("b") in excluded
            }
            drop_cols = [c for c in list(dict.fromkeys(excluded)) + sorted(derived) if c in X.columns]
            if drop_cols:
                X = X.drop(columns=drop_cols)
                emit(
                    {
                        "type": "agent_message",
                        "agent": "trainer",
                        "text": f"Excluded {len(drop_cols)} disabled feature(s): {', '.join(drop_cols)}.",
                    }
                )
        else:
            drop_cols = []

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.seed, stratify=None
        )

        engineer = FeatureEngineer(target_column=target)
        Xtr, ytr = engineer.fit_transform(pd.concat([X_train, y_train], axis=1))
        Xte, yte = engineer.transform(pd.concat([X_test, y_test], axis=1))
        ytr = ytr if ytr is not None else y_train
        yte = yte if yte is not None else y_test

        model = instantiate_model(self.task_type, model_name, params)
        start = time.time()
        model.fit(Xtr, ytr)
        training_time = time.time() - start

        y_pred = model.predict(Xte)
        y_proba = model.predict_proba(Xte) if hasattr(model, "predict_proba") else None
        metrics = compute_metrics(self.task_type, yte, y_pred, y_proba)

        feature_names = engineer.get_feature_names()
        importance = get_feature_importance(model, feature_names)

        # Persist best estimator + engineer for export/download.
        with self.tracker.run(run_name=f"iter_{iteration + 1}_{model_name}") as run:
            run_id = run["run_id"]
            self.tracker.log_params(
                {
                    "model": model_name,
                    "hyperparameters": params,
                    "iteration": iteration + 1,
                    "task_type": self.task_type,
                    "n_rows_train": len(Xtr),
                    "n_features": len(feature_names),
                    "test_size": self.test_size,
                }
            )
            self.tracker.log_metrics(metrics)
            self.tracker.log_metrics({"target_metric": metrics.get(target_spec.metric, 0.0), "training_time": training_time})
            self.tracker.set_tags(
                {
                    "model": model_name,
                    "iteration": iteration + 1,
                    "papers": "|".join(self._recent_papers(store)),
                    "status": "completed",
                }
            )
            model_path = self.model_dir / f"iter_{iteration + 1}_{model_name}.pkl"
            with open(model_path, "wb") as fh:
                pickle.dump({"model": model, "engineer": engineer, "feature_names": feature_names, "metrics": metrics}, fh)
            self.tracker.log_artifact(model_path)
            dataset_snapshot = store.export_csv(self.model_dir / f"iter_{iteration + 1}_dataset_snapshot.csv")
            self.tracker.log_artifact(dataset_snapshot)

        record = {
            "iteration": iteration + 1,
            "model": model_name,
            "hyperparameters": json_safe(params),
            "metrics": json_safe(metrics),
            "target_metric": safe_float(metrics.get(target_spec.metric)),
            "training_time_seconds": round(training_time, 3),
            "n_features": len(feature_names),
            "feature_importance": json_safe(importance),
            "run_id": run_id,
            "model_path": str(model_path),
            "dataset_snapshot": str(dataset_snapshot),
            "target_met": met_target(metrics, target_spec),
        }

        # Keep best model artifact.
        if self.best is None or self._better(record, target_spec):
            self.best = record
            self.best_cache = {
                "model": model,
                "engineer": engineer,
                "feature_names": feature_names,
                "record": record,
            }

        emit(
            {
                "type": "iteration",
                "iteration": record["iteration"],
                "model": model_name,
                "params": params,
                "metrics": metrics,
                "target_metric": record["target_metric"],
                "target_met": record["target_met"],
                "training_time_seconds": record["training_time_seconds"],
                "mlflow_run_id": record["run_id"],
            }
        )
        emit(
            {
                "type": "leaderboard",
                "runs": self.tracker.list_runs(),
                "best": self.best,
            }
        )
        if importance:
            emit(
                {
                    "type": "feature_importance",
                    "feature_importance": importance,
                    "iteration": record["iteration"],
                }
            )
        return record

    def _better(self, record: dict[str, Any], spec: TargetSpec) -> bool:
        current = safe_float(self.best.get("target_metric") if self.best else -float("inf"))
        candidate = safe_float(record.get("target_metric"))
        if spec.direction == "minimize":
            return candidate < current
        return candidate > current

    @staticmethod
    def _recent_papers(store: CentralStore) -> list[str]:
        return [f.get("paper_id", "") for f in store.feature_log[-5:]]

    def export_best_model(self) -> str | None:
        """Write the best cached estimator to a .pkl file; returns its path."""
        if not self.best_cache.get("model") or not self.best:
            return None
        out = Path(settings.automl_output_dir) / "best_model.pkl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as fh:
            pickle.dump(self.best_cache, fh)
        return str(out)