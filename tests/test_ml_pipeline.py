import tempfile
from pathlib import Path
from uuid import UUID

import pandas as pd
import pytest

from ml.preprocessing.profiler import profile_dataset
from ml.features.engineering import FeatureEngineer
from ml.experiments.runner import ExperimentRunner
from ml.evaluation.ranking import rank_models, generate_model_report


@pytest.fixture
def sample_classification_data():
    data = {
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 10,
        "feature2": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1] * 10,
        "feature3": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"] * 10,
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 10,
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_regression_data():
    data = {
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 10,
        "feature2": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1] * 10,
        "feature3": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"] * 10,
        "target": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 11.0] * 10,
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_csv_file(sample_classification_data):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        sample_classification_data.to_csv(f.name, index=False)
        yield f.name
    Path(f.name).unlink()


class TestDatasetProfiler:
    def test_profile_classification_dataset(self, sample_classification_data):
        profile = profile_dataset(sample_classification_data, target_column="target")

        assert "overview" in profile
        assert profile["overview"]["n_rows"] == 100
        assert profile["overview"]["n_columns"] == 4
        assert "columns" in profile
        assert "target" in profile["columns"]
        assert profile["target_analysis"]["type"] == "classification"
        assert profile["target_analysis"]["n_classes"] == 2

    def test_profile_regression_dataset(self, sample_regression_data):
        profile = profile_dataset(sample_regression_data, target_column="target")

        assert profile["target_analysis"]["type"] == "regression"

    def test_profile_from_csv(self, temp_csv_file):
        profile = profile_dataset(temp_csv_file, target_column="target")

        assert profile["overview"]["n_rows"] == 100
        assert profile["overview"]["n_columns"] == 4


class TestFeatureEngineer:
    def test_fit_transform_classification(self, sample_classification_data):
        engineer = FeatureEngineer(target_column="target")
        X, y = engineer.fit_transform(sample_classification_data)

        assert X.shape[0] == 100
        assert X.shape[1] > 0
        assert y is not None
        assert len(y) == 100

    def test_fit_transform_regression(self, sample_regression_data):
        engineer = FeatureEngineer(target_column="target")
        X, y = engineer.fit_transform(sample_regression_data)

        assert X.shape[0] == 100
        assert X.shape[1] > 0
        assert y is not None
        assert len(y) == 100

    def test_get_feature_names(self, sample_classification_data):
        engineer = FeatureEngineer(target_column="target")
        engineer.fit_transform(sample_classification_data)

        names = engineer.get_feature_names()
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)


class TestExperimentRunner:
    def test_run_classification_experiment(self, sample_classification_data):
        runner = ExperimentRunner(
            target_column="target",
            task_type="classification",
            models_to_run=["logistic_regression", "random_forest"],
            cv_folds=3,
        )

        report = runner.run(sample_classification_data)

        assert "summary" in report
        assert report["summary"]["total_models"] == 2
        assert "ranking" in report
        assert len(report["ranking"]) == 2
        assert all("rank" in r for r in report["ranking"])

    def test_run_regression_experiment(self, sample_regression_data):
        runner = ExperimentRunner(
            target_column="target",
            task_type="regression",
            models_to_run=["linear_regression", "random_forest"],
            cv_folds=3,
        )

        report = runner.run(sample_regression_data)

        assert report["summary"]["total_models"] == 2
        assert len(report["ranking"]) == 2


class TestModelRanking:
    def test_rank_models(self):
        results = [
            {"name": "model_a", "model_type": "logistic_regression", "metrics": {"f1": 0.9, "accuracy": 0.95}, "training_time_seconds": 1.0},
            {"name": "model_b", "model_type": "random_forest", "metrics": {"f1": 0.85, "accuracy": 0.92}, "training_time_seconds": 2.0},
            {"name": "model_c", "model_type": "xgboost", "metrics": {"f1": 0.92, "accuracy": 0.96}, "training_time_seconds": 3.0},
        ]

        ranked = rank_models(results, primary_metric="f1")

        assert ranked[0]["name"] == "model_c"
        assert ranked[0]["rank"] == 1
        assert ranked[1]["name"] == "model_a"
        assert ranked[1]["rank"] == 2
        assert ranked[2]["name"] == "model_b"
        assert ranked[2]["rank"] == 3

    def test_generate_model_report(self):
        results = [
            {"name": "model_a", "model_type": "logistic_regression", "metrics": {"f1": 0.9}, "training_time_seconds": 1.0, "rank": 1},
            {"name": "model_b", "model_type": "random_forest", "metrics": {"f1": 0.85}, "training_time_seconds": 2.0, "rank": 2},
        ]

        report = generate_model_report(results)

        assert report["summary"]["total_models"] == 2
        assert report["summary"]["best_model"] == "model_a"
        assert "ranking" in report
        assert "comparison_table" in report
        assert "pareto_frontier" in report


class TestComputeMetrics:
    """Trainer regression: binary ROC-AUC must be real, precision/recall computed."""

    def test_binary_roc_auc_is_not_stored_as_zero(self, sample_classification_data):
        import numpy as np
        from backend.agents.trainer import instantiate_model
        from ml.features.engineering import FeatureEngineer
        from ml.models.registry import compute_metrics
        from sklearn.model_selection import train_test_split

        df = sample_classification_data
        X, y = df.drop(columns=["target"]), df["target"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        engineer = FeatureEngineer(target_column="target")
        Xtr, ytr = engineer.fit_transform(pd.concat([X_train, y_train], axis=1))
        Xte, yte = engineer.transform(pd.concat([X_test, y_test], axis=1))
        ytr = ytr if ytr is not None else y_train
        yte = yte if yte is not None else y_test

        model = instantiate_model("classification", "logistic_regression")
        model.fit(Xtr, ytr)
        y_proba = model.predict_proba(Xte)
        metrics = compute_metrics("classification", yte, model.predict(Xte), y_proba)

        assert metrics["roc_auc"] > 0.5, "binary ROC-AUC must not be 0.0"
        assert metrics["precision"] > 0.5
        assert metrics["recall"] > 0.5

    def test_multiclass_rock_auc(self):
        import numpy as np
        import pandas as pd
        from ml.models.registry import compute_metrics
        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        X, y = make_classification(
            n_samples=1200, n_features=10, n_informative=6, n_classes=3, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)
        y_df = pd.Series(y_test)
        metrics = compute_metrics("classification", y_df, model.predict(X_test), y_proba)

        assert metrics["roc_auc"] > 0.5


class TestIntegrationFlow:
    def test_full_classification_flow(self, sample_classification_data):
        profile = profile_dataset(sample_classification_data, target_column="target")
        assert profile["target_analysis"]["type"] == "classification"

        runner = ExperimentRunner(
            target_column="target",
            task_type="classification",
            models_to_run=["logistic_regression", "random_forest"],
            cv_folds=3,
        )

        report = runner.run(sample_classification_data)
        assert report["summary"]["total_models"] == 2

        best = report["ranking"][0]
        assert best["metrics"]["f1"] >= 0.5

    def test_full_regression_flow(self, sample_regression_data):
        profile = profile_dataset(sample_regression_data, target_column="target")
        assert profile["target_analysis"]["type"] == "regression"

        runner = ExperimentRunner(
            target_column="target",
            task_type="regression",
            models_to_run=["linear_regression", "random_forest"],
            cv_folds=3,
        )

        report = runner.run(sample_regression_data)
        assert report["summary"]["total_models"] == 2

        best = report["ranking"][0]
        assert best["metrics"]["r2"] >= 0.5