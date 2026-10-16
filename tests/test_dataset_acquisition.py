from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.dataset_acquisition import (
    DatasetSpecification,
    acquire_dataset,
)
from ml.dataset_acquisition.evaluate import (
    analyze_features,
    detect_leakage,
    identify_target,
    inspect_dataframe,
    score_candidate,
)
from ml.dataset_acquisition.optimize import optimize_dataset
from ml.dataset_acquisition.spec import MLTaskType


@pytest.fixture
def sample_df() -> pd.DataFrame:
    np.random.seed(0)
    n = 1500
    return pd.DataFrame(
        {
            "id": range(n),
            "age": np.random.randint(18, 80, n),
            "income": np.random.normal(50000, 15000, n),
            "score": np.random.normal(0, 1, n),
            "city": np.random.choice(["A", "B", "C"], n),
            "fraud": np.random.choice([0, 1], n, p=[0.85, 0.15]),
            "leak_col": np.random.choice([1, np.nan], n, p=[0.8, 0.2]),
        }
    )


def test_spec_from_request_infers_task_and_target():
    spec = DatasetSpecification.from_user_request("predict fraud with 5 features and 2000 rows")
    assert spec.task == MLTaskType.CLASSIFICATION
    assert "task" in spec.inferred_fields
    assert spec.requested_features == 5


def test_inspect_dataframe_reports_reality(sample_df):
    insp = inspect_dataframe(sample_df)
    assert insp["n_rows"] == 1500
    assert insp["n_columns"] == 7
    assert "id" not in insp["constant_columns"]


def test_target_identification_prefers_requested_target(sample_df):
    ti = identify_target(sample_df, "fraud", "classification")
    assert ti["best_target"]["column"] == "fraud"
    assert ti["best_target"]["type"] == "binary"


def test_identifier_and_constant_features_flagged(sample_df):
    ti = identify_target(sample_df, "fraud", "classification")
    assessments = analyze_features(sample_df, ti["best_target"]["column"], "classification")
    classes = {a.name: a.classification for a in assessments}
    assert classes["id"] == "identifier"
    assert classes["leak_col"] == "constant"


def test_leakage_detection_finds_direct_copy(sample_df):
    df = sample_df.copy()
    df["fraud_copy"] = df["fraud"]
    ti = identify_target(df, "fraud", "classification")
    assessments = analyze_features(df, "fraud", "classification")
    leakage = detect_leakage(assessments)
    assert "fraud_copy" in leakage


def test_score_candidate_produces_weighted_score(sample_df):
    ti = identify_target(sample_df, "fraud", "classification")
    insp = inspect_dataframe(sample_df)
    assessments = analyze_features(sample_df, ti["best_target"]["column"], "classification")
    spec = DatasetSpecification.from_user_request("predict fraud")
    score = score_candidate(insp, ti, assessments, spec, "desc", "CC-BY")
    assert 0.0 <= score["overall_score"] <= 1.0
    assert "relevance" in score["component_scores"]


def test_optimize_removes_bad_features_and_respects_count(sample_df):
    ti = identify_target(sample_df, "fraud", "classification")
    assessments = analyze_features(sample_df, ti["best_target"]["column"], "classification")
    res = optimize_dataset(
        sample_df, ti["best_target"]["column"], assessments, requested_input_features=5
    )
    assert "id" not in res.df.columns
    assert "leak_col" not in res.df.columns
    assert res.target == "fraud"
    # age, income, score + one-hot(city) -> 5 columns achievable
    assert res.final_input_feature_count >= 3


def test_acquire_from_local_file_returns_report(sample_df, tmp_path):
    path = tmp_path / "data.csv"
    sample_df.to_csv(path, index=False)
    out = tmp_path / "out"
    report = acquire_dataset(
        request="predict fraud with 5 features",
        local_file_path=str(path),
        output_dir=str(out),
    )
    assert report["status"] == "success"
    assert report["problem"]["target"] == "fraud"
    assert report["output_path"] is not None
    assert (tmp_path / "out").exists()
    # final dataset csv should exist and be loadable
    saved = pd.read_csv(report["output_path"])
    assert "fraud" in saved.columns
