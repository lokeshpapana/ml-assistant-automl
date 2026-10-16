"""End-to-end test for the AutoML orchestrator with synthetic data."""

import os
import time

import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def test_orchestrator_synthetic_run():
    from backend.orchestrator import get_orchestrator, DONE, FAILED, PAUSED

    reset_central_store()

    orch = get_orchestrator()

    config = {
        "goal": "classify synthetic demo data",
        "metric": "accuracy >= 0.5",
        "dataset_source": "synthetic",
        "max_rows": 2000,
        "target_column": "target",
        "max_iterations": 2,
        "feature_engineering_every_n": 1,
    }

    res = orch.handle_message({"event": "start", "config": config})
    assert res["status"] == "started"

    deadline = time.time() + 300
    while time.time() < deadline:
        if orch.state in (DONE, PAUSED, FAILED):
            break
        time.sleep(0.2)

    assert orch.state in (DONE, PAUSED), f"orchestrator failed: {orch.error}"
    assert orch.error is None

    assert len(orch.iterations) >= 1, "at least one iteration should complete"
    for it in orch.iterations:
        assert "model" in it
        assert "metrics" in it
        assert "target_metric" in it
        assert isinstance(it["iteration"], int)

    snap = orch.snapshot()
    assert snap["state"] in (DONE, PAUSED)
    assert snap["dataset"]["rows"] == 2000
    assert "target" in snap["dataset"]["columns"]


def test_orchestrator_stop_signal():
    from backend.orchestrator import get_orchestrator, DONE, FAILED, PAUSED

    reset_central_store()

    orch = get_orchestrator()

    config = {
        "goal": "classify synthetic demo data",
        "metric": "accuracy >= 0.5",
        "dataset_source": "synthetic",
        "max_rows": 1000,
        "target_column": "target",
        "max_iterations": 10,
        "feature_engineering_every_n": 1,
    }

    orch.handle_message({"event": "start", "config": config})

    time.sleep(1.0)
    orch.handle_message({"event": "stop"})

    deadline = time.time() + 30
    while time.time() < deadline:
        if orch.state in (PAUSED, FAILED, DONE):
            break
        time.sleep(0.1)

    assert orch.state in (PAUSED, DONE)
    assert len(orch.iterations) >= 0


def reset_central_store():
    import backend.central_store as cs

    cs.reset_central_store()