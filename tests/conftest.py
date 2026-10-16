"""Test isolation.

The dev server, the DuckDB central store and ``./mlruns`` are all shared, cwd
relative artifacts. Tests therefore run against throwaway locations so they
never touch developer data and never contend for locks with a running server.
Environment is set BEFORE any backend module is imported, because settings are
read at import time.
"""

import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="mlassist-tests-"))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_TMP / 'test.db'}"
os.environ["AUTOML_OUTPUT_DIR"] = str(_TMP / "automl_outputs")
os.environ["CENTRAL_STORE_PATH"] = str(_TMP / "central_store.duckdb")
os.environ["CENTRAL_STORE_IN_MEMORY"] = "true"
os.environ["MLFLOW_TRACKING_URI"] = str(_TMP / "mlruns")
os.environ["UPLOAD_DIR"] = str(_TMP / "uploads")
os.environ["LLM_PROVIDER"] = "local"
os.environ["ENABLE_OFFLINE_PAPER_BANK"] = "true"

Path(os.environ["UPLOAD_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["AUTOML_OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)

# Create the schema for the throwaway DB up front. ASGITransport does not run
# the app lifespan, so `init_db()` would otherwise only happen when a test
# happens to exercise it (and any single-test run would fail on missing tables).
# Importing the app first is required: Base.metadata is only populated once the
# model modules are imported.
import asyncio  # noqa: E402

import backend.main  # noqa: E402,F401  (registers every model on Base.metadata)
from backend.database import close_db, init_db  # noqa: E402


async def _create_schema() -> None:
    await init_db()
    await close_db()


asyncio.run(_create_schema())


import time  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _quiesce_orchestrator():
    """Leave the singleton orchestrator idle between tests.

    The orchestrator is process-wide with one worker thread, so a test that
    leaves a run in flight makes the next test observe a ``busy`` engine and a
    stale interview (the classic source of order-dependent test failures).
    """
    from backend.orchestrator import get_orchestrator

    orch = get_orchestrator()
    if orch.state in ("TRAINING", "RESEARCH", "DATA_FETCH", "STARTING"):
        orch.request_stop("test-teardown")
        deadline = time.time() + 120
        while time.time() < deadline:
            if orch.state not in ("TRAINING", "RESEARCH", "DATA_FETCH", "STARTING"):
                break
            time.sleep(0.2)
    orch.interviewer.reset()
    orch._interview_project_id = None
    yield
