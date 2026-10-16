"""Phase B tests: project_id propagation, entry validation, run isolation.

Covers: workspace shell route, /ui regression, context extraction shapes,
invalid-project rejection (no orphaned runs), session/project mismatch,
event tagging, busy-guard isolation between two projects, WS helper path.
"""

import asyncio
import uuid

import pandas as pd
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from backend.main import app
from backend.orchestrator import get_orchestrator
from backend.project_context import extract_run_context
from backend.ws_bus import get_event_bus


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture(autouse=True)
def _store_has_table():
    """snapshot() needs the central table to exist; ensure it cheaply."""
    from backend.central_store import get_central_store

    store = get_central_store()
    try:
        store.shape()
    except Exception:
        store.load_dataframe(pd.DataFrame({"a": [1, 2], "target": [0, 1]}))


@pytest.mark.asyncio
async def test_workspace_route_and_ui_regression(client):
    r = await client.get("/workspace")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "AutoML Workspace" in r.text
    # Phase B shipped honest placeholders; Phase E replaced them with real pages.
    # Either way the shell must not claim persistence it does not have.
    assert "resumable" not in r.text.lower()  # never claim "resumable" (not a durability claim we make)
    # Resume is now implemented from persisted checkpoints, so the shell exposes
    # the action and never implies the app itself was restarted into a run.
    assert "Resume" in r.text
    assert "your application restarted" not in r.text.lower()
    # every project page route is present as a real view.
    # The UI redesign consolidated the shell into these canonical views; each
    # legacy Phase-E view maps to one of them without losing its capability:
    #   overview -> home, analysis/evaluation -> report (+ compare),
    #   research/features -> model-lab pipeline & research drawers,
    #   chat, data, experiments, report -> unchanged.
    for page in ("home", "chat", "data", "lab", "experiments",
                 "monitor", "reports", "report", "compare", "code",
                 "settings", "projects", "dataset", "run"):
        assert f'id="v-{page}"' in r.text, f"missing view for {page}"

    r = await client.get("/ui")  # classic UI untouched
    assert r.status_code == 200
    assert "Interactive AutoML" in r.text


def test_extract_run_context_shapes():
    assert extract_run_context(None) == {"project_id": None, "session_id": None, "config": {}}
    flat = extract_run_context({"goal": "g", "project_id": "p"})
    assert flat["project_id"] == "p" and flat["config"] == {"goal": "g"}
    nested = extract_run_context({"project_id": "top", "config": {"goal": "g", "project_id": "n"}})
    assert nested["project_id"] == "top"  # top-level wins


def test_emit_tags_run_context():
    orch = get_orchestrator()
    orch.project_id, orch.session_id, orch.run_id = "pa", "sa", "ra"
    try:
        orch._emit({"type": "probe-phase-b"})
        ev = [e for e in get_event_bus().replay("t", all_projects=True) if e.get("type") == "probe-phase-b"][-1]
        assert (ev["project_id"], ev["session_id"], ev["run_id"]) == ("pa", "sa", "ra")
    finally:
        orch.project_id, orch.session_id, orch.run_id = None, None, None


@pytest.mark.asyncio
async def test_start_rejects_unknown_project(client):
    before = get_orchestrator().snapshot()["run_context"]
    r = await client.post("/automl/start", json={"project_id": str(uuid.uuid4()), "goal": "x"})
    assert r.status_code == 404
    after = get_orchestrator().snapshot()["run_context"]
    assert after == before  # no orphaned run started


@pytest.mark.asyncio
async def test_start_rejects_malformed_project(client):
    r = await client.post("/automl/start", json={"project_id": "not-a-uuid"})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_ws_start_helper_rejects_invalid():
    from backend.api.automl import _validate_ws_start

    sent = []

    class FakeWS:
        async def send_json(self, payload):
            sent.append(payload)

    before = get_orchestrator().snapshot()["run_context"]
    msg = await _validate_ws_start(
        {"event": "start", "project_id": str(uuid.uuid4()), "config": {"goal": "x"}},
        FakeWS(),
    )
    assert msg is None  # run NOT started
    assert sent and sent[0]["status"] == "error" and sent[0]["code"] == 404
    assert get_orchestrator().snapshot()["run_context"] == before


@pytest.mark.asyncio
async def test_session_project_mismatch_rejected(client):
    # NOTE: POST /sessions has a pre-existing response-serialization bug
    # (returns MetaData instead of dict; untouched per minimal-change rule),
    # so the session row is created directly.
    from uuid import UUID

    from backend.database import async_session_maker
    from backend.database.models import Session as SessionRow

    suffix = uuid.uuid4().hex[:6]
    pa = (await client.post("/projects", json={"name": f"PhaseB-C-{suffix}"})).json()["id"]
    pb = (await client.post("/projects", json={"name": f"PhaseB-D-{suffix}"})).json()["id"]
    async with async_session_maker() as db:
        row = SessionRow(project_id=UUID(pb), name="s1")
        db.add(row)
        await db.commit()
        await db.refresh(row)
        sid = str(row.id)
    r = await client.post("/automl/start", json={"project_id": pa, "session_id": sid, "goal": "x"})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_two_project_runs_stay_isolated(client):
    orch = get_orchestrator()
    bus = get_event_bus()
    suffix = uuid.uuid4().hex[:6]
    ida = (await client.post("/projects", json={"name": f"PhaseB-A-{suffix}"})).json()["id"]
    idb = (await client.post("/projects", json={"name": f"PhaseB-B-{suffix}"})).json()["id"]

    def tiny_cfg(tag):
        return {
            "goal": f"Phase B probe {tag}",
            "dataset_source": "synthetic",
            "max_rows": 300,
            "max_iterations": 50,  # long enough to observe busy-guard
            "target_column": "target",
        }

    async def wait_state(*states, timeout=120):
        for _ in range(int(timeout / 0.2)):
            if orch.state in states:
                return orch.state
            await asyncio.sleep(0.2)
        raise AssertionError(f"orchestrator stuck in {orch.state}")

    # A starts via REST (validated path).
    r = await client.post("/automl/start", json={"project_id": ida, "config": tiny_cfg("A")})
    assert r.status_code == 200 and r.json()["status"] == "started"
    await wait_state("DATA_FETCH", "RESEARCH", "TRAINING")
    assert orch.snapshot()["run_context"]["project_id"] == ida
    run_a = orch.snapshot()["run_context"]["run_id"]
    assert run_a

    # B starts while A runs -> busy, no orphan, A's context intact.
    r = await client.post("/automl/start", json={"project_id": idb, "goal": "y"})
    assert r.json()["status"] == "busy"
    assert orch.snapshot()["run_context"]["project_id"] == ida

    # Stop A, wait for PAUSED, then start B via WS-shaped message (merge path).
    orch.request_stop()
    await wait_state("PAUSED", "DONE", "FAILED")
    res = orch.handle_message({"event": "start", "project_id": idb, "config": tiny_cfg("B")})
    assert res["status"] == "started"
    for _ in range(100):
        ctx = orch.snapshot()["run_context"]
        if ctx["run_id"] and ctx["run_id"] != run_a:
            break
        await asyncio.sleep(0.2)
    assert ctx["project_id"] == idb and ctx["run_id"] != run_a

    # Isolation: every tagged event belongs to exactly one project; B has none of A's.
    events_a = [e for e in bus.replay("t", all_projects=True) if e.get("run_id") == run_a]
    assert events_a, "expected tagged events for run A"
    assert all(e.get("project_id") == ida for e in events_a)
    assert not [e for e in bus.replay("t", all_projects=True) if e.get("project_id") == idb and e.get("run_id") == run_a]

    orch.request_stop()
    await wait_state("PAUSED", "DONE", "FAILED")
