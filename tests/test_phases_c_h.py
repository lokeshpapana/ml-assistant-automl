"""Phases C-H regression tests: memory, chat persistence, run events, features,
diagnostics, import compatibility, report, and the bugs fixed along the way."""

import time
import uuid

import pandas as pd
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from backend import runlog
from backend.main import app
from backend.orchestrator import get_orchestrator

os_env = None


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture(autouse=True)
def _store():
    from backend.central_store import get_central_store

    store = get_central_store()
    try:
        store.shape()
    except Exception:
        store.load_dataframe(pd.DataFrame({"a": [1, 2], "target": [0, 1]}))


async def _project(client, prefix):
    r = await client.post("/projects", json={"name": f"{prefix}-{uuid.uuid4().hex[:6]}"})
    return r.json()["id"]


# --------------------------------------------------------------- Phase F memory

def test_memory_appends_persist():
    """Regression: nested-list mutation before assignment made SQLAlchemy skip
    the UPDATE, so every memory write after the first was silently lost."""
    assert runlog.enabled()
    pid = str(uuid.uuid4())
    runlog.remember(pid, "decisions", "one")
    runlog.remember(pid, "decisions", "two")
    runlog.remember(pid, "decisions", "two")  # dedupe
    mem = runlog.get_memory(pid)
    assert mem["decisions"] == ["one", "two"]


def test_memory_is_isolated_per_project():
    """Regression: shallow copy of the default template leaked one project's
    decisions into every other project."""
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    runlog.remember(a, "decisions", "A-only decision")
    runlog.remember(a, "goal", "A goal")
    assert runlog.get_memory(b)["decisions"] == []
    assert runlog.get_memory(b)["goal"] == ""


def test_memory_scalar_section_overwrites():
    pid = str(uuid.uuid4())
    runlog.remember(pid, "target", "churn")
    runlog.remember(pid, "target", "price")
    assert runlog.get_memory(pid)["target"] == "price"


def test_memory_default_has_no_cross_leak_after_many_writes():
    runlog.remember(str(uuid.uuid4()), "decisions", "x" * 3)
    fresh = runlog.get_memory(str(uuid.uuid4()))
    assert fresh["decisions"] == []


# ------------------------------------------------------------- json safety

def test_json_safe_converts_numpy():
    import numpy as np

    from backend.agents.common import json_safe

    out = json_safe({"a": np.float32(1.5), "b": [np.int64(2)], "c": np.float64("nan")})
    assert out == {"a": 1.5, "b": [2], "c": None}
    import json

    json.dumps(out)  # must not raise


def test_snapshot_is_json_serializable_after_run():
    import json

    orch = get_orchestrator()
    snap = orch.snapshot()
    json.dumps(snap)  # numpy metrics/importances must already be coerced


# --------------------------------------------------------------- Phase C chat

@pytest.mark.asyncio
async def test_session_messages_endpoint_and_clear(client):
    pid = await _project(client, "chatproj")
    created = await client.post(f"/sessions?project_id={pid}", json={"name": "s1"})
    assert created.status_code == 201
    sid = created.json()["id"]
    runlog.save_chat(pid, sid, "user", "hello")
    runlog.save_chat(pid, sid, "agent", "hi there", agent="interviewer")
    r = await client.get(f"/sessions/{sid}/messages")
    assert r.status_code == 200
    msgs = r.json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "agent"]
    d = await client.delete(f"/sessions/{sid}/messages")
    assert d.status_code == 204
    r = await client.get(f"/sessions/{sid}/messages")
    assert r.json()["messages"] == []


@pytest.mark.asyncio
async def test_session_metadata_serializes_as_dict(client):
    """Regression (bug 8): POST /sessions raised ResponseValidationError."""
    pid = await _project(client, "sessmeta")
    r = await client.post(f"/sessions?project_id={pid}", json={"name": "abc"})
    assert r.status_code == 201
    assert r.json()["metadata"] == {}
    lst = await client.get(f"/sessions?project_id={pid}")
    assert lst.status_code == 200
    assert isinstance(lst.json()["items"][0]["metadata"], dict)


@pytest.mark.asyncio
async def test_chat_confirm_gate_requires_explicit_yes():
    """Regression: any chat text at the confirm gate used to start a run."""
    from backend.agents.interviewer import InterviewerAgent

    ag = InterviewerAgent()
    ag.ingest_chat("predict churn from my csv at 90 percent accuracy")
    ag.ingest_chat("5000 rows")
    ag.ingest_chat("auto")
    assert ag.awaiting == "confirm"
    ag.ingest_chat("use xgboost")  # follow-up instruction, not a confirmation
    assert ag.awaiting == "confirm"
    assert ag.is_confirmed() is False
    reply, advanced = ag.ingest_chat("start")
    assert advanced is True
    assert ag.is_confirmed() is True


def test_parse_action_recognizes_intents():
    from backend.agents.common import Requirements
    from backend.agents.interviewer import InterviewerAgent

    ag = InterviewerAgent()
    req = Requirements()
    a = ag.parse_action("use xgboost", req)
    assert a and a["type"] == "select_model" and a["model"] == "xgboost" and not a["destructive"]
    a = ag.parse_action("try random forest", req)
    assert a and a["model"] == "random_forest"
    a = ag.parse_action("remove feature age", req)
    assert a and a["type"] == "remove_feature" and a["column"] == "age" and a["destructive"]
    a = ag.parse_action("undo that", req)
    assert a and a["type"] == "undo_last_feature"
    a = ag.parse_action("set the target to churn", req)
    assert a and a["type"] == "set_target_column" and a["column"] == "churn"
    a = ag.parse_action("remove column feature", req)
    assert a is None  # keyword, not a real column
    assert ag.parse_action("what is the weather", req) is None


# ------------------------------------------------------------ Phase D events

def test_candidate_ladder_params_match_estimators():
    """Regression: hunt-phase rungs passed `learning_rate` to RandomForest,
    raising TypeError and failing any run that reached them."""
    from ml.models.registry import instantiate_model

    from backend.agents.trainer import build_candidate_ladder

    for task in ("classification", "regression"):
        for rung in build_candidate_ladder(task):
            model = instantiate_model(task, rung["model"], rung["params"])
            assert model is not None, rung


@pytest.mark.asyncio
async def test_resume_continues_from_checkpoint(client):
    """Resume must continue the model ladder from the persisted checkpoint,
    record lineage, and not retrain models the source run already tried."""
    from backend.orchestrator import get_orchestrator

    pid = await _project(client, "resume")
    cfg = {"goal": "resume probe", "dataset_source": "synthetic", "max_rows": 500,
           "max_iterations": 50, "target_column": "target", "metric": "accuracy >= 0.99999"}
    r = await client.post("/automl/start", json={"project_id": pid, "config": cfg})
    assert r.json()["status"] == "started"
    run_id = None
    trained = 0
    for _ in range(900):
        s = (await client.get("/automl/state")).json()
        run_id = (s.get("run_context") or {}).get("run_id") or run_id
        trained = len(s.get("iterations") or [])
        if s["state"] in ("TRAINING", "RESEARCH", "DATA_FETCH") and trained >= 2:
            break
        if s["state"] in ("DONE", "PAUSED", "FAILED"):
            break
        await asyncio_sleep()
    await client.post("/automl/stop")
    for _ in range(400):
        s = (await client.get("/automl/state")).json()
        if s["state"] in ("PAUSED", "DONE", "FAILED"):
            break
        await asyncio_sleep()
    assert s["state"] in ("PAUSED", "DONE")
    first_models = [i["model"] for i in (s.get("iterations") or [])]
    assert len(first_models) >= 1, first_models

    r = await client.post("/automl/resume", json={"project_id": pid, "run_id": run_id})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["status"] == "started", out
    assert out["resumed_from"] == run_id
    assert out["resume_index"] == len(first_models)

    new_run = None
    for _ in range(900):
        s2 = (await client.get("/automl/state")).json()
        rc = s2.get("run_context") or {}
        if rc.get("run_id") and rc["run_id"] != run_id:
            new_run = rc["run_id"]
            break
        if s2["state"] in ("DONE", "PAUSED", "FAILED"):
            await asyncio_sleep()
    assert new_run and new_run != run_id
    rec = (await client.get(f"/runs/{new_run}")).json()
    assert rec["run"]["config"]["resumes_run_id"] == run_id
    assert rec["run"]["config"]["resume_index"] == len(first_models)
    # The resumed campaign must not repeat the models the source already trained.
    resumed_models = [e["metadata"]["model"] for e in rec["events"] if e["event_type"] == "model_training"]
    if resumed_models:
        assert resumed_models[0] not in first_models, (first_models, resumed_models)
    orch = get_orchestrator()
    if orch.state in ("TRAINING", "RESEARCH", "DATA_FETCH"):
        orch.request_stop()


@pytest.mark.asyncio
async def test_resume_rejects_foreign_project_and_unknown_run(client):
    a = await _project(client, "resA")
    b = await _project(client, "resB")
    run = runlog.save_run("run-dddddddddddd", a, None, {"goal": "x", "metric": "accuracy"})
    assert run
    r = await client.post("/automl/resume", json={"project_id": b, "run_id": "run-dddddddddddd"})
    assert r.status_code == 400, r.text  # belongs to another project
    r = await client.post("/automl/resume", json={"project_id": a, "run_id": "run-eeeeeeeeeeee"})
    assert r.status_code == 404
    r = await client.post("/automl/resume", json={"project_id": "nope", "run_id": "run-dddddddddddd"})
    assert r.status_code == 400
    r = await client.post("/automl/resume", json={})
    assert r.status_code == 200
    # Ad-hoc (no project): either nothing to resume, or a global run was picked.
    assert r.json()["status"] in ("no_resumable_run", "already_finished", "not_found",
                                  "no_config", "started", "busy")
    if r.json()["status"] == "started":
        get_orchestrator().request_stop()


@pytest.mark.asyncio
async def test_trainer_resume_offset_skips_used_rungs():
    from backend.agents.trainer import TrainerAgent

    fresh = TrainerAgent(task_type="classification")
    resumed = TrainerAgent(task_type="classification", start_index=2)
    # Selection, not the ladder itself, is offset by the resume checkpoint.
    assert fresh.model_for(0)["model"] == fresh.ladder[0]["model"]
    assert resumed.model_for(0)["model"] == fresh.ladder[2]["model"]
    assert resumed.model_for(1)["model"] == fresh.ladder[3]["model"]
    assert TrainerAgent(task_type="classification", start_index=0).model_for(0)["model"] == fresh.ladder[0]["model"]


@pytest.mark.asyncio
async def test_import_review_returns_diff_preview(client):
    """REVIEW mode must show what *would* be applied, not only a verdict."""
    a = await _project(client, "diffA")
    b = await _project(client, "diffB")
    run = runlog.save_run("run-cccccccccccc", a, None, {"metric": "accuracy", "dataset_source": "synthetic"})
    assert run
    runlog.update_run("run-cccccccccccc", status="done", best={}, research=[],
                      feature_history=[{"name": "f_p", "kind": "product", "a": "a", "b": "b",
                                        "paper_id": "p", "technique": "interaction"}])
    r = await client.post(f"/projects/{b}/imports", json={
        "source_project_id": a, "mode": "review", "source_run_id": "run-cccccccccccc",
        "resources": [{"type": "feature_pipeline"}, {"type": "experiment_config"}]})
    assert r.status_code == 200
    for res in r.json()["results"]:
        assert "would_apply" in res
        assert res["applied"] is False
    fp = next(x for x in r.json()["results"] if x["type"] == "feature_pipeline")
    assert fp["would_apply"]["items"], "feature pipeline diff must list concrete items"
    assert any("f_p" in str(i) for i in fp["would_apply"]["items"])


def test_concurrent_chat_writes_are_never_dropped():
    """Regression: a transient SQLite lock silently discarded chat turns and
    events, because the sync writer and the async app engine share one file."""
    assert runlog.enabled()
    pid, sid = str(uuid.uuid4()), str(uuid.uuid4())
    for i in range(30):
        runlog.save_chat(pid, sid, "user", f"m{i}")
        runlog.save_chat(pid, sid, "agent", f"r{i}", agent="interviewer")
    msgs = runlog.list_chat(session_id=sid, limit=100)
    assert len(msgs) == 60, f"expected 60 persisted turns, got {len(msgs)}"


def test_event_bus_replay_is_project_scoped():
    """Regression: EventBus.replay() broadcast the whole buffer to every new
    connection, so project A's events (including chat replies) surfaced in
    project B's UI."""
    from backend.ws_bus import EventBus

    bus = EventBus()
    bus.publish({"type": "status", "project_id": "A", "state": "TRAINING"})
    bus.publish({"type": "status", "project_id": "B", "state": "DATA_FETCH"})
    bus.publish({"type": "status", "project_id": None, "state": "IDLE"})

    only_a = bus.replay("c1", project_id="A")
    assert [e["project_id"] for e in only_a] == ["A"]
    unassigned = bus.replay("c1", project_id=None)
    assert [e["project_id"] for e in unassigned] == [None]
    everything = bus.replay("c1", all_projects=True)
    assert len(everything) == 3


@pytest.mark.asyncio
async def test_interview_state_does_not_leak_between_projects(client):
    """Regression: the singleton orchestrator's interview state (pending goal,
    metric and 'explicit' flags) survived a project switch, so a new project
    inherited the previous project's requirements."""
    a = await _project(client, "leakA")
    b = await _project(client, "leakB")
    orch = get_orchestrator()

    orch.handle_message({"event": "chat", "project_id": a, "session_id": None,
                         "message": "predict churn at 75 percent accuracy from csv, 900 rows, target is churn"})
    assert orch.interviewer.requirements.metric == "accuracy"
    assert orch.interviewer.requirements.target_value == 0.75

    orch.handle_message({"event": "chat", "project_id": b, "session_id": None,
                         "message": "predict rent at 88 percent accuracy from csv, 1200 rows, target is rent"})
    assert orch.interviewer.requirements.target_value == 0.88
    assert orch.interviewer.requirements.target_column == "rent"


@pytest.mark.asyncio
async def test_run_lifecycle_persists_events_and_record(client):
    pid = await _project(client, "runlife")
    cfg = {"goal": "test run", "dataset_source": "synthetic", "max_rows": 400,
           "max_iterations": 1, "target_column": "target", "metric": "accuracy >= 0.995"}
    r = await client.post("/automl/start", json={"project_id": pid, "config": cfg})
    assert r.json()["status"] == "started"
    run_id = None
    for _ in range(900):
        s = (await client.get("/automl/state")).json()
        rc = s.get("run_context") or {}
        run_id = rc.get("run_id") or run_id
        if s["state"] in ("DONE", "PAUSED", "FAILED"):
            break
        await asyncio_sleep()
    assert run_id
    r = await client.get(f"/runs/{run_id}")
    assert r.status_code == 200
    rec = r.json()
    types = {e["event_type"] for e in rec["events"]}
    for required in ("run_started", "data_loading", "data_validation", "model_training", "run_completed"):
        assert required in types, types
    assert rec["run"]["progress"] > 0
    assert rec["run"]["status"] in ("done", "paused")
    assert rec["mlflow_run_ids"], "MLflow lineage must be linked"
    # Replay works from persisted data alone (no live state needed).
    assert rec["run"]["dataset"]["rows"] > 0


async def asyncio_sleep():
    import asyncio

    await asyncio.sleep(0.25)


@pytest.mark.asyncio
async def test_run_id_validation(client):
    assert (await client.get("/runs/run-zzzzzzzzzzzz")).status_code == 400
    assert (await client.get("/runs/run-000000000000")).status_code == 404


@pytest.mark.asyncio
async def test_progress_is_stage_based_not_fake(client):
    pid = await _project(client, "prog")
    cfg = {"goal": "g", "dataset_source": "synthetic", "max_rows": 300,
           "max_iterations": 2, "target_column": "target", "metric": "accuracy >= 0.995"}
    await client.post("/automl/start", json={"project_id": pid, "config": cfg})
    run_id = None
    for _ in range(900):
        s = (await client.get("/automl/state")).json()
        run_id = (s.get("run_context") or {}).get("run_id") or run_id
        if s["state"] in ("DONE", "PAUSED", "FAILED"):
            break
        await asyncio_sleep()
    rec = (await client.get(f"/runs/{run_id}")).json()
    progs = [e["progress"] for e in rec["events"] if e.get("progress") is not None]
    assert progs == sorted(progs), "progress must be monotonic"
    assert rec["run"]["progress"] == 100.0


# ------------------------------------------------------------- Phase E features

@pytest.mark.asyncio
async def test_feature_state_and_lifecycle(client):
    store = get_orchestrator().store
    store.load_dataframe(pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "target": [0, 1, 0]}))
    r = await client.get("/automl/features")
    assert r.status_code == 200
    assert "target" in r.json()["columns"]
    r = await client.post("/automl/features/disable", json={"column": "a"})
    assert "Disabled" in r.json()["note"]
    assert "a" in r.json()["excluded"]
    r = await client.post("/automl/features/enable", json={"column": "a"})
    assert "Re-enabled" in r.json()["note"]
    r = await client.post("/automl/features/remove", json={"column": "b"})
    assert r.status_code == 200
    r = await client.get("/automl/features")
    assert "b" not in r.json()["columns"]
    assert any(op["op"] == "drop" for op in r.json()["ops"])


@pytest.mark.asyncio
async def test_target_protection_and_validation(client):
    store = get_orchestrator().store
    store.load_dataframe(pd.DataFrame({"a": [1.0, 2.0], "target": [0, 1]}))
    orch = get_orchestrator()
    orch.requirements.target_column = "target"
    r = await client.post("/automl/features/remove", json={"column": "target"})
    assert r.status_code == 200 and "target" in r.json()["note"].lower()
    assert (await client.post("/automl/target", json={"column": "nope"})).status_code == 400
    r = await client.post("/automl/target", json={"column": "a"})
    assert r.json()["target"] == "a"


@pytest.mark.asyncio
async def test_upload_validation_matrix(client):
    pid = await _project(client, "uploads")
    files = {
        "bad.exe": (b"MZ binary", "evil.exe", "application/octet-stream"),
        "broken.csv": (b"a,b\n\x00\xff\xfe", "broken.csv", "text/csv"),
        "ok.csv": (b"a,b,target\n1,2,0\n3,4,1\n", "ok.csv", "text/csv"),
    }
    for _, (data, name, ctype) in files.items():
        r = await client.post(
            f"/datasets?project_id={pid}",
            files={"file": (name, data, ctype)},
        )
        if name == "ok.csv":
            assert r.status_code == 201, r.text
            assert r.json()["rows"] == 2
        else:
            assert r.status_code == 400, (name, r.status_code, r.text)
            assert r.json()["detail"]


@pytest.mark.asyncio
async def test_dataset_filename_is_unique_and_safe(client):
    pid = await _project(client, "fname")
    r1 = await client.post(f"/datasets?project_id={pid}", files={"file": ("a.csv", b"x,target\n1,0\n", "text/csv")})
    r2 = await client.post(f"/datasets?project_id={pid}", files={"file": ("a.csv", b"y,target\n1,1\n", "text/csv")})
    assert r1.status_code == 201 and r2.status_code == 201
    assert r1.json()["id"] != r2.json()["id"]  # regression: placeholder UUID(int=0) collided
    assert r1.json()["file_path"] != r2.json()["file_path"]


# ------------------------------------------------------------ Phase G imports

@pytest.mark.asyncio
async def test_import_modes_and_provenance(client, tmp_path):
    a = await _project(client, "impA")
    b = await _project(client, "impB")
    # Source project with a real artifact.
    orch = get_orchestrator()
    store = orch.store
    store.load_dataframe(pd.DataFrame({"a": [1.0, 2.0, 3.0], "target": [0, 1, 0]}))
    orch.store.add_feature({"kind": "product", "a": "a", "name": "f_prod_a", "paper_id": "2106.01663",
                            "technique": "interaction", "b": "a"})
    model = tmp_path / "m.pkl"
    model.write_bytes(b"x")
    run = runlog.save_run("run-aaaaaaaaaaaa", a, None, {"metric": "accuracy", "dataset_source": "synthetic"})
    assert run
    runlog.update_run("run-aaaaaaaaaaaa", status="done", progress=100.0,
                      best={"model": "random_forest", "iteration": 1, "model_path": str(model),
                            "dataset_snapshot": "", "target_metric": 0.9, "metrics": {"accuracy": 0.9}},
                      feature_history=[{"name": "f_prod_a", "kind": "product", "a": "a", "b": "a",
                                        "paper_id": "2106.01663", "technique": "interaction"}],
                      research=[{"paper_id": "2106.01663", "title": "t", "technique": "interaction", "feature": "f_prod_a"}])

    r = await client.post(f"/projects/{b}/imports", json={
        "source_project_id": a, "mode": "review",
        "resources": [{"type": "feature_pipeline"}, {"type": "model_artifact"},
                      {"type": "research"}, {"type": "bogus"}]})
    assert r.status_code == 200
    res = {x["type"]: x for x in r.json()["results"]}
    assert all(not x["applied"] for x in r.json()["results"]), "review must not apply"
    assert res["bogus"]["verdict"] == "incompatible"
    assert all(x["reasons"] for x in r.json()["results"])

    r = await client.post(f"/projects/{b}/imports", json={
        "source_project_id": a, "mode": "smart",
        "resources": [{"type": "feature_pipeline"}, {"type": "model_artifact"}]})
    res = r.json()["results"]
    assert any(x["applied"] for x in res), res
    assert all(x["reasons"] for x in res)

    r = await client.get(f"/projects/{b}/imports")
    imports = r.json()["imports"]
    assert imports, "provenance must be recorded"
    assert imports[0]["source_project_id"] == a
    assert imports[0]["source_run_id"] == "run-aaaaaaaaaaaa"
    assert imports[0]["mode"] in ("quick", "smart", "review")


@pytest.mark.asyncio
async def test_import_rejects_self_and_unknown(client):
    a = await _project(client, "selfimp")
    r = await client.post(f"/projects/{a}/imports", json={
        "source_project_id": a, "mode": "quick", "resources": [{"type": "research"}]})
    assert r.status_code == 400
    r = await client.post(f"/projects/{a}/imports", json={
        "source_project_id": str(uuid.uuid4()), "mode": "quick", "resources": [{"type": "research"}]})
    assert r.status_code == 404
    r = await client.post(f"/projects/{a}/imports", json={
        "source_project_id": a, "mode": "bogus", "resources": [{"type": "research"}]})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_import_cross_project_compatibility_uses_own_context(client):
    """Regression: the import check compared against the *source* project's
    leftover working table (the central store is a global singleton)."""
    a = await _project(client, "compatA")
    b = await _project(client, "compatB")
    orch = get_orchestrator()
    orch.project_id = a  # live table belongs to A
    orch.store.load_dataframe(pd.DataFrame({"a": [1.0, 2.0], "target": [0, 1]}))
    runlog.save_run("run-bbbbbbbbbbbb", a, None, {"metric": "accuracy"})
    runlog.update_run("run-bbbbbbbbbbbb", status="done",
                      feature_history=[{"name": "f_x", "kind": "product", "a": "a", "b": "a",
                                        "paper_id": "x", "technique": "t"}],
                      best={}, research=[])
    r = await client.post(f"/projects/{b}/imports", json={
        "source_project_id": a, "mode": "review", "resources": [{"type": "feature_pipeline"}]})
    verdict = r.json()["results"][0]
    # B has no working table of its own: must be staged/partially compatible,
    # never "compatible" based on A's leftover columns.
    assert verdict["verdict"] == "partially_compatible", verdict
    assert "no working table" in " ".join(verdict["reasons"]).lower()
    orch.project_id = None


# ------------------------------------------------------------ Phase E reports

@pytest.mark.asyncio
async def test_report_formats_and_honesty(client):
    pid = await _project(client, "rep")
    for fmt in ("json", "markdown", "html"):
        r = await client.get(f"/projects/{pid}/report?format={fmt}")
        assert r.status_code == 200 and len(r.text) > 50
    r = await client.get(f"/projects/{pid}/report?format=json")
    data = r.json()
    assert data["limitations"], "must state limitations even with no data"
    assert data["run"] is None
    assert (await client.get(f"/projects/{pid}/report?format=xml")).status_code == 422


# ------------------------------------------------- model-type enum consistency

def test_model_type_enum_matches_registry():
    """Regression (bug 2): DB ModelType was missing registry names, so POST
    /models 400'd for ladder models like gradient_boosting/ridge."""
    from backend.database.models import ModelType
    from ml.models.registry import get_available_models

    registry = set(get_available_models("classification")) | set(get_available_models("regression"))
    enum_values = {m.value for m in ModelType}
    missing = registry - enum_values
    assert not missing, f"registry models absent from DB ModelType: {missing}"
    # canonical direction: every enum value is a real registry model
    stale = {v for v in enum_values if v != "decision_tree"} - registry
    assert not stale, f"DB ModelType values with no registry implementation: {stale}"


# ---------------------------------------------------------- legacy research spec

def test_central_store_accepts_source_alias():
    """Regression (bug 3): freq_encode/quantile_bin/date parts read `source`
    while the researcher emits `a`."""
    from backend.central_store import CentralStore

    store = CentralStore(memory=True)
    store.load_dataframe(pd.DataFrame({
        "cat": ["x", "y", "x", "y"], "num": [1.0, 2.0, 3.0, 4.0], "target": [0, 1, 0, 1]}))
    assert store.add_feature({"kind": "freq_encode", "a": "cat", "name": "f_freq_cat"})["name"]
    assert store.add_feature({"kind": "quantile_bin", "a": "num", "name": "f_bin_num"})["name"]
    # legacy alias still works
    assert store.add_feature({"kind": "freq_encode", "source": "cat", "name": "f_freq_cat2"})["name"]


def test_data_fetcher_rejects_empty_and_missing_target(tmp_path):
    from backend.agents.common import Requirements
    from backend.agents.data_fetcher import DataFetcherAgent

    good = tmp_path / "ok.csv"
    good.write_text("a,b,target\n1,2,0\n3,4,1\n5,6,0\n")
    agent = DataFetcherAgent()

    empty = tmp_path / "empty.csv"
    empty.write_text("")
    req = Requirements(dataset_source="upload", dataset_path=str(empty))
    with pytest.raises(ValueError):
        agent.fetch(req, lambda e: None)

    req = Requirements(dataset_source="upload", dataset_path=str(good), target_column="does_not_exist")
    with pytest.raises(ValueError, match="not found in the dataset"):
        agent.fetch(req, lambda e: None)


# --------------------------------------------------------------- migration stub

def test_alembic_state_is_documented():
    """Regression (bug 10): alembic has a [alembic] section but no
    ``script_location`` and no revision files, so `alembic upgrade head`
    cannot work. It must stay an explicitly documented stub: the live schema is
    created by Base.metadata.create_all() in backend/database/__init__.py."""
    from pathlib import Path

    ini = (Path("backend") / "alembic.ini").read_text()
    configured = {
        line.split("=", 1)[0].strip()
        for line in ini.splitlines()
        if "=" in line and not line.strip().startswith("#")
    }
    versions = list((Path("backend") / "alembic" / "versions").glob("*.py"))
    assert "script_location" not in configured, "alembic.ini gained a script_location"
    assert not versions, "alembic gained revisions; the stub is no longer a stub"
    assert "create_all" in (Path("backend") / "database" / "__init__.py").read_text(), (
        "the real schema path must remain create_all() on startup"
    )
