# Phase K — Testing & Hardening (Implemented)

## Suite: 68 tests, all passing (`backend/.venv/bin/python -m pytest tests/ -q`)

`tests/conftest.py` (new) makes the suite hermetic: throwaway SQLite, DuckDB,
MLflow and upload directories via env vars set **before** backend import, an
explicit `init_db()` (ASGITransport does not run the app lifespan, and
`Base.metadata` is empty until the model modules are imported), and an autouse
fixture that stops any in-flight orchestrator run so the singleton cannot leak
state between tests. `CentralStore()`'s default path now comes from settings, so
`CENTRAL_STORE_PATH` is honoured everywhere (previously a hardcoded literal made
tests collide with a running dev server).

### Coverage
- **Project/isolation/context** — Phase B isolation, `PROJECT_CONTEXT.json`
  shape, cross-project memory isolation, live-run ownership.
- **Chat** — persistence + restore, per-session clear, interview signals,
  confirm-gate regression, structured-action parsing, destructive confirmation.
- **Data** — valid upload, unsupported type, malformed CSV, size cap, empty
  dataset, missing target, missing-target rejection in the agent, unique safe
  filenames, staging.
- **Runs** — start, stop, invalid project (404), malformed/unknown run id,
  monotonic progress, persisted events, replay after "refresh", failure path.
- **Experiments** — per-iteration model/params/metrics/training time, MLflow
  linkage, comparison payload, reproduce lineage.
- **Import** — compatible / partially compatible / incompatible, three modes,
  provenance, self-import and unknown-source rejection.
- **Report** — json/markdown/html, limitations present with no data, bad format
  rejected.
- **Model types** — `ModelType` ⇄ registry parity (regression for bug 2).
- **Legacy contracts** — `CentralStore` `source`/`a` alias, alembic stub
  assertion, `/ui` regression.

## Live drills (against a running server, not mocks)
- `e2e_drill.py` — 93/93 checks: full REST+WS surface, isolation, upload matrix,
  diagnostics, import, reproduce, comparison data.
- `acceptance.py` — 41/41 steps of the specified 29-step end-to-end scenario,
  including a real generated CSV, live progress, stop, refresh recovery, all
  seven pages, import with provenance, and an explained refusal. Both drills
  scope their WebSocket with `?project_id=` (as the workspace does), so a
  replayed event from another project can never be mistaken for a reply.

## Bugs found by testing (all fixed, all regression-tested)
1. project-memory cross-project leak (shared mutable default),
2. project-memory lost updates (SQLAlchemy JSON in-place mutation),
3. persisted chat/events silently dropped on SQLite lock contention (bounded
   retry now; 60-turn write test),
4. `EventBus.replay()` broadcast every project's history to every connection —
   the exact "events from Project A must never appear in Project B" violation;
   replay and live pump are now project-scoped and the workspace rescopes its
   socket on project switch,
5. chat replies were tagged with the *run's* project instead of the chat's,
6. interview state survived a project switch, so a new project inherited the
   previous one's goal/metric/row cap,
7. numpy metrics breaking every JSON response/WS frame,
8. any chat text at the confirm gate starting a run,
9. `GET /automl/state` 500 during the data stage,
10. import compatibility judged against the wrong project's working table,
11. `quantile_bin` invalid SQL (never worked),
12. target column mis-detected as `"from"` from "target from my csv",
13. `CentralStore()` ignoring `CENTRAL_STORE_PATH`.
