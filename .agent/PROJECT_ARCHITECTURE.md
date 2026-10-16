# Project Architecture (post–Phase K)

## TRACK 1 — Persistent domain model
FastAPI CRUD for `Project → Session → Dataset → Experiment → Model`, plus the
five tables added for the interactive engine: `runs`, `run_events`,
`chat_messages`, `project_memory`, `project_imports`. SQLite by default
(`create_all` on startup), optional Postgres via `DATABASE_URL`; Celery+Redis
tasks exist for the `/ml/*` async paths (external worker). No auth.

## TRACK 2 — In-memory execution system
Singleton `AutoMLOrchestrator` on a daemon thread: interview → data fetch →
research/feature engineering → training ladder, cooperative STOP, MLflow/JSON
tracking, `CentralStore` (one DuckDB table), `EventBus` → `WS /ws/automl`.
Agents are rule-based; `LLM_PROVIDER=local` by default.

## PHASE B–K BRIDGE (read-only validation, then write-through)
1. `backend/project_context.py` validates `project_id`/`session_id` (404/400) at
   both entry points before any run starts; no orphan runs.
2. Orchestrator carries `project_id`/`session_id`/`run_id` as explicit
   association labels; every WS event and `snapshot()` is tagged, so a UI can
   never mistake one project's run for another's.
3. `backend/runlog.py` performs best-effort write-through from the worker thread
   into SQLite: run record, append-only events, chat turns, structured memory,
   import provenance. Failures warn, never block training.
4. Reads go the other way: the UI, reports, comparison and diagnostics read
   durable rows/artifacts, so a browser refresh loses nothing.

## Single source of truth per fact
| Fact | Owner |
|---|---|
| Dataset contents | `CentralStore` (DuckDB, one table) + per-iteration CSV snapshots |
| Run params/metrics/artifacts | MLflow (`iter_N_model` runs; `mlflow_run_ids` linked in events) |
| Run status/progress/timeline | `runs` + `run_events` |
| Best model of a run | `trainer.best` in memory, `runs.best` persisted (same value) |
| Chat history | `chat_messages` (per session) |
| Project knowledge | `project_memory` (structured sections) |
| Imported artifacts | `project_imports` (provenance) |
| Project/session/dataset rows | `projects`/`sessions`/`datasets` |

## Deliberately not done
Auth, per-project concurrency, mid-run resume, cross-validation, PDF export,
alembic migrations (schema is `create_all`; stub is asserted by test).
