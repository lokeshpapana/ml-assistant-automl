# Current UI Audit

Date: 2026-09-22. Method: direct code inspection (4 parallel auditors + verifier reads).
Repo: `/home/lokeshpapapa/Downloads/projects/ml_assistant`. All paths below are repo-relative.

> **Headline finding:** the codebase contains **two parallel, disconnected systems**.
> Track 1 (persistent CRUD): `Project → Session → Dataset → Experiment → Model` rows in
> SQLite, REST CRUD, Celery worker, `ExperimentRunner` — but `POST /experiments/{id}/run`
> is a stub (`TODO`, no worker triggered) and the UI never touches it.
> Track 2 (interactive AutoML): singleton `AutoMLOrchestrator` + 4 agents + DuckDB
> `CentralStore` + MLflow/JSON + WebSocket — this is what `/ui` drives, and it is
> **entirely in-memory with zero linkage to Track 1**. Every Phase-3+ feature
> (project workspaces, persistence, import/reuse, checkpoints) must bridge these tracks.

---

## Frontend architecture

- **No framework, no build, no CDN.** Pure HTML + inline `<style>` + inline vanilla JS
  (`"use strict"`). Verified: no `*.tsx/jsx/vue`, no `package.json`, no `<script src>`,
  no `import/require`, no tailwind/bootstrap/React/Vue/Next, no streamlit/gradio/electron
  anywhere outside `.venv`.
- **Two static pages, no router:** `backend/static/automl_ui.html` (~659 lines, the
  Interactive AutoML dashboard) and `backend/static/acquire_ui.html` (122 lines,
  single-shot dataset-acquisition form → JSON dump). No frontend routes exist;
  `/projects/:id/...` pages do not exist in any form.
- **Serving:** `backend/main.py:53-57` mounts `StaticFiles` at `/static`;
  `backend/api/automl.py:113-116` serves `automl_ui.html` as `HTMLResponse` at `GET /ui`;
  `backend/api/dataset_acquisition.py:98-104` serves `acquire_ui.html` at
  `GET /ml/dataset-acquisition/ui`. No separate frontend server.
- **Current `/ui` layout** (`automl_ui.html:205-336`): sticky header (brand, `statePill`,
  `llmPill`, `mlflowPill`, export button) → amber modebar → 3-column grid
  (`292px 1fr 336px`, `automl_ui.html:70-74`): left *Quick start* (goal, preset chips,
  demo/start/stop, `<details>` advanced), center *Agent collaboration* (5-step stepper +
  4 tabs Chat/Metrics/Dataset/Research), right *Run insights* (best card, leaderboard,
  feature importance). ~30 element IDs, 22 named JS functions + 8 listeners.
- **Client→server contract (exact):** WS `ws(s)://{host}/ws/automl` sends
  `{event:"chat",message}`, `{event:"start",config:{goal,metric,dataset_source,
  dataset_query,max_rows,target_column,max_iterations}}`, `{event:"stop"}`;
  REST: `POST /automl/upload` (FormData), `GET /automl/state`, `GET
  /automl/export-best-model` (navigation). Handles 11 inbound types: `chat_response`,
  `agent_message`, `status`, `iteration`, `research_card`, `dataset_preview`,
  `leaderboard`, `feature_importance`, `best_model_ready`, `snapshot`, `ack`.
- **Realtime:** WebSocket only. No SSE (`grep StreamingResponse|text/event-stream`
  → zero hits). Auto-reconnect on close (`setTimeout(connectWS,2000)`).
- **Responsive:** one breakpoint `@media (max-width:1180px)` → single column, chat
  first. Header has no wrap (overflows on phones). `acquire_ui.html` has no media
  queries at all.
- **Other entry point:** `scripts/acquire_cli.py` (72 lines, argparse CLI calling the
  same `acquire_dataset()` core as the REST route; `--target` parsed but never passed).

## Backend architecture

- **FastAPI** (`backend/main.py:28-34`, docs at `/docs`, `/redoc`). Lifespan only does
  `init_db()` / `close_db()` (`main.py:21-25`). Exactly one middleware
  (`CORSMiddleware`, `main.py:36-42`). **No exception handlers, no auth middleware,
  no rate limiting, no logging middleware.**
- **8 routers, no include-time prefixes** (`main.py:44-51`): `projects /projects`,
  `sessions /sessions`, `datasets /datasets`, `experiments /experiments`, `models
  /models`, `ml /ml`, `dataset-acquisition /ml/dataset-acquisition`, `automl`
  (absolute `/automl/*`, `/ui`, `/ws/automl`). 47 HTTP endpoints + 1 WS + `/health`
  + `/` (full table in auditor report; shapes in `backend/api/schemas.py`).
- **Two execution engines (see headline):** (a) Celery+Redis tasks
  (`backend/tasks/ml_tasks.py`: `run_experiment_task`, `profile_dataset_task`,
  `acquire_dataset_task`; broker `redis://…/0`, `backend/tasks/__init__.py:7-25`)
  invoked from `/ml/*` with `async_mode=True` default — **requires an external
  worker + Redis, not managed by FastAPI**; without them async calls hang pending
  (sync fallbacks exist). (b) In-process `threading.Thread` AutoML worker
  (`orchestrator.py:108-109`, daemon, no join/executor).
- **Config:** `backend/config/settings.py` — SQLite default
  (`sqlite+aiosqlite:///./automl_app.db`), `LLM_PROVIDER=local`,
  `MLFLOW_TRACKING_URI=./mlruns`, `CENTRAL_STORE_PATH=./automl_outputs/central_store.duckdb`.
  Fully offline-capable by default; Postgres/Redis only via `docker-compose.yml`.
- **LLM use is rule-based, not model calls.** `InterviewerAgent` *holds* a `BaseLLM`
  (`interviewer.py:72`) but **never calls `.chat()`**; all replies come from the
  stage machine + `_absorb_signals` regexes. Data/Research/Trainer agents emit
  f-string templates. `LocalLLM.chat()` is an unused deterministic echo.

## Current routing

- Backend: REST prefixes above + `WS /ws/automl` + `GET /ui` + `GET
  /ml/dataset-acquisition/ui` + `/health` + `/`. No versioning (`/v1`), no
  `/projects/:id/chat`-style nested routes (project sub-resources are flat query
  params: `POST /sessions?project_id=`, `POST /datasets?project_id=&session_id?`).
- Frontend: none (two static pages). There is **no route to preserve** — greenfield
  for `/projects`, `/projects/:id/*`, `/projects/:id/runs/:runId`.
- Since the UI is static files, new "pages" can be separate static HTML shells
  reusing one shared JS core (recommended, §Recommended architecture), keeping the
  no-build constraint.

## Current project lifecycle

- DB model `Project` (`database/models.py:27-64`): `id/name/description/status
  (active|archived|deleted)/created_at/updated_at` + `sessions/datasets/experiments`
  relationships (cascade delete). Full CRUD via `backend/api/projects.py:18-99`
  (soft-delete → `DELETED`, `204`).
- **The interactive AutoML engine has no project concept.** Its "project" is an
  in-memory `Requirements` dataclass (`agents/common.py:87-102`) + `meta` dict,
  reset on every `start()` (`orchestrator.py:86-91`). It never reads/writes
  `Project/Session/Dataset/Experiment/Model` rows.
- **Sessions** (`sessions` table, `metadata_` JSON) are generic containers, never
  written by the orchestrator; `backend/sessions/` dir is empty.
- Net: project lifecycle exists in Track 1 (create → list → patch → soft-delete)
  and is **absent** in Track 2 (start → run → vanish on restart).

## Current agent lifecycle

- Singleton `AutoMLOrchestrator` (`orchestrator.py:354-359`). States:
  `IDLE→STARTING→DATA_FETCH→RESEARCH→TRAINING→{DONE|PAUSED|FAILED}`
  (`orchestrator.py:29-36,131-255`). **`INTERVIEW` is defined but never assigned.**
- `start(config)` (`:68-110`): busy-guard (`TRAINING|DATA_FETCH|RESEARCH` → `{busy}`),
  `interviewer.reset + ingest_form`, `confirm=True`, clears all run state, spawns one
  daemon thread running `_run_loop`.
- `_run_loop` (`:131-240`): `data_fetcher.fetch` → build `TargetSpec` + `TrainerAgent`
  → loop {research cadence (`feature_engineering_every_n`, one-shot `fe_done`) →
  `trainer.train_iteration` → break on `target_met` or `max_iterations` or
  `stop_event`} → `_finalize` (exports `best_model.pkl`, emits `best_model_ready`).
- **Interviewer pre-loop:** `ingest_form` (from Start button) or `ingest_chat`
  (stage machine goal→metric→source→rows→target_column→confirm + indirect-signal
  absorption + update-on-correction; training launches only on explicit
  start/yes/confirm). `awaiting` property was missing and added during prior work.
- **Stop:** cooperative `threading.Event` (`request_stop`, `:112-127`); finishes the
  current step, then `PAUSED`. **PAUSED is terminal** — no `resume` event; `start()`
  always restarts from iteration 0.
- **Failure:** outer `except → FAILED + error string + agent message` (`:242-251`);
  per-feature research errors are swallowed with a "Skipped" note
  (`researcher.py:97-98`).
- **Agent→tool communication:** no agent-to-agent messages. Orchestrator calls agents
  synchronously in one thread, injecting shared `Requirements`, `CentralStore`,
  `TargetSpec`, and an `emit` callback → `EventBus.publish`. Tools = direct Python
  (DuckDB store, MLflow tracker, httpx research). No function-calling, no MCP use
  at runtime (MCP server exists but the agent uses in-process functions).

## Current dataset lifecycle

- Track 1: `POST /datasets?project_id=` saves to `./uploads/{project_id}/{dataset_id}{ext}`
  (**bug:** filename uses placeholder `UUID(int=0)`, `datasets.py:44-46`), row
  `status=READY` immediately without validation; `GET` list; `DELETE` removes row +
  file. Profiling via `POST /ml/datasets/{id}/profile` (Celery or sync
  `ml.preprocessing.profiler.profile_dataset` → writes `rows/columns/column_info/profile`).
- Track 2: `DataFetcherAgent.fetch` (`data_fetcher.py:69-125`) obtains upload/local /
  OpenML / Kaggle / synthetic → random downsample to `max_rows` → `store.load_dataframe`
  (**drops the single table**, `central_store.py:73-79,123-126`) → `detect_task`
  (explicit target → candidate-name match on `target,label,class,y,outcome,result,
  fraud,price,churn,score` → last column; `common.py:126-149`) → emits
  `dataset_preview`. Single table `central_dataset` in DuckDB
  (`./automl_outputs/central_store.duckdb`) or SQLite fallback; in-place
  `ALTER/UPDATE` features (`add_feature`, 15 kinds).
- No dataset versioning, no lineage, no quality gates before training.

## Current experiment lifecycle

- Track 1: `Experiment` rows (`PENDING→RUNNING→COMPLETED|FAILED|CANCELLED`,
  `config/metrics/started_at/completed_at/error_message`, `models.py:162-211`);
  `Model` rows (`model_type/hyperparameters/metrics/training_time/model_path/
  feature_importance`). `POST /experiments/{id}/run` only flips the flag (**stub**,
  `experiments.py:123-124` TODO). Real execution lives in `ml/experiments/runner.py`
  (`ExperimentRunner`: split → `FeatureEngineer` → registry loop → rank → report
  dict) reachable via `/ml/*` sync path or Celery. Comparison endpoint
  `GET /models/experiment/{id}/comparison` sorts by f1/accuracy/roc_auc.
- Track 2: "experiments" are **iterations** (13-rung ladder:
  default→tuned→hunt configs, `trainer.py:36-66`, `iteration % len`), each an MLflow
  run (`iter_{n}_{model}` + params/metrics/tags/papers + `.pkl` + dataset CSV
  snapshot artifacts) or JSON-fallback equivalent (`fallback_runs/runs.json`).
  Best tracked in `trainer.best`; `best_model.pkl` exported on finalize.
- **Known defects:** `leaderboard()` reads `self.best` (always `None`;
  `snapshot()` correctly reads `trainer.best`); DB `ModelType` has `decision_tree`
  (registry lacks it) while registry has `gradient_boosting/linear_regression/ridge`
  (DB enum lacks them → `POST /models` 400s for ladder models); researcher emits
  feature specs with `a/b` keys but `CentralStore` special branches read `source`
  (`central_store.py:225,235,243`); `ml_tasks.py:53` `db.get(id)` missing model
  class; async acquisition `register` can't resolve `project_id` (never in kwargs).

## Current progress tracking

- **Push-only, in-memory.** `EventBus` (`ws_bus.py:18-82`): `subscribe/unsubscribe/
  replay/publish/publish_sync/send`, `threading.Lock`, `loop.call_soon_threadsafe`
  fan-out, `max_history=2000`. WS handler sends `snapshot` + full `replay` on
  connect, then live-pumps (`automl.py:121-155`).
- Snapshot (`orchestrator.snapshot`, `:311-326`): state, requirements, timestamps,
  error, iterations, feature_history, best, dataset preview(10), tracker info,
  mlflow runs. No percentages, no ETA, no per-step durations, no checkpoint list.
- **Nothing survives restart:** bus history, iterations, Requirements all
  process-local. `PAUSED/DONE` runs cannot be resumed or even re-viewed (snapshot
  only reflects the *current* in-memory run).
- UI mirrors this 1:1 (stepper, feed, chart, leaderboard) — adequate for live view,
  absent for history.

## Existing APIs

47 HTTP + 1 WS (enumerated with file:line in the audit working notes; shapes in
`backend/api/schemas.py`). By area: projects 5, sessions 5, datasets 4,
experiments 6 (incl. stub run), models 5 (incl. comparison), ml 7 (profile/run/
results/best-model/task status+result), dataset-acquisition 2 (acquire + its own
`/ui`), automl 11 REST (`start/stop/state/leaderboard/dataset-preview/upload/
export-best-model/papers/papers/{id}`) + `GET /ui` + `WS /ws/automl`, plus
`GET /health`, `GET /`. Pagination uniform offset/limit on 5 list endpoints.
**No auth on any endpoint** (`backend/auth/` empty; secret key dormant in settings).

## Existing reusable components

- Backend: `FeatureEngineer` (+`OutlierHandler`, `DateTimeFeatureExtractor`),
  `MODEL_REGISTRY` + `compute_metrics` + `get_feature_importance`,
  `DatasetProfiler`, acquisition `spec/sources/evaluate/optimize/agent/report`,
  `ExperimentRunner`, `rank_models`/`select_best_model`/pareto, `CentralStore`,
  `MLflowTracker`, `EventBus`, `ResearchClient` + offline bank, `parse_target_spec/
  met_target/detect_task`, suggestion regexes in UI (`suggestionsFor`).
- Frontend: **none reusable** — one monolithic inline `<script>` (~300 lines);
  no components, no shared JS module, no CSS system (single inline stylesheet,
  CSS variables only). `acquire_ui.html` shares nothing with `automl_ui.html`.
- Tests (32 passing): `test_main`, `test_central_store`, `test_orchestrator`
  (synthetic run + stop), `test_ml_pipeline`, `test_dataset_acquisition`,
  `test_dataset_formats`.

## Current UI problems

1. Static-dashboard feel: primary interaction is a *form* (left) + *tabs* (center),
   chat is one of four tabs rather than the workspace. No project concept anywhere.
2. No persistence of runs: refresh loses everything except the live in-memory run;
   no history, no run pages, no resume.
3. Dead/lying controls: `llmPill` never updates; export button 404s pre-training
   with a raw browser error; Start stays enabled while busy (only a toast);
   upload trusts responses without `res.ok` checks; server accepts any suffix/size.
4. Avatar mismatch: `data-fetcher`/`research` events fall back to grey orchestrator
   avatar (`AGENTS` keys are `data_fetcher`/`researcher`).
5. Single-iteration runs never plot; non-numeric metrics render `NaN`; preview
   capped 10 cols/6 rows with no full view; leaderboard capped at 30 with no
  [...]...[truncated 6158 chars]