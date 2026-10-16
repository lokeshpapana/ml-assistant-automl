# Phase B Audit — Project-Aware Workspace Foundation

Date: 2026-09-22. Scope: ONLY what Phase B needs (shell, sidebar, routing, start +
`project_id` propagation, run context, WS tagging, activity panel, run view).
Prior full-system reports: `.agent/UI_AUDIT.md`, `.agent/PROJECT_MODEL.md`.

## Current frontend architecture

- No framework/build/CDN: `backend/static/automl_ui.html` (~659 lines, inline CSS +
  inline vanilla JS) served at `GET /ui` (`backend/api/automl.py:113-116`); `GET
  /static` mount in `backend/main.py:53-57`. Second page `acquire_ui.html` is
  unrelated (dataset acquisition form). No frontend router exists.
- UI state is local JS variables (`automl_ui.html` ~line 340: `state = {chart, state}`);
  project concept absent. Server state arrives via `GET /automl/state` + WS
  `snapshot`/`replay` on connect (`automl.py:131-133`).
- Client→server: WS `{event:"chat"|"start"|"stop"}` (`orchestrator.handle_message`,
  `orchestrator.py:271-307`); REST `POST /automl/upload`, `GET /automl/state`,
  `GET /automl/export-best-model`. Eleven inbound event types handled (see
  `.agent/UI_AUDIT.md` → Existing APIs/contract table).

## Current backend architecture

- FastAPI, 8 routers (`backend/main.py:44-51`), one `CORSMiddleware`, no auth, no
  exception handlers, no rate limiting. Lifespan = `init_db`/`close_db` only.
- Two engines: (a) Celery+Redis (`backend/tasks/`) used by `/ml/*` async paths —
  needs external worker, untouched by Phase B; (b) in-process daemon thread running
  the singleton `AutoMLOrchestrator` (`orchestrator.py:108-109`, `get_orchestrator`
  `:354-359`) — this is what Phase B tags with project/run context.
- Config defaults fully local: SQLite `./automl_app.db`, `LLM_PROVIDER=local`,
  MLflow `./mlruns` (JSON fallback active), DuckDB central store.

## Relevant API routes (Phase B surface)

| Route | File:line | Phase B use |
|---|---|---|
| `POST /projects` / `GET /projects` (paginated, `status?` filter) | `api/projects.py:18,30` | Sidebar list + create. Real data, no changes needed |
| `GET /projects/{id}` | `api/projects.py:57` | Workspace header + Overview page |
| `GET /sessions?project_id=` / `POST /sessions?project_id=` | `api/sessions.py:18,36` | Optional session linkage (validate only) |
| `GET /datasets?project_id=` / `GET /experiments?project_id=` | `api/datasets.py:70` / `api/experiments.py:46` | Overview counts (read-only) |
| `POST /automl/start` (body IS the config dict) | `api/automl.py:46-48` | **Must change**: extract + validate `project_id`/`session_id`, then start |
| `GET /automl/state`, `WS /ws/automl` | `api/automl.py:57,121` | State + events; snapshot/emit must carry run context |
| `GET /ui` | `api/automl.py:113` | **Untouched** (regression-tested) |

## Project lifecycle (today)

`Project{name*,description?,status=ACTIVE,created_at,updated_at}` with
`sessions/datasets/experiments` cascade relations (`database/models.py:27-64`).
CRUD: create→list→get→patch→soft-delete (`api/projects.py:18-99`). Nothing in the
AutoML track references it.

## AutoML run lifecycle (today)

`start(config)` (`orchestrator.py:68-110`): busy-guard → `interviewer.reset` →
`ingest_form(config)` (**ignores unknown keys** — verified `interviewer.py:132-158`,
so `project_id` inside config is safely ignored today) → clears run state →
emits `STARTING` → daemon thread `_run_loop` (`:131-240`):
`DATA_FETCH → [RESEARCH] → TRAINING × N → DONE | PAUSED (stop_event) | FAILED
(exception)`. `PAUSED` is terminal; `start()` always restarts from iteration 0.
Singleton → **one run at a time globally**; concurrent start returns `{busy}`.

## WebSocket lifecycle (today)

`automl_websocket` (`api/automl.py:121-155`): accept → subscribe (server loop) →
send `snapshot` + full `replay` → pump task forwards queue → inbound JSON →
`handle_message` → `ack`. No project/run association on any event; `_emit`
(`orchestrator.py:343-347`) stamps only `state` + `ts`. History = volatile
≤2000-event list.

## Current project_id flow

Nowhere. Enumerated: (1) UI never sends it (no project concept in
`automl_ui.html`); (2) `POST /automl/start` forwards the raw dict;
(3) `ingest_form` drops unknown keys; (4) orchestrator stores no context;
(5) events carry no context; (6) Celery `acquire_dataset_task` receives
`register_as_dataset: bool` but never `project_id` (`dataset_acquisition.py:55`
vs `ml_tasks.py:166` — pre-existing gap, out of Phase B scope).

## Missing project_id links (the Phase B work list)

1. Entry validation: neither REST nor WS start validates `project_id` (nothing to
   validate against in the flow — the check itself is missing).
2. Run context: orchestrator has no `project_id`/`run_id`/`session_id` fields.
3. Event tagging: `_emit` stamps no context; cross-project filtering impossible.
4. Snapshot: no `run_context` block for the activity panel / run view.
5. UI: no sidebar, no selection state, no routing, no per-project filtering.

## Files that must change (minimum set)

1. `backend/orchestrator.py` — add `project_id/session_id/run_id` fields; set them
   in `start()` from config; stamp every `_emit`; include `run_context` in
   `snapshot()` (+ `leaderboard()` for consistency). No thread/state-machine changes.
2. `backend/api/automl.py` — REST start: extract `project_id`/`session_id`
   (top-level or nested in config; old flat-config bodies keep working), validate,
   404/400 on invalid, pass through. WS handler: same validation for
   `event=="start"` before `handle_message` (send error `ack`, start nothing).
   Add `GET /workspace` serving the new shell. `GET /ui` byte-identical.
3. NEW `backend/project_context.py` — one shared async validator
   (`validate_run_context(project_id, session_id, db)`): UUID parse → Project
   exists (404) → optional session exists AND belongs to project (400). Used by
   both entry points. No other backend changes.
4. NEW `backend/static/workspace.html` — self-contained shell (sidebar, hash
   routing, chat, activity rail, run view, placeholders). Duplicates needed JS
   deliberately so `automl_ui.html` is never touched.
5. NEW `tests/test_phase_b.py` — validation, propagation, isolation, regression
   tests (httpx `ASGITransport`, `asyncio_mode=auto` already configured).

## Files that should remain untouched

- `backend/static/automl_ui.html`, `acquire_ui.html` (regression surface).
- `backend/orchestrator.py` thread/state-machine logic (only additive context).
- `backend/ws_bus.py`, `backend/central_store.py`, `backend/mlflow_tracker.py`,
  all of `backend/agents/*`, `backend/research/*` (no changes needed).
- Track 1 CRUD (`projects/sessions/datasets/experiments/models` routers + models),
  `backend/tasks/*`, `ml/*`, `docker-compose.yml`, `alembic/`.
- `backend/main.py` (no new mounts/routers needed; `/workspace` lives on the
  existing automl router).

## Phase B risks

1. **Singleton = one global run.** "Project B starts while A runs" can only yield
   `busy`. Isolation is therefore *tagging + filtering*, not concurrency. The UI
   must not imply parallel runs. (Mitigation: busy toast names the owning project.)
2. **No persistence → honest UX burden.** Refresh loses run activity; run view only
   renders the *live* run. Every label must say "live/active", never
   "saved/resumable". The `replay` buffer is the only history.
3. **Old-/ui runs have `project_id=null`.** Workspace filters them out of project
   views (correct isolation) — verify this in testing, or classic-UI runs will
   "leak" into project panels.
4. **UUID vs example `{project_id: 17}`.** Real ids are UUID strings; validator
   must 400/404 (not 500) on malformed ids. `session_id` must be checked against
   the *same* project (confused-deputy across projects).
5. **WS handler validation errors** must be sent as `ack`-style errors, never raise
   (raising kills the socket loop for all of that client's pending work).
6. **Scope creep into Track 1 writes.** Tempting to persist iterations→Experiment
   rows "while here" — explicitly Phase E. Resist.
