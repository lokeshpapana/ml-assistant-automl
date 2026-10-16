# UI_FINAL_REPORT — Interactive AutoML Platform

Date: 2026-09-28 · Scope: Phase B → K (C–K implemented on top of the existing
engine; nothing was rebuilt, no second backend/frontend was created).

## 1. What was implemented
- **Phase C** persistent per-project chat (own table, per-session history,
  restore, sessions, rule-based compaction, structured validated actions,
  confirmation gating, project-state reference resolution).
- **Phase D** persisted run/event log with 10 canonical event types, stage-based
  monotonic progress, real ETA from observed iteration times, `GET /runs/{id}`,
  and **resume from checkpoint** (`POST /automl/resume`, chat "resume the run",
  run-page button): the next ladder rung is derived from the persisted
  `model_training` events, so a resumed campaign continues instead of repeating
  work, and records `resumes_run_id` for lineage.
- **Phase E** seven real pages (Data, Analysis, Research, Features, Experiments,
  Evaluation, Report) + Run view + Compare + Import, all backed by stored data
  and recomputed evidence.
- **Phase F** structured per-project memory + compact `PROJECT_CONTEXT.json`.
- **Phase G** import with quick/smart/review modes, compatibility verdicts with
  reasons, adaptation or refusal, and provenance rows.
- **Phase H** comparison from the run's own objective, reproduce with lineage,
  and recomputed model diagnostics (confusion/ROC/PR/residuals/permutation
  importance/error examples) with task-validity guards.
- **Phase I** full UI consistency pass + closing batch: activity rail expandable
  to a full page, expandable per-event run timeline, experiment
  filter/sort, real import-review diff, persistent agent chat drawer.
- **Phase J** responsive drawers/mobile.
- **Phase K** hermetic test suite (69 tests) + two live drills (93 and 41 checks).

## 2. What was fixed (10 audit bugs + 13 found by testing)
Audit list: `leaderboard()` always-None best · `ModelType`⇄registry mismatch ·
researcher `source` vs `a/b` gap · Celery `project_id` gap · `db.get` bug ·
upload validation · dead `llmPill` · `POST /sessions` serialization ·
placeholder dataset filename · alembic stub (documented + asserted).
Found by testing: project-memory cross-project leak · project-memory lost
updates · chat/events silently dropped on SQLite lock contention · **EventBus
replay broadcasting every project's history to every client** · chat replies
tagged with the run's project instead of the chat's · interview state surviving
a project switch · numpy metrics breaking JSON/WS · confirm-gate started runs on
any text · `/automl/state` 500 during data load · import judged against the
wrong project's table · `quantile_bin` invalid SQL · target mis-detected as
`"from"` · `CentralStore()` ignoring `CENTRAL_STORE_PATH` · **hunt-phase ladder
rungs passed `learning_rate` to RandomForest, failing any run past iteration 11**
(inside the default 12-iteration budget).

## 3. Architecture (one source of truth)
```
UI (workspace.html hash routes | classic /ui)
  ↓ REST / WS /ws/automl            WS = live only; REST = durable truth
API (projects/sessions/datasets/runs/imports/automl)
  ↓ validated project_id · session_id · run_id (backend/project_context.py)
Orchestrator (singleton thread) → agents → CentralStore (DuckDB, one table)
                                       → MLflow (params/metrics/artifacts lineage)
  ↓ write-through (backend/runlog.py, best-effort, never blocks training)
SQLite: projects, sessions, datasets, experiments, models, runs, run_events,
        chat_messages, project_memory, project_imports
```
MLflow keeps lineage; `runs`/`run_events` are the operational record;
`project_memory` is the retrieval unit. No competing trackers.

## 4. New routes
`GET /workspace` (shell) · `GET /runs`, `/runs/{id}`, `/runs/{id}/diagnostics`,
`/runs/{id}/research` · `GET/POST /projects/{id}/imports`,
`GET /projects/{id}/resources`, `GET /projects/{id}/imports`,
`GET /projects/{id}/report?format=` , `GET /projects/{id}/context` ·
`GET/DELETE /sessions/{id}/messages` · `POST /datasets/{id}/stage` ·
`GET /automl/features`, `POST /automl/features/{disable,enable,remove,undo}`,
`POST /automl/target`, `POST /automl/resume`. UI routes: `#/projects/:id/{overview,chat,data,analysis,
research,features,experiments,evaluation,report,import}`,
`#/projects/:id/runs/:runId`, `#/projects/:id/compare/:ids`,
`#/projects/:id/activity`.

## 5. Database changes
Four new tables (`runs`, `run_events`, `chat_messages`, `project_memory`,
`project_imports`), created by the existing `create_all` startup path. No data
migrated or destroyed. `ModelType` gained the canonical registry names
(`gradient_boosting`, `linear_regression`, `ridge`); `decision_tree` kept and
marked legacy.

## 6. UI changes
Workspace rebuilt into nine real project pages plus run/compare views; chat kept
as the primary surface with a floating agent button on other pages; session
management added; every panel shows real data or an explicit reason it is
absent; responsive drawers for narrow screens.

## 7. Testing results
`pytest tests/ -q` → **73 passed** (hermetic: temp DB/store/MLflow, autouse
orchestrator quiescing, works with a dev server running). Live drill → **93/93**.
Spec acceptance scenario → **41/41 steps** (real CSV upload → run → stop →
refresh → reproduce → all pages → import + provenance → explained refusal).
Cross-project isolation is verified three ways: unit test, live drill, and the
project-scoped WebSocket the workspace actually opens.

## 8. Known limitations (stated in-product, not hidden)
- One run at a time process-wide (singleton); a second start returns `busy`.
- Resume restarts the *campaign* from the last checkpoint (it re-loads data and
  restarts the ladder at the next rung); it does not restore in-process state,
  and a crash mid-iteration repeats that one iteration.
- Cross-validation is not used in the AutoML track (single holdout split) —
  listed in every report's limitations.
- Import verdicts rely on recorded schemas/configs; a resource with no recorded
  provenance is reported as unknown rather than assumed compatible.
- `Session.metadata_` is still the only unwritten Track-1 field (chat moved to a
  table for scalability).
- Alembic remains a documented stub; `create_all` is the live schema path.
- No authentication/authorization (see `SECURITY_LIMITATIONS.md`).

## 9. Remaining TODOs
Cross-validation option, per-project concurrency, PDF report export, a
regression suite for the Celery track, and auth before any shared deployment.
(Resume is implemented; the remaining "TODO" from the spec is mid-iteration
crash-resume, which is not attempted because the trainer has no per-iteration
transactional checkpoint.)

## 10. How to run
```bash
cd /home/lokeshpapapa/Downloads/projects/ml_assistant
backend/.venv/bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
# workspace: http://localhost:8000/workspace   classic: /ui   API docs: /docs
backend/.venv/bin/python -m pytest tests/ -q     # 73 tests
```
Nothing cloud is required: SQLite, DuckDB, offline paper bank, local MLflow
(JSON fallback), `LLM_PROVIDER=local`.

## 11. Reproduce an experiment
Experiments page → `Reproduce` (or chat "reproduce run …"). The new run copies
the source `runs.config` and records `reproduces_run_id`; the run view and
project memory show the link. Compare both runs at
`#/projects/:id/compare/<runA>,<runB>`.

## 12. Reproduce a model / import
Model: `GET /automl/export-best-model` (button enabled once a run has a best
model), or the artifact path in `GET /runs/{id}` → `artifacts[]`.
Import: Import page → pick source project → `Inspect source resources` (review)
→ choose QUICK/SMART/REVIEW. Applied artifacts are listed with provenance
(source project, source run, mode, verdict, adapted flag).

## 13. Clean-workspace redesign pass (latest)

`backend/static/workspace.html` was rewritten as a single clean, menu-driven
workspace to fix the "full of details, not user friendly" feedback:

- **Grouped sidebar navigation** - Workspace (Overview / Chat / Activity),
  Pipeline (Data / Analysis / Research / Features / Experiments), Results
  (Evaluation / Report / Import) and a Run group (Live run / Compare).
  Every item has an icon + single label.
- **Progressive disclosure** - page content is rendered into collapsible
  `<details class="sect">` sections, so raw tables/JSON are one click away
  instead of dominating the page.
- **Compact KPI overview** - headline numbers first, drill-down below.
- **Persistent chat drawer** - the agent is reachable from any page via the
  floating `Agent` button without losing your place.
- **Responsive** - the drawer/panels collapse on small screens.

### Bugs found and fixed while validating the rewrite

The first rebuild was **non-functional**: a mismatched quote in the
`loadFeaturesPage` string concatenation (`'">Remove</button></td></tr>"`)
left the rest of the script unterminated, so the whole UI script failed to
parse and nothing on the page worked. Also fixed:

- `mountChat()` only moved the composer on the first switch (it looked inside
  `v-chat`, where the composer no longer lived), so the composer never came
  back from the drawer. The composer now has `id="chatComposer"` and moves
  with the other chat nodes.
- Sidebar project buttons called `goto("chat")`, which reused the
  already-selected project; clicking a *different* project did nothing. They
  now navigate to `#/projects/<id>/chat`.
- `buildNav()` rendered each item's label twice; it now emits
  `<span class="ic">icon</span><span>label</span>` via an `ICONS` map.
- An unknown page in the URL hash (`#/projects/:id/typo`) threw on a null
  view element; `parseHash()` now falls back to `chat`.
- Added the `profiling` event to the activity feed so that pipeline step is
  visible.

### Validation performed

- JS parses cleanly (esprima) - 1 script block, OK.
- HTML well-formed (balanced tags, no stray closes).
- Every `$("id")` reference resolves to a real element; every `LOADERS` page
  has a matching `v-*` view.
- All read endpoints the UI calls return 200 with real data against the
  `Customer Churn Demo` project (context, resources, imports, report,
  datasets, sessions, messages, runs, run detail, diagnostics, research,
  target, features).
- WebSocket `/ws/automl?project_id=` delivers a project-scoped snapshot and
  live events; a chat round-trip and a full run
  (STARTING -> DATA_FETCH -> RESEARCH -> TRAINING -> DONE) were verified live.
- Tests: `73 passed` (full suite); `41 passed` for the workspace/phases suites
  after the final edits.

### Known limitations (unchanged)

No auth, one run at a time, resume restarts from the persisted checkpoint
(not mid-iteration), single holdout split, no PDF export, Alembic is a stub.

## 14. Premium enrichment pass (2026-10-07)

Keep the dark base, add richness (user's chosen direction), still strict
UI/UX + honest data — no invented numbers, no backend behavior changes except
the trainer metrics fix below.

- **KPI cards** - `kpi(l, v, i, sub, spark)` supports an honest subtitle and an
  inline `sparkline(vals)` trend; Overview now has real Dataset/Target cards plus
  a "Metric trend" card built from actual recorded runs (oldest -> newest,
  green/red by direction).
- **Charts** - `lineChart` upgraded: area gradient, light gridlines, adaptive
  tick precision (0.90 -> 0.91 prints distinct labels), hover crosshair +
  tooltip pill (`initChartHover()` on `g.hv`), draw-in animation via
  `pathLength=1` + `stroke-dasharray:1`. Compare charts wrapped in `.grid2`,
  height 180.
- **Depth/motion/type** - card depth (inset highlight + shadow + hover lift),
  KPI radial glow + gradient numerals (`background-clip:text`,
  `font-variant-numeric: tabular-nums`), primary button gradient + glow,
  staggered `rise` animation, `:focus-visible` ring, `[hidden]{display:none!important}`.
- **Layout fixes verified programmatically** - composer pinned to chat bottom,
  380px mobile `phead` bug, `#collapseBtn` hidden <=900px, project-name flex
  growth on small widths.

### Bugs found and fixed during enrichment

- Compare page crashed with `Cannot access 'sel' before initialization`
  (shadowed global); renamed to `picked`.
- Deep-link to Compare with a single run now auto-selects a second run so the
  charts render.
- `.btn{display:inline-flex}` was overriding the `hidden` attribute, leaking the
  Agent FAB onto every page; fixed with a global `[hidden]` rule.
- Stored metrics could print a bare `0` (trainer bug, see below) next to a real
  recomputed value; the report page now shows `recomputed from saved
  predictions` notes instead of a confusing zero.

### Trainer metrics fix (backend, root cause)

`ml/models/registry.py` `compute_metrics`:
- **Binary ROC-AUC was always stored as 0.0.** The `roc_auc` metric passed the
  full `(n,2)` probability matrix to `roc_auc_score(..., multi_class="ovr")`,
  which sklearn rejects for binary -> the swallowed `except` wrote `0.0`.
  Fixed: binary uses `y_proba[:, 1]` (positive class); multiclass still uses
  `multi_class="ovr"`.
- **precision/recall were never computed**, so any campaign targeting
  `recall`/`precision` recorded `target_metric: 0.0`. Added both
  (`average="weighted"`, `zero_division=0`).
- Verified: binary `roc_auc ≈ 0.955`, multiclass `0.826`, regression metrics
  unchanged. Two regression tests added (`TestComputeMetrics`).

## 15. Deployment (LAN)

Running server: `backend/.venv/bin/python -m uvicorn backend.main:app
--host 0.0.0.0 --port 8000` (started with setsid/nohup, pid changes per start).

- **This machine**: `http://localhost:8000/workspace`
- **LAN**: `http://<host-ip>:8000/workspace` (find via
  `ip -4 addr show scope global`)
- **Security caveat**: no auth (see `SECURITY_LIMITATIONS.md`); anyone on the
  network can start real ML runs. Stop the server with
  `pkill -f "uvicorn backend.main:app"`.

## 16. Final validation (this pass)

- 12 workspace routes render with 0 JS errors / 0 failed requests.
- Interaction suite: modals/Esc, palette, composer pin, drawer open/close,
  compare charts + hover tooltip, report KPIs, settings tab persistence,
  sidebar collapse.
- `pytest tests/ -q` -> **75 passed** (73 + 2 new trainer-metric tests).
- Server restarted to load the trainer fix; in-memory leaderboard resets to
  IDLE (honest), persisted experiments are unaffected.

## 17. Dataset transparency pass (this session)

User wanted: see the dataset + its details, know exactly what data is being
used, and get every feature with its details (including when asking chat about
rows).

### Root-cause fix (backend)
- **Bug**: on process restart the working table survives in the DuckDB file,
  but `CentralStore._columns/_dtypes` stayed empty until a `load_*` call, so
  `/automl/dataset-preview` returned `columns: []` / `dtypes: {}` for a live
  381,100-row table -> the Data page and dataset detail showed nothing.
- **Fix**: `_connect()` now calls `_refresh_schema_if_exists()` (central_store.py)
  which re-reads `DESCRIBE central_dataset` when the persisted table exists.
  Verified: reopened store reports all 13 columns + dtypes.
- **Test**: `test_reopen_persisted_file_reports_schema` added (tests/test_central_store.py).

### Real per-column statistics (backend)
- New `CentralStore.column_stats()` — SQL aggregates computed from the actual
  single-source table (nulls, null %, uniques, min/mean/std/max, or top
  categorical values). No estimation.
- New `GET /automl/dataset-details?project_id=` returns preview + stats +
  provenance (source/path/imported_snapshot_of) + resolved target.

### Chat answers dataset questions
- `orchestrator.handle_message` detects question-shaped dataset/rows/features
  messages (never hijacks interview answers like "5000 rows" or action intents)
  and replies with every feature + its real type/statistics + provenance.
- The `chat_response` WS event now carries a structured `dataset` payload, and
  `workspace.chat.js` renders it as a `datasetFactsCard` (rows, per-feature
  type/stats, provenance, target Ⓣ, Inspect button).
- Verified over the live WS: "which features does the dataset have?" -> 13 real
  features with stats; Playwright confirms the card renders with 0 JS errors.

### Dataset detail page (UI)
- Overview: new "Column statistics" cards grid (real range/mean/σ/top values,
  nulls, uniques) + "Imported from run" provenance in Source.
- Columns tab now shows real per-column stats inline.
- Statistics tab: real per-column table (Range/top values, Nulls, Unique) with
  an honest fallback when no live stats exist yet.

### Validation
- New `dataset-details` endpoint: 200, 13 columns, target `target`, provenance
  `recorded`.
- Playwright: Data page lists Working table (381,100 rows); dataset detail
  shows numbers, features, provenance, target; Stats tab renders; chat card
  renders; **0 JS errors**.
- `pytest tests/ -q` -> **76 passed** (75 + new reconnect test).
