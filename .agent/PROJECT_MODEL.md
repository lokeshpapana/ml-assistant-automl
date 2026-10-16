# Project Model — How a Project Is Currently Represented

All paths repo-relative. Sources: `backend/database/models.py` (verified by direct
read), `backend/api/schemas.py`, `backend/api/projects.py`, `backend/agents/common.py`,
`backend/orchestrator.py`.

## 1. Track 1 — persistent relational project (the only durable representation)

```
Project (projects) 1──┐
  id: Uuid PK (uuid4)  │
  name: String(255)    ├── Session (sessions) ── metadata_: JSON (free-form, unused by agents)
  description: Text?   │     project_id FK CASCADE, name, description, created/updated
  status: ACTIVE|ARCHIVED|DELETED
  created_at / updated_at (tz, server_default)
                       ├── Dataset (datasets) ── project_id FK CASCADE, session_id? FK SET NULL
                       │     name, original_filename, file_path (./uploads/...), file_size_bytes,
                       │     mime_type, status UPLOADING|READY|PROCESSING|FAILED,
                       │     rows?, columns?, column_info? JSON, profile? JSON
                       └── Experiment (experiments) ── project_id FK, session_id?, dataset_id FK
                             name, description?, status PENDING|RUNNING|COMPLETED|FAILED|CANCELLED,
                             config JSON, metrics? JSON, started/completed_at?, error_message?
                             └── Model (models) ── experiment_id FK
                                   name, model_type (7-value enum), hyperparameters JSON,
                                   metrics JSON, training_time_seconds?, model_path?,
                                   feature_importance? JSON, created_at (no updated_at)
```

- API shapes (`backend/api/schemas.py:12-175`): `ProjectCreate{name*,description?}`,
  `ProjectResponse{id,status,created_at,updated_at+base}`, `SessionCreate/Response`,
  `DatasetResponse{…rows?,columns?,column_info?,profile?}`, `ExperimentCreate{name*,
  description?,config={},dataset_id*}` + `ExperimentResponse{…metrics?…}`,
  `ModelCreate{name*,model_type,hyperparameters}` + `ModelResponse{…}`.
- Lifecycle (`backend/api/projects.py:18-99`): create → list (paginated, filterable by
  status) → get → patch → soft-delete (`DELETED`). Delete cascades to sessions,
  datasets, experiments, models (ORM `delete-orphan` + FK `CASCADE`).
- What a project does **NOT** contain today: no link to AutoML runs, no chat history,
  no requirements/goals, no research records, no feature records, no reports, no
  memory, no tasks, no checkpoints, no provenance, no autonomy/mode setting, no
  imported-resource records. `Session.metadata_` JSON exists and is empty by
  convention — the natural (if interim) home for agent state until a dedicated table
  is added.

## 2. Track 2 — ephemeral run representation (what `/ui` actually drives)

No project. A run is a bundle of process-local objects, all reset on `start()`:

```
AutoMLOrchestrator (singleton, backend/orchestrator.py:39-64)
  state, started_at, finished_at, error, upload_path
  requirements: Requirements (agents/common.py:87-102)
    goal, target_column?, task_type, metric_expr, metric, target_value, direction,
    max_rows, dataset_source, dataset_path, dataset_query, confirm,
    max_iterations, feature_engineering_every_n, meta{} (explicit-flags live here)
  iterations: [ {iteration, model, hyperparameters, metrics{}, target_metric,
                 training_time_seconds, n_features, feature_importance?,
                 run_id, model_path, target_met} ]          (trainer.py:167-179)
  feature_history: [ {name, kind, a?, b?, paper_id?, paper_title?, technique?} ]
  trainer.best: best record (same shape) + best_cache {model, engineer, feature_names}
  store: CentralStore → ONE table "central_dataset" (DuckDB file or SQLite)
  tracker: MLflowTracker → MLflow file store or fallback_runs/runs.json
  bus history: last ≤2000 events (volatile)
```

- Serialized view = `snapshot()` (`orchestrator.py:311-326`): state, requirements,
  timestamps, error, iterations, feature_history, best, dataset preview(10),
  tracker info, mlflow runs. This dict is the **only** "project state" the UI knows.
- On-disk residue per run (write-only, never read back): `automl_outputs/models/
  iter_{n}_{model}.pkl`, `iter_{n}_dataset_snapshot.csv`, `best_model.pkl`,
  `central_store.duckdb`, `fallback_runs/runs.json`, `./mlruns/*` (when enabled).

## 3. Identity and linkage gaps (why Phases 3–4 need backend work)

| Needed (spec)              | Today                                              | Gap                                   |
|----------------------------|----------------------------------------------------|---------------------------------------|
| Project has a dataset      | `Dataset.project_id` exists; AutoML ignores it     | DataFetcher never reads/writes rows   |
| Project has experiments    | `Experiment` rows exist; AutoML writes none        | iterations live only in memory/MLflow |
| Project has models         | `Model` rows exist; AutoML writes none             | `.pkl` files unregistered             |
| Run identity (`runs/:runId`)| No run id; one global run at a time               | singleton orchestrator                |
| Chat history               | `EventBus` 2000-event volatile buffer              | never persisted                       |
| Requirements/goals         | `Requirements` in-memory; `meta` flags             | never persisted                       |
| Research records           | `feature_history` in-memory                        | never persisted                       |
| Reports                    | dicts only, never stored                           | no table/endpoint                     |
| Memory/context             | `Requirements.meta` + defaults                     | volatile; full resend n/a (rule-based)|
| Import/provenance          | nothing                                            | no tables/endpoints                   |
| Autonomy mode              | hardcoded confirm-gate                             | no setting anywhere                   |

## 4. Minimal durable project representation (recommendation, not yet implemented)

Reuse existing tables; add linkage, not a parallel universe:

- `Session.metadata_` (already JSON): store `{automl: {requirements, stage,
  autonomy_mode, run_ids, memory_summary, decisions[]}}` — zero-migration first step.
- New nullable FKs/columns (one small migration or `create_all` in dev):
  `Experiment.orchestrator_run_id` (string, groups iterations of one campaign),
  `Model.artifact_path` reuse of `model_path` (already exists — point it at the
  `.pkl`), `Dataset.central_table` marker is unnecessary (single table).
- New tables only when Phases G/H demand them: `project_memory` (scoped key facts),
  `project_imports` (provenance: source project/resource, compatibility verdict,
  diff, importer, timestamp), `run_events` (append-only operational timeline backing
  `/runs/:runId`), `tasks` ( TASK-xxx with subtasks/status).
- Rule: Track 2 remains the execution engine; Track 1 becomes the system of record.
  The orchestrator writes *summaries* (records, events, artifacts paths) to the DB;
  bulk data (datasets, matrices, pickles) stays in files with paths in rows.
