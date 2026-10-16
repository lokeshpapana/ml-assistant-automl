# Phase D — Persisted Run Event Log (Implemented)

Every run is now replayable from the database alone.

## Schema
- `runs(run_id PK, project_id?, session_id?, status, stage, progress, eta_seconds,
  config JSON, dataset JSON, best JSON, feature_history JSON, research JSON,
  error, started_at, completed_at, created_at, updated_at)`
- `run_events(id, run_id FK CASCADE, ts, event_type, status, message, progress,
  stage, duration_ms, metadata JSON)` — append-only, indexed `(run_id, ts)`.

## Canonical event types (exactly the specified set, plus what the code emits)
`run_started, data_loading, data_validation, profiling, preprocessing*,
feature_engineering, research_started, research_completed, model_training,
model_evaluation*, model_checkpoint*, model_selected, report_generated,
run_completed, run_failed, run_stopped` — `*` = reserved, not emitted today
(no separate preprocess/checkpoint step exists in this engine; `model_training`
carries the preprocess snapshot, and each iteration is itself a checkpointed
artifact). Nothing is emitted that does not happen.

## Progress and ETA
- Stage-based: IDLE 0 · STARTING 2 · DATA_FETCH 20 · RESEARCH 40 · TRAINING 60
  (interpolated by `iteration / max_iterations`) · terminal 100.
- `profiling` → 30, `data_validation` → 25, `feature_engineering` → 50,
  `model_selected` → 100. Monotonic per run (`_max_progress` clamp) so a
  re-emitted preview cannot make the log go backwards.
- ETA is computed ONLY from observed per-iteration training times and remaining
  iterations; otherwise `null` (never a guess).

## APIs
- `GET /runs` (filter by project/status, paginated), `GET /runs/{run_id}` (run +
  events + artifacts + `mlflow_run_ids` + `stages_seen`), `GET /runs/{run_id}/research`.
- WS: unchanged; now every event also carries `project_id/session_id/run_id`.

## Files
`backend/runlog.py` (new), `backend/database/models.py` (2 new tables),
`backend/orchestrator.py` (`_persist_event`, `_set_state` write-through,
`_progress_for`, `_eta_seconds`, `_safe_preview`), `backend/api/runs.py` (new),
`backend/agents/trainer.py` (`training_time_seconds`, `dataset_snapshot`,
`mlflow_run_id` in the record/emit).

## Tests
`test_run_lifecycle_persists_events_and_record`, `test_progress_is_stage_based_not_fake`
(monotonic + terminal 100), `test_run_id_validation` (400 malformed / 404 unknown),
plus drill checks for all 8 event types and MLflow lineage.

## Resume from checkpoint (added after Phase K review)
- `POST /automl/resume {project_id, run_id?, session_id?}` — validates the
  project, then `AutoMLOrchestrator.resume_from()` derives the next ladder rung
  from the source run's persisted `model_training` event count, copies the stored
  config, and starts a new campaign recording `resumes_run_id` + `resume_index`.
- `TrainerAgent.model_for(iteration)` = `ladder[(start_index + iteration) % n]`,
  so a resumed campaign does not retrain models the source already tried.
- Refusals are explicit: unknown run (404), run of another project (400), invalid
  project (400), finished run with no progress, and missing config.
- Surfaces: run-view "Resume from checkpoint" button, chat "resume the run" /
  "Resume" quick reply, and the activity rail. Chat resume is orchestrator-side
  and re-verifies the run's project, so it cannot be used to cross projects.
- Verified live: a 40-iteration run stopped, resumed at index 40, lineage stored,
  next model differs from the last one used.
