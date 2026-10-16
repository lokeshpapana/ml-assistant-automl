# Phase F — Project Memory (Implemented)

## Storage
One `project_memory` row per project (`project_id PK`, `data JSON`, `updated_at`)
— the same SQLite database, no second store.

## Sections
`goal, dataset, target, task_type, constraints, preferences, decisions, rejected,
feature_decisions, model_decisions, research, best, warnings, excluded_features,
pending_features, pending_model, pending_config, chat_digest`.

Written at stage boundaries (interview confirm, data ready, research applied,
each best-model change, finalize, failure) by the orchestrator, not by the UI.

## Compact context endpoint
`GET /projects/{id}/context` returns `PROJECT_CONTEXT.json`:
`project_id, goal, task, dataset, target, constraints, decisions[-10],
experiments[-10], best_model, research[-10], warnings, current_run,
pending_features, excluded_features, pending_model`.
No chat history, no event log, no dataset rows — verified < 6 KB in the drill.

## Two silent-data-loss bugs found and fixed here
1. **Cross-project leak**: `dict(_MEMORY_DEFAULT)` was a *shallow* copy, so the
   list sections were shared with the module template. Writing project A's
   decisions made them visible for project B. Fixed with `copy.deepcopy` +
   `_memory_base()`.
2. **Lost updates**: the nested list was mutated in place while still
   referenced by the ORM instance, so SQLAlchemy's JSON change detection saw no
   delta and **never issued the UPDATE** — every memory write after the first was
   silently dropped. Fixed by deep-copying on read
   (`test_memory_appends_persist`, `test_memory_is_isolated_per_project`).

## Token discipline
Chat turns are stored once and replayed locally; the context document is
assembled per request; compaction digests history past 300 turns. No code path
resends the full project state.
