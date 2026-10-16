# Phase G — Import From Project (Implemented)

## Resources
`dataset, dataset_snapshot, feature_pipeline, model_artifact, experiment_config,
research` — assembled from real rows/artifacts via
`GET /projects/{id}/resources` (datasets, runs, memory, prior imports).

## Modes
- **QUICK** — applies only `compatible` resources; anything else is refused with
  the reason appended.
- **SMART** — also applies `partially_compatible` with an explicit adaptation:
  feature specs are trimmed to existing columns (or *staged* when the target has
  no working table yet), a model artifact with an unconfirmed task is copied and
  marked staged (never activated).
- **REVIEW** — dry run: verdicts + reasons + a diff of what would be applied,
  and nothing is written.

## Compatibility verdicts
`compatible` / `partially_compatible` / `incompatible`, each with concrete
reasons: missing columns (listed), task-type mismatch, absent artifact, unknown
resource type, deleted source project, self-import (400), unknown run (404),
run belonging to another project (400).

**Cross-project bug found and fixed**: the check compared incoming features
against `CentralStore.columns()`, but the store is a global singleton — it still
held the *source* project's table, so a fresh target was judged "compatible"
against someone else's columns. Now the store is only trusted when the live run
actually belongs to the target project (`test_import_cross_project_compatibility_uses_own_context`).

## Provenance
Every applied import writes a `project_imports` row:
`target_project_id, source_project_id, source_run_id, source_experiment_id,
resource_type, resource_ref, mode, verdict, verdict_reasons, adapted,
created_at`. Source ids are plain values (not FKs) so provenance survives source
deletion. Listed at `GET /projects/{id}/imports` and rendered on the Import page.

Imports are also *chat-addressable*: "import the preprocessing from Project A"
is resolvable through the same `/projects/{id}/imports` endpoint (the agent
parses the intent; the endpoint does the verification).
