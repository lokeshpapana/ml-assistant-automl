# Phase E — Real Project Pages (Implemented)

Every page renders stored data. No page fabricates a value; missing data is
stated as such.

## Endpoints added
| Page | Source | Endpoint |
|---|---|---|
| Data | project datasets + live table + run record | `GET /datasets?project_id`, `GET /automl/features`, `POST /automl/target`, `POST /datasets/{id}/stage` |
| Analysis | snapshot preview + recomputed metrics + importances | `GET /automl/dataset-preview`, `GET /runs/{id}/diagnostics` |
| Research | run research findings with provenance | `GET /runs/{id}/research` |
| Features | live table + persisted per-run pipeline + journal | `GET /automl/features`, `/disable`, `/enable`, `/remove`, `/undo` |
| Experiments | runs + persisted `model_training` events + MLflow ids | `GET /runs?project_id`, `GET /runs/{id}` |
| Evaluation | recomputed from stored `.pkl` + snapshot CSV | `GET /runs/{id}/diagnostics` |
| Report | assembled from runs/memory/events | `GET /projects/{id}/report?format=json\|markdown\|html` |

## Honesty rules enforced in code
- Research cards are labelled `OFFLINE KNOWLEDGE` / `LIVE RESEARCH` /
  `HEURISTIC` by checking the curated bank — an offline result is never
  presented as live.
- Diagnostics **recompute** metrics from the stored model + dataset snapshot;
  stored metrics are echoed separately as `stored_metrics`. ROC/PR/confusion
  are omitted for regression (and for multiclass without probabilities) with an
  explicit `omitted` reason instead of drawing a misleading plot.
- Feature operations are journalled in `CentralStore.op_log` and every engine
  change is written to project memory → reproducible.
- Target column cannot be removed/disabled; removal attempts return a reason.

## Bug fixes landed here
- `quantile_bin` never worked: window function in `UPDATE ... SET` is invalid
  SQL (`NTILE(4) OVER (...)` → BinderException). Now `UPDATE ... FROM` a
  windowed projection matched on `rowid`, with a pandas fallback.
- `leaderboard()` returned `best=None` forever (read `self.best`, which is never
  set) — now reads `trainer.best`, same source as `snapshot()`.
- numpy scalars in metrics broke JSON serialization of `/automl/state` and every
  WS payload (`PydanticSerializationError: numpy.float32`); added
  `json_safe()` at the emit/snapshot boundary.
- `snapshot()` 500'd while the data stage dropped/recreated the central table;
  now returns an explicit `unavailable` preview.
