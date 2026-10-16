# Phase H — Experiment Comparison, Clone/Reproduce, Diagnostics (Implemented)

## Data source
No competing tracker: `run_events(event_type='model_training')` holds model,
params, metrics, target metric, training time and the `mlflow_run_id` for every
iteration; MLflow remains the params/metrics/artifact lineage of record.

## Comparison (`/#/projects/:id/compare/:runIds`)
Side-by-side run / iteration / model / params / feature count / objective metric
/ run duration / training time, plus metric bars. The objective is the run's own
configured metric — the best model is **derived**, never hard-coded, and the
selected objective is printed on the page. Conflicting objectives (e.g. F1 up,
latency up) are shown side by side rather than hidden.

## Clone / Reproduce
`Reproduce` re-starts a run with the source run's stored config plus
`reproduces_run_id`, so the new run references the original in
`runs.config.reproduces_run_id`, in the run view, and in project memory
decisions. Data provenance (dataset source/target/row cap), preprocessing
snapshot and features come from the same config, so the only difference is
model search order — which is stated in the run config.

## Diagnostics (`GET /runs/{run_id}/diagnostics`)
Recomputed from the stored `best_model.pkl` bundle + the per-iteration dataset
snapshot CSV (no trust in stored metrics):
- classification: accuracy, weighted precision/recall/F1, confusion matrix,
  classification report, ROC + PR curves with AUC (binary + probabilities only),
  prediction distribution, worst error examples with feature values;
- regression: MAE, MSE, RMSE, R², residual stats, predicted-vs-actual series,
  decile means; ROC/PR/confusion explicitly omitted with a reason;
- permutation importance (300-row sample, 3 repeats) on top of the model's own
  `feature_importance`;
- a `glossary` explaining every metric, rendered next to its value.
