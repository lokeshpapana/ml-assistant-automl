"""Local MLflow experiment tracking for every training iteration.

Wraps MLflow's file-based tracking (no cloud) and degrades gracefully to a JSON
run store if MLflow is unavailable, so the engine never breaks offline.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from backend.config import settings

try:  # pragma: no cover - import guard
    import mlflow

    _MLFLOW_AVAILABLE = True
except Exception:  # pragma: no cover
    mlflow = None
    _MLFLOW_AVAILABLE = False


class MLflowTracker:
    def __init__(self):
        self._lock = threading.RLock()
        self._fallback_runs: list[dict[str, Any]] = []
        self._fallback_dir = Path(settings.automl_output_dir) / "fallback_runs"
        self._fallback_dir.mkdir(parents=True, exist_ok=True)
        self._experiment_id: str | None = None
        self._enabled = _MLFLOW_AVAILABLE
        self._active_run = None
        if self._enabled:
            try:
                mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
                experiment = mlflow.set_experiment(settings.mlflow_experiment_name)
                self._experiment_id = experiment.experiment_id
            except Exception:
                self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def info(self) -> dict[str, Any]:
        return {
            "enabled": self._enabled,
            "tracking_uri": settings.mlflow_tracking_uri if self._enabled else "json-fallback",
            "experiment": settings.mlflow_experiment_name if self._enabled else "fallback",
            "experiment_id": self._experiment_id,
        }

    @contextmanager
    def run(self, run_name: str, run_id: str | None = None) -> Iterator[dict[str, Any]]:
        """Open a tracking run. Yields {id, is_new} for the caller."""
        if not self._enabled:
            entry = {
                "run_id": run_id or f"fallback-{int(time.time() * 1000)}",
                "run_name": run_name,
                "params": {},
                "metrics": {},
                "tags": {},
                "artifacts": [],
                "start_time": datetime.utcnow().isoformat(),
            }
            with self._lock:
                self._active_run = entry
            try:
                yield {"run_id": entry["run_id"], "is_new": True}
            finally:
                entry["end_time"] = datetime.utcnow().isoformat()
                with self._lock:
                    self._fallback_runs.append(entry)
                    self._active_run = None
                    self._persist_fallback()
            return

        with mlflow.start_run(run_name=run_name, run_id=run_id) as active:
            with self._lock:
                self._active_run = {"run_id": active.info.run_id, "run_name": run_name}
            yield {"run_id": active.info.run_id, "is_new": bool(run_id is None)}

    # ------------------------------------------------------------ logging

    def log_params(self, params: dict[str, Any]) -> None:
        with self._lock:
            if self._enabled and self._active_run:
                try:
                    mlflow.log_params({k: self._serializable(v) for k, v in params.items()})
                except Exception:
                    pass
                return
            if self._active_run:
                self._active_run["params"].update(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        with self._lock:
            if self._enabled and self._active_run:
                try:
                    mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
                except Exception:
                    pass
                return
            if self._active_run:
                self._active_run["metrics"].update({k: float(v) for k, v in metrics.items()})

    def set_tags(self, tags: dict[str, str]) -> None:
        with self._lock:
            if self._enabled and self._active_run:
                try:
                    mlflow.set_tags(dict(tags))
                except Exception:
                    pass
                return
            if self._active_run:
                self._active_run["tags"].update(tags)

    def log_artifact(self, path: str | Path) -> None:
        p = Path(path)
        if not p.exists():
            return
        with self._lock:
            if self._enabled and self._active_run:
                try:
                    mlflow.log_artifact(str(p))
                except Exception:
                    pass
                return
            if self._active_run:
                self._active_run["artifacts"].append(str(p))
                self._persist_fallback()

    def register_model(self, model_name: str, artifact_dir: str | Path) -> None:
        """Best-effort registration into the local MLflow model registry."""
        if not self._enabled:
            return
        try:
            import mlflow.sklearn  # noqa: F401

            mlflow.sklearn.log_model(
                self._active_best_estimator if hasattr(self, "_active_best_estimator") else None,
                artifact_path="model",
                registered_model_name=model_name,
            )
        except Exception:
            pass

    # ------------------------------------------------------------ queries

    def list_runs(self, experiment_name: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if self._enabled:
                try:
                    runs = mlflow.search_runs(
                        [self._experiment_id] if self._experiment_id else None,
                        order_by=["attributes.start_time DESC"],
                        output_format="list",
                    )
                    out = []
                    for r in runs:
                        out.append(
                            {
                                "run_id": r.info.run_id,
                                "run_name": r.data.tags.get("mlflow.runName", r.info.run_id),
                                "status": r.info.status,
                                "start_time": r.info.start_time,
                                "params": dict(r.data.params),
                                "metrics": dict(r.data.metrics),
                                "tags": dict(r.data.tags),
                            }
                        )
                    return out
                except Exception:
                    pass
            return list(self._fallback_runs)

    def get_best_run(self, metric: str | None = None) -> dict[str, Any] | None:
        runs = self.list_runs()
        metric = metric or "target_metric"
        valid = [r for r in runs if metric in r.get("metrics", {})]
        if not valid:
            return None
        return max(valid, key=lambda r: r["metrics"][metric])

    def _persist_fallback(self) -> None:
        target = self._fallback_dir / "runs.json"
        payload = self._fallback_runs
        try:
            target.write_text(json.dumps(payload, indent=2, default=str))
        except Exception:
            pass

    @staticmethod
    def _serializable(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return json.dumps(list(value))
        if isinstance(value, dict):
            return json.dumps(value)
        return str(value)


_tracker_lock = threading.Lock()
_tracker: MLflowTracker | None = None


def get_tracker() -> MLflowTracker:
    global _tracker
    with _tracker_lock:
        if _tracker is None:
            _tracker = MLflowTracker()
        return _tracker
