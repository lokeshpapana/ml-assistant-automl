"""Interactive AutoML Multi-Agent Orchestrator.

Coordinates the four agents, maintains a thread-safe state machine, and listens
for real-time human interrupt commands via a ``threading.Event``. The training
loop checks ``stop_event.is_set()`` before every new iteration and before any
new paper-research call, then pauses gracefully and keeps the best model + MLflow
metrics intact.
"""

from __future__ import annotations

import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from backend.agents import (
    DataFetcherAgent,
    InterviewerAgent,
    ResearchAgent,
    TrainerAgent,
)
from backend.agents.common import Requirements, json_safe, met_target, parse_target_spec
from backend.central_store import get_central_store
from backend.config import settings
from backend.mlflow_tracker import get_tracker
from backend.ws_bus import get_event_bus
from backend import runlog

IDLE = "IDLE"
INTERVIEW = "INTERVIEW"
DATA_FETCH = "DATA_FETCH"
RESEARCH = "RESEARCH"
TRAINING = "TRAINING"
PAUSED = "PAUSED"
DONE = "DONE"
FAILED = "FAILED"

# Canonical persisted event types. Anything not listed here is operational noise
# (progress chatter, previews already covered) and stays in the WS stream only.
_STATUS_EVENTS = {
    "STARTING": "run_started",
    "DATA_FETCH": "data_loading",
    "RESEARCH": "research_started",
    "PAUSED": "run_stopped",
    "DONE": "run_completed",
    "FAILED": "run_failed",
}
_STATUS_EVENT = {
    "STARTING": "run_started",
    "DATA_FETCH": "data_loading",
    "RESEARCH": "research_started",
    "PAUSED": "run_stopped",
    "DONE": "run_completed",
    "FAILED": "run_failed",
}
_STAGE_PROGRESS = {
    "IDLE": 0.0, "STARTING": 2.0, "DATA_FETCH": 20.0, "RESEARCH": 40.0,
    "TRAINING": 60.0, "PAUSED": 100.0, "DONE": 100.0, "FAILED": 100.0,
}
_CHAT_CONFIRM = {"yes", "yeah", "yep", "confirm", "confirmed", "do it", "go ahead"}

# Keys that constitute a real run configuration. A start payload carrying ONLY
# context keys (e.g. a chat-driven "yes" with project_id) must preserve the
# live interview instead of resetting to empty defaults.
_RUN_CONFIG_KEYS = {
    "goal", "metric", "metric_expr", "dataset_source", "dataset_path",
    "dataset_query", "max_rows", "target_column", "max_iterations",
    "feature_engineering_every_n", "chat_mode",
}


def _safe(obj: Any) -> Any:
    """Best-effort JSON-safe coercion for persisted records (numpy-safe)."""
    try:
        from backend.agents.common import json_ready

        return json_ready(obj)
    except Exception:
        return {"unserializable": str(type(obj))}


class AutoMLOrchestrator:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.stop_event = threading.Event()
        self.bus = get_event_bus()
        self.tracker = get_tracker()
        self.store = get_central_store()

        self.interviewer = InterviewerAgent()
        self.data_fetcher = DataFetcherAgent()
        self.researcher = ResearchAgent()

        self.state: str = IDLE
        self.requirements: Requirements = Requirements()
        self.iterations: list[dict[str, Any]] = []
        self.feature_history: list[dict[str, Any]] = []
        self.best: dict[str, Any] | None = None
        self.trainer: TrainerAgent | None = None
        self.error: str | None = None
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self.upload_path: str | None = None
        # Phase B run context: lightweight association labels (NOT persistence,
        # NOT resume). project_id/session_id arrive via the start config after
        # entry-point validation; run_id is generated per start.
        self.project_id: str | None = None
        self.session_id: str | None = None
        self.run_id: str | None = None
        self._research_completed_logged: bool = False
        self._max_progress: float = 0.0
        self._resume_index: int = 0
        self._interview_project_id: str | None = None
        self._chat_pid: str | None = None
        self._chat_sid: str | None = None
        self._worker: threading.Thread | None = None

        self.workdir = Path(settings.automl_output_dir)
        self.workdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ control

    def start(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Begin a campaign. Config is optional; falls back to interview defaults."""
        with self._lock:
            if self.state in (TRAINING, DATA_FETCH, RESEARCH):
                return {"status": "busy", "state": self.state}

            incoming = config or {}
            # Chat-driven starts ("yes") carry no run keys: keep the confirmed
            # interview instead of resetting to empty defaults (which used to
            # silently discard everything the user just told the agent).
            keep_interview = (
                self.interviewer.is_confirmed()
                and bool(self.interviewer.requirements.goal)
                and not any(k in incoming for k in _RUN_CONFIG_KEYS)
            )
            req = self.interviewer.requirements if keep_interview else Requirements()
            self.interviewer.reset(req)
            if incoming and not keep_interview:
                confirm_feedback = self.interviewer.ingest_form(incoming)
                if isinstance(config, dict) and config.get("chat_mode"):
                    pass
            else:
                confirm_feedback = self.interviewer.confirm_summary()

            self.requirements = self.interviewer.requirements
            self.requirements.confirm = True
            # Chat action and direct config both express a starting-model hint.
            cfg_hint = (incoming or {}).get("model_hint")
            if cfg_hint:
                self.requirements.meta["model_hint"] = cfg_hint
            if self.project_id:
                # Re-apply persisted soft-exclusions (Features page / chat).
                mem = runlog.get_memory(self.project_id)
                excl = mem.get("excluded_features") or []
                if excl:
                    self.requirements.meta["excluded_features"] = list(excl)

            self.stop_event.clear()
            self.iterations = []
            self.feature_history = []
            self.best = None
            self.error = None
            self.finished_at = None
            self.started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
            # Run context for this campaign (validated at the REST/WS entry
            # points; stored here so every event and snapshot carries it).
            cfg = config or {}
            self.project_id = str(cfg.get("project_id")) if cfg.get("project_id") else None
            self.session_id = str(cfg.get("session_id")) if cfg.get("session_id") else None
            self.run_id = f"run-{uuid.uuid4().hex[:12]}"
            self._research_completed_logged = False
            self._max_progress = 0.0
            # 0 for a fresh campaign; >0 continues a resumed run's ladder.
            self._resume_index = max(0, int(cfg.get("resume_index") or 0))

            # Persist the run record + seed memory (best-effort, never blocks start).
            runlog.save_run(
                self.run_id,
                self.project_id,
                self.session_id,
                {
                    "goal": self.requirements.goal,
                    "metric": self.requirements.metric,
                    "target_value": self.requirements.target_value,
                    "direction": self.requirements.direction,
                    "dataset_source": self.requirements.dataset_source,
                    "dataset_query": self.requirements.dataset_query,
                    "max_rows": self.requirements.max_rows,
                    "max_iterations": self.requirements.max_iterations,
                    "task_type": self.requirements.task_type,
                    "target_column": self.requirements.target_column,
                    "model_hint": (self.requirements.meta or {}).get("model_hint"),
                    "reproduces_run_id": cfg.get("reproduces_run_id"),
                    "resumes_run_id": cfg.get("resumes_run_id"),
                    "resume_index": cfg.get("resume_index"),
                },
            )
            if self.project_id:
                runlog.remember(self.project_id, "goal", self.requirements.goal)
                if cfg.get("reproduces_run_id"):
                    runlog.remember(
                        self.project_id,
                        "decisions",
                        f"Run {self.run_id} reproduces run {cfg.get('reproduces_run_id')}.",
                    )
                runlog.remember(
                    self.project_id,
                    "decisions",
                    f"Run {self.run_id} started: target {self.requirements.metric} "
                    f"{'>=' if self.requirements.direction == 'maximize' else '<='} "
                    f"{self.requirements.target_value} from {self.requirements.dataset_source}.",
                )

            if not self.requirements.dataset_path and self.upload_path:
                if self.requirements.dataset_source in ("upload", "local", ""):
                    self.requirements.dataset_source = "upload"
                    self.requirements.dataset_path = self.upload_path

            self._emit({"type": "status", "state": "STARTING", "message": "Orchestrator starting…"})
            self._emit(
                {
                    "type": "agent_message",
                    "agent": "interviewer",
                    "text": confirm_feedback,
                }
            )

            self._worker = threading.Thread(target=self._run_loop, daemon=True, name="automl-orchestrator")
            self._worker.start()
            return {"status": "started", "state": self.state}

    def resume_from(
        self, run_id: str | None = None, project_id: str | None = None, session_id: str | None = None
    ) -> dict[str, Any]:
        """Continue a stopped run from its latest checkpoint.

        Checkpoint = the persisted event log: the next ladder rung is derived
        from how many ``model_training`` events the source run already recorded,
        so resumed work continues instead of repeating (or restarting from
        scratch). The new run records ``resumes_run_id`` for lineage.
        """
        if self.state in (TRAINING, DATA_FETCH, RESEARCH):
            return {"status": "busy", "state": self.state}
        if not run_id:
            for status in ("paused", "failed", "running", "done"):
                runs = runlog.list_runs(project_id=project_id, status=status, limit=1)
                if runs:
                    run_id = runs[0]["run_id"]
                    break
        if not run_id:
            return {"status": "no_resumable_run", "detail": "No previous run recorded for this project."}
        source = runlog.get_run(run_id)
        if source is None:
            return {"status": "not_found", "detail": f"Run {run_id} is not stored."}
        if project_id and (source.get("project_id") or "") != project_id:
            return {"status": "wrong_project", "detail": "That run belongs to a different project."}
        done = sum(1 for e in runlog.list_events(run_id) if e.get("event_type") == "model_training")
        if done <= 0 and source.get("status") in ("done",):
            return {"status": "already_finished", "detail": "That run already reached DONE."}
        cfg = dict(source.get("config") or {})
        if not cfg:
            return {"status": "no_config", "detail": "The stored run has no config to resume from."}
        cfg.pop("resumes_run_id", None)
        cfg.pop("reproduces_run_id", None)
        cfg["resumes_run_id"] = run_id
        cfg["resume_index"] = done
        if project_id:
            cfg["project_id"] = project_id
        if session_id:
            cfg["session_id"] = session_id
        # A resume is a new campaign that continues the ladder; start() records
        # the lineage and owns the requirements from here on.
        result = self.start(cfg)
        if result.get("status") == "started" and project_id:
            runlog.remember(
                project_id, "decisions",
                f"Run {self.run_id} resumes {run_id} from checkpoint (iteration {done}).",
            )
        return {**result, "resumed_from": run_id, "resume_index": done}

    def request_stop(self, reason: str = "user-requested") -> None:
        self._emit(
            {
                "type": "status",
                "state": "STOP_SIGNAL",
                "message": "Interrupt received. Pausing after the current step.",
            }
        )
        self._emit(
            {
                "type": "agent_message",
                "agent": "orchestrator",
                "text": f"STOP signal received ({reason}). Finishing the current step safely.",
            }
        )

    # ------------------------------------------------------------------ worker

    def _run_loop(self) -> None:
        try:
            self._set_state(DATA_FETCH)
            summary = self.data_fetcher.fetch(self.requirements, self._emit)

            dataset_info = {
                "source": summary.get("source"),
                "query": self.requirements.dataset_query,
                "path": self.requirements.dataset_path,
                "rows": summary.get("rows"),
                "columns": summary.get("columns"),
                "target_column": summary.get("target_column"),
                "task_type": summary.get("task_type"),
            }
            self._emit({"type": "profiling", "summary": _safe(dataset_info)})
            runlog.update_run(self.run_id, dataset=_safe(dataset_info))
            if self.project_id:
                runlog.remember(self.project_id, "dataset", dataset_info)
                runlog.remember(self.project_id, "target", summary.get("target_column"))
                runlog.remember(self.project_id, "task_type", summary.get("task_type"))

            req = self.requirements
            target_spec = parse_target_spec(
                {"metric": req.metric, "value": req.target_value, "direction": req.direction}
            )
            self.trainer = TrainerAgent(
                task_type=req.task_type,
                test_size=settings.automl_test_size,
                target_column=req.target_column,
                model_hint=(req.meta or {}).get("model_hint"),
                # Resume continues the ladder where the source run stopped.
                start_index=self._resume_index,
            )
            self._emit(
                {
                    "type": "agent_message",
                    "agent": "orchestrator",
                    "text": (
                        f"Target configured: {target_spec.metric} "
                        f"{target_spec.op} {target_spec.value} ({target_spec.direction})."
                    ),
                }
            )

            fe_every_n = max(1, req.feature_engineering_every_n)
            iteration = 0
            fe_done = False

            while True:
                if self.stop_event.is_set():
                    break

                # Research & feature-engineering stage on configured cadence.
                if not fe_done and (iteration % fe_every_n == 0):
                    self._set_state(RESEARCH)
                    before_cols = self.store.shape()[1]
                    applied = self.researcher.run(self.store, req, self._emit, max_features=6)
                    self.feature_history.extend(applied)
                    if self.project_id and applied:
                        runlog.remember(
                            self.project_id,
                            "research",
                            [
                                {
                                    "paper_id": f.get("paper_id"),
                                    "title": f.get("paper_title"),
                                    "technique": f.get("technique"),
                                    "feature": f.get("name"),
                                }
                                for f in applied
                            ],
                        )
                    if applied:
                        fe_done = True
                        self._emit(
                            {
                                "type": "dataset_preview",
                                "event": "dataset_preview",
                                "data": self.store.preview(limit=25),
                            }
                        )
                        self._emit(
                            {
                                "type": "agent_message",
                                "agent": "orchestrator",
                                "text": (
                                    f"Feature engineering added {len(applied)} column(s) in place "
                                    f"({before_cols} -> {self.store.shape()[1]} columns)."
                                ),
                            }
                        )

                if self.stop_event.is_set():
                    break

                self._set_state(TRAINING)
                if iteration >= req.max_iterations:
                    self._emit(
                        {
                            "type": "agent_message",
                            "agent": "orchestrator",
                            "text": f"Reached max iterations ({req.max_iterations}).",
                        }
                    )
                    break

                record = self.trainer.train_iteration(
                    self.store,
                    target_spec,
                    iteration,
                    self._emit,
                    time_budget=settings.automl_model_time_budget,
                    excluded_features=(req.meta or {}).get("excluded_features"),
                    feature_provenance=self.feature_history,
                )
                self.iterations.append(record)
                if record["target_met"]:
                    self._set_state(DONE)
                    self._emit(
                        {
                            "type": "agent_message",
                            "agent": "orchestrator",
                            "text": (
                                f"Target reached: {target_spec.metric}={record['target_metric']:.4f} "
                                f"({target_spec.op} {target_spec.value})."
                            ),
                        }
                    )
                    break
                iteration += 1

            if self.stop_event.is_set():
                self._set_state(PAUSED)
            elif self.state == TRAINING:
                # Exceeded iterations without meeting target.
                self._set_state(DONE)
                self._emit(
                    {
                        "type": "agent_message",
                        "agent": "orchestrator",
                        "text": "Training completed; target value was not reached within iteration budget.",
                    }
                )

            self._finalize()

        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            self._set_state(FAILED)
            self._emit(
                {
                    "type": "agent_message",
                    "agent": "orchestrator",
                    "text": f"Run failed: {self.error}",
                }
            )
            from datetime import datetime, timezone

            runlog.update_run(
                self.run_id, status="failed", error=self.error,
                completed_at=datetime.now(timezone.utc),
            )
            if self.project_id:
                runlog.remember(self.project_id, "warnings", f"Run {self.run_id} failed: {self.error}")

        finally:
            self.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
            self._emit({"type": "status", "state": self.state, "message": f"Orchestrator final state: {self.state}"})

    def _finalize(self) -> None:
        from datetime import datetime, timezone

        best_path = self.trainer.export_best_model() if self.trainer else None
        if best_path:
            self._emit({"type": "best_model_ready", "path": best_path})
        self._emit(
            {
                "type": "status",
                "state": self.state,
                "message": "Best model preserved; metrics remain in MLflow.",
            }
        )
        best = _safe(self.trainer.best) if self.trainer and self.trainer.best else None
        runlog.update_run(
            self.run_id,
            status="paused" if self.stop_event.is_set() else "done",
            stage=self.state,
            progress=100.0,
            eta_seconds=None,
            best=best,
            feature_history=_safe(self.feature_history),
            research=_safe(
                [
                    {
                        "paper_id": f.get("paper_id"),
                        "title": f.get("paper_title"),
                        "technique": f.get("technique"),
                        "feature": f.get("name"),
                    }
                    for f in self.feature_history
                ]
            ),
            completed_at=datetime.now(timezone.utc),
        )
        if self.project_id and best:
            runlog.remember(self.project_id, "best", best)
            runlog.remember(
                self.project_id,
                "decisions",
                f"Run {self.run_id} best: {best.get('model')} "
                f"(iter #{best.get('iteration')}, target={best.get('target_metric')}).",
            )

    # ------------------------------------------------------------------ messages

    def handle_message(self, message: dict[str, Any]) -> dict[str, Any]:
        event = message.get("event") or message.get("type")

        if event == "stop":
            self.request_stop()
            return {"status": "stop_signal_sent", "state": self.state}

        if event == "start":
            config = message.get("config") or message.get("data") or {}
            if isinstance(config, dict):
                # Top-level run context (validated at the entry point) flows
                # into the config so start() picks it up uniformly.
                config = dict(config)
                for key in ("project_id", "session_id"):
                    if message.get(key) is not None and config.get(key) is None:
                        config[key] = message[key]
            return self.start(config)

        if event == "chat":
            text = message.get("message") or message.get("content") or ""
            if not text:
                return {"status": "ok"}
            pid = message.get("project_id") or self.project_id
            sid = message.get("session_id") or self.session_id
            self._chat_pid, self._chat_sid = pid, sid
            # A project switch must not inherit the previous project's interview
            # state (pending goal/metric/"explicit" flags), otherwise a new
            # project silently starts from the last one's requirements.
            if pid and pid != getattr(self, "_interview_project_id", None):
                if getattr(self, "_interview_project_id", None) is not None:
                    self.interviewer.reset()
                    self.interviewer.memory = runlog.get_memory(pid) if pid else {}
                self._interview_project_id = pid
            runlog.save_chat(pid, sid, "user", text)
            if self.state in (TRAINING, RESEARCH, DATA_FETCH):
                if text.strip().lower() in {"stop", "stop training", "halt", "pause"}:
                    self.request_stop("chat-command")
                    return {"status": "stop_signal_sent"}
                self._agent_reply(
                    "I'm in the middle of training — use the STOP button to pause, or wait.",
                    pid, sid,
                )
                return {"status": "ok"}
            lowered = text.strip().lower()
            # 0.2 Informational dataset questions (what data / rows / features)
            # get a real answer from the live working table instead of being
            # forced into the interview wizard. Only question-shaped messages
            # about the data qualify, so "5000 rows" (a row-cap answer) still
            # flows to the normal interview.
            if self._is_dataset_question(lowered):
                summary, payload = self._dataset_fact_sheet(pid)
                self._agent_reply(summary, pid, sid, dataset=payload)
                return {"status": "ok"}
            # 0. Resume/continue from the last checkpoint (chat-addressable).
            if re.search(r"\b(resume|continue)\b", lowered) and (
                "run" in lowered or "training" in lowered or "paused" in lowered or lowered.strip() in ("resume", "continue")
            ):
                result = self.resume_from(project_id=pid, session_id=sid)
                if result.get("status") == "started":
                    self._agent_reply(
                        f"Resuming {result['resumed_from']} from checkpoint "
                        f"(continuing at iteration {result['resume_index'] + 1}).",
                        pid, sid,
                    )
                else:
                    self._agent_reply(
                        f"I can't resume: {result.get('detail') or result.get('status')}.", pid, sid
                    )
                return {"status": "ok"}
            # 1. Pending destructive-action confirmation wins over everything else
            # (including the interview's own start-yes), so "yes" can never both
            # confirm an action and launch a run.
            if self.interviewer.pending_action and lowered in _CHAT_CONFIRM:
                action = self.interviewer.pending_action
                self.interviewer.pending_action = None
                self._agent_reply(self._apply_chat_action(action), pid, sid)
                return {"status": "ok"}
            # 2. Structured action intents (validated, store-safe).
            self.interviewer.memory = runlog.get_memory(pid) if pid else {}
            action = self.interviewer.parse_action(text, self.requirements)
            prefix = ""
            if action:
                if action.get("destructive"):
                    self.interviewer.pending_action = action
                    self._agent_reply(action["prompt"], pid, sid)
                    return {"status": "ok"}
                prefix = self._apply_chat_action(action) + "\n"
            # 3. Normal interview flow.
            reply, advanced = self.interviewer.ingest_chat(text)
            self.requirements = self.interviewer.requirements
            self._agent_reply(prefix + reply if prefix else reply, pid, sid)
            if advanced and self.interviewer.is_confirmed():
                cfg: dict[str, Any] = {}
                if self._chat_pid:
                    cfg["project_id"] = self._chat_pid
                if self._chat_sid:
                    cfg["session_id"] = self._chat_sid
                return self.start(cfg or None)
            return {"status": "ok"}

        if event in ("state", "snapshot"):
            return {"status": "ok", "snapshot": self.snapshot()}

        return {"status": "unknown_event", "event": event}

    # ------------------------------------------------------- chat persistence

    def _is_dataset_question(self, lowered: str) -> bool:
        """True when a chat line asks a factual question about the data.

        Question-shaped + dataset-shaped: the text must contain an
        interrogative/imperative AND a data noun. A bare "5000 rows" answer is
        not a question, so it keeps flowing through the interview.
        """
        asks = ("what", "which", "how many", "how much", "show", "list", "tell", "describe",
                "details", "detail", "columns?", "features?", "info", "information", "looks like")
        nouns = ("dataset", "data", "rows", "row", "column", "feature", "table", "preview",
                 "records", "samples", "variables", "attributes")
        has_ask = any(a in lowered for a in asks)
        has_noun = any(n in lowered for n in nouns)
        if not (has_ask and has_noun):
            return False
        # Never hijack interview answers or action intents.
        if re.search(r"\b(use|upload|change|set|remove|drop|delete|disable|enable)\b", lowered):
            return False
        return True

    def _dataset_fact_sheet(self, pid: str | None = None) -> tuple[str, dict | None]:
        """A factual description of the live working table, computed from the
        real single-source store — never estimated or summarised from memory.

        Returns (text_payload, structured_payload) so the UI can render the
        facts as a card as well as prose.
        """
        try:
            cols = self.store.columns()
            dtypes = self.store.dtypes()
            stats = self.store.column_stats()["columns"]
        except Exception as exc:  # noqa: BLE001
            return f"Right now there's no working dataset loaded to describe: {exc}", None
        try:
            preview = self.store.preview(limit=1)
            rows = preview.get("rows", 0)
        except Exception:
            rows = 0
        prov = {}
        if pid:
            try:
                prov = runlog.get_memory(str(pid)).get("dataset") or {}
            except Exception:
                prov = {}
        prov_parts = []
        if prov.get("source"):
            prov_parts.append(f"source={prov['source']}")
        if prov.get("path"):
            prov_parts.append(f"file={prov['path']}")
        if prov.get("imported_snapshot_of"):
            prov_parts.append(f"imported snapshot of run {prov['imported_snapshot_of']}")
        prov_str = (" (" + ", ".join(prov_parts) + ")") if prov_parts else ""
        target = self.requirements.target_column
        if not target and pid:
            try:
                target = runlog.get_memory(str(pid)).get("target")
            except Exception:
                target = None
        lines = [
            f"The working table holds {rows:,} rows across {len(cols)} columns{prov_str}.\n"
            "Here is every feature with its real type and statistics:"
        ]
        for c in cols:
            st = stats.get(c, {})
            dt = st.get("type") or dtypes.get(c, "?")
            parts = [f"**{c}** ({dt})"]
            if st.get("nulls"):
                parts.append(f"{st['nulls']:,} null")
            if "uniques" in st and st["uniques"] is not None:
                parts.append(f"{st['uniques']:,} unique")
            if "min" in st:
                parts.append(f"range {self._num(st['min'])}–{self._num(st['max'])} (mean {self._num(st['mean'])})")
            elif st.get("top"):
                parts.append("top: " + ", ".join(f"'{v}'" for v in st["top"][:5]))
            if c == target:
                parts.append("**target**")
            lines.append("• " + " · ".join(parts))
        payload = {
            "rows": rows,
            "columns": list(cols),
            "preview": {"rows": rows, "columns": list(cols)},
            "stats": stats,
            "dtypes": dtypes,
            "provenance": prov or None,
            "target": target,
        } if (cols and stats) else None
        return "\n".join(lines), payload

    @staticmethod
    def _num(v: Any) -> str:
        if v is None:
            return "—"
        try:
            f = float(v)
        except (TypeError, ValueError):
            return str(v)
        if abs(f) >= 1e5 or (abs(f) > 0 and abs(f) < 1e-4):
            return f"{f:.3e}"
        return f"{f:,.4f}".rstrip("0").rstrip(".")

    def _agent_reply(self, text: str, pid: str | None, sid: str | None, dataset: dict | None = None) -> None:
        """Emit an agent turn, persist it, and compact oversized sessions.

        The event is tagged with the *chat's* project (not the currently running
        one): conversation traffic must stay with the project it belongs to, or a
        project-scoped client would receive another project's replies.
        """
        payload: dict[str, Any] = {
            "type": "chat_response",
            "role": "agent",
            "text": text,
            "stage": self.interviewer.session.stage,
            "project_id": pid,
            "session_id": sid,
        }
        if dataset:
            payload["dataset"] = dataset
        self._emit(payload)
        runlog.save_chat(pid, sid, "agent", text, agent="interviewer")
        if sid:
            runlog.compact_session(sid, pid)

    def _apply_chat_action(self, action: dict[str, Any]) -> str:
        """Execute a validated structured action. Never raises: failures become
        user-facing notes. Only whitelisted types run; column names were
        syntax-validated at parse time and are re-checked against the store."""
        atype = action.get("type")
        req = self.requirements

        if atype == "select_model":
            model = action.get("model", "")
            req.meta["model_hint"] = model
            if self.project_id:
                runlog.remember(self.project_id, "model_decisions", f"Chat selected {model} as next starting model.")
            return f"Noted — I'll start with {model} on the next run. Anything else before we train?"

        if atype == "remove_feature":
            return self.remove_feature(action.get("column", ""))

        if atype == "undo_last_feature":
            return self.undo_last_feature()

        if atype == "set_target_column":
            col = action.get("column", "")
            cols = self.store.columns()
            if col not in cols:
                return f"Column '{col}' isn't in the working dataset. Available: {', '.join(cols)}."
            req.target_column = col
            if self.project_id:
                runlog.remember(self.project_id, "target", col)
            return f"Target set to '{col}'."

        if atype == "use_dataset":
            dataset = (self.interviewer.memory or {}).get("dataset") or {}
            if not dataset.get("source"):
                return "I don't have a previous dataset recorded for this project yet."
            req.dataset_source = dataset.get("source") or req.dataset_source
            req.dataset_query = dataset.get("query", "") or req.dataset_query
            req.dataset_path = dataset.get("path", "") or req.dataset_path
            return (
                f"Reusing the previous dataset ({dataset.get('source')}"
                f"{', ' + str(dataset.get('query')) if dataset.get('query') else ''})."
            )

        return "I don't know how to do that yet."

    # ------------------------------------------------------- feature operations
    # Shared by chat actions and the Features page endpoints. Every method
    # returns a user-facing note and never raises.

    @staticmethod
    def _valid_column(col: str) -> bool:
        import re

        return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", col or ""))

    def _feature_pid(self, project_id: str | None) -> str | None:
        return project_id or self.project_id

    def disable_feature(self, column: str, project_id: str | None = None) -> str:
        """Soft-exclude a column from future training (reversible via enable)."""
        if not self._valid_column(column):
            return f"'{column}' isn't a valid column name."
        if column not in self.store.columns():
            return f"Column '{column}' isn't in the working dataset."
        if column == self.requirements.target_column:
            return f"I won't disable '{column}' — it's the prediction target."
        excluded = self.requirements.meta.setdefault("excluded_features", [])
        if column not in excluded:
            excluded.append(column)
        pid = self._feature_pid(project_id)
        if pid:
            mem = runlog.get_memory(pid)
            current = mem.get("excluded_features") or []
            if column not in current:
                runlog.remember(pid, "excluded_features", current + [column])
            runlog.remember(pid, "feature_decisions", f"Disabled feature '{column}' (excluded from training).")
        return f"Disabled '{column}' — it will be skipped in the next run. Re-enable anytime."

    def enable_feature(self, column: str, project_id: str | None = None) -> str:
        if not self._valid_column(column):
            return f"'{column}' isn't a valid column name."
        excluded = self.requirements.meta.get("excluded_features") or []
        if column in excluded:
            self.requirements.meta["excluded_features"] = [c for c in excluded if c != column]
        pid = self._feature_pid(project_id)
        if pid:
            mem = runlog.get_memory(pid)
            current = mem.get("excluded_features") or []
            if column in current:
                runlog.remember(pid, "excluded_features", [c for c in current if c != column])
            runlog.remember(pid, "feature_decisions", f"Re-enabled feature '{column}'.")
        return f"Re-enabled '{column}'."

    def remove_feature(self, column: str, project_id: str | None = None) -> str:
        """Hard-drop a column from the working table (logged, reproducible)."""
        if not self._valid_column(column):
            return f"I can't remove '{column}' — that isn't a valid column name."
        cols = self.store.columns()
        if column not in cols:
            return f"Column '{column}' isn't in the working dataset. Available: {', '.join(cols)}."
        if column == self.requirements.target_column:
            return f"I won't remove '{column}' — it's the prediction target. Change the target first if needed."
        self.store.drop_column(column)
        pid = self._feature_pid(project_id)
        if pid:
            runlog.remember(pid, "feature_decisions", f"Removed feature '{column}'.")
        return f"Removed feature '{column}'. The change is logged; re-fetch or re-run research to restore it."

    def undo_last_feature(self, project_id: str | None = None) -> str:
        added = [f for f in self.store.feature_log if f.get("name") in self.store.columns()]
        if not added:
            return "There's no engineered feature to undo right now."
        col = added[-1]["name"]
        self.store.drop_column(col)
        pid = self._feature_pid(project_id)
        if pid:
            runlog.remember(pid, "feature_decisions", f"Undid feature '{col}'.")
        return f"Undone — dropped '{col}'."

    def excluded_features(self) -> list[str]:
        return list(self.requirements.meta.get("excluded_features") or [])

    # ------------------------------------------------------------------ snapshot

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            draft = {
                "state": self.state,
                "run_context": self.run_context(),
                "requirements": self.requirements.to_dict(),
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "error": self.error,
                "iterations": json_safe(self.iterations),
                "feature_history": json_safe(self.feature_history),
                "best": json_safe(self.trainer.best if self.trainer else self.best),
                # The store table is dropped and rebuilt inside the data stage, so
                # a preview can legitimately fail mid-run. Report that honestly
                # instead of 500-ing the endpoint the UI polls on refresh.
                "dataset": self._safe_preview(),
                "tracker": self.tracker.info(),
                "mlflow_runs": self.tracker.list_runs(),
                "llm": {"provider": settings.llm_provider},
            }
            return draft

    def _safe_preview(self, limit: int = 10) -> dict:
        try:
            return self.store.preview(limit=limit)
        except Exception as exc:  # noqa: BLE001
            return {
                "rows": 0, "columns": [], "dtypes": {}, "preview": [],
                "limit": limit, "unavailable": str(exc),
            }

    def leaderboard(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "run_context": self.run_context(),
            # Single source of truth for "best" is trainer.best (snapshot agrees).
            "best": json_safe(self.trainer.best if self.trainer else self.best),
            "runs": self.tracker.list_runs(),
            "iterations": self.iterations,
        }

    def run_context(self) -> dict[str, Any]:
        """Lightweight association labels for the current/last run.

        In-memory only: exists for the duration of the process, survives on the
        finished run for display, but is NOT persistence and NOT resume.
        """
        with self._lock:
            return {
                "project_id": self.project_id,
                "session_id": self.session_id,
                "run_id": self.run_id,
                "status": self.state,
                "started_at": self.started_at,
            }

    # ------------------------------------------------------------------ helpers

    def _set_state(self, state: str) -> None:
        with self._lock:
            self.state = state
        runlog.update_run(
            self.run_id,
            stage=state,
            status={"DONE": "done", "FAILED": "failed", "PAUSED": "paused"}.get(state, "running"),
            progress=_STAGE_PROGRESS.get(state, 0.0),
        )
        self._emit({"type": "status", "state": state, "message": f"State → {state}"})

    def _progress_for(self, state: str, iteration: int | None = None) -> float:
        base = _STAGE_PROGRESS.get(state, 0.0)
        if state == "TRAINING" and iteration:
            max_iters = self.requirements.max_iterations or 0
            if max_iters > 0:
                return min(95.0, 60.0 + 35.0 * iteration / max_iters)
        return base

    def _eta_seconds(self) -> float | None:
        """Real ETA from observed iteration durations; None when unknowable."""
        times = [it.get("training_time_seconds") for it in self.iterations]
        times = [t for t in times if isinstance(t, (int, float)) and t > 0]
        max_iters = self.requirements.max_iterations or 0
        remaining = max(0, max_iters - len(self.iterations))
        if not times or remaining <= 0 or self.state != "TRAINING":
            return None
        return round(sum(times) / len(times) * remaining, 1)

    def _emit(self, event: dict[str, Any]) -> None:
        # Everything crossing the WS/API boundary must be plain JSON: numpy
        # scalars (metrics, importances) are not serializable by pydantic.
        try:
            payload = json_safe(event)
        except Exception:  # noqa: BLE001
            payload = {"type": "status", "text": str(event)[:200]}
        if not isinstance(payload, dict):
            payload = {"type": "status", "text": str(payload)[:200]}
        payload.setdefault("state", self.state)
        payload.setdefault("ts", time.time())
        # Tag every event so UIs can associate activity with the correct
        # project/run without trusting frontend filtering alone.
        payload.setdefault("project_id", self.project_id)
        payload.setdefault("session_id", self.session_id)
        payload.setdefault("run_id", self.run_id)
        self.bus.publish(payload)
        self._persist_event(payload)

    def _persist_event(self, payload: dict[str, Any]) -> None:
        """Write-through of operational events. Best-effort; training never blocks."""
        if not self.run_id:
            return
        etype = payload.get("type")
        state = payload.get("state") or self.state
        canonical: str | None = None
        metadata: dict[str, Any] = {}
        progress: float | None = None

        if etype == "status" and state in _STATUS_EVENTS:
            canonical = _STATUS_EVENTS[state]
            progress = _STAGE_PROGRESS.get(state, 0.0)
        elif etype == "dataset_preview":
            canonical = "data_validation"
            data = payload.get("data") or {}
            metadata = {"rows": data.get("rows"), "columns": data.get("columns")}
            progress = 25.0
        elif etype == "profiling":
            canonical = "profiling"
            metadata = {"summary": payload.get("summary")}
            progress = 30.0
        elif etype == "research_card":
            canonical = "feature_engineering"
            metadata = {"card": payload.get("card")}
            progress = 50.0
        elif etype == "iteration":
            canonical = "model_training"
            metadata = {
                "iteration": payload.get("iteration"),
                "model": payload.get("model"),
                "params": payload.get("params"),
                "metrics": payload.get("metrics"),
                "target_metric": payload.get("target_metric"),
                "target_met": payload.get("target_met"),
                "training_time_seconds": payload.get("training_time_seconds"),
                "mlflow_run_id": payload.get("mlflow_run_id"),
            }
            progress = self._progress_for(state, payload.get("iteration"))
        elif etype == "best_model_ready":
            canonical = "model_selected"
            metadata = {"path": payload.get("path"), "best": _safe(self.trainer.best if self.trainer else None)}
            progress = 100.0

        if canonical is None:
            return
        # Progress is monotonic per run: some stages re-emit earlier events (e.g.
        # a dataset preview after research starts), and a log that goes backwards
        # misrepresents the run.
        if progress is not None:
            self._max_progress = max(getattr(self, "_max_progress", 0.0), float(progress))
            progress = self._max_progress
        # First TRAINING sighting after research closes the research stage honestly.
        if state == "TRAINING" and self.feature_history and not self._research_completed_logged:
            self._research_completed_logged = True
            runlog.log_event(
                self.run_id, "research_completed",
                message=f"Research applied {len(self.feature_history)} feature(s).",
                stage=state, progress=55.0,
                metadata={"features": _safe(self.feature_history)},
            )
        runlog.log_event(
            self.run_id, canonical,
            message=payload.get("message") or payload.get("text"),
            status=state, stage=state, progress=progress,
            duration_ms=(payload.get("training_time_seconds") or 0) * 1000
            if etype == "iteration" and payload.get("training_time_seconds") else None,
            metadata=_safe(metadata),
        )
        if etype == "iteration":
            runlog.update_run(
                self.run_id,
                progress=progress,
                eta_seconds=self._eta_seconds(),
                best=_safe(self.trainer.best if self.trainer else None),
            )


_orchestrator_lock = threading.Lock()
_orchestrator: AutoMLOrchestrator | None = None


def get_orchestrator() -> AutoMLOrchestrator:
    global _orchestrator
    with _orchestrator_lock:
        if _orchestrator is None:
            _orchestrator = AutoMLOrchestrator()
        return _orchestrator