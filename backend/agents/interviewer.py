"""Agent 1: Interviewer & Requirements Agent.

Collects the topic / goal description, maximum dataset row limit, target metric &
value, and the dataset source. Confirms everything before signalling the
orchestrator to proceed. Works through a deterministic local state machine when no
LLM key is configured.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from backend.agents.common import (
    _TARGET_COLUMN_CANDIDATES,
    Requirements,
    parse_target_spec,
)
from backend.agents.llm import BaseLLM, get_llm
from backend.config import settings

_QUESTION_ORDER = ["goal", "metric", "source", "rows", "target_column", "confirm"]

_INT_MATCH = re.compile(r"(\d[\d,]*)")

# A row count is a bare whole number, optionally with a k/m suffix. Decimals such
# as the "0.92" in "accuracy >= 0.92" must never be read as a row count.
_ROWS_INT_RE = re.compile(r"(?<![\d.])(\d[\d,]*)\s*(?:k|m)?(?![\d.])", re.IGNORECASE)

_METRIC_NAME_RE = (
    r"accuracy|acc|f1|precision|recall|auc|roc(?:_auc)?|logloss|rmse|mae|mse|r2(?:_score)?"
)
# "90% accuracy" / "90 percent accuracy" / "85 pct f1"
_PCT_FIRST_RE = re.compile(
    rf"(?P<num>\d+(?:\.\d+)?)\s*(?:%|percent|pct)\s*(?P<met>{_METRIC_NAME_RE})", re.IGNORECASE
)
# "rmse under 1.2" / "accuracy of 0.9" / "f1 at least 0.85"
_WORDY_METRIC_RE = re.compile(
    rf"(?P<met>{_METRIC_NAME_RE})\s+(?P<word>under|below|at most|over|above|at least|of|is|around|about)\s+(?P<num>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
# "accuracy 0.9" / "rmse <= 1.2" embedded in a longer sentence
_LOOSE_METRIC_RE = re.compile(
    rf"(?P<met>{_METRIC_NAME_RE})\s*(?P<op>>=|<=|>|<|=)?\s*(?P<num>\d+(?:\.\d+)?)\s*(?P<pct>%)?",
    re.IGNORECASE,
)
# "10000 rows" / "10k samples" / "1m records"
_ROWS_RE = re.compile(
    r"(?P<num>\d[\d,]*)\s*(?P<suffix>k|m)?\s*(?:rows|samples|records|data\s*points|examples)",
    re.IGNORECASE,
)
# "target is price" / "label column churn" — require the explicit marker
# ("is" or "column"), otherwise ordinary goal wording like "predict target from
# my csv" would be misread as a column named "from".
_TARGET_IS_RE = re.compile(
    r"(?:target|label|outcome)(?:\s+column)?\s+is\s+(?P<col>[a-zA-Z_][a-zA-Z0-9_]*)"
    r"|(?:target|label|outcome)\s+column\s+(?P<col2>[a-zA-Z_][a-zA-Z0-9_]*)",
    re.IGNORECASE,
)

_WORDY_OPS = {
    "under": "<=",
    "below": "<=",
    "at most": "<=",
    "over": ">=",
    "above": ">=",
    "at least": ">=",
}

_AFFIRMATIVES = {
    "start",
    "go",
    "begin",
    "yes",
    "ok",
    "confirm",
    "yes start",
    "let's go",
    "lets go",
    "do it",
    "proceed",
    "start automl",
    "run it",
    "start training",
    "start the run",
    "start the training",
    "y",
    "yeah",
    "sure",
}

_SOURCE_KEYWORDS = {
    "synthetic": ["synthetic", "generate", "demo", "no data", "new data", "simulate"],
    "upload": ["upload", "file", "csv", "my file", "local"],
    "openml": ["openml", "search", "find", "fetch", "dataset online"],
    "kaggle": ["kaggle"],
}


@dataclass
class InterviewerSession:
    requirements: Requirements = field(default_factory=Requirements)
    stage: str = "goal"
    awaiting: str = "goal"
    messages: list[dict[str, str]] = field(default_factory=list)


class InterviewerAgent:
    def __init__(self, llm: BaseLLM | None = None):
        self.llm: BaseLLM = llm or get_llm()
        self.session = InterviewerSession()
        self.pending_action: dict[str, Any] | None = None
        self.memory: dict[str, Any] = {}
        self.dataset_context: dict[str, Any] = {}

    def reset(self, defaults: Requirements | None = None) -> None:
        self.session = InterviewerSession(requirements=defaults or Requirements())
        self.pending_action = None

    @property
    def requirements(self) -> Requirements:
        return self.session.requirements

    @property
    def awaiting(self) -> str:
        return self.session.awaiting

    @awaiting.setter
    def awaiting(self, value: str) -> None:
        self.session.awaiting = value

    def apply_defaults(self) -> Requirements:
        req = self.requirements
        # A row cap must always be a usable number: never 0 or negative.
        if req.max_rows <= 0:
            req.max_rows = settings.automl_default_max_rows
        req.metric = req.metric or settings.automl_default_metric
        if req.target_value <= 0 and req.metric not in ("rmse", "mae", "mse"):
            req.target_value = settings.automl_default_target_value
        req.max_iterations = req.max_iterations or settings.automl_max_iterations
        req.feature_engineering_every_n = (
            req.feature_engineering_every_n or settings.automl_feature_engineering_every_n
        )
        return req

    def next_question(self) -> str:
        req = self.requirements
        if not req.goal:
            return "What ML task would you like to build? Describe your topic and goal (e.g. 'predict customer churn')."
        if not req.metric_expr:
            return (
                "What target metric and value should I train towards? "
                "Examples: 'accuracy >= 0.90', 'f1 >= 0.85', 'RMSE <= 1.2' (or type 'auto')."
            )
        if req.dataset_source == "":
            return (
                "Where should I get the dataset? Choose one: 'upload' (I upload a file), "
                "'openml' (search a dataset), 'kaggle' (needs credentials), or 'synthetic' (generate a demo dataset)."
            )
        if self.awaiting == "rows" and not req.meta.get("rows_explicit"):
            total = (self.dataset_context or {}).get("rows")
            if total:
                return (
                    f"Your working table holds {int(total):,} rows. How many should I cap "
                    "training at? Examples: '10k rows', '50000 rows', or 'auto' for the "
                    "50,000 default."
                )
            return (
                "How many rows should I cap the dataset at? "
                "Examples: '10k rows', '50000 rows', or 'auto' for the 50,000 default."
            )
        if self.awaiting == "target_column" and req.target_column is None:
            return (
                "Which column should I predict? Reply with the column name, "
                "or 'auto' and I'll detect it from the data."
            )
        if not req.confirm:
            return (
                "Shall I proceed? I'll cap the dataset to "
                f"{req.max_rows:,} rows, target {req.metric} {'>=' if req.direction == 'maximize' else '<='} "
                f"{req.target_value}, source={req.dataset_source}. Reply 'start' / 'yes' to begin."
            )
        return "All requirements confirmed."

    def ingest_form(self, data: dict[str, Any]) -> str:
        """Populate requirements directly from the left-panel form."""
        req = self.requirements
        req.goal = (data.get("goal") or req.goal or "").strip()
        metric_expr = (data.get("metric") or data.get("metric_expr") or "").strip()
        if metric_expr:
            self._apply_metric(metric_expr)
        src = (data.get("dataset_source") or "").strip().lower()
        if src:
            req.dataset_source = self._normalize_source(src)
        req.dataset_path = (data.get("dataset_path") or req.dataset_path or "").strip()
        req.dataset_query = (data.get("dataset_query") or req.dataset_query or "").strip()
        try:
            rows = int(data.get("max_rows") or 0) or req.max_rows
            req.max_rows = min(max(rows, 100), 5_000_000)
        except (TypeError, ValueError):
            pass
        req.target_column = (data.get("target_column") or req.target_column or "").strip() or None
        try:
            iters = int(data.get("max_iterations") or 0)
            if iters > 0:
                req.max_iterations = iters
        except (TypeError, ValueError):
            pass
        req.confirm = True
        self.apply_defaults()
        return self.confirm_summary()

    def ingest_chat(self, text: str) -> tuple[str, bool]:
        """Handle a chat line, grounded in the live dataset and project memory.

        Before the interview state machine runs we read whatever the loaded
        dataset and the project's stored memory already tell us (data source,
        target column, task type, prior goal). That context is announced once and
        the matching questions are skipped, so the agent reasons about *this* data
        and *this* project instead of asking blind for a row cap and a target.
        """
        self._infer_from_dataset()
        briefing = self._dataset_briefing()
        reply, advanced = self._ingest_chat_inner(text)
        if briefing:
            reply = f"{briefing}\n\n{reply}"
        return reply, advanced

    def _ingest_chat_inner(self, text: str) -> tuple[str, bool]:
        """The deterministic interview state machine (see ``ingest_chat``)."""
        req = self.requirements
        text = text.strip()
        lower = text.lower()

        if lower in {"stop", "pause", "stop training", "halt"}:
            return "Stopping current run.", False

        if lower in _AFFIRMATIVES:
            req.confirm = True
            self.apply_defaults()
            return self.confirm_summary(), True

        # Absorb indirect signals from free text before stage handling, so one
        # message like "predict churn at 90% accuracy from my csv, 10k rows"
        # can fill several requirements at once.
        notes = self._absorb_signals(text)
        # A message that carried a target metric must not have its digits stolen
        # as a row count by a later stage ("accuracy >= 0.92" -> 0 rows).
        metric_in_msg = bool(req.meta.get("_fresh_metric"))

        if self.awaiting == "goal":
            req.goal = text
            self.awaiting = "metric"
        elif self.awaiting == "metric":
            if lower in {"auto", "automatic", "best"}:
                self._apply_metric("")
            elif not req.meta.pop("_fresh_metric", False):
                # _absorb_signals already set a proper spec; don't clobber it
                # with a raw-text parse.
                self._apply_metric(text)
            self.awaiting = "source"
        elif self.awaiting == "source":
            if not req.meta.get("source_explicit"):
                req.dataset_source = self._normalize_source(text)
            self.awaiting = "rows"
        elif self.awaiting == "rows":
            match = None if metric_in_msg else _ROWS_INT_RE.search(text)
            if match:
                req.max_rows = int(match.group(1).replace(",", ""))
                req.meta["rows_explicit"] = True
            elif lower in {"auto", "automatic", "default"}:
                req.meta["rows_explicit"] = True
            self.awaiting = "target_column"
        elif self.awaiting == "target_column":
            if lower in {"auto", "don't know", "dont know", "none", "n/a"} or not text:
                req.target_column = None
                self.awaiting = "confirm"
            elif _ROWS_RE.search(text) and not req.meta.get("target_explicit"):
                # Rows answer arrived one stage late — it was already absorbed
                # above; stay here and ask for the target column properly.
                self.awaiting = "target_column"
            else:
                if not req.meta.get("target_explicit"):
                    req.target_column = text
                self.awaiting = "confirm"
        elif self.awaiting == "confirm":
            # Last-minute correction at the confirm gate must not launch training.
            if notes:
                self.apply_defaults()
                return self._with_notes(notes) + self.next_question(), False
            # Only an explicit affirmative confirms. Anything else (e.g. "use
            # xgboost") is a follow-up instruction that must NOT start a run.
            if lower in _AFFIRMATIVES:
                req.confirm = True
                self.apply_defaults()
                return self.confirm_summary(), True
            return self.next_question(), False

        # Skip questions whose answers were already inferred indirectly.
        self._skip_filled_stages()
        reply = self.next_question()
        if self.awaiting == "confirm":
            return self._with_notes(notes) + reply, False
        return self._with_notes(notes) + reply, True

    # ------------------------------------------------------- indirect input

    def _absorb_signals(self, text: str) -> list[str]:
        """Extract metric / source / row-cap / target signals from free text.

        Returns human-readable notes of what was inferred, so the agent can
        announce its autonomous decisions in the reply. A newly found signal
        that differs from the current value *updates* it (with an "updated"
        note) — this is what makes last-minute corrections work.
        """
        req = self.requirements
        notes: list[str] = []
        lower = text.lower()

        # --- target metric ---
        spec = self._sniff_metric(text)
        if spec and spec != req.metric_expr:
            first = not req.meta.get("metric_explicit")
            req.metric_expr = spec
            self._apply_metric(spec)
            req.meta["metric_explicit"] = True
            req.meta["_fresh_metric"] = True
            notes.append(f"{'targeting' if first else 'updated target to'} {spec}")

        # --- dataset source (explicit keyword hits only; switching allowed) ---
        for source, keywords in _SOURCE_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                if source != req.dataset_source or not req.meta.get("source_explicit"):
                    verb = "using" if not req.meta.get("source_explicit") else "switched to"
                    req.dataset_source = source
                    notes.append(f"{verb} {source} data")
                req.meta["source_explicit"] = True
                break

        # --- row cap ("10k rows", "1m records") ---
        m = _ROWS_RE.search(text)
        if m:
            try:
                rows = int(m.group("num").replace(",", ""))
                suffix = (m.group("suffix") or "").lower()
                if suffix == "k":
                    rows *= 1_000
                elif suffix == "m":
                    rows *= 1_000_000
                rows = min(max(rows, 100), 5_000_000)
                if rows != req.max_rows or not req.meta.get("rows_explicit"):
                    verb = (
                        "capping data at"
                        if not req.meta.get("rows_explicit")
                        else "updated row cap to"
                    )
                    notes.append(f"{verb} {rows:,} rows")
                req.max_rows = rows
                req.meta["rows_explicit"] = True
            except (TypeError, ValueError):
                pass

        # --- target column ("target is price") ---
        m = _TARGET_IS_RE.search(text)
        if m:
            col = m.group("col") or m.group("col2")
            if col:
                if col != req.target_column:
                    notes.append(f"target column '{col}'")
                req.target_column = col
                req.meta["target_explicit"] = True

        return notes

    @staticmethod
    def _sniff_metric(text: str) -> str | None:
        """Turn indirect metric phrasing into a strict ``metric op value`` string."""
        from backend.agents.common import METRIC_SYNONYMS, MINIMIZE_METRICS

        def canon(name: str) -> str:
            return METRIC_SYNONYMS.get(name.lower(), name.lower())

        m = _PCT_FIRST_RE.search(text)
        if m:
            return f"{canon(m.group('met'))} >= {float(m.group('num')) / 100:.4g}"

        m = _WORDY_METRIC_RE.search(text)
        if m:
            metric = canon(m.group("met"))
            word = m.group("word").lower()
            value = float(m.group("num"))
            if word in _WORDY_OPS:
                return f"{metric} {_WORDY_OPS[word]} {value:.4g}"
            if value > 1 and metric not in MINIMIZE_METRICS:
                value /= 100
            op = "<=" if metric in MINIMIZE_METRICS else ">="
            return f"{metric} {op} {value:.4g}"

        m = _LOOSE_METRIC_RE.search(text)
        if m:
            # Avoid matching a bare number glued to an unrelated word salad:
            # require an explicit operator, a % sign, or "of/is/:/=" nearby.
            op, pct = m.group("op"), m.group("pct")
            context = text[max(0, m.start() - 6) : m.end() + 2].lower()
            glued = bool(op or pct or any(w in context for w in ("of", "is", "=", ":", "at", "to")))
            if not glued:
                return None
            metric = canon(m.group("met"))
            value = float(m.group("num"))
            if pct or (value > 1 and metric not in MINIMIZE_METRICS):
                value = value / 100 if value > 1 else value
            op = op or ("<=" if metric in MINIMIZE_METRICS else ">=")
            return f"{metric} {op} {value:.4g}"
        return None

    def _skip_filled_stages(self) -> None:
        """Advance past questions already answered via indirect signals."""
        req = self.requirements
        order = ["goal", "metric", "source", "rows", "target_column", "confirm"]
        filled = {
            "goal": bool(req.goal),
            "metric": bool(req.metric_expr),
            "source": req.dataset_source != "",
            "rows": bool(req.meta.get("rows_explicit")),
            "target_column": req.target_column is not None,
        }
        while self.awaiting in filled and filled[self.awaiting]:
            if self.awaiting == "metric":
                req.meta.pop("_fresh_metric", None)
            self.awaiting = order[order.index(self.awaiting) + 1]

    # ------------------------------------------------------- dataset awareness

    def set_dataset_context(self, ctx: dict[str, Any] | None) -> None:
        """Attach a structured description of the live dataset + project memory.

        Supplied by the orchestrator (which owns the store). When absent the
        interview behaves exactly as before.
        """
        self.dataset_context = ctx or {}

    def _infer_from_dataset(self) -> list[str]:
        """Fill requirements from the real dataset and the project's memory.

        Only clearly-known facts are applied — the data source recorded in
        provenance, the project's remembered target (or an explicit target-like
        column), and the detected task type. Ambiguous choices are left for the
        interview to ask about, never invented.
        """
        ctx = self.dataset_context or {}
        if not ctx.get("available"):
            return []
        req = self.requirements
        cols = [str(c) for c in (ctx.get("columns") or [])]
        if not cols:
            return []
        notes: list[str] = []

        # The project already has a working table: don't ask "where should I get
        # the dataset?" as if no data existed.
        prov = ctx.get("provenance") or {}
        src = str(prov.get("source") or "").strip().lower()
        if src and not req.meta.get("source_explicit"):
            req.dataset_source = src
            if prov.get("path"):
                req.dataset_path = str(prov["path"])
            req.meta["source_explicit"] = True
            notes.append(f"using the loaded {src} data")

        # Target column: prefer the project's remembered target, else the first
        # explicit target/label-like column. Never silently invent one.
        if req.target_column is None:
            remembered = ctx.get("target")
            candidate: str | None = None
            if remembered and str(remembered) in cols:
                candidate = str(remembered)
            else:
                lower_map = {c.lower(): c for c in cols}
                for cand in _TARGET_COLUMN_CANDIDATES:
                    if cand in lower_map:
                        candidate = lower_map[cand]
                        break
            if candidate:
                req.target_column = candidate
                req.meta["target_inferred"] = True
                notes.append(f"target column '{candidate}'")

        # Task type detected from the real target values.
        task = ctx.get("task_type")
        if task and task != "auto":
            req.task_type = task

        return notes

    def _dataset_briefing(self) -> str:
        """One-time, grounded summary of the live dataset and project memory."""
        ctx = self.dataset_context or {}
        req = self.requirements
        if not ctx.get("available") or req.meta.get("dataset_announced"):
            return ""
        cols = [str(c) for c in (ctx.get("columns") or [])]
        if not cols:
            return ""
        req.meta["dataset_announced"] = True

        rows = int(ctx.get("rows") or 0)
        prov = ctx.get("provenance") or {}
        stats = ctx.get("stats") or {}
        target = req.target_column or ctx.get("target")
        task = ctx.get("task_type") or "auto"

        src = prov.get("source")
        path = prov.get("path")
        origin = f" (source: {src}{', ' + str(path) if path else ''})" if src else ""
        lines = [
            f"I can see your working dataset — **{rows:,} rows × {len(cols)} columns**{origin}."
        ]

        if req.goal:
            lines.append(f"Project goal on record: **{req.goal}**.")

        if target and str(target) in cols:
            st = stats.get(str(target), {})
            extra = []
            if st.get("uniques") is not None:
                extra.append(f"{int(st['uniques']):,} distinct values")
            if st.get("type"):
                extra.append(str(st["type"]))
            suffix = f" ({', '.join(extra)})" if extra else ""
            task_s = "" if task in ("", "auto") else f", looks like a **{task}** task"
            lines.append(f"Target: **{target}**{suffix}{task_s}.")
        else:
            lines.append("I couldn't pin down a target column from the data, so I'll ask.")

        shown = ", ".join(f"`{c}`" for c in cols[:20])
        more = "" if len(cols) <= 20 else f" (+{len(cols) - 20} more)"
        lines.append(f"Columns: {shown}{more}.")
        return "\n".join(lines)

    @staticmethod
    def _with_notes(notes: list[str]) -> str:
        if not notes:
            return ""
        return "Noted — " + ", ".join(notes) + ". "

    def confirm_summary(self) -> str:
        req = self.requirements
        if req.metric_expr:
            metric_desc = req.metric_expr
        else:
            metric_desc = (
                f"{req.metric} {'>=' if req.direction == 'maximize' else '<='} {req.target_value}"
            )
        return (
            f"Requirements confirmed:\n"
            f"  • Goal: {req.goal or '(auto)'}\n"
            f"  • Target: {metric_desc}\n"
            f"  • Dataset: {req.dataset_source}"
            f"{f' ({req.dataset_path})' if req.dataset_path else ''}"
            f"{f' query={req.dataset_query}' if req.dataset_query else ''}\n"
            f"  • Max rows: {req.max_rows:,}\n"
            f"  • Max iterations: {req.max_iterations}\n"
            f"Starting the AutoML engine now."
        )

    def _apply_metric(self, text: str) -> None:
        req = self.requirements
        req.metric_expr = text
        spec = parse_target_spec(text or None, default_metric=settings.automl_default_metric)
        req.metric = spec.metric
        req.target_value = spec.value if text else settings.automl_default_target_value
        req.direction = spec.direction
        self.apply_defaults()

    def _normalize_source(self, text: str) -> str:
        lower = text.strip().lower()
        for source, keywords in _SOURCE_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                return source
        return "synthetic"

    def is_confirmed(self) -> bool:
        self.apply_defaults()
        return bool(self.requirements.confirm and self.requirements.goal)

    # ------------------------------------------------------- structured actions

    _MODEL_ALIASES = {
        "random forest": "random_forest",
        "randomforest": "random_forest",
        "rf": "random_forest",
        "xgboost": "xgboost",
        "xgb": "xgboost",
        "lightgbm": "lightgbm",
        "lgbm": "lightgbm",
        "logistic regression": "logistic_regression",
        "svm": "svm",
        "support vector": "svm",
        "neural network": "neural_network",
        "mlp": "neural_network",
        "gradient boosting": "gradient_boosting",
        "gbm": "gradient_boosting",
        "linear regression": "linear_regression",
        "ridge": "ridge",
    }
    _COLUMN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def parse_action(self, text: str, requirements: Requirements) -> dict[str, Any] | None:
        """Parse a tool-like instruction into a validated action dict.

        Returns ``None`` for ordinary interview text. Destructive actions
        (remove/undo) are returned with ``destructive=True`` so the caller
        asks for confirmation; they are executed only via an explicit confirm.
        Column names are syntax-validated here; existence/ownership checks
        happen at execution time (orchestrator owns the store).
        """
        lower = text.strip().lower()

        m = re.search(
            r"(?:use|try|switch to|train with)\s+(?:the\s+)?"
            r"(random forest|randomforest|rf|xgboost|xgb|lightgbm|lgbm|logistic regression"
            r"|svm|support vector|neural network|mlp|gradient boosting|gbm|linear regression|ridge)"
            r"(?:\s+model)?",
            lower,
        )
        if m:
            name = self._MODEL_ALIASES[m.group(1)]
            try:
                from ml.models.registry import get_available_models

                known = set(get_available_models("classification")) | set(
                    get_available_models("regression")
                )
            except Exception:
                known = set(self._MODEL_ALIASES.values())
            if name in known:
                return {"type": "select_model", "model": name, "destructive": False}
            return None

        m = re.search(
            r"(?:remove|drop|delete|exclude)\s+(?:the\s+)?(?:feature\s+|column\s+)?([A-Za-z_][A-Za-z0-9_]*)",
            text.strip(),
        )
        if m:
            col = m.group(1)
            if col.lower() in {"feature", "column", "target", "model", "dataset"}:
                return None
            return {
                "type": "remove_feature",
                "column": col,
                "destructive": True,
                "prompt": (
                    f"Remove feature '{col}' from the working dataset? "
                    "Reply 'yes' to confirm. (Logged and reproducible; "
                    "re-fetch or re-run research to restore it.)"
                ),
            }

        if re.search(r"\bundo\b|\brevert\s+(?:the\s+|my\s+)?last\b", lower):
            return {
                "type": "undo_last_feature",
                "destructive": True,
                "prompt": "Undo the last feature change? Reply 'yes' to confirm.",
            }

        m = re.search(
            r"(?:set|change|make)\s+(?:the\s+)?target(?:\s+column)?\s+to\s+([A-Za-z_][A-Za-z0-9_]*)",
            text.strip(),
            re.IGNORECASE,
        )
        if m:
            return {"type": "set_target_column", "column": m.group(1), "destructive": False}

        if re.search(
            r"use (?:the )?same dataset|reuse (?:the )?(?:last|previous) dataset|use (?:the )?last dataset",
            lower,
        ):
            dataset = (self.memory or {}).get("dataset") or {}
            if dataset.get("source"):
                return {"type": "use_dataset", "destructive": False}
            return None

        return None
