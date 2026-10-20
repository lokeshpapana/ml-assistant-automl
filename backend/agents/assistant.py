"""Conversational layer for the AutoML assistant.

Design boundary (deliberate):

* The deterministic interviewer owns **facts and control flow**. The dataset
  briefing is emitted verbatim by this module; the model never gets a chance to
  paraphrase or drop it.
* The LLM only adds **conversational framing** around the deterministic
  ``notes`` (decisions taken this turn) and must end with the deterministic
  ``required_question``. Any required part the model drops is re-appended
  verbatim, so a bad generation can never lose information or derail the state
  machine.
* Provider failures (bad key, retired model, rate limit, timeout, 503, malformed
  body) degrade to a deterministic fallback that is explicitly labelled
  ``fallback`` — it is never presented as an LLM answer.
"""

from __future__ import annotations

import json
import time
from typing import Any

from backend.agents.llm import BaseLLM, get_llm

_SYSTEM_PROMPT = """You are the AutoML Assistant inside an ML workspace.

Help the user design and eventually run a machine-learning experiment on their
dataset. You discuss, clarify and plan. You do not train.

Hard rules:
- State ONLY facts from the provided context. Never invent numbers, column
  names, metrics, model results or dataset statistics.
- The dataset summary is already shown to the user, so do not repeat its row
  counts or column names — just talk about the plan.
- Be warm, concise and specific. Ask at most one or two focused questions.
- Never offer to start training or claim a run is approved.
- End with the required next step. If it is a question, include it verbatim as
  the final line."""

_RETRIES = 3
_BACKOFF = (0.5, 1.5, 3.0)
_TRANSIENT = (
    "503",
    "unavailable",
    "timeout",
    "timed out",
    "429",
    "rate",
    "temporarily",
    "overloaded",
    "connection",
    "reset",
    "empty response",
)


class AssistantAgent:
    """Phrases a deterministic reply plan into a conversational message."""

    def __init__(self, llm: BaseLLM | None = None):
        self.llm: BaseLLM = llm or get_llm()
        self.last_error: str | None = None

    @property
    def provider_name(self) -> str:
        return getattr(self.llm, "name", "unknown")

    @property
    def available(self) -> bool:
        if self.provider_name == "local":
            return False
        try:
            return bool(self.llm.available())
        except Exception:  # noqa: BLE001
            return False

    def compose(self, plan: dict[str, Any]) -> tuple[str, str]:
        """Return ``(message, source)`` with source in {"llm", "fallback"}."""
        briefing = str(plan.get("briefing") or "").strip()
        notes = [str(n).strip() for n in (plan.get("notes") or []) if str(n).strip()]
        question = str(plan.get("required_question") or "").strip()
        context = plan.get("context") or {}
        required = notes + ([question] if question else [])

        if not briefing and not required:
            return "", "fallback"

        if not self.available:
            self.last_error = None
            return self._fallback(briefing, required), "fallback"

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "notes": notes,
                        "required_question": question,
                        "context": context,
                    },
                    default=str,
                ),
            },
        ]

        text, error = self._call_with_retry(messages)
        self.last_error = error
        if text is None:
            return self._fallback(briefing, required), "fallback"

        # Re-append any required part the model dropped, verbatim.
        for part in required:
            if part and part not in text:
                text = f"{text.rstrip()}\n\n{part}"
        message = self._assemble(briefing, text)
        return message, "llm"

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _assemble(briefing: str, body: str) -> str:
        if briefing and briefing not in body:
            return f"{briefing}\n\n{body.strip()}"
        return body.strip()

    @staticmethod
    def _fallback(briefing: str, required: list[str]) -> str:
        parts = [p for p in [briefing, *required] if p]
        return "\n\n".join(parts)

    def _call_with_retry(self, messages: list[dict[str, str]]) -> tuple[str | None, str | None]:
        last: BaseException | None = None
        for attempt in range(_RETRIES):
            try:
                raw = self.llm.chat(messages, temperature=0.3, max_tokens=600, timeout=30)
                text = "" if raw is None else str(raw).strip()
                if not text:
                    raise ValueError("empty response from provider")
                return text, None
            except Exception as exc:  # noqa: BLE001
                last = exc
                if attempt < _RETRIES - 1 and self._is_transient(exc):
                    time.sleep(_BACKOFF[attempt])
                    continue
                break
        return None, f"{type(last).__name__}: {last}"

    @staticmethod
    def _is_transient(exc: BaseException) -> bool:
        text = str(exc).lower()
        return any(token in text for token in _TRANSIENT)
