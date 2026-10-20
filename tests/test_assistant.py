"""Conversation-layer tests.

The LLM is exercised with fakes so the suite never touches a real provider.
The invariants under test are the safety boundaries: the deterministic briefing
and required next step are never lost, prose-only LLM output, failures degrade
to a labelled fallback, and transient errors are retried while permanent ones
are not.
"""

from backend.agents.assistant import AssistantAgent
from backend.agents.llm import BaseLLM, LocalLLM


class FakeLLM(BaseLLM):
    name = "fake"

    def __init__(self, reply: str | None = "ok", error: str | None = None,
                 transient_attempts: int = 0):
        self.reply = reply
        self.error = error
        self.transient_attempts = transient_attempts
        self.calls = 0

    def available(self) -> bool:
        return True

    def chat(self, messages, **kwargs):
        self.calls += 1
        if self.error:
            if self.calls <= self.transient_attempts:
                raise TimeoutError("temporarily unavailable")
            if self.transient_attempts == 0:
                raise RuntimeError(self.error)
        return self.reply


def _plan(question="What metric should I target?"):
    return {
        "briefing": "Working dataset: 381,100 rows × 13 columns (source: upload).",
        "notes": ["Noted — targeting accuracy >= 0.90."],
        "required_question": question,
        "context": {"stage": "metric", "dataset": None, "memory": {}},
    }


def test_llm_path_preserves_briefing_and_question():
    ag = AssistantAgent(llm=FakeLLM(reply="Sounds good. What metric should I target?"))
    text, source = ag.compose(_plan())
    assert source == "llm"
    assert "381,100 rows × 13 columns" in text
    assert "What metric should I target?" in text
    assert ag.last_error is None


def test_briefing_is_emitted_even_when_model_ignores_context():
    ag = AssistantAgent(llm=FakeLLM(reply="Great, let's proceed."))
    text, source = ag.compose(_plan())
    assert source == "llm"
    assert "381,100 rows × 13 columns" in text


def test_deterministic_question_is_reappended_if_model_drops_it():
    ag = AssistantAgent(llm=FakeLLM(reply="Sure, let's proceed with that plan."))
    text, source = ag.compose(_plan())
    assert source == "llm"
    assert text.rstrip().endswith("What metric should I target?")


def test_local_provider_uses_deterministic_fallback():
    ag = AssistantAgent(llm=LocalLLM())
    text, source = ag.compose(_plan())
    assert source == "fallback"
    assert "381,100 rows × 13 columns" in text
    assert "accuracy >= 0.90" in text
    assert "What metric should I target?" in text
    assert ag.last_error is None


def test_provider_error_degrades_to_fallback_and_records_reason():
    ag = AssistantAgent(llm=FakeLLM(error="401 invalid api key"))
    text, source = ag.compose(_plan())
    assert source == "fallback"
    assert "381,100 rows × 13 columns" in text
    assert ag.last_error is not None


def test_transient_error_is_retried_then_succeeds():
    ag = AssistantAgent(llm=FakeLLM(reply="Recovered. What metric should I target?",
                                    error="503 overloaded", transient_attempts=2))
    text, source = ag.compose(_plan())
    assert source == "llm"
    assert ag.llm.calls == 3


def test_permanent_error_is_not_retried():
    ag = AssistantAgent(llm=FakeLLM(error="400 malformed request"))
    text, source = ag.compose(_plan())
    assert source == "fallback"
    assert ag.llm.calls == 1


def test_empty_response_is_treated_as_failure():
    ag = AssistantAgent(llm=FakeLLM(reply="   "))
    text, source = ag.compose(_plan())
    assert source == "fallback"


def test_plan_without_content_is_fallback_without_calling_llm():
    ag = AssistantAgent(llm=FakeLLM(reply="should not be used"))
    text, source = ag.compose({"briefing": "", "notes": [], "required_question": "", "context": {}})
    assert (text, source) == ("", "fallback")
    assert ag.llm.calls == 0


# ---------------------------------------------------- orchestrator integration


def _chat_event(orch, events):
    return [e for e in events if e.get("type") == "chat_response"][-1]


def test_orchestrator_chat_uses_conversational_layer(monkeypatch):
    from backend.orchestrator import get_orchestrator

    orch = get_orchestrator()
    monkeypatch.setattr(orch, "assistant", AssistantAgent(llm=FakeLLM(reply="Let's go.")))
    events: list = []
    monkeypatch.setattr(orch, "_emit", lambda e: events.append(e))

    res = orch.handle_message(
        {"event": "chat", "project_id": None, "session_id": None,
         "message": "predict churn at 90 percent accuracy"}
    )
    assert res["status"] == "ok"
    chat = _chat_event(orch, events)
    assert chat["source"] == "llm"
    # The deterministic next question survives the LLM framing.
    assert orch.interviewer._last_question and orch.interviewer._last_question in chat["text"]
    # A chat message must never launch training.
    assert orch.state == "IDLE"


def test_orchestrator_chat_falls_back_and_surfaces_provider_error(monkeypatch):
    from backend.orchestrator import get_orchestrator

    orch = get_orchestrator()
    monkeypatch.setattr(
        orch, "assistant", AssistantAgent(llm=FakeLLM(error="401 invalid api key"))
    )
    events: list = []
    monkeypatch.setattr(orch, "_emit", lambda e: events.append(e))

    res = orch.handle_message(
        {"event": "chat", "project_id": None, "session_id": None,
         "message": "predict churn"}
    )
    assert res["status"] == "ok"
    chat = _chat_event(orch, events)
    assert chat["source"] == "fallback"
    assert chat.get("provider_error")
    assert orch.state == "IDLE"


def test_llm_cannot_launch_training(monkeypatch):
    """Even if the model says 'start', only the deterministic gate can run."""
    from backend.orchestrator import get_orchestrator

    orch = get_orchestrator()
    monkeypatch.setattr(orch, "assistant", AssistantAgent(llm=FakeLLM(reply="start training now")))
    events: list = []
    monkeypatch.setattr(orch, "_emit", lambda e: events.append(e))

    orch.handle_message(
        {"event": "chat", "project_id": None, "session_id": None, "message": "predict churn"}
    )
    assert orch.state == "IDLE"
    assert all(e.get("type") != "status" or e.get("state") != "TRAINING" for e in events)
