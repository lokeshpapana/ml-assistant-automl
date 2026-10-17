"""LLM provider selection and request-shaping tests.

Providers talk plain HTTP (httpx) so tests monkeypatch ``httpx.post`` and assert
the request we would send, without any network or API key.
"""

import httpx

from backend.agents.llm import GeminiProvider, get_llm
from backend.config import settings


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_gemini_provider_shapes_request(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(url=url, headers=headers, json=json)
        return _FakeResponse(
            {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = GeminiProvider("KEY", "gemini-2.0-flash")
    out = provider.chat(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
            {"role": "user", "content": "bye"},
        ]
    )

    assert out == "hello"
    assert "gemini-2.0-flash:generateContent" in captured["url"]
    assert captured["headers"]["x-goog-api-key"] == "KEY"
    body = captured["json"]
    assert body["systemInstruction"]["parts"][0]["text"] == "sys"
    # Neutral roles map onto Gemini's user/model vocabulary.
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]
    assert body["contents"][1]["parts"][0]["text"] == "yo"


def test_get_llm_selects_gemini_when_key_present(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_api_key", "KEY")
    assert get_llm().name == "gemini"


def test_get_llm_falls_back_to_local_without_gemini_key(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_api_key", "")
    assert get_llm().name == "local"
