"""LLM provider abstraction for the multi-agent engine.

Default provider is ``local`` — fully offline/deterministic, no API key, no cloud.
``openai`` and ``anthropic`` providers are implemented over plain HTTP (httpx) and
are only activated when the matching key is present in the environment.
"""

from __future__ import annotations

from typing import Any

from backend.config import settings


class BaseLLM:
    name = "base"

    def available(self) -> bool:  # pragma: no cover - interface
        return False

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        raise NotImplementedError


class LocalLLM(BaseLLM):
    """Deterministic no-LLM provider used for offline/local testing."""

    name = "local"

    def available(self) -> bool:
        return True

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        last = messages[-1]["content"] if messages else ""
        text = f"(local) Acknowledged: {last}"
        return text


class OpenAIProvider(BaseLLM):
    name = "openai"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def available(self) -> bool:
        return bool(self.api_key)

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        import httpx

        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.2),
                "max_tokens": kwargs.get("max_tokens", 600),
            },
            timeout=kwargs.get("timeout", 60),
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class AnthropicProvider(BaseLLM):
    name = "anthropic"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def available(self) -> bool:
        return bool(self.api_key)

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        import httpx

        # Remove role prefixes Anthropic doesn't accept.
        system = ""
        cleaned: list[dict[str, str]] = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                cleaned.append({"role": m["role"], "content": m["content"]})

        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", 600),
            "messages": cleaned,
        }
        if system:
            body["system"] = system
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=body,
            timeout=kwargs.get("timeout", 60),
        )
        resp.raise_for_status()
        data = resp.json()
        return "".join(block.get("text", "") for block in data.get("content", []))


def get_llm() -> BaseLLM:
    provider = settings.llm_provider.lower()
    if provider == "openai" and settings.openai_api_key:
        return OpenAIProvider(settings.openai_api_key, settings.openai_model)
    if provider == "anthropic" and settings.anthropic_api_key:
        return AnthropicProvider(settings.anthropic_api_key, settings.anthropic_model)
    return LocalLLM()