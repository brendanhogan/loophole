from __future__ import annotations

import os
from typing import Any, Protocol


class LLMBackend(Protocol):
    """Protocol for LLM backends."""

    def call(self, system: str, user_message: str, temperature: float, max_tokens: int) -> str: ...


class AnthropicBackend:
    """Anthropic Claude backend."""

    def __init__(self, model: str):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model

    def call(self, system: str, user_message: str, temperature: float, max_tokens: int) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text


def resolve_openai_compatible_api_key(base_url: str) -> str | None:
    """
    API key for OpenAI-compatible HTTP APIs (OpenRouter, Venice, local proxies, etc.).

    Prefer OPENAI_API_KEY; for Venice's hosted API also accept VENICE_API_KEY.
    """
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if key:
        return key
    if "venice.ai" in base_url.lower():
        key = (os.environ.get("VENICE_API_KEY") or "").strip()
        if key:
            return key
    return None


class OpenAICompatibleBackend:
    """OpenAI Chat Completions-compatible HTTP API (e.g. Venice at https://api.venice.ai/api/v1)."""

    def __init__(self, *, base_url: str, api_key: str, model: str):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key)
        self.model = model

    def call(self, system: str, user_message: str, temperature: float, max_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
        )
        msg = response.choices[0].message
        content: Any = msg.content
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        return _openai_message_content_as_text(content)


def _openai_message_content_as_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return str(content)


VENICE_DEFAULT_BASE_URL = "https://api.venice.ai/api/v1"

PROVIDER_DEFAULTS = {
    "anthropic": "claude-sonnet-4-20250514",
    "venice": "llama-3.3-70b",
}


class LLMClient:
    """
    Unified LLM client supporting multiple backends.

    Provider selection (in priority order):
    1. LOOPHOLE_PROVIDER environment variable ("anthropic" or "venice")
    2. model.provider in config.yaml
    3. Default: "anthropic" (backward compatible)

    Venice uses the OpenAI-compatible surface: set base_url (default https://api.venice.ai/api/v1)
    and OPENAI_API_KEY or VENICE_API_KEY per upstream review feedback.
    """

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = 4096,
        provider: str | None = None,
        base_url: str | None = None,
    ):
        self.provider = (
            os.environ.get("LOOPHOLE_PROVIDER")
            or provider
            or "anthropic"
        ).lower()

        if self.provider not in ("anthropic", "venice"):
            raise ValueError(f"Unknown provider: {self.provider}. Use 'anthropic' or 'venice'.")

        resolved_model = model or PROVIDER_DEFAULTS.get(self.provider)

        if self.provider == "anthropic":
            self.backend: LLMBackend = AnthropicBackend(resolved_model)
        else:
            resolved_base = (
                (os.environ.get("LOOPHOLE_BASE_URL") or "").strip()
                or (base_url or "").strip()
                or VENICE_DEFAULT_BASE_URL
            )
            api_key = resolve_openai_compatible_api_key(resolved_base)
            if not api_key:
                raise ValueError(
                    "Venice / OpenAI-compatible provider requires OPENAI_API_KEY "
                    "(or VENICE_API_KEY when using api.venice.ai)."
                )
            self.backend = OpenAICompatibleBackend(
                base_url=resolved_base,
                api_key=api_key,
                model=resolved_model,
            )

        self.model = resolved_model
        self.max_tokens = max_tokens

    def call(self, system: str, user_message: str, temperature: float = 0.5) -> str:
        return self.backend.call(system, user_message, temperature, self.max_tokens)
