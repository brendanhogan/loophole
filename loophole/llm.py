from __future__ import annotations

import os
from typing import Any, cast, Protocol


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


class VeniceBackend:
    """Venice AI backend."""

    def __init__(self, model: str | None):
        from venice_sdk import VeniceClient, create_client

        self.client: VeniceClient = create_client()
        self.model = model or self.client.config.default_model or "llama-3.3-70b"

    def call(self, system: str, user_message: str, temperature: float, max_tokens: int) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]
        raw = self.client.chat.complete(
            messages=messages,
            model=self.model,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            stream=False,
        )
        resp = cast(dict[str, Any], raw)
        choices = resp.get("choices") or []
        if not choices:
            return ""
        return self._choice_text(cast(dict[str, Any], choices[0]))

    @staticmethod
    def _choice_text(choice: dict[str, Any]) -> str:
        msg = choice.get("message") or {}
        content = msg.get("content")
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
        return str(content or "")


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
    """

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = 4096,
        provider: str | None = None,
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
            self.backend = VeniceBackend(resolved_model)

        self.model = resolved_model
        self.max_tokens = max_tokens

    def call(self, system: str, user_message: str, temperature: float = 0.5) -> str:
        return self.backend.call(system, user_message, temperature, self.max_tokens)
