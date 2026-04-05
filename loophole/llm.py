from __future__ import annotations

from typing import Any, cast

from venice_sdk import VeniceClient, create_client


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


class LLMClient:
    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = 4096,
    ):
        self.client: VeniceClient = create_client()
        self.model = model or self.client.config.default_model or "llama-3.3-70b"
        self.max_tokens = max_tokens

    def call(self, system: str, user_message: str, temperature: float = 0.5) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]
        raw = self.client.chat.complete(
            messages=messages,
            model=self.model,
            temperature=temperature,
            max_completion_tokens=self.max_tokens,
            stream=False,
        )
        resp = cast(dict[str, Any], raw)
        choices = resp.get("choices") or []
        if not choices:
            return ""
        return _choice_text(cast(dict[str, Any], choices[0]))
