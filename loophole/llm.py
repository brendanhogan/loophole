from __future__ import annotations

import os
import anthropic
from openai import OpenAI


class LLMClient:
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.use_openai = False

        if os.environ.get("OPENROUTER_API_KEY"):
            self.use_openai = True
            self.openai_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY"),
            )
        elif os.environ.get("OPENAI_API_KEY"):
            self.use_openai = True
            self.openai_client = OpenAI(
                base_url=os.environ.get("OPENAI_BASE_URL"),
                api_key=os.environ.get("OPENAI_API_KEY"),
            )
        else:
            self.anthropic_client = anthropic.Anthropic()

    def call(self, system: str, user_message: str, temperature: float = 0.5) -> str:
        if self.use_openai:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_message},
                ],
            )
            return response.choices[0].message.content
        else:
            response = self.anthropic_client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text
