from __future__ import annotations

import re
from typing import Any

from loophole.agents.base import BaseAgent
from loophole.senate.models import Bill
from loophole.senate.prompts import DRAFTER_SYSTEM, DRAFTER_USER


class Drafter(BaseAgent):
    """Plain-English idea → legal-format bill text."""

    def _build_system_prompt(self, **kwargs: Any) -> str:
        return DRAFTER_SYSTEM

    def _build_user_message(self, state: Any = None, **kwargs: Any) -> str:
        return DRAFTER_USER.format(user_prompt=kwargs["user_prompt"])

    def draft(self, user_prompt: str) -> Bill:
        raw = self.llm.call(
            self._build_system_prompt(),
            self._build_user_message(user_prompt=user_prompt),
            temperature=self.temperature,
        )
        name = _extract_tag(raw, "bill_name") or "Untitled Bill"
        summary = _extract_tag(raw, "summary") or ""
        text = _extract_tag(raw, "bill_text") or raw
        return Bill(name=name, text=text, version=1, summary=summary)


def _extract_tag(text: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None
