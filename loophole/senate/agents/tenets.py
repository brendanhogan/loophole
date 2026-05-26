from __future__ import annotations

import re
from typing import Any

from loophole.agents.base import BaseAgent
from loophole.senate.models import Bill, CoreTenets
from loophole.senate.prompts import TENETS_EXTRACTOR_SYSTEM, TENETS_EXTRACTOR_USER


class TenetsExtractor(BaseAgent):
    """Extract the immutable core tenets that constrain iteration."""

    def _build_system_prompt(self, **kwargs: Any) -> str:
        return TENETS_EXTRACTOR_SYSTEM

    def _build_user_message(self, state: Any = None, **kwargs: Any) -> str:
        return TENETS_EXTRACTOR_USER.format(
            user_prompt=kwargs["user_prompt"],
            bill_text=kwargs["bill"].text,
        )

    def extract(self, bill: Bill, user_prompt: str) -> CoreTenets:
        raw = self.llm.call(
            self._build_system_prompt(),
            self._build_user_message(bill=bill, user_prompt=user_prompt),
            temperature=self.temperature,
        )
        tenets = _extract_items(raw, "tenets", "tenet")
        return CoreTenets(tenets=tenets, extracted_from_version=bill.version)


def _extract_tag(text: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _extract_items(text: str, parent_tag: str, item_tag: str) -> list[str]:
    parent = _extract_tag(text, parent_tag)
    if not parent:
        return []
    return [
        m.group(1).strip()
        for m in re.finditer(rf"<{item_tag}>(.*?)</{item_tag}>", parent, re.DOTALL)
    ]
