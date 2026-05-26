from __future__ import annotations

import re
from typing import Any

from loophole.agents.base import BaseAgent
from loophole.senate.models import Citation, IssueStance, MoralConstitution, Senator
from loophole.senate.prompts import BIOGRAPHER_SYSTEM, BIOGRAPHER_USER


class Biographer(BaseAgent):
    """Synthesize a senator's moral constitution from public-record knowledge."""

    def _build_system_prompt(self, **kwargs: Any) -> str:
        return BIOGRAPHER_SYSTEM

    def _build_user_message(self, state: Any = None, **kwargs: Any) -> str:
        senator: Senator = kwargs["senator"]
        return BIOGRAPHER_USER.format(
            full_name=senator.full_name,
            party=senator.party.value,
            state=senator.state,
            role=senator.role or "Senator",
        )

    def build(self, senator: Senator) -> MoralConstitution:
        system = self._build_system_prompt()
        user_msg = self._build_user_message(senator=senator)
        raw = self.llm.call(system, user_msg, temperature=self.temperature)
        return _parse_constitution(raw, senator)


def _parse_constitution(raw: str, senator: Senator) -> MoralConstitution:
    core_values = _extract_items(raw, "core_values")
    red_lines = _extract_items(raw, "red_lines")
    allies = _extract_items(raw, "typical_allies")
    negotiation_style = _extract_tag(raw, "negotiation_style") or ""
    voice_notes = _extract_tag(raw, "voice_notes") or ""
    confidence = (_extract_tag(raw, "confidence") or "medium").strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"

    top_issues: list[IssueStance] = []
    issues_block = _extract_tag(raw, "top_issues") or ""
    for issue_match in re.finditer(r"<issue>(.*?)</issue>", issues_block, re.DOTALL):
        block = issue_match.group(1)
        name = _extract_tag(block, "name") or ""
        stance = _extract_tag(block, "stance") or ""
        if name and stance:
            top_issues.append(IssueStance(issue=name, stance=stance))

    citations: list[Citation] = []
    citations_block = _extract_tag(raw, "citations") or ""
    for cm in re.finditer(r"<citation>(.*?)</citation>", citations_block, re.DOTALL):
        block = cm.group(1)
        claim = _extract_tag(block, "claim") or ""
        source_kind = _extract_tag(block, "source_kind") or "public_statement"
        detail = _extract_tag(block, "detail") or ""
        if claim and detail:
            citations.append(
                Citation(claim=claim, source_kind=source_kind, detail=detail)
            )

    return MoralConstitution(
        senator_full_name=senator.full_name,
        core_values=core_values,
        top_issues=top_issues,
        red_lines=red_lines,
        negotiation_style=negotiation_style,
        typical_allies=allies,
        voice_notes=voice_notes,
        citations=citations,
        confidence=confidence,
    )


def _extract_tag(text: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _extract_items(text: str, parent_tag: str) -> list[str]:
    parent = _extract_tag(text, parent_tag)
    if not parent:
        return []
    return [
        m.group(1).strip()
        for m in re.finditer(r"<item>(.*?)</item>", parent, re.DOTALL)
    ]
