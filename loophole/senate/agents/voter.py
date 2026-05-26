from __future__ import annotations

import re
from typing import Any

from loophole.agents.base import BaseAgent
from loophole.senate.models import (
    Amendment,
    BillReaction,
    MoralConstitution,
    Senator,
    VoteStance,
)
from loophole.senate.personas import StateMood
from loophole.senate.prompts import VOTER_SYSTEM, VOTER_USER


class Voter(BaseAgent):
    """Simulate one senator's reaction to a bill, grounded in their constitution."""

    def _build_system_prompt(self, **kwargs: Any) -> str:
        return VOTER_SYSTEM

    def _build_user_message(self, state: Any = None, **kwargs: Any) -> str:
        senator: Senator = kwargs["senator"]
        constitution: MoralConstitution = kwargs["constitution"]
        bill_name: str = kwargs["bill_name"]
        bill_text: str = kwargs["bill_text"]
        state_mood: StateMood | None = kwargs.get("state_mood")

        return VOTER_USER.format(
            full_name=senator.full_name,
            short_name=senator.short_name,
            party=senator.party.value,
            state=senator.state,
            core_values=_bulleted(constitution.core_values),
            top_issues=_format_issues(constitution.top_issues),
            red_lines=_bulleted(constitution.red_lines),
            negotiation_style=constitution.negotiation_style,
            typical_allies=", ".join(constitution.typical_allies) or "(none recorded)",
            voice_notes=constitution.voice_notes,
            bill_name=bill_name,
            bill_text=bill_text,
            constituent_block=_format_constituent_block(state_mood, senator.state),
        )

    def react(
        self,
        senator: Senator,
        constitution: MoralConstitution,
        bill_name: str,
        bill_text: str,
        state_mood: StateMood | None = None,
    ) -> BillReaction:
        system = self._build_system_prompt()
        user_msg = self._build_user_message(
            senator=senator,
            constitution=constitution,
            bill_name=bill_name,
            bill_text=bill_text,
            state_mood=state_mood,
        )
        raw = self.llm.call(system, user_msg, temperature=self.temperature)
        return _parse_reaction(raw, senator)


def _format_constituent_block(state_mood: StateMood | None, state_code: str) -> str:
    if state_mood is None or state_mood.total == 0:
        return ""
    lines = [
        "",
        f"YOUR CONSTITUENTS BACK HOME — sampled poll of {state_code} residents",
        f"(treat as one signal among many; weigh per your usual style):",
        f"  - {state_mood.yes_count} support",
        f"  - {state_mood.no_count} oppose",
        f"  - {state_mood.unsure_count} unsure",
        f"  Aggregate: {state_mood.label}",
        "",
    ]
    if state_mood.sample_voices:
        lines.append("Sample voices from your state:")
        for v in state_mood.sample_voices[:3]:
            voice = v.reasoning[:180].strip() + ("…" if len(v.reasoning) > 180 else "")
            lines.append(f'  - [{v.vote}] "{voice}"')
        lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _bulleted(items: list[str]) -> str:
    if not items:
        return "(none recorded)"
    return "\n".join(f"  - {item}" for item in items)


def _format_issues(issues: list) -> str:
    if not issues:
        return "(none recorded)"
    parts = []
    for issue in issues:
        parts.append(f"  - {issue.issue}: {issue.stance}")
    return "\n".join(parts)


def _parse_reaction(raw: str, senator: Senator) -> BillReaction:
    vote_text = (_extract_tag(raw, "vote") or "undecided").strip().lower()
    try:
        vote = VoteStance(vote_text)
    except ValueError:
        vote = VoteStance.UNDECIDED

    confidence_text = _extract_tag(raw, "confidence") or "0.5"
    try:
        confidence = float(confidence_text.strip())
    except ValueError:
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    reasoning = _extract_tag(raw, "reasoning") or ""
    supported = _extract_items(raw, "supported")
    opposed = _extract_items(raw, "opposed")

    amendments: list[Amendment] = []
    amendments_block = _extract_tag(raw, "amendments") or ""
    for am in re.finditer(r"<amendment>(.*?)</amendment>", amendments_block, re.DOTALL):
        block = am.group(1)
        description = _extract_tag(block, "description") or ""
        rationale = _extract_tag(block, "rationale") or ""
        flip_text = (_extract_tag(block, "would_flip_vote") or "false").strip().lower()
        if description:
            amendments.append(
                Amendment(
                    description=description,
                    rationale=rationale,
                    would_flip_vote=(flip_text == "true"),
                )
            )

    return BillReaction(
        senator_full_name=senator.full_name,
        vote=vote,
        confidence=confidence,
        reasoning=reasoning,
        key_provisions_supported=supported,
        key_provisions_opposed=opposed,
        amendments=amendments,
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
