"""Constituent voter — a sampled Nemotron persona reacts to a bill."""

from __future__ import annotations

import re
from typing import Any

from loophole.agents.base import BaseAgent
from loophole.senate.personas import ConstituentVote, Persona


CONSTITUENT_SYSTEM = """\
You are simulating how an ordinary American would react to a proposed law. \
You will be given a persona — demographic facts plus a short biography — \
and the text of a bill. React as that person would react.

GROUND RULES:
1. Reason from the persona's actual circumstances. A retiree on fixed income \
worries about prices. A small-business owner worries about regulation. A \
union member thinks about labor protections. Don't impose generic political \
positions — start from how this specific person's life would be affected.
2. The vote field uses three values: yes, no, unsure. Be decisive when you \
reasonably can. Most Americans have an instinctive yes-or-no on most policy \
questions even without policy expertise — even if your gut answer is rough, \
pick the direction it leans. Reserve "unsure" for cases of genuine internal \
conflict (the bill helps you AND hurts you, or you actively disagree with \
yourself), NOT for "I haven't thought about this much."
3. Reasoning is 1-2 sentences in the persona's voice — informal, grounded \
in their daily life. NOT in the voice of a senator or policy expert.
4. Key concern is a 3-8 word phrase summarizing their top concern about \
this bill, useful for aggregating views. E.g., "premium cost", "small \
business impact", "fairness to taxpayers".

Output ONLY in these tags:

<vote>yes OR no OR unsure</vote>
<reasoning>[1-2 sentences in the persona's voice]</reasoning>
<key_concern>[3-8 word phrase]</key_concern>"""


CONSTITUENT_USER = """\
PERSONA: {intro}

About them:
{about}

Lifestyle & values:
{persona_block}

---

THE BILL — "{bill_name}":

{bill_text}

---

How would this person react? Vote, reasoning in their voice, and their top \
concern about the bill."""


class ConstituentVoter(BaseAgent):
    """Cheap agent: persona + bill → vote + reasoning. Use a smaller model."""

    def _build_system_prompt(self, **kwargs: Any) -> str:
        return CONSTITUENT_SYSTEM

    def _build_user_message(self, state: Any = None, **kwargs: Any) -> str:
        persona: Persona = kwargs["persona"]
        bill_name: str = kwargs["bill_name"]
        bill_text: str = kwargs["bill_text"]

        about_lines = []
        if persona.education_level:
            about_lines.append(f"  Education: {persona.education_level.replace('_', ' ')}")
        if persona.bachelors_field:
            about_lines.append(f"  Field: {persona.bachelors_field}")
        if persona.marital_status:
            about_lines.append(f"  Marital status: {persona.marital_status.replace('_', ' ')}")
        if persona.cultural_background:
            about_lines.append(f"  Background: {_clip(persona.cultural_background, 280)}")

        persona_block = "\n".join(
            [
                _clip(persona.persona, 380),
                _clip(persona.career_goals_and_ambitions, 220),
                _clip(persona.hobbies_and_interests, 200),
            ]
        ).strip()

        return CONSTITUENT_USER.format(
            intro=persona.one_line_intro(),
            about="\n".join(about_lines) or "  (no additional demographics)",
            persona_block=persona_block or "(no extended persona)",
            bill_name=bill_name,
            bill_text=_clip(bill_text, 4000),
        )

    def vote(self, persona: Persona, bill_name: str, bill_text: str) -> ConstituentVote:
        raw = self.llm.call(
            self._build_system_prompt(),
            self._build_user_message(persona=persona, bill_name=bill_name, bill_text=bill_text),
            temperature=self.temperature,
        )
        vote_text = (_extract_tag(raw, "vote") or "unsure").strip().lower()
        if vote_text not in {"yes", "no", "unsure"}:
            vote_text = "unsure"
        return ConstituentVote(
            persona_uuid=persona.uuid,
            state=persona.state,
            vote=vote_text,
            reasoning=_extract_tag(raw, "reasoning") or "",
            key_concern=_extract_tag(raw, "key_concern") or "",
        )


def _clip(text: str | None, n: int) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= n:
        return text
    return text[:n].rsplit(" ", 1)[0] + "…"


def _extract_tag(text: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None
