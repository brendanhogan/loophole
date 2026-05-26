from __future__ import annotations

import re
from typing import Any

from loophole.agents.base import BaseAgent
from loophole.senate.models import (
    AppliedAmendment,
    Bill,
    BillReaction,
    CoreTenets,
    ProposedAmendment,
    RejectedAmendment,
    VoteStance,
)
from loophole.senate.prompts import REVISER_SYSTEM, REVISER_USER


class Reviser(BaseAgent):
    """Given (bill, tenets, reactions), produce a revised bill plus applied/rejected lists."""

    def _build_system_prompt(self, **kwargs: Any) -> str:
        return REVISER_SYSTEM

    def _build_user_message(self, state: Any = None, **kwargs: Any) -> str:
        return REVISER_USER.format(
            bill_name=kwargs["bill"].name,
            version=kwargs["bill"].version,
            bill_text=kwargs["bill"].text,
            tenets=_format_tenets(kwargs["tenets"]),
            amendments=_format_amendments(kwargs["amendments"]),
            load_bearing=_format_load_bearing(kwargs.get("load_bearing") or []),
        )

    def revise(
        self,
        bill: Bill,
        tenets: CoreTenets,
        amendments: list[ProposedAmendment],
        load_bearing: list[tuple[str, int]] | None = None,
    ) -> tuple[Bill, list[AppliedAmendment], list[RejectedAmendment], str]:
        raw = self.llm.call(
            self._build_system_prompt(),
            self._build_user_message(
                bill=bill,
                tenets=tenets,
                amendments=amendments,
                load_bearing=load_bearing or [],
            ),
            temperature=self.temperature,
        )

        applied: list[AppliedAmendment] = []
        for m in re.finditer(r"<amendment>(.*?)</amendment>", _extract_tag(raw, "applied") or "", re.DOTALL):
            block = m.group(1)
            desc = _extract_tag(block, "description") or ""
            how = _extract_tag(block, "how_applied") or ""
            if desc:
                applied.append(AppliedAmendment(description=desc, how_applied=how))

        rejected: list[RejectedAmendment] = []
        for m in re.finditer(r"<amendment>(.*?)</amendment>", _extract_tag(raw, "rejected") or "", re.DOTALL):
            block = m.group(1)
            desc = _extract_tag(block, "description") or ""
            tenet = _extract_tag(block, "violates_tenet") or ""
            rationale = _extract_tag(block, "rationale") or ""
            if desc:
                rejected.append(
                    RejectedAmendment(description=desc, violates_tenet=tenet, rationale=rationale)
                )

        revision_summary = _extract_tag(raw, "revision_summary") or ""
        revised_text = _extract_tag(raw, "revised_bill") or bill.text

        # If nothing applied, keep the bill text as-is (the model may still hand back
        # the original text inside <revised_bill>; treating it as unchanged is fine).
        new_version = bill.version + 1 if applied else bill.version
        revised = Bill(
            name=bill.name,
            text=revised_text,
            version=new_version,
            summary=bill.summary,
        )
        return revised, applied, rejected, revision_summary


def cluster_flip_amendments(reactions: list[BillReaction]) -> list[ProposedAmendment]:
    """Group senator-proposed amendments by similar description.

    Uses a simple normalized first-N-words key as the clustering signal.
    Good enough for our purposes — if two senators independently word the
    same demand slightly differently, the LLM Reviser will recognize the
    overlap and treat them as one. The clustering here is mostly to compute
    flip_count and to surface the most-demanded amendments first.
    """

    def key(s: str) -> str:
        words = re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()
        return " ".join(words[:5])

    bucket: dict[str, ProposedAmendment] = {}
    for r in reactions:
        for a in r.amendments:
            if not a.would_flip_vote:
                continue
            k = key(a.description)
            short_name = r.senator_full_name.split()[-1]
            if k in bucket:
                bucket[k].proposed_by.append(short_name)
                bucket[k].flip_count += 1
            else:
                bucket[k] = ProposedAmendment(
                    description=a.description,
                    rationale=a.rationale,
                    proposed_by=[short_name],
                    flip_count=1,
                )
    return sorted(bucket.values(), key=lambda p: -p.flip_count)


def cluster_load_bearing_provisions(reactions: list[BillReaction]) -> list[tuple[str, int]]:
    """Aggregate what current yes-voters cite as the reasons they're yes.

    Returns a list of (provision_description, supporter_count) pairs, ordered
    by count desc. The reviser uses these as a "do not erode" signal — if
    half the chamber's yes-vote rests on a particular provision, the next
    revision shouldn't quietly water it down to placate the no's.
    """

    def key(s: str) -> str:
        words = re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()
        return " ".join(words[:6])

    counts: dict[str, int] = {}
    canonical: dict[str, str] = {}  # key → first-seen full text
    for r in reactions:
        if r.vote not in (VoteStance.YES, VoteStance.LEAN_YES):
            continue
        for prov in r.key_provisions_supported:
            k = key(prov)
            if not k:
                continue
            counts[k] = counts.get(k, 0) + 1
            canonical.setdefault(k, prov)

    items = [(canonical[k], counts[k]) for k in counts]
    items.sort(key=lambda kv: -kv[1])
    return items


def _format_tenets(tenets: CoreTenets) -> str:
    if not tenets.tenets:
        return "(none specified)"
    return "\n".join(f"  {i+1}. {t}" for i, t in enumerate(tenets.tenets))


def _format_amendments(amendments: list[ProposedAmendment]) -> str:
    if not amendments:
        return "(no amendments proposed)"
    parts = []
    for i, a in enumerate(amendments, 1):
        proposers = ", ".join(a.proposed_by[:5])
        more = f" + {len(a.proposed_by)-5} more" if len(a.proposed_by) > 5 else ""
        parts.append(
            f"  [{i}] (would flip {a.flip_count} senators: {proposers}{more})\n"
            f"      {a.description}\n"
            f"      Rationale: {a.rationale}"
        )
    return "\n\n".join(parts)


def _format_load_bearing(items: list[tuple[str, int]]) -> str:
    if not items:
        return "(none — this is the first round, or no current yes-voters cited specific provisions)"
    # Cap to top 10 to keep the prompt compact
    parts = []
    for i, (provision, count) in enumerate(items[:10], 1):
        parts.append(f"  [{i}] ({count} yes-voter{'s' if count != 1 else ''}) {provision}")
    return "\n".join(parts)


def _extract_tag(text: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None
