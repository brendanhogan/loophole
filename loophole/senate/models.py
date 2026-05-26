from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Party(str, Enum):
    DEMOCRAT = "D"
    REPUBLICAN = "R"
    INDEPENDENT = "I"


class VoteStance(str, Enum):
    YES = "yes"
    LEAN_YES = "lean_yes"
    UNDECIDED = "undecided"
    LEAN_NO = "lean_no"
    NO = "no"


class Senator(BaseModel):
    """Identity record for a sitting US senator."""

    full_name: str
    short_name: str  # last name, used for display
    party: Party
    state: str  # 2-letter postal code
    senate_class: int | None = None  # 1, 2, or 3
    role: str | None = None  # e.g. "Majority Leader"
    caucus: Party | None = None  # for independents, who they caucus with
    bioguide_id: str | None = None  # Library of Congress ID
    photo_url: str | None = None  # official portrait, theunitedstates.io CDN


class Citation(BaseModel):
    """A grounding reference for a claim in the moral constitution."""

    claim: str  # the specific assertion this supports
    source_kind: str  # "vote" | "speech" | "interview" | "bill_sponsored" | "public_statement"
    detail: str  # short factual anchor, e.g. "Voted NO on Inflation Reduction Act, Aug 2022"


class IssueStance(BaseModel):
    issue: str
    stance: str  # one paragraph in the senator's voice


class MoralConstitution(BaseModel):
    """Structured persona derived from a senator's public record."""

    senator_full_name: str
    core_values: list[str] = Field(default_factory=list)  # 3-5 short statements
    top_issues: list[IssueStance] = Field(default_factory=list)
    red_lines: list[str] = Field(default_factory=list)  # non-negotiable positions
    negotiation_style: str = ""  # one paragraph
    typical_allies: list[str] = Field(default_factory=list)  # other senator short_names
    voice_notes: str = ""  # rhetorical style, how they argue
    citations: list[Citation] = Field(default_factory=list)
    confidence: str = "medium"  # "high" | "medium" | "low" — biographer's self-assessment
    generated_at: datetime = Field(default_factory=datetime.now)


class Amendment(BaseModel):
    description: str  # what change they want
    rationale: str  # why, in their voice
    would_flip_vote: bool  # would adopting this move them toward yes?


class BillReaction(BaseModel):
    """One senator's reaction to a specific bill."""

    senator_full_name: str
    vote: VoteStance
    confidence: float = 0.5  # 0..1
    reasoning: str  # 1 paragraph in their voice
    key_provisions_supported: list[str] = Field(default_factory=list)
    key_provisions_opposed: list[str] = Field(default_factory=list)
    amendments: list[Amendment] = Field(default_factory=list)


class SenateSession(BaseModel):
    session_id: str
    bill_name: str
    bill_text: str
    bill_summary: str | None = None
    reactions: list[BillReaction] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)

    @property
    def tally(self) -> dict[str, int]:
        counts: dict[str, int] = {v.value: 0 for v in VoteStance}
        for r in self.reactions:
            counts[r.vote.value] += 1
        return counts


# ---------------------------------------------------------------------------
# Iteration mode — bill drafting + tenet-constrained revision loop
# ---------------------------------------------------------------------------


class Bill(BaseModel):
    """A versioned bill that gets revised across iteration rounds."""

    name: str  # e.g. "Sunshine Protection Act of 2026"
    text: str
    version: int = 1
    summary: str = ""  # 1-2 sentence summary for the UI


class CoreTenets(BaseModel):
    """Immutable goals extracted from the bill — amendments cannot violate these."""

    tenets: list[str] = Field(default_factory=list)
    extracted_from_version: int = 1


class ProposedAmendment(BaseModel):
    """An amendment proposed by one or more senators, surfaced for the reviser."""

    description: str
    rationale: str
    proposed_by: list[str] = Field(default_factory=list)  # senator short_names
    flip_count: int = 0  # how many senators said this would flip their vote


class AppliedAmendment(BaseModel):
    description: str
    how_applied: str  # 1-2 sentences on how the reviser incorporated it


class RejectedAmendment(BaseModel):
    description: str
    violates_tenet: str  # which tenet (verbatim)
    rationale: str  # why it violates


class IterationRound(BaseModel):
    """One pass through the chamber, plus the revision that produced the next bill."""

    round_number: int  # 0-indexed
    bill: Bill  # the bill voted on this round
    reactions: list[BillReaction] = Field(default_factory=list)
    applied: list[AppliedAmendment] = Field(default_factory=list)
    rejected: list[RejectedAmendment] = Field(default_factory=list)
    revision_summary: str = ""  # plain-language: what changed for next round
    # Hill-climb bookkeeping: was this round's bill accepted as the new best,
    # or did it regress and get reverted?
    outcome: str = "accepted"  # accepted | regressed | passed | stuck
    target_to_beat: int | None = None  # the best yes_total at the time this round was tried

    @property
    def tally(self) -> dict[str, int]:
        counts = {v.value: 0 for v in VoteStance}
        for r in self.reactions:
            counts[r.vote.value] += 1
        return counts

    @property
    def yes_total(self) -> int:
        t = self.tally
        return t["yes"] + t["lean_yes"]


class ConstituentVoteRecord(BaseModel):
    """One sampled constituent's vote — stored in IterationSession for the viz."""

    persona_uuid: str
    state: str
    vote: str  # "yes" | "no" | "unsure"
    reasoning: str = ""
    key_concern: str = ""
    persona_intro: str = ""  # short demographic summary for the viz


class StateMoodRecord(BaseModel):
    """Per-state aggregate of constituent votes, serializable form of StateMood."""

    state: str
    yes_count: int = 0
    no_count: int = 0
    unsure_count: int = 0
    label: str = "unpolled"  # strong support | lean support | split | lean oppose | strong oppose
    sample_voices: list[ConstituentVoteRecord] = Field(default_factory=list)

    @property
    def total(self) -> int:
        return self.yes_count + self.no_count + self.unsure_count


class IterationSession(BaseModel):
    session_id: str
    user_prompt: str  # the user's original plain-English idea
    tenets: CoreTenets
    rounds: list[IterationRound] = Field(default_factory=list)
    final_status: str = "in_progress"  # in_progress | passed | stuck | max_rounds | converged
    created_at: datetime = Field(default_factory=datetime.now)
    state_moods: list[StateMoodRecord] = Field(default_factory=list)
    constituents_polled: bool = False
    # Index into .rounds for the bill that achieved the highest yes_total.
    # None until at least one round has run. Used to surface the "best" bill
    # instead of just the last-tried one.
    best_round_idx: int | None = None

    @property
    def final_bill(self) -> Bill | None:
        if not self.rounds:
            return None
        if self.best_round_idx is not None and 0 <= self.best_round_idx < len(self.rounds):
            return self.rounds[self.best_round_idx].bill
        return self.rounds[-1].bill

    @property
    def best_round(self) -> IterationRound | None:
        if not self.rounds:
            return None
        if self.best_round_idx is not None and 0 <= self.best_round_idx < len(self.rounds):
            return self.rounds[self.best_round_idx]
        return max(self.rounds, key=lambda r: r.yes_total)

    @property
    def state_mood_by_state(self) -> dict[str, StateMoodRecord]:
        return {m.state: m for m in self.state_moods}
