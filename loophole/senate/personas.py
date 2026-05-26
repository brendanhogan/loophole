"""Constituent persona handling — load, sample, and aggregate Nemotron-USA personas."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

PERSONAS_PATH = Path(__file__).parent / "data" / "personas_pool.jsonl"


class Persona(BaseModel):
    uuid: str
    state: str
    sex: str | None = None
    age: int | None = None
    marital_status: str | None = None
    education_level: str | None = None
    bachelors_field: str | None = None
    occupation: str | None = None
    city: str | None = None
    zipcode: str | None = None
    professional_persona: str | None = None
    persona: str | None = None
    cultural_background: str | None = None
    career_goals_and_ambitions: str | None = None
    hobbies_and_interests: str | None = None

    def one_line_intro(self) -> str:
        bits = []
        if self.age:
            bits.append(f"{self.age}-year-old")
        if self.sex:
            bits.append(self.sex.lower())
        if self.occupation:
            bits.append(self.occupation.replace("_", " "))
        if self.city and self.state:
            bits.append(f"in {self.city}, {self.state}")
        elif self.state:
            bits.append(f"in {self.state}")
        return " ".join(bits) if bits else f"resident of {self.state}"


class ConstituentVote(BaseModel):
    persona_uuid: str
    state: str
    vote: str  # "yes" | "no" | "unsure"
    reasoning: str  # 1-2 sentences in the persona's voice
    key_concern: str = ""  # short, optional summary of their primary concern


class StateMood(BaseModel):
    state: str
    yes_count: int = 0
    no_count: int = 0
    unsure_count: int = 0
    sample_voices: list[ConstituentVote] = Field(default_factory=list)

    @property
    def total(self) -> int:
        return self.yes_count + self.no_count + self.unsure_count

    @property
    def yes_pct(self) -> float:
        return (self.yes_count / self.total) if self.total else 0.0

    @property
    def no_pct(self) -> float:
        return (self.no_count / self.total) if self.total else 0.0

    @property
    def label(self) -> str:
        if self.total == 0:
            return "unpolled"
        # If most respondents are unsure, label that way honestly
        unsure_pct = self.unsure_count / self.total
        if unsure_pct >= 0.5:
            return "no clear opinion"
        # Otherwise, weight by share among the decided respondents
        decided = self.yes_count + self.no_count
        if decided == 0:
            return "no clear opinion"
        yes_share = self.yes_count / decided
        if yes_share >= 0.7:
            return "strong support"
        if yes_share >= 0.55:
            return "lean support"
        if yes_share >= 0.45:
            return "split"
        if yes_share >= 0.30:
            return "lean oppose"
        return "strong oppose"


def load_persona_pool(path: Path | str | None = None) -> dict[str, list[Persona]]:
    """Load every persona from the JSONL pool, grouped by state."""
    p = Path(path) if path else PERSONAS_PATH
    pool: dict[str, list[Persona]] = defaultdict(list)
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            try:
                persona = Persona(**data)
            except Exception:
                continue
            pool[persona.state].append(persona)
    return dict(pool)


def sample_personas(
    pool: dict[str, list[Persona]],
    per_state: int = 10,
    seed: int = 42,
) -> dict[str, list[Persona]]:
    """Deterministic per-state sample of N personas."""
    rng = random.Random(seed)
    out: dict[str, list[Persona]] = {}
    for state, personas in pool.items():
        if len(personas) <= per_state:
            out[state] = personas[:]
        else:
            out[state] = rng.sample(personas, per_state)
    return out


def aggregate_state_mood(votes: list[ConstituentVote]) -> dict[str, StateMood]:
    """Group constituent votes into per-state mood snapshots."""
    by_state: dict[str, StateMood] = {}
    for v in votes:
        mood = by_state.setdefault(v.state, StateMood(state=v.state))
        if v.vote == "yes":
            mood.yes_count += 1
        elif v.vote == "no":
            mood.no_count += 1
        else:
            mood.unsure_count += 1
        if len(mood.sample_voices) < 3:
            mood.sample_voices.append(v)
    return by_state
