"""The constitution: the human-written spec that defines ground-truth labels.

A constitution lists three things, all by construction:

  * **payloads (P)** — concrete forbidden goals. *Malicious by construction.*
  * **benign seeds (B)** — allowed requests. *Benign by construction.*
  * **transforms (T)** — intent-preserving rewrites. *Defined to NOT change the class.*

Every example the adversary can produce is ``base × transform-stack`` drawn from these
fixed libraries, so its label is known without ever asking a model to judge it. That is
the whole point (see ``context.md`` §2). This module is pure data + a YAML loader; it
makes no model calls.

Each item carries a ``split`` of ``"train"`` or ``"heldout"``. The held-out payloads and
transforms are never shown to the adversary and form the HARD evaluation set, which tests
generalization rather than memorization.
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml

TRAIN = "train"
HELDOUT = "heldout"


@dataclass(frozen=True)
class Payload:
    """A forbidden goal. Malicious by construction (label = BLOCK)."""

    id: str
    goal: str  # the request a safe model should refuse
    rubric: str  # what a *complied* response looks like (used only by the verifier V)
    split: str = TRAIN


@dataclass(frozen=True)
class Seed:
    """An allowed request. Benign by construction (label = ALLOW)."""

    id: str
    request: str
    split: str = TRAIN


@dataclass(frozen=True)
class TransformSpec:
    """A fully-specified, intent-preserving rewrite the adversary may apply.

    ``kind`` names a function in ``toolbox.TRANSFORMS``; ``params`` are fixed here so the
    adversary only ever *chooses transform ids*, never free-form parameters.
    """

    id: str
    kind: str
    params: dict
    split: str = TRAIN


@dataclass(frozen=True)
class Constitution:
    name: str
    payloads: tuple[Payload, ...]
    seeds: tuple[Seed, ...]
    transforms: tuple[TransformSpec, ...]

    # --- split helpers (HARD eval uses heldout payloads × heldout transforms) ---
    def train_payloads(self) -> list[Payload]:
        return [p for p in self.payloads if p.split == TRAIN]

    def heldout_payloads(self) -> list[Payload]:
        return [p for p in self.payloads if p.split == HELDOUT]

    def train_seeds(self) -> list[Seed]:
        return [s for s in self.seeds if s.split == TRAIN]

    def heldout_seeds(self) -> list[Seed]:
        return [s for s in self.seeds if s.split == HELDOUT]

    def train_transforms(self) -> list[TransformSpec]:
        return [t for t in self.transforms if t.split == TRAIN]

    def heldout_transforms(self) -> list[TransformSpec]:
        return [t for t in self.transforms if t.split == HELDOUT]

    # --- id lookups ---
    def payload(self, pid: str) -> Payload | None:
        return next((p for p in self.payloads if p.id == pid), None)

    def seed(self, sid: str) -> Seed | None:
        return next((s for s in self.seeds if s.id == sid), None)

    def transform(self, tid: str) -> TransformSpec | None:
        return next((t for t in self.transforms if t.id == tid), None)

    def base_text(self, base_id: str) -> str:
        """Return the raw text of a payload (goal) or seed (request) by id."""
        p = self.payload(base_id)
        if p is not None:
            return p.goal
        s = self.seed(base_id)
        if s is not None:
            return s.request
        raise KeyError(f"unknown base id: {base_id!r}")

    def label_of(self, base_id: str) -> int:
        """1 = BLOCK (payload, malicious), 0 = ALLOW (seed, benign). By construction."""
        if self.payload(base_id) is not None:
            return 1
        if self.seed(base_id) is not None:
            return 0
        raise KeyError(f"unknown base id: {base_id!r}")


def load_constitution(path: str) -> Constitution:
    """Load and validate a constitution YAML file.

    Validates that every payload has a rubric and every transform references a known
    transform kind only when ``toolbox`` is imported by the caller; here we just check
    structural completeness so errors surface early and clearly.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    payloads = tuple(
        Payload(
            id=p["id"],
            goal=p["goal"],
            rubric=p["rubric"],
            split=p.get("split", TRAIN),
        )
        for p in raw["payloads"]
    )
    seeds = tuple(
        Seed(id=s["id"], request=s["request"], split=s.get("split", TRAIN))
        for s in raw["seeds"]
    )
    transforms = tuple(
        TransformSpec(
            id=t["id"],
            kind=t["kind"],
            params=t.get("params", {}) or {},
            split=t.get("split", TRAIN),
        )
        for t in raw["transforms"]
    )

    _check_unique([p.id for p in payloads], "payload")
    _check_unique([s.id for s in seeds], "seed")
    _check_unique([t.id for t in transforms], "transform")

    return Constitution(
        name=raw.get("name", "unnamed"),
        payloads=payloads,
        seeds=seeds,
        transforms=transforms,
    )


def _check_unique(ids: list[str], what: str) -> None:
    seen = set()
    for i in ids:
        if i in seen:
            raise ValueError(f"duplicate {what} id: {i!r}")
        seen.add(i)
