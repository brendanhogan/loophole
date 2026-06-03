"""The replay buffer: C's accumulated, class-balanced training set.

Two kinds of examples flow in:
  * **hard** — boundary cases mined by the adversary (where C was wrong). Never evicted, so
    C cannot forget old attacks.
  * **easy** — a fresh stream of random ``base × short-stack`` renders each round, giving C the
    bulk distribution and keeping the classes balanced.

Labels are always set by ``toolbox.render`` (construction), never by a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .action import Action
from .constitution import Constitution
from .toolbox import ALLOW, BLOCK, render


@dataclass
class Example:
    prompt: str
    label: int  # ALLOW (0) or BLOCK (1)
    source: str  # "seed" | "easy" | "hard"
    round_added: int
    base_id: str
    transform_ids: tuple[str, ...]
    track: str


class ReplayBuffer:
    def __init__(self):
        self.examples: list[Example] = []

    def __len__(self) -> int:
        return len(self.examples)

    def add(self, examples: list[Example]) -> None:
        self.examples.extend(examples)

    def counts(self) -> dict:
        n_block = sum(1 for e in self.examples if e.label == BLOCK)
        return {
            "total": len(self.examples),
            "block": n_block,
            "allow": len(self.examples) - n_block,
            "hard": sum(1 for e in self.examples if e.source == "hard"),
        }

    def render_easy_stream(
        self, const: Constitution, n: int, round_num: int, rng, max_stack: int = 2
    ) -> list[Example]:
        """Render ``n`` random in-distribution examples (both classes, short stacks)."""
        bases = [(p.id, "payload") for p in const.train_payloads()] + [
            (s.id, "seed") for s in const.train_seeds()
        ]
        transforms = [t.id for t in const.train_transforms()]
        out = []
        for _ in range(n):
            base_id, _ = bases[rng.integers(len(bases))]
            depth = int(rng.integers(0, max_stack + 1))
            stack = tuple(transforms[i] for i in rng.choice(len(transforms), size=depth, replace=False)) if depth else ()
            r = render(Action(base_id, stack), const)
            out.append(_to_example(r, "easy", round_num))
        return out

    def sample_balanced(self, n: int, rng) -> tuple[list[str], list[int]]:
        """Sample up to ``n`` examples forced to a 50/50 ALLOW/BLOCK split."""
        block = [e for e in self.examples if e.label == BLOCK]
        allow = [e for e in self.examples if e.label == ALLOW]
        if not block or not allow:
            chosen = list(self.examples)
        else:
            per = n // 2
            chosen = _sample(block, per, rng) + _sample(allow, per, rng)
        rng.shuffle(chosen)
        return [e.prompt for e in chosen], [e.label for e in chosen]


def example_from_render(rendered, source: str, round_num: int) -> Example:
    """Public helper so the loop can turn mined renders into buffer examples."""
    return _to_example(rendered, source, round_num)


def _to_example(rendered, source: str, round_num: int) -> Example:
    return Example(
        prompt=rendered.prompt,
        label=rendered.label,
        source=source,
        round_added=round_num,
        base_id=rendered.base_id,
        transform_ids=rendered.transform_ids,
        track=rendered.track,
    )


def _sample(pool: list[Example], k: int, rng) -> list[Example]:
    if k <= 0 or not pool:
        return []
    replace = k > len(pool)
    idx = rng.choice(len(pool), size=k, replace=replace)
    return [pool[i] for i in idx]
