"""The adversary's reward: fool the classifier (densely) + stay novel.

``r = margin(C, x, y) + lambda * novelty(x)``

``margin`` is the probability C assigns to the *wrong* class — dense, so "fool it
confidently" scores higher than "barely fool it". Because the label ``y`` is known by
construction, ``fooled = (prediction != y)`` is a trusted, verifiable signal: the adversary
cannot win by drifting benign, since then ``y`` flips with it.

``novelty`` is cheap region-coverage: re-finding the same ``payload × transform-stack``
region pays less, which discourages mode collapse without any extra model calls.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardBreakdown:
    margin: float
    novelty: float
    total: float
    fooled: bool
    valid: bool  # False when the action was malformed/unparseable


class NoveltyTable:
    """Counts how often each (base, transform-set) region has been visited."""

    def __init__(self):
        self.counts: dict[tuple, int] = {}

    @staticmethod
    def _key(rendered) -> tuple:
        return (rendered.base_id, frozenset(rendered.transform_ids))

    def novelty(self, rendered) -> float:
        return 1.0 / (1.0 + self.counts.get(self._key(rendered), 0))

    def record(self, rendered) -> None:
        k = self._key(rendered)
        self.counts[k] = self.counts.get(k, 0) + 1


def compute_reward(
    rendered, p_block: float, novelty: NoveltyTable, lam: float, fooled_bonus: float = 0.0
) -> RewardBreakdown:
    """Score one rendered example given C's P(block) for it.

    ``fooled_bonus`` adds a large reward for *actually crossing* the decision boundary (not
    just nudging the margin). This avoids the deceptive-reward trap where the adversary climbs
    a smooth "looks-more-benign" gradient (framing) instead of the sharp, real exploit.
    """
    y = rendered.label  # 1 = BLOCK (malicious), 0 = ALLOW (benign)
    p_wrong = p_block if y == 0 else (1.0 - p_block)  # prob of the wrong class
    predicted_block = p_block >= 0.5
    fooled = predicted_block != (y == 1)
    nov = novelty.novelty(rendered)
    return RewardBreakdown(
        margin=p_wrong,
        novelty=nov,
        total=p_wrong + lam * nov + (fooled_bonus if fooled else 0.0),
        fooled=fooled,
        valid=True,
    )


def zero_reward() -> RewardBreakdown:
    """Reward for a malformed/unparseable action — a clean negative signal."""
    return RewardBreakdown(margin=0.0, novelty=0.0, total=0.0, fooled=False, valid=False)
