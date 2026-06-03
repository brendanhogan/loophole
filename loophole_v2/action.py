"""The adversary's action and the parser that extracts it from model output.

The adversary is assigned a *base* (a specific payload or seed) and must choose an
ordered **stack of transform ids** to apply. That choice — and only that choice — is its
action. Keeping the action to "pick ids from a fixed menu" is what preserves the
labels-by-construction guarantee: no free-form model text ever enters the rendered prompt.

Qwen3 models emit a ``<think>...</think>`` reasoning block followed by the answer. We keep
the full generation for logging (it shows the model's strategy), but for *rendering* we
only need the final JSON action: ``{"transform_ids": ["t1", "t3"]}``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


@dataclass(frozen=True)
class Action:
    base_id: str
    transform_ids: tuple[str, ...]


def strip_think(generation: str) -> tuple[str, str]:
    """Split a Qwen3 generation into (think_text, answer_text)."""
    think = " ".join(m.group(0) for m in _THINK_RE.finditer(generation))
    answer = _THINK_RE.sub("", generation).strip()
    # Also handle an unclosed <think> (truncated generation): drop everything up to it.
    if "<think>" in answer:
        answer = answer.split("<think>")[0].strip()
    return think.strip(), answer


def parse_action(
    generation: str, base_id: str, allowed_transform_ids: list[str]
) -> Action | None:
    """Parse the last JSON object in a generation into a validated Action.

    Returns ``None`` on any failure (no JSON, malformed, or an unknown/disallowed
    transform id). A ``None`` action earns reward 0 in the loop — a clean signal that
    teaches the policy to emit well-formed, in-menu actions.
    """
    _, answer = strip_think(generation)
    allowed = set(allowed_transform_ids)

    # Take the LAST JSON-looking block; models often restate it after reasoning.
    blocks = _JSON_BLOCK_RE.findall(answer)
    for block in reversed(blocks):
        try:
            obj = json.loads(block)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "transform_ids" not in obj:
            continue
        ids = obj["transform_ids"]
        if not isinstance(ids, list) or not all(isinstance(t, str) for t in ids):
            continue
        if any(t not in allowed for t in ids):
            return None  # referenced a transform outside the assigned menu
        return Action(base_id=base_id, transform_ids=tuple(ids))
    return None
