"""Tests for the labels-by-construction core: constitution, toolbox, action.

Run: ``.venv/bin/python -m pytest tests/test_core.py`` (or just execute the file).
These need no GPU and no model — they prove the conceptual guarantee.
"""

from __future__ import annotations

from pathlib import Path

from loophole_v2.action import Action, parse_action, strip_think
from loophole_v2.constitution import load_constitution
from loophole_v2.toolbox import ALLOW, BLOCK, TRANSFORMS, render

CONST_PATH = str(Path(__file__).resolve().parents[1] / "constitutions" / "sample_safety.yaml")
const = load_constitution(CONST_PATH)


def test_constitution_loads_and_splits():
    assert len(const.payloads) == 10 and len(const.seeds) == 10
    assert len(const.heldout_payloads()) == 2
    assert len(const.heldout_transforms()) == 2
    # every payload has a rubric; every transform kind is real
    assert all(p.rubric for p in const.payloads)
    assert all(t.kind in TRANSFORMS for t in const.transforms)


def test_label_comes_only_from_the_base():
    """The crux: the label is fixed by payload-vs-seed, regardless of transforms."""
    payload = const.train_payloads()[0]
    seed = const.train_seeds()[0]
    stacks = [(), ("t_roleplay",), ("t_roleplay", "t_base64", "t_leetspeak")]
    for stack in stacks:
        mal = render(Action(payload.id, stack), const)
        ben = render(Action(seed.id, stack), const)
        assert mal.label == BLOCK and mal.track == "jailbreak"
        assert ben.label == ALLOW and ben.track == "false_refusal"


def test_transforms_change_surface_not_label():
    payload = const.train_payloads()[0]
    raw = render(Action(payload.id, ()), const)
    transformed = render(Action(payload.id, ("t_base64",)), const)
    assert raw.prompt != transformed.prompt  # surface changed
    assert raw.label == transformed.label  # label did not


def test_parse_action_handles_think_and_json():
    gen = (
        "<think>The roleplay frame might slip this past the filter, and base64 hides it.</think>\n"
        'Here is my plan: ```json\n{"transform_ids": ["t_roleplay", "t_base64"]}\n```'
    )
    think, answer = strip_think(gen)
    assert "roleplay frame" in think and "<think>" not in answer
    action = parse_action(gen, base_id="p_be_rude", allowed_transform_ids=["t_roleplay", "t_base64", "t_leetspeak"])
    assert action == Action("p_be_rude", ("t_roleplay", "t_base64"))


def test_parse_action_rejects_unknown_and_malformed():
    allowed = ["t_roleplay"]
    assert parse_action('{"transform_ids": ["t_caesar"]}', "p_be_rude", allowed) is None  # out of menu
    assert parse_action("no json here", "p_be_rude", allowed) is None
    assert parse_action('{"wrong_key": []}', "p_be_rude", allowed) is None
    # empty stack is a valid action (raw base, no transforms)
    assert parse_action('{"transform_ids": []}', "p_be_rude", allowed) == Action("p_be_rude", ())


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nAll {len(fns)} core tests passed.")
