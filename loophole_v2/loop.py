"""The outer min-max loop — the whole game on one screen.

  init:   train C0 on a constitution-seeded set;  D <- that set;  pop <- {C0}
  repeat:
    Phase A — adversary hunts C's blind spots (C frozen); mine hard examples (labels trusted)
    Gate    — sampled verifier drops malicious examples a transform accidentally neutered
    Phase B — D <- D + hard + fresh easy stream;  retrain C from base;  pop <- pop + {C}
    measure — fooled-rate, accuracy on general & hard eval sets, universality

Phase A is pluggable: a ``GreedyAdversary`` (no RL, milestone 4) or the GRPO policy
(milestone 5). Everything is logged for the blog.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .adversary import GreedyAdversary, sample_goals
from .classifier import Classifier
from .config import Config
from .constitution import load_constitution
from .eval import build_general_set, build_hard_set, evaluate
from .logging_jsonl import JsonlLogger
from .population import Population
from .replay_buffer import ReplayBuffer, example_from_render
from .reward import NoveltyTable, compute_reward
from .toolbox import render


def run(cfg: Config) -> str:
    rng = np.random.default_rng(cfg.seed)
    const = load_constitution(cfg.constitution_path)
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "ckpts").mkdir(exist_ok=True)
    logger = JsonlLogger(cfg.run_dir)
    _save_config(cfg, run_dir)

    # Fixed eval sets (built once so the curves are comparable across rounds).
    general = build_general_set(const, cfg.eval_n, np.random.default_rng(cfg.seed + 1), cfg.max_stack)
    hard = build_hard_set(const, cfg.eval_n, np.random.default_rng(cfg.seed + 2), cfg.max_stack)
    # A fixed set of points for the decision-boundary view (recolored by C every round).
    boundary_set = general + hard

    # --- init: seed the buffer and train C0 ---
    buffer = ReplayBuffer()
    buffer.add(buffer.render_easy_stream(const, cfg.easy_per_round, 0, rng))
    pop = Population(cfg.pop_size, cfg.device)
    classifier = _train_classifier(cfg, buffer, rng, logger, round_num=0)
    ckpt = classifier.save(str(run_dir / "ckpts" / "round_000"))
    pop.add(ckpt)
    _measure(classifier, general, hard, const, logger, round_num=0, fooled_rate=None)
    _log_boundary(classifier, boundary_set, logger, round_num=0, log_static=True)

    # The adversary. Greedy baseline by default; GRPO policy when cfg.use_grpo.
    adversary = None if cfg.use_grpo else GreedyAdversary(cfg.max_stack)
    if cfg.use_grpo:
        from .adversary import Adversary  # lazy: only load the 9B model when needed

        adversary = Adversary(cfg.adversary_model, device=cfg.device, lr=cfg.grpo_lr)
    novelty = NoveltyTable()

    for rnd in range(1, cfg.rounds + 1):
        # --- Phase A: mine hard examples against the population ---
        if cfg.use_grpo:
            from .grpo import train_phase_a

            hard_examples, fooled_rate = train_phase_a(
                cfg, adversary, const, pop, novelty, rng, logger, rnd
            )
        else:
            hard_examples, fooled_rate = _greedy_mine(
                cfg, adversary, const, pop, novelty, rng, logger, rnd
            )

        # --- Gate (M7): sampled verifier drops neutered malicious examples ---
        if cfg.verifier_model and cfg.verifier_sample_rate > 0:
            from .verifier import gate_hard_examples

            hard_examples = gate_hard_examples(cfg, hard_examples, const, rng, logger, rnd)

        # --- Phase B: grow the buffer and retrain C from base ---
        buffer.add(hard_examples)
        buffer.add(buffer.render_easy_stream(const, cfg.easy_per_round, rnd, rng))
        del classifier
        _free_gpu()
        classifier = _train_classifier(cfg, buffer, rng, logger, round_num=rnd)
        ckpt = classifier.save(str(run_dir / "ckpts" / f"round_{rnd:03d}"))
        pop.add(ckpt)

        _measure(classifier, general, hard, const, logger, rnd, fooled_rate)
        _log_boundary(classifier, boundary_set, logger, rnd, log_static=False)

    logger.close()
    print(f"\nDone. Run dir: {run_dir}")
    return str(run_dir)


def _greedy_mine(cfg, adversary, const, pop, novelty, rng, logger, rnd):
    """Phase A without RL: sample random stacks, keep the ones that fool C."""
    classifier = pop.latest()
    goals = sample_goals(const, cfg.goals_per_round, rng)
    mined, total, fooled = [], 0, 0
    for goal in goals:
        actions = adversary.propose(goal, cfg.group_size, rng)
        rendered = [render(a, const) for a in actions]
        probs = classifier.predict_proba([r.prompt for r in rendered])
        for r, p in zip(rendered, probs):
            p_block = float(p[1])
            rb = compute_reward(r, p_block, novelty, cfg.novelty_lambda)
            novelty.record(r)
            total += 1
            fooled += int(rb.fooled)
            logger.log("rollouts", {
                "round": rnd, "phase": "A", "track": r.track,
                "base_id": r.base_id, "transform_ids": list(r.transform_ids),
                "true_label": r.label, "p_block": p_block,
                "predicted_label": int(p_block >= 0.5), "fooled": rb.fooled,
                "reward": {"margin": rb.margin, "novelty": rb.novelty, "total": rb.total},
            })
            if rb.fooled:
                mined.append(example_from_render(r, "hard", rnd))
    return mined, fooled / max(total, 1)


def _train_classifier(cfg, buffer, rng, logger, round_num) -> Classifier:
    """Retrain C from the base checkpoint on a class-balanced buffer sample."""
    texts, labels = buffer.sample_balanced(cfg.train_sample, rng)
    classifier = Classifier(cfg.classifier_model, device=cfg.device)
    metrics = classifier.train(
        texts, labels, epochs=cfg.c_epochs, lr=cfg.c_lr, batch_size=cfg.c_batch_size
    )
    counts = buffer.counts()
    logger.log("classifier_train", {"round": round_num, **metrics, **counts})
    return classifier


def _measure(classifier, general, hard, const, logger, round_num, fooled_rate):
    res = evaluate(classifier, general, hard, const, round_num)
    record = {
        "round": round_num, "fooled_rate": fooled_rate,
        "general_acc": res.general_acc, "general_fpr": res.general_fpr,
        "general_miss": res.general_miss, "hard_acc": res.hard_acc,
        "hard_fpr": res.hard_fpr, "hard_miss": res.hard_miss,
        "max_universality": res.max_universality, "universality": res.universality,
    }
    logger.log("rounds", record)
    logger.log("eval", record)
    fr = "  init" if fooled_rate is None else f"{fooled_rate:5.2f}"
    print(
        f"round {round_num:2d} | fooled {fr} | gen_acc {res.general_acc:.3f} "
        f"| hard_acc {res.hard_acc:.3f} | gen_fpr {res.general_fpr:.3f} "
        f"| max_univ {res.max_universality:.3f}"
    )


def _log_boundary(classifier, boundary_set, logger, round_num, log_static):
    """Log C's P(block) for a fixed point set, so the report can show the boundary moving.

    The 2D layout (computed at report time) is fixed; only these probabilities change per
    round, which is exactly the decision surface shifting over a stationary map.
    """
    probs = classifier.predict_proba([e.prompt for e in boundary_set])
    for i, e in enumerate(boundary_set):
        if log_static:
            logger.log("boundary_points", {
                "idx": i, "prompt": e.prompt, "true_label": e.label,
                "track": e.track, "base_id": e.base_id, "transform_ids": list(e.transform_ids),
            })
        logger.log("boundary", {"round": round_num, "idx": i, "p_block": float(probs[i][1])})


def _save_config(cfg, run_dir):
    from dataclasses import asdict

    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)


def _free_gpu():
    import gc

    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
