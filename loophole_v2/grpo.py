"""Hand-rolled minimal GRPO — Phase A of the loop.

Modeled on DeepSeekRL-Extended (group-relative advantages, a REINFORCE-with-ratio loss, an
optional KL to a frozen reference). The reference here is free: disabling the LoRA adapter
turns the policy back into the base model, so we read its logps without a second copy.

The trajectory we learn from is the whole completion — the Qwen ``<think>`` reasoning plus
the final JSON action — so the policy is rewarded for whatever token sequence produced a
classifier-fooling action.
"""

from __future__ import annotations

import numpy as np
import torch

from .adversary import Group, sample_goals
from .replay_buffer import example_from_render
from .reward import compute_reward, zero_reward


def selective_log_softmax(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Log-prob of the chosen token per position, without materializing a full softmax."""
    gathered = torch.gather(logits, -1, index.unsqueeze(-1)).squeeze(-1)
    return gathered - torch.logsumexp(logits, dim=-1)


def completion_logps(model, input_ids, attention_mask, prompt_len: int) -> torch.Tensor:
    """Per-token log-probs over just the completion tokens, shape (group, comp_len)."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    # Token at position j is predicted by the logits at position j-1; completion tokens live
    # at positions [prompt_len, T), so we need logits at [prompt_len-1, T-1).
    comp_logits = logits[:, prompt_len - 1 : -1, :].float()
    targets = input_ids[:, prompt_len:]
    return selective_log_softmax(comp_logits, targets)


def group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Group-relative baseline: normalize rewards within the group (no critic)."""
    return (rewards - rewards.mean()) / (rewards.std() + 1e-4)


def grpo_group_loss(adversary, group: Group, rewards: torch.Tensor, cfg) -> tuple[torch.Tensor, dict]:
    """The GRPO loss for one goal's group of rollouts."""
    device = adversary.device
    rollouts = group.rollouts
    lengths = torch.tensor([r.length for r in rollouts], device=device)
    comp_ids = torch.stack([r.completion_ids for r in rollouts]).to(device)  # (G, L_c)
    g, comp_len = comp_ids.shape
    comp_mask = (torch.arange(comp_len, device=device)[None, :] < lengths[:, None]).float()

    prompt = group.prompt_ids.repeat(g, 1)  # (G, P)
    prompt_len = prompt.shape[1]
    input_ids = torch.cat([prompt, comp_ids], dim=1)
    attn = torch.cat([torch.ones_like(prompt), comp_mask.long()], dim=1)

    advantages = group_advantages(rewards).unsqueeze(1)  # (G, 1)
    logps = completion_logps(adversary.model, input_ids, attn, prompt_len)

    if cfg.grpo_kl_beta > 0:
        with torch.no_grad(), adversary.model.disable_adapter():
            ref_logps = completion_logps(adversary.model, input_ids, attn, prompt_len)
        kl = torch.exp(ref_logps - logps) - (ref_logps - logps) - 1.0
    else:
        kl = torch.zeros_like(logps)

    ratio = torch.exp(logps - logps.detach())  # == 1 at the sampling point; carries the grad
    per_token = ratio * advantages - cfg.grpo_kl_beta * kl
    loss = -((per_token * comp_mask).sum(1) / comp_mask.sum(1).clamp(min=1)).mean()

    kl_mean = float((kl * comp_mask).sum() / comp_mask.sum().clamp(min=1))
    return loss, {"kl": kl_mean}


def score_group(group: Group, classifier, const, novelty, cfg, rnd, logger, mined) -> tuple[list, int, int]:
    """Reward every rollout in a group, log it, and collect fooling examples as hard data."""
    valid_idx = [i for i, r in enumerate(group.rollouts) if r.rendered is not None]
    probs = (
        classifier.predict_proba([group.rollouts[i].rendered.prompt for i in valid_idx])
        if valid_idx
        else np.zeros((0, 2))
    )
    p_block = {i: float(probs[j][1]) for j, i in enumerate(valid_idx)}

    rewards, fooled, parsed = [0.0] * len(group.rollouts), 0, 0
    for i, r in enumerate(group.rollouts):
        if r.rendered is None:
            rb, pb = zero_reward(), None
        else:
            pb = p_block[i]
            rb = compute_reward(r.rendered, pb, novelty, cfg.novelty_lambda, cfg.reward_fooled_bonus)
            novelty.record(r.rendered)
            parsed += 1
            if rb.fooled:
                fooled += 1
                mined.append(example_from_render(r.rendered, "hard", rnd))
        rewards[i] = rb.total
        logger.log("rollouts", {
            "round": rnd, "phase": "A", "track": group.goal.track,
            "base_id": group.goal.base_id,
            "transform_ids": list(r.action.transform_ids) if r.action else None,
            "parse_ok": r.rendered is not None,
            "think_text": r.think_text, "completion_text": r.completion_text,
            "p_block": pb, "true_label": (r.rendered.label if r.rendered else None),
            "fooled": rb.fooled,
            "reward": {"margin": rb.margin, "novelty": rb.novelty, "total": rb.total},
        })
    return rewards, fooled, parsed


def train_phase_a(cfg, adversary, const, pop, novelty, rng, logger, rnd) -> tuple[list, float]:
    """A few GRPO steps against a sampled classifier from the population. Returns mined hard
    examples and the round's fooled-rate."""
    goals_per_step = max(1, cfg.goals_per_round // cfg.phase_a_steps)
    mined, total, fooled_total = [], 0, 0

    for step in range(cfg.phase_a_steps):
        classifier = pop.sample(rng)  # train against the population, not just the latest
        goals = sample_goals(const, goals_per_step, rng)
        adversary.optimizer.zero_grad()

        losses, kls, r_means, r_stds, parsed_total, step_fooled, step_n = [], [], [], [], 0, 0, 0
        for goal in goals:
            group = adversary.sample_group(
                goal, const, cfg.group_size, cfg.grpo_temperature, cfg.grpo_max_new_tokens
            )
            rewards, fooled, parsed = score_group(group, classifier, const, novelty, cfg, rnd, logger, mined)
            rt = torch.tensor(rewards, device=adversary.device, dtype=torch.float32)
            loss, st = grpo_group_loss(adversary, group, rt, cfg)
            (loss / len(goals)).backward()

            losses.append(loss.item())
            kls.append(st["kl"])
            r_means.append(float(rt.mean()))
            r_stds.append(float(rt.std()))
            parsed_total += parsed
            step_fooled += fooled
            step_n += len(group.rollouts)

        trainable = [p for p in adversary.model.parameters() if p.requires_grad]
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        adversary.optimizer.step()

        total += step_n
        fooled_total += step_fooled
        # NOTE: policy_loss ~ 0 is expected — group-normalized advantages are mean-zero, so the
        # loss *value* vanishes while its *gradient* (the learning signal) does not. Watch
        # grad_norm and reward_mean instead.
        logger.log("grpo_steps", {
            "round": rnd, "step": step,
            "policy_loss": float(np.mean(losses)), "kl": float(np.mean(kls)),
            "grad_norm": grad_norm,
            "reward_mean": float(np.mean(r_means)), "reward_std": float(np.mean(r_stds)),
            "fooled_rate": step_fooled / max(step_n, 1),
            "parse_rate": parsed_total / max(step_n, 1), "lr": cfg.grpo_lr,
        })
        print(
            f"  grpo step {step} | grad_norm {grad_norm:.3f} | reward {np.mean(r_means):.3f} "
            f"| fooled {step_fooled / max(step_n, 1):.2f} | parse {parsed_total / max(step_n, 1):.2f}"
        )

    return mined, fooled_total / max(total, 1)
