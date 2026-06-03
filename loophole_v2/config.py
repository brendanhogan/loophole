"""All run hyperparameters in one readable place."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # --- data / io ---
    constitution_path: str = "constitutions/sample_safety.yaml"
    run_dir: str = "runs/dev"
    seed: int = 0
    device: str = "cuda"

    # --- outer loop ---
    rounds: int = 8

    # --- adversary mining (a "group" of actions per assigned goal) ---
    goals_per_round: int = 16
    group_size: int = 8
    max_stack: int = 3  # max transform-stack depth

    # --- classifier C (retrained from base each round) ---
    classifier_model: str = "Qwen/Qwen3-0.6B"
    c_epochs: int = 2
    c_lr: float = 1e-5
    c_batch_size: int = 16
    easy_per_round: int = 200
    train_sample: int = 512

    # --- reward ---
    novelty_lambda: float = 0.3
    reward_fooled_bonus: float = 0.0  # bonus for actually crossing the decision boundary

    # --- checkpoint population (anti-cycling; M6) ---
    pop_size: int = 5

    # --- GRPO adversary (M5) ---
    use_grpo: bool = False
    adversary_model: str = "Qwen/Qwen3.5-9B"
    phase_a_steps: int = 4
    grpo_lr: float = 1e-5
    grpo_kl_beta: float = 0.02
    grpo_temperature: float = 1.0
    grpo_max_new_tokens: int = 512

    # --- verifier V (sampled gate; M7) ---
    verifier_model: str = ""  # empty => verifier disabled
    verifier_sample_rate: float = 0.0

    # --- eval ---
    eval_n: int = 240  # size of each fixed eval set (general / hard)
