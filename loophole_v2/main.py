"""CLI: ``loophole-v2 run`` to play the game, ``loophole-v2 report`` to build the HTML."""

from __future__ import annotations

import typer

from .config import Config
from .loop import run as run_loop

app = typer.Typer(add_completion=False, help="Loophole v2 — adversarial classifier training.")


@app.command()
def run(
    constitution: str = "constitutions/sample_safety.yaml",
    run_dir: str = "runs/dev",
    rounds: int = 8,
    grpo: bool = typer.Option(False, help="Use the GRPO adversary instead of the greedy baseline."),
    seed: int = 0,
    goals_per_round: int = Config.goals_per_round,
    group_size: int = Config.group_size,
    phase_a_steps: int = Config.phase_a_steps,
    grpo_lr: float = Config.grpo_lr,
    easy_per_round: int = Config.easy_per_round,
    verifier_model: str = typer.Option("", help="Enable the verifier V (e.g. a claude/gpt/ollama model)."),
    verifier_sample_rate: float = 0.0,
):
    """Play the min-max game and write logs + checkpoints to RUN_DIR."""
    cfg = Config(
        constitution_path=constitution,
        run_dir=run_dir,
        rounds=rounds,
        use_grpo=grpo,
        seed=seed,
        goals_per_round=goals_per_round,
        group_size=group_size,
        phase_a_steps=phase_a_steps,
        grpo_lr=grpo_lr,
        easy_per_round=easy_per_round,
        verifier_model=verifier_model,
        verifier_sample_rate=verifier_sample_rate,
    )
    run_loop(cfg)


@app.command()
def report(run_dir: str = "runs/dev", out: str = ""):
    """Generate a self-contained HTML report from a run directory."""
    from .report import generate

    path = generate(run_dir, out or None)
    print(f"Wrote {path}")


if __name__ == "__main__":
    app()
