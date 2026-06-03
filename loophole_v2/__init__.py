"""Loophole v2 — adversarial classifier training (GAN-style min-max).

A learned adversary mines decision-boundary attacks whose labels are guaranteed
by construction (no judge in the loop); a small classifier hardens against them
each round. See ``context.md`` / ``plan.md`` at the repo root for the full design.
"""

__version__ = "0.2.0"
