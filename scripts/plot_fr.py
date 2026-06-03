"""Average the false-refusal experiment over seeds and plot it in a hand-drawn (xkcd) style.

Plots: false-refusal rate, classifier accuracy, time-to-harden, category coverage, trigger rate.
Usage: python scripts/plot_fr.py [n_seeds]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from loophole_v2.logging_jsonl import read_jsonl

n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
out = Path("runs/expfr_compare")
out.mkdir(parents=True, exist_ok=True)

SMART, RAND = "#2b6cb0", "#dd6b20"  # blue / orange
LAB = {"grpo": "SMART adversary (learned)", "greedy": "RANDOM adversary"}
COL = {"grpo": SMART, "greedy": RAND}


def series(mode, field):
    rows = []
    for s in range(n):
        p = Path(f"runs/expfr_{mode}_s{s}/logs/rounds.jsonl")
        if p.exists():
            rows.append([(r[field] if r[field] is not None else np.nan) for r in read_jsonl(p)])
    if not rows:
        return None
    L = max(len(r) for r in rows)
    rows = [r for r in rows if len(r) == L]
    return np.array(rows, dtype=float)


def band(ax, mode, arr, marker="o"):
    x = np.arange(arr.shape[1])
    m, sd = np.nanmean(arr, 0), np.nanstd(arr, 0)
    ax.plot(x, m, marker + "-", color=COL[mode], lw=3, ms=7, label=f"{LAB[mode]} (n={arr.shape[0]})")
    ax.fill_between(x, m - sd, m + sd, color=COL[mode], alpha=0.15, lw=0)


def style(ax, title, ylab, ylim=None, legend=True):
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("training round")
    ax.set_ylabel(ylab)
    if ylim:
        ax.set_ylim(*ylim)
    if legend:
        ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.25)


with plt.xkcd(scale=2.2, length=100, randomness=3):
    # 1) false-refusal rate
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=130)
    for mode in ("grpo", "greedy"):
        a = series(mode, "false_refusal")
        if a is not None:
            band(ax, mode, a)
    ax.annotate("the smart attacker\nfixes it almost\nimmediately", xy=(2, 0.25), xytext=(4.2, 0.62),
                fontsize=11, color=SMART, arrowprops=dict(arrowstyle="->", color=SMART, lw=2))
    style(ax, "WRONGLY BLOCKING GOOD CUSTOMERS\n(and how fast the filter learns to stop)",
          "false-refusal rate", (-0.03, 1.08))
    fig.tight_layout(); fig.savefig(out / "false_refusal.png", bbox_inches="tight"); plt.close(fig)

    # 2) classifier accuracy
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=130)
    for mode in ("grpo", "greedy"):
        a = series(mode, "general_acc")
        if a is not None:
            band(ax, mode, a, marker="s")
    style(ax, "THE FILTER GETS GOOD FASTER\nwhen trained by the smart attacker",
          "overall accuracy", (0.45, 1.03))
    fig.tight_layout(); fig.savefig(out / "classifier_accuracy.png", bbox_inches="tight"); plt.close(fig)

    # 3) time-to-harden (rounds until false-refusal first drops below 0.30)
    THRESH = 0.30
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=130)
    means, errs, cols, labs = [], [], [], []
    for mode in ("grpo", "greedy"):
        a = series(mode, "false_refusal")
        if a is None:
            continue
        times = []
        for row in a:
            hit = np.where(row <= THRESH)[0]
            times.append(hit[0] if len(hit) else a.shape[1] - 1)
        means.append(np.mean(times)); errs.append(np.std(times)); cols.append(COL[mode]); labs.append(LAB[mode].split(" ")[0])
    ax.bar(labs, means, yerr=errs, color=cols, capsize=8, width=0.55, alpha=0.9)
    for i, m in enumerate(means):
        ax.text(i, m + 0.2, f"{m:.0f} rounds", ha="center", fontsize=12)
    style(ax, "HOW LONG UNTIL THE FILTER STOPS\nOVER-REFUSING? (lower = better)", "rounds to harden", legend=False)
    ax.set_xlabel("")
    fig.tight_layout(); fig.savefig(out / "time_to_harden.png", bbox_inches="tight"); plt.close(fig)

    # 4) category coverage
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=130)
    for mode in ("grpo", "greedy"):
        a = series(mode, "categories")
        if a is not None:
            band(ax, mode, np.maximum.accumulate(a, axis=1), marker="^")  # cumulative categories found
    ax.axhline(6, ls="--", color="gray", lw=1.5)
    ax.text(0.2, 6.05, "all 6 weak spots", fontsize=9, color="gray")
    style(ax, "FINDING THE FILTER'S BLIND SPOTS\n(6 hidden trigger types)", "blind-spot categories found", (-0.2, 6.5))
    fig.tight_layout(); fig.savefig(out / "category_coverage.png", bbox_inches="tight"); plt.close(fig)

    # 5) trigger rate (efficiency)
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=130)
    for mode in ("grpo", "greedy"):
        a = series(mode, "trigger_rate")
        if a is not None:
            band(ax, mode, np.nan_to_num(a), marker="D")
    style(ax, "WHY: THE SMART ATTACKER RARELY MISSES\n(random wastes its shots on decoys)", "fraction of attacks that land")
    fig.tight_layout(); fig.savefig(out / "trigger_rate.png", bbox_inches="tight"); plt.close(fig)

print("wrote 5 plots to", out)
