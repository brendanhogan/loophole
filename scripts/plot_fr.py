"""Average the false-refusal experiment over seeds and plot it (publication style).

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

# --- palette + style ---
SMART, RAND = "#0F7173", "#E8743B"   # deep teal / warm orange
INK, MUTE = "#22303C", "#7A8B99"
LAB = {"grpo": "Learned adversary", "greedy": "Random adversary"}
COL = {"grpo": SMART, "greedy": RAND}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12, "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.labelsize": 12.5, "axes.labelcolor": INK, "text.color": INK,
    "axes.edgecolor": MUTE, "xtick.color": MUTE, "ytick.color": MUTE,
    "axes.linewidth": 1.1, "figure.dpi": 150, "savefig.dpi": 150,
})


def newax(figsize=(8.8, 5.2)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E7ECEF", lw=1.0, zorder=0)
    ax.set_axisbelow(True)
    return fig, ax


def title(ax, head, sub=None):
    ax.text(0, 1.12, head, transform=ax.transAxes, fontsize=15, fontweight="bold", color=INK, va="bottom")
    if sub:
        ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=10.5, color=MUTE, va="bottom")


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


def band(ax, mode, arr, marker="o", x0=0):
    x = np.arange(x0, arr.shape[1])
    a = arr[:, x0:]
    m, sd = np.nanmean(a, 0), np.nanstd(a, 0)
    ax.fill_between(x, m - sd, m + sd, color=COL[mode], alpha=0.13, lw=0, zorder=2)
    ax.plot(x, m, "-", color=COL[mode], lw=2.6, zorder=3, label=f"{LAB[mode]}  (n={arr.shape[0]})")
    ax.plot(x, m, marker, color=COL[mode], ms=6.5, mec="white", mew=1.2, zorder=4)


def legend(ax, **kw):
    lg = ax.legend(frameon=False, fontsize=11.5, **kw)
    for t in lg.get_texts():
        t.set_color(INK)


# 1) false-refusal rate (start at the seeded baseline; an untrained filter refuses nothing) ----
fig, ax = newax()
for mode in ("grpo", "greedy"):
    a = series(mode, "false_refusal")
    if a is not None:
        band(ax, mode, a, x0=1)
ax.annotate("learned attacker hardens\nthe filter almost immediately",
            xy=(3, 0.27), xytext=(5.2, 0.66), fontsize=11, color=SMART,
            arrowprops=dict(arrowstyle="-|>", color=SMART, lw=1.8, connectionstyle="arc3,rad=-0.2"))
ax.set_xlabel("training round"); ax.set_ylabel("false-refusal rate"); ax.set_ylim(-0.03, 1.05)
title(ax, "Wrongly blocking good customers", "share of harmless requests the filter refuses · mean ± std over 5 seeds")
legend(ax, loc="upper right")
fig.tight_layout(); fig.savefig(out / "false_refusal.png", bbox_inches="tight"); plt.close(fig)

# 2) classifier accuracy (from the UNTRAINED model at round 0) ----------------
fig, ax = newax()
for mode in ("grpo", "greedy"):
    a = series(mode, "general_acc")
    if a is not None:
        band(ax, mode, a, marker="s", x0=0)
ax.axvline(1, color=MUTE, lw=1, ls=(0, (3, 3)), zorder=1)
ax.text(1.12, 0.52, "untrained → seed-trained\n→ adversarial rounds", fontsize=9.5, color=MUTE)
ax.set_xlabel("training round"); ax.set_ylabel("overall accuracy"); ax.set_ylim(0.44, 1.01)
title(ax, "The filter gets accurate faster", "balanced allow/block accuracy, from a randomly-initialized model · mean ± std over 5 seeds")
legend(ax, loc="lower right")
fig.tight_layout(); fig.savefig(out / "classifier_accuracy.png", bbox_inches="tight"); plt.close(fig)

# 3) time-to-harden ----------------------------------------------------------
THRESH = 0.30
fig, ax = plt.subplots(figsize=(6.4, 5.2))
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="#E7ECEF", lw=1.0, zorder=0); ax.set_axisbelow(True)
labs, means, errs, cols = [], [], [], []
for mode in ("grpo", "greedy"):
    a = series(mode, "false_refusal")
    if a is None:
        continue
    # count adversarial rounds (from the seeded baseline at round 1) until fr drops below threshold
    t = [(np.where(r[1:] <= THRESH)[0][0] if np.any(r[1:] <= THRESH) else a.shape[1] - 2) for r in a]
    labs.append(LAB[mode].split()[0]); means.append(np.mean(t)); errs.append(np.std(t)); cols.append(COL[mode])
bars = ax.bar(labs, means, yerr=errs, color=cols, width=0.6, zorder=3,
              error_kw=dict(ecolor=INK, lw=1.5, capsize=7), edgecolor="white", lw=1.5)
for i, m in enumerate(means):
    ax.text(i, m + max(errs) * 0.12 + 0.15, f"{m:.0f} rounds", ha="center", fontsize=13, color=INK, fontweight="bold")
ax.set_ylabel("rounds to harden"); ax.set_ylim(0, max(means) + max(errs) + 1.2)
title(ax, "Time to stop over-refusing", "rounds until false-refusal drops below 30% · lower is better")
fig.tight_layout(); fig.savefig(out / "time_to_harden.png", bbox_inches="tight"); plt.close(fig)

# 4) category coverage -------------------------------------------------------
fig, ax = newax()
for mode in ("grpo", "greedy"):
    a = series(mode, "categories")
    if a is not None:
        band(ax, mode, np.maximum.accumulate(a, axis=1), marker="^", x0=2)
ax.axhline(6, ls=(0, (4, 3)), color=MUTE, lw=1.3)
ax.text(0.15, 6.08, "all 6 blind spots", fontsize=10.5, color=MUTE)
ax.set_xlabel("training round"); ax.set_ylabel("blind-spot categories found"); ax.set_ylim(-0.2, 6.6)
title(ax, "Finding the filter's blind spots", "6 hidden trigger categories · mean ± std over 5 seeds")
legend(ax, loc="lower right")
fig.tight_layout(); fig.savefig(out / "category_coverage.png", bbox_inches="tight"); plt.close(fig)

# 5) trigger rate ------------------------------------------------------------
fig, ax = newax()
for mode in ("grpo", "greedy"):
    a = series(mode, "trigger_rate")
    if a is not None:
        band(ax, mode, np.nan_to_num(a), marker="D", x0=2)
ax.set_xlabel("training round"); ax.set_ylabel("fraction of attacks that land"); ax.set_ylim(-0.03, 1.02)
title(ax, "Why: the learned attacker rarely misses", "random wastes its attempts on decoy transforms · mean ± std over 5 seeds")
legend(ax, loc="upper right")
fig.tight_layout(); fig.savefig(out / "trigger_rate.png", bbox_inches="tight"); plt.close(fig)

print("wrote 5 publication-style plots to", out)
