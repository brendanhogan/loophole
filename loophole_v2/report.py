"""Self-contained HTML report — the blog artifact.

Reads the JSONL logs of a run and renders one standalone ``report.html`` (all images
base64-embedded, no external assets): training curves, a decision-boundary slider over
rounds, transform-usage stats, and the strategies the adversary discovered. Pure CPU —
everything it needs was logged during the run.
"""

from __future__ import annotations

import base64
import io
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .embed import project_2d  # noqa: E402
from .logging_jsonl import read_jsonl  # noqa: E402

_BG, _FG, _ACCENT = "#0f1115", "#e6e6e6", "#7aa2f7"


def generate(run_dir: str, out: str | None = None) -> str:
    logs = Path(run_dir) / "logs"
    rounds = read_jsonl(logs / "rounds.jsonl") if (logs / "rounds.jsonl").exists() else []
    steps = read_jsonl(logs / "grpo_steps.jsonl") if (logs / "grpo_steps.jsonl").exists() else []
    rollouts = read_jsonl(logs / "rollouts.jsonl") if (logs / "rollouts.jsonl").exists() else []
    points = read_jsonl(logs / "boundary_points.jsonl") if (logs / "boundary_points.jsonl").exists() else []
    boundary = read_jsonl(logs / "boundary.jsonl") if (logs / "boundary.jsonl").exists() else []

    panels = [
        ("Convergence — fooled-rate & accuracy per round", _curves_panel(rounds)),
        ("Universality — most transferable transform per round", _universality_panel(rounds)),
    ]
    if steps:
        panels.append(("GRPO — reward & gradient signal per step", _grpo_panel(steps)))
    if rollouts:
        panels.append(("Transform usage (and how often each fooled C)", _transform_panel(rollouts)))

    boundary_imgs, boundary_rounds = _boundary_frames(points, boundary)
    strategies = _top_strategies(rollouts)

    html = _render_html(run_dir, panels, boundary_imgs, boundary_rounds, strategies, rounds)
    out_path = Path(out) if out else Path(run_dir) / "report.html"
    out_path.write_text(html)
    return str(out_path)


# --- matplotlib panels (each returns a base64 PNG) ---------------------------

def _fig():
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=110)
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    for s in ax.spines.values():
        s.set_color("#333")
    ax.tick_params(colors=_FG)
    ax.xaxis.label.set_color(_FG)
    ax.yaxis.label.set_color(_FG)
    ax.title.set_color(_FG)
    return fig, ax


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _curves_panel(rounds) -> str:
    fig, ax = _fig()
    r = [x["round"] for x in rounds]
    ax.plot(r, [x.get("general_acc") for x in rounds], "-o", color="#9ece6a", label="general acc")
    ax.plot(r, [x.get("hard_acc") for x in rounds], "-o", color=_ACCENT, label="hard acc")
    ax.plot(r, [x.get("general_fpr") for x in rounds], "--", color="#f7768e", label="general FPR")
    fooled_r = [x["round"] for x in rounds if x.get("fooled_rate") is not None]
    fooled_v = [x["fooled_rate"] for x in rounds if x.get("fooled_rate") is not None]
    ax.plot(fooled_r, fooled_v, "-s", color="#e0af68", label="fooled-rate")
    ax.set_xlabel("round")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(facecolor=_BG, labelcolor=_FG, edgecolor="#333", fontsize=8)
    return _png(fig)


def _universality_panel(rounds) -> str:
    fig, ax = _fig()
    r = [x["round"] for x in rounds]
    ax.plot(r, [x.get("max_universality", 0) for x in rounds], "-o", color="#bb9af7",
            label="max universality (a single transform's fooled-fraction over all payloads)")
    ax.set_xlabel("round")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(facecolor=_BG, labelcolor=_FG, edgecolor="#333", fontsize=8)
    return _png(fig)


def _grpo_panel(steps) -> str:
    fig, ax = _fig()
    x = list(range(len(steps)))
    ax.plot(x, [s.get("reward_mean") for s in steps], "-o", color="#9ece6a", label="reward mean")
    ax.plot(x, [s.get("fooled_rate") for s in steps], "-s", color="#e0af68", label="fooled-rate")
    ax.set_xlabel("GRPO step (across rounds)")
    ax.set_ylabel("reward / fooled")
    ax2 = ax.twinx()
    ax2.plot(x, [s.get("grad_norm") for s in steps], ":", color=_ACCENT, label="grad norm")
    ax2.tick_params(colors=_ACCENT)
    ax2.set_ylabel("grad norm", color=_ACCENT)
    ax.legend(facecolor=_BG, labelcolor=_FG, edgecolor="#333", fontsize=8, loc="upper right")
    return _png(fig)


def _transform_panel(rollouts) -> str:
    used, fooled = Counter(), Counter()
    for r in rollouts:
        for t in (r.get("transform_ids") or []):
            used[t] += 1
            if r.get("fooled"):
                fooled[t] += 1
    if not used:
        fig, ax = _fig()
        return _png(fig)
    labels = [t for t, _ in used.most_common()]
    fig, ax = _fig()
    ax.bar(labels, [used[t] for t in labels], color="#414868", label="used")
    ax.bar(labels, [fooled[t] for t in labels], color="#e0af68", label="fooled C")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.legend(facecolor=_BG, labelcolor=_FG, edgecolor="#333", fontsize=8)
    return _png(fig)


def _boundary_frames(points, boundary) -> tuple[list[str], list[int]]:
    """One scatter+contour PNG per round over a fixed 2D layout, recolored by P(block)."""
    if not points or not boundary:
        return [], []
    coords = project_2d([p["prompt"] for p in sorted(points, key=lambda p: p["idx"])])
    labels = [p["true_label"] for p in sorted(points, key=lambda p: p["idx"])]
    by_round = defaultdict(dict)
    for b in boundary:
        by_round[b["round"]][b["idx"]] = b["p_block"]

    imgs, rnds = [], []
    for rnd in sorted(by_round):
        pb = [by_round[rnd].get(i, 0.5) for i in range(len(coords))]
        imgs.append(_boundary_png(coords, labels, pb, rnd))
        rnds.append(rnd)
    return imgs, rnds


def _boundary_png(coords, labels, p_block, rnd) -> str:
    import numpy as np
    from sklearn.neighbors import KNeighborsRegressor

    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=110)
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    x, y, pb = coords[:, 0], coords[:, 1], np.array(p_block)
    # Smooth decision surface: fit P(block) over the fixed layout and evaluate on a mesh, so
    # the 0.5 contour reads as a clean boundary rather than jagged point-to-point interpolation.
    try:
        knn = KNeighborsRegressor(n_neighbors=min(15, len(coords)), weights="distance").fit(coords, pb)
        pad_x = 0.05 * (x.max() - x.min() + 1e-9)
        pad_y = 0.05 * (y.max() - y.min() + 1e-9)
        gx = np.linspace(x.min() - pad_x, x.max() + pad_x, 220)
        gy = np.linspace(y.min() - pad_y, y.max() + pad_y, 220)
        xx, yy = np.meshgrid(gx, gy)
        zz = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=np.linspace(0, 1, 11), cmap="coolwarm", alpha=0.55, vmin=0, vmax=1)
        ax.contour(xx, yy, zz, levels=[0.5], colors="white", linewidths=1.8)
    except Exception:
        pass
    lab = np.array(labels)
    ax.scatter(x[lab == 0], y[lab == 0], marker="o", s=22, facecolors="none",
               edgecolors="#9ece6a", linewidths=1.0, label="allow (true)")
    ax.scatter(x[lab == 1], y[lab == 1], marker="x", s=26, color="#f7768e", label="block (true)")
    ax.set_title(f"round {rnd}", color=_FG)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(facecolor=_BG, labelcolor=_FG, edgecolor="#333", fontsize=8, loc="upper right")
    return _png(fig)


def _top_strategies(rollouts, k: int = 12) -> list[dict]:
    agg = defaultdict(lambda: {"count": 0, "fooled": 0, "reward": 0.0})
    for r in rollouts:
        ids = r.get("transform_ids")
        if ids is None:
            continue
        key = " + ".join(ids) if ids else "(no transforms)"
        a = agg[key]
        a["count"] += 1
        a["fooled"] += int(bool(r.get("fooled")))
        a["reward"] += float((r.get("reward") or {}).get("total", 0.0))
    rows = []
    for key, a in agg.items():
        rows.append({
            "stack": key, "count": a["count"], "fooled": a["fooled"],
            "fool_rate": a["fooled"] / a["count"] if a["count"] else 0.0,
            "mean_reward": a["reward"] / a["count"] if a["count"] else 0.0,
        })
    rows.sort(key=lambda r: (r["fooled"], r["mean_reward"]), reverse=True)
    return rows[:k]


# --- HTML assembly -----------------------------------------------------------

def _img(b64: str, style: str = "") -> str:
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;{style}"/>'


def _render_html(run_dir, panels, boundary_imgs, boundary_rounds, strategies, rounds) -> str:
    panel_html = "".join(
        f'<section><h2>{title}</h2>{_img(b64)}</section>' for title, b64 in panels
    )

    if boundary_imgs:
        frames = ",".join(f'"{b}"' for b in boundary_imgs)
        max_idx = len(boundary_imgs) - 1
        boundary_html = f"""
        <section><h2>Decision boundary over rounds</h2>
        <p class="muted">Fixed TF-IDF/PCA layout of the eval points; color = current C's P(block),
        white line = the 0.5 boundary. The points stay put — the surface moves as C hardens.</p>
        <input id="sl" type="range" min="0" max="{max_idx}" value="0" style="width:60%"
               oninput="document.getElementById('bd').src='data:image/png;base64,'+F[this.value];
                        document.getElementById('rl').textContent='round '+R[this.value]"/>
        <span id="rl" class="muted">round {boundary_rounds[0]}</span><br/>
        <img id="bd" src="data:image/png;base64,{boundary_imgs[0]}" style="max-width:520px"/>
        <script>const F=[{frames}]; const R=[{",".join(str(r) for r in boundary_rounds)}];</script>
        </section>"""
    else:
        boundary_html = ""

    rows = "".join(
        f"<tr><td><code>{s['stack']}</code></td><td>{s['count']}</td>"
        f"<td>{s['fooled']}</td><td>{s['fool_rate']:.2f}</td><td>{s['mean_reward']:.3f}</td></tr>"
        for s in strategies
    )
    strat_html = f"""
    <section><h2>Strategies the adversary discovered</h2>
    <table><thead><tr><th>transform stack</th><th>tried</th><th>fooled C</th>
    <th>fool-rate</th><th>mean reward</th></tr></thead><tbody>{rows}</tbody></table>
    </section>""" if rows else ""

    final = rounds[-1] if rounds else {}
    summary = (
        f"final general acc <b>{final.get('general_acc', 0):.3f}</b> · "
        f"hard acc <b>{final.get('hard_acc', 0):.3f}</b> · "
        f"general FPR <b>{final.get('general_fpr', 0):.3f}</b> · "
        f"{len(rounds)} rounds"
    ) if rounds else "no rounds logged"

    return f"""<!doctype html><html><head><meta charset="utf-8"/>
<title>Loophole v2 — {run_dir}</title>
<style>
  body{{background:{_BG};color:{_FG};font-family:-apple-system,Segoe UI,Roboto,sans-serif;
       max-width:900px;margin:24px auto;padding:0 16px;line-height:1.5}}
  h1{{color:{_ACCENT}}} h2{{color:{_ACCENT};font-size:1.05rem;margin-top:1.6em}}
  .muted{{color:#8a8f98;font-size:.85rem}}
  section{{border-top:1px solid #20232b;padding-top:.4em}}
  table{{width:100%;border-collapse:collapse;font-size:.85rem}}
  th,td{{text-align:left;padding:4px 8px;border-bottom:1px solid #20232b}}
  code{{color:#e0af68}}
</style></head><body>
<h1>Loophole v2 — adversarial classifier training</h1>
<p class="muted">{run_dir} · {summary}</p>
{boundary_html}
{panel_html}
{strat_html}
<p class="muted">Labels are guaranteed by construction; the classifier is hardened against a
learned adversary that mines the decision boundary. See context.md / plan.md.</p>
</body></html>"""
