"""Generate the landing page: full interactive chamber + bill library cards.

The chamber at the top is the main 'always-there' view of the simulator —
hover any seat for a tooltip with the senator's photo and meta, click for
their full moral constitution. Bills live below as cards linking out to
their iteration reports.
"""

from __future__ import annotations

import html
import json
import math
from datetime import datetime
from pathlib import Path

from loophole.senate.models import IterationSession, MoralConstitution, Party, Senator
from loophole.senate.roster import load_roster
from loophole.senate.session import ConstitutionStore

ROW_SIZES = [22, 24, 26, 28]
ROW_DEM_COUNT = [10, 11, 12, 14]


def scan_iterations(sessions_dir: Path | str = "sessions") -> list[IterationSession]:
    base = Path(sessions_dir)
    out: list[IterationSession] = []
    if not base.exists():
        return out
    for p in sorted(base.iterdir(), reverse=True):
        if not p.name.startswith("iterate_"):
            continue
        state = p / "state.json"
        if not state.exists():
            continue
        try:
            sess = IterationSession.model_validate_json(state.read_text())
        except Exception:
            continue
        out.append(sess)
    return out


def generate_landing_html(
    sessions: list[IterationSession],
    senators: list[Senator] | None = None,
    constitutions: dict[str, MoralConstitution] | None = None,
    output_path: str = "sessions/index.html",
) -> str:
    senators = senators or load_roster()
    if constitutions is None:
        try:
            constitutions = ConstitutionStore().load_all()
        except Exception:
            constitutions = {}

    # Full chamber payload — for the hero, hoverable/clickable
    full_seats = _layout_seats_full(senators, constitutions)

    # Thumbnail layout (compact) — for bill cards
    thumb_seats = _layout_seats_thumbnail(senators)

    cards = []
    for sess in sessions:
        if not sess.rounds:
            continue
        final = sess.rounds[-1]
        first_bill = sess.rounds[0].bill
        yes_total = final.yes_total
        no_total = final.tally["no"] + final.tally["lean_no"]
        verdict, kind = _verdict(sess.final_status, yes_total)
        seat_votes = {r.senator_full_name: r.vote.value for r in final.reactions}
        report_link = f"{sess.session_id}/iteration.html"
        cards.append({
            "session_id": sess.session_id,
            "bill_name": first_bill.name,
            "summary": first_bill.summary,
            "user_prompt": sess.user_prompt,
            "rounds": len(sess.rounds),
            "yes": yes_total,
            "no": no_total,
            "verdict": verdict,
            "kind": kind,
            "report_link": report_link,
            "seat_svg": _build_thumbnail_svg(thumb_seats, seat_votes),
            "created_at": sess.created_at.strftime("%b %d, %Y") if sess.created_at else "",
        })

    payload = {
        "full_seats": full_seats,
        "total_senators": len(senators),
        "n_d": sum(1 for s in senators if s.party == Party.DEMOCRAT),
        "n_r": sum(1 for s in senators if s.party == Party.REPUBLICAN),
        "n_i": sum(1 for s in senators if s.party == Party.INDEPENDENT),
    }
    payload_json = json.dumps(payload).replace("</", "<\\/")

    page = _PAGE_TEMPLATE.format(
        cards_html=_render_cards(cards) or _render_empty_state(),
        generated_at=datetime.now().strftime("%b %d, %Y · %I:%M %p"),
        total=len(cards),
        plural="" if len(cards) == 1 else "s",
        payload=payload_json,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return str(out)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def _is_dem_caucus(s: Senator) -> bool:
    return s.party == Party.DEMOCRAT or (
        s.party == Party.INDEPENDENT and s.caucus == Party.DEMOCRAT
    )


def _layout_seats_full(senators: list[Senator], constitutions: dict[str, MoralConstitution]) -> list[dict]:
    """Full-size chamber layout, with constitution and metadata for tooltip/panel."""
    dems = sorted([s for s in senators if _is_dem_caucus(s)], key=lambda s: s.short_name)
    reps = sorted([s for s in senators if not _is_dem_caucus(s)], key=lambda s: s.short_name)

    cx, cy = 500.0, 480.0
    base_radius = 180.0
    row_step = 75.0

    seats: list[dict] = []
    dem_idx = rep_idx = 0
    for row_idx, row_size in enumerate(ROW_SIZES):
        radius = base_radius + row_idx * row_step
        dem_n = ROW_DEM_COUNT[row_idx]
        rep_n = row_size - dem_n
        aisle = 0.08
        left_start = math.pi - 0.04
        left_end = math.pi / 2 + aisle / 2
        right_start = math.pi / 2 - aisle / 2
        right_end = 0.04

        for k in range(dem_n):
            t = k / max(dem_n - 1, 1)
            angle = left_start + t * (left_end - left_start)
            s = dems[dem_idx]
            dem_idx += 1
            seats.append(_seat_full(s, constitutions.get(s.full_name), cx, cy, radius, angle))
        for k in range(rep_n):
            t = k / max(rep_n - 1, 1)
            angle = right_start + t * (right_end - right_start)
            s = reps[rep_idx]
            rep_idx += 1
            seats.append(_seat_full(s, constitutions.get(s.full_name), cx, cy, radius, angle))
    return seats


def _seat_full(s: Senator, c: MoralConstitution | None, cx: float, cy: float, radius: float, angle: float) -> dict:
    return {
        "full_name": s.full_name,
        "short_name": s.short_name,
        "party": s.party.value,
        "state": s.state,
        "role": s.role,
        "photo_url": s.photo_url,
        "x": round(cx + radius * math.cos(angle), 2),
        "y": round(cy - radius * math.sin(angle), 2),
        "constitution": _serialize_constitution(c),
    }


def _serialize_constitution(c: MoralConstitution | None) -> dict | None:
    if c is None:
        return None
    return {
        "core_values": c.core_values,
        "top_issues": [{"issue": i.issue, "stance": i.stance} for i in c.top_issues],
        "red_lines": c.red_lines,
        "negotiation_style": c.negotiation_style,
        "typical_allies": c.typical_allies,
        "voice_notes": c.voice_notes,
        "confidence": c.confidence,
    }


def _layout_seats_thumbnail(senators: list[Senator]) -> list[dict]:
    """Compact thumbnail layout for bill cards. Tighter, fewer pixels per seat."""
    dems = sorted([s for s in senators if _is_dem_caucus(s)], key=lambda s: s.short_name)
    reps = sorted([s for s in senators if not _is_dem_caucus(s)], key=lambda s: s.short_name)

    # Wider, shorter — fits in a card thumbnail strip
    cx, cy = 250.0, 175.0
    base_radius = 78.0
    row_step = 19.0

    seats: list[dict] = []
    dem_idx = rep_idx = 0
    for row_idx, row_size in enumerate(ROW_SIZES):
        radius = base_radius + row_idx * row_step
        dem_n = ROW_DEM_COUNT[row_idx]
        rep_n = row_size - dem_n
        aisle = 0.14
        left_start = math.pi - 0.05
        left_end = math.pi / 2 + aisle / 2
        right_start = math.pi / 2 - aisle / 2
        right_end = 0.05

        for k in range(dem_n):
            t = k / max(dem_n - 1, 1)
            angle = left_start + t * (left_end - left_start)
            s = dems[dem_idx]
            dem_idx += 1
            seats.append({"full_name": s.full_name, "party": s.party.value,
                          "x": round(cx + radius * math.cos(angle), 2),
                          "y": round(cy - radius * math.sin(angle), 2)})
        for k in range(rep_n):
            t = k / max(rep_n - 1, 1)
            angle = right_start + t * (right_end - right_start)
            s = reps[rep_idx]
            rep_idx += 1
            seats.append({"full_name": s.full_name, "party": s.party.value,
                          "x": round(cx + radius * math.cos(angle), 2),
                          "y": round(cy - radius * math.sin(angle), 2)})
    return seats


def _build_thumbnail_svg(seat_layout, seat_votes) -> str:
    seats = []
    for s in seat_layout:
        vote = seat_votes.get(s["full_name"], "absent")
        seats.append(
            f'<circle cx="{s["x"]}" cy="{s["y"]}" r="2.4" '
            f'class="thumb-vote-{vote}"/>'
        )
    return (
        '<svg class="thumbnail-svg" viewBox="0 0 500 200" '
        'preserveAspectRatio="xMidYMid meet">'
        + "".join(seats)
        + '</svg>'
    )


def _verdict(status: str, yes_total: int) -> tuple[str, str]:
    if status == "passed":
        return f"Passes {yes_total}", "passes"
    if status == "stuck":
        return f"Stalls at {yes_total}", "stalls"
    if status == "max_rounds":
        return f"Falls short ({yes_total})", "falls-short"
    return f"In progress ({yes_total})", "in-progress"


def _render_cards(cards: list[dict]) -> str:
    if not cards:
        return ""
    parts = []
    for c in cards:
        parts.append(f"""
        <a class="bill-card" href="{html.escape(c['report_link'])}">
            <div class="card-thumb">{c['seat_svg']}</div>
            <div class="card-body">
                <div class="card-verdict verdict-{c['kind']}">{html.escape(c['verdict'])} – {c['no']}</div>
                <h3 class="card-name">{html.escape(c['bill_name'])}</h3>
                <p class="card-summary">{html.escape(c['summary'] or '')}</p>
                <div class="card-prompt"><span class="card-prompt-mark">"</span>{html.escape(c['user_prompt'])}<span class="card-prompt-mark">"</span></div>
                <div class="card-meta">
                    <span>{c['rounds']} round{"" if c['rounds']==1 else "s"} of revision</span>
                    <span class="dot">·</span>
                    <span>{html.escape(c['created_at'])}</span>
                </div>
            </div>
        </a>
        """)
    return "\n".join(parts)


def _render_empty_state() -> str:
    return """
    <div class="empty-state">
        <p>No iterations yet. Run one from the command line:</p>
        <pre class="cmd">loophole-senate iterate "your bill idea in plain English"</pre>
    </div>
    """


_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Loophole · The Senate Floor</title>
<style>
:root {{
    --bg: #fbf9f3;
    --bg-2: #f4f0e4;
    --surface: #ffffff;
    --ink: #15161a;
    --ink-soft: #3a3d44;
    --ink-mute: #6b7080;
    --rule: #d8d3c4;
    --rule-soft: #e8e3d3;
    --accent: #8b3a2e;
    --accent-soft: #c47862;
    --gold: #b08940;
    --party-d: #245b9f;
    --party-r: #b9311f;
    --party-i: #5a4d7a;
    --vote-yes: #2d7d3a;
    --vote-lean-yes: #6ba76f;
    --vote-undecided: #c0a248;
    --vote-lean-no: #c87560;
    --vote-no: #a32a18;
    --vote-absent: #b5b0a0;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html {{ scroll-behavior: smooth; }}
body {{
    background: var(--bg);
    color: var(--ink);
    font-family: 'Charter','Iowan Old Style','Georgia','Times New Roman',serif;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
}}
.sans {{ font-family: 'Inter','Helvetica Neue',-apple-system,sans-serif; }}
.mono {{ font-family: 'iA Writer Mono','SF Mono','JetBrains Mono',Menlo,monospace; }}
.wrap {{ max-width: 1180px; margin: 0 auto; padding: 4rem 2rem 6rem; }}

/* MASTHEAD */
.masthead {{ text-align: center; border-bottom: 2px solid var(--ink); padding-bottom: 2rem; margin-bottom: 2.5rem; }}
.eyebrow {{ font-family: 'Inter',sans-serif; font-size: 0.72rem; letter-spacing: 0.3em; text-transform: uppercase; color: var(--accent); font-weight: 700; margin-bottom: 1.2rem; }}
.masthead h1 {{ font-size: 4rem; font-weight: 600; letter-spacing: -0.025em; line-height: 1; color: var(--ink); margin-bottom: 0.6rem; font-variant: small-caps; }}
.masthead .deck {{ font-style: italic; color: var(--ink-soft); font-size: 1.15rem; max-width: 660px; margin: 0.6rem auto 0; line-height: 1.55; }}
.rule-flourish {{ display: flex; align-items: center; justify-content: center; gap: 1rem; margin: 1.6rem 0 0.4rem; }}
.rule-flourish .line {{ height: 1px; width: 60px; background: var(--gold); }}
.rule-flourish .dot {{ width: 6px; height: 6px; background: var(--gold); border-radius: 50%; }}
.masthead-meta {{ font-family: 'Inter',sans-serif; font-size: 0.78rem; letter-spacing: 0.16em; text-transform: uppercase; color: var(--ink-mute); margin-top: 1rem; }}

/* SECTION HEADERS */
.section-rule {{ display: flex; align-items: center; margin: 3.5rem 0 1.5rem; }}
.section-rule h2 {{ font-family: 'Charter','Georgia',serif; font-size: 1.6rem; font-weight: 600; color: var(--ink); padding-right: 1.2rem; letter-spacing: -0.01em; }}
.section-rule .line {{ flex: 1; height: 1px; background: var(--rule); }}

/* CHAMBER HERO */
.chamber-hero {{
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: 6px;
    padding: 1.75rem 2rem 1.25rem;
    margin-top: 1rem;
}}
.chamber-explain {{
    font-style: italic;
    color: var(--ink-soft);
    font-size: 0.98rem;
    max-width: 760px;
    margin-bottom: 1.25rem;
    line-height: 1.55;
}}
.party-pills {{
    display: flex; gap: 0.6rem; flex-wrap: wrap;
    margin-bottom: 1rem;
    font-family: 'Inter',sans-serif;
    font-size: 0.78rem;
    color: var(--ink-mute);
}}
.party-pill {{
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.25rem 0.7rem;
    border-radius: 99px;
    background: var(--bg-2);
    border: 1px solid var(--rule);
}}
.party-pill .dot {{ width: 8px; height: 8px; border-radius: 50%; }}
.party-pill .dot.d {{ background: var(--party-d); }}
.party-pill .dot.r {{ background: var(--party-r); }}
.party-pill .dot.i {{ background: var(--party-i); }}

.chamber-svg {{ width: 100%; height: auto; display: block; }}
.chamber-svg .dais {{ fill: var(--bg-2); stroke: var(--rule); stroke-width: 1; }}
.chamber-svg .dais-label {{ fill: var(--ink-mute); font-family: 'Inter',sans-serif; font-size: 11px; text-anchor: middle; letter-spacing: 0.18em; text-transform: uppercase; }}
.chamber-svg .seat {{ cursor: pointer; transition: transform 0.15s ease; transform-origin: center; transform-box: fill-box; }}
.chamber-svg .seat:hover {{ transform: scale(1.45); }}
.chamber-svg .seat.selected {{ transform: scale(1.55); filter: drop-shadow(0 0 6px var(--accent)); }}
.chamber-svg .seat-ring {{ fill: none; stroke-width: 1.5; }}
.chamber-svg .seat-d .seat-ring {{ stroke: var(--party-d); }}
.chamber-svg .seat-r .seat-ring {{ stroke: var(--party-r); }}
.chamber-svg .seat-i .seat-ring {{ stroke: var(--party-i); }}
.chamber-svg .seat-d .seat-fill {{ fill: var(--party-d); fill-opacity: 0.7; }}
.chamber-svg .seat-r .seat-fill {{ fill: var(--party-r); fill-opacity: 0.7; }}
.chamber-svg .seat-i .seat-fill {{ fill: var(--party-i); fill-opacity: 0.75; }}

/* Tooltip */
.tooltip {{ position: fixed; pointer-events: none; background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 0.6rem 0.7rem; font-size: 0.85rem; max-width: 320px; box-shadow: 0 10px 30px rgba(20,22,30,0.12); opacity: 0; transition: opacity 0.1s ease; z-index: 100; }}
.tooltip.visible {{ opacity: 1; }}
.tooltip-row {{ display: flex; gap: 0.7rem; align-items: flex-start; }}
.tooltip-photo {{ width: 44px; height: 56px; border-radius: 3px; object-fit: cover; background: var(--bg-2); border: 1px solid var(--rule-soft); flex-shrink: 0; }}
.tooltip-name {{ font-family: 'Charter','Georgia',serif; font-weight: 600; color: var(--ink); font-size: 0.95rem; }}
.tooltip-meta {{ font-family: 'Inter',sans-serif; color: var(--ink-mute); font-size: 0.72rem; margin-top: 0.18rem; letter-spacing: 0.04em; }}
.tooltip-hint {{ font-family: 'Inter',sans-serif; font-size: 0.7rem; color: var(--accent); margin-top: 0.45rem; letter-spacing: 0.05em; }}

/* Side panel */
.panel {{ position: fixed; top: 0; right: 0; width: min(540px, 94vw); height: 100vh; background: var(--surface); border-left: 1px solid var(--rule); transform: translateX(100%); transition: transform 0.25s ease; overflow-y: auto; z-index: 50; box-shadow: -8px 0 32px rgba(20,22,30,0.08); }}
.panel.open {{ transform: translateX(0); }}
.panel-close {{ position: absolute; top: 1rem; right: 1rem; background: var(--surface); border: 1px solid var(--rule); border-radius: 50%; color: var(--ink-mute); width: 34px; height: 34px; cursor: pointer; font-size: 1.1rem; line-height: 1; }}
.panel-close:hover {{ color: var(--accent); border-color: var(--accent); }}
.panel-body {{ padding: 2rem 1.75rem 3rem; }}
.panel-header {{ display: flex; gap: 1.1rem; align-items: flex-start; margin-bottom: 0.4rem; }}
.panel-photo {{ width: 96px; height: 120px; border-radius: 4px; object-fit: cover; background: var(--bg-2); border: 1px solid var(--rule); flex-shrink: 0; }}
.panel-id {{ min-width: 0; flex: 1; }}
.panel-name {{ font-family: 'Charter','Georgia',serif; font-size: 1.65rem; color: var(--ink); line-height: 1.15; font-weight: 600; letter-spacing: -0.01em; }}
.panel-meta {{ color: var(--ink-mute); margin-top: 0.4rem; font-size: 0.86rem; font-family: 'Inter',sans-serif; letter-spacing: 0.04em; }}
.confidence-tag {{ display: inline-block; font-size: 0.66rem; padding: 1px 7px; border-radius: 99px; text-transform: uppercase; letter-spacing: 0.1em; margin-left: 0.4rem; font-weight: 600; }}
.confidence-tag.high {{ background: rgba(45,125,58,0.12); color: var(--vote-yes); }}
.confidence-tag.medium {{ background: rgba(192,162,72,0.18); color: var(--vote-undecided); }}
.confidence-tag.low {{ background: rgba(163,42,24,0.12); color: var(--vote-no); }}
.panel-section {{ margin-top: 1.75rem; }}
.panel-section h3 {{ font-size: 0.72rem; color: var(--accent); text-transform: uppercase; letter-spacing: 0.16em; margin-bottom: 0.7rem; font-weight: 600; font-family: 'Inter',sans-serif; }}
.const-block {{ background: var(--bg-2); border: 1px solid var(--rule-soft); border-radius: 4px; padding: 0.9rem 1.1rem; margin-bottom: 0.55rem; }}
.const-block ul {{ list-style: none; padding: 0; }}
.const-block ul li {{ padding: 0.28rem 0; font-size: 0.92rem; line-height: 1.5; color: var(--ink-soft); }}
.const-block ul li::before {{ content: "—  "; color: var(--ink-mute); }}
.issue {{ padding: 0.55rem 0; border-bottom: 1px solid var(--rule-soft); }}
.issue:last-child {{ border-bottom: none; }}
.issue .issue-name {{ color: var(--ink); font-size: 0.95rem; font-weight: 600; margin-bottom: 0.25rem; }}
.issue .issue-stance {{ color: var(--ink-soft); font-size: 0.88rem; line-height: 1.5; }}
.voice {{ color: var(--ink-soft); font-size: 0.92rem; line-height: 1.55; font-style: italic; }}

/* GRID */
.bills-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); gap: 1.4rem; }}
.bill-card {{ background: var(--surface); border: 1px solid var(--rule); border-radius: 4px; overflow: hidden; text-decoration: none; color: inherit; display: flex; flex-direction: column; transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease; cursor: pointer; }}
.bill-card:hover {{ transform: translateY(-3px); box-shadow: 0 12px 32px rgba(20,22,30,0.08); border-color: var(--ink-mute); }}
.card-thumb {{ background: var(--bg-2); border-bottom: 1px solid var(--rule-soft); padding: 0.7rem 1rem 0.4rem; }}
.thumbnail-svg {{ width: 100%; height: 100px; }}
.thumb-vote-yes {{ fill: var(--vote-yes); }}
.thumb-vote-lean_yes {{ fill: var(--vote-lean-yes); }}
.thumb-vote-undecided {{ fill: var(--vote-undecided); }}
.thumb-vote-lean_no {{ fill: var(--vote-lean-no); }}
.thumb-vote-no {{ fill: var(--vote-no); }}
.thumb-vote-absent {{ fill: var(--vote-absent); }}

.card-body {{ padding: 1.2rem 1.4rem 1.4rem; flex: 1; display: flex; flex-direction: column; }}
.card-verdict {{ font-family: 'Inter',sans-serif; font-size: 0.74rem; letter-spacing: 0.16em; text-transform: uppercase; font-weight: 700; margin-bottom: 0.5rem; }}
.verdict-passes {{ color: var(--vote-yes); }}
.verdict-falls-short {{ color: var(--vote-undecided); }}
.verdict-stalls {{ color: var(--vote-no); }}
.verdict-in-progress {{ color: var(--ink-mute); }}
.card-name {{ font-family: 'Charter','Georgia',serif; font-size: 1.3rem; line-height: 1.2; font-weight: 600; color: var(--ink); margin-bottom: 0.5rem; letter-spacing: -0.005em; }}
.card-summary {{ color: var(--ink-soft); font-size: 0.93rem; line-height: 1.5; margin-bottom: 0.9rem; }}
.card-prompt {{ font-style: italic; color: var(--ink-mute); font-size: 0.86rem; line-height: 1.5; border-left: 2px solid var(--gold); padding: 0.2rem 0.85rem; margin-bottom: 0.9rem; }}
.card-prompt-mark {{ color: var(--gold); font-style: normal; }}
.card-meta {{ font-family: 'Inter',sans-serif; font-size: 0.74rem; color: var(--ink-mute); letter-spacing: 0.04em; margin-top: auto; padding-top: 0.6rem; border-top: 1px solid var(--rule-soft); }}
.card-meta .dot {{ margin: 0 0.4rem; color: var(--rule); }}

/* CTA */
.cta-section {{ margin-top: 3rem; padding: 2.5rem 2.5rem; background: var(--surface); border: 1px solid var(--rule); border-left: 4px solid var(--accent); border-radius: 4px; }}
.cta-section h2 {{ font-family: 'Charter','Georgia',serif; font-size: 1.7rem; font-weight: 600; margin-bottom: 0.7rem; letter-spacing: -0.01em; }}
.cta-section p {{ color: var(--ink-soft); font-size: 1rem; line-height: 1.6; margin-bottom: 1rem; }}
.cta-section pre {{ background: var(--bg); border: 1px solid var(--rule); border-radius: 4px; padding: 0.9rem 1.1rem; font-family: 'iA Writer Mono','SF Mono',monospace; font-size: 0.88rem; color: var(--ink); overflow-x: auto; margin: 0.8rem 0; }}
.cta-section .ann {{ font-family: 'Inter',sans-serif; font-size: 0.78rem; color: var(--ink-mute); letter-spacing: 0.04em; margin-top: 0.6rem; }}

.empty-state {{ text-align: center; padding: 3rem 2rem; color: var(--ink-soft); }}
.empty-state .cmd {{ display: inline-block; margin-top: 0.8rem; background: var(--surface); border: 1px solid var(--rule); padding: 0.6rem 0.9rem; border-radius: 4px; font-family: 'iA Writer Mono','SF Mono',monospace; font-size: 0.9rem; }}

footer {{ margin-top: 5rem; padding-top: 2rem; border-top: 1px solid var(--rule); text-align: center; color: var(--ink-mute); font-family: 'Inter',sans-serif; font-size: 0.78rem; letter-spacing: 0.02em; }}

@media (max-width: 720px) {{
    .wrap {{ padding: 2.5rem 1.2rem 4rem; }}
    .masthead h1 {{ font-size: 2.6rem; }}
    .cta-section, .chamber-hero {{ padding: 1.6rem; }}
}}
</style>
</head>
<body>
<div class="wrap">

<header class="masthead">
    <div class="eyebrow">Loophole · A Senate Simulator</div>
    <h1>The Senate Floor</h1>
    <div class="rule-flourish"><div class="line"></div><div class="dot"></div><div class="line"></div></div>
    <p class="deck">A plain-English bill, drafted by an LLM, polled across 100 simulated senators (each modeled from their public record), then iteratively revised — without violating the user's stated tenets — until passage or impasse.</p>
    <div class="masthead-meta">{total} Bill{plural} on the Floor · Updated {generated_at}</div>
</header>

<!-- HERO CHAMBER -->
<section class="chamber-hero">
    <div class="party-pills">
        <span class="party-pill"><span class="dot d"></span>Democrats <span id="pp-d"></span></span>
        <span class="party-pill"><span class="dot r"></span>Republicans <span id="pp-r"></span></span>
        <span class="party-pill"><span class="dot i"></span>Independents <span id="pp-i"></span></span>
    </div>
    <p class="chamber-explain">Hover any seat for the senator's photo and biography. Click for their full moral constitution — the values, top issues, red lines, and rhetorical style the simulator uses to decide how they'd vote.</p>
    <svg class="chamber-svg" viewBox="0 0 1000 560" preserveAspectRatio="xMidYMid meet">
        <rect x="380" y="490" width="240" height="40" rx="4" class="dais"></rect>
        <text x="500" y="515" class="dais-label">Presiding Officer</text>
        <g id="seats"></g>
    </svg>
</section>

<!-- BILLS -->
<div class="section-rule"><h2>Bills on the Floor</h2><div class="line"></div></div>
<div class="bills-grid">
{cards_html}
</div>

<!-- CTA -->
<section class="cta-section">
    <h2>Try Your Own</h2>
    <p>Have an idea? Drop it into the iterate command. The system drafts your idea into a bill, extracts the immutable goals, then watches 100 simulated senators vote and propose amendments. The reviser tries to win more votes — without violating your goals.</p>
    <pre>loophole-senate iterate "your bill idea in plain english"</pre>
    <p class="ann">The library above refreshes automatically when the run completes.</p>
</section>

<footer>
    Loophole · Senate Simulator · Each senator's reasoning is LLM-synthesized from public-record knowledge. Not a prediction of actual votes.
</footer>

</div>

<div class="tooltip" id="tooltip"></div>
<aside class="panel" id="panel">
    <button class="panel-close" id="panel-close" aria-label="Close">×</button>
    <div class="panel-body" id="panel-body"></div>
</aside>

<script id="payload" type="application/json">{payload}</script>
<script>
(function() {{
    const data = JSON.parse(document.getElementById('payload').textContent);
    const seatsEl = document.getElementById('seats');
    const tooltip = document.getElementById('tooltip');
    const panel = document.getElementById('panel');
    const panelBody = document.getElementById('panel-body');
    const panelClose = document.getElementById('panel-close');

    // Party pills counts
    document.getElementById('pp-d').textContent = data.n_d;
    document.getElementById('pp-r').textContent = data.n_r;
    document.getElementById('pp-i').textContent = data.n_i;

    // Render seats
    const partyClass = (p) => p === 'D' ? 'seat-d' : p === 'R' ? 'seat-r' : 'seat-i';
    data.full_seats.forEach((seat, idx) => {{
        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        g.setAttribute('class', `seat ${{partyClass(seat.party)}}`);
        const ring = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        ring.setAttribute('cx', seat.x);
        ring.setAttribute('cy', seat.y);
        ring.setAttribute('r', 11);
        ring.setAttribute('class', 'seat-ring');
        const fill = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        fill.setAttribute('cx', seat.x);
        fill.setAttribute('cy', seat.y);
        fill.setAttribute('r', 8);
        fill.setAttribute('class', 'seat-fill');
        g.appendChild(ring);
        g.appendChild(fill);
        seatsEl.appendChild(g);

        g.addEventListener('mouseenter', (e) => showTooltip(e, seat));
        g.addEventListener('mousemove', moveTooltip);
        g.addEventListener('mouseleave', hideTooltip);
        g.addEventListener('click', () => openPanel(seat, g));
    }});

    function showTooltip(e, seat) {{
        const partyLabel = seat.party === 'D' ? 'Democrat' : seat.party === 'R' ? 'Republican' : 'Independent';
        const photo = seat.photo_url
            ? `<img class="tooltip-photo" src="${{escAttr(seat.photo_url)}}" loading="eager" alt="" onerror="this.style.visibility='hidden'">`
            : '';
        tooltip.innerHTML = `
            <div class="tooltip-row">
                ${{photo}}
                <div>
                    <div class="tooltip-name">${{escHtml(seat.full_name)}}</div>
                    <div class="tooltip-meta">${{partyLabel}} · ${{seat.state}}${{seat.role ? ' · ' + escHtml(seat.role) : ''}}</div>
                    <div class="tooltip-hint">Click for full constitution</div>
                </div>
            </div>
        `;
        tooltip.classList.add('visible');
        moveTooltip(e);
    }}
    function moveTooltip(e) {{
        const pad = 14;
        let x = e.clientX + pad, y = e.clientY + pad;
        const rect = tooltip.getBoundingClientRect();
        if (x + rect.width > window.innerWidth - 10) x = e.clientX - rect.width - pad;
        if (y + rect.height > window.innerHeight - 10) y = e.clientY - rect.height - pad;
        tooltip.style.left = x + 'px';
        tooltip.style.top = y + 'px';
    }}
    function hideTooltip() {{ tooltip.classList.remove('visible'); }}

    let selected = null;
    function openPanel(seat, g) {{
        if (selected) selected.classList.remove('selected');
        g.classList.add('selected');
        selected = g;

        const partyLabel = seat.party === 'D' ? 'Democrat' : seat.party === 'R' ? 'Republican' : 'Independent';
        const c = seat.constitution;
        const portrait = seat.photo_url
            ? `<img class="panel-photo" src="${{escAttr(seat.photo_url)}}" loading="lazy" alt="${{escAttr(seat.full_name)}}" onerror="this.style.visibility='hidden'">`
            : '';
        const sections = [];
        sections.push(`
            <div class="panel-header">
                ${{portrait}}
                <div class="panel-id">
                    <div class="panel-name">${{escHtml(seat.full_name)}}</div>
                    <div class="panel-meta">${{partyLabel}} · ${{seat.state}}${{seat.role ? ' · ' + escHtml(seat.role) : ''}}${{c ? `<span class="confidence-tag ${{c.confidence}}">confidence: ${{c.confidence}}</span>` : ''}}</div>
                </div>
            </div>
        `);
        if (c) {{
            const issuesHtml = (c.top_issues || []).map(i => `
                <div class="issue"><div class="issue-name">${{escHtml(i.issue)}}</div><div class="issue-stance">${{escHtml(i.stance)}}</div></div>
            `).join('');
            sections.push(`
                <div class="panel-section">
                    <h3>Moral constitution</h3>
                    ${{c.core_values && c.core_values.length ? `<div class="const-block"><h3 style="margin-bottom:0.5rem">Core values</h3><ul>${{c.core_values.map(v => `<li>${{escHtml(v)}}</li>`).join('')}}</ul></div>` : ''}}
                    ${{issuesHtml ? `<div class="const-block"><h3 style="margin-bottom:0.5rem">Top issues</h3>${{issuesHtml}}</div>` : ''}}
                    ${{c.red_lines && c.red_lines.length ? `<div class="const-block"><h3 style="margin-bottom:0.5rem">Red lines</h3><ul>${{c.red_lines.map(v => `<li>${{escHtml(v)}}</li>`).join('')}}</ul></div>` : ''}}
                    ${{c.negotiation_style ? `<div class="const-block"><h3 style="margin-bottom:0.5rem">Negotiation style</h3><div class="voice">${{escHtml(c.negotiation_style)}}</div></div>` : ''}}
                    ${{c.voice_notes ? `<div class="const-block"><h3 style="margin-bottom:0.5rem">Voice</h3><div class="voice">${{escHtml(c.voice_notes)}}</div></div>` : ''}}
                    ${{c.typical_allies && c.typical_allies.length ? `<div class="const-block"><h3 style="margin-bottom:0.5rem">Typical allies</h3><div class="voice">${{c.typical_allies.map(escHtml).join(' · ')}}</div></div>` : ''}}
                </div>
            `);
        }} else {{
            sections.push(`<div class="panel-section"><h3>Moral constitution</h3><p class="voice">Not yet generated.</p></div>`);
        }}
        panelBody.innerHTML = sections.join('');
        panel.classList.add('open');
    }}
    panelClose.addEventListener('click', closePanel);
    document.addEventListener('keydown', (e) => {{ if (e.key === 'Escape') closePanel(); }});
    function closePanel() {{
        panel.classList.remove('open');
        if (selected) selected.classList.remove('selected');
        selected = null;
    }}

    function escHtml(s) {{
        if (s === null || s === undefined) return '';
        return String(s).replace(/[&<>"']/g, c => ({{
            '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
        }})[c]);
    }}
    function escAttr(s) {{ return escHtml(s); }}
}})();
</script>
</body>
</html>"""


def _render_page(cards_html: str, generated_at: str, total: int, payload: str) -> str:
    return _PAGE_TEMPLATE.format(
        cards_html=cards_html,
        generated_at=generated_at,
        total=total,
        plural="" if total == 1 else "s",
        payload=payload,
    )
