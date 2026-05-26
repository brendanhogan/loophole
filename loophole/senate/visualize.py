"""Generate the self-contained HTML chamber visualizer.

Renders the 100-seat Senate semicircle (4 concentric arcs, 22/24/26/28 seats).
Seats are positioned by party (D-caucus left, R right) and alphabetized within
each row, so geometry stays stable across runs. Vote outcome is encoded by
fill color. Click a seat → side panel with full reasoning and constitution.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

from loophole.senate.models import (
    BillReaction,
    MoralConstitution,
    Party,
    SenateSession,
    Senator,
    VoteStance,
)

# Seat layout: 4 concentric arcs that sum to 100.
# Inner-to-outer. Roughly matches the real Senate chamber's curve.
ROW_SIZES = [22, 24, 26, 28]

# Within each row, D-caucus on viewer-left, R on viewer-right, proportional
# to the overall 47/53 split (Sanders & King caucus with Dems).
ROW_DEM_COUNT = [10, 11, 12, 14]  # sums to 47


def generate_html(
    state: SenateSession,
    senators: list[Senator],
    constitutions: dict[str, MoralConstitution],
    output_path: str | None = None,
) -> str:
    reactions_by_name: dict[str, BillReaction] = {r.senator_full_name: r for r in state.reactions}
    has_bill = bool(state.reactions)

    seats = _layout_seats(senators, constitutions, reactions_by_name)
    tally = _compute_tally(state.reactions) if has_bill else None
    flip_amendments = _collect_flip_amendments(state.reactions, senators) if has_bill else []

    payload = {
        "bill_name": state.bill_name,
        "has_bill": has_bill,
        "tally": tally,
        "total_voting": len([r for r in state.reactions]),
        "total_senators": len(senators),
        "seats": seats,
        "flip_amendments": flip_amendments,
        "generated_at": datetime.now().strftime("%b %d, %Y · %I:%M %p"),
    }

    payload_json = json.dumps(payload).replace("</", "<\\/")

    title = state.bill_name if has_bill else "Senate Roster"
    page = _HTML_TEMPLATE.format(
        title=_escape(title),
        payload=payload_json,
    )

    if output_path:
        out = Path(output_path)
    else:
        out = Path("sessions") / state.session_id / "chamber.html"
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return str(out)


def _layout_seats(
    senators: list[Senator],
    constitutions: dict[str, MoralConstitution],
    reactions_by_name: dict[str, BillReaction],
) -> list[dict]:
    """Place every senator on the semicircle and pair with their data."""
    # Partition into D-caucus and R. Independents go with their caucus.
    def is_dem_caucus(s: Senator) -> bool:
        if s.party == Party.DEMOCRAT:
            return True
        if s.party == Party.INDEPENDENT and s.caucus == Party.DEMOCRAT:
            return True
        return False

    dems = sorted([s for s in senators if is_dem_caucus(s)], key=lambda s: s.short_name)
    reps = sorted([s for s in senators if not is_dem_caucus(s)], key=lambda s: s.short_name)

    # SVG viewport
    cx, cy = 500.0, 480.0
    base_radius = 180.0
    row_step = 75.0

    seats: list[dict] = []
    dem_idx = 0
    rep_idx = 0

    for row_idx, row_size in enumerate(ROW_SIZES):
        radius = base_radius + row_idx * row_step
        dem_n = ROW_DEM_COUNT[row_idx]
        rep_n = row_size - dem_n

        # Sweep across the full semicircle from left (π) to right (0).
        # Insert a small gap in the middle (the aisle).
        aisle = 0.08  # radians of gap
        left_span = math.pi - aisle - 0.08  # leave a tiny border on each side
        right_span = math.pi - aisle - 0.08
        # left half goes from π - 0.04 → (π/2 + aisle/2)
        # right half goes from (π/2 - aisle/2) → 0.04
        left_start = math.pi - 0.04
        left_end = math.pi / 2 + aisle / 2
        right_start = math.pi / 2 - aisle / 2
        right_end = 0.04

        # Place Dems on left (alphabetical, fanning from outer-left toward aisle)
        for k in range(dem_n):
            t = k / max(dem_n - 1, 1)
            angle = left_start + t * (left_end - left_start)
            senator = dems[dem_idx]
            dem_idx += 1
            seats.append(_seat_record(senator, constitutions, reactions_by_name, cx, cy, radius, angle))

        # Place Reps on right (alphabetical, fanning from aisle toward outer-right)
        for k in range(rep_n):
            t = k / max(rep_n - 1, 1)
            angle = right_start + t * (right_end - right_start)
            senator = reps[rep_idx]
            rep_idx += 1
            seats.append(_seat_record(senator, constitutions, reactions_by_name, cx, cy, radius, angle))

    return seats


def _seat_record(
    senator: Senator,
    constitutions: dict[str, MoralConstitution],
    reactions_by_name: dict[str, BillReaction],
    cx: float,
    cy: float,
    radius: float,
    angle: float,
) -> dict:
    x = cx + radius * math.cos(angle)
    y = cy - radius * math.sin(angle)
    constitution = constitutions.get(senator.full_name)
    reaction = reactions_by_name.get(senator.full_name)
    return {
        "full_name": senator.full_name,
        "short_name": senator.short_name,
        "party": senator.party.value,
        "state": senator.state,
        "role": senator.role,
        "photo_url": senator.photo_url,
        "x": round(x, 2),
        "y": round(y, 2),
        "constitution": _serialize_constitution(constitution),
        "reaction": _serialize_reaction(reaction),
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
        "citations": [
            {"claim": ct.claim, "source_kind": ct.source_kind, "detail": ct.detail}
            for ct in c.citations
        ],
    }


def _serialize_reaction(r: BillReaction | None) -> dict | None:
    if r is None:
        return None
    return {
        "vote": r.vote.value,
        "confidence": r.confidence,
        "reasoning": r.reasoning,
        "supported": r.key_provisions_supported,
        "opposed": r.key_provisions_opposed,
        "amendments": [
            {
                "description": a.description,
                "rationale": a.rationale,
                "would_flip_vote": a.would_flip_vote,
            }
            for a in r.amendments
        ],
    }


def _compute_tally(reactions: list[BillReaction]) -> dict:
    by_vote = {v.value: 0 for v in VoteStance}
    for r in reactions:
        by_vote[r.vote.value] += 1

    # Simple pass/fail projection — counts lean_yes as yes-leaning, lean_no as no-leaning.
    yes_total = by_vote["yes"] + by_vote["lean_yes"]
    no_total = by_vote["no"] + by_vote["lean_no"]
    undecided = by_vote["undecided"]

    return {
        "by_vote": by_vote,
        "yes_total": yes_total,
        "no_total": no_total,
        "undecided": undecided,
        "needed_for_passage": 51,
        "needed_for_cloture": 60,
    }


def _collect_flip_amendments(
    reactions: list[BillReaction], senators: list[Senator]
) -> list[dict]:
    """Amendments marked as vote-flipping. Sorted by which senators are closest to flipping."""
    by_name = {s.full_name: s for s in senators}
    vote_weight = {
        "lean_no": 1,
        "no": 2,
        "undecided": 3,
        "lean_yes": 4,
        "yes": 5,
    }
    out: list[dict] = []
    for r in reactions:
        for a in r.amendments:
            if not a.would_flip_vote:
                continue
            s = by_name.get(r.senator_full_name)
            out.append({
                "senator_short": s.short_name if s else r.senator_full_name,
                "senator_party": s.party.value if s else "",
                "senator_state": s.state if s else "",
                "current_vote": r.vote.value,
                "description": a.description,
                "rationale": a.rationale,
                "_sort": vote_weight.get(r.vote.value, 99),
            })
    out.sort(key=lambda x: (x["_sort"], x["senator_short"]))
    for x in out:
        x.pop("_sort", None)
    return out


def _escape(text: str) -> str:
    import html
    return html.escape(text)


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Loophole · Senate — {title}</title>
<style>
:root {{
    --bg: #0e1320;
    --bg-2: #141a2b;
    --surface: #1a2238;
    --border: #2a3454;
    --border-soft: #1f273f;
    --text: #e8e4d4;
    --text-dim: #8a93a8;
    --text-mute: #5a6378;
    --text-bright: #faf6e8;
    --accent: #d4af6c;
    --accent-dim: #a08654;
    --party-d: #4f7fc7;
    --party-r: #c75858;
    --party-i: #9b8db5;
    --vote-yes: #5fb878;
    --vote-lean-yes: #8fc99e;
    --vote-undecided: #b5a06a;
    --vote-lean-no: #d18585;
    --vote-no: #d5524a;
    --vote-absent: #3d4663;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{ height: 100%; }}
body {{
    background: radial-gradient(ellipse at top, var(--bg-2) 0%, var(--bg) 65%);
    color: var(--text);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    line-height: 1.55;
    min-height: 100vh;
    overflow-x: hidden;
}}
.headline {{
    font-family: 'Iowan Old Style', 'Palatino', 'Georgia', serif;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: var(--text-bright);
}}
.mono {{
    font-family: 'SF Mono', 'JetBrains Mono', Menlo, Consolas, monospace;
}}

.page {{ max-width: 1400px; margin: 0 auto; padding: 2.5rem 2rem 4rem; }}

/* ----- Header ----- */
header.hero {{ margin-bottom: 1.5rem; }}
.eyebrow {{
    color: var(--accent);
    font-size: 0.75rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}}
h1.title {{ font-size: 2.4rem; line-height: 1.15; margin-bottom: 0.4rem; }}
.subtitle {{ color: var(--text-dim); font-size: 0.95rem; }}

/* ----- Tally bar ----- */
.tally {{
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 2rem;
    align-items: center;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin: 1.5rem 0;
}}
.tally-bar {{
    display: flex;
    height: 14px;
    border-radius: 999px;
    overflow: hidden;
    background: var(--bg-2);
    border: 1px solid var(--border-soft);
}}
.tally-seg {{ height: 100%; }}
.tally-yes {{ background: var(--vote-yes); }}
.tally-lean-yes {{ background: var(--vote-lean-yes); }}
.tally-undecided {{ background: var(--vote-undecided); }}
.tally-lean-no {{ background: var(--vote-lean-no); }}
.tally-no {{ background: var(--vote-no); }}
.tally-counts {{
    display: flex;
    gap: 1.25rem;
    font-size: 0.85rem;
    color: var(--text-dim);
    margin-top: 0.75rem;
}}
.tally-counts strong {{ color: var(--text-bright); font-weight: 600; margin-right: 0.3rem; }}
.tally-counts .dot {{
    display: inline-block;
    width: 8px; height: 8px; border-radius: 50%;
    margin-right: 0.4rem;
    vertical-align: middle;
}}
.tally-projection {{
    text-align: right;
    font-size: 0.85rem;
    color: var(--text-dim);
}}
.tally-projection .big {{
    display: block;
    font-family: 'Iowan Old Style', 'Palatino', 'Georgia', serif;
    font-size: 2rem;
    color: var(--text-bright);
    line-height: 1;
    margin-bottom: 0.2rem;
}}
.tally-projection.pass .big {{ color: var(--vote-yes); }}
.tally-projection.fail .big {{ color: var(--vote-no); }}

/* ----- Chamber ----- */
.chamber-wrap {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem 1.5rem 1rem;
    position: relative;
}}
.chamber-svg {{ width: 100%; height: auto; display: block; }}
.dais {{
    fill: var(--bg-2);
    stroke: var(--border);
    stroke-width: 1;
}}
.dais-label {{
    fill: var(--text-mute);
    font-size: 11px;
    text-anchor: middle;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}}
.seat {{
    cursor: pointer;
    transition: transform 0.15s ease, filter 0.15s ease;
    transform-origin: center;
    transform-box: fill-box;
}}
.seat:hover {{ transform: scale(1.4); filter: brightness(1.2); }}
.seat.selected {{
    transform: scale(1.5);
    filter: drop-shadow(0 0 6px var(--accent));
}}
.seat-ring {{
    fill: none;
    stroke-width: 1.5;
}}
.seat-d .seat-ring {{ stroke: var(--party-d); }}
.seat-r .seat-ring {{ stroke: var(--party-r); }}
.seat-i .seat-ring {{ stroke: var(--party-i); }}
.seat-yes {{ fill: var(--vote-yes); }}
.seat-lean_yes {{ fill: var(--vote-lean-yes); }}
.seat-undecided {{ fill: var(--vote-undecided); }}
.seat-lean_no {{ fill: var(--vote-lean-no); }}
.seat-no {{ fill: var(--vote-no); }}
.seat-absent {{ fill: var(--vote-absent); }}
.seat-roster {{ fill: var(--text-mute); }}

/* Legend ribbon */
.legend {{
    display: flex;
    justify-content: center;
    gap: 1.5rem;
    margin: 0.5rem 0 0.5rem;
    font-size: 0.8rem;
    color: var(--text-dim);
    flex-wrap: wrap;
}}
.legend-item {{ display: flex; align-items: center; gap: 0.45rem; }}
.legend-dot {{
    width: 12px; height: 12px; border-radius: 50%;
    border: 1.5px solid var(--text-mute);
}}

/* Tooltip */
.tooltip {{
    position: fixed;
    pointer-events: none;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 0.7rem;
    font-size: 0.85rem;
    max-width: 360px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    opacity: 0;
    transition: opacity 0.1s ease;
    z-index: 100;
}}
.tooltip.visible {{ opacity: 1; }}
.tooltip-row {{ display: flex; gap: 0.7rem; align-items: flex-start; }}
.tooltip-photo {{
    width: 48px; height: 60px;
    border-radius: 4px;
    object-fit: cover;
    background: var(--bg-2);
    flex-shrink: 0;
}}
.tooltip-text {{ min-width: 0; }}
.tooltip-name {{ font-weight: 600; color: var(--text-bright); }}
.tooltip-meta {{ color: var(--text-dim); font-size: 0.78rem; margin-top: 0.1rem; }}
.tooltip-vote {{
    display: inline-block;
    padding: 1px 8px;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 600;
    margin-top: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #0e1320;
}}
.tooltip-quote {{
    color: var(--text);
    font-style: italic;
    margin-top: 0.45rem;
    font-size: 0.8rem;
    line-height: 1.45;
}}

/* Side panel */
.panel {{
    position: fixed;
    top: 0; right: 0;
    width: min(520px, 92vw);
    height: 100vh;
    background: var(--bg-2);
    border-left: 1px solid var(--border);
    transform: translateX(100%);
    transition: transform 0.25s ease;
    overflow-y: auto;
    z-index: 50;
    box-shadow: -8px 0 32px rgba(0,0,0,0.4);
}}
.panel.open {{ transform: translateX(0); }}
.panel-close {{
    position: absolute;
    top: 1rem; right: 1rem;
    background: none;
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text-dim);
    width: 32px; height: 32px;
    cursor: pointer;
    font-size: 1rem;
    transition: all 0.15s;
}}
.panel-close:hover {{ color: var(--text-bright); border-color: var(--accent); }}
.panel-body {{ padding: 2rem 1.75rem; }}
.panel-header {{ display: flex; gap: 1.1rem; align-items: flex-start; margin-bottom: 0.25rem; }}
.panel-photo {{
    width: 92px; height: 115px;
    border-radius: 6px;
    object-fit: cover;
    background: var(--surface);
    border: 1px solid var(--border);
    flex-shrink: 0;
}}
.panel-id {{ min-width: 0; flex: 1; }}
.panel-name {{ font-family: 'Iowan Old Style','Palatino','Georgia',serif; font-size: 1.65rem; color: var(--text-bright); line-height: 1.15; }}
.panel-meta {{ color: var(--text-dim); margin-top: 0.4rem; font-size: 0.88rem; }}
.panel-section {{ margin-top: 1.75rem; }}
.panel-section h3 {{
    font-size: 0.72rem;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.16em;
    margin-bottom: 0.7rem;
    font-weight: 600;
}}
.vote-chip {{
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem 0.85rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #0e1320;
    margin-top: 0.6rem;
}}
.vote-chip.yes {{ background: var(--vote-yes); }}
.vote-chip.lean_yes {{ background: var(--vote-lean-yes); }}
.vote-chip.undecided {{ background: var(--vote-undecided); }}
.vote-chip.lean_no {{ background: var(--vote-lean-no); }}
.vote-chip.no {{ background: var(--vote-no); }}
.vote-conf {{ font-weight: 400; color: rgba(14,19,32,0.6); margin-left: 0.4rem; }}
.reasoning {{
    color: var(--text);
    font-size: 0.95rem;
    line-height: 1.6;
    font-style: italic;
    padding-left: 0.85rem;
    border-left: 2px solid var(--accent-dim);
}}
.provision-list {{ list-style: none; padding: 0; margin: 0; }}
.provision-list li {{
    padding: 0.45rem 0.75rem;
    border-radius: 6px;
    margin-bottom: 0.35rem;
    font-size: 0.88rem;
    line-height: 1.45;
}}
.provision-list.supported li {{ background: rgba(95,184,120,0.08); border-left: 3px solid var(--vote-yes); }}
.provision-list.opposed li {{ background: rgba(213,82,74,0.08); border-left: 3px solid var(--vote-no); }}
.amendment {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.6rem;
}}
.amendment.flips {{ border-left: 3px solid var(--accent); }}
.amendment .desc {{ font-weight: 500; color: var(--text-bright); margin-bottom: 0.3rem; font-size: 0.92rem; }}
.amendment .rationale {{ color: var(--text-dim); font-size: 0.85rem; font-style: italic; line-height: 1.5; }}
.amendment .flip-flag {{
    display: inline-block;
    font-size: 0.65rem;
    color: var(--accent);
    margin-top: 0.4rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}}

/* Constitution sub-sections */
.constitution-block {{ background: var(--surface); border: 1px solid var(--border-soft); border-radius: 8px; padding: 0.9rem 1.1rem; margin-bottom: 0.6rem; }}
.constitution-block .label {{ font-size: 0.72rem; color: var(--accent); text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 0.5rem; }}
.constitution-block ul {{ list-style: none; padding: 0; margin: 0; }}
.constitution-block ul li {{ padding: 0.25rem 0; font-size: 0.88rem; line-height: 1.5; color: var(--text); }}
.constitution-block ul li::before {{ content: "—  "; color: var(--text-mute); }}
.issue {{ padding: 0.55rem 0; border-bottom: 1px solid var(--border-soft); }}
.issue:last-child {{ border-bottom: none; }}
.issue .issue-name {{ color: var(--text-bright); font-size: 0.9rem; font-weight: 500; margin-bottom: 0.25rem; }}
.issue .issue-stance {{ color: var(--text-dim); font-size: 0.85rem; line-height: 1.5; }}
.voice {{ color: var(--text); font-size: 0.88rem; line-height: 1.55; font-style: italic; }}
.confidence-tag {{
    display: inline-block;
    font-size: 0.7rem;
    padding: 1px 8px;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-left: 0.5rem;
}}
.confidence-tag.high {{ background: rgba(95,184,120,0.18); color: var(--vote-yes); }}
.confidence-tag.medium {{ background: rgba(181,160,106,0.18); color: var(--vote-undecided); }}
.confidence-tag.low {{ background: rgba(213,82,74,0.18); color: var(--vote-no); }}

/* Flip-amendments panel (below chamber) */
.flips-section {{ margin-top: 2.5rem; }}
.flips-section h2 {{
    font-family: 'Iowan Old Style','Palatino','Georgia',serif;
    font-size: 1.3rem;
    color: var(--text-bright);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}}
.flips-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 0.75rem; }}
.flip-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
}}
.flip-card .who {{
    font-size: 0.82rem;
    color: var(--text-dim);
    margin-bottom: 0.45rem;
}}
.flip-card .who strong {{ color: var(--text-bright); }}
.flip-card .desc {{ color: var(--text); font-size: 0.92rem; font-weight: 500; margin-bottom: 0.35rem; }}
.flip-card .rationale {{ color: var(--text-dim); font-size: 0.83rem; font-style: italic; line-height: 1.5; }}

/* Footer */
footer {{ margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid var(--border-soft); color: var(--text-mute); font-size: 0.78rem; text-align: center; }}
footer a {{ color: var(--accent-dim); text-decoration: none; }}

@media (max-width: 720px) {{
    .page {{ padding: 1.5rem 1rem 3rem; }}
    h1.title {{ font-size: 1.7rem; }}
    .tally {{ grid-template-columns: 1fr; }}
    .tally-projection {{ text-align: left; }}
}}
</style>
</head>
<body>
<div class="page">
    <header class="hero">
        <div class="eyebrow">Loophole · Senate Simulation</div>
        <h1 id="bill-title" class="title headline"></h1>
        <p id="subtitle" class="subtitle"></p>
    </header>

    <section id="tally-section" class="tally" hidden>
        <div>
            <div class="tally-bar" id="tally-bar"></div>
            <div class="tally-counts" id="tally-counts"></div>
        </div>
        <div class="tally-projection" id="projection">
            <span class="big" id="projection-big"></span>
            <span id="projection-detail"></span>
        </div>
    </section>

    <section class="chamber-wrap">
        <svg class="chamber-svg" viewBox="0 0 1000 560" preserveAspectRatio="xMidYMid meet">
            <rect x="380" y="490" width="240" height="40" rx="4" class="dais"></rect>
            <text x="500" y="515" class="dais-label">Presiding Officer</text>
            <g id="seats"></g>
        </svg>
        <div class="legend" id="legend"></div>
    </section>

    <section class="flips-section" id="flips-section" hidden>
        <h2>What would change votes</h2>
        <p class="subtitle" style="margin-bottom: 1rem;">Amendments senators say would move them toward yes.</p>
        <div class="flips-grid" id="flips-grid"></div>
    </section>

    <footer>
        Generated <span id="generated-at"></span> · Each senator's constitution and reaction is LLM-synthesized from public-record knowledge. Treat as a model, not a prediction.
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

    // --- Header ---
    document.getElementById('bill-title').textContent = data.has_bill ? data.bill_name : 'United States Senate · Roster';
    document.getElementById('generated-at').textContent = data.generated_at || '';
    const subtitle = document.getElementById('subtitle');
    if (data.has_bill) {{
        subtitle.textContent = `${{data.total_voting}} of ${{data.total_senators}} senators responded.`;
    }} else {{
        subtitle.textContent = `Click any seat to view that senator's moral constitution.`;
    }}

    // --- Tally ---
    if (data.has_bill && data.tally) {{
        const tallySection = document.getElementById('tally-section');
        tallySection.hidden = false;
        const t = data.tally.by_vote;
        const total = Math.max(1, t.yes + t.lean_yes + t.undecided + t.lean_no + t.no);
        const bar = document.getElementById('tally-bar');
        bar.innerHTML = [
            ['yes', t.yes], ['lean-yes', t.lean_yes], ['undecided', t.undecided],
            ['lean-no', t.lean_no], ['no', t.no]
        ].map(([cls, n]) => `<div class="tally-seg tally-${{cls}}" style="width:${{(n/total*100).toFixed(2)}}%"></div>`).join('');
        const counts = document.getElementById('tally-counts');
        counts.innerHTML = [
            ['yes', 'Yes', t.yes],
            ['lean-yes', 'Lean Yes', t.lean_yes],
            ['undecided', 'Undecided', t.undecided],
            ['lean-no', 'Lean No', t.lean_no],
            ['no', 'No', t.no],
        ].map(([cls, label, n]) => `<span><span class="dot tally-${{cls}}" style="background:var(--vote-${{cls}})"></span><strong>${{n}}</strong>${{label}}</span>`).join('');

        // Projection
        const projection = document.getElementById('projection');
        const proj = document.getElementById('projection-big');
        const detail = document.getElementById('projection-detail');
        const yesTotal = data.tally.yes_total;
        const noTotal = data.tally.no_total;
        proj.textContent = `${{yesTotal}}–${{noTotal}}`;
        if (yesTotal >= 60) {{
            detail.textContent = `Cloture invoked, passes`;
            projection.classList.add('pass');
        }} else if (yesTotal >= 51) {{
            detail.textContent = `Majority but cloture in doubt`;
        }} else {{
            detail.textContent = `Falls short of majority`;
            projection.classList.add('fail');
        }}
    }}

    // --- Legend ---
    const legend = document.getElementById('legend');
    const legendItems = data.has_bill ? [
        ['yes', 'Yes'], ['lean_yes', 'Lean Yes'], ['undecided', 'Undecided'],
        ['lean_no', 'Lean No'], ['no', 'No'], ['absent', 'No response']
    ] : [
        ['party-d', 'Democratic caucus'], ['party-r', 'Republican'], ['party-i', 'Independent']
    ];
    legend.innerHTML = legendItems.map(([cls, label]) => {{
        const color = data.has_bill ? `var(--vote-${{cls.replace('_','-')}})` : `var(--${{cls}})`;
        return `<span class="legend-item"><span class="legend-dot" style="background:${{color}}"></span>${{label}}</span>`;
    }}).join('');

    // --- Seats ---
    const partyClass = (p) => p === 'D' ? 'seat-d' : p === 'R' ? 'seat-r' : 'seat-i';
    const voteClass = (seat) => {{
        if (!data.has_bill) return 'seat-roster';
        if (!seat.reaction) return 'seat-absent';
        return `seat-${{seat.reaction.vote}}`;
    }};

    data.seats.forEach((seat, idx) => {{
        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        g.setAttribute('class', `seat ${{partyClass(seat.party)}}`);
        g.setAttribute('data-idx', idx);

        const ring = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        ring.setAttribute('cx', seat.x);
        ring.setAttribute('cy', seat.y);
        ring.setAttribute('r', 11);
        ring.setAttribute('class', 'seat-ring');

        const fill = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        fill.setAttribute('cx', seat.x);
        fill.setAttribute('cy', seat.y);
        fill.setAttribute('r', 8);
        fill.setAttribute('class', voteClass(seat));

        g.appendChild(ring);
        g.appendChild(fill);
        seatsEl.appendChild(g);

        g.addEventListener('mouseenter', (e) => showTooltip(e, seat));
        g.addEventListener('mousemove', moveTooltip);
        g.addEventListener('mouseleave', hideTooltip);
        g.addEventListener('click', () => openPanel(seat, g));
    }});

    // --- Tooltip ---
    function showTooltip(e, seat) {{
        const partyLabel = seat.party === 'D' ? 'Democrat' : seat.party === 'R' ? 'Republican' : 'Independent';
        let voteHtml = '';
        let quoteHtml = '';
        if (seat.reaction) {{
            const v = seat.reaction.vote;
            const bg = `var(--vote-${{v.replace('_','-')}})`;
            voteHtml = `<span class="tooltip-vote" style="background:${{bg}}">${{v.replace('_', ' ')}} · ${{(seat.reaction.confidence*100).toFixed(0)}}%</span>`;
            const firstSentence = (seat.reaction.reasoning || '').split(/(?<=[.!?])\\s/)[0];
            if (firstSentence) quoteHtml = `<div class="tooltip-quote">"${{escapeHtml(firstSentence)}}"</div>`;
        }} else if (data.has_bill) {{
            voteHtml = `<span class="tooltip-vote" style="background:var(--vote-absent); color: var(--text)">No response</span>`;
        }}
        const photoHtml = seat.photo_url
            ? `<img class="tooltip-photo" src="${{escapeAttr(seat.photo_url)}}" alt="" loading="eager" onerror="this.style.visibility='hidden'">`
            : '';
        tooltip.innerHTML = `
            <div class="tooltip-row">
                ${{photoHtml}}
                <div class="tooltip-text">
                    <div class="tooltip-name">${{escapeHtml(seat.full_name)}}</div>
                    <div class="tooltip-meta">${{partyLabel}} · ${{seat.state}}${{seat.role ? ' · ' + escapeHtml(seat.role) : ''}}</div>
                    ${{voteHtml}}
                    ${{quoteHtml}}
                </div>
            </div>
        `;
        tooltip.classList.add('visible');
        moveTooltip(e);
    }}
    function moveTooltip(e) {{
        const pad = 14;
        let x = e.clientX + pad;
        let y = e.clientY + pad;
        const rect = tooltip.getBoundingClientRect();
        if (x + rect.width > window.innerWidth - 10) x = e.clientX - rect.width - pad;
        if (y + rect.height > window.innerHeight - 10) y = e.clientY - rect.height - pad;
        tooltip.style.left = x + 'px';
        tooltip.style.top = y + 'px';
    }}
    function hideTooltip() {{ tooltip.classList.remove('visible'); }}

    // --- Panel ---
    let selected = null;
    function openPanel(seat, g) {{
        if (selected) selected.classList.remove('selected');
        g.classList.add('selected');
        selected = g;

        const partyLabel = seat.party === 'D' ? 'Democrat' : seat.party === 'R' ? 'Republican' : 'Independent';
        const c = seat.constitution;
        const r = seat.reaction;
        const sections = [];

        const portraitHtml = seat.photo_url
            ? `<img class="panel-photo" src="${{escapeAttr(seat.photo_url)}}" alt="${{escapeAttr(seat.full_name)}}" loading="lazy" onerror="this.style.visibility='hidden'">`
            : '';
        sections.push(`
            <div class="panel-header">
                ${{portraitHtml}}
                <div class="panel-id">
                    <div class="panel-name">${{escapeHtml(seat.full_name)}}</div>
                    <div class="panel-meta">${{partyLabel}} · ${{seat.state}}${{seat.role ? ' · ' + escapeHtml(seat.role) : ''}}${{c ? `<span class="confidence-tag ${{c.confidence}}">confidence: ${{c.confidence}}</span>` : ''}}</div>
                </div>
            </div>
        `);

        if (r) {{
            sections.push(`
                <div class="panel-section">
                    <h3>Vote on this bill</h3>
                    <span class="vote-chip ${{r.vote}}">${{r.vote.replace('_',' ')}}<span class="vote-conf">${{(r.confidence*100).toFixed(0)}}%</span></span>
                </div>
                <div class="panel-section">
                    <h3>In their words</h3>
                    <div class="reasoning">${{escapeHtml(r.reasoning || '')}}</div>
                </div>
            `);
            if (r.supported && r.supported.length) {{
                sections.push(`
                    <div class="panel-section">
                        <h3>Provisions they support</h3>
                        <ul class="provision-list supported">${{r.supported.map(p => `<li>${{escapeHtml(p)}}</li>`).join('')}}</ul>
                    </div>
                `);
            }}
            if (r.opposed && r.opposed.length) {{
                sections.push(`
                    <div class="panel-section">
                        <h3>Provisions they oppose</h3>
                        <ul class="provision-list opposed">${{r.opposed.map(p => `<li>${{escapeHtml(p)}}</li>`).join('')}}</ul>
                    </div>
                `);
            }}
            if (r.amendments && r.amendments.length) {{
                sections.push(`
                    <div class="panel-section">
                        <h3>Amendments they would demand</h3>
                        ${{r.amendments.map(a => `
                            <div class="amendment ${{a.would_flip_vote ? 'flips' : ''}}">
                                <div class="desc">${{escapeHtml(a.description)}}</div>
                                <div class="rationale">${{escapeHtml(a.rationale)}}</div>
                                ${{a.would_flip_vote ? '<div class="flip-flag">→ would change my vote</div>' : ''}}
                            </div>
                        `).join('')}}
                    </div>
                `);
            }}
        }} else if (data.has_bill) {{
            sections.push(`<div class="panel-section"><h3>Vote</h3><div class="reasoning">No response recorded for this bill.</div></div>`);
        }}

        if (c) {{
            const issuesHtml = (c.top_issues || []).map(i => `
                <div class="issue">
                    <div class="issue-name">${{escapeHtml(i.issue)}}</div>
                    <div class="issue-stance">${{escapeHtml(i.stance)}}</div>
                </div>
            `).join('');
            sections.push(`
                <div class="panel-section">
                    <h3>Moral constitution</h3>
                    ${{c.core_values && c.core_values.length ? `<div class="constitution-block"><div class="label">Core values</div><ul>${{c.core_values.map(v => `<li>${{escapeHtml(v)}}</li>`).join('')}}</ul></div>` : ''}}
                    ${{issuesHtml ? `<div class="constitution-block"><div class="label">Top issues</div>${{issuesHtml}}</div>` : ''}}
                    ${{c.red_lines && c.red_lines.length ? `<div class="constitution-block"><div class="label">Red lines</div><ul>${{c.red_lines.map(v => `<li>${{escapeHtml(v)}}</li>`).join('')}}</ul></div>` : ''}}
                    ${{c.negotiation_style ? `<div class="constitution-block"><div class="label">Negotiation style</div><div class="voice">${{escapeHtml(c.negotiation_style)}}</div></div>` : ''}}
                    ${{c.voice_notes ? `<div class="constitution-block"><div class="label">Voice</div><div class="voice">${{escapeHtml(c.voice_notes)}}</div></div>` : ''}}
                    ${{c.typical_allies && c.typical_allies.length ? `<div class="constitution-block"><div class="label">Typical allies</div><div class="voice">${{c.typical_allies.map(escapeHtml).join(' · ')}}</div></div>` : ''}}
                </div>
            `);
        }} else {{
            sections.push(`<div class="panel-section"><h3>Moral constitution</h3><div class="reasoning">Not yet generated. Run <span class="mono">loophole-senate build-constitutions --only ${{seat.short_name}}</span></div></div>`);
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

    // --- Flip amendments ---
    if (data.has_bill && data.flip_amendments && data.flip_amendments.length) {{
        const flipsSection = document.getElementById('flips-section');
        flipsSection.hidden = false;
        const grid = document.getElementById('flips-grid');
        grid.innerHTML = data.flip_amendments.map(a => `
            <div class="flip-card">
                <div class="who">
                    <strong>${{escapeHtml(a.senator_short)}}</strong>
                    (${{a.senator_party}}-${{a.senator_state}}) ·
                    currently <em>${{a.current_vote.replace('_',' ')}}</em>
                </div>
                <div class="desc">${{escapeHtml(a.description)}}</div>
                <div class="rationale">${{escapeHtml(a.rationale)}}</div>
            </div>
        `).join('');
    }}

    function escapeHtml(s) {{
        if (s === null || s === undefined) return '';
        return String(s).replace(/[&<>"']/g, (c) => ({{
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
        }})[c]);
    }}
    function escapeAttr(s) {{ return escapeHtml(s); }}
}})();
</script>
</body>
</html>"""
