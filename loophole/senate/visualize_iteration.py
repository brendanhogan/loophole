"""NYT-style narrative HTML for an iteration session.

Renders the full convergence story: hero outcome, tenets, then one panel
per round (mini-chamber + what-changed + flipper photo cards), then the
final bill diff against the original.
"""

from __future__ import annotations

import html
import json
import math
from datetime import datetime
from difflib import unified_diff
from pathlib import Path

from loophole.senate.models import (
    BillReaction,
    IterationRound,
    IterationSession,
    MoralConstitution,
    Party,
    Senator,
    VoteStance,
)

ROW_SIZES = [22, 24, 26, 28]
ROW_DEM_COUNT = [10, 11, 12, 14]

VOTE_WEIGHT = {
    VoteStance.NO: 0,
    VoteStance.LEAN_NO: 1,
    VoteStance.UNDECIDED: 2,
    VoteStance.LEAN_YES: 3,
    VoteStance.YES: 4,
}

# String-keyed version, used after a card is built (card stores vote as string).
_VOTE_WEIGHT_BY_STR = {v.value: w for v, w in VOTE_WEIGHT.items()}


def VOTE_WEIGHT_GET(vote_str: str) -> int:
    return _VOTE_WEIGHT_BY_STR.get(vote_str, 5)


def generate_iteration_html(
    iter_session: IterationSession,
    senators: list[Senator],
    constitutions: dict[str, MoralConstitution],
    output_path: str | None = None,
) -> str:
    senator_by_name = {s.full_name: s for s in senators}
    state_mood_map = iter_session.state_mood_by_state  # may be empty if no constituents
    has_constituents = bool(iter_session.constituents_polled and state_mood_map)

    # Compute seat layout once (positions are constant; only fill colors change per round)
    seat_layout = _layout_seats(senators)

    # Per-round payloads
    rounds_data = []
    prev_reactions: dict[str, BillReaction] = {}
    for round_record in iter_session.rounds:
        reactions_by_name = {r.senator_full_name: r for r in round_record.reactions}

        # Find flippers: anyone whose vote moved toward yes vs previous round.
        # For round 0, there's no previous — call out the strong-yes votes as
        # "founding supporters" to give the first panel something to anchor on.
        flippers: list[dict] = []
        if round_record.round_number == 0:
            for r in round_record.reactions:
                if r.vote == VoteStance.YES and r.confidence >= 0.7:
                    flippers.append(_flipper_card(senator_by_name.get(r.senator_full_name), r, prev=None, label="opening yes"))
            flippers = flippers[:6]
        else:
            for r in round_record.reactions:
                prev = prev_reactions.get(r.senator_full_name)
                if prev is None:
                    continue
                cur_weight = VOTE_WEIGHT[r.vote]
                prev_weight = VOTE_WEIGHT[prev.vote]
                if cur_weight - prev_weight >= 2:  # meaningful flip
                    flippers.append(
                        _flipper_card(
                            senator_by_name.get(r.senator_full_name),
                            r,
                            prev=prev,
                            label=f"{prev.vote.value.replace('_',' ')} → {r.vote.value.replace('_',' ')}",
                        )
                    )
            flippers.sort(key=lambda f: -f.get("_weight_change", 0))
            flippers = flippers[:6]

        # Dissenters: senators voting no/lean_no this round, ranked by how loud
        # their pushback is (no > lean_no, then confidence). Each gets a label
        # describing how they got here (opening no / holdout / moved against).
        dissenters: list[dict] = []
        for r in round_record.reactions:
            if r.vote not in (VoteStance.NO, VoteStance.LEAN_NO):
                continue
            prev = prev_reactions.get(r.senator_full_name)
            if round_record.round_number == 0:
                label = "opening no" if r.vote == VoteStance.NO else "opening lean no"
                shift = 0
            elif prev is None:
                label = "no" if r.vote == VoteStance.NO else "lean no"
                shift = 0
            else:
                cur_w = VOTE_WEIGHT[r.vote]
                prev_w = VOTE_WEIGHT[prev.vote]
                shift = cur_w - prev_w  # negative = moved against
                if shift <= -2:
                    label = f"{prev.vote.value.replace('_',' ')} → {r.vote.value.replace('_',' ')}"
                elif prev.vote in (VoteStance.NO, VoteStance.LEAN_NO):
                    label = "holdout"
                else:
                    label = f"{prev.vote.value.replace('_',' ')} → {r.vote.value.replace('_',' ')}"
            dissenters.append(
                _dissent_card(
                    senator_by_name.get(r.senator_full_name),
                    r,
                    prev=prev,
                    label=label,
                    shift=shift,
                )
            )
        # Sort: hardest no first (vote weight asc), then biggest negative shift,
        # then highest confidence — this surfaces the strongest pushback first.
        dissenters.sort(key=lambda d: (VOTE_WEIGHT_GET(d["vote"]), d.get("_shift", 0), -d.get("_confidence", 0)))
        dissenters = [d for d in dissenters if d][:6]

        # Vote colors per seat for this round
        seat_votes = {}
        seat_alignment = {}  # "aligned" | "out_of_step" | "unknown"
        for seat in seat_layout:
            r = reactions_by_name.get(seat["full_name"])
            seat_votes[seat["full_name"]] = r.vote.value if r else "absent"
            if has_constituents and r is not None:
                state = senator_by_name.get(seat["full_name"]).state if senator_by_name.get(seat["full_name"]) else None
                mood = state_mood_map.get(state) if state else None
                seat_alignment[seat["full_name"]] = _alignment(r.vote.value, mood)
            else:
                seat_alignment[seat["full_name"]] = "unknown"

        tally = round_record.tally
        # Count out-of-step senators for this round
        out_of_step_count = sum(1 for v in seat_alignment.values() if v == "out_of_step")

        # Per-senator reactions for this round, for the click-popup. We embed
        # only what the popup needs (vote, reasoning, confidence) so the JSON
        # stays manageable.
        seat_reactions = {
            r.senator_full_name: {
                "vote": r.vote.value,
                "reasoning": r.reasoning,
                "confidence": r.confidence,
            }
            for r in round_record.reactions
        }

        rounds_data.append({
            "number": round_record.round_number + 1,
            "version": round_record.bill.version,
            "bill_name": round_record.bill.name,
            "bill_text": round_record.bill.text,
            "tally": tally,
            "yes_total": round_record.yes_total,
            "no_total": tally["no"] + tally["lean_no"],
            "undecided": tally["undecided"],
            "seat_votes": seat_votes,
            "seat_alignment": seat_alignment,
            "seat_reactions": seat_reactions,
            "out_of_step_count": out_of_step_count,
            "outcome": getattr(round_record, "outcome", "accepted"),
            "target_to_beat": getattr(round_record, "target_to_beat", None),
            "applied": [
                {"description": a.description, "how_applied": a.how_applied}
                for a in round_record.applied
            ],
            "rejected": [
                {
                    "description": a.description,
                    "violates_tenet": a.violates_tenet,
                    "rationale": a.rationale,
                }
                for a in round_record.rejected
            ],
            "revision_summary": round_record.revision_summary,
            "flippers": flippers,
            "dissenters": dissenters,
        })
        prev_reactions = reactions_by_name

    # Final bill diff — use the BEST round's bill, not the last one. With the
    # hill-climbing loop the last round may be a regression that was reverted.
    original_text = iter_session.rounds[0].bill.text if iter_session.rounds else ""
    best_round = getattr(iter_session, "best_round", None)
    if best_round is None and iter_session.rounds:
        best_round = max(iter_session.rounds, key=lambda r: r.yes_total)
    final_text = best_round.bill.text if best_round else ""
    final_diff = _compute_diff_html(original_text, final_text)

    # Outcome headline keyed off the best round.
    status_text, status_kind = _status_kind(iter_session.final_status, best_round)
    best_round_number = (best_round.round_number + 1) if best_round else None

    # Per-state mood data (only present if constituents were polled)
    state_moods_data = []
    public_yes = public_no = public_unsure = 0
    if has_constituents:
        for state, mood in sorted(state_mood_map.items(), key=lambda kv: -kv[1].yes_count / max(1, kv[1].yes_count + kv[1].no_count + kv[1].unsure_count)):
            tot = mood.yes_count + mood.no_count + mood.unsure_count
            state_moods_data.append({
                "state": state,
                "yes": mood.yes_count,
                "no": mood.no_count,
                "unsure": mood.unsure_count,
                "label": mood.label,
                "voices": [
                    {"vote": v.vote, "reasoning": v.reasoning, "key_concern": v.key_concern, "intro": v.persona_intro}
                    for v in mood.sample_voices[:3]
                ],
                "kind": _mood_kind(mood.label),
            })
            public_yes += mood.yes_count
            public_no += mood.no_count
            public_unsure += mood.unsure_count

    payload = {
        "user_prompt": iter_session.user_prompt,
        "tenets": iter_session.tenets.tenets,
        "final_status": iter_session.final_status,
        "status_text": status_text,
        "status_kind": status_kind,
        "final_yes": best_round.yes_total if best_round else 0,
        "final_no": (best_round.tally["no"] + best_round.tally["lean_no"]) if best_round else 0,
        "total_rounds": len(iter_session.rounds),
        "best_round_number": best_round_number,
        "final_bill_name": (best_round.bill.name if best_round else ""),
        "final_bill_text": final_text,
        "original_bill_text": original_text,
        "final_diff_html": final_diff,
        "seat_layout": seat_layout,
        "rounds": rounds_data,
        "has_constituents": has_constituents,
        "state_moods": state_moods_data,
        "public_yes_pct": (public_yes / max(1, public_yes + public_no + public_unsure)) if has_constituents else 0,
        "public_no_pct": (public_no / max(1, public_yes + public_no + public_unsure)) if has_constituents else 0,
        "public_total": public_yes + public_no + public_unsure,
        "generated_at": datetime.now().strftime("%b %d, %Y · %I:%M %p"),
    }

    payload_json = json.dumps(payload).replace("</", "<\\/")

    page = _PAGE_TEMPLATE.format(
        title=html.escape(iter_session.final_bill.name if iter_session.final_bill else "Senate Iteration"),
        payload=payload_json,
    )

    if output_path:
        out = Path(output_path)
    else:
        out = Path("sessions") / iter_session.session_id / "iteration.html"
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return str(out)


def _flipper_card(
    senator: Senator | None,
    reaction: BillReaction,
    prev: BillReaction | None,
    label: str,
) -> dict:
    if not senator:
        return {}
    weight_change = VOTE_WEIGHT[reaction.vote] - (VOTE_WEIGHT[prev.vote] if prev else 0)
    return {
        "full_name": senator.full_name,
        "short_name": senator.short_name,
        "party": senator.party.value,
        "state": senator.state,
        "photo_url": senator.photo_url,
        "vote": reaction.vote.value,
        "label": label,
        "reasoning": reaction.reasoning,
        "_weight_change": weight_change,
    }


def _dissent_card(
    senator: Senator | None,
    reaction: BillReaction,
    prev: BillReaction | None,
    label: str,
    shift: int,
) -> dict:
    if not senator:
        return {}
    return {
        "full_name": senator.full_name,
        "short_name": senator.short_name,
        "party": senator.party.value,
        "state": senator.state,
        "photo_url": senator.photo_url,
        "vote": reaction.vote.value,
        "label": label,
        "reasoning": reaction.reasoning,
        "_shift": shift,
        "_confidence": reaction.confidence or 0,
    }


def _layout_seats(senators: list[Senator]) -> list[dict]:
    def is_dem_caucus(s: Senator) -> bool:
        return s.party == Party.DEMOCRAT or (
            s.party == Party.INDEPENDENT and s.caucus == Party.DEMOCRAT
        )

    dems = sorted([s for s in senators if is_dem_caucus(s)], key=lambda s: s.short_name)
    reps = sorted([s for s in senators if not is_dem_caucus(s)], key=lambda s: s.short_name)

    cx, cy = 500.0, 460.0
    base_radius = 175.0
    row_step = 70.0

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
            seats.append(_seat_record(dems[dem_idx], cx, cy, radius, angle))
            dem_idx += 1
        for k in range(rep_n):
            t = k / max(rep_n - 1, 1)
            angle = right_start + t * (right_end - right_start)
            seats.append(_seat_record(reps[rep_idx], cx, cy, radius, angle))
            rep_idx += 1
    return seats


def _seat_record(senator: Senator, cx: float, cy: float, radius: float, angle: float) -> dict:
    return {
        "full_name": senator.full_name,
        "short_name": senator.short_name,
        "party": senator.party.value,
        "state": senator.state,
        "photo_url": senator.photo_url,
        "x": round(cx + radius * math.cos(angle), 2),
        "y": round(cy - radius * math.sin(angle), 2),
    }


def _alignment(senator_vote: str, mood) -> str:
    """Return 'aligned' | 'out_of_step' | 'unknown' comparing senator vote to state mood."""
    if mood is None or mood.total == 0:
        return "unknown"
    # Direction: -1 = oppose, +1 = support, 0 = mixed
    sen_dir = 0
    if senator_vote in ("yes", "lean_yes"):
        sen_dir = 1
    elif senator_vote in ("no", "lean_no"):
        sen_dir = -1

    yp = mood.yes_count / mood.total
    np_ = mood.no_count / mood.total
    if yp >= 0.55:
        mood_dir = 1
    elif np_ >= 0.55:
        mood_dir = -1
    else:
        mood_dir = 0

    if sen_dir == 0 or mood_dir == 0:
        return "aligned"  # treat as not-out-of-step if either is ambiguous
    return "aligned" if sen_dir == mood_dir else "out_of_step"


def _mood_kind(label: str) -> str:
    return {
        "strong support": "strong-support",
        "lean support": "lean-support",
        "split": "split",
        "lean oppose": "lean-oppose",
        "strong oppose": "strong-oppose",
        "no clear opinion": "unpolled",
        "unpolled": "unpolled",
    }.get(label, "split")


def _status_kind(status: str, final_round: IterationRound | None) -> tuple[str, str]:
    if status == "passed":
        return "Passes", "passes"
    if status == "stuck":
        return "Stalls", "stalls"
    if status == "converged":
        # Hill-climb found a local maximum it can't improve on without violating
        # tenets. Show as "falls short" if it's below majority, "holds" otherwise.
        if final_round and final_round.yes_total >= 51:
            return "Holds at majority", "falls-short"
        return "Falls short", "falls-short"
    if status == "max_rounds":
        return "Falls short", "falls-short"
    return "In progress", "in-progress"


def _compute_diff_html(before: str, after: str) -> str:
    diff = unified_diff(before.splitlines(), after.splitlines(), n=2, lineterm="")
    rows = []
    for line in diff:
        if line.startswith("---") or line.startswith("+++"):
            continue
        esc = html.escape(line)
        if line.startswith("@@"):
            rows.append(f'<div class="diff-hunk">{esc}</div>')
        elif line.startswith("+"):
            rows.append(f'<div class="diff-add">{esc}</div>')
        elif line.startswith("-"):
            rows.append(f'<div class="diff-del">{esc}</div>')
        else:
            rows.append(f'<div class="diff-ctx">{esc}</div>')
    return "\n".join(rows) if rows else '<div class="diff-ctx">(no textual changes)</div>'


_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Loophole · {title}</title>
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
.sans {{ font-family: 'Inter','Helvetica Neue',-apple-system,BlinkMacSystemFont,sans-serif; }}
.mono {{ font-family: 'iA Writer Mono','SF Mono','JetBrains Mono',Menlo,monospace; }}
.wrap {{ max-width: 920px; margin: 0 auto; padding: 4rem 1.5rem 6rem; }}

.eyebrow {{
    font-family: 'Inter',sans-serif;
    font-size: 0.72rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 1rem;
    font-weight: 600;
}}

/* HERO ----------------------------------------------------------------- */
.hero {{ border-bottom: 1px solid var(--rule); padding-bottom: 3rem; margin-bottom: 3rem; }}
.hero-headline {{
    font-size: 3.4rem;
    line-height: 1.05;
    letter-spacing: -0.015em;
    font-weight: 600;
    color: var(--ink);
    margin-bottom: 1.2rem;
}}
.hero-headline .verdict {{ color: var(--accent); font-style: italic; }}
.hero-headline .verdict.passes {{ color: var(--vote-yes); }}
.hero-headline .verdict.falls-short {{ color: var(--vote-undecided); }}
.hero-headline .verdict.stalls {{ color: var(--vote-no); }}

.hero-deck {{ font-size: 1.18rem; color: var(--ink-soft); margin-bottom: 1.6rem; line-height: 1.55; max-width: 720px; }}

.hero-quote {{
    border-left: 3px solid var(--accent);
    padding: 0.4rem 1.2rem;
    margin: 1.5rem 0 2rem;
    font-style: italic;
    color: var(--ink-soft);
    font-size: 1.1rem;
}}
.hero-quote-attr {{
    font-family: 'Inter',sans-serif;
    font-style: normal;
    font-size: 0.78rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--ink-mute);
    margin-top: 0.4rem;
}}

.hero-stats {{
    display: flex;
    gap: 3rem;
    flex-wrap: wrap;
    padding: 1.5rem 0;
    border-top: 1px solid var(--rule-soft);
    border-bottom: 1px solid var(--rule-soft);
    margin-top: 2rem;
}}
.hero-stat {{ }}
.hero-stat .num {{
    font-size: 2.8rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    line-height: 1;
    color: var(--ink);
}}
.hero-stat.passes .num {{ color: var(--vote-yes); }}
.hero-stat.falls-short .num {{ color: var(--vote-undecided); }}
.hero-stat.stalls .num {{ color: var(--vote-no); }}
.hero-stat .label {{
    font-family: 'Inter',sans-serif;
    font-size: 0.74rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--ink-mute);
    margin-top: 0.35rem;
    font-weight: 500;
}}

/* TENETS --------------------------------------------------------------- */
.tenets-block {{ margin: 3rem 0; }}
.section-eyebrow {{
    font-family: 'Inter',sans-serif;
    font-size: 0.74rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--ink-mute);
    margin-bottom: 0.6rem;
    font-weight: 600;
}}
.section-title {{
    font-size: 1.7rem;
    letter-spacing: -0.01em;
    color: var(--ink);
    margin-bottom: 1.5rem;
    line-height: 1.2;
}}
.tenets-list {{ list-style: none; counter-reset: tenet; }}
.tenets-list li {{
    counter-increment: tenet;
    position: relative;
    padding: 0.9rem 0 0.9rem 3rem;
    border-top: 1px solid var(--rule-soft);
    font-size: 1.05rem;
    color: var(--ink-soft);
    line-height: 1.5;
}}
.tenets-list li:last-child {{ border-bottom: 1px solid var(--rule-soft); }}
.tenets-list li::before {{
    content: counter(tenet, decimal-leading-zero);
    position: absolute;
    left: 0;
    top: 1rem;
    font-family: 'Inter',sans-serif;
    font-size: 0.85rem;
    color: var(--gold);
    font-weight: 600;
    letter-spacing: 0.06em;
}}

/* ROUND ---------------------------------------------------------------- */
.round {{ margin: 4.5rem 0; padding-top: 3rem; border-top: 2px solid var(--ink); }}
.round-header {{ margin-bottom: 1.5rem; }}
.round-number {{
    font-family: 'Inter',sans-serif;
    font-size: 0.74rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--accent);
    font-weight: 700;
    margin-bottom: 0.5rem;
}}
.round-headline {{
    font-size: 2rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: var(--ink);
    line-height: 1.15;
    margin-bottom: 0.8rem;
}}
.round-tally {{
    display: inline-flex;
    align-items: baseline;
    gap: 0.5rem;
    font-family: 'Inter',sans-serif;
    margin-bottom: 0.5rem;
}}
.round-tally .num-yes {{ font-size: 2.4rem; font-weight: 600; color: var(--vote-yes); line-height: 1; }}
.round-tally .num-no {{ font-size: 2.4rem; font-weight: 600; color: var(--vote-no); line-height: 1; }}
.round-tally .num-sep {{ font-size: 1.8rem; color: var(--ink-mute); font-weight: 300; }}
.round-tally .num-und {{ font-size: 0.85rem; color: var(--ink-mute); margin-left: 0.6rem; }}

/* mini-chamber svg */
.minichamber {{
    width: 100%;
    max-width: 720px;
    height: auto;
    display: block;
    margin: 1.5rem auto;
}}
.minichamber .dais {{ fill: var(--bg-2); stroke: var(--rule); stroke-width: 1; }}
.minichamber .dais-label {{ fill: var(--ink-mute); font-size: 11px; text-anchor: middle; letter-spacing: 0.15em; text-transform: uppercase; font-family: 'Inter',sans-serif; }}
.minichamber .seat-ring {{ fill: none; stroke-width: 1.5; }}
.minichamber .seat-d .seat-ring {{ stroke: var(--party-d); }}
.minichamber .seat-r .seat-ring {{ stroke: var(--party-r); }}
.minichamber .seat-i .seat-ring {{ stroke: var(--party-i); }}
.minichamber .vote-yes {{ fill: var(--vote-yes); }}
.minichamber .vote-lean_yes {{ fill: var(--vote-lean-yes); }}
.minichamber .vote-undecided {{ fill: var(--vote-undecided); }}
.minichamber .vote-lean_no {{ fill: var(--vote-lean-no); }}
.minichamber .vote-no {{ fill: var(--vote-no); }}
.minichamber .vote-absent {{ fill: var(--vote-absent); }}
/* Out-of-step alignment overlay: dashed gold halo */
.minichamber .alignment-overlay {{
    fill: none;
    stroke: var(--gold);
    stroke-width: 1.5;
    stroke-dasharray: 2,1.5;
}}
.out-of-step-note {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-family: 'Inter',sans-serif;
    font-size: 0.78rem;
    color: var(--gold);
    letter-spacing: 0.04em;
    margin-left: 0.6rem;
}}
.out-of-step-note .dash {{
    width: 14px; height: 14px;
    border-radius: 50%;
    border: 1.5px dashed var(--gold);
    display: inline-block;
}}

/* Constituents section */
.constituents-block {{ margin: 4rem 0 3rem; padding-top: 3rem; border-top: 2px solid var(--ink); }}
.public-vs-senate {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    margin: 1.8rem 0 2.2rem;
}}
.public-stat {{
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: 6px;
    padding: 1.4rem 1.5rem;
}}
.public-stat .label {{
    font-family: 'Inter',sans-serif;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: var(--ink-mute);
    font-weight: 600;
    margin-bottom: 0.6rem;
}}
.public-stat .bar {{
    height: 14px;
    border-radius: 99px;
    overflow: hidden;
    display: flex;
    background: var(--bg-2);
    border: 1px solid var(--rule-soft);
}}
.public-stat .bar .seg-yes {{ background: var(--vote-yes); }}
.public-stat .bar .seg-no {{ background: var(--vote-no); }}
.public-stat .bar .seg-unsure {{ background: var(--vote-undecided); }}
.public-stat .pct-text {{
    font-family: 'Inter',sans-serif;
    font-size: 0.85rem;
    color: var(--ink-soft);
    margin-top: 0.5rem;
    display: flex;
    gap: 1rem;
}}
.public-stat .pct-text strong {{ color: var(--ink); font-weight: 600; }}

.state-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 0.5rem;
    margin-top: 1rem;
}}
.state-cell {{
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: 4px;
    padding: 0.55rem 0.7rem;
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
}}
.state-cell.strong-support {{ border-left: 3px solid var(--vote-yes); }}
.state-cell.lean-support {{ border-left: 3px solid var(--vote-lean-yes); }}
.state-cell.split {{ border-left: 3px solid var(--vote-undecided); }}
.state-cell.lean-oppose {{ border-left: 3px solid var(--vote-lean-no); }}
.state-cell.strong-oppose {{ border-left: 3px solid var(--vote-no); }}
.state-cell.unpolled {{ border-left: 3px solid var(--rule); }}
.state-cell .state-code {{
    font-family: 'Charter','Georgia',serif;
    font-size: 1rem;
    font-weight: 600;
    color: var(--ink);
    letter-spacing: 0.04em;
}}
.state-cell .state-bar {{
    height: 6px;
    border-radius: 99px;
    overflow: hidden;
    display: flex;
    background: var(--bg-2);
}}
.state-cell .state-bar .seg-yes {{ background: var(--vote-yes); }}
.state-cell .state-bar .seg-no {{ background: var(--vote-no); }}
.state-cell .state-bar .seg-unsure {{ background: var(--vote-undecided); }}
.state-cell .state-label {{
    font-family: 'Inter',sans-serif;
    font-size: 0.7rem;
    color: var(--ink-mute);
    text-transform: capitalize;
}}

.voices-section {{ margin-top: 2rem; }}
.voice-card {{
    background: var(--surface);
    border: 1px solid var(--rule);
    border-left: 3px solid var(--gold);
    border-radius: 4px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
}}
.voice-card .who {{
    font-family: 'Inter',sans-serif;
    font-size: 0.78rem;
    color: var(--ink-mute);
    letter-spacing: 0.04em;
    margin-bottom: 0.4rem;
}}
.voice-card .who strong {{ color: var(--ink); font-weight: 600; }}
.voice-card .quote {{
    color: var(--ink);
    font-size: 0.95rem;
    font-style: italic;
    line-height: 1.55;
}}
.voice-card .key-concern {{
    display: inline-block;
    margin-top: 0.5rem;
    font-family: 'Inter',sans-serif;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--accent);
    background: var(--bg-2);
    padding: 2px 8px;
    border-radius: 99px;
}}

/* what-changed */
.changes {{ margin-top: 2.5rem; }}
.changes-subhead {{
    font-family: 'Inter',sans-serif;
    font-size: 0.74rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--ink-mute);
    font-weight: 600;
    margin-bottom: 0.7rem;
}}
.revision-summary {{
    font-style: italic;
    color: var(--ink-soft);
    font-size: 1.05rem;
    line-height: 1.55;
    padding: 0.8rem 1.2rem;
    border-left: 3px solid var(--gold);
    margin-bottom: 1.5rem;
    background: var(--bg-2);
}}
.amendment-block {{ margin-bottom: 1.5rem; }}
.amendment-card {{
    padding: 1.1rem 1.3rem;
    border: 1px solid var(--rule);
    border-radius: 4px;
    margin-bottom: 0.6rem;
    background: var(--surface);
}}
.amendment-card.applied {{ border-left: 3px solid var(--vote-yes); }}
.amendment-card.rejected {{ border-left: 3px solid var(--vote-no); }}
.amendment-desc {{ font-size: 1rem; font-weight: 500; color: var(--ink); margin-bottom: 0.4rem; line-height: 1.45; }}
.amendment-meta {{ font-family: 'Inter',sans-serif; font-size: 0.78rem; color: var(--ink-mute); line-height: 1.55; }}
.amendment-meta .tag {{ font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; }}
.amendment-meta .tag.applied {{ color: var(--vote-yes); }}
.amendment-meta .tag.rejected {{ color: var(--vote-no); }}

/* Flippers */
.flippers {{ margin-top: 2.5rem; }}
.flipper-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 1rem;
}}
.flipper-card {{
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: 4px;
    padding: 1rem 1.1rem;
}}
.flipper-head {{ display: flex; gap: 0.9rem; align-items: flex-start; margin-bottom: 0.7rem; }}
.flipper-photo {{
    width: 64px; height: 80px;
    border-radius: 3px;
    object-fit: cover;
    background: var(--bg-2);
    border: 1px solid var(--rule-soft);
    flex-shrink: 0;
}}
.flipper-id {{ min-width: 0; }}
.flipper-name {{
    font-family: 'Charter','Georgia',serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--ink);
    line-height: 1.15;
    margin-bottom: 0.2rem;
}}
.flipper-meta {{ font-family: 'Inter',sans-serif; font-size: 0.74rem; color: var(--ink-mute); letter-spacing: 0.04em; }}
.flipper-label {{
    display: inline-block;
    font-family: 'Inter',sans-serif;
    font-size: 0.66rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 2px 7px;
    border-radius: 99px;
    margin-top: 0.4rem;
    background: var(--bg-2);
    color: var(--accent);
    font-weight: 600;
}}
.flipper-quote {{
    font-size: 0.93rem;
    font-style: italic;
    color: var(--ink-soft);
    line-height: 1.55;
    padding-top: 0.6rem;
    border-top: 1px solid var(--rule-soft);
}}

/* Clickable seats */
.minichamber .seat.clickable {{ cursor: pointer; }}
.minichamber .seat.clickable:hover .seat-ring {{ stroke-width: 2.5; }}
.minichamber .seat.clickable:hover circle:nth-of-type(2) {{ filter: brightness(1.08); }}

/* Round outcome banner */
.round-outcome-banner {{
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    margin-top: 0.6rem;
    padding: 0.4rem 0.85rem;
    border-radius: 4px;
    font-family: 'Inter',sans-serif;
    font-size: 0.78rem;
    letter-spacing: 0.04em;
    font-weight: 600;
}}
.round-outcome-banner.accepted {{
    background: #e7efe5;
    color: var(--vote-yes);
    border: 1px solid #c8dac3;
}}
.round-outcome-banner.regressed {{
    background: #fbeae5;
    color: var(--vote-no);
    border: 1px solid #f0c8bd;
}}
.round-outcome-banner.passed {{
    background: var(--vote-yes);
    color: white;
    border: 1px solid var(--vote-yes);
}}
.round-outcome-banner .icon {{ font-size: 0.9rem; }}

/* Senator detail side panel */
.sen-panel {{
    position: fixed; inset: 0; z-index: 1000;
    display: flex; justify-content: flex-end;
}}
.sen-panel[hidden] {{ display: none; }}
.sen-panel-backdrop {{
    position: absolute; inset: 0;
    background: rgba(20, 22, 26, 0.32);
    backdrop-filter: blur(2px);
}}
.sen-panel-body {{
    position: relative;
    width: min(440px, 100%);
    height: 100vh;
    background: var(--surface);
    border-left: 1px solid var(--rule);
    box-shadow: -8px 0 24px rgba(0,0,0,0.08);
    padding: 2.5rem 2rem 2rem;
    overflow-y: auto;
    animation: slideIn 0.18s ease-out;
}}
@keyframes slideIn {{
    from {{ transform: translateX(20px); opacity: 0; }}
    to   {{ transform: translateX(0); opacity: 1; }}
}}
.sen-panel-close {{
    position: absolute; top: 0.6rem; right: 0.85rem;
    background: transparent; border: none;
    font-size: 1.7rem; color: var(--ink-mute);
    cursor: pointer; line-height: 1; padding: 0.3rem 0.5rem;
}}
.sen-panel-close:hover {{ color: var(--ink); }}
.sen-panel-head {{
    display: flex; gap: 1rem; align-items: flex-start;
    padding-bottom: 1rem; border-bottom: 1px solid var(--rule-soft);
    margin-bottom: 1.2rem;
}}
.sen-photo {{
    width: 80px; height: 100px;
    object-fit: cover; background: var(--bg-2);
    border: 1px solid var(--rule-soft);
    border-radius: 3px; flex-shrink: 0;
}}
.sen-name {{
    font-family: 'Charter','Georgia',serif;
    font-size: 1.4rem; font-weight: 600;
    color: var(--ink); line-height: 1.15;
    margin-bottom: 0.25rem;
}}
.sen-meta {{
    font-family: 'Inter',sans-serif;
    font-size: 0.78rem; color: var(--ink-mute);
    letter-spacing: 0.04em;
}}
.sen-round-label {{
    font-family: 'Inter',sans-serif;
    font-size: 0.7rem; color: var(--accent);
    letter-spacing: 0.16em; text-transform: uppercase;
    margin-top: 0.55rem; font-weight: 600;
}}
.sen-vote-pill {{
    display: inline-block;
    padding: 0.3rem 0.8rem; border-radius: 4px;
    font-family: 'Inter',sans-serif;
    font-size: 0.78rem; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    margin-bottom: 1rem;
}}
.sen-vote-pill.vote-yes      {{ background: var(--vote-yes);      color: white; }}
.sen-vote-pill.vote-lean_yes {{ background: var(--vote-lean-yes); color: white; }}
.sen-vote-pill.vote-undecided{{ background: var(--vote-undecided); color: var(--ink); }}
.sen-vote-pill.vote-lean_no  {{ background: var(--vote-lean-no);  color: white; }}
.sen-vote-pill.vote-no       {{ background: var(--vote-no);       color: white; }}
.sen-vote-pill.vote-absent   {{ background: var(--vote-absent);   color: var(--ink); }}
.sen-reasoning {{
    font-family: 'Charter','Georgia',serif;
    font-size: 1.02rem; line-height: 1.65;
    color: var(--ink-soft);
    font-style: italic;
    padding: 0.4rem 0 1.4rem;
    border-bottom: 1px solid var(--rule-soft);
    margin-bottom: 1.4rem;
}}
.sen-traj-block {{ }}
.sen-traj-label {{
    font-family: 'Inter',sans-serif;
    font-size: 0.7rem; color: var(--ink-mute);
    letter-spacing: 0.16em; text-transform: uppercase;
    margin-bottom: 0.7rem; font-weight: 600;
}}
.sen-trajectory {{
    display: flex; gap: 0.5rem; flex-wrap: wrap;
}}
.sen-traj-pill {{
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 0.25rem 0.6rem; border-radius: 3px;
    font-family: 'Inter',sans-serif; font-size: 0.72rem;
    font-weight: 600; letter-spacing: 0.04em;
    background: var(--bg-2); color: var(--ink-soft);
    cursor: pointer;
    border: 1px solid transparent;
}}
.sen-traj-pill:hover {{ border-color: var(--rule); }}
.sen-traj-pill.current {{ border-color: var(--accent); color: var(--accent); }}
.sen-traj-pill .swatch {{
    display: inline-block; width: 9px; height: 9px; border-radius: 99px;
}}

/* Dissenters — mirror flipper layout but tagged with no-vote color */
.dissenters {{ margin-top: 2.5rem; }}
.dissent-card {{
    background: var(--surface);
    border: 1px solid var(--rule);
    border-left: 3px solid var(--vote-no);
    border-radius: 4px;
    padding: 1rem 1.1rem;
}}
.dissent-label {{
    display: inline-block;
    font-family: 'Inter',sans-serif;
    font-size: 0.66rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 2px 7px;
    border-radius: 99px;
    margin-top: 0.4rem;
    background: #fbeae5;
    color: var(--vote-no);
    font-weight: 600;
}}
.dissent-label.moved-against {{
    background: #f4dccf;
    color: #7a1d0d;
}}
.dissent-label.holdout {{
    background: var(--bg-2);
    color: var(--ink-soft);
}}

/* Final bill block */
.final-bill {{ margin-top: 5rem; padding-top: 3rem; border-top: 2px solid var(--ink); }}
.bill-meta {{
    font-family: 'Inter',sans-serif;
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    color: var(--ink-mute);
    margin-bottom: 0.6rem;
}}
.bill-text-block {{
    background: var(--surface);
    border: 1px solid var(--rule);
    padding: 2rem 2.5rem;
    font-family: 'Charter','Georgia',serif;
    font-size: 1rem;
    line-height: 1.7;
    white-space: pre-wrap;
    color: var(--ink-soft);
    margin: 1.5rem 0 2rem;
    border-radius: 4px;
}}
.diff-block {{
    background: var(--surface);
    border: 1px solid var(--rule);
    padding: 1rem 1.4rem;
    font-family: 'iA Writer Mono','SF Mono',monospace;
    font-size: 0.83rem;
    line-height: 1.5;
    overflow-x: auto;
    max-height: 540px;
    overflow-y: auto;
    border-radius: 4px;
}}
.diff-add {{ color: var(--vote-yes); background: rgba(45,125,58,0.06); }}
.diff-del {{ color: var(--vote-no); background: rgba(163,42,24,0.06); }}
.diff-hunk {{ color: var(--gold); }}
.diff-ctx {{ color: var(--ink-mute); }}

/* Footer */
footer {{ margin-top: 5rem; padding-top: 2rem; border-top: 1px solid var(--rule); color: var(--ink-mute); font-size: 0.78rem; font-family: 'Inter',sans-serif; text-align: center; }}

@media (max-width: 720px) {{
    .wrap {{ padding: 2.5rem 1.2rem 4rem; }}
    .hero-headline {{ font-size: 2.3rem; }}
    .round-headline {{ font-size: 1.4rem; }}
    .hero-stats {{ gap: 1.5rem; }}
    .hero-stat .num {{ font-size: 2rem; }}
    .bill-text-block {{ padding: 1.2rem 1.4rem; }}
}}
</style>
</head>
<body>
<div class="wrap">

<!-- HERO -->
<header class="hero">
    <div class="eyebrow">Loophole · A Legislative Simulation</div>
    <h1 id="headline" class="hero-headline"></h1>
    <p id="deck" class="hero-deck"></p>
    <div class="hero-quote">
        "<span id="prompt-text"></span>"
        <div class="hero-quote-attr">— The user's idea, in plain English</div>
    </div>
    <div class="hero-stats" id="hero-stats"></div>
</header>

<!-- TENETS -->
<section class="tenets-block">
    <div class="section-eyebrow">The Constraints</div>
    <h2 class="section-title">The non-negotiables</h2>
    <p class="hero-deck" style="margin-bottom: 1.2rem;">Before iteration began, the system extracted these immutable goals from the draft. Amendments that violate any of them are rejected — even if they would have flipped a dozen senators.</p>
    <ol id="tenets-list" class="tenets-list"></ol>
</section>

<!-- CONSTITUENTS -->
<section id="constituents-block" class="constituents-block" hidden>
    <div class="section-eyebrow">The Public</div>
    <h2 class="section-title">What the constituents said</h2>
    <p class="hero-deck" style="margin-bottom: 0.8rem;">Before any senator voted, a sample of <span id="const-total"></span> ordinary Americans (drawn from Nvidia's Nemotron-USA persona dataset, weighted across all 50 states) reacted to the initial bill. The chamber sees this distribution before voting — but is not bound to it.</p>
    <div id="public-vs-senate" class="public-vs-senate"></div>

    <h3 class="section-eyebrow" style="margin-top: 2rem;">State by state</h3>
    <div id="state-grid" class="state-grid"></div>

    <div class="voices-section">
        <h3 class="section-eyebrow">A few voices</h3>
        <div id="voices-list"></div>
    </div>
</section>

<!-- ROUNDS -->
<div id="rounds"></div>

<!-- FINAL BILL -->
<section class="final-bill">
    <div class="section-eyebrow">The Final Text</div>
    <h2 id="final-bill-name" class="section-title"></h2>
    <div class="bill-meta" id="final-bill-meta"></div>
    <div id="final-bill-text" class="bill-text-block"></div>

    <div class="section-eyebrow" style="margin-top: 3rem;">What Changed From The Original Draft</div>
    <div id="final-diff" class="diff-block"></div>
</section>

<footer>
    Generated <span id="generated-at"></span> ·
    Each senator's reasoning is LLM-synthesized from their public-record constitution.
    This is a model of how a chamber might iterate, not a prediction of what will pass.
</footer>

</div>

<!-- SENATOR DETAIL SIDE PANEL (populated on seat click) -->
<div id="senator-panel" class="sen-panel" hidden>
    <div class="sen-panel-backdrop" id="senator-panel-backdrop"></div>
    <aside class="sen-panel-body" role="dialog" aria-modal="true" aria-labelledby="sen-name">
        <button class="sen-panel-close" id="senator-panel-close" aria-label="Close">×</button>
        <div class="sen-panel-head">
            <img id="sen-photo" class="sen-photo" alt="" onerror="this.style.visibility='hidden'">
            <div>
                <div id="sen-name" class="sen-name"></div>
                <div id="sen-meta" class="sen-meta"></div>
                <div id="sen-round-label" class="sen-round-label"></div>
            </div>
        </div>
        <div id="sen-vote-pill" class="sen-vote-pill"></div>
        <div id="sen-reasoning" class="sen-reasoning"></div>
        <div class="sen-traj-block">
            <div class="sen-traj-label">Vote trajectory across rounds</div>
            <div id="sen-trajectory" class="sen-trajectory"></div>
        </div>
    </aside>
</div>

<script id="payload" type="application/json">{payload}</script>
<script>
(function() {{
    const data = JSON.parse(document.getElementById('payload').textContent);

    // ----- HERO -----
    const verdictClass = data.status_kind;
    document.getElementById('headline').innerHTML =
        `<span class="verdict ${{verdictClass}}">${{data.status_text}}.</span> ${{escapeHtml(data.final_bill_name || '')}}`;

    const deck = document.getElementById('deck');
    if (data.final_status === 'passed') {{
        deck.textContent = `After ${{data.total_rounds}} round${{data.total_rounds===1?'':'s'}} of revision, the bill clears the chamber on its final vote. The path there required dropping amendments that would have violated the user's core tenets.`;
    }} else if (data.final_status === 'stuck') {{
        deck.textContent = `After ${{data.total_rounds}} round${{data.total_rounds===1?'':'s'}}, the chamber stalled — every remaining vote-changing amendment would have violated one of the user's core tenets. To pass, the user would have to negotiate on what they thought was non-negotiable.`;
    }} else if (data.final_status === 'converged') {{
        const bestNum = data.best_round_number || data.total_rounds;
        deck.textContent = `Hill-climbed across ${{data.total_rounds}} rounds. The best version found — Round ${{bestNum}}, ${{data.final_yes}}-yes — could not be improved further without crossing one of the user's core tenets.`;
    }} else {{
        deck.textContent = `Hit the round cap with the chamber still divided. More iteration might have closed the gap.`;
    }}

    document.getElementById('prompt-text').textContent = data.user_prompt;

    const heroStats = document.getElementById('hero-stats');
    const bestRoundLabel = data.best_round_number
        ? `Yes-votes, best round (R${{data.best_round_number}})`
        : 'Yes-votes, final round';
    heroStats.innerHTML = `
        <div class="hero-stat ${{verdictClass}}"><div class="num">${{data.final_yes}}</div><div class="label">${{bestRoundLabel}}</div></div>
        <div class="hero-stat"><div class="num">${{data.final_no}}</div><div class="label">Opposed</div></div>
        <div class="hero-stat"><div class="num">${{data.total_rounds}}</div><div class="label">Rounds of revision</div></div>
    `;

    // ----- TENETS -----
    const tenetsList = document.getElementById('tenets-list');
    tenetsList.innerHTML = data.tenets.map(t => `<li>${{escapeHtml(t)}}</li>`).join('');

    // ----- CONSTITUENTS -----
    if (data.has_constituents) {{
        const block = document.getElementById('constituents-block');
        block.hidden = false;
        document.getElementById('const-total').textContent = data.public_total;

        const pubYesPct = (data.public_yes_pct * 100).toFixed(0);
        const pubNoPct = (data.public_no_pct * 100).toFixed(0);
        const pubUnsurePct = (100 - pubYesPct - pubNoPct).toFixed(0);
        const senateYesPct = ((data.final_yes / (data.final_yes + data.final_no || 1)) * 100).toFixed(0);
        const senateNoPct = (100 - senateYesPct).toFixed(0);
        document.getElementById('public-vs-senate').innerHTML = `
            <div class="public-stat">
                <div class="label">The Public</div>
                <div class="bar">
                    <div class="seg-yes" style="width:${{pubYesPct}}%"></div>
                    <div class="seg-unsure" style="width:${{pubUnsurePct}}%"></div>
                    <div class="seg-no" style="width:${{pubNoPct}}%"></div>
                </div>
                <div class="pct-text"><span><strong>${{pubYesPct}}%</strong> support</span><span><strong>${{pubNoPct}}%</strong> oppose</span><span><strong>${{pubUnsurePct}}%</strong> unsure</span></div>
            </div>
            <div class="public-stat">
                <div class="label">The Senate (final round)</div>
                <div class="bar">
                    <div class="seg-yes" style="width:${{senateYesPct}}%"></div>
                    <div class="seg-no" style="width:${{senateNoPct}}%"></div>
                </div>
                <div class="pct-text"><span><strong>${{senateYesPct}}%</strong> yes</span><span><strong>${{senateNoPct}}%</strong> no</span></div>
            </div>
        `;

        const grid = document.getElementById('state-grid');
        grid.innerHTML = data.state_moods.map(m => {{
            const tot = m.yes + m.no + m.unsure || 1;
            const yp = (m.yes/tot*100).toFixed(1);
            const np = (m.no/tot*100).toFixed(1);
            const up = (m.unsure/tot*100).toFixed(1);
            return `<div class="state-cell ${{m.kind}}">
                <div class="state-code">${{m.state}}</div>
                <div class="state-bar">
                    <div class="seg-yes" style="width:${{yp}}%"></div>
                    <div class="seg-unsure" style="width:${{up}}%"></div>
                    <div class="seg-no" style="width:${{np}}%"></div>
                </div>
                <div class="state-label">${{escapeHtml(m.label)}}</div>
            </div>`;
        }}).join('');

        // A few voices: pick one from a strong-support, split, and strong-oppose state if available
        const voices = [];
        const moods = data.state_moods.slice();
        for (const m of moods) {{
            if (m.voices && m.voices.length) {{
                voices.push({{...m.voices[0], state: m.state, mood_label: m.label}});
                if (voices.length >= 3) break;
            }}
        }}
        document.getElementById('voices-list').innerHTML = voices.map(v => `
            <div class="voice-card">
                <div class="who"><strong>${{escapeHtml(v.intro || v.state)}}</strong> · <em>${{v.vote}}</em></div>
                <div class="quote">"${{escapeHtml(v.reasoning)}}"</div>
                ${{v.key_concern ? `<span class="key-concern">${{escapeHtml(v.key_concern)}}</span>` : ''}}
            </div>
        `).join('');
    }}

    // ----- ROUNDS -----
    const roundsContainer = document.getElementById('rounds');
    data.rounds.forEach((rd, i) => {{
        const isLast = i === data.rounds.length - 1;
        const roundEl = document.createElement('section');
        roundEl.className = 'round';

        let headline;
        if (rd.yes_total >= 67) headline = 'Clears cloture and passage';
        else if (rd.yes_total >= 51) headline = 'Carries a simple majority — but not 60';
        else headline = 'Falls short of a majority';

        const seatsSvg = buildMiniChamberSvg(rd.seat_votes, rd.seat_alignment || {{}}, i);
        const outOfStepNote = (rd.out_of_step_count && data.has_constituents)
            ? `<span class="out-of-step-note"><span class="dash"></span>${{rd.out_of_step_count}} voting against their state</span>`
            : '';

        const appliedHtml = rd.applied.map(a => `
            <div class="amendment-card applied">
                <div class="amendment-desc">${{escapeHtml(a.description)}}</div>
                <div class="amendment-meta"><span class="tag applied">Applied</span> · ${{escapeHtml(a.how_applied)}}</div>
            </div>
        `).join('');

        const rejectedHtml = rd.rejected.map(a => `
            <div class="amendment-card rejected">
                <div class="amendment-desc">${{escapeHtml(a.description)}}</div>
                <div class="amendment-meta"><span class="tag rejected">Rejected</span> · violates tenet: "${{escapeHtml(a.violates_tenet)}}" — ${{escapeHtml(a.rationale)}}</div>
            </div>
        `).join('');

        const flippersHtml = rd.flippers.map(f => `
            <div class="flipper-card">
                <div class="flipper-head">
                    ${{f.photo_url ? `<img class="flipper-photo" src="${{escapeAttr(f.photo_url)}}" alt="" loading="lazy" onerror="this.style.visibility='hidden'">` : ''}}
                    <div class="flipper-id">
                        <div class="flipper-name">${{escapeHtml(f.full_name)}}</div>
                        <div class="flipper-meta">${{f.party}}–${{f.state}}</div>
                        <span class="flipper-label">${{escapeHtml(f.label)}}</span>
                    </div>
                </div>
                <div class="flipper-quote">${{firstTwoSentences(f.reasoning)}}</div>
            </div>
        `).join('');

        const dissentersHtml = (rd.dissenters || []).map(d => {{
            let labelCls = '';
            if (d.label === 'holdout') labelCls = 'holdout';
            else if (d._shift !== undefined && d._shift <= -2) labelCls = 'moved-against';
            return `
            <div class="dissent-card flipper-card">
                <div class="flipper-head">
                    ${{d.photo_url ? `<img class="flipper-photo" src="${{escapeAttr(d.photo_url)}}" alt="" loading="lazy" onerror="this.style.visibility='hidden'">` : ''}}
                    <div class="flipper-id">
                        <div class="flipper-name">${{escapeHtml(d.full_name)}}</div>
                        <div class="flipper-meta">${{d.party}}–${{d.state}}</div>
                        <span class="flipper-label dissent-label ${{labelCls}}">${{escapeHtml(d.label)}}</span>
                    </div>
                </div>
                <div class="flipper-quote">${{firstTwoSentences(d.reasoning)}}</div>
            </div>
        `;}}).join('');

        // What-changed block (skip for the last round since no revision happens after)
        let changesHtml = '';
        if (!isLast && (rd.applied.length || rd.rejected.length || rd.revision_summary)) {{
            changesHtml = `
                <div class="changes">
                    <div class="changes-subhead">The revision that followed</div>
                    ${{rd.revision_summary ? `<div class="revision-summary">${{escapeHtml(rd.revision_summary)}}</div>` : ''}}
                    ${{rd.applied.length ? `<div class="amendment-block"><div class="changes-subhead">Applied (${{rd.applied.length}})</div>${{appliedHtml}}</div>` : ''}}
                    ${{rd.rejected.length ? `<div class="amendment-block"><div class="changes-subhead">Rejected — violates a tenet (${{rd.rejected.length}})</div>${{rejectedHtml}}</div>` : ''}}
                </div>
            `;
        }}

        const flippersLabel = i === 0 ? 'Opening yes-votes' : 'Senators who moved toward yes';
        const flippersBlock = rd.flippers.length
            ? `<div class="flippers"><div class="changes-subhead">${{flippersLabel}}</div><div class="flipper-grid">${{flippersHtml}}</div></div>`
            : '';

        const dissentersLabel = i === 0 ? 'Opening no-votes' : 'Senators pushing back';
        const dissentersBlock = (rd.dissenters && rd.dissenters.length)
            ? `<div class="dissenters"><div class="changes-subhead">${{dissentersLabel}}</div><div class="flipper-grid">${{dissentersHtml}}</div></div>`
            : '';

        // Outcome banner (hill-climb): accepted = new best, regressed = reverted
        let outcomeBannerHtml = '';
        if (rd.outcome === 'regressed') {{
            const ttb = (rd.target_to_beat !== null && rd.target_to_beat !== undefined) ? rd.target_to_beat : '?';
            outcomeBannerHtml = `<div class="round-outcome-banner regressed"><span class="icon">↓</span>Regression · ${{rd.yes_total}} ≤ best ${{ttb}} — reverted</div>`;
        }} else if (rd.outcome === 'passed') {{
            outcomeBannerHtml = `<div class="round-outcome-banner passed"><span class="icon">✓</span>Passes · ${{rd.yes_total}} yes</div>`;
        }} else if (rd.outcome === 'accepted' && i > 0) {{
            const ttb = (rd.target_to_beat !== null && rd.target_to_beat !== undefined) ? rd.target_to_beat : null;
            if (ttb !== null && rd.yes_total > ttb) {{
                outcomeBannerHtml = `<div class="round-outcome-banner accepted"><span class="icon">↑</span>New best · ${{rd.yes_total}} > prior ${{ttb}}</div>`;
            }}
        }}

        roundEl.innerHTML = `
            <div class="round-header">
                <div class="round-number">Round ${{rd.number}} · Bill v${{rd.version}}</div>
                <h2 class="round-headline">${{headline}}.</h2>
                <div class="round-tally">
                    <span class="num-yes">${{rd.yes_total}}</span>
                    <span class="num-sep">–</span>
                    <span class="num-no">${{rd.no_total}}</span>
                    ${{rd.undecided ? `<span class="num-und">${{rd.undecided}} undecided</span>` : ''}}
                    ${{outOfStepNote}}
                </div>
                ${{outcomeBannerHtml}}
            </div>
            ${{seatsSvg}}
            ${{flippersBlock}}
            ${{dissentersBlock}}
            ${{changesHtml}}
        `;
        roundsContainer.appendChild(roundEl);
    }});

    // ----- SENATOR DETAIL PANEL (click on any seat) -----
    const senatorByName = {{}};
    data.seat_layout.forEach(s => {{ senatorByName[s.full_name] = s; }});
    const photoByName = {{}};
    data.seat_layout.forEach(s => {{
        if (s.photo_url) photoByName[s.full_name] = s.photo_url;
    }});
    // Fallback: also pick up photos surfaced in flipper/dissent cards
    data.rounds.forEach(rd => {{
        (rd.flippers || []).concat(rd.dissenters || []).forEach(c => {{
            if (c && c.full_name && c.photo_url) photoByName[c.full_name] = c.photo_url;
        }});
    }});

    const panel = document.getElementById('senator-panel');
    const panelClose = () => {{ panel.hidden = true; document.body.style.overflow = ''; }};
    document.getElementById('senator-panel-close').addEventListener('click', panelClose);
    document.getElementById('senator-panel-backdrop').addEventListener('click', panelClose);
    document.addEventListener('keydown', e => {{ if (e.key === 'Escape' && !panel.hidden) panelClose(); }});

    function openPanel(fullName, roundIdx) {{
        const s = senatorByName[fullName];
        if (!s) return;
        const rd = data.rounds[roundIdx];
        const reaction = (rd && rd.seat_reactions && rd.seat_reactions[fullName]) || null;
        document.getElementById('sen-name').textContent = fullName;
        document.getElementById('sen-meta').textContent = `${{s.party === 'D' ? 'Democrat' : s.party === 'R' ? 'Republican' : 'Independent'}} · ${{s.state}}`;
        document.getElementById('sen-round-label').textContent = `Round ${{roundIdx + 1}} · Bill v${{rd.version}}`;
        const photoEl = document.getElementById('sen-photo');
        const photoUrl = photoByName[fullName] || '';
        if (photoUrl) {{ photoEl.src = photoUrl; photoEl.style.visibility = 'visible'; }}
        else {{ photoEl.removeAttribute('src'); photoEl.style.visibility = 'hidden'; }}
        const vote = reaction ? reaction.vote : 'absent';
        const pill = document.getElementById('sen-vote-pill');
        pill.className = `sen-vote-pill vote-${{vote}}`;
        pill.textContent = `Vote: ${{vote.replace('_', ' ')}}${{reaction && reaction.confidence ? ` · confidence ${{Math.round(reaction.confidence*100)}}%` : ''}}`;
        document.getElementById('sen-reasoning').textContent = reaction && reaction.reasoning
            ? `"${{reaction.reasoning}}"`
            : '(No recorded reasoning for this round.)';

        // Trajectory across all rounds
        const traj = document.getElementById('sen-trajectory');
        traj.innerHTML = data.rounds.map((r, j) => {{
            const reac = r.seat_reactions && r.seat_reactions[fullName];
            const v = reac ? reac.vote : 'absent';
            const swatchColor = voteColor(v);
            const cur = j === roundIdx ? ' current' : '';
            return `<button class="sen-traj-pill${{cur}}" data-round="${{j}}"><span class="swatch" style="background:${{swatchColor}}"></span>R${{j+1}} · ${{v.replace('_',' ')}}</button>`;
        }}).join('');
        traj.querySelectorAll('.sen-traj-pill').forEach(btn => {{
            btn.addEventListener('click', () => openPanel(fullName, parseInt(btn.dataset.round, 10)));
        }});

        panel.hidden = false;
        document.body.style.overflow = 'hidden';
    }}

    function voteColor(v) {{
        const map = {{
            'yes': '#2d7d3a', 'lean_yes': '#6ba76f', 'undecided': '#c0a248',
            'lean_no': '#c87560', 'no': '#a32a18', 'absent': '#b5b0a0'
        }};
        return map[v] || '#b5b0a0';
    }}

    // Delegate clicks on all seat groups (event delegation so we don't have to
    // re-bind after each round render).
    document.getElementById('rounds').addEventListener('click', e => {{
        const seat = e.target.closest('g.seat.clickable');
        if (!seat) return;
        const name = seat.getAttribute('data-name');
        const roundIdx = parseInt(seat.getAttribute('data-round'), 10);
        if (!name || isNaN(roundIdx)) return;
        openPanel(name, roundIdx);
    }});

    // ----- FINAL BILL -----
    document.getElementById('final-bill-name').textContent = data.final_bill_name || '';
    document.getElementById('final-bill-meta').textContent =
        `Version ${{data.rounds[data.rounds.length-1].version}} after ${{data.total_rounds}} round${{data.total_rounds===1?'':'s'}} · final tally ${{data.final_yes}}–${{data.final_no}}`;
    document.getElementById('final-bill-text').textContent = data.final_bill_text || '';
    document.getElementById('final-diff').innerHTML = data.final_diff_html || '';
    document.getElementById('generated-at').textContent = data.generated_at || '';

    // ----- helpers -----
    function buildMiniChamberSvg(seatVotes, seatAlignment, roundIdx) {{
        const dais = `<rect x="380" y="470" width="240" height="34" rx="3" class="dais"></rect>
                      <text x="500" y="492" class="dais-label">Presiding Officer</text>`;
        const seats = data.seat_layout.map(s => {{
            const vote = seatVotes[s.full_name] || 'absent';
            const align = (seatAlignment && seatAlignment[s.full_name]) || 'unknown';
            const partyCls = s.party === 'D' ? 'seat-d' : s.party === 'R' ? 'seat-r' : 'seat-i';
            const overlay = align === 'out_of_step'
                ? `<circle cx="${{s.x}}" cy="${{s.y}}" r="12.5" class="alignment-overlay"/>`
                : '';
            return `<g class="seat clickable ${{partyCls}}" data-name="${{escapeAttr(s.full_name)}}" data-round="${{roundIdx}}">
                ${{overlay}}
                <circle cx="${{s.x}}" cy="${{s.y}}" r="9.5" class="seat-ring"/>
                <circle cx="${{s.x}}" cy="${{s.y}}" r="7" class="vote-${{vote}}"/>
                <circle cx="${{s.x}}" cy="${{s.y}}" r="13" class="seat-hit" fill="transparent"/>
                <title>${{escapeAttr(s.short_name)}} (${{s.party}}-${{s.state}}) · ${{vote.replace('_',' ')}}${{align==='out_of_step' ? ' · OUT OF STEP with state' : ''}} · click for reasoning</title>
            </g>`;
        }}).join('');
        return `<svg class="minichamber" viewBox="0 0 1000 530" preserveAspectRatio="xMidYMid meet">${{dais}}<g>${{seats}}</g></svg>`;
    }}

    function firstTwoSentences(text) {{
        if (!text) return '';
        const parts = text.split(/(?<=[.!?])\\s+/);
        return escapeHtml(parts.slice(0, 2).join(' '));
    }}

    function escapeHtml(s) {{
        if (s === null || s === undefined) return '';
        return String(s).replace(/[&<>"']/g, c => ({{
            '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
        }})[c]);
    }}
    function escapeAttr(s) {{ return escapeHtml(s); }}
}})();
</script>
</body>
</html>"""
