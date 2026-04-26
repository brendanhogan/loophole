"""Generate an HTML visualization of a reverse morals session."""

from __future__ import annotations

import difflib
import html
from pathlib import Path

from loophole.reverse.models import CaseResolution, CaseType, ReverseSession


def _escape(text: str) -> str:
    return html.escape(text)


def _compute_diff_html(before: str, after: str) -> str:
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    diff = difflib.unified_diff(before_lines, after_lines, n=3)
    lines_html = []
    for line in diff:
        if line.startswith("---") or line.startswith("+++"):
            continue
        stripped = html.escape(line.rstrip("\n"))
        if line.startswith("@@"):
            lines_html.append(f'<div class="diff-hunk">{stripped}</div>')
        elif line.startswith("+"):
            lines_html.append(f'<div class="diff-add">{stripped}</div>')
        elif line.startswith("-"):
            lines_html.append(f'<div class="diff-del">{stripped}</div>')
        else:
            lines_html.append(f'<div class="diff-ctx">{stripped}</div>')
    if not lines_html:
        return '<div class="diff-ctx">(no changes)</div>'
    return "\n".join(lines_html)


def generate_html(
    state: ReverseSession, output_path: str | None = None
) -> str:
    contradictions = [f for f in state.findings if f.case_type == CaseType.CONTRADICTION]
    gaps = [f for f in state.findings if f.case_type == CaseType.GAP]

    # Build tensions cards
    tensions_html = ""
    if state.tensions:
        for i, t in enumerate(state.tensions, 1):
            badge = "CONTRADICTION" if t.case_type == CaseType.CONTRADICTION else "GAP"
            badge_cls = "badge-contradiction" if t.case_type == CaseType.CONTRADICTION else "badge-gap"
            tensions_html += f"""
            <div class="tension-card">
                <div class="tension-header">
                    <span class="badge {badge_cls}">{badge}</span>
                    <span class="tension-title">Tension #{i}</span>
                    <span class="principles-tag">{_escape(', '.join(t.principles_involved))}</span>
                </div>
                <div class="tension-scenario">{_escape(t.scenario)}</div>
                <div class="tension-note">
                    <strong>Why it's unresolvable:</strong> {_escape(t.tension_note or '')}
                </div>
            </div>"""
    else:
        tensions_html = '<p class="empty">No tensions identified yet.</p>'

    # Build timeline of refinements
    timeline_html = ""
    refined = [f for f in state.findings if f.resolution == CaseResolution.REFINED]
    for f in refined:
        badge = "CONTRADICTION" if f.case_type == CaseType.CONTRADICTION else "GAP"
        badge_cls = "badge-contradiction" if f.case_type == CaseType.CONTRADICTION else "badge-gap"

        # Find the principles diff
        diff_html = ""
        idx = None
        for j, finding in enumerate(state.findings):
            if finding.id == f.id and finding.resolution == CaseResolution.REFINED:
                # Count how many refined findings came before this one
                refined_before = sum(
                    1 for fi in state.findings[:j]
                    if fi.resolution == CaseResolution.REFINED
                )
                # principles_history[0] is initial, [1] is after first refinement, etc.
                before_idx = refined_before
                after_idx = refined_before + 1
                if after_idx < len(state.principles_history):
                    diff_html = _compute_diff_html(
                        state.principles_history[before_idx].text,
                        state.principles_history[after_idx].text,
                    )
                break

        timeline_html += f"""
        <div class="timeline-item">
            <div class="timeline-header">
                <span class="badge {badge_cls}">{badge}</span>
                Finding #{f.id} — Round {f.round}
            </div>
            <div class="timeline-scenario">{_escape(f.scenario)}</div>
            <div class="timeline-instruction">
                <strong>Your instruction:</strong> {_escape(f.user_instruction or '')}
            </div>
            {f'<div class="diff-block">{diff_html}</div>' if diff_html else ''}
        </div>"""

    # Initial principles
    initial_text = _escape(state.principles_history[0].text) if state.principles_history else "(none)"

    # Final principles
    final_text = _escape(state.current_principles.text)

    page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Loophole Reverse — {_escape(state.document_name)}</title>
<style>
:root {{
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #c9d1d9; --text-dim: #8b949e; --text-bright: #f0f6fc;
    --red: #f85149; --yellow: #d29922; --green: #3fb950;
    --blue: #58a6ff; --purple: #bc8cff; --magenta: #f778ba;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; line-height: 1.6; padding: 2rem; max-width: 900px; margin: 0 auto; }}
h1 {{ color: var(--text-bright); font-size: 1.8rem; margin-bottom: 0.3rem; }}
h2 {{ color: var(--text-bright); font-size: 1.3rem; margin: 2rem 0 1rem; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }}
.subtitle {{ color: var(--text-dim); margin-bottom: 1.5rem; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; }}
.badge-contradiction {{ background: rgba(248,81,73,0.2); color: var(--red); }}
.badge-gap {{ background: rgba(210,153,34,0.2); color: var(--yellow); }}
.badge-doc {{ background: rgba(88,166,255,0.2); color: var(--blue); }}

/* Stats grid */
.stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1.5rem 0; }}
.stat {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; text-align: center; }}
.stat-value {{ font-size: 1.8rem; font-weight: 700; color: var(--text-bright); }}
.stat-label {{ font-size: 0.8rem; color: var(--text-dim); text-transform: uppercase; }}
.stat-red .stat-value {{ color: var(--red); }}
.stat-yellow .stat-value {{ color: var(--yellow); }}
.stat-purple .stat-value {{ color: var(--purple); }}

/* Tensions */
.tension-card {{ background: var(--surface); border: 1px solid var(--purple); border-left: 4px solid var(--purple); border-radius: 8px; padding: 1.2rem; margin-bottom: 1rem; }}
.tension-header {{ display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.8rem; }}
.tension-title {{ font-weight: 600; color: var(--purple); }}
.principles-tag {{ font-size: 0.8rem; color: var(--text-dim); margin-left: auto; }}
.tension-scenario {{ margin-bottom: 0.8rem; white-space: pre-wrap; }}
.tension-note {{ color: var(--text-dim); font-style: italic; }}

/* Timeline */
.timeline-item {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1.2rem; margin-bottom: 1rem; }}
.timeline-header {{ font-weight: 600; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem; }}
.timeline-scenario {{ margin-bottom: 0.5rem; white-space: pre-wrap; }}
.timeline-instruction {{ color: var(--green); margin-bottom: 0.5rem; }}

/* Code/principles blocks */
.code-block {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1.2rem; white-space: pre-wrap; font-family: 'SF Mono', SFMono-Regular, Consolas, monospace; font-size: 0.85rem; overflow-x: auto; max-height: 400px; overflow-y: auto; }}

/* Legal text */
.legal-text {{ max-height: 300px; overflow-y: auto; }}

/* Diffs */
.diff-block {{ background: #0d1117; border: 1px solid var(--border); border-radius: 6px; padding: 0.8rem; margin-top: 0.5rem; font-family: 'SF Mono', SFMono-Regular, Consolas, monospace; font-size: 0.8rem; overflow-x: auto; }}
.diff-add {{ color: var(--green); background: rgba(63,185,80,0.1); }}
.diff-del {{ color: var(--red); background: rgba(248,81,73,0.1); }}
.diff-hunk {{ color: var(--blue); }}
.diff-ctx {{ color: var(--text-dim); }}

.empty {{ color: var(--text-dim); font-style: italic; }}
.footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border); color: var(--text-dim); font-size: 0.85rem; text-align: center; }}
</style>
</head>
<body>

<h1>Loophole — Reverse Morals</h1>
<p class="subtitle">Extract the moral DNA of a legal text &nbsp; <span class="badge badge-doc">{_escape(state.document_name)}</span></p>

<div class="stats">
    <div class="stat">
        <div class="stat-value">{state.current_round}</div>
        <div class="stat-label">Rounds</div>
    </div>
    <div class="stat">
        <div class="stat-value">{len(state.findings)}</div>
        <div class="stat-label">Findings</div>
    </div>
    <div class="stat stat-red">
        <div class="stat-value">{len(contradictions)}</div>
        <div class="stat-label">Contradictions</div>
    </div>
    <div class="stat stat-purple">
        <div class="stat-value">{len(state.tensions)}</div>
        <div class="stat-label">Tensions</div>
    </div>
</div>

<h2>Legal Text</h2>
<div class="code-block legal-text">{_escape(state.legal_text[:5000])}{'...' if len(state.legal_text) > 5000 else ''}</div>

<h2>Initial Principles (v1)</h2>
<div class="code-block">{initial_text}</div>

<h2>Genuine Tensions</h2>
{tensions_html}

<h2>Refinement Timeline</h2>
{timeline_html if timeline_html else '<p class="empty">No refinements yet.</p>'}

<h2>Final Principles (v{state.current_principles.version})</h2>
<div class="code-block">{final_text}</div>

<div class="footer">
    {len(state.findings)} findings over {state.current_round} rounds &middot;
    Principles v{state.current_principles.version} &middot;
    {len(state.tensions)} genuine tensions &middot;
    Generated by <strong>Loophole</strong>
</div>

</body>
</html>"""

    if output_path:
        out = Path(output_path)
    else:
        out = Path("sessions") / state.session_id / "report.html"
        out.parent.mkdir(parents=True, exist_ok=True)

    out.write_text(page_html, encoding="utf-8")
    return str(out)
