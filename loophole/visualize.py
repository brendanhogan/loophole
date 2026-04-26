"""Generate an HTML visualization of a Loophole session."""

from __future__ import annotations

import difflib
import html
from pathlib import Path

from loophole.models import CaseStatus, CaseType, LegalCode, SessionState


def _compute_diff_html(before: str, after: str) -> str:
    """Produce a git-style unified diff with red/green line highlighting."""
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)

    diff = difflib.unified_diff(before_lines, after_lines, n=3)

    lines_html = []
    for line in diff:
        # Skip the --- / +++ header lines
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
        return '<div class="diff-ctx">(no textual changes detected)</div>'

    return "\n".join(lines_html)


def _build_version_map(state: SessionState) -> dict[int, int]:
    """Map each resolved case's position in state.cases → the code_history index
    of the version it produced.

    We use position in state.cases (not case.id) as the key because case IDs
    can collide across types (e.g., loophole #1 and overreach #1).

    In the main loop, each resolved case triggers legislator.revise(), which
    appends a new LegalCode to code_history.  code_history[0] is the initial
    draft (v1).  The i-th resolved case (in processing order) produced
    code_history[i+1] if it exists.
    """
    mapping: dict[int, int] = {}
    resolved_idx = 0
    for cases_idx, case in enumerate(state.cases):
        if case.status in (CaseStatus.AUTO_RESOLVED, CaseStatus.USER_RESOLVED):
            history_idx = resolved_idx + 1  # +1 because [0] is initial
            if history_idx < len(state.code_history):
                mapping[cases_idx] = history_idx
            resolved_idx += 1
    return mapping


def generate_html(state: SessionState, output_path: str | None = None) -> str:
    cases_html = []
    case_to_history = _build_version_map(state)

    for cases_idx, case in enumerate(state.cases):
        if case.status not in (CaseStatus.AUTO_RESOLVED, CaseStatus.USER_RESOLVED):
            continue

        # Get before/after code for diff
        history_idx = case_to_history.get(cases_idx)
        before_code = None
        after_code = None
        if history_idx and history_idx < len(state.code_history):
            before_code = state.code_history[history_idx - 1]
            after_code = state.code_history[history_idx]

        case_type_label = "LOOPHOLE" if case.case_type == CaseType.LOOPHOLE else "OVERREACH"
        case_type_desc = "Legal but Immoral" if case.case_type == CaseType.LOOPHOLE else "Illegal but Moral"
        color = "#ef4444" if case.case_type == CaseType.LOOPHOLE else "#eab308"
        attack_bg = "#1c1012" if case.case_type == CaseType.LOOPHOLE else "#1c1a0e"
        resolved_by = "Judge (auto)" if case.resolved_by == "judge" else "Human (escalated)"
        resolved_badge_color = "#10b981" if case.resolved_by == "judge" else "#8b5cf6"

        diff_section = ""
        if before_code and after_code:
            diff_html = _compute_diff_html(before_code.text, after_code.text)
            diff_section = f"""
            <div class="section">
                <div class="section-label">Code Diff (v{before_code.version} &rarr; v{after_code.version})</div>
                <div class="diff-box">
                    {diff_html}
                </div>
            </div>
            """

        cases_html.append(f"""
        <div class="case">
            <div class="case-header">
                <div class="case-badge" style="background: {color};">
                    Case #{case.id} &mdash; {case_type_label}
                </div>
                <div class="case-meta">
                    <span class="round-badge">Round {case.round}</span>
                    <span class="resolved-badge" style="background: {resolved_badge_color};">{resolved_by}</span>
                </div>
            </div>
            <div class="case-type-desc" style="color: {color};">{case_type_desc}</div>

            <div class="section">
                <div class="section-label">The Attack</div>
                <div class="scenario-box" style="border-left: 4px solid {color}; background: {attack_bg};">
                    <p class="scenario-text">{html.escape(case.scenario)}</p>
                    <p class="explanation-text">{html.escape(case.explanation)}</p>
                </div>
            </div>

            <div class="section">
                <div class="section-label">Resolution</div>
                <div class="resolution-box">
                    <p>{html.escape(case.resolution or "")}</p>
                </div>
            </div>

            {diff_section}
        </div>
        """)

    # Stats
    total = len(state.cases)
    loopholes = len([c for c in state.cases if c.case_type == CaseType.LOOPHOLE])
    overreaches = len([c for c in state.cases if c.case_type == CaseType.OVERREACH])

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Loophole &mdash; {html.escape(state.domain)}</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: #0a0a0a;
        color: #e5e5e5;
        line-height: 1.6;
    }}
    .container {{
        max-width: 840px;
        margin: 0 auto;
        padding: 40px 24px;
    }}

    /* ---- Header ---- */
    .header {{
        text-align: center;
        margin-bottom: 48px;
    }}
    .header h1 {{
        font-size: 48px;
        font-weight: 700;
        letter-spacing: -2px;
        margin-bottom: 8px;
        background: linear-gradient(135deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .header .subtitle {{
        font-size: 18px;
        color: #737373;
        margin-bottom: 24px;
    }}
    .header .domain-badge {{
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        background: #1a1a2e;
        border: 1px solid #333;
        font-size: 14px;
        color: #a5b4fc;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    /* ---- Stats ---- */
    .stats {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 48px;
    }}
    .stat {{
        background: #141414;
        border: 1px solid #262626;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }}
    .stat-value {{
        font-size: 32px;
        font-weight: 700;
        color: #fff;
    }}
    .stat-label {{
        font-size: 12px;
        color: #737373;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }}

    /* ---- Principles ---- */
    .principles {{
        background: #141414;
        border: 1px solid #262626;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 48px;
    }}
    .principles h2 {{
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #737373;
        margin-bottom: 12px;
    }}
    .principles p {{
        color: #d4d4d4;
        white-space: pre-wrap;
        font-size: 14px;
    }}

    /* ---- Initial code ---- */
    .initial-code {{
        background: #141414;
        border: 1px solid #262626;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 48px;
    }}
    .initial-code h2 {{
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #737373;
        margin-bottom: 12px;
    }}
    .initial-code-block {{
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 16px;
        font-family: 'SF Mono', 'Fira Code', 'Menlo', monospace;
        font-size: 12px;
        line-height: 1.6;
        color: #c9d1d9;
        white-space: pre-wrap;
        overflow-x: auto;
        max-height: 500px;
        overflow-y: auto;
    }}

    /* ---- Timeline ---- */
    .timeline {{
        position: relative;
        padding-left: 32px;
    }}
    .timeline::before {{
        content: '';
        position: absolute;
        left: 8px;
        top: 0;
        bottom: 0;
        width: 2px;
        background: #262626;
    }}

    /* ---- Case card ---- */
    .case {{
        position: relative;
        background: #141414;
        border: 1px solid #262626;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
    }}
    .case::before {{
        content: '';
        position: absolute;
        left: -28px;
        top: 28px;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: #3b82f6;
        border: 2px solid #0a0a0a;
    }}
    .case-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
        flex-wrap: wrap;
        gap: 8px;
    }}
    .case-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 13px;
        font-weight: 600;
        color: #fff;
    }}
    .case-meta {{
        display: flex;
        gap: 8px;
    }}
    .round-badge {{
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        background: #1e293b;
        color: #94a3b8;
    }}
    .resolved-badge {{
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        color: #fff;
    }}
    .case-type-desc {{
        font-size: 13px;
        font-weight: 500;
        margin-bottom: 16px;
    }}
    .section {{
        margin-bottom: 16px;
    }}
    .section-label {{
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #636363;
        margin-bottom: 8px;
        font-weight: 600;
    }}

    /* ---- Attack box (dark bg, high contrast text) ---- */
    .scenario-box {{
        padding: 16px;
        border-radius: 8px;
    }}
    .scenario-text {{
        font-size: 14px;
        margin-bottom: 12px;
        color: #f5f5f5;
    }}
    .explanation-text {{
        font-size: 13px;
        color: #b0b0b0;
        font-style: italic;
    }}

    /* ---- Resolution box ---- */
    .resolution-box {{
        padding: 16px;
        border-radius: 8px;
        background: #0a1a0a;
        border-left: 4px solid #10b981;
    }}
    .resolution-box p {{
        font-size: 14px;
        color: #d4d4d4;
    }}

    /* ---- Diff view (git-style) ---- */
    .diff-box {{
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 4px 0;
        font-family: 'SF Mono', 'Fira Code', 'Menlo', monospace;
        font-size: 12px;
        line-height: 1.6;
        overflow-x: auto;
        max-height: 400px;
        overflow-y: auto;
    }}
    .diff-hunk {{
        padding: 4px 16px;
        color: #79c0ff;
        background: #161b22;
        font-weight: 600;
        border-top: 1px solid #21262d;
        border-bottom: 1px solid #21262d;
        margin: 2px 0;
    }}
    .diff-add {{
        padding: 1px 16px;
        color: #aff5b4;
        background: #12261e;
    }}
    .diff-del {{
        padding: 1px 16px;
        color: #ffa0a0;
        background: #2d1215;
    }}
    .diff-ctx {{
        padding: 1px 16px;
        color: #8b949e;
    }}

    /* ---- Final code ---- */
    .final-code {{
        background: #141414;
        border: 1px solid #262626;
        border-radius: 12px;
        padding: 24px;
        margin-top: 48px;
    }}
    .final-code h2 {{
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 16px;
        color: #10b981;
    }}
    .final-code-block {{
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 16px;
        font-family: 'SF Mono', 'Fira Code', 'Menlo', monospace;
        font-size: 12px;
        line-height: 1.6;
        color: #c9d1d9;
        white-space: pre-wrap;
        overflow-x: auto;
    }}

    /* ---- Footer ---- */
    .footer {{
        text-align: center;
        margin-top: 48px;
        color: #525252;
        font-size: 13px;
    }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>Loophole</h1>
        <div class="subtitle">Adversarial Moral-Legal Code System</div>
        <div class="domain-badge">{html.escape(state.domain)}</div>
    </div>

    <div class="stats">
        <div class="stat">
            <div class="stat-value">{state.current_round}</div>
            <div class="stat-label">Rounds</div>
        </div>
        <div class="stat">
            <div class="stat-value">{total}</div>
            <div class="stat-label">Cases</div>
        </div>
        <div class="stat">
            <div class="stat-value" style="color: #ef4444;">{loopholes}</div>
            <div class="stat-label">Loopholes</div>
        </div>
        <div class="stat">
            <div class="stat-value" style="color: #eab308;">{overreaches}</div>
            <div class="stat-label">Overreach</div>
        </div>
    </div>

    <div class="principles">
        <h2>Moral Principles</h2>
        <p>{html.escape(state.moral_principles)}</p>
    </div>

    <div class="initial-code">
        <h2>Initial Legal Code (v1)</h2>
        <div class="initial-code-block">{html.escape(state.code_history[0].text if state.code_history else state.current_code.text)}</div>
    </div>

    <div class="timeline">
        {"".join(cases_html)}
    </div>

    <div class="final-code">
        <h2>Final Legal Code (v{state.current_code.version})</h2>
        <div class="final-code-block">{html.escape(state.current_code.text)}</div>
    </div>

    <div class="footer">
        Generated by <strong>Loophole</strong> &mdash; {state.current_round} rounds, {total} adversarial cases, code v{state.current_code.version}
    </div>
</div>
</body>
</html>"""

    if output_path is None:
        output_path = f"sessions/{state.session_id}/report.html"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(page, encoding="utf-8")
    return output_path
