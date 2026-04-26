from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from loophole.models import Case, CaseStatus, LegalCode, SessionState


class SessionManager:
    def __init__(self, base_dir: str = "sessions"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_session(self, session_id: str, domain: str, principles: str, initial_code: LegalCode) -> SessionState:
        state = SessionState(
            session_id=session_id,
            domain=domain,
            moral_principles=principles,
            current_code=initial_code,
            code_history=[initial_code],
        )
        self.save(state)
        return state

    def save(self, state: SessionState) -> None:
        session_dir = self.base_dir / state.session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Machine-readable state
        (session_dir / "state.json").write_text(
            state.model_dump_json(indent=2), encoding="utf-8"
        )

        # Human-readable legal code
        (session_dir / "current_code.md").write_text(
            f"# Legal Code v{state.current_code.version}\n\n"
            f"*Domain: {state.domain}*\n\n"
            f"{state.current_code.text}\n",
            encoding="utf-8",
        )

        # Human-readable case log
        (session_dir / "case_log.md").write_text(
            _render_case_log(state),
            encoding="utf-8",
        )

    def load(self, session_id: str) -> SessionState:
        state_path = self.base_dir / session_id / "state.json"
        return SessionState.model_validate_json(state_path.read_text(encoding="utf-8"))

    def list_sessions(self) -> list[dict]:
        sessions = []
        for p in sorted(self.base_dir.iterdir()):
            state_path = p / "state.json"
            if state_path.exists():
                data = json.loads(state_path.read_text(encoding="utf-8"))
                sessions.append({
                    "id": data["session_id"],
                    "domain": data["domain"],
                    "round": data["current_round"],
                    "cases": len(data["cases"]),
                    "code_version": data["current_code"]["version"],
                })
        return sessions


def _render_case_log(state: SessionState) -> str:
    lines = [
        f"# Case Log — {state.domain}",
        f"*Session: {state.session_id}*\n",
    ]

    status_labels = {
        CaseStatus.AUTO_RESOLVED: "Auto-resolved by Judge",
        CaseStatus.USER_RESOLVED: "Resolved by User",
        CaseStatus.ESCALATED: "ESCALATED — awaiting user",
        CaseStatus.PENDING: "Pending",
    }

    for case in state.cases:
        label = status_labels.get(case.status, case.status.value)
        lines.append(f"## Case #{case.id} ({case.case_type.value}) — Round {case.round}")
        lines.append(f"**Status:** {label}\n")
        lines.append(f"**Scenario:** {case.scenario}\n")
        lines.append(f"**Problem:** {case.explanation}\n")
        if case.resolution:
            lines.append(f"**Resolution:** {case.resolution}\n")
        lines.append("---\n")

    return "\n".join(lines)
