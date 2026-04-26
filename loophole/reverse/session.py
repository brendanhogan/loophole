from __future__ import annotations

import json
from pathlib import Path

from loophole.reverse.models import PrinciplesList, ReverseSession


class ReverseSessionManager:
    def __init__(self, base_dir: str = "sessions"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_session(
        self,
        session_id: str,
        document_name: str,
        legal_text: str,
        initial_principles: PrinciplesList,
    ) -> ReverseSession:
        state = ReverseSession(
            session_id=session_id,
            document_name=document_name,
            legal_text=legal_text,
            current_principles=initial_principles,
            principles_history=[initial_principles],
        )
        self.save(state)
        return state

    def save(self, state: ReverseSession) -> None:
        session_dir = self.base_dir / state.session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        (session_dir / "state.json").write_text(
            state.model_dump_json(indent=2), encoding="utf-8"
        )

        (session_dir / "current_principles.md").write_text(
            f"# Extracted Principles v{state.current_principles.version}\n\n"
            f"*Document: {state.document_name}*\n\n"
            f"{state.current_principles.text}\n",
            encoding="utf-8",
        )

        (session_dir / "finding_log.md").write_text(
            _render_finding_log(state), encoding="utf-8"
        )

        (session_dir / "tensions.md").write_text(
            _render_tensions(state), encoding="utf-8"
        )

    def load(self, session_id: str) -> ReverseSession:
        state_path = self.base_dir / session_id / "state.json"
        return ReverseSession.model_validate_json(state_path.read_text(encoding="utf-8"))

    def list_sessions(self) -> list[dict]:
        sessions = []
        for p in sorted(self.base_dir.iterdir()):
            state_path = p / "state.json"
            if not state_path.exists():
                continue
            data = json.loads(state_path.read_text(encoding="utf-8"))
            if "legal_text" not in data:
                continue
            sessions.append({
                "id": data["session_id"],
                "document": data["document_name"],
                "round": data["current_round"],
                "findings": len(data["findings"]),
                "tensions": len(data["tensions"]),
                "principles_version": data["current_principles"]["version"],
            })
        return sessions


def _render_finding_log(state: ReverseSession) -> str:
    lines = [f"# Finding Log — {state.document_name}\n"]
    for f in state.findings:
        status = f.resolution.value.upper()
        lines.append(f"## Finding #{f.id} [{status}] — {f.case_type.value}")
        lines.append(f"Round {f.round}\n")
        lines.append(f"**Scenario:** {f.scenario}\n")
        lines.append(f"**Explanation:** {f.explanation}\n")
        lines.append(f"**Principles:** {', '.join(f.principles_involved)}\n")
        if f.user_instruction:
            lines.append(f"**User instruction:** {f.user_instruction}\n")
        if f.tension_note:
            lines.append(f"**Tension note:** {f.tension_note}\n")
        lines.append("---\n")
    return "\n".join(lines)


def _render_tensions(state: ReverseSession) -> str:
    lines = [f"# Genuine Tensions — {state.document_name}\n"]
    if not state.tensions:
        lines.append("No tensions identified yet.\n")
        return "\n".join(lines)
    for i, t in enumerate(state.tensions, 1):
        lines.append(f"## Tension {i} ({t.case_type.value})")
        lines.append(f"**Scenario:** {t.scenario}\n")
        lines.append(f"**Why it's unresolvable:** {t.tension_note}\n")
        lines.append(f"**Principles involved:** {', '.join(t.principles_involved)}\n")
        lines.append("---\n")
    return "\n".join(lines)
