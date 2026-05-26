"""Persistence for senate sessions and the shared constitutions store."""

from __future__ import annotations

import json
from pathlib import Path

from loophole.senate.models import MoralConstitution, SenateSession


CONSTITUTIONS_DIR_DEFAULT = "sessions/_senate_constitutions"


class ConstitutionStore:
    """Constitutions are expensive to generate — store them once, reuse across bills."""

    def __init__(self, base_dir: str = CONSTITUTIONS_DIR_DEFAULT):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, senator_full_name: str) -> Path:
        slug = senator_full_name.lower().replace(",", "").replace(" ", "_").replace(".", "")
        return self.base_dir / f"{slug}.json"

    def has(self, senator_full_name: str) -> bool:
        return self._path_for(senator_full_name).exists()

    def save(self, constitution: MoralConstitution) -> None:
        self._path_for(constitution.senator_full_name).write_text(
            constitution.model_dump_json(indent=2)
        )

    def load(self, senator_full_name: str) -> MoralConstitution:
        return MoralConstitution.model_validate_json(
            self._path_for(senator_full_name).read_text()
        )

    def load_all(self) -> dict[str, MoralConstitution]:
        out: dict[str, MoralConstitution] = {}
        for p in self.base_dir.glob("*.json"):
            c = MoralConstitution.model_validate_json(p.read_text())
            out[c.senator_full_name] = c
        return out


class SenateSessionManager:
    def __init__(self, base_dir: str = "sessions"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_session(
        self,
        session_id: str,
        bill_name: str,
        bill_text: str,
    ) -> SenateSession:
        state = SenateSession(
            session_id=session_id,
            bill_name=bill_name,
            bill_text=bill_text,
        )
        self.save(state)
        return state

    def save(self, state: SenateSession) -> None:
        session_dir = self.base_dir / state.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "state.json").write_text(state.model_dump_json(indent=2))

    def load(self, session_id: str) -> SenateSession:
        return SenateSession.model_validate_json(
            (self.base_dir / session_id / "state.json").read_text()
        )

    def list_sessions(self) -> list[dict]:
        sessions = []
        for p in sorted(self.base_dir.iterdir()):
            state_path = p / "state.json"
            if not state_path.exists():
                continue
            try:
                data = json.loads(state_path.read_text())
            except json.JSONDecodeError:
                continue
            # senate-style sessions track reactions
            if "reactions" not in data:
                continue
            sessions.append({
                "id": data["session_id"],
                "bill": data.get("bill_name", ""),
                "reactions": len(data.get("reactions", [])),
            })
        return sessions
