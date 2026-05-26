"""Load the senate roster from the bundled JSON file."""

from __future__ import annotations

import json
from pathlib import Path

from loophole.senate.models import Senator

ROSTER_PATH = Path(__file__).parent / "data" / "senators.json"


def load_roster(path: Path | str | None = None) -> list[Senator]:
    p = Path(path) if path else ROSTER_PATH
    data = json.loads(p.read_text())
    return [Senator(**row) for row in data["senators"]]
