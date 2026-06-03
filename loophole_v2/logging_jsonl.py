"""Structured, append-only logging — the backbone of the "log everything" requirement.

One JSONL file per record type under ``<run_dir>/logs/`` (rollouts, grpo_steps,
classifier_train, verifier, eval, rounds). Every record gets a timestamp. This is the raw
material the HTML report and the blog are built from, so we log generously: full model
generations and think-traces, every tool call, every classifier probability, every loss.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class JsonlLogger:
    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.logs_dir = self.run_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._files: dict[str, Any] = {}

    def log(self, record_type: str, record: dict) -> None:
        """Append one record to ``logs/<record_type>.jsonl`` and flush immediately."""
        f = self._files.get(record_type)
        if f is None:
            f = open(self.logs_dir / f"{record_type}.jsonl", "a")
            self._files[record_type] = f
        stamped = {"ts": time.time(), **record}
        f.write(json.dumps(stamped, default=str) + "\n")
        f.flush()

    def close(self) -> None:
        for f in self._files.values():
            f.close()
        self._files.clear()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_jsonl(path: str) -> list[dict]:
    """Read a JSONL file back into a list of dicts (used by the report)."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
