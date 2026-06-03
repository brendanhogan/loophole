"""A population of past classifier checkpoints.

Training the adversary against a *sample* of past C's (not just the latest) is self-play
hygiene: it stops rock-paper-scissors cycling and pushes the adversary toward attacks that
beat many classifiers, which makes the mined data broadly useful. Checkpoints live on disk;
a small in-memory cache avoids reloading the same 0.6B model repeatedly.
"""

from __future__ import annotations

from .classifier import Classifier


class Population:
    def __init__(self, max_size: int = 5, device: str = "cuda"):
        self.max_size = max_size
        self.device = device
        self.paths: list[str] = []
        self._cache: dict[str, Classifier] = {}

    def add(self, path: str) -> None:
        self.paths.append(path)
        # Keep only the most recent ``max_size`` members eligible for sampling.
        if len(self.paths) > self.max_size:
            dropped = self.paths.pop(0)
            self._cache.pop(dropped, None)

    def _load(self, path: str) -> Classifier:
        if path not in self._cache:
            self._cache[path] = Classifier.load(path, device=self.device)
        return self._cache[path]

    def latest(self) -> Classifier:
        return self._load(self.paths[-1])

    def sample(self, rng) -> Classifier:
        path = self.paths[int(rng.integers(len(self.paths)))]
        return self._load(path)
