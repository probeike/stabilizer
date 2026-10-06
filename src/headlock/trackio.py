"""track.json: raw per-frame head positions, normalized to the tracked frame (0-1, origin top-left)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np


@dataclass
class Track:
    video: str                 # file that was tracked (may be a proxy)
    source: str                # full-res media the XML should reference
    source_size: tuple[int, int]
    fps: Fraction
    head: np.ndarray           # (N, 2) normalized, NaN where unreliable
    conf: np.ndarray           # (N,)
    fraction: float = 2 / 3
    has_audio: bool = False

    def head_px(self, size: tuple[int, int] | None = None) -> np.ndarray:
        return self.head * np.array(size or self.source_size, dtype=float)

    def save(self, path: str | Path) -> None:
        data = {
            "version": 1,
            "video": self.video,
            "source": self.source,
            "source_size": list(self.source_size),
            "fps": f"{self.fps.numerator}/{self.fps.denominator}",
            "fraction": self.fraction,
            "has_audio": self.has_audio,
            "head": [None if np.isnan(p).any() else [round(float(p[0]), 6), round(float(p[1]), 6)]
                     for p in self.head],
            "conf": [round(float(c), 4) for c in self.conf],
        }
        Path(path).write_text(json.dumps(data, indent=None, separators=(",", ":")))

    @classmethod
    def load(cls, path: str | Path) -> "Track":
        data = json.loads(Path(path).read_text())
        head = np.array([[np.nan, np.nan] if p is None else p for p in data["head"]], dtype=float)
        return cls(
            video=data["video"],
            source=data["source"],
            source_size=tuple(data["source_size"]),
            fps=Fraction(data["fps"]),
            head=head.reshape(-1, 2),
            conf=np.array(data["conf"], dtype=float),
            fraction=data.get("fraction", 2 / 3),
            has_audio=data.get("has_audio", False),
        )
