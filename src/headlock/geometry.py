"""Premiere Pro Motion math for pinning a tracked point to a fixed spot in the sequence frame.

Conventions: pixel coordinates, origin top-left, Y down (same as Premiere and OpenCV).
The clip's Anchor Point stays at its default (clip center). A source pixel ``q`` lands in the
sequence at ``P + s * (q - c)`` where ``P`` is Motion > Position, ``s`` is Scale/100 and ``c``
is the clip center. Pinning head ``h`` to lock point ``L`` gives ``P = L - s * (h - c)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EDGES = ("left", "top", "right", "bottom")


def lock_point(seq_size: tuple[int, int], lock: tuple[float, float]) -> np.ndarray:
    """Lock position in sequence pixels from fractions of the sequence frame (from top-left)."""
    return np.array([lock[0] * seq_size[0], lock[1] * seq_size[1]], dtype=float)


def positions(head_px: np.ndarray, src_size: tuple[int, int], seq_size: tuple[int, int],
              scale: float, lock: tuple[float, float]) -> np.ndarray:
    """Per-frame Motion > Position (sequence px) that keeps ``head_px`` at the lock point."""
    c = np.array(src_size, dtype=float) / 2
    return lock_point(seq_size, lock) - scale * (np.asarray(head_px, dtype=float) - c)


def visible_rect(head_px: np.ndarray, seq_size: tuple[int, int], scale: float,
                 lock: tuple[float, float]) -> np.ndarray:
    """Source-pixel rectangle (left, top, right, bottom) shown in the sequence on each frame."""
    h = np.asarray(head_px, dtype=float)
    L = lock_point(seq_size, lock)
    far = np.array(seq_size, dtype=float) - L
    return np.column_stack([h[:, 0] - L[0] / scale, h[:, 1] - L[1] / scale,
                            h[:, 0] + far[0] / scale, h[:, 1] + far[1] / scale])


def min_scale(head_px: np.ndarray, src_size: tuple[int, int], seq_size: tuple[int, int],
              lock: tuple[float, float]) -> float:
    """Smallest Scale (as a fraction) at which the sequence frame never shows outside the source.

    From ``visible_rect``: left >= 0  <=>  s >= Lx / hx, right <= W  <=>  s >= (Ws - Lx) / (W - hx),
    and likewise vertically. Returns inf if the head ever sits on/outside a source edge that the
    lock point needs room beyond.
    """
    h = np.asarray(head_px, dtype=float)
    L = lock_point(seq_size, lock)
    near, far = L, np.array(seq_size, dtype=float) - L
    room_near = h
    room_far = np.array(src_size, dtype=float) - h
    with np.errstate(divide="ignore", invalid="ignore"):
        need = np.concatenate([
            np.where(room_near > 0, near / room_near, np.where(near > 0, np.inf, 0.0)),
            np.where(room_far > 0, far / room_far, np.where(far > 0, np.inf, 0.0)),
        ], axis=1)
    return float(np.max(need))


@dataclass
class Feasibility:
    scale: float
    min_scale: float
    overflow: dict[str, float]      # worst overflow beyond each source edge, source px (0 = fits)
    worst_frame: dict[str, int]     # frame index of that worst overflow
    window: tuple[float, float]     # source-pixel size of the visible window at this scale

    @property
    def fits(self) -> bool:
        return all(v <= 0.5 for v in self.overflow.values())

    def report(self) -> str:
        lines = [
            f"Scale {self.scale * 100:.1f}% -> source window {self.window[0]:.0f}x{self.window[1]:.0f} px",
            f"Minimum scale that always fits: "
            + (f"{self.min_scale * 100:.1f}%" if self.min_scale < 50 else "none (head reaches the source edge)"),
        ]
        if self.fits:
            lines.append("Crop fits inside the source on every frame.")
        else:
            for e in EDGES:
                if self.overflow[e] > 0.5:
                    lines.append(f"  exceeds {e:<6} by {self.overflow[e]:.0f} px (worst at frame {self.worst_frame[e]})")
        if self.scale > 1.0:
            lines.append("Warning: scale > 100% upsamples the source (softer image).")
        return "\n".join(lines)


def check(head_px: np.ndarray, src_size: tuple[int, int], seq_size: tuple[int, int],
          scale: float, lock: tuple[float, float]) -> Feasibility:
    rect = visible_rect(head_px, seq_size, scale, lock)
    W, H = src_size
    over = np.column_stack([-rect[:, 0], -rect[:, 1], rect[:, 2] - W, rect[:, 3] - H])
    worst = over.argmax(axis=0)
    return Feasibility(
        scale=scale,
        min_scale=min_scale(head_px, src_size, seq_size, lock),
        overflow={e: max(0.0, float(over[worst[i], i])) for i, e in enumerate(EDGES)},
        worst_frame={e: int(worst[i]) for i, e in enumerate(EDGES)},
        window=(seq_size[0] / scale, seq_size[1] / scale),
    )


def auto_scale(head_px: np.ndarray, src_size: tuple[int, int], seq_size: tuple[int, int],
               lock: tuple[float, float], margin: float = 1.02, max_scale: float = 1.0) -> float:
    """Smallest scale that always fits (plus margin), capped at ``max_scale``.

    If the cap wins, the crop will show past the source edge on some frames; ``check`` reports where.
    """
    s = min_scale(head_px, src_size, seq_size, lock) * margin
    return float(min(s, max_scale))
