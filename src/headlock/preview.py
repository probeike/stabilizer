"""Render a stabilized preview and synthetic test clips without opening Premiere."""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from . import geometry
from .trackio import Track
from .video import VideoWriter, iter_frames, probe


def render(video: str | Path, head_norm: np.ndarray, raw_norm: np.ndarray | None,
           src_size: tuple[int, int], seq_size: tuple[int, int], scale: float,
           lock: tuple[float, float], out: str | Path) -> None:
    """Apply the same pin transform Premiere will, on decoded frames.

    Green crosshair = lock point (where the smoothed head is pinned). Red dot = raw, unsmoothed
    head estimate, so you can see what the smoothing removed.
    """
    info = probe(video)
    # Decode only as large as the preview needs.
    max_h = min(info.height, math.ceil(seq_size[1] / scale * info.height / src_size[1]))
    L = geometry.lock_point(seq_size, lock)
    with VideoWriter(out, *seq_size, info.fps) as writer:
        for i, frame in enumerate(iter_frames(video, max_height=max_h)):
            if i >= len(head_norm):
                break
            fh, fw = frame.shape[:2]
            k = scale * src_size[0] / fw  # frame px -> sequence px
            h = head_norm[i] * (fw, fh)
            M = np.array([[k, 0, L[0] - k * h[0]], [0, k, L[1] - k * h[1]]])
            img = cv2.warpAffine(frame, M, seq_size, flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 0, 40))
            cx, cy = int(round(L[0])), int(round(L[1]))
            cv2.line(img, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 1, cv2.LINE_AA)
            cv2.line(img, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 1, cv2.LINE_AA)
            if raw_norm is not None and not np.isnan(raw_norm[i]).any():
                r = L + k * (raw_norm[i] * (fw, fh) - h)
                cv2.circle(img, (int(round(r[0])), int(round(r[1]))), 4, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(img, str(i), (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(img)


def make_test_clip(out: str | Path, size: tuple[int, int] = (3840, 2160),
                   fps: Fraction = Fraction(30000, 1001), seconds: float = 10.0) -> Track:
    """Grid clip with a dot moving on a known Lissajous path, plus its exact track.

    Exporting this track should produce an XML where the dot stays dead still in Premiere.
    """
    W, H = size
    n = int(round(seconds * fps))
    bg = np.full((H, W, 3), 70, np.uint8)
    step = max(W // 24, 16)
    for x in range(0, W, step):
        cv2.line(bg, (x, 0), (x, H), (120, 120, 120), max(1, W // 1920))
    for y in range(0, H, step):
        cv2.line(bg, (0, y), (W, y), (120, 120, 120), max(1, W // 1920))
    cv2.rectangle(bg, (0, 0), (W - 1, H - 1), (0, 200, 255), max(4, W // 400))

    t = np.arange(n) / float(fps)
    # Stay well inside the frame so a 1080p crop fits at moderate scale.
    xs = W * (0.5 + 0.18 * np.sin(2 * np.pi * 0.23 * t))
    ys = H * (0.42 + 0.12 * np.sin(2 * np.pi * 0.37 * t + 0.6))
    radius = max(6, W // 110)
    with VideoWriter(out, W, H, fps) as writer:
        for i in range(n):
            img = bg.copy()
            c = (int(round(xs[i])), int(round(ys[i])))
            cv2.circle(img, c, radius, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, c, max(2, radius // 4), (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(img, f"frame {i}", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, W / 1000, (255, 255, 255),
                        max(2, W // 640), cv2.LINE_AA)
            writer.write(img)
    # Pixel centres -> normalized; drawn integer centres so the expected pin is exact to the pixel.
    head = np.column_stack([np.round(xs), np.round(ys)]) / (W, H)
    return Track(video=str(out), source=str(out), source_size=size, fps=fps,
                 head=head, conf=np.ones(n), has_audio=False)
