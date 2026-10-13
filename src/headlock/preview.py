"""Render stabilized previews, the 360 reframe, and synthetic test clips."""

from __future__ import annotations

import math
import sys
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from . import geometry, sphere
from .trackio import Track
from .video import VideoWriter, iter_frames, probe, wants_16bit


def _overlay(img: np.ndarray, L: np.ndarray, raw: np.ndarray | None, i: int) -> None:
    """Green crosshair at the lock point, red dot at the raw head, frame number top-left."""
    cx, cy = int(round(L[0])), int(round(L[1]))
    cv2.line(img, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 1, cv2.LINE_AA)
    if raw is not None and not np.isnan(raw).any():
        cv2.circle(img, (int(round(raw[0])), int(round(raw[1]))), 4, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(img, str(i), (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)


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
    frames = tqdm(iter_frames(video, max_height=max_h), total=len(head_norm), desc="Preview", unit="f",
                  disable=not sys.stderr.isatty())
    with VideoWriter(out, *seq_size, info.fps) as writer:
        for i, frame in enumerate(frames):
            if i >= len(head_norm):
                break
            fh, fw = frame.shape[:2]
            k = scale * src_size[0] / fw  # frame px -> sequence px
            h = head_norm[i] * (fw, fh)
            M = np.array([[k, 0, L[0] - k * h[0]], [0, k, L[1] - k * h[1]]])
            img = cv2.warpAffine(frame, M, seq_size, flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 0, 40))
            raw = None if raw_norm is None else L + k * (raw_norm[i] * (fw, fh) - h)
            _overlay(img, L, raw, i)
            writer.write(img)


def render_360(video: str | Path, dirs: np.ndarray, raw_dirs: np.ndarray | None,
               out_size: tuple[int, int], fov: float, lock: tuple[float, float], out: str | Path,
               overlays: bool = False, audio: bool = False) -> None:
    """Reframe an equirect video into a flat view that keeps ``dirs`` (smoothed head) on the lock point.

    The view turns (yaw, then pitch, no roll) to follow the head, so nothing is ever cropped past
    an edge. ``overlays`` draws the lock point and raw head as in ``render``; without it the result
    is the deliverable: full quality, the source's audio, ProRes 10-bit if ``out`` is a .mov.
    """
    info = probe(video)
    f = sphere.focal(out_size, fov)
    # Decode just past the view's centre sampling rate (px per radian * 2 pi); more is wasted work.
    max_h = math.ceil(2 * np.pi * f * 1.25 / 2)
    depth = 16 if wants_16bit(out) and not overlays else 8
    L = geometry.lock_point(out_size, lock)
    yaw, pitch = sphere.aim(dirs, sphere.pixels_to_rays(L, out_size, f))
    R = sphere.rotation(yaw, pitch)
    rays = sphere.view_rays(out_size, f)
    frames = tqdm(iter_frames(video, max_height=max_h, depth=depth), total=len(dirs),
                  desc="Preview" if overlays else "Rendering", unit="f", disable=not sys.stderr.isatty())
    with VideoWriter(out, *out_size, info.fps, audio_from=video if audio else None, depth=depth,
                     crf=20 if overlays else 16) as writer:
        for i, frame in enumerate(frames):
            if i >= len(dirs):
                break
            mx, my = sphere.equirect_maps(R[i], rays, (frame.shape[1], frame.shape[0]))
            img = cv2.remap(frame, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
            if overlays:
                raw = None if raw_dirs is None else sphere.rays_to_pixels(R[i].T @ raw_dirs[i], out_size, f)
                _overlay(img, L, raw, i)
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


def make_test_clip_360(out: str | Path, size: tuple[int, int] = (3840, 1920),
                       fps: Fraction = Fraction(30000, 1001), seconds: float = 10.0) -> Track:
    """Equirect grid with a dot wandering back and forth across the +-180 deg seam, plus its track.

    Grid lines every 15 deg; the equator is yellow, the forward meridian (centre) cyan. Exporting
    the track should give a reframe where the dot stays dead still on the lock point.
    """
    W, H = size
    n = int(round(seconds * fps))
    bg = np.full((H, W, 3), 70, np.uint8)
    thick = max(1, W // 1920)
    for k in range(24):
        cv2.line(bg, (k * W // 24, 0), (k * W // 24, H), (120, 120, 120), thick)
    for k in range(12):
        cv2.line(bg, (0, k * H // 12), (W, k * H // 12), (120, 120, 120), thick)
    cv2.line(bg, (0, H // 2), (W, H // 2), (0, 220, 255), 2 * thick)
    cv2.line(bg, (W // 2, 0), (W // 2, H), (255, 220, 0), 2 * thick)

    t = np.arange(n) / float(fps)
    lon = np.pi + 0.6 * np.sin(2 * np.pi * 0.23 * t)            # around the seam behind the camera
    lat = -0.15 + 0.25 * np.sin(2 * np.pi * 0.37 * t + 0.6)
    xs = np.round(((lon / (2 * np.pi) + 0.5) % 1.0) * W - 0.5)
    ys = np.round((0.5 - lat / np.pi) * H - 0.5)
    radius = max(6, W // 220)
    with VideoWriter(out, W, H, fps) as writer:
        for i in range(n):
            img = bg.copy()
            rx = int(round(radius / np.cos(lat[i])))   # a round dot on the sphere is wider here
            for x in (xs[i] - W, xs[i], xs[i] + W):    # drawn three times so it wraps the seam
                c = (int(x), int(ys[i]))
                cv2.ellipse(img, c, (rx, radius), 0, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.ellipse(img, c, (max(2, rx // 4), max(2, radius // 4)), 0, 0, 360, (0, 0, 255), -1,
                            cv2.LINE_AA)
            cv2.putText(img, f"frame {i}", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, W / 2000, (255, 255, 255),
                        max(2, W // 1280), cv2.LINE_AA)
            writer.write(img)
    # Drawn pixel index -> continuous coords (pixel i covers [i, i+1)) -> normalized.
    head = np.column_stack([(xs + 0.5) / W, (ys + 0.5) / H])
    return Track(video=str(out), source=str(out), source_size=size, fps=fps,
                 head=head, conf=np.ones(n), has_audio=False, projection="equirect")
