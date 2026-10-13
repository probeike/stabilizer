"""Live framing preview: play a tracked clip as it will come out, while the framing options change.

Like Insta360 Studio's preview, each frame is redrawn through the virtual camera as it plays, so a
new field of view, head height or smoothing shows on the next frame. Studio does this on the GPU;
here it is the export's own maths on the CPU at a preview size (decode, then one ``cv2.remap``),
which is enough to judge the framing. The export re-renders at full quality. No sound.

``Framer`` turns a track and the options into a ``render(i, frame)``; ``Playback`` decodes from
any frame on a worker thread and hands each rendered frame to a callback.
"""

from __future__ import annotations

import threading
import time
from argparse import Namespace
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import cli, geometry, sphere
from .preview import _overlay
from .trackio import Track
from .video import iter_frames

PREVIEW_LONG_SIDE = 640
DECODE_HEIGHT_360 = 960     # equirect frames decoded at 1920 x 960 for the preview
DECODE_HEIGHT_FLAT = 1080


@dataclass(frozen=True)
class Options:
    """The framing options ``headlock export`` takes."""
    out_size: tuple[int, int]
    lock: tuple[float, float]
    fov: float = 100.0
    smooth: str = "savgol"
    cutoff_hz: float = 6.0
    scale: str = "auto"
    max_scale: float = 1.0


def preview_size(out_size: tuple[int, int], long_side: int = PREVIEW_LONG_SIDE) -> tuple[int, int]:
    """The output's shape at ``long_side`` px on its longer side (even sizes)."""
    k = long_side / max(out_size)
    return 2 * round(out_size[0] * k / 2), 2 * round(out_size[1] * k / 2)


class Framer:
    """Renders frame ``i`` of the tracked clip as the export would, at preview size.

    Smoothing, scale and aim are computed for every frame up front (fast), the same way the
    export does, so what plays is what will render.
    """

    def __init__(self, track: Track, opts: Options, long_side: int = PREVIEW_LONG_SIDE, marks: bool = False):
        self.track, self.opts, self.marks = track, opts, marks
        self.size = preview_size(opts.out_size, long_side)
        self.L = geometry.lock_point(self.size, opts.lock)
        self.n = len(track.head)
        self.note = ""
        args = Namespace(output_resolution=opts.out_size, lock_x=opts.lock[0], lock_y=opts.lock[1],
                         scale=opts.scale, max_scale=opts.max_scale, fov=opts.fov, smooth=opts.smooth,
                         cutoff_hz=opts.cutoff_hz)
        if track.projection == "equirect":
            self.f = sphere.focal(self.size, opts.fov)
            self.rays = sphere.view_rays(self.size, self.f)
            yaw, pitch = sphere.aim(cli._smoothed(track, args), sphere.pixels_to_rays(self.L, self.size, self.f))
            self.R = sphere.rotation(yaw, pitch).astype(np.float32)
            self.raw = sphere.uv_to_dirs(track.head)
        else:
            self.head, self.scale, feas = cli._prepare(track, args)
            if not feas.fits:
                self.note = "At this zoom the frame shows past the picture's edge in places (purple)."

    @property
    def decode_height(self) -> int:
        return DECODE_HEIGHT_360 if self.track.projection == "equirect" else DECODE_HEIGHT_FLAT

    def __call__(self, i: int, frame: np.ndarray) -> np.ndarray:
        i = min(i, self.n - 1)
        fh, fw = frame.shape[:2]
        raw = None
        if self.track.projection == "equirect":
            mx, my = sphere.equirect_maps(self.R[i], self.rays, (fw, fh))
            img = cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            if self.marks:
                raw = sphere.rays_to_pixels(self.R[i].T @ self.raw[i], self.size, self.f)
        else:
            # As preview.render: source px -> output px is scale * src_w / fw, then down to preview size.
            k = self.scale * self.track.source_size[0] / fw * self.size[0] / self.opts.out_size[0]
            h = self.head[i] * (fw, fh)
            M = np.array([[k, 0, self.L[0] - k * h[0]], [0, k, self.L[1] - k * h[1]]])
            img = cv2.warpAffine(frame, M, self.size, flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 0, 40))
            if self.marks:
                raw = self.L + k * (self.track.head[i] * (fw, fh) - h)
        if self.marks:
            _overlay(img, self.L, raw, i)
        return img


def to_ppm(img: np.ndarray) -> bytes:
    """BGR image -> binary PPM, which Tk's PhotoImage reads directly."""
    h, w = img.shape[:2]
    return b"P6 %d %d 255\n" % (w, h) + np.ascontiguousarray(img[..., ::-1]).tobytes()


class Playback:
    """Plays ``video`` from any frame on a worker thread.

    Each frame goes through the current ``render(i, frame)`` and then to ``on_frame(i, result)``
    (on the worker thread). Playing keeps to ``fps``, skipping the render of a few frames it is
    late for; if decoding itself can't keep up, it plays slower instead of freezing.
    ``set_render`` redraws the paused frame straight away, or the next one while playing.
    """

    def __init__(self, video: str | Path, fps: Fraction, count: int, render: Callable, on_frame: Callable,
                 decode_height: int | None):
        self.video, self.fps, self.count = str(video), float(fps), count
        self.decode_height = decode_height
        self._render, self._on_frame = render, on_frame
        self._cv = threading.Condition()
        self._seek: int | None = 0
        self._playing = False
        self._dirty = False
        self._stop = False
        self.position = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def playing(self) -> bool:
        return self._playing

    def play(self) -> None:
        with self._cv:
            if self.position >= self.count - 1:
                self._seek = 0
            self._playing = True
            self._cv.notify()

    def pause(self) -> None:
        with self._cv:
            self._playing = False
            self._cv.notify()

    def seek(self, i: int) -> None:
        with self._cv:
            self._seek = max(0, min(int(i), self.count - 1))
            self._cv.notify()

    def set_render(self, render: Callable) -> None:
        with self._cv:
            self._render, self._dirty = render, True
            self._cv.notify()

    def close(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        frames, current, index, clock, skipped = None, None, -1, None, 0
        try:
            while True:
                with self._cv:
                    while not (self._stop or self._seek is not None or self._playing or self._dirty):
                        clock = None   # paused: the playback clock restarts on play
                        self._cv.wait()
                    if self._stop:
                        return
                    seek, self._seek = self._seek, None
                    playing, render = self._playing, self._render
                    show, self._dirty = self._dirty, False
                if seek is not None:
                    if frames is not None:
                        frames.close()
                    frames = iter_frames(self.video, max_height=self.decode_height, start=seek, fast=True)
                    current, index, clock = None, seek - 1, None
                if playing or seek is not None:
                    frame = next(frames, None)
                    if frame is None:   # end of the clip
                        with self._cv:
                            self._playing = False
                        if current is None:
                            continue
                    else:
                        current, index, show = frame, index + 1, True
                if not show or current is None:
                    continue
                self.position = index
                if playing:
                    now = time.monotonic()
                    if clock is None:
                        clock = (now, index)
                    late = now - (clock[0] + (index - clock[1]) / self.fps)
                    if late > 1 / self.fps:
                        if skipped < 4:
                            skipped += 1
                            continue   # behind: decode on, draw the next frame that is on time
                        clock = (now, index)   # still behind: decoding is the limit, so just play slower
                    skipped = 0
                    if late < 0:
                        with self._cv:
                            self._cv.wait(timeout=-late)
                self._on_frame(index, render(index, current))
        finally:
            if frames is not None:
                frames.close()
