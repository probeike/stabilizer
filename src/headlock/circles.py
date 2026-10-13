"""Chorus circles: three extra singing clips in soft-edged circles along the bottom of a vertical video.

Pure pieces, each testable without Resolve:
- ``slots``: where the circles go on the main video's frame (centres at 1/6, 1/2 and 5/6 of the width);
- ``framing_from_keypoints`` / ``find_framing``: a fixed framing centred on the singer's head;
- ``onset_envelope`` / ``sync``: line the chorus clips up with each other by their audio;
- ``build_setting``: the Fusion nodes that cut one clip into its circle.

Fusion graph per chorus clip (output is the full frame, transparent outside the circle, so it
composites over V1 with no Edit-page sizing)::

    MediaIn1 -> CircleFace (Transform) -> CircleZoom (Merge fg) -> CirclePlace (Merge fg) -> MediaOut1
                head moved to the centre   bg CircleSquare (DxD)    bg CircleCanvas (WxH)
                                           Size = zoom, masked      Center = the slot
                                           by CircleMask (ellipse)

CircleFace puts the head at the image centre and a Merge scales its foreground about its centre,
so changing CircleZoom's Size zooms around the face, which stays in the middle of the circle.
Merge maths as in ``resolve_fusion``: Size 1 maps one foreground pixel to one background pixel and
the foreground's centre lands at Center (normalized, origin bottom-left, Y up).
"""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .head import HEAD_TOP, NECK, head_centers
from .resolve_fusion import MEDIA_IN

NAMES = ("left", "middle", "right")
FACE_FILL = 0.55        # neck -> head-top spans this fraction of the circle's diameter
SAMPLES = 40            # frames sampled for the fixed framing
MIN_FACES = 5           # confident samples needed before trusting the face

SR = 16000              # audio sample rate for sync
HOP = 160               # 10 ms envelope hop
N_FFT = 1024            # 64 ms analysis window
PEAK_WIDTH = 7          # envelope steps (70 ms): onsets this close count as together
MIN_SYNC_CONFIDENCE = 1.5


@dataclass(frozen=True)
class Slot:
    name: str                   # "left", "middle" or "right"
    center: tuple[int, int]     # canvas px, origin top-left
    diameter: float             # visible circle (half-transparent edge), canvas px
    square: int                 # side of the transparent square the circle is cut in; even, so it lands on whole px
    soft: float                 # width of the soft edge, fraction of the diameter

    @property
    def index(self) -> int:
        return NAMES.index(self.name)


def slots(canvas: tuple[int, int], diameter: float = 0.30, y: float = 0.85,
          soft: float = 0.04) -> list[Slot]:
    """Left, middle and right circles. ``diameter`` is a fraction of the width (1/3 would touch),
    ``y`` the centres' height as a fraction from the top."""
    W, H = canvas
    d = diameter * W
    square = 2 * math.ceil(d * (1 + 2 * soft) / 2)   # room for the soft edge on both sides
    cy = round(y * H)
    return [Slot(name, (round(W * k / 6), cy), d, square, soft) for name, k in zip(NAMES, (1, 3, 5))]


# ---- framing ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class Framing:
    head: tuple[float, float]   # head centre, source px, origin top-left
    zoom: float                 # CircleZoom Size: source px -> canvas px
    found: bool                 # False = no confident face, framed on the picture's centre
    note: str = ""


def _min_zoom(head: tuple[float, float], src_size: tuple[int, int], diameter: float) -> float:
    """Smallest zoom at which the circle around ``head`` stays inside the picture."""
    room = min(head[0], src_size[0] - head[0], head[1], src_size[1] - head[1])
    return diameter / 2 / max(room, 1.0)


def framing_from_keypoints(kpts: np.ndarray, scores: np.ndarray, frame_size: tuple[int, int],
                           src_size: tuple[int, int], diameter: float, min_conf: float = 0.3) -> Framing:
    """Median head centre and size over sampled frames -> a fixed framing.

    kpts: (N, 26, 2) Halpe26 keypoints in decoded-frame px (``frame_size``); scores (N, 26).
    The neck -> head-top length sets the zoom so it fills ``FACE_FILL`` of the circle. Never
    zooms out so far that the circle would show past the picture's edge.
    """
    k = src_size[0] / frame_size[0]
    centre = (src_size[0] / 2, src_size[1] / 2)
    if len(kpts):
        heads, _, _ = head_centers(kpts, scores, min_conf)
        conf = np.minimum(scores[:, NECK], scores[:, HEAD_TOP])
        good = ~np.isnan(heads).any(axis=1) & (conf >= min_conf)
    else:
        good = np.zeros(0, bool)
    if good.sum() < MIN_FACES:
        zoom = _min_zoom(centre, src_size, diameter)
        return Framing(centre, zoom, False,
                       f"no clear face in {good.sum()}/{len(good)} sampled frames; framed on the picture's centre")
    head = tuple(float(v) for v in np.median(heads[good], axis=0) * k)
    head_len = float(np.median(np.linalg.norm(kpts[good, HEAD_TOP] - kpts[good, NECK], axis=1))) * k
    zoom = FACE_FILL * diameter / max(head_len, 1.0)
    floor = _min_zoom(head, src_size, diameter)
    note = ""
    if zoom < floor:
        zoom, note = floor, "zoomed in past the face-size default so the circle stays inside the picture"
    return Framing(head, zoom, True, note)


def find_framing(path: str | Path, diameter: float, estimator, samples: int = SAMPLES,
                 min_conf: float = 0.3, desc: str = "Finding the face") -> Framing:
    """Run the pose model on ``samples`` frames spread across the clip (a fixed framing doesn't need more)."""
    from tqdm import tqdm

    from .pose import BoxStabilizer
    from .video import iter_frames, probe

    info = probe(path)
    step = max(1, info.frame_count // samples)
    kpts, scores, size = [], [], None
    frames = tqdm(iter_frames(path, max_height=1080, step=step), total=math.ceil(info.frame_count / step),
                  desc=desc, unit="f", disable=not sys.stderr.isatty())
    for frame in frames:
        size = (frame.shape[1], frame.shape[0])
        estimator.boxes = BoxStabilizer()   # samples are far apart: pick the singer afresh each time
        kp, sc = estimator(frame)
        kpts.append(kp)
        scores.append(sc)
    if size is None:
        raise ValueError(f"no frames decoded from {path}")
    return framing_from_keypoints(np.array(kpts).reshape(-1, 26, 2), np.array(scores).reshape(-1, 26),
                                  size, (info.width, info.height), diameter, min_conf)


# ---- audio sync ------------------------------------------------------------------------------

def read_audio(path: str | Path, sr: int = SR, start: float = 0.0,
               duration: float | None = None) -> np.ndarray | None:
    """First audio stream as mono float32 at ``sr`` (optionally ``duration`` s from ``start`` s);
    None if the file has no audio."""
    span = (["-ss", f"{start:.6f}"] if start > 0 else []) + (["-t", f"{duration:.6f}"] if duration else [])
    proc = subprocess.run(["ffmpeg", "-v", "error", *span, "-i", str(path), "-map", "0:a:0", "-ac", "1",
                           "-ar", str(sr), "-f", "f32le", "-"], capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        return None
    return np.frombuffer(proc.stdout, "<f4")


def onset_envelope(x: np.ndarray, sr: int = SR, hop: int = HOP, n_fft: int = N_FFT) -> np.ndarray:
    """Where notes and words start: one value per ``hop`` samples, zero mean, unit variance.

    Spectral flux of the log-magnitude spectrum rises at onsets whatever the pitch, so takes singing
    different harmony parts still line up, and the log makes it indifferent to how loud each take
    was recorded. Only the onset peaks are kept, each widened to ``PEAK_WIDTH`` steps, because two
    takes never start a word at exactly the same moment: widening (unlike blurring) tolerates that
    slop without flattening the peaks, which keeps the right lag well ahead of chance matches.
    """
    from scipy.ndimage import maximum_filter1d, uniform_filter1d

    # No padding: silence before the first sample would read as an onset at every take's start,
    # and those would line up with each other instead of the singing.
    x = np.asarray(x, dtype=np.float32)
    x = np.pad(x, (0, max(0, n_fft - len(x))))
    n = 1 + (len(x) - n_fft) // hop
    window = np.hanning(n_fft).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop][:n]
    logmag = np.empty((n, n_fft // 2 + 1), np.float32)
    for a in range(0, n, 2048):   # in blocks, so long takes don't need gigabytes
        logmag[a:a + 2048] = np.log1p(1000 * np.abs(np.fft.rfft(frames[a:a + 2048] * window, axis=1)))
    flux = np.concatenate([[0.0], np.maximum(np.diff(logmag, axis=0), 0).sum(axis=1)])
    flux -= uniform_filter1d(flux, size=int(round(0.5 * sr / hop)), mode="nearest")  # onsets, not loudness
    peaks = maximum_filter1d(np.maximum(flux / (flux.std() + 1e-9), 0), PEAK_WIDTH)
    return (peaks - peaks.mean()) / (peaks.std() + 1e-9)


def best_lag(ref: np.ndarray, other: np.ndarray, guard: int) -> tuple[float, float]:
    """Lag (in envelope steps) that lines ``other`` up with ``ref``, and how much it beats the runner-up.

    A lag k means ``other``'s step n matches ``ref``'s step n + k. The runner-up is the best peak
    more than ``guard`` steps away. The lag is refined to a fraction of a step.
    """
    from scipy.fft import irfft, next_fast_len, rfft

    nfft = next_fast_len(len(ref) + len(other) - 1)
    c = irfft(rfft(ref, nfft) * np.conj(rfft(other, nfft)), nfft)
    c = np.concatenate([c[nfft - len(other) + 1:], c[:len(ref)]])   # lags -(len(other)-1) .. len(ref)-1
    lags = np.arange(-(len(other) - 1), len(ref))
    i = int(np.argmax(c))
    far = np.abs(lags - lags[i]) > guard
    runner_up = c[far].max() if far.any() else 0.0
    confidence = float(c[i] / runner_up) if runner_up > 0 else math.inf
    frac = 0.0
    if 0 < i < len(c) - 1:   # parabola through the peak and its neighbours
        a, b, d = c[i - 1], c[i], c[i + 1]
        denom = a - 2 * b + d
        frac = 0.5 * (a - d) / denom if denom < 0 else 0.0
    return float(lags[i] + frac), confidence


@dataclass(frozen=True)
class SyncResult:
    offsets: list[float]        # seconds after the earliest clip that each clip starts
    confidence: list[float]     # per clip; the reference (first with audio) gets inf
    notes: list[str]


def sync(audio: list[np.ndarray | None], names: list[str] | None = None, sr: int = SR,
         hop: int = HOP) -> SyncResult:
    """Line the clips up with each other by their audio, against the first clip that has any.

    Offsets are shifted so the earliest clip starts at 0. A clip without audio gets offset 0.
    """
    names = names or [f"clip {i + 1}" for i in range(len(audio))]
    notes: list[str] = []
    envs = [None if a is None or len(a) < N_FFT else onset_envelope(a, sr, hop) for a in audio]
    ref = next((i for i, e in enumerate(envs) if e is not None), None)
    offsets = [0.0] * len(audio)
    conf = [math.inf] * len(audio)
    if ref is None:
        return SyncResult(offsets, conf, ["no clip has audio; nothing to sync, all start together"])
    guard = int(round(0.5 * sr / hop))
    for i, env in enumerate(envs):
        if env is None:
            notes.append(f"{names[i]} has no audio; placed at the start, line it up by hand")
            continue
        if i == ref:
            continue
        lag, conf[i] = best_lag(envs[ref], env, guard)
        offsets[i] = lag * hop / sr
        if conf[i] < MIN_SYNC_CONFIDENCE:
            notes.append(f"{names[i]}: sync is uncertain (peak only {conf[i]:.2f}x the next best); check it by ear")
    with_audio = [o for o, e in zip(offsets, envs) if e is not None]
    lo = min(with_audio)
    offsets = [o - lo if e is not None else 0.0 for o, e in zip(offsets, envs)]
    return SyncResult(offsets, conf, notes)


# ---- Fusion ----------------------------------------------------------------------------------

def _background(name: str, size: tuple[int, int], pos: tuple[int, int]) -> str:
    """A transparent Background of ``size``."""
    return (
        f"\t\t{name} = Background {{\n"
        "\t\t\tNameSet = true,\n"
        "\t\t\tInputs = {\n"
        "\t\t\t\tUseFrameFormatSettings = Input { Value = 0, },\n"
        f"\t\t\t\tWidth = Input {{ Value = {size[0]}, }},\n"
        f"\t\t\t\tHeight = Input {{ Value = {size[1]}, }},\n"
        "\t\t\t\tTopLeftRed = Input { Value = 0, },\n"
        "\t\t\t\tTopLeftGreen = Input { Value = 0, },\n"
        "\t\t\t\tTopLeftBlue = Input { Value = 0, },\n"
        "\t\t\t\tTopLeftAlpha = Input { Value = 0, },\n"
        "\t\t\t},\n"
        f"\t\t\tViewInfo = OperatorInfo {{ Pos = {{ {pos[0]}, {pos[1]} }} }},\n"
        "\t\t},\n"
    )


def build_setting(slot: Slot, canvas: tuple[int, int], src_size: tuple[int, int], framing: Framing) -> str:
    """The .setting text for one circle (the same Lua table text Fusion puts on the clipboard)."""
    W, H = canvas
    D = slot.square
    # Head in Fusion coords (x from the left, Y up). A Transform's Center 0.5 is no move, so
    # Center = 1 - head moves the head to the middle of the picture.
    hx = framing.head[0] / src_size[0]
    hy = 1.0 - framing.head[1] / src_size[1]
    place = (slot.center[0] / W, 1.0 - slot.center[1] / H)
    circle = slot.diameter / D                      # mask units: fractions of the square's width
    soft = slot.soft * slot.diameter / D
    return (
        "{\n"
        "\tTools = ordered() {\n"
        "\t\tCircleFace = Transform {\n"
        "\t\t\tNameSet = true,\n"
        "\t\t\tInputs = {\n"
        f"\t\t\t\tCenter = Input {{ Value = {{ {1.0 - hx:.7f}, {1.0 - hy:.7f} }}, }},\n"
        f"\t\t\t\tInput = Input {{ SourceOp = \"{MEDIA_IN}\", Source = \"Output\", }},\n"
        "\t\t\t},\n"
        "\t\t\tViewInfo = OperatorInfo { Pos = { 110, 0 } },\n"
        "\t\t},\n"
        + _background("CircleSquare", (D, D), (220, -66))
        + "\t\tCircleMask = EllipseMask {\n"
        "\t\t\tNameSet = true,\n"
        "\t\t\tInputs = {\n"
        f"\t\t\t\tSoftEdge = Input {{ Value = {soft:.7f}, }},\n"
        "\t\t\t\tOutputSize = Input { Value = FuID { \"Custom\" }, },\n"
        "\t\t\t\tUseFrameFormatSettings = Input { Value = 0, },\n"
        f"\t\t\t\tMaskWidth = Input {{ Value = {D}, }},\n"
        f"\t\t\t\tMaskHeight = Input {{ Value = {D}, }},\n"
        "\t\t\t\tPixelAspect = Input { Value = { 1, 1 }, },\n"
        "\t\t\t\tClippingMode = Input { Value = FuID { \"None\" }, },\n"
        "\t\t\t\tCenter = Input { Value = { 0.5, 0.5 }, },\n"
        f"\t\t\t\tWidth = Input {{ Value = {circle:.7f}, }},\n"
        f"\t\t\t\tHeight = Input {{ Value = {circle:.7f}, }},\n"
        "\t\t\t},\n"
        "\t\t\tViewInfo = OperatorInfo { Pos = { 220, 66 } },\n"
        "\t\t},\n"
        "\t\tCircleZoom = Merge {\n"
        "\t\t\tNameSet = true,\n"
        "\t\t\tInputs = {\n"
        "\t\t\t\tBackground = Input { SourceOp = \"CircleSquare\", Source = \"Output\", },\n"
        "\t\t\t\tForeground = Input { SourceOp = \"CircleFace\", Source = \"Output\", },\n"
        "\t\t\t\tCenter = Input { Value = { 0.5, 0.5 }, },\n"
        f"\t\t\t\tSize = Input {{ Value = {framing.zoom:.7f}, }},\n"
        "\t\t\t\tEffectMask = Input { SourceOp = \"CircleMask\", Source = \"Mask\", },\n"
        "\t\t\t},\n"
        "\t\t\tViewInfo = OperatorInfo { Pos = { 330, 0 } },\n"
        "\t\t},\n"
        + _background("CircleCanvas", (W, H), (440, -66))
        + "\t\tCirclePlace = Merge {\n"
        "\t\t\tNameSet = true,\n"
        "\t\t\tInputs = {\n"
        "\t\t\t\tBackground = Input { SourceOp = \"CircleCanvas\", Source = \"Output\", },\n"
        "\t\t\t\tForeground = Input { SourceOp = \"CircleZoom\", Source = \"Output\", },\n"
        f"\t\t\t\tCenter = Input {{ Value = {{ {place[0]:.7f}, {place[1]:.7f} }}, }},\n"
        "\t\t\t\tSize = Input { Value = 1, },\n"
        "\t\t\t},\n"
        "\t\t\tViewInfo = OperatorInfo { Pos = { 550, 0 } },\n"
        "\t\t},\n"
        "\t},\n"
        "\tActiveTool = \"CircleZoom\"\n"
        "}\n"
    )


def anchor_point(slot: Slot, canvas: tuple[int, int]) -> tuple[float, float]:
    """Edit-page Anchor Point that puts the circle's centre under the Inspector's Zoom.

    Resolve measures it in timeline px from the frame centre; Y is taken as up (to verify in Resolve).
    """
    W, H = canvas
    return slot.center[0] - W / 2, H / 2 - slot.center[1]
