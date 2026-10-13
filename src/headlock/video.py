"""Video probing and decoding via ffmpeg subprocesses."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: Fraction
    frame_count: int
    has_audio: bool = False

    @property
    def ntsc(self) -> bool:
        """True for 1000/1001 rates (23.976, 29.97, 59.94)."""
        return self.fps.denominator == 1001

    @property
    def timebase(self) -> int:
        """Integer timebase as used by FCP7 XML (30 for 29.97)."""
        return round(float(self.fps))


def probe(path: str | Path) -> VideoInfo:
    path = Path(path)
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=codec_type,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration"
            ":stream_tags=rotate:stream_side_data=rotation:format=duration",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    data = json.loads(out)
    stream = next(s for s in data["streams"] if s["codec_type"] == "video")
    has_audio = any(s["codec_type"] == "audio" for s in data["streams"])
    fps = Fraction(stream.get("r_frame_rate") or stream["avg_frame_rate"])
    if stream.get("nb_frames", "N/A") not in ("N/A", None):
        frame_count = int(stream["nb_frames"])
    else:
        duration = float(stream.get("duration") or data["format"]["duration"])
        frame_count = round(duration * fps)
    width, height = int(stream["width"]), int(stream["height"])
    # Phones store portrait video sideways with a rotation flag; ffmpeg decodes it upright, so report
    # the upright size.
    rotation = next((float(d["rotation"]) for d in stream.get("side_data_list", []) if "rotation" in d),
                    float(stream.get("tags", {}).get("rotate", 0)))
    if round(rotation) % 180 == 90:
        width, height = height, width
    return VideoInfo(path, width, height, fps, frame_count, has_audio)


def iter_frames(path: str | Path, max_height: int | None = 1080, depth: int = 8,
                step: int = 1) -> Iterator[np.ndarray]:
    """Yield BGR frames, downscaled so height <= max_height (aspect preserved).

    depth 8 -> uint8 frames; depth 16 -> uint16, which keeps 10-bit (e.g. log) sources intact.
    step > 1 yields only every step-th frame (0, step, 2*step, ...), dropped before scaling.
    """
    info = probe(path)
    w, h = info.width, info.height
    if max_height and h > max_height:
        w = round(w * max_height / h / 2) * 2
        h = max_height
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0"]
    filters = [f"select=not(mod(n\\,{step}))"] if step > 1 else []
    if (w, h) != (info.width, info.height):
        filters.append(f"scale={w}:{h}:flags=area")
    if filters:
        cmd += ["-vf", ",".join(filters)]
    if step > 1:
        cmd += ["-fps_mode", "passthrough"]
    dtype, pix_fmt = (np.dtype("<u2"), "bgr48le") if depth == 16 else (np.dtype(np.uint8), "bgr24")
    cmd += ["-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]
    frame_bytes = w * h * 3 * dtype.itemsize
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield np.frombuffer(buf, dtype).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def wants_16bit(path: str | Path) -> bool:
    """.mov outputs are ProRes 422 HQ (10-bit), worth feeding 16-bit frames."""
    return Path(path).suffix.lower() == ".mov"


class VideoWriter:
    """Pipe BGR frames into an ffmpeg encode: ProRes 422 HQ for .mov, otherwise H.264.

    ``audio_from`` copies that file's audio (re-encoded: PCM in .mov, AAC otherwise).
    ``depth`` 16 takes uint16 frames, as ``iter_frames(depth=16)`` yields.
    """

    def __init__(self, path: str | Path, width: int, height: int, fps: Fraction,
                 audio_from: str | Path | None = None, depth: int = 8, crf: int = 20):
        cmd = [
            "ffmpeg", "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr48le" if depth == 16 else "bgr24", "-s", f"{width}x{height}",
            "-r", f"{fps.numerator}/{fps.denominator}", "-i", "-",
        ]
        if audio_from:
            cmd += ["-i", str(audio_from), "-map", "0:v", "-map", "1:a:0?"]
        if wants_16bit(path):
            cmd += ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"]
            cmd += ["-c:a", "pcm_s16le"] if audio_from else []
        else:
            cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", str(crf), "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart"]
            cmd += ["-c:a", "aac", "-b:a", "256k"] if audio_from else []
        self.proc = subprocess.Popen(cmd + [str(path)], stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg encode failed")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
