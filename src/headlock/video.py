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
            ":format=duration",
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
    return VideoInfo(path, int(stream["width"]), int(stream["height"]), fps, frame_count, has_audio)


def iter_frames(path: str | Path, max_height: int | None = 1080) -> Iterator[np.ndarray]:
    """Yield BGR uint8 frames, downscaled so height <= max_height (aspect preserved)."""
    info = probe(path)
    w, h = info.width, info.height
    if max_height and h > max_height:
        w = round(w * max_height / h / 2) * 2
        h = max_height
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0"]
    if (w, h) != (info.width, info.height):
        cmd += ["-vf", f"scale={w}:{h}:flags=area"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    frame_bytes = w * h * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


class VideoWriter:
    """Pipe BGR frames into an ffmpeg H.264 encode."""

    def __init__(self, path: str | Path, width: int, height: int, fps: Fraction):
        self.proc = subprocess.Popen(
            [
                "ffmpeg", "-v", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
                "-r", f"{fps.numerator}/{fps.denominator}", "-i", "-",
                "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

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
