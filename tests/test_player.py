import shutil
import threading
import time
from fractions import Fraction

import cv2
import numpy as np
import pytest

from headlock import geometry, player, sphere
from headlock.trackio import Track
from headlock.video import VideoWriter


def _dot_centre(img):
    """Centroid of the strongly red pixels."""
    w = np.clip(img[..., 2].astype(float) - img[..., :2].max(axis=2) - 80, 0, None)
    ys, xs = np.mgrid[0:img.shape[0], 0:img.shape[1]] + 0.5
    return np.array([(w * xs).sum(), (w * ys).sum()]) / w.sum()


def test_preview_size_keeps_the_shape():
    assert player.preview_size((1080, 1920)) == (360, 640)
    assert player.preview_size((3840, 2160)) == (640, 360)


@pytest.mark.parametrize("fov,lock_y", [(100, 0.35), (70, 0.5)])
def test_360_preview_puts_the_head_on_the_lock_point(fov, lock_y):
    W, H = 1920, 960
    uv = np.array([0.93, 0.41])                      # near the seam, a little above the horizon
    frame = np.full((H, W, 3), 60, np.uint8)
    for x in (uv[0] * W - W, uv[0] * W):              # drawn twice so it wraps the seam
        cv2.circle(frame, (int(x), int(uv[1] * H)), 6, (0, 0, 255), -1)
    track = Track("v.mp4", "v.mp4", (W, H), Fraction(30), np.tile(uv, (10, 1)), np.ones(10), projection="equirect")
    opts = player.Options((1080, 1920), (0.5, lock_y), fov=fov, smooth="none")
    framer = player.Framer(track, opts)
    img = framer(3, frame)
    assert img.shape == (640, 360, 3)
    np.testing.assert_allclose(_dot_centre(img), geometry.lock_point((360, 640), (0.5, lock_y)), atol=1.5)


def test_flat_preview_matches_the_export_pin():
    src = (1920, 1080)
    n = 20
    head = np.column_stack([np.linspace(0.4, 0.6, n), np.full(n, 0.3)])
    track = Track("v.mp4", "v.mp4", src, Fraction(30), head, np.ones(n))
    framer = player.Framer(track, player.Options((1080, 1920), (0.5, 0.2), smooth="none", scale="1.5"))
    i = 7
    frame = np.full((540, 960, 3), 60, np.uint8)      # decoded at half the source size
    cv2.circle(frame, tuple(int(v) for v in head[i] * (960, 540)), 3, (0, 0, 255), -1)
    np.testing.assert_allclose(_dot_centre(framer(i, frame)), geometry.lock_point((360, 640), (0.5, 0.2)), atol=1.5)
    assert framer.note    # 1.5x of a landscape clip into a vertical frame shows past its edges


def test_marks_draw_without_error():
    uv = np.array([[0.5, 0.5]] * 5)
    track = Track("v.mp4", "v.mp4", (1920, 960), Fraction(30), uv, np.ones(5), projection="equirect")
    img = player.Framer(track, player.Options((1080, 1920), (0.5, 0.35)), marks=True)(0, np.zeros((960, 1920, 3), np.uint8))
    assert img.any()


def test_to_ppm():
    img = np.zeros((2, 3, 3), np.uint8)
    img[0, 0] = (255, 0, 0)                            # blue in BGR
    ppm = player.to_ppm(img)
    assert ppm.startswith(b"P6 3 2 255\n") and ppm[len(b"P6 3 2 255\n"):][:3] == bytes([0, 0, 255])


@pytest.fixture
def ramp(tmp_path):
    path = tmp_path / "ramp.mp4"
    with VideoWriter(path, 64, 48, Fraction(30)) as w:
        for i in range(30):
            w.write(np.full((48, 64, 3), 8 * i, np.uint8))
    return path


class _Shown:
    def __init__(self):
        self.frames, self.cond = [], threading.Condition()

    def __call__(self, i, img):
        with self.cond:
            self.frames.append((i, img))
            self.cond.notify_all()

    def wait(self, pred, timeout=5.0):
        with self.cond:
            assert self.cond.wait_for(lambda: pred(self.frames), timeout), self.frames[-3:]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg/ffprobe")
def test_playback_seeks_plays_and_redraws(ramp):
    shown = _Shown()
    level = lambda i, f: (i, round(float(f.mean()) / 8))   # noqa: E731
    p = player.Playback(ramp, Fraction(30), 30, level, shown, decode_height=None)
    try:
        shown.wait(lambda fs: fs)                          # opens on frame 0
        assert shown.frames[0][1] == (0, 0)
        p.seek(12)
        shown.wait(lambda fs: fs[-1][0] == 12)
        assert shown.frames[-1][1] == (12, 12)
        p.set_render(lambda i, f: "redrawn")               # paused: the same frame again, newly rendered
        shown.wait(lambda fs: fs[-1] == (12, "redrawn"))
        p.set_render(level)
        start = time.monotonic()
        p.play()
        shown.wait(lambda fs: fs[-1][0] == 29)
        assert time.monotonic() - start > 0.4              # kept to 30 fps, not as fast as it decodes
        assert all(v == (i, i) for i, v in shown.frames if isinstance(v, tuple))   # each index is its frame
        deadline = time.monotonic() + 2
        while p.playing and time.monotonic() < deadline:   # stops once it reads past the end
            time.sleep(0.01)
        assert not p.playing
        p.play()                                           # play again from the top
        shown.wait(lambda fs: fs[-1][0] == 29 and any(i == 0 for i, _ in fs[3:]))
    finally:
        p.close()
