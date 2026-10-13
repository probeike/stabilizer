import re
import shutil
import subprocess
from argparse import Namespace
from fractions import Fraction

import numpy as np
import pytest

from headlock import cli, geometry, sphere
from headlock.trackio import Track
from headlock.video import iter_frames


def _flat_track(size=(3840, 2160), n=30):
    t = np.arange(n)
    head = np.column_stack([0.5 + 0.05 * np.sin(t / 5), 0.3 + 0.03 * np.cos(t / 7)])
    return Track("v.mp4", "v.mp4", size, Fraction(30), head, np.ones(n))


def _args(**kw):
    base = dict(output_resolution=None, lock_x=0.5, lock_y=None, scale="auto", max_scale=1.0, fov=100.0,
                smooth="savgol", cutoff_hz=6.0, target="resolve", preview=None)
    return Namespace(**{**base, **kw})


@pytest.mark.parametrize("size,projection,want_res,want_lock_y", [
    ((3840, 2160), "flat", (1920, 1080), 0.2),
    ((2160, 3840), "flat", (1080, 1920), 0.2),
    ((5760, 2880), "equirect", (1080, 1920), 0.35),
])
def test_defaults_follow_the_source(size, projection, want_res, want_lock_y):
    track = _flat_track(size)
    track.projection = projection
    args = _args()
    cli._fill_defaults(track, args)
    assert args.output_resolution == want_res and args.lock_y == want_lock_y


def test_export_writes_resolve_setting_by_default(tmp_path):
    track = _flat_track()
    track.save(tmp_path / "take.track.json")
    cli.main(["export", str(tmp_path / "take.track.json")])
    text = (tmp_path / "take.headlock.setting").read_text()
    assert "HeadlockMerge = Merge" in text and text.count("Linear = true") == 2 * 30
    cli.main(["export", str(tmp_path / "take.track.json"), "--target", "premiere"])
    assert (tmp_path / "take.headlock.xml").read_text().count("<keyframe>") == 30


def test_360_smoothing_is_seamless():
    n = 120
    t = np.arange(n) / 30
    lon = np.pi + 0.5 * np.sin(2 * np.pi * 0.4 * t)        # swings back and forth over the seam
    lat = 0.2 * np.sin(2 * np.pi * 0.3 * t)
    truth = sphere.uv_to_dirs(np.column_stack([(lon / (2 * np.pi) + 0.5) % 1, 0.5 - lat / np.pi]))
    uv = sphere.dirs_to_uv(truth)
    uv[[10, 11, 50]] = np.nan
    track = Track("v.mp4", "v.mp4", (5760, 2880), Fraction(30), uv, np.ones(n), projection="equirect")
    d = cli._smoothed(track, _args())
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1)
    err_deg = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", d, truth), -1, 1)))
    assert err_deg.max() < 0.5


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg/ffprobe")
def test_360_render_end_to_end(tmp_path):
    """Synthetic 360 clip -> track.json -> rendered 9:16: the dot must sit on the lock point."""
    clip = tmp_path / "dot360.mp4"
    cli.main(["test-clip", str(clip), "--360", "--size", "1024x512", "--fps", "10", "--seconds", "2"])
    out = tmp_path / "out.mp4"
    cli.main(["export", str(tmp_path / "dot360.track.json"), "--smooth", "none",
              "--output-resolution", "360x640", "-o", str(out)])
    L = geometry.lock_point((360, 640), (0.5, 0.35))
    x0, y0 = int(L[0]) - 30, int(L[1]) - 30
    ys, xs = np.mgrid[0:60, 0:60] + 0.5
    n = 0
    for frame in iter_frames(out, max_height=None):
        # Weight by redness: only the dot's red centre counts, not the grid or the yellow equator
        # (4:2:0 chroma smears thin yellow lines toward white, but nothing else is strongly red).
        win = frame[y0:y0 + 60, x0:x0 + 60].astype(float)
        w = np.clip(win[..., 2] - win[..., :2].max(axis=2) - 80, 0, None)
        assert w.sum() > 0
        # 2 px (~0.4 deg): the clip and the output are both lossy H.264, which nudges a tiny dot by ~1 px.
        # The exact geometry is checked uncompressed in test_sphere; a wrong frame, flipped axis or
        # mis-aim here would be tens of px off.
        np.testing.assert_allclose([x0 + (w * xs).sum() / w.sum(), y0 + (w * ys).sum() / w.sum()], L, atol=2.0)
        n += 1
    assert n == 20


def _clip(path, size, fps, seconds, wav=None):
    cmd = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"color=c=gray:s={size}:r={fps}:d={seconds}"]
    cmd += ["-i", str(wav), "-shortest", "-c:a", "pcm_s16le"] if wav else []
    subprocess.run(cmd + ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)


class _FakePose:
    """Every frame: a head at (60%, 40%) of the picture, neck to head-top a tenth of its height."""

    def __init__(self, *args):
        self.boxes = None

    def __call__(self, frame):
        from headlock.head import HEAD_TOP, NECK
        h, w = frame.shape[:2]
        k, s = np.zeros((26, 2)), np.zeros(26)
        k[NECK], k[HEAD_TOP] = [0.6 * w, (0.4 + 0.1 * 2 / 3) * h], [0.6 * w, (0.4 - 0.1 / 3) * h]
        s[[NECK, HEAD_TOP]] = 0.9
        return k, s


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg/ffprobe")
def test_circles_end_to_end(tmp_path, monkeypatch, capsys):
    """Three takes of one rhythm, started at different points: synced, framed, and built in (fake) Resolve."""
    from types import SimpleNamespace

    from fake_resolve import FakeResolve, readstring
    from scipy.io import wavfile
    from test_circles import _take

    from headlock import circles, pose, resolve_api

    rng = np.random.default_rng(5)
    onsets = np.cumsum(rng.uniform(0.15, 0.6, 40))
    starts = {"left": 1.0, "middle": 0.0, "right": 2.5}       # song time each take started recording
    clips = {}
    for name, start in starts.items():
        wav = tmp_path / f"{name}.wav"
        wavfile.write(wav, circles.SR, _take(onsets, 8, start, 220, 0.5, rng, jitter=0.03))
        clips[name] = tmp_path / f"take_{name}.mov"
        _clip(clips[name], "180x320", 30, 8, wav)
    main = tmp_path / "song.mp4"
    _clip(main, "216x384", "30000/1001", 10)
    monkeypatch.setattr(pose, "PoseEstimator", _FakePose)
    argv = ["circles", str(main), *(a for n, p in clips.items() for a in (f"--{n}", str(p)))]

    cli.main(argv + ["--no-resolve"])
    out = capsys.readouterr().out
    for name, start in starts.items():
        said = re.search(rf"{name}: take_{name}.mov starts ([0-9.]+)s", out)
        assert float(said[1]) == pytest.approx(start, abs=1 / 30)
        text = (tmp_path / f"take_{name}.{name}.circle.setting").read_text()
        assert "CirclePlace = Merge" in text

    resolve = FakeResolve()
    monkeypatch.setattr(resolve_api, "connect", lambda: (resolve, SimpleNamespace(readstring=readstring)))
    cli.main(argv + ["--at", "0:02"])
    assert 'Built the timeline "song circles"' in capsys.readouterr().out
    tl = resolve.project.current
    assert tl.settings["timelineResolutionWidth"] == "216" and tl.settings["timelineFrameRate"] == "29.97"
    records = {(i.clip.path.rsplit("/", 1)[-1], i.media_type): i.record - 86400 for i in tl.items}
    for name, start in starts.items():
        want = round((2 + start) * 30000 / 1001)
        assert abs(records[(f"take_{name}.mov", 1)] - want) <= 1
        assert records[(f"take_{name}.mov", 2)] == records[(f"take_{name}.mov", 1)]
    comp = next(i for i in tl.items if i.clip.path.endswith("take_left.mov") and i.media_type == 1).comps[0]
    face = comp.tools["CircleFace"].inputs["Center"]
    assert (face[1], face[2]) == pytest.approx((1 - 0.6, 1 - (1 - 0.4)), abs=0.01)   # head moved to the middle
