import shutil
import subprocess
from fractions import Fraction

import numpy as np
import pytest

from headlock.video import VideoWriter, iter_frames, probe


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg/ffprobe")
def test_iter_frames_step_yields_every_nth_frame(tmp_path):
    path = tmp_path / "ramp.mp4"
    with VideoWriter(path, 64, 48, Fraction(30)) as w:
        for i in range(30):
            w.write(np.full((48, 64, 3), 8 * i, np.uint8))
    levels = [float(f.mean()) / 8 for f in iter_frames(path, max_height=24, step=4)]
    np.testing.assert_allclose(levels, range(0, 30, 4), atol=0.3)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg/ffprobe")
def test_phone_rotation_flag_reports_the_upright_size(tmp_path):
    """Phones store portrait clips sideways plus a rotation flag; ffmpeg decodes them upright."""
    stored = tmp_path / "stored.mp4"
    with VideoWriter(stored, 64, 32, Fraction(30)) as w:
        for _ in range(3):
            w.write(np.zeros((32, 64, 3), np.uint8))
    phone = tmp_path / "phone.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-display_rotation:v:0", "90", "-i", str(stored), "-c", "copy",
                    str(phone)], check=True)
    info = probe(phone)
    assert (info.width, info.height) == (32, 64)
    assert next(iter_frames(phone, max_height=None)).shape == (64, 32, 3)
