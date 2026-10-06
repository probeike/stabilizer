import numpy as np
import pytest

from headlock.smooth import fill_gaps, find_gaps, savgol_window, smooth

FPS = 30.0


def _signal(n=300):
    t = np.arange(n) / FPS
    x = 1000 + 200 * np.sin(2 * np.pi * 0.5 * t)
    y = 500 + 100 * np.sin(2 * np.pi * 0.8 * t + 1)
    return np.column_stack([x, y])


@pytest.mark.parametrize("method", ["savgol", "butter"])
def test_jitter_removed(method):
    clean = _signal()
    rng = np.random.default_rng(0)
    noisy = clean + rng.uniform(-3, 3, clean.shape)
    out = smooth(noisy, FPS, method, 6.0)
    inner = slice(15, -15)
    rms_in = np.sqrt(np.mean((noisy[inner] - clean[inner]) ** 2))
    rms_out = np.sqrt(np.mean((out[inner] - clean[inner]) ** 2))
    # A light 6 Hz cutoff at 30 fps still passes ~40% of white-noise bandwidth.
    assert rms_out < 0.65 * rms_in


@pytest.mark.parametrize("method", ["savgol", "butter"])
def test_zero_phase_on_fast_move(method):
    # A fast 150 ms head whip of 100 px; zero-phase filtering must keep it centred in time.
    n = 120
    t = np.arange(n)
    x = 100 / (1 + np.exp(-(t - 60) / 1.2))
    track = np.column_stack([x, np.zeros(n)])
    out = smooth(track, FPS, method, 6.0)
    mid_in = np.argmin(np.abs(x - 50))
    mid_out = np.argmin(np.abs(out[:, 0] - 50))
    assert mid_out == mid_in
    assert out[:, 0].max() < 100 + 15  # bounded overshoot
    assert out[:, 0].min() > -15


def test_savgol_window():
    assert savgol_window(30, 6) in (7, 9)
    assert savgol_window(60, 6) > savgol_window(30, 6)
    assert savgol_window(30, 6) % 2 == 1


def test_fill_gaps():
    track = np.column_stack([np.arange(10.0), np.arange(10.0) * 2])
    holed = track.copy()
    holed[[0, 1, 4, 5, 9]] = np.nan
    assert find_gaps(~np.isnan(holed).any(axis=1)) == [(0, 2), (4, 6), (9, 10)]
    out = fill_gaps(holed)
    np.testing.assert_allclose(out[4:6], track[4:6])
    np.testing.assert_allclose(out[0], track[2])
    np.testing.assert_allclose(out[9], track[8])
    assert not np.isnan(out).any()
