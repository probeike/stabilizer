import numpy as np
import pytest

from headlock import geometry as g

SRC = (3840, 2160)
SEQ = (1920, 1080)
LOCK = (0.5, 0.2)


def test_head_lands_on_lock_point():
    rng = np.random.default_rng(0)
    heads = rng.uniform([0, 0], SRC, size=(50, 2))
    for s in (0.5, 0.86, 1.3):
        P = g.positions(heads, SRC, SEQ, s, LOCK)
        c = np.array(SRC) / 2
        landed = P + s * (heads - c)
        np.testing.assert_allclose(landed, np.tile(g.lock_point(SEQ, LOCK), (50, 1)))


def test_centered_head_fits_and_reports_window():
    heads = np.tile([1920.0, 900.0], (10, 1))
    f = g.check(heads, SRC, SEQ, 0.86, LOCK)
    assert f.fits
    assert f.window == pytest.approx((1920 / 0.86, 1080 / 0.86))


def test_overflow_near_left_edge_and_auto_scale_fixes_it():
    heads = np.array([[1920.0, 900.0], [500.0, 900.0], [1920.0, 900.0]])
    f = g.check(heads, SRC, SEQ, 0.86, LOCK)
    assert not f.fits
    # visible left edge = hx - Lx/s = 500 - 960/0.86
    assert f.overflow["left"] == pytest.approx(960 / 0.86 - 500)
    assert f.worst_frame["left"] == 1
    assert f.overflow["right"] == 0

    s = g.auto_scale(heads, SRC, SEQ, LOCK, max_scale=4.0)
    assert s == pytest.approx(960 / 500 * 1.02)
    assert g.check(heads, SRC, SEQ, s, LOCK).fits


def test_min_scale_is_tight():
    rng = np.random.default_rng(1)
    heads = rng.uniform([800, 400], [3000, 1500], size=(200, 2))
    s = g.min_scale(heads, SRC, SEQ, LOCK)
    assert g.check(heads, SRC, SEQ, s, LOCK).fits
    assert not g.check(heads, SRC, SEQ, s * 0.99, LOCK).fits


def test_head_on_edge_is_infeasible():
    heads = np.array([[0.0, 900.0]])
    assert g.min_scale(heads, SRC, SEQ, LOCK) == np.inf
    assert g.auto_scale(heads, SRC, SEQ, LOCK) == 1.0  # capped, check() then reports overflow
