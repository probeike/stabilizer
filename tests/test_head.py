import numpy as np

from headlock.head import HEAD_TOP, L_EAR, NECK, R_EAR, calibrate_fraction, head_centers


def _frames(n, ear_conf=0.95, ear_frac=0.6):
    kpts = np.zeros((n, 26, 2))
    scores = np.full((n, 26), 0.9)
    kpts[:, NECK] = [500, 400]
    kpts[:, HEAD_TOP] = [500, 250]          # neck->top = (0, -150)
    ear_y = 400 - ear_frac * 150
    kpts[:, L_EAR] = [470, ear_y]
    kpts[:, R_EAR] = [530, ear_y]
    scores[:, [L_EAR, R_EAR]] = ear_conf
    return kpts, scores


def test_calibrated_fraction_matches_ears_so_blend_has_no_step():
    kpts, scores = _frames(40, ear_frac=0.6)
    assert abs(calibrate_fraction(kpts, scores, 0.3) - 0.6) < 1e-9
    # Ears fade out halfway through (dancer turns away): the centre must not move.
    scores[20:, [L_EAR, R_EAR]] = 0.1
    center, conf, frac = head_centers(kpts, scores)
    np.testing.assert_allclose(center, np.tile([500, 310], (40, 1)))


def test_default_fraction_without_confident_ears():
    kpts, scores = _frames(40, ear_conf=0.2)
    center, _, frac = head_centers(kpts, scores)
    assert abs(frac - 2 / 3) < 1e-9
    np.testing.assert_allclose(center[0], [500, 300])


def test_low_confidence_becomes_nan_unless_ears_solid():
    kpts, scores = _frames(30)
    scores[5, [NECK, HEAD_TOP]] = 0.1              # ears still solid -> ear midpoint
    scores[6, [NECK, HEAD_TOP, L_EAR, R_EAR]] = 0.1  # nothing reliable -> NaN
    center, conf, _ = head_centers(kpts, scores, min_conf=0.3)
    np.testing.assert_allclose(center[5], [500, 310])
    assert np.isnan(center[6]).all()
    assert not np.isnan(np.delete(center, 6, axis=0)).any()
