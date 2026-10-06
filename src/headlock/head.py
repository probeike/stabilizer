"""Head-centre estimation from Halpe26 keypoints."""

from __future__ import annotations

import numpy as np

L_EAR, R_EAR, HEAD_TOP, NECK = 3, 4, 17, 18

DEFAULT_FRACTION = 2 / 3   # how far from neck to head-top the skull centre sits
EAR_LO, EAR_HI = 0.5, 0.8  # ear confidence ramp for blending in the ear midpoint


def calibrate_fraction(kpts: np.ndarray, scores: np.ndarray, min_conf: float) -> float:
    """Neck->top fraction that best matches the ear midpoint, measured on frames where both are confident.

    Using the dancer's own fraction instead of a fixed 2/3 means the neck/top estimate and the ear
    midpoint agree, so the blend weight changing (head turning) doesn't make the point step.
    """
    neck, top = kpts[:, NECK], kpts[:, HEAD_TOP]
    ears = (kpts[:, L_EAR] + kpts[:, R_EAR]) / 2
    axis = top - neck
    length2 = np.einsum("ij,ij->i", axis, axis)
    good = (
        (np.minimum(scores[:, L_EAR], scores[:, R_EAR]) >= EAR_HI)
        & (np.minimum(scores[:, NECK], scores[:, HEAD_TOP]) >= max(min_conf, EAR_HI))
        & (length2 > 1.0)
    )
    if good.sum() < 10:
        return DEFAULT_FRACTION
    frac = np.einsum("ij,ij->i", ears[good] - neck[good], axis[good]) / length2[good]
    return float(np.clip(np.median(frac), 0.4, 0.85))


def head_centers(kpts: np.ndarray, scores: np.ndarray, min_conf: float = 0.3,
                 fraction: float | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-frame head centre (N, 2), NaN where unreliable, plus confidence (N,) and the fraction used.

    kpts: (N, 26, 2) pixel coordinates; scores: (N, 26).
    """
    kpts = np.asarray(kpts, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if fraction is None:
        fraction = calibrate_fraction(kpts, scores, min_conf)

    neck, top = kpts[:, NECK], kpts[:, HEAD_TOP]
    primary = neck + fraction * (top - neck)
    primary_conf = np.minimum(scores[:, NECK], scores[:, HEAD_TOP])

    ears = (kpts[:, L_EAR] + kpts[:, R_EAR]) / 2
    ear_conf = np.minimum(scores[:, L_EAR], scores[:, R_EAR])
    w = np.clip((ear_conf - EAR_LO) / (EAR_HI - EAR_LO), 0.0, 1.0)

    primary_ok = primary_conf >= min_conf
    # Primary unreliable but ears solid: use the ears alone.
    w = np.where(primary_ok, w, np.where(ear_conf >= EAR_HI, 1.0, 0.0))
    center = w[:, None] * ears + (1 - w[:, None]) * primary
    conf = np.where(primary_ok, primary_conf, np.where(ear_conf >= EAR_HI, ear_conf, primary_conf))

    ok = primary_ok | (ear_conf >= EAR_HI)
    center[~ok] = np.nan
    return center, conf, fraction
