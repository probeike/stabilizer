"""Head tracking in 360 (equirectangular) video.

The pose model is trained on ordinary photos, so rather than running it on the stretched equirect
frame it runs on a perspective view that is re-aimed at the head every frame. The view never sits
on the +-180 deg seam or near a pole, so the dancer/runner is never split or smeared.
"""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from . import geometry, sphere
from .head import DEFAULT_FRACTION, head_centers

VIEW_SIZE = (960, 960)
VIEW_FOV = 120.0
VIEW_LOCK = (0.5, 0.35)  # head a little above centre so the shoulders are in view too


def _area(b: np.ndarray) -> float:
    return float(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))


def find_person(est, frame: np.ndarray) -> np.ndarray | None:
    """Direction to the largest detected person in an equirect frame, or None.

    Detects twice, once with the frame rolled half a turn, so someone standing on the seam is
    whole in one of the two. The largest box is the selfie-taker (closest to the camera).
    """
    h, w = frame.shape[:2]
    best = None
    for shift in (0, w // 2):
        img = np.roll(frame, -shift, axis=1) if shift else frame
        for b in np.asarray(est.det(img), dtype=float).reshape(-1, 4):
            if best is None or _area(b) > best[0]:
                cx = ((b[0] + b[2]) / 2 + shift) % w
                cy = b[1] + 0.2 * (b[3] - b[1])  # upper body, nearer the head than the box centre
                best = (_area(b), cx / w, cy / h)
    return None if best is None else sphere.uv_to_dirs(np.array(best[1:]))


def track_equirect(frames: Iterable[np.ndarray], est, min_conf: float = 0.3):
    """Track the head through equirect frames.

    Returns (head_uv (N, 2) normalized equirect, NaN where unreliable; conf (N,); neck->top
    fraction; keypoints (N, 26, 2) and scores (N, 26) in tracking-view pixels).
    """
    f = sphere.focal(VIEW_SIZE, VIEW_FOV)
    rays = sphere.view_rays(VIEW_SIZE, f)
    lock_ray = sphere.pixels_to_rays(geometry.lock_point(VIEW_SIZE, VIEW_LOCK), VIEW_SIZE, f)
    target = None  # direction the view is aimed at (last good head)
    kpts, scores, rots = [], [], []
    for frame in frames:
        if target is None:
            target = find_person(est, frame)
            if target is None:
                kpts.append(np.zeros((26, 2)))
                scores.append(np.zeros(26))
                rots.append(np.eye(3))
                continue
        R = sphere.rotation(*sphere.aim(target, lock_ray))
        mx, my = sphere.equirect_maps(R, rays, (frame.shape[1], frame.shape[0]))
        view = cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        k, s = est(view)
        kpts.append(k)
        scores.append(s)
        rots.append(R)
        c, _, _ = head_centers(k[None], s[None], min_conf, fraction=DEFAULT_FRACTION)
        if not np.isnan(c).any():
            target = R @ sphere.pixels_to_rays(c[0], VIEW_SIZE, f)

    if not kpts:
        raise ValueError("no frames decoded")
    kpts, scores, rots = np.array(kpts), np.array(scores), np.array(rots)
    center, conf, fraction = head_centers(kpts, scores, min_conf)
    dirs = np.einsum("nij,nj->ni", rots, sphere.pixels_to_rays(center, VIEW_SIZE, f))
    return sphere.dirs_to_uv(dirs), conf, fraction, kpts, scores
