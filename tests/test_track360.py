import cv2
import numpy as np

from headlock import sphere
from headlock.head import HEAD_TOP, L_EAR, NECK, R_EAR
from headlock.track360 import track_equirect

EQ = (2048, 1024)


class FakeEstimator:
    """Stands in for YOLOX + RTMPose: the white dot is the 'head'.

    ``det`` boxes each white blob (so a dot cut by the seam gives two small boxes); calling it
    returns neck/top/ears placed symmetrically around the dot so the head centre is the dot.
    """

    def det(self, img):
        n, _, stats, _ = cv2.connectedComponentsWithStats((img[..., 0] > 127).astype(np.uint8))
        return np.array([[x, y, x + w, y + h] for x, y, w, h, _ in stats[1:]], dtype=float).reshape(-1, 4)

    def __call__(self, view):
        k, s = np.zeros((26, 2)), np.zeros(26)
        ys, xs = np.nonzero(view[..., 0] > 127)
        if len(xs) == 0:
            return k, s
        c = np.array([xs.mean() + 0.5, ys.mean() + 0.5])
        k[NECK], k[HEAD_TOP] = c + [0, 30], c - [0, 30]
        k[L_EAR], k[R_EAR] = c - [20, 0], c + [20, 0]
        s[[NECK, HEAD_TOP, L_EAR, R_EAR]] = 0.95
        return k, s


def _frames(dirs, radius_deg=2.0):
    W, H = EQ
    xs, ys = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    rays = sphere.uv_to_dirs(np.stack([xs / W, ys / H], axis=-1))
    for d in dirs:
        img = np.zeros((H, W, 3), np.uint8)
        if d is not None:
            img[np.degrees(np.arccos(np.clip(rays @ d, -1, 1))) < radius_deg] = 255
        yield img


def test_tracks_head_across_the_seam():
    lon = np.linspace(179.5, 195, 10)   # first sighting straddles the +-180 seam, then crosses it
    lat = np.linspace(-15, 10, 10)
    want = sphere.uv_to_dirs(np.column_stack([(lon / 360 + 0.5) % 1, 0.5 - lat / 180]))
    # Frame 0 has nobody in shot yet; tracking starts on frame 1.
    uv, conf, fraction, kpts, scores = track_equirect(_frames([None, *want]), FakeEstimator())

    assert len(uv) == 11
    assert np.isnan(uv[0]).all()
    got = sphere.uv_to_dirs(uv[1:])
    err_deg = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", got, want), -1, 1)))
    assert err_deg.max() < 0.25, err_deg
    assert abs(fraction - 0.5) < 1e-6
