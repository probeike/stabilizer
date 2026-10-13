import cv2
import numpy as np
import pytest

from headlock import geometry, sphere

SIZE = (270, 480)   # small 9:16 view
FOV = 100.0


def _dir(lon_deg, lat_deg):
    return sphere.uv_to_dirs(np.array([lon_deg / 360 + 0.5, 0.5 - lat_deg / 180]))


def _dots_equirect(dirs, size=(2048, 1024), radius_deg=1.5):
    """Equirect frame with a round white dot (on the sphere) at each direction."""
    W, H = size
    xs, ys = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    rays = sphere.uv_to_dirs(np.stack([xs / W, ys / H], axis=-1))
    img = np.zeros((H, W, 3), np.uint8)
    for d in dirs:
        img[np.degrees(np.arccos(np.clip(rays @ d, -1, 1))) < radius_deg] = 255
    return img


def _reframe(img, d, lock, size=SIZE, fov=FOV):
    f = sphere.focal(size, fov)
    L = geometry.lock_point(size, lock)
    R = sphere.rotation(*sphere.aim(d, sphere.pixels_to_rays(L, size, f)))
    mx, my = sphere.equirect_maps(R, sphere.view_rays(size, f), (img.shape[1], img.shape[0]))
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP), L


def _centroid(mask):
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean() + 0.5, ys.mean() + 0.5])


def test_uv_dirs_roundtrip_and_axes():
    uv = np.array([[0.5, 0.5], [0.0001, 0.3], [0.9999, 0.7], [0.25, 0.1]])
    d = sphere.uv_to_dirs(uv)
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1)
    np.testing.assert_allclose(sphere.dirs_to_uv(d), uv, atol=1e-12)
    np.testing.assert_allclose(_dir(0, 0), [0, 0, 1], atol=1e-12)    # image centre = forward
    np.testing.assert_allclose(_dir(90, 0), [1, 0, 0], atol=1e-12)   # right of centre = +x
    np.testing.assert_allclose(_dir(0, 90), [0, 1, 0], atol=1e-12)   # top row = up
    assert np.isnan(sphere.dirs_to_uv(np.full(3, np.nan))).all()


@pytest.mark.parametrize("lock", [(0.5, 0.35), (0.5, 0.5), (0.3, 0.2)])
def test_aim_turns_lock_ray_onto_target_without_roll(lock):
    rng = np.random.default_rng(0)
    d = _dir(rng.uniform(-180, 180, 300), rng.uniform(-60, 60, 300))
    f = sphere.focal(SIZE, FOV)
    l = sphere.pixels_to_rays(geometry.lock_point(SIZE, lock), SIZE, f)
    R = sphere.rotation(*sphere.aim(d, l))
    np.testing.assert_allclose(R @ l, d, atol=1e-9)
    np.testing.assert_allclose(R[:, 1, 0], 0, atol=1e-12)  # camera x axis stays level: no roll
    np.testing.assert_allclose(R @ R.transpose(0, 2, 1), np.broadcast_to(np.eye(3), R.shape), atol=1e-12)


def test_pixel_ray_roundtrip():
    f = sphere.focal(SIZE, FOV)
    px = np.array([[0.0, 0.0], [135.0, 240.0], [270.0, 480.0], [12.5, 400.25]])
    np.testing.assert_allclose(sphere.rays_to_pixels(sphere.pixels_to_rays(px, SIZE, f), SIZE, f), px, atol=1e-9)
    assert np.isnan(sphere.rays_to_pixels(np.array([0.0, 0.0, -1.0]), SIZE, f)).all()


@pytest.mark.parametrize("lon,lat,lock", [(0, 0, (0.5, 0.35)), (179.8, -20, (0.5, 0.35)),
                                          (-120, 35, (0.3, 0.2)), (60, -40, (0.7, 0.6))])
def test_reframe_puts_head_on_lock_point(lon, lat, lock):
    d = _dir(lon, lat)
    view, L = _reframe(_dots_equirect([d]), d, lock)
    np.testing.assert_allclose(_centroid(view[..., 0] > 127), L, atol=0.5)


def test_reframe_is_not_mirrored_or_flipped():
    head, right, up = _dir(170, -10), _dir(178, -10), _dir(170, -2)
    lock = (0.5, 0.5)
    view_r, L = _reframe(_dots_equirect([right]), head, lock)
    view_u, _ = _reframe(_dots_equirect([up]), head, lock)
    assert _centroid(view_r[..., 0] > 127)[0] > L[0] + 20   # east of the head shows to its right
    assert _centroid(view_u[..., 0] > 127)[1] < L[1] - 20   # above the head shows higher up
