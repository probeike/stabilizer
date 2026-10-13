"""Sphere math for 360 (equirectangular) sources.

Directions are unit vectors with x right, y up, z forward. In the equirect image longitude 0
(forward) is the horizontal centre and increases to the right; latitude +90 deg is the top row.
Normalized equirect coordinates (u, v) run 0-1 from the top-left, like the flat head track.

A view is a pinhole camera with the same axes, turned by pitch (about x) then yaw (about y) and
never rolled, so a level 360 export (Insta360 horizon lock) keeps a level horizon in the view.
Pixel coordinates are continuous, origin top-left, Y down; pixel i covers [i, i+1).
"""

from __future__ import annotations

import numpy as np


def uv_to_dirs(uv: np.ndarray) -> np.ndarray:
    """Normalized equirect coords (..., 2) -> unit directions (..., 3). NaN stays NaN."""
    uv = np.asarray(uv, dtype=float)
    lon = (uv[..., 0] - 0.5) * 2 * np.pi
    lat = (0.5 - uv[..., 1]) * np.pi
    c = np.cos(lat)
    return np.stack([c * np.sin(lon), np.sin(lat), c * np.cos(lon)], axis=-1)


def dirs_to_uv(d: np.ndarray) -> np.ndarray:
    """Directions (..., 3), any length -> normalized equirect coords (..., 2), u in [0, 1)."""
    d = np.asarray(d, dtype=float)
    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = np.arcsin(np.clip(d[..., 1] / np.linalg.norm(d, axis=-1), -1.0, 1.0))
    return np.stack([(lon / (2 * np.pi) + 0.5) % 1.0, 0.5 - lat / np.pi], axis=-1)


def focal(size: tuple[int, int], fov_deg: float) -> float:
    """Focal length (px) that spans ``fov_deg`` across the longer side of a (w, h) view."""
    return max(size) / 2 / np.tan(np.radians(fov_deg) / 2)


def pixels_to_rays(px: np.ndarray, size: tuple[int, int], f: float) -> np.ndarray:
    """Camera-space unit rays through view pixel coords (..., 2)."""
    px = np.asarray(px, dtype=float)
    w, h = size
    r = np.stack([(px[..., 0] - w / 2) / f, -(px[..., 1] - h / 2) / f, np.ones(px.shape[:-1])], axis=-1)
    return r / np.linalg.norm(r, axis=-1, keepdims=True)


def rays_to_pixels(r: np.ndarray, size: tuple[int, int], f: float) -> np.ndarray:
    """Camera-space rays (..., 3) -> view pixel coords (..., 2); NaN for rays behind the camera."""
    r = np.asarray(r, dtype=float)
    w, h = size
    with np.errstate(divide="ignore", invalid="ignore"):
        px = np.stack([w / 2 + f * r[..., 0] / r[..., 2], h / 2 - f * r[..., 1] / r[..., 2]], axis=-1)
    px[~(r[..., 2] > 0)] = np.nan
    return px


def view_rays(size: tuple[int, int], f: float) -> np.ndarray:
    """(h, w, 3) float32 camera-space rays through every pixel centre of a view."""
    w, h = size
    xs, ys = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    return pixels_to_rays(np.stack([xs, ys], axis=-1), size, f).astype(np.float32)


def rotation(yaw: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    """Camera-to-world rotation(s) Ry(yaw) @ Rx(pitch), shape (..., 3, 3)."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    zero, one = np.zeros_like(cy), np.ones_like(cy)
    return np.stack([
        np.stack([cy, -sy * sp, sy * cp], axis=-1),
        np.stack([zero, cp, sp], axis=-1),
        np.stack([-sy, -cy * sp, cy * cp], axis=-1),
    ], axis=-2)


def aim(dirs: np.ndarray, lock_ray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(yaw, pitch) that turn camera ray ``lock_ray`` onto each world direction in ``dirs``.

    Pitch first: it must bring the lock ray's height (y) to the target's, which yaw can't change.
    With A = |(ly, lz)| and alpha = atan2(lz, ly), ly*cos(p) + lz*sin(p) = A*cos(p - alpha) = dy.
    Of the two roots, ``alpha - acos`` is the one that doesn't flip the camera over a pole.
    Yaw then lines up the horizontal angles.
    """
    d = np.asarray(dirs, dtype=float)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    lx, ly, lz = np.asarray(lock_ray, dtype=float) / np.linalg.norm(lock_ray)
    A = np.hypot(ly, lz)
    pitch = np.arctan2(lz, ly) - np.arccos(np.clip(d[..., 1] / A, -1.0, 1.0))
    mz = -ly * np.sin(pitch) + lz * np.cos(pitch)
    yaw = np.arctan2(d[..., 0], d[..., 2]) - np.arctan2(lx, mz)
    return yaw, pitch


def equirect_maps(R: np.ndarray, rays: np.ndarray, src_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """``cv2.remap`` maps that sample an equirect frame of ``src_size`` along ``rays`` turned by R.

    Use with ``borderMode=cv2.BORDER_WRAP`` so samples straddling the +-180 deg seam wrap around.
    """
    world = rays @ np.asarray(R, dtype=np.float32).T
    lon = np.arctan2(world[..., 0], world[..., 2])
    lat = np.arcsin(np.clip(world[..., 1], -1.0, 1.0))
    W, H = src_size
    mx = lon * np.float32(W / (2 * np.pi)) + np.float32(W / 2 - 0.5)
    my = np.clip((0.5 - lat * np.float32(1 / np.pi)) * np.float32(H) - np.float32(0.5), 0, H - 1)
    return mx.astype(np.float32), my.astype(np.float32)
