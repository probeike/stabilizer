"""Gap filling and zero-phase smoothing of a 2D track."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, savgol_filter, sosfiltfilt

METHODS = ("savgol", "butter", "none")


def find_gaps(valid: np.ndarray) -> list[tuple[int, int]]:
    """Runs of invalid frames as half-open (start, end) index ranges."""
    gaps, start = [], None
    for i, ok in enumerate(valid):
        if not ok and start is None:
            start = i
        elif ok and start is not None:
            gaps.append((start, i))
            start = None
    if start is not None:
        gaps.append((start, len(valid)))
    return gaps


def fill_gaps(track: np.ndarray) -> np.ndarray:
    """Replace NaN rows with PCHIP interpolation; hold the nearest valid value past the ends."""
    track = np.asarray(track, dtype=float)
    valid = ~np.isnan(track).any(axis=1)
    if valid.all():
        return track.copy()
    if valid.sum() == 0:
        raise ValueError("No valid head positions in the whole clip.")
    t = np.arange(len(track))
    tv = t[valid]
    out = track.copy()
    if len(tv) == 1:
        out[:] = track[valid][0]
        return out
    interp = PchipInterpolator(tv, track[valid], axis=0, extrapolate=False)
    out[~valid] = interp(t[~valid])
    out[t < tv[0]] = track[tv[0]]
    out[t > tv[-1]] = track[tv[-1]]
    return out


def savgol_window(fps: float, cutoff_hz: float, polyorder: int = 3) -> int:
    """Odd window length whose -3 dB cutoff is ~cutoff_hz (Schafer 2011 approximation)."""
    n = ((polyorder + 1) * fps / cutoff_hz + 4.6) / 3.2
    n = max(int(round(n)), polyorder + 2)
    return n if n % 2 else n + 1


def smooth(track: np.ndarray, fps: float, method: str = "savgol", cutoff_hz: float = 6.0) -> np.ndarray:
    """Zero-phase smoothing along time (axis 0). Input must have no NaNs."""
    track = np.asarray(track, dtype=float)
    if method == "none" or len(track) < 5:
        return track.copy()
    if method == "savgol":
        window = min(savgol_window(fps, cutoff_hz), len(track) - (1 - len(track) % 2))
        if window <= 3:
            return track.copy()
        return savgol_filter(track, window, 3, axis=0, mode="interp")
    if method == "butter":
        if cutoff_hz >= fps / 2:
            return track.copy()
        sos = butter(2, cutoff_hz, fs=fps, output="sos")
        padlen = min(3 * (2 * len(sos) + 1), len(track) - 1)
        return sosfiltfilt(sos, track, axis=0, padlen=padlen)
    raise ValueError(f"Unknown smoothing method {method!r}; choose from {METHODS}")
