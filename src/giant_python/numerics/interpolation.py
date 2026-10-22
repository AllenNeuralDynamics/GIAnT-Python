"""Shared nearest-neighbor interpolation."""

import numpy as np


def nearest_interp(
    x: np.ndarray, xp: np.ndarray, yp: np.ndarray
) -> np.ndarray:
    """Nearest-neighbor interpolation (MATLAB ``interp1(...,'nearest')``).

    Parameters
    ----------
    x : ndarray
        Query points.
    xp : ndarray
        Sample positions (assumed sorted ascending).
    yp : ndarray
        Sample values aligned with ``xp``.

    Returns
    -------
    ndarray
        ``yp`` sampled at the nearest ``xp`` for each ``x``.
    """
    if len(xp) == 1:
        return yp
    x_bds = xp[:-1] / 2.0 + xp[1:] / 2.0
    idx = np.searchsorted(x_bds, x, side="left")
    idx = np.clip(idx, 0, len(xp) - 1)
    return yp[idx]
