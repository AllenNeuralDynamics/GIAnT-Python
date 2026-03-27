"""Weighted 2-D cross-correlation with NaN support."""

from typing import Tuple

import numpy as np


def xcorr2_nans_weighted(
    frame: np.ndarray,
    freshness: np.ndarray,
    template: np.ndarray,
    shifts_center: np.ndarray,
    d_shift: float,
) -> Tuple[np.ndarray, float]:
    """Perform weighted, NaN-aware local normalised cross-correlation.

    Estimates the sub-pixel translation between *frame* and *template* by
    evaluating the weighted normalised cross-correlation over a local search
    window of ±*d_shift* pixels centred on *shifts_center*. Corresponds to
    xcorr2_nans_weighted.m in GIAnT-MATLAB.

    Parameters
    ----------
    frame : ndarray of shape (H, W)
        Floating image to align; may contain NaNs.
    freshness : ndarray of shape (H, W)
        Per-pixel effective sample count (photon-count weighting) for
        *frame*.
    template : ndarray of shape (H, W)
        Fixed reference image.
    shifts_center : ndarray of shape (2,)
        ``[row, col]`` centre of the local search window.
    d_shift : float
        Half-width of the search window in pixels (rounded to integer).

    Returns
    -------
    motion : ndarray of shape (2,)
        Estimated ``[row, col]`` shift of *frame* relative to *template*.
    r : float
        Peak normalised cross-correlation coefficient.
    """
    raise NotImplementedError
