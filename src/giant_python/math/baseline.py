"""Fluorescence baseline (F0) estimation.

Shared by both source-extraction backends. Ports ``computeF0.m`` (GIAnT-MATLAB)
/ ``compute_f0`` from ``extractSLAP2IntegrationSources.py``.
"""

import numpy as np


def compute_f0(
    f_in: np.ndarray,
    denoise_window: int,
    hull_window: int,
) -> np.ndarray:
    """Estimate the slowly varying baseline F0 of fluorescence traces.

    Median-filters to reduce noise, applies a rolling convex-hull-like min
    envelope on decimated grids, discards doubtful samples near NaNs, then
    smooths/fills and PCHIP-interpolates back to the full time grid.

    Parameters
    ----------
    f_in : ndarray of shape (T, ...)
        Fluorescence with time along axis 0; NaNs allowed.
    denoise_window : int
        Median-filter window, in time samples.
    hull_window : int
        Window controlling the convex-hull-like min envelope.

    Returns
    -------
    ndarray
        Baseline F0, same shape as *f_in*.
    """
    raise NotImplementedError
