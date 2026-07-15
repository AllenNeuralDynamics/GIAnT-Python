"""Activity-image peak detection via integrated-Gaussian fitting.

Shared by both source-extraction backends (standard and integration). Ports
``getActImPeaks.m`` (GIAnT-MATLAB) / ``get_act_im_peaks`` and its
integrated-Gaussian helpers from ``extractSLAP2IntegrationSources.py``.
"""

from typing import Optional, Tuple

import numpy as np


def gaussian_peaks_integrated(
    theta: np.ndarray,
    yx: np.ndarray,
) -> np.ndarray:
    """Evaluate integrated isotropic 2-D Gaussians over unit pixels.

    Port of ``gaussianPeaksIntegrated`` from getActImPeaks.m.

    Parameters
    ----------
    theta : ndarray of shape (N, 4)
        Per-peak parameters ``[amp, mu_y, mu_x, sigma]``.
    yx : ndarray of shape (M, 2)
        Pixel-centre coordinates ``[y, x]``.

    Returns
    -------
    ndarray of shape (M,)
        Predicted values at the given pixel centres.
    """
    raise NotImplementedError


def detect_peaks_2d(
    act_im_2d: np.ndarray,
    exclusion_mask: np.ndarray,
    mu_bg: float,
    sigma_bg: float,
    peak_thresh: float,
    peak_th: float,
) -> np.ndarray:
    """Detect Gaussian peaks in a single activity-image plane.

    Performs initial local-maximum detection followed by iterative
    residual peak finding with bounded Levenberg-Marquardt curve fitting.

    Parameters
    ----------
    act_im_2d : ndarray of shape (H, W)
        One plane of the activity image; may contain NaNs.
    exclusion_mask : ndarray of shape (H, W)
        Pixels to exclude from detection.
    mu_bg, sigma_bg : float
        Background mean and (MAD-normalised) standard deviation.
    peak_thresh : float
        Absolute amplitude threshold (``mu_bg + peak_th * sigma_bg``).
    peak_th : float
        Threshold in MAD-normalised standard deviations.

    Returns
    -------
    ndarray of shape (N, 4)
        Fitted peaks ``[amp, mu_y, mu_x, sigma]`` (``(0, 4)`` if none).
    """
    raise NotImplementedError


def get_act_im_peaks(
    act_im: np.ndarray,
    peak_th: float = 3.0,
    exclusion_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Find Gaussian peaks across a 3-D ``(Z, H, W)`` activity image.

    Background statistics and the detection threshold are computed once
    across all planes for uniform sensitivity, then each plane is processed
    independently via :func:`detect_peaks_2d`.

    Parameters
    ----------
    act_im : ndarray of shape (Z, H, W)
        Activity image; may contain NaNs.
    peak_th : float
        Threshold in MAD-normalised standard deviations.
    exclusion_mask : ndarray, optional
        Mask broadcastable to ``(Z, H, W)`` of pixels to exclude.

    Returns
    -------
    ndarray of shape (N, 3)
        Source seeds ``[z, mu_y, mu_x]`` (``(0, 3)`` if none detected).
    """
    raise NotImplementedError
