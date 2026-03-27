"""Frame interpolation with photon-count weighting."""

from typing import Tuple

import numpy as np


def interp_frame(
    m1: np.ndarray,
    view_c: np.ndarray,
    view_r: np.ndarray,
    freshness: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate an image and compute the per-pixel variance multiplier.

    Performs bilinear interpolation of a 2-D image at arbitrary coordinates,
    weighting each contributing pixel by its *freshness* (effective photon
    count). Also returns a variance multiplier proportional to the local
    measurement noise. Corresponds to interpFrame.m in GIAnT-MATLAB.

    Parameters
    ----------
    m1 : ndarray of shape (H, W)
        Source image to interpolate.
    view_c : ndarray
        Column coordinates at which to evaluate the interpolation.
    view_r : ndarray
        Row coordinates at which to evaluate the interpolation.
    freshness : ndarray of shape (H, W)
        Effective sample count for each pixel of *m1*.

    Returns
    -------
    im : ndarray
        Interpolated image evaluated at (*view_r*, *view_c*).
    v_im : ndarray
        Variance multiplier; multiply a pixel intensity by this value to
        obtain a quantity proportional to the pixel's variance.
    """
    raise NotImplementedError
