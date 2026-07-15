"""FFT-based and NaN-aware cross-correlation registration kernels.

Ports of dftregistration_clipped.m, xcorr2_nans.m, xcorr2_nans3d.m, and
xcorr2_nans_weighted.m from GIAnT-MATLAB.
"""

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
    window of +/-*d_shift* pixels centred on *shifts_center*. Corresponds to
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


def xcorr2_nans(
    frame: np.ndarray,
    template: np.ndarray,
) -> np.ndarray:
    """Compute NaN-aware 2-D normalised cross-correlation.

    Port of xcorr2_nans.m in GIAnT-MATLAB.

    Parameters
    ----------
    frame : ndarray of shape (H, W)
        Image to align; may contain NaNs.
    template : ndarray of shape (H, W)
        Fixed reference image.

    Returns
    -------
    ndarray
        Cross-correlation surface.
    """
    raise NotImplementedError


def xcorr2_nans3d(
    frame: np.ndarray,
    template: np.ndarray,
) -> np.ndarray:
    """Compute NaN-aware 3-D normalised cross-correlation.

    Port of xcorr2_nans3d.m in GIAnT-MATLAB.

    Parameters
    ----------
    frame : ndarray of shape (Z, H, W)
        Volume to align; may contain NaNs.
    template : ndarray of shape (Z, H, W)
        Fixed reference volume.

    Returns
    -------
    ndarray
        Cross-correlation volume.
    """
    raise NotImplementedError


def dft_register_clipped(
    buf_fft: np.ndarray,
    target_fft: np.ndarray,
    usfac: int = 1,
    max_shift: float = np.inf,
) -> np.ndarray:
    """Estimate sub-pixel translation by upsampled DFT cross-correlation.

    Port of dftregistration_clipped.m in GIAnT-MATLAB: efficient sub-pixel
    image registration via upsampled DFT, with the shift clipped to
    ``+/-max_shift``.

    Parameters
    ----------
    buf_fft : ndarray
        FFT of the image to register.
    target_fft : ndarray
        FFT of the reference image.
    usfac : int
        Upsampling factor (sub-pixel resolution = ``1 / usfac``).
    max_shift : float
        Maximum allowed shift magnitude, in pixels.

    Returns
    -------
    ndarray
        Estimated ``[row, col]`` shift.
    """
    raise NotImplementedError
