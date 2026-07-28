"""Numerical kernels for GIAnT-Python.

Pure, side-effect-free functions ported closely from GIAnT-MATLAB so they can
be validated against MATLAB outputs function-by-function. Watch MATLAB<->NumPy
axis-order and 0- vs 1-based indexing differences when porting.
"""

from .baseline import compute_f0
from .interpolation import interp_frame
from .peaks import detect_peaks_2d, gaussian_peaks_integrated, get_act_im_peaks
from .registration import (
    dft_register_clipped,
    xcorr2_nans,
    xcorr2_nans3d,
    xcorr2_nans_weighted,
)

__all__ = [
    "compute_f0",
    "detect_peaks_2d",
    "dft_register_clipped",
    "gaussian_peaks_integrated",
    "get_act_im_peaks",
    "interp_frame",
    "xcorr2_nans",
    "xcorr2_nans3d",
    "xcorr2_nans_weighted",
]
