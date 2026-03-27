"""ScanImage TIFF data reader with fallback to standard TIFF reading."""

from pathlib import Path
from typing import Union

import numpy as np


def scanimagetiff_data_wrapper(
    reader: object,
    tif_file: Union[str, Path],
) -> np.ndarray:
    """Read a ScanImage TIFF stack, falling back to standard TIFF reading.

    Attempts to read image data via *reader*. On failure, reads all pages of
    the TIFF using standard image I/O. Corresponds to
    ScanImageTiffDataWrapper.m in GIAnT-MATLAB.

    Parameters
    ----------
    reader : object
        ScanImage TIFF reader object (e.g. from the ``scanimage-tiff-reader``
        Python package).
    tif_file : str or Path
        Path to the ScanImage TIFF file used to construct *reader*.

    Returns
    -------
    ndarray
        Image stack in the native on-disk datatype, shaped
        ``(W, H, n_frames)`` to match ScanImage convention.
    """
    raise NotImplementedError
