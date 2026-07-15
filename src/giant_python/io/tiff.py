"""ScanImage TIFF reading (port of ScanImageTiffWrapper / ...DataWrapper).

Also the intended home for the fast TIFF writers (ports of Fast_Tiff_Write /
Fast_BigTiff_Write) used to emit registered movies.
"""

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np


def scanimagetiff_wrapper(
    fn: Union[str, Path],
) -> Tuple[np.ndarray, List[str]]:
    """Read image data and per-frame metadata from a ScanImage TIFF file.

    Opens the file with a ScanImage TIFF reader, extracts the image stack
    via :func:`scanimagetiff_data_wrapper`, and returns the per-frame
    metadata strings. Corresponds to ScanImageTiffWrapper.m in GIAnT-MATLAB.

    Parameters
    ----------
    fn : str or Path
        Path to the ScanImage TIFF file.

    Returns
    -------
    data : ndarray
        Image stack read from the file.
    meta : str
        File-level metadata string embedded in the TIFF.
    """
    raise NotImplementedError


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
