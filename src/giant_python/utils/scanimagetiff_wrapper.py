"""Convenience wrapper for reading ScanImage TIFF files."""

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
