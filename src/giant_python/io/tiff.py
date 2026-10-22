"""Native TIFF array reading without acquisition-specific transformations."""

from pathlib import Path
from typing import Union

import numpy as np


def read_tiff(path: Union[str, Path]) -> np.ndarray:
    """Read a TIFF eagerly, preserving native dtype, rank and page ordering.

    Reference intensity scaling, channel reshaping and PSF cropping are
    caller policies, not generic TIFF decoding. Import tifffile only on use.
    """
    import tifffile

    return tifffile.imread(path)
