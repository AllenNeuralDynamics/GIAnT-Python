"""Input/output for GIAnT-Python.

This package is the compatibility contract with GIAnT-MATLAB: it reads and
writes the shared HDF5 structures, ScanImage TIFF stacks, and SLAP2 binary
data files. Pay special attention to 0- vs 1-based indices and array
transposition when porting from MATLAB.
"""

from .hdf5 import load_struct_h5, save_struct_h5
from .slap2 import (
    get_online_motion,
    read_integration_trial_data,
    ref_pixs_to_drc,
)
from .tiff import scanimagetiff_data_wrapper, scanimagetiff_wrapper

__all__ = [
    "get_online_motion",
    "load_struct_h5",
    "read_integration_trial_data",
    "ref_pixs_to_drc",
    "save_struct_h5",
    "scanimagetiff_data_wrapper",
    "scanimagetiff_wrapper",
]
