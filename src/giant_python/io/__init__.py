"""Input/output for GIAnT-Python.

This package is the compatibility contract with GIAnT-MATLAB: it reads and
writes the shared HDF5 structures, ScanImage TIFF stacks, and SLAP2 binary
data files. Pay special attention to 0- vs 1-based indices and array
transposition when porting from MATLAB.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from .hdf5 import load_struct_h5
from .slap2 import (
    Slap2Reader,
    load_alignment_data_h5,
    open_slap2_file,
)
from .tiff import read_tiff

if TYPE_CHECKING:
    from .annotations import read_annotations_h5, save_annotations_h5
    from .experiment_summary import (
        read_summary,
        write_summary,
    )
    from .trial_table import read_trial_table

_SCHEMA_EXPORTS = {
    "read_trial_table": ".trial_table",
    "read_annotations_h5": ".annotations",
    "save_annotations_h5": ".annotations",
    "read_summary": ".experiment_summary",
    "write_summary": ".experiment_summary",
}


def __getattr__(name):
    """Load model codecs lazily, avoiding schema/model import cycles."""
    if name not in _SCHEMA_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_SCHEMA_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "Slap2Reader",
    "load_alignment_data_h5",
    "load_struct_h5",
    "open_slap2_file",
    "read_annotations_h5",
    "read_summary",
    "read_tiff",
    "read_trial_table",
    "save_annotations_h5",
    "write_summary",
]
