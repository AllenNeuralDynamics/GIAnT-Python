"""Generic struct <-> HDF5 (de)serialization (port of load/saveStructFromH5).

This module is the foundation of MATLAB interoperability: the model
``from_h5`` / ``to_h5`` methods build on these helpers. Round-trip tests
against real MATLAB-produced ``.h5`` files are the acceptance criterion.

Serialization conventions (shared with GIAnT-MATLAB; see the package README)
------------------------------------------------------------------------
* Writers set ``/row_major`` to ``1`` (Python / h5py) or ``0`` (MATLAB).
* Dimension tuples in the README match h5py ``shape`` when ``row_major=1``.
* Vectors documented as ``1 x N`` or ``N x 1`` are written as rank-2
  datasets with those shapes (never squeezed to ``(N,)``), so downstream
  code can use the same size checks on either toolbox's outputs after
  consulting ``row_major``.
"""

from pathlib import Path
from typing import Union


def load_struct_h5(path: Union[str, Path]) -> dict:
    """Load a (possibly nested) MATLAB-style struct from an HDF5 file.

    Recursively reads HDF5 groups into nested dictionaries, applying the
    ``row_major`` / axis-permutation conventions required for MATLAB
    compatibility. Corresponds to loadStructFromH5.m in GIAnT-MATLAB.

    Parameters
    ----------
    path : str or Path
        Path to the HDF5 file.

    Returns
    -------
    dict
        Nested dictionary mirroring the HDF5 group hierarchy.
    """
    raise NotImplementedError


def save_struct_h5(struct: dict, path: Union[str, Path]) -> None:
    """Write a (possibly nested) dict to an HDF5 file as a MATLAB-style struct.

    Recursively writes nested dictionaries to HDF5 groups/datasets so that the
    result is readable by GIAnT-MATLAB. Corresponds to saveStructToH5.m in
    GIAnT-MATLAB.

    Sets ``/row_major = 1``. Rank-2 vector shapes (``1 x N`` / ``N x 1``) are
    preserved on disk; singleton dimensions are not squeezed.

    Parameters
    ----------
    struct : dict
        Nested dictionary to serialize.
    path : str or Path
        Destination path for the HDF5 file.
    """
    raise NotImplementedError
