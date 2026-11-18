"""MATLAB-style HDF5 decoding, struct writing and parameter group writing.

This module is the foundation of MATLAB interoperability: the model
``from_h5`` / ``to_h5`` methods build on these helpers.

Serialization conventions
-------------------------
Shared with GIAnT-MATLAB; see the package README.

* Writers set ``/row_major`` to ``1`` (Python / h5py) or ``0`` (MATLAB).
* Dimension tuples in the README match h5py ``shape`` when ``row_major=1``.
* Vectors documented as ``1 x N`` or ``N x 1`` are written as rank-2
  datasets with those shapes (never squeezed to ``(N,)``), so downstream
  code can use the same size checks on either toolbox's outputs after
  consulting ``row_major``.
"""

from pathlib import Path
from typing import Union, cast

import h5py
import numpy as np


def _is_string_dataset(ds: h5py.Dataset) -> bool:
    """Return whether an h5py dataset holds (fixed/variable-length) strings."""
    if h5py.check_string_dtype(ds.dtype) is not None:
        return True
    return ds.dtype.kind in ("S", "U", "O")


def _decode_one(x) -> str:
    """Decode a single bytes/str scalar to ``str``."""
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8", "replace")
    return str(x)


def _decode_strings(val):
    """Decode a scalar or ndarray of (variable-length) strings to ``str``.

    Parameters
    ----------
    val : bytes, str, or ndarray
        Raw value read from a string dataset.

    Returns
    -------
    str or ndarray of object
        Decoded scalar string, or an object array of decoded strings.
    """
    if isinstance(val, (bytes, str, np.bytes_, np.str_)):
        return _decode_one(val)
    arr = np.asarray(val, dtype=object)
    if arr.size == 0:
        return arr
    return np.vectorize(_decode_one, otypes=[object])(arr)


def _read_dataset(ds: h5py.Dataset, row_major: bool):
    """Read one dataset in MATLAB ``size()`` orientation.

    Scalar strings/paths (e.g. ``datadr``) are returned as plain ``str``;
    string grids (e.g. ``filename``, ``fn_adata``) stay as object arrays with
    MATLAB shape. Column-major files have their 2-D+ axes transposed to recover
    the README orientation.

    Parameters
    ----------
    ds : h5py.Dataset
        The dataset to read.
    row_major : bool
        Whether the file was written row-major (``True``) or column-major.

    Returns
    -------
    object
        The decoded value (``str``, scalar, or ndarray).
    """
    val = ds[()]
    if _is_string_dataset(ds):
        val = _decode_strings(val)
        if isinstance(val, np.ndarray):
            if (not row_major) and val.ndim >= 2:
                val = val.T
            if val.size == 1:
                return val.reshape(-1)[0]
        return val
    val = np.asarray(val)
    if (not row_major) and val.ndim >= 2:
        val = val.T
    return val


def _read_group(grp: h5py.Group, row_major: bool) -> dict:
    """Read an h5py group into a nested dict (skipping ``row_major``)."""
    out = {}
    for key, item in grp.items():
        if key == "row_major":
            continue
        if isinstance(item, h5py.Group):
            out[key] = _read_group(item, row_major)
        else:
            out[key] = _read_dataset(item, row_major)
    return out


def load_struct_h5(path: Union[str, Path]) -> dict:
    """Load a (possibly nested) MATLAB-style struct from an HDF5 file.

    Recursively reads HDF5 groups into nested dictionaries, applying the
    ``row_major`` / axis-permutation conventions required for MATLAB
    compatibility. Corresponds to loadStructFromH5.m in GIAnT-MATLAB.

    Groups become nested dicts, string datasets become ``str`` (scalars) or
    object arrays (grids), and numeric datasets are returned in MATLAB
    ``size()`` orientation. The ``/row_major`` flag (absent => column-major)
    selects whether 2-D+ axes are transposed.

    Parameters
    ----------
    path : str or Path
        Path to the HDF5 file.

    Returns
    -------
    dict
        Nested dictionary mirroring the HDF5 group hierarchy.
    """
    with h5py.File(path, "r") as f:
        row_major = False
        if "row_major" in f:
            flag = cast(h5py.Dataset, f["row_major"])
            row_major = bool(np.asarray(flag[()]).reshape(-1)[0])
        return _read_group(f, row_major)


def to_serializable(val):
    """Convert nested NumPy values to ordinary Python parameter values."""
    if isinstance(val, dict):
        return {k: to_serializable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return type(val)(to_serializable(v) for v in val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, np.generic):
        return val.item()
    return val


def _write_preserved_array(grp: h5py.Group, key: str, value) -> None:
    """Write a NumPy value with its original shape and numeric dtype."""
    arr = np.asarray(value)
    if arr.dtype.kind in ("U", "S", "O"):
        grp.create_dataset(
            key,
            data=np.asarray(_decode_strings(arr), dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
    else:
        grp.create_dataset(key, data=arr)


def _write_sequence(grp: h5py.Group, key: str, value) -> None:
    """Skip empty sequences and preserve persisted element stringification."""
    arr = np.asarray(value)
    if arr.size == 0:
        return
    if arr.dtype.kind in ("U", "S", "O"):
        grp.create_dataset(
            key,
            data=np.array([str(x) for x in value], dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
    else:
        grp.create_dataset(key, data=arr)


def _matlab_array(value) -> np.ndarray:
    """A value as an array in MATLAB ``size()`` orientation (rank >= 2).

    Python scalars become ``1 x 1`` (``int`` and ``float`` as double, the
    MATLAB default; ``bool`` as int8, as GIAnT-MATLAB writes logicals); 1-D
    arrays become ``1 x N`` rows.
    """
    if isinstance(value, (bool, np.bool_)):
        return np.array([[np.int8(value)]])
    arr = np.asarray(value)
    if arr.dtype == np.bool_:
        arr = arr.astype(np.int8)
    if isinstance(value, (int, float)):
        arr = arr.astype(np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def write_matlab_value(grp: h5py.Group, key: str, value) -> None:
    """Write one value as GIAnT-MATLAB ``saveStructToH5`` would.

    Dictionaries become groups; strings and string arrays become variable
    length UTF-8 strings; numeric and boolean arrays are given in MATLAB
    orientation and stored with reversed dimensions (column-major), so
    MATLAB's ``h5read`` returns the original ``size()``. Empty values are
    skipped, as MATLAB skips empty fields.

    Parameters
    ----------
    grp : h5py.Group
        Destination group.
    key : str
        Dataset or group name.
    value : dict, str, bool, int, float or array_like
        The value. Use NumPy dtypes to choose the stored class (e.g.
        ``np.float32`` for MATLAB single).
    """
    if isinstance(value, dict):
        sub = grp.create_group(key)
        for k, v in value.items():
            write_matlab_value(sub, str(k), v)
        return
    str_dt = h5py.string_dtype(encoding="utf-8")
    if isinstance(value, str):
        grp.create_dataset(
            key, data=np.array([[value]], dtype=object), dtype=str_dt
        )
        return
    arr = _matlab_array(value)
    if arr.size == 0:
        return
    if arr.dtype.kind in ("U", "S", "O"):
        text = np.vectorize(_decode_one, otypes=[object])(arr)
        grp.create_dataset(
            key, data=np.ascontiguousarray(text.T), dtype=str_dt
        )
        return
    grp.create_dataset(key, data=np.ascontiguousarray(arr.T))


def save_struct_h5(path: Union[str, Path], struct: dict) -> None:
    """Write a nested dict as a GIAnT-MATLAB struct file (column-major).

    Counterpart of ``saveStructToH5.m``: the file is created anew and gets
    ``/row_major = 0``, so both :func:`load_struct_h5` and MATLAB's
    ``loadStructFromH5`` read the values back in MATLAB orientation.

    Parameters
    ----------
    path : str or Path
        Destination file (overwritten).
    struct : dict
        Values as accepted by :func:`write_matlab_value`.
    """
    with h5py.File(path, "w") as f:
        for key, value in struct.items():
            write_matlab_value(f, str(key), value)
        f.create_dataset("row_major", data=np.zeros((1, 1), np.uint8))


def write_dict_to_h5group(grp: h5py.Group, d: dict) -> None:
    """Write params, skipping empty sequences/None and stringifying unknowns.

    This is deliberately NOT a general lossless struct serializer. Its policy
    preserves the existing experiment-summary params contract. Decoded arrays
    retain their dtype/rank (including uint8 boolean parameter scalars).
    """
    str_dt = h5py.string_dtype(encoding="utf-8")
    for key, v in d.items():
        key = str(key)
        if isinstance(v, dict):
            write_dict_to_h5group(grp.create_group(key), v)
        elif isinstance(v, (np.ndarray, np.generic)):
            _write_preserved_array(grp, key, v)
        elif isinstance(v, bool):
            grp.create_dataset(key, data=np.uint8(v))
        elif isinstance(v, str):
            grp.create_dataset(key, data=v, dtype=str_dt)
        elif isinstance(v, (int, float)):
            grp.create_dataset(key, data=v)
        elif isinstance(v, (list, tuple)):
            _write_sequence(grp, key, v)
        elif v is not None:
            grp.create_dataset(key, data=str(v), dtype=str_dt)
