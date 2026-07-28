"""HDF5 struct (de)serialization for the BandSILo backend.

BandRegistration.m (GIAnT-MATLAB) writes its outputs as HDF5 via
``saveStructToH5.m`` rather than the old ``.mat`` files. The readers below
mirror ``loadStructFromH5.m``: groups become nested dicts, string datasets
become ``str`` (or object arrays), and numeric datasets are returned in MATLAB
``size()`` orientation. MATLAB writes column-major (the ``/row_major`` flag is
``0`` or absent), so h5py reads datasets with axes reversed and we transpose to
recover the original layout.

Ported from ``extractSLAP2IntegrationSources.py`` (``load_struct_from_h5``,
``load_alignment_data_h5``, ``_write_dict_to_h5group``, ``to_serializable``).
This module is intentionally private to :mod:`giant_python.bandsilo`; the
generic ``giant_python.io.hdf5`` helpers remain a separate concern.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import h5py
import numpy as np


def to_serializable(val):
    """Recursively convert numpy types/arrays to JSON-friendly Python values.

    Parameters
    ----------
    val : object
        A scalar, numpy value/array, or (possibly nested) dict/list/tuple.

    Returns
    -------
    object
        The same structure with numpy scalars converted to Python scalars and
        numpy arrays converted to (nested) lists.
    """
    if isinstance(val, dict):
        return {k: to_serializable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return type(val)(to_serializable(v) for v in val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, np.generic):
        return val.item()
    return val


def _h5_is_string_dataset(ds: h5py.Dataset) -> bool:
    """Return whether an h5py dataset holds (fixed/variable-length) strings."""
    if h5py.check_string_dtype(ds.dtype) is not None:
        return True
    return ds.dtype.kind in ("S", "U", "O")


def _decode_one(x) -> str:
    """Decode a single bytes/str scalar to ``str``."""
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8", "replace")
    return str(x)


def _decode_h5_strings(val):
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


def _read_h5_dataset(ds: h5py.Dataset, row_major: bool):
    """Read one dataset in MATLAB ``size()`` orientation.

    Scalar strings/paths (e.g. ``datadr``) are returned as plain ``str``;
    string grids (e.g. ``filename``, ``fn_adata``) stay as object arrays with
    MATLAB shape.

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
    if _h5_is_string_dataset(ds):
        val = _decode_h5_strings(val)
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


def _read_h5_group(grp: h5py.Group, row_major: bool) -> dict:
    """Recursively read an h5py group into a nested dict."""
    out = {}
    for key, item in grp.items():
        if key == "row_major":
            continue
        if isinstance(item, h5py.Group):
            out[key] = _read_h5_group(item, row_major)
        else:
            out[key] = _read_h5_dataset(item, row_major)
    return out


def load_struct_from_h5(path: Union[str, Path]) -> dict:
    """Load a (possibly nested) MATLAB struct from HDF5 into a nested dict.

    Python port of GIAnT-MATLAB ``loadStructFromH5.m``. Honors the
    ``/row_major`` layout flag (absent => column-major, so 2-D+ axes are
    transposed to recover MATLAB/README orientation).

    Parameters
    ----------
    path : str or Path
        Path to the HDF5 file.

    Returns
    -------
    dict
        The nested struct as a dict of dicts / ndarrays / scalars.
    """
    with h5py.File(path, "r") as f:
        row_major = False
        if "row_major" in f:
            row_major = bool(np.asarray(f["row_major"][()]).reshape(-1)[0])
        return _read_h5_group(f, row_major)


def write_dict_to_h5group(grp: h5py.Group, d: dict) -> None:
    """Write a (JSON-serializable) dict into an h5 group key-by-key.

    Nested dicts become subgroups; scalars/arrays become datasets; bools are
    stored as uint8; empty/None values are skipped; anything unexpected is
    stringified.

    Parameters
    ----------
    grp : h5py.Group
        Destination group.
    d : dict
        The dict to serialize.
    """
    str_dt = h5py.string_dtype(encoding="utf-8")
    for key, v in d.items():
        key = str(key)
        if isinstance(v, dict):
            write_dict_to_h5group(grp.create_group(key), v)
        elif isinstance(v, bool):
            grp.create_dataset(key, data=np.uint8(v))
        elif isinstance(v, str):
            grp.create_dataset(key, data=v, dtype=str_dt)
        elif isinstance(v, (int, float)):
            grp.create_dataset(key, data=v)
        elif isinstance(v, (list, tuple)):
            arr = np.asarray(v)
            if arr.size == 0:
                continue
            if arr.dtype.kind in ("U", "S", "O"):
                grp.create_dataset(
                    key,
                    data=np.array([str(x) for x in v], dtype=object),
                    dtype=str_dt,
                )
            else:
                grp.create_dataset(key, data=arr)
        elif v is None:
            continue
        else:
            grp.create_dataset(key, data=str(v), dtype=str_dt)


def _reshape_1d(src: dict, key: str):
    """Return ``src[key]`` flattened to 1-D, or ``None`` if missing."""
    v = src.get(key)
    return None if v is None else np.asarray(v).reshape(-1)


def load_alignment_data_h5(path: Union[str, Path]) -> dict:
    """Load a ``*_ALIGNMENTDATA.h5`` file (aData) into flat arrays/scalars.

    Mirrors the aData struct written by BandRegistration.m. Motion estimates
    are stored as column vectors and online-motion shifts live under the nested
    ``slap2`` group in the new format; everything is flattened to 1-D here.

    Parameters
    ----------
    path : str or Path
        Path to the ``*_ALIGNMENTDATA.h5`` file.

    Returns
    -------
    dict
        Keys ``DSframes``, ``motionDSr/c/z``, ``onlineYshift/Xshift/Zshift``,
        ``numChannels``, ``alignHz``.
    """
    s = load_struct_from_h5(path)
    slap2 = s.get("slap2", {}) or {}
    return {
        "DSframes": _reshape_1d(s, "DSframes"),
        "motionDSr": _reshape_1d(s, "motionDSr"),
        "motionDSc": _reshape_1d(s, "motionDSc"),
        "motionDSz": _reshape_1d(s, "motionDSz"),
        "onlineYshift": _reshape_1d(slap2, "onlineMotionYshift"),
        "onlineXshift": _reshape_1d(slap2, "onlineMotionXshift"),
        "onlineZshift": _reshape_1d(slap2, "onlineMotionZshift"),
        "numChannels": int(np.asarray(s["numChannels"]).reshape(-1)[0]),
        "alignHz": float(np.asarray(s["alignHz"]).reshape(-1)[0]),
    }


def _resolve_fn_adata(fn_adata: np.ndarray, moco_save_dr: str) -> np.ndarray:
    """Resolve per-trial aData basenames to absolute paths under moco dir.

    Parameters
    ----------
    fn_adata : ndarray of object, shape (n_dmds, n_trials)
        Alignment-data basenames (may contain ``''``/``None``).
    moco_save_dr : str
        ``<savedr>/motion_correction`` directory.

    Returns
    -------
    ndarray of object, shape (n_dmds, n_trials)
        Absolute paths (``''`` where the entry was empty).
    """
    n_dmds, n_trials = fn_adata.shape
    out = np.empty((n_dmds, n_trials), dtype=object)
    for d in range(n_dmds):
        for t in range(n_trials):
            entry = fn_adata[d, t]
            if entry in (None, ""):
                out[d, t] = ""
            else:
                out[d, t] = os.path.join(moco_save_dr, str(entry))
    return out


def load_trial_table(path: Union[str, Path]) -> dict:
    """Load and normalize a GIAnT ``trial_table.h5`` for BandSILo.

    Reads the trial table written by BandRegistration.m and assembles the
    normalized structure consumed by the band backend (mirrors the
    setup block of ``main`` in ``extractSLAP2IntegrationSources.py``).

    Parameters
    ----------
    path : str or Path
        Path to the ``trial_table.h5`` file.

    Returns
    -------
    dict
        Keys: ``datadr``, ``result_dr``, ``moco_save_dr``,
        ``annotation_save_dr``, ``src_extr_save_dr``, ``align_params``,
        ``n_dmds``, ``n_trials``, and ``trial_table`` (a dict with
        ``filename``, ``first_line``, ``last_line``, ``fn_adata`` (absolute
        paths), ``datadr``, ``savedr``, ``ref_stack``).
    """
    tt = load_struct_from_h5(path)

    datadr = str(tt["datadr"])
    result_dr = str(tt["savedr"])
    moco_save_dr = os.path.join(result_dr, "motion_correction")

    motion_correction = tt.get("motion_correction", {}) or {}
    slap2_info = tt.get("slap2_info", {}) or {}

    align_params = {
        key: to_serializable(val)
        for key, val in (
            motion_correction.get("align_params", {}) or {}
        ).items()
    }

    filenames = np.atleast_2d(np.asarray(tt["filename"], dtype=object))
    first_line = np.atleast_2d(np.asarray(slap2_info["first_line"]))
    last_line = np.atleast_2d(np.asarray(slap2_info["last_line"]))
    fn_adata = np.atleast_2d(
        np.asarray(motion_correction["fn_adata"], dtype=object)
    )

    n_dmds, n_trials = filenames.shape
    fn_adata_full = _resolve_fn_adata(fn_adata, moco_save_dr)

    trial_table = {
        "filename": filenames,
        "first_line": first_line,
        "last_line": last_line,
        "fn_adata": fn_adata_full,
        "datadr": datadr,
        "savedr": result_dr,
        "ref_stack": slap2_info.get("ref_stack", {}) or {},
    }

    return {
        "datadr": datadr,
        "result_dr": result_dr,
        "moco_save_dr": moco_save_dr,
        "annotation_save_dr": os.path.join(result_dr, "annotations"),
        "src_extr_save_dr": os.path.join(result_dr, "source_extraction"),
        "align_params": align_params,
        "n_dmds": int(n_dmds),
        "n_trials": int(n_trials),
        "trial_table": trial_table,
    }
