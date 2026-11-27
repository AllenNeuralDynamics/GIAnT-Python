"""The statistics file of a registered trial (``<stem>_REGSTATS-80Hz.h5``).

A sidecar next to the registered movie, so that no reader of the movie or
the alignment file changes. Arrays are stored in GIAnT's MATLAB layout, as
the movie is: a map that is ``(channels, rows, cols)`` in Python is the
HDF5 dataset ``(channels, cols, rows)``, which MATLAB's ``h5read`` returns
as ``rows x cols x channels``. :func:`read_stats` returns Python's
orientation.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Union

import h5py
import numpy as np

STATS_VERSION = 1
"""Version of the statistics' definitions and file layout."""

LAYOUT = (
    "maps/<name>: HDF5 (channels, cols, rows), MATLAB h5read rows x cols x "
    "channels; svd/weight*: (channels, components, cols, rows); "
    "svd/time_courses: (channels, components, time bins); gain/*: "
    "(channels, brightness groups)"
)
"""Description stored with every file."""

_REGISTERED = "_REGISTERED_DOWNSAMPLED-"


def stats_path(movie_path: Union[str, Path]) -> Path:
    """The statistics file of a registered movie.

    ``<stem>_REGISTERED_DOWNSAMPLED-80Hz.h5`` (or ``.tif``) gives
    ``<stem>_REGSTATS-80Hz.h5`` in the same folder; any other name
    ``<name>_REGSTATS.h5``.
    """
    p = Path(movie_path)
    if _REGISTERED in p.name:
        stem, rest = p.name.split(_REGISTERED, 1)
        rate = rest.rsplit(".", 1)[0]
        return p.with_name(f"{stem}_REGSTATS-{rate}.h5")
    return p.with_name(f"{p.stem}_REGSTATS.h5")


def alignment_digest(ds_frames, motion_r, motion_c) -> str:
    """SHA-256 of the frame times and motion (float64), which identify the
    registration the statistics were made from."""
    h = hashlib.sha256()
    for a in (ds_frames, motion_r, motion_c):
        h.update(np.ascontiguousarray(a, dtype=np.float64).ravel().tobytes())
    return h.hexdigest()


def _to_file(a: np.ndarray) -> np.ndarray:
    """Python ``(..., rows, cols)`` to the file's ``(..., cols, rows)``."""
    return np.ascontiguousarray(np.swapaxes(a, -1, -2))


def _dataset(grp: h5py.Group, name: str, data: np.ndarray) -> None:
    """Write one array (deflated if it is large)."""
    data = np.asarray(data)
    opts = {}
    if data.size >= 4096:
        opts = dict(compression="gzip", compression_opts=1, shuffle=True)
    grp.create_dataset(name, data=data, **opts)


def write_stats(path: Union[str, Path], result: dict, attrs: dict) -> None:
    """Write a trial's statistics.

    Parameters
    ----------
    path : str or Path
        Destination (overwritten).
    result : dict
        :func:`.maps.finish_maps`'s output.
    attrs : dict
        File attributes (provenance); JSON-serializable values.
    """
    with h5py.File(path, "w") as f:
        f.attrs["stats_version"] = STATS_VERSION
        f.attrs["layout"] = LAYOUT
        f.attrs["created"] = datetime.now().astimezone().isoformat()
        for k, v in {**result["attrs"], **attrs}.items():
            f.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
        g = f.create_group("maps")
        for k, v in result["maps"].items():
            _dataset(g, k, _to_file(np.asarray(v, np.float32)))
        g = f.create_group("svd")
        for k, v in result["svd"].items():
            v = np.asarray(v)
            if k.startswith("weight"):
                v = _to_file(v.astype(np.float32))
            _dataset(g, k, v)
        g = f.create_group("gain")
        for k, v in result["gain"].items():
            _dataset(g, k, np.asarray(v))


def _attr(v):
    """A file attribute as a Python value (JSON decoded where it is)."""
    if isinstance(v, bytes):
        v = v.decode()
    if isinstance(v, str) and v[:1] in "[{":
        try:
            return json.loads(v)
        except ValueError:
            return v
    if isinstance(v, np.generic):
        return v.item()
    return v


def read_stats(path: Union[str, Path]) -> dict:
    """A statistics file, with maps as ``(channels, rows, cols)``.

    Returns
    -------
    dict
        ``attrs``, ``maps``, ``svd`` (weights ``(channels, components,
        rows, cols)``) and ``gain``.
    """
    out = {"maps": {}, "svd": {}, "gain": {}}
    with h5py.File(path, "r") as f:
        out["attrs"] = {k: _attr(v) for k, v in f.attrs.items()}
        for k, ds in f["maps"].items():
            out["maps"][k] = _to_file(ds[()])
        for k, ds in f["svd"].items():
            v = ds[()]
            out["svd"][k] = _to_file(v) if k.startswith("weight") else v
        for k, ds in f["gain"].items():
            out["gain"][k] = ds[()]
    return out
