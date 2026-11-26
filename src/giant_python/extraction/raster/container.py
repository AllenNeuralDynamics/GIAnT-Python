"""The container: one trial-path's inputs of source extraction, in one file.

A container holds everything a subproblem solver needs, so that solving
needs neither the raw data nor the SLAP2 reader nor the trial table:

* the data at the path's selected pixels and its variance multiplier, stored
  once (overlapping subproblems share pixels instead of copying them);
* the frames' lines, discards and motion, and the pixels' positions in the
  consensus frame and in the trial's registered frame;
* the path's sources and each subproblem as index lists (its pixels, its
  sources) with its crop of the consensus map;
* the trial's global and user-ROI traces;
* the parameters, the code version and a hash of the settings;
* reserved, empty datasets for the F0 network's baseline at the pixels.

File layout (HDF5, row-major, zero-based indices; lines are one-based)::

    /                      attributes: format, format_version, code_version,
                           settings_hash, params (JSON), source (JSON),
                           photon_scale, consensus_shape
    /data/y                (channels, pixels, frames) float32, counts /
                           photon_scale, channel 0 the activity channel
    /data/finv             (pixels, frames) float32 variance multiplier
    /frames/lines          (frames,) one-based line of each frame
    /frames/discard        (frames,) frames discarded for motion
    /frames/motion_r, motion_c
    /pixels/consensus_rc   (pixels, 2) row, col in the consensus frame, in
                           MATLAB's column-major order of the map
    /pixels/trial_rc       (pixels, 2) position in the registered frame
    /sources/rc            (sources, 2) row, col in the consensus frame
    /subproblems/pixel_index, pixel_offset    pixels of subproblem k:
                           pixel_index[pixel_offset[k]:pixel_offset[k + 1]]
    /subproblems/source_index, source_offset  its sources, likewise
    /subproblems/crop      (subproblems, 4) row0, col0, rows, cols
    /traces/global_f       (channels, frames), file channel order
    /traces/roi_f, roi_fsvd  (rois, channels, frames), analysis order
    /baseline/values       (pixels, 0), resizable: the F0 network's baseline
                           at the pixels, at the registration's frames
    /baseline/frame_lines  (0,), resizable: those frames' lines

``/baseline`` has the attribute ``filled = 0`` until a later stage maps the
network's baseline onto the pixels and writes it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

import h5py
import numpy as np

from ... import __version__
from .trial_data import TrialData

FORMAT = "giant-silo-container"
FORMAT_VERSION = 1

COMPRESSION = "lzf"
"""Default filter of the data: see :func:`write_container`."""

CHUNK_PIXELS = 16
"""Pixels per chunk of the data (a chunk holds all frames)."""


def _jsonable(value):
    """A value as JSON types: NumPy to Python, dataclasses to dicts."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def settings_hash(params) -> str:
    """SHA-256 of settings, independent of the order of their keys.

    Parameters
    ----------
    params : dict or dataclass
        The settings (JSON types, NumPy values or dataclasses).

    Returns
    -------
    str
        Hexadecimal digest, with the container format's version.
    """
    record = {"format_version": FORMAT_VERSION, "params": _jsonable(params)}
    text = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Subproblem:
    """One subproblem's inputs, in the convention shared by the SILo stages.

    Attributes
    ----------
    y : ndarray of float32
        ``(channels, pixels, frames)`` data in counts / photonScale, channel
        0 the activity channel.
    finv : ndarray
        ``(pixels, frames)`` variance multiplier.
    sel : ndarray of bool
        The subproblem's pixels in its crop of the consensus map; its true
        elements in column-major order are the rows of ``y``.
    sources_rc : ndarray of int
        ``(n, 2)`` zero-based row and column of its sources in the crop.
    pixels : ndarray of int
        Its pixels, as indices into the trial's selected-pixel list.
    sources : ndarray of int
        Its sources, as indices into the path's source list.
    row0, col0 : int
        Zero-based origin of the crop in the consensus frame.
    """

    y: np.ndarray
    finv: np.ndarray
    sel: np.ndarray
    sources_rc: np.ndarray
    pixels: np.ndarray
    sources: np.ndarray
    row0: int
    col0: int


@dataclass
class Container:
    """A container's content.

    Attributes
    ----------
    y, finv : ndarray or None
        The data and its variance multiplier at the selected pixels (None
        when read without data).
    frame_lines : ndarray of int
        One-based line of each frame.
    discard : ndarray of bool
        Frames discarded for motion.
    motion_r, motion_c : ndarray
        Motion applied at each frame.
    pixel_rc : ndarray of int
        ``(pixels, 2)`` consensus row and column of each pixel.
    trial_rc : ndarray
        ``(pixels, 2)`` position of each pixel in the registered frame.
    sources_rc : ndarray
        ``(sources, 2)`` consensus row and column of the path's sources.
    subproblem_pixels, subproblem_sources : list of ndarray
        Each subproblem's pixels and sources, as indices into the lists.
    crops : ndarray of int
        ``(subproblems, 4)`` row0, col0, rows, cols of each crop.
    consensus_shape : tuple
        ``(rows, cols)`` of the consensus frame.
    global_f, roi_f, roi_fsvd : ndarray
        The trial's global and user-ROI traces (see ``TrialData``).
    photon_scale : float
        Counts per unit of ``y``.
    params, source : dict
        The settings and the caller's record of where the data came from.
    code_version, settings_hash : str
        Version of the package that wrote the file; hash of ``params``.
    baseline_filled : bool
        Whether the network's baseline has been written.
    """

    y: Optional[np.ndarray]
    finv: Optional[np.ndarray]
    frame_lines: np.ndarray
    discard: np.ndarray
    motion_r: np.ndarray
    motion_c: np.ndarray
    pixel_rc: np.ndarray
    trial_rc: np.ndarray
    sources_rc: np.ndarray
    subproblem_pixels: list
    subproblem_sources: list
    crops: np.ndarray
    consensus_shape: tuple
    global_f: np.ndarray
    roi_f: np.ndarray
    roi_fsvd: np.ndarray
    photon_scale: float
    params: dict = field(default_factory=dict)
    source: dict = field(default_factory=dict)
    code_version: str = ""
    settings_hash: str = ""
    baseline_filled: bool = False

    @property
    def n_subproblems(self) -> int:
        """Number of subproblems."""
        return len(self.subproblem_pixels)

    def subproblem(self, k: int) -> Subproblem:
        """Subproblem ``k`` of a container read with its data."""
        if self.y is None:
            raise ValueError("the container was read without its data")
        pix = self.subproblem_pixels[k]
        return _subproblem(
            self.y[:, pix],
            self.finv[pix],
            self.pixel_rc[pix],
            self.sources_rc,
            pix,
            self.subproblem_sources[k],
            self.crops[k],
        )


def crop_of(pixel_rc: np.ndarray) -> np.ndarray:
    """Bounding box ``[row0, col0, rows, cols]`` of pixels."""
    if pixel_rc.shape[0] == 0:
        return np.zeros(4, dtype=np.int64)
    lo, hi = pixel_rc.min(0), pixel_rc.max(0)
    return np.array([lo[0], lo[1], hi[0] - lo[0] + 1, hi[1] - lo[1] + 1])


def _subproblem(y, finv, pixel_rc, sources_rc, pixels, sources, crop):
    """Assemble a :class:`Subproblem` from its rows and index lists."""
    row0, col0, rows, cols = (int(v) for v in crop)
    sel = np.zeros((rows, cols), dtype=bool)
    sel[pixel_rc[:, 0] - row0, pixel_rc[:, 1] - col0] = True
    return Subproblem(
        y=y,
        finv=finv,
        sel=sel,
        sources_rc=sources_rc[sources] - np.array([row0, col0]),
        pixels=np.asarray(pixels),
        sources=np.asarray(sources),
        row0=row0,
        col0=col0,
    )


def _ragged(lists, n_max: int, what: str):
    """Concatenated index lists and their offsets, checked."""
    lists = [np.asarray(v, dtype=np.int64).ravel() for v in lists]
    for k, v in enumerate(lists):
        if v.size and (v.min() < 0 or v.max() >= n_max):
            raise ValueError(f"subproblem {k}: {what} outside 0..{n_max - 1}")
        if np.any(np.diff(v) <= 0):
            raise ValueError(f"subproblem {k}: {what} must increase")
    offsets = np.concatenate([[0], np.cumsum([v.size for v in lists])])
    flat = np.concatenate(lists) if lists else np.zeros(0, np.int64)
    return flat.astype(np.int32), offsets.astype(np.int64)


def _data_kwargs(shape, chunk_pixels, compression, compression_opts):
    """Chunking and filter of a ``(..., pixels, frames)`` dataset."""
    if shape[-2] == 0 or shape[-1] == 0:
        return {}
    chunks = (
        (1,) * (len(shape) - 2)
        + (min(chunk_pixels, shape[-2]),)
        + (shape[-1],)
    )
    kwargs = {"chunks": chunks}
    if compression:
        kwargs.update(
            compression=compression,
            compression_opts=compression_opts,
            shuffle=True,
        )
    return kwargs


def write_container(
    path,
    trial: TrialData,
    sources_rc,
    subproblem_pixels: Sequence,
    subproblem_sources: Sequence,
    consensus_shape,
    params=None,
    source: Optional[dict] = None,
    compression: Optional[str] = COMPRESSION,
    compression_opts=None,
    chunk_pixels: int = CHUNK_PIXELS,
) -> Path:
    """Write one trial-path's container.

    The file is written under a temporary name and moved into place when
    complete; a failed write removes it and leaves an existing container
    untouched.

    Parameters
    ----------
    path : str or Path
        Destination (replaced if it exists).
    trial : TrialData
        The trial's data from :func:`~.trial_data.load_trial`.
    sources_rc : array_like
        ``(sources, 2)`` zero-based consensus row and column of the path's
        sources.
    subproblem_pixels, subproblem_sources : sequence of array_like
        Each subproblem's pixels (increasing indices into the trial's
        selected-pixel list) and sources (indices into ``sources_rc``).
    consensus_shape : (int, int)
        Rows and columns of the consensus frame.
    params : dict or dataclass, optional
        The settings that produced the container and that solving it will
        use; saved as JSON, and hashed.
    source : dict, optional
        A record of where the data came from (file, trial, path).
    compression : {"lzf", "gzip", None}
        Filter of the data (with byte shuffling). On a 780125 trial-path
        (13909 pixels, 5633 frames, 897 MB in memory) the default, LZF,
        gives 597 MB in 4 s; ``"gzip"`` (level ``compression_opts`` 1 to 4)
        544 to 528 MB in 17 to 23 s; None 901 MB in 1 s.
    compression_opts : int, optional
        Level for ``"gzip"``.
    chunk_pixels : int
        Pixels per chunk; a chunk holds all frames of its pixels, so a
        subproblem is read as the few chunks around its pixels (small
        chunks read single subproblems faster, at the same file size).

    Returns
    -------
    Path
        The file written.

    Raises
    ------
    ValueError
        If the subproblem lists do not match, are not increasing, or point
        outside the pixel or source lists.
    """
    path = Path(path)
    sources_rc = np.asarray(sources_rc).reshape(-1, 2)
    n_px = trial.y.shape[1]
    if len(subproblem_pixels) != len(subproblem_sources):
        raise ValueError("one source list is needed per subproblem")
    pix_flat, pix_off = _ragged(subproblem_pixels, n_px, "pixels")
    src_flat, src_off = _ragged(
        subproblem_sources, sources_rc.shape[0], "sources"
    )
    crops = np.array(
        [
            crop_of(trial.pixel_rc[pix_flat[a:b]])
            for a, b in zip(pix_off[:-1], pix_off[1:])
        ],
        dtype=np.int32,
    ).reshape(-1, 4)
    params = _jsonable(params or {})
    tmp = path.with_name(path.name + ".partial")
    try:
        with h5py.File(tmp, "w") as f:
            f.attrs["format"] = FORMAT
            f.attrs["format_version"] = FORMAT_VERSION
            f.attrs["code_version"] = __version__
            f.attrs["settings_hash"] = settings_hash(params)
            f.attrs["params"] = json.dumps(params, sort_keys=True)
            f.attrs["source"] = json.dumps(
                _jsonable(source or {}), sort_keys=True
            )
            f.attrs["photon_scale"] = float(trial.photon_scale)
            f.attrs["consensus_shape"] = np.asarray(consensus_shape, np.int64)
            for name, a in (("y", trial.y), ("finv", trial.finv)):
                f.create_dataset(
                    f"data/{name}",
                    data=np.asarray(a, dtype=np.float32),
                    **_data_kwargs(
                        a.shape, chunk_pixels, compression, compression_opts
                    ),
                )
            f["frames/lines"] = np.asarray(trial.frame_lines, np.int64)
            f["frames/discard"] = np.asarray(trial.discard, np.uint8)
            f["frames/motion_r"] = np.asarray(trial.motion_r, np.float64)
            f["frames/motion_c"] = np.asarray(trial.motion_c, np.float64)
            f["pixels/consensus_rc"] = np.asarray(trial.pixel_rc, np.int32)
            f["pixels/trial_rc"] = np.asarray(trial.trial_rc, np.float64)
            f["sources/rc"] = sources_rc
            f["subproblems/pixel_index"] = pix_flat
            f["subproblems/pixel_offset"] = pix_off
            f["subproblems/source_index"] = src_flat
            f["subproblems/source_offset"] = src_off
            f["subproblems/crop"] = crops
            f["traces/global_f"] = trial.global_f
            f["traces/roi_f"] = trial.roi_f
            f["traces/roi_fsvd"] = trial.roi_fsvd
            _reserve_baseline(
                f, n_px, chunk_pixels, compression, compression_opts
            )
        os.replace(tmp, path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    return path


def _reserve_baseline(f, n_px, chunk_pixels, compression, compression_opts):
    """Create the empty, resizable datasets of the network's baseline."""
    g = f.create_group("baseline")
    kwargs = {}
    if compression:
        kwargs.update(
            compression=compression,
            compression_opts=compression_opts,
            shuffle=True,
        )
    g.create_dataset(
        "values",
        shape=(n_px, 0),
        maxshape=(n_px, None),
        dtype=np.float32,
        chunks=(max(1, min(chunk_pixels, n_px)), 256),
        fillvalue=np.nan,
        **kwargs,
    )
    g.create_dataset(
        "frame_lines", shape=(0,), maxshape=(None,), dtype=np.float64
    )
    g.attrs["filled"] = 0
    g.attrs["units"] = "counts / photon_scale"


def _check_format(f) -> None:
    """Reject files that are not containers of a version this code reads."""
    if f.attrs.get("format") != FORMAT:
        raise ValueError(f"{f.filename} is not a SILo container")
    version = int(f.attrs["format_version"])
    if version > FORMAT_VERSION:
        raise ValueError(
            f"{f.filename} has container format {version}; this code reads "
            f"up to {FORMAT_VERSION}"
        )


def _split(flat: np.ndarray, offsets: np.ndarray) -> list:
    """The index lists of concatenated lists and their offsets."""
    return [
        flat[a:b].astype(np.int64) for a, b in zip(offsets[:-1], offsets[1:])
    ]


def read_container(path, data: bool = True) -> Container:
    """Read a container.

    Parameters
    ----------
    path : str or Path
        The container.
    data : bool
        Read ``y`` and ``finv`` too (they are most of the file). Without
        them, :func:`read_subproblem` reads single subproblems from the
        file.

    Returns
    -------
    Container
    """
    with h5py.File(path, "r") as f:
        _check_format(f)
        sub = f["subproblems"]
        return Container(
            y=f["data/y"][()] if data else None,
            finv=f["data/finv"][()] if data else None,
            frame_lines=f["frames/lines"][()],
            discard=f["frames/discard"][()].astype(bool),
            motion_r=f["frames/motion_r"][()],
            motion_c=f["frames/motion_c"][()],
            pixel_rc=f["pixels/consensus_rc"][()].astype(np.int64),
            trial_rc=f["pixels/trial_rc"][()],
            sources_rc=f["sources/rc"][()],
            subproblem_pixels=_split(
                sub["pixel_index"][()], sub["pixel_offset"][()]
            ),
            subproblem_sources=_split(
                sub["source_index"][()], sub["source_offset"][()]
            ),
            crops=sub["crop"][()].astype(np.int64),
            consensus_shape=tuple(int(v) for v in f.attrs["consensus_shape"]),
            global_f=f["traces/global_f"][()],
            roi_f=f["traces/roi_f"][()],
            roi_fsvd=f["traces/roi_fsvd"][()],
            photon_scale=float(f.attrs["photon_scale"]),
            params=json.loads(f.attrs["params"]),
            source=json.loads(f.attrs["source"]),
            code_version=str(f.attrs["code_version"]),
            settings_hash=str(f.attrs["settings_hash"]),
            baseline_filled=bool(f["baseline"].attrs["filled"]),
        )


def _read_rows(ds, rows: np.ndarray) -> np.ndarray:
    """Rows of the pixel axis (second to last) of a dataset.

    The span of the rows is read in one piece and indexed in memory: a
    subproblem's pixels lie close together in the list, and HDF5 reads a
    contiguous span much faster than a list of rows.
    """
    if rows.size == 0:
        return ds[..., :0, :]
    a, b = int(rows[0]), int(rows[-1]) + 1
    return ds[..., a:b, :][..., rows - a, :]


def read_subproblem(
    container: Union[Container, str, Path, h5py.File], k: int
) -> Subproblem:
    """Subproblem ``k`` of a container, in the shared convention.

    Parameters
    ----------
    container : Container, str, Path or h5py.File
        A container read with its data, or the file: then only the
        subproblem's part of the data is read.
    k : int
        Zero-based index of the subproblem.

    Returns
    -------
    Subproblem
    """
    if isinstance(container, Container):
        return container.subproblem(k)
    if not isinstance(container, h5py.File):
        with h5py.File(container, "r") as f:
            return read_subproblem(f, k)
    f = container
    _check_format(f)
    sub = f["subproblems"]
    n = sub["crop"].shape[0]
    if not 0 <= k < n:
        raise IndexError(f"subproblem {k} is not one of {n}")
    p0, p1 = (int(v) for v in sub["pixel_offset"][k : k + 2])
    s0, s1 = (int(v) for v in sub["source_offset"][k : k + 2])
    pix = sub["pixel_index"][p0:p1].astype(np.int64)
    rc = _read_rows(f["pixels/consensus_rc"], pix).astype(np.int64)
    return _subproblem(
        _read_rows(f["data/y"], pix),
        _read_rows(f["data/finv"], pix),
        rc,
        f["sources/rc"][()],
        pix,
        sub["source_index"][s0:s1].astype(np.int64),
        sub["crop"][k],
    )
