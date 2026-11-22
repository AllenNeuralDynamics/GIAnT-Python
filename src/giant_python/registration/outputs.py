"""Registration outputs in GIAnT-MATLAB's file layout.

* The registered movie: a BigTIFF with one float32 page per channel per
  frame (``Fast_BigTiff_Write``), or an HDF5 file whose ``/data`` holds the
  same pages (``saveTif = false``).
* ``<trial>_ALIGNMENTDATA.h5``: the alignment struct written column-major by
  ``saveStructToH5``. Its per-frame variance factor is streamed to disk while
  frames are registered; the file is complete only after :meth:`finish`.
  The settings that produced it are recorded as a JSON root attribute
  (:data:`SETTINGS_ATTR`), which struct readers ignore.

Both are written under temporary names (``<name>.partial``) and moved into
place together by :func:`publish` once both are complete, the alignment file
last. A run that fails before then leaves an existing pair untouched; one
that fails while publishing leaves no alignment file, so the trial is not
mistaken for registered.
"""

import json
import math
import os
import warnings
import zlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np

from ..io.hdf5 import write_matlab_value

PIXELS_PER_CM = 4e4
"""Resolution tag of the registered TIFF (250 nm pixels), as in MATLAB."""

SETTINGS_ATTR = "align_params"
"""Root attribute of an alignment file: JSON of :func:`settings_record`."""


def settings_record(params) -> dict:
    """The settings that produced a trial's outputs, as saved with them.

    The trial table's ``align_params`` fields (MATLAB names) without
    ``overwriteExisting``, which only decides whether outputs are made;
    NaN (no junction limit) is stored as None.

    Parameters
    ----------
    params : MultiRoiRegistrationParams
        Settings of the registration.

    Returns
    -------
    dict
        JSON-serializable record.
    """
    record = params.to_matlab()
    record.pop("overwriteExisting")
    return {
        key: None if isinstance(value, float) and math.isnan(value) else value
        for key, value in record.items()
    }


class _Background:
    """Run writes in order on one thread, with a bounded backlog."""

    def __init__(self, max_pending: int = 64):
        """Start the thread."""
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._pending = deque()
        self._max = max_pending

    def submit(self, fn, *args) -> None:
        """Queue ``fn(*args)``; wait while too many writes are pending."""
        self._pending.append(self._pool.submit(fn, *args))
        while len(self._pending) > self._max:
            self._pending.popleft().result()

    def drain(self) -> None:
        """Wait for queued writes, re-raising their errors."""
        while self._pending:
            self._pending.popleft().result()

    def close(self) -> None:
        """Wait for queued writes and stop the thread."""
        try:
            self.drain()
        finally:
            self._pool.shutdown(wait=True)


_COMPRESSORS = None


def _compressors() -> ThreadPoolExecutor:
    """Shared pool that deflates chunks in parallel (zlib releases the GIL)."""
    global _COMPRESSORS
    if _COMPRESSORS is None:
        _COMPRESSORS = ThreadPoolExecutor(
            max_workers=max(1, min(8, (os.cpu_count() or 2) // 2))
        )
    return _COMPRESSORS


def _deflate(page: np.ndarray, level: int) -> bytes:
    """One frame's chunk, transposed to MATLAB layout and deflated."""
    return zlib.compress(np.ascontiguousarray(page.T).tobytes(), level)


class _ChunkedFrames:
    """An HDF5 dataset of frames, one chunk each, optionally deflated.

    Deflated chunks are compressed on a thread pool and stored in order
    with ``write_direct_chunk``; the dataset carries HDF5's standard deflate
    filter, so any reader (MATLAB's ``h5read`` included) decompresses them.
    """

    def __init__(self, group, name, n_frames, shape, level: int):
        """Create the dataset ``(n_frames, cols, rows)`` (MATLAB layout)."""
        rows, cols = shape
        self.level = int(level)
        opts = {}
        if self.level:
            opts = dict(compression="gzip", compression_opts=self.level)
        self.ds = group.create_dataset(
            name,
            shape=(n_frames, cols, rows),
            dtype=np.float32,
            chunks=(1, cols, rows) if self.level else None,
            **opts,
        )

    def prepare(self, page: np.ndarray):
        """Start compressing a page; returns what :meth:`store` needs."""
        page = np.asarray(page, np.float32)
        if not self.level:
            return page
        return _compressors().submit(_deflate, page, self.level)

    def store(self, index: int, prepared) -> None:
        """Write a prepared page (call in frame order, from one thread)."""
        if not self.level:
            self.ds[index] = prepared.T
            return
        self.ds.id.write_direct_chunk((index, 0, 0), prepared.result())


def partial_path(path: Path) -> Path:
    """Temporary name of an output until it is published."""
    return path.with_name(path.name + ".partial")


class MovieWriter:
    """Append registered pages to a TIFF or HDF5 movie, under a temporary name.

    Parameters
    ----------
    path : str or Path
        Final destination; the pages go to ``<path>.partial`` until
        :func:`publish`.
    shape : (int, int)
        Page shape ``(rows, cols)``.
    n_pages : int
        Pages to be written (frames x channels); needed for HDF5.
    tiff : bool
        Write a BigTIFF (else HDF5 ``/data``, one chunk per page, MATLAB
        dimensions ``[rows cols pages]``).
    deflate : int
        HDF5 movies: deflate level (0 for none; MATLAB writes 4).
    """

    def __init__(
        self,
        path: Union[str, Path],
        shape,
        n_pages: int,
        tiff: bool = True,
        deflate: int = 1,
    ):
        """Open the temporary file."""
        self.path = Path(path)
        self.tmp = partial_path(self.path)
        self.tiff = tiff
        self.n = 0
        self._closed = False
        if self.tmp.exists():
            self.tmp.unlink()
        if tiff:
            import tifffile

            self._tw = tifffile.TiffWriter(self.tmp, bigtiff=True)
        else:
            self._h5 = h5py.File(self.tmp, "w")
            self._frames = _ChunkedFrames(
                self._h5, "data", n_pages, shape, deflate
            )
        self._bg = _Background()

    def write(self, page: np.ndarray) -> None:
        """Append one ``(rows, cols)`` page.

        Pages are stored as MATLAB's ``Fast_BigTiff_Write`` stores them:
        column-major, so a TIFF reader that is not MATLAB's (e.g.
        ``tifffile``) returns each page transposed, ``(cols, rows)``.
        """
        page = np.asarray(page, np.float32)
        if not self.tiff:
            page = self._frames.prepare(page)
        self._bg.submit(self._write, page, self.n)
        self.n += 1

    def _write(self, page, index: int) -> None:
        """Write one page (on the background thread)."""
        if self.tiff:
            self._tw.write(
                np.ascontiguousarray(page.T),
                photometric="minisblack",
                resolution=(PIXELS_PER_CM, PIXELS_PER_CM),
                resolutionunit="CENTIMETER",
            )
        else:
            self._frames.store(index, page)

    def close(self) -> None:
        """Finish queued pages and close the (still temporary) file."""
        if self._closed:
            return
        self._closed = True
        try:
            self._bg.close()
        finally:
            if self.tiff:
                self._tw.close()
            else:
                self._h5.close()

    def abort(self) -> None:
        """Close and delete the temporary file."""
        try:
            self.close()
        except Exception:
            pass
        finally:
            if self.tmp.exists():
                self.tmp.unlink()


class AlignmentFile:
    """Build ``_ALIGNMENTDATA.h5`` under a temporary name.

    Parameters
    ----------
    path : str or Path
        Final destination; the file is ``<path>.partial`` until
        :func:`publish`.
    var_shape : (int, int)
        Shape of each frame's variance factor ``(rows, cols)``.
    n_frames : int
        Number of registered frames.
    deflate : int
        Deflate level of ``varFacDS`` (0 for none, as MATLAB writes it).
    """

    def __init__(
        self,
        path: Union[str, Path],
        var_shape,
        n_frames: int,
        deflate: int = 1,
    ):
        """Create the temporary file and the variance-factor dataset."""
        self.path = Path(path)
        self.tmp = partial_path(self.path)
        if self.tmp.exists():
            self.tmp.unlink()
        self._f = h5py.File(self.tmp, "w")
        self._var = _ChunkedFrames(
            self._f.require_group("slap2"),
            "varFacDS",
            n_frames,
            var_shape,
            deflate,
        )
        self._bg = _Background()

    def write_var_factor(self, index: int, var_factor: np.ndarray) -> None:
        """Queue frame ``index``'s ``(rows, cols)`` variance factor."""
        self._bg.submit(self._var.store, index, self._var.prepare(var_factor))

    def finish(self, fields: dict, settings: Optional[dict] = None) -> None:
        """Write the remaining fields and close the (still temporary) file.

        Parameters
        ----------
        fields : dict
            Struct fields as accepted by ``write_matlab_value``; entries of a
            ``'slap2'`` dict go into the existing ``/slap2`` group.
        settings : dict, optional
            :func:`settings_record` of the registration, saved as the root
            attribute :data:`SETTINGS_ATTR`.
        """
        self._bg.close()
        fields = dict(fields)
        slap2 = fields.pop("slap2", {})
        for key, value in fields.items():
            write_matlab_value(self._f, key, value)
        grp = self._f["slap2"]
        for key, value in slap2.items():
            write_matlab_value(grp, key, value)
        self._f.create_dataset("row_major", data=np.zeros((1, 1), np.uint8))
        if settings is not None:
            self._f.attrs[SETTINGS_ATTR] = json.dumps(settings, sort_keys=True)
        self._f.close()

    def abort(self) -> None:
        """Close and delete the temporary file."""
        try:
            self._bg.close()
        except Exception:
            pass
        try:
            self._f.close()
        finally:
            if self.tmp.exists():
                self.tmp.unlink()


def publish(movie: MovieWriter, adata: AlignmentFile) -> None:
    """Move a finished movie and alignment file into place, as a pair.

    The previous alignment file is removed first and the new one is moved
    in last. A pair whose alignment file exists is therefore complete and
    consistent; if publishing is interrupted, no alignment file remains and
    the trial is registered again.

    Parameters
    ----------
    movie : MovieWriter
        A closed movie.
    adata : AlignmentFile
        A finished alignment file.
    """
    adata.path.unlink(missing_ok=True)
    os.replace(movie.tmp, movie.path)
    os.replace(adata.tmp, adata.path)


def downsample2(image: np.ndarray) -> np.ndarray:
    """Sum 2 x 2 blocks (trailing odd row/column dropped; NaN propagates)."""
    h, w = image.shape[-2] // 2 * 2, image.shape[-1] // 2 * 2
    x = image[..., :h, :w]
    return (x[..., 0::2, 0::2] + x[..., 0::2, 1::2] + x[..., 1::2, 0::2]) + x[
        ..., 1::2, 1::2
    ]


def matlab_round(x):
    """Round half away from zero, as MATLAB's ``round``."""
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def rec_neg_err(movie_ds: np.ndarray, align_hz: float) -> np.ndarray:
    """Per-frame negative reconstruction error (MultiRoiRegistration).

    Frames are taken in chunks of at most 10 s; each chunk's template is its
    NaN-ignoring median (NaN where more than 20% of frames are NaN). The
    error is the RMS of the template's excess over the frame, relative to
    the template, both scaled by ``sqrt(max(0, template) + 10)``. As in
    MATLAB, pixels where either is NaN count as zero in the numerator's mean
    but are left out of the denominator's.

    Parameters
    ----------
    movie_ds : ndarray of float32
        ``(frames, rows, cols)`` spatially downsampled registered frames.
    align_hz : float
        Frame rate.

    Returns
    -------
    ndarray of float64
        ``(frames,)`` errors.
    """
    n = movie_ds.shape[0]
    out = np.full(n, np.nan)
    offset = np.float32(10)
    n_chunks = int(np.ceil(n / (align_hz * 10)))
    edges = matlab_round(np.linspace(1, n + 1, n_chunks + 1)).astype(int)
    for a, b in zip(edges[:-1] - 1, edges[1:] - 1):
        chunk = movie_ds[a:b]
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            template = np.nanmedian(chunk, axis=0).astype(np.float32)
            nan_frac = np.isnan(chunk).mean(0)
            template[nan_frac > 0.2] = np.nan
            gamma = np.sqrt(np.maximum(0, template) + offset)
            excess = (template - chunk) / gamma
            num = np.where(np.isnan(excess), 0, np.maximum(0, excess)) ** 2
            den = (template / gamma) ** 2 + chunk * 0
            den_mean = np.nanmean(den.reshape(den.shape[0], -1), axis=1)
            num_mean = num.reshape(num.shape[0], -1).mean(axis=1)
            out[a:b] = np.sqrt(num_mean / den_mean)
    return out
