"""Small files that SILo's stages hand to each other.

* The per-trial localization file (the ``localize`` stage's output, one per
  trial-path): what the plan stage and the per-trial summary need of a
  localized trial.
* The seeds file: seeds given to the plan stage instead of its own seeding
  (MATLAB's ``thetaf``, for parity runs).

Layout of a localization file (HDF5, row-major; images in MATLAB's
orientation, zero-based coordinates)::

    /             attributes: format, format_version, code_version,
                  settings_hash, movie, path, trial (one-based), align_hz,
                  num_channels, z_depth, peaks_fitted (0/1), half_ims (0/1)
    /mean_im      (rows, cols, channels) mean image, file channel order
    /act_im       (rows, cols) activity image
    /summary      (rows, cols) the image the trial's peaks are fitted in
    /discard      (frames,) uint8: registered frames discarded for motion
    /peaks        (n, 4) [amplitude, row, col, sigma], when fitted with the
                  localization
    /half_ims     (2, rows, cols, channels) float32: mean images of two
                  halves of the frames (MATLAB's halfIMs), when the run's
                  across-trial alignment is patchwise

Peaks fitted later (``giant silo`` fits them in the background) go to a
sidecar ``<name>.peaks.h5`` (``/peaks``), so that the localization file is
never written while the plan stage reads it.

Layout of a seeds file: ``/Path<n>/seeds`` ``(n, 4)`` ``[amplitude, row,
col, sigma]`` per imaging path (numpy orientation), with the root attribute
``index_base`` (0 or 1) saying whether rows and columns are zero- or
one-based (MATLAB's ``thetaf`` is one-based).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from ... import __version__

LOCALIZATION_FORMAT = "giant-silo-localization"
LOCALIZATION_VERSION = 1


@dataclass
class LocalizationRecord:
    """One localized trial-path, as the later stages use it.

    Attributes
    ----------
    mean_im : ndarray of shape (rows, cols, channels)
    act_im, summary : ndarray of shape (rows, cols)
    discard : ndarray of bool, shape (frames,)
    peaks : ndarray of shape (n, 4) or None
        The trial's own peaks (zero-based, in its registered frame), when
        fitted.
    attrs : dict
        The file's attributes.
    half_ims : ndarray of shape (2, rows, cols, channels) or None
        Mean images of two halves of the frames, when stored.
    """

    mean_im: np.ndarray
    act_im: np.ndarray
    summary: np.ndarray
    discard: np.ndarray
    peaks: Optional[np.ndarray]
    attrs: dict
    half_ims: Optional[np.ndarray] = None

    def trial_images(self):
        """The plan stage's input
        (:class:`~giant_python.extraction.raster.plan.TrialImages`)."""
        from .plan import TrialImages

        return TrialImages(
            self.mean_im, self.act_im, self.peaks, self.discard, self.half_ims
        )


def _value(v):
    """An HDF5 attribute as a Python value."""
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.generic):
        return v.item()
    return v


def write_localization(
    path,
    mean_im: np.ndarray,
    act_im: np.ndarray,
    summary: np.ndarray,
    discard: np.ndarray,
    peaks: Optional[np.ndarray],
    attrs: dict,
    half_ims: Optional[np.ndarray] = None,
) -> Path:
    """Write a localization file (written whole, then moved into place).

    Parameters
    ----------
    path : str or Path
        Destination (replaced).
    mean_im, act_im, summary, discard, peaks, half_ims :
        See :class:`LocalizationRecord`; ``peaks`` None if not fitted,
        ``half_ims`` None if not computed.
    attrs : dict
        Attributes (``settings_hash``, ``movie``, ``path``, ``trial``,
        ``align_hz``, ``num_channels``, ``z_depth``).

    Returns
    -------
    Path
    """
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.partial")
    try:
        with h5py.File(tmp, "w") as f:
            f.attrs["format"] = LOCALIZATION_FORMAT
            f.attrs["format_version"] = LOCALIZATION_VERSION
            f.attrs["code_version"] = __version__
            for key, value in attrs.items():
                f.attrs[key] = value
            f.attrs["peaks_fitted"] = int(peaks is not None)
            f.attrs["half_ims"] = int(half_ims is not None)
            opts = dict(compression="gzip", compression_opts=4, shuffle=True)
            f.create_dataset("mean_im", data=np.asarray(mean_im), **opts)
            f.create_dataset("act_im", data=np.asarray(act_im), **opts)
            f.create_dataset("summary", data=np.asarray(summary), **opts)
            f["discard"] = np.asarray(discard, dtype=np.uint8)
            if peaks is not None:
                f["peaks"] = np.asarray(peaks, dtype=np.float64).reshape(-1, 4)
            if half_ims is not None:
                f.create_dataset(
                    "half_ims", data=np.asarray(half_ims, np.float32), **opts
                )
        side = peaks_file(path)
        if side.exists():  # peaks of an earlier localization
            side.unlink()
        os.replace(tmp, path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    return path


def peaks_file(path) -> Path:
    """The sidecar of a localization file's peaks fitted later."""
    path = Path(path)
    return path.with_name(path.stem + ".peaks.h5")


def _sidecar_peaks(path) -> Optional[np.ndarray]:
    """Peaks from a localization file's sidecar, or None."""
    side = peaks_file(path)
    if not side.exists():
        return None
    with h5py.File(side, "r") as f:
        return f["peaks"][()]


def localization_attrs(path) -> Optional[dict]:
    """A localization file's attributes, or None if it cannot be read.

    ``peaks_fitted`` is 1 when the peaks are in the file or its sidecar.
    """
    try:
        with h5py.File(path, "r") as f:
            if _value(f.attrs.get("format")) != LOCALIZATION_FORMAT:
                return None
            attrs = {k: _value(v) for k, v in f.attrs.items()}
    except OSError:
        return None
    if not attrs.get("peaks_fitted") and peaks_file(path).exists():
        attrs["peaks_fitted"] = 1
    return attrs


def read_localization(
    path, peaks: bool = True, half_ims: bool = True
) -> LocalizationRecord:
    """Read a localization file (and its peaks' sidecar, if there is one).

    Parameters
    ----------
    path : str or Path
        The file.
    peaks : bool
        Read the sidecar's peaks too (the plan stage does not: a sidecar may
        be being written then).
    half_ims : bool
        Read the half images, when the file has them (only the patchwise
        alignment uses them).

    Raises
    ------
    ValueError
        If the file is not a localization file of a version this code
        reads.
    """
    with h5py.File(path, "r") as f:
        if _value(f.attrs.get("format")) != LOCALIZATION_FORMAT:
            raise ValueError(f"{path} is not a SILo localization file")
        if int(f.attrs["format_version"]) > LOCALIZATION_VERSION:
            raise ValueError(
                f"{path} has localization format "
                f"{int(f.attrs['format_version'])}; this code reads up to "
                f"{LOCALIZATION_VERSION}"
            )
        rec = LocalizationRecord(
            mean_im=f["mean_im"][()],
            act_im=f["act_im"][()],
            summary=f["summary"][()],
            discard=f["discard"][()].astype(bool),
            peaks=f["peaks"][()] if "peaks" in f else None,
            attrs={k: _value(v) for k, v in f.attrs.items()},
            half_ims=(
                f["half_ims"][()] if half_ims and "half_ims" in f else None
            ),
        )
    if rec.peaks is None and peaks:
        rec.peaks = _sidecar_peaks(path)
        if rec.peaks is not None:
            rec.attrs["peaks_fitted"] = 1
    return rec


def add_peaks(path, peaks: np.ndarray) -> Path:
    """Store a trial's peaks fitted later, in the localization file's
    sidecar (written whole, then moved into place).

    Parameters
    ----------
    path : str or Path
        The localization file.
    peaks : ndarray of shape (n, 4)
        ``[amplitude, row, col, sigma]``, zero-based.

    Returns
    -------
    Path
        The sidecar.
    """
    side = peaks_file(path)
    tmp = side.with_name(f"{side.name}.{os.getpid()}.partial")
    try:
        with h5py.File(tmp, "w") as f:
            f.attrs["format"] = LOCALIZATION_FORMAT + " peaks"
            f["peaks"] = np.asarray(peaks, dtype=np.float64).reshape(-1, 4)
        os.replace(tmp, side)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    return side


def write_seeds(path, seeds: dict, index_base: int = 0) -> Path:
    """Write a seeds file.

    Parameters
    ----------
    path : str or Path
        Destination (replaced).
    seeds : dict
        One-based path number to an ``(n, 4)`` array ``[amplitude, row,
        col, sigma]``.
    index_base : {0, 1}
        Whether rows and columns are zero- or one-based.

    Returns
    -------
    Path
    """
    if index_base not in (0, 1):
        raise ValueError("index_base must be 0 or 1")
    with h5py.File(path, "w") as f:
        f.attrs["index_base"] = int(index_base)
        for d, s in seeds.items():
            f[f"Path{int(d)}/seeds"] = np.asarray(s, np.float64).reshape(-1, 4)
    return Path(path)


def read_seeds(path, n_paths: int) -> list:
    """Each path's seeds from a seeds file, zero-based.

    Parameters
    ----------
    path : str or Path
        The file.
    n_paths : int
        Imaging paths of the session.

    Returns
    -------
    list of (ndarray or None)
        Index ``d`` holds path ``d + 1``'s seeds, None where the file has
        none (that path seeds itself).
    """
    out: list = [None] * n_paths
    with h5py.File(path, "r") as f:
        base = int(f.attrs.get("index_base", 0))
        if base not in (0, 1):
            raise ValueError(f"{path}: index_base must be 0 or 1")
        for d in range(n_paths):
            key = f"Path{d + 1}/seeds"
            if key in f:
                s = np.asarray(f[key][()], np.float64).reshape(-1, 4).copy()
                s[:, 1:3] -= base
                out[d] = s
    return out
