"""Users' ROIs of a raster session: ``annotations.h5`` read, and moved into
the consensus frame.

Port of the non-GUI path of GIAnT-MATLAB's ``loadOrCreateAnnotations.m``
(main 5a0673b): an existing ``<savedr>/annotations/annotations.h5`` is read
(``loadAnnotationsH5.m``) and each path's ROIs are moved from the trial they
were drawn on into SILo's consensus frame (``roisToConsensusFrame.m``,
``shiftRoiData.m``). ROIs whose label contains ``SOMA`` mask seeding; every
ROI gets a trace. The GUI that draws ROIs, and the import of
post-acquisition annotations, are not ported: without a file a session has
no user ROIs, as MATLAB's SILo when the GUI is not available.

The file's layout (written by ``saveAnnotationsH5.m``): ``/Path<n>`` with
``dr`` and ``fn`` (the annotation trial's registered file stem), ``n_rois``
and ``roi_###`` groups holding ``type``, ``label``, ``mask`` and optional
``position``, ``center``, ``semi_axes``, ``rotation_angle``, ``radius``.
Strings are stored as UTF-16 code units, arrays column-major (MATLAB's
``h5write``); ``/coords_zero_indexed`` (1 in current files) says whether
``position`` and ``center`` are zero-based ``[row, col]``.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePath
from typing import Optional, Sequence

import h5py
import numpy as np

from .alignment import TrialWarp, mask_to_consensus


@dataclass
class UserRoi:
    """One user ROI.

    Attributes
    ----------
    label : str
        The ROI's label (``Label``); ``SOMA`` in it marks a soma.
    type : str
        ``'polygon'``, ``'circle'``, ``'ellipse'`` or as stored.
    mask : ndarray of bool, shape (rows, cols)
        Pixels of the ROI (MATLAB's orientation).
    center, position : ndarray or None
        Geometry in MATLAB's ``images.roi`` convention: ``[x, y]`` (column,
        row), one-based. Moved with the mask into the consensus frame.
    radius, rotation_angle : float or None
    semi_axes : ndarray or None
    """

    label: str
    type: str
    mask: np.ndarray
    center: Optional[np.ndarray] = None
    position: Optional[np.ndarray] = None
    radius: Optional[float] = None
    semi_axes: Optional[np.ndarray] = None
    rotation_angle: Optional[float] = None

    @property
    def is_soma(self) -> bool:
        """Whether SILo treats the ROI as a soma (no sources seeded)."""
        return "SOMA" in self.label.upper()


@dataclass
class PathAnnotations:
    """One imaging path's ROIs (an entry of MATLAB's ``ROIs``).

    Attributes
    ----------
    fn : str
        The annotation trial's registered file stem; empty if the path
        was not annotated.
    dr : str
        The folder of that file when it was annotated.
    rois : list of UserRoi
    path : int
        The path's number in the file (``Path<n>`` or ``DMD<n>``; 0 if not
        read from a file).
    """

    fn: str = ""
    dr: str = ""
    rois: list = field(default_factory=list)
    path: int = 0


def _text(f: h5py.File, path: str, default: str = "") -> str:
    """A string stored as MATLAB character codes (``uint16``)."""
    if path not in f:
        return default
    raw = np.asarray(f[path][()])
    if raw.dtype.kind in ("S", "O", "U"):
        value = raw.ravel()[0] if raw.size else ""
        return value.decode() if isinstance(value, bytes) else str(value)
    # MATLAB characters are UTF-16 code units
    codes = np.asarray(raw.ravel(), dtype="<u2")
    return codes.tobytes().decode("utf-16-le", errors="replace")


def _array(f: h5py.File, path: str) -> Optional[np.ndarray]:
    """A numeric array in MATLAB's orientation, or None."""
    if path not in f:
        return None
    a = np.asarray(f[path][()])
    return a.T if a.ndim >= 2 else a


def _coords(a: Optional[np.ndarray], zero_indexed: bool):
    """H5 coordinates as ``images.roi`` ``[x, y]`` (one-based)."""
    if a is None:
        return None
    a = np.asarray(a, dtype=np.float64).reshape(-1, 2)
    xy = np.column_stack([a[:, 1] + 1, a[:, 0] + 1]) if zero_indexed else a
    return xy[0] if xy.shape[0] == 1 else xy


def _scalar(a: Optional[np.ndarray]) -> Optional[float]:
    """First element of an array as a float, or None."""
    return None if a is None or a.size == 0 else float(a.ravel()[0])


def _type(raw: str) -> str:
    """The ROI type without MATLAB's ``images.roi.`` prefix."""
    t = raw.strip().lower()
    return t[len("images.roi.") :] if t.startswith("images.roi.") else t


def read_annotations(path) -> list:
    """Read a SILo ``annotations.h5`` (``loadAnnotationsH5.m``).

    Parameters
    ----------
    path : str or Path
        The file.

    Returns
    -------
    list of PathAnnotations
        One entry per ``Path<n>`` (or ``DMD<n>``) group, in the order of
        their numbers (``PathAnnotations.path``).
    """
    with h5py.File(path, "r") as f:
        zero = True
        if "coords_zero_indexed" in f:
            zero = bool(np.asarray(f["coords_zero_indexed"][()]).ravel()[0])
        else:
            zero = False
        groups = []
        for name in f:
            m = re.fullmatch(r"(?:Path|DMD)(\d+)", name)
            if m and isinstance(f[name], h5py.Group):
                groups.append((int(m.group(1)), name))
        out = []
        for number, name in sorted(groups):
            n_rois = _scalar(_array(f, f"{name}/n_rois")) or 0
            rois = []
            for k in range(1, int(n_rois) + 1):
                r = f"{name}/roi_{k:03d}"
                mask = _array(f, f"{r}/mask")
                rois.append(
                    UserRoi(
                        label=_text(f, f"{r}/label"),
                        type=_type(_text(f, f"{r}/type")),
                        mask=(
                            np.zeros((0, 0), bool)
                            if mask is None
                            else mask.astype(bool)
                        ),
                        center=_coords(_array(f, f"{r}/center"), zero),
                        position=_coords(_array(f, f"{r}/position"), zero),
                        radius=_scalar(_array(f, f"{r}/radius")),
                        semi_axes=_array(f, f"{r}/semi_axes"),
                        rotation_angle=_scalar(
                            _array(f, f"{r}/rotation_angle")
                        ),
                    )
                )
            out.append(
                PathAnnotations(
                    fn=_text(f, f"{name}/fn"),
                    dr=_text(f, f"{name}/dr"),
                    rois=rois,
                    path=number,
                )
            )
        return out


def annotation_trial(fn: str, fn_reg_ds: Sequence[str]) -> Optional[int]:
    """The zero-based trial an annotation was drawn on, if it is known.

    As ``roisToConsensusFrame.m``: ``fn`` is matched whole against the
    stems of the registered files, then without its extension.

    Parameters
    ----------
    fn : str
        The annotation's file name (``PathAnnotations.fn``).
    fn_reg_ds : sequence of str
        The path's registered files (trial table ``fn_reg_ds``), in trial
        order.

    Returns
    -------
    int or None
    """
    name = PurePath(str(fn).replace("\\", "/")).name
    if not name:
        return None
    stems = [
        PurePath(str(f).replace("\\", "/")).stem if f else ""
        for f in fn_reg_ds
    ]
    for candidate in (name, PurePath(name).stem):
        if candidate in stems:
            return stems.index(candidate)
    return None


def _geometry_to_consensus(xy: np.ndarray, warp: TrialWarp, trial: int):
    """``images.roi`` points (``[x, y]``, one-based) of a trial's image in
    the consensus frame, through the inverse of the trial's map (MATLAB's
    ``trialWarpInverse`` in ``roisToConsensusFrame``)."""
    pts = xy.reshape(-1, 2)
    rows, cols = warp.inverse(trial, pts[:, 1] - 1, pts[:, 0] - 1)
    moved = np.column_stack([cols + 1, rows + 1])
    return moved.reshape(xy.shape)


def rois_to_consensus(
    annotations: PathAnnotations,
    fn_reg_ds: Sequence[str],
    warp: TrialWarp,
    shape: tuple,
) -> list:
    """Move a path's ROIs into the consensus frame (``roisToConsensusFrame``).

    Each mask is moved by minus its trial's shift, rounded, and cropped or
    padded to ``shape``; geometry (``center``, ``position``) is moved by
    the exact shift. Through an affine map or a patchwise field, each
    consensus pixel takes the mask's value at the nearest pixel of its
    position in the trial, and geometry goes through the map's inverse
    (MATLAB's ``warpRoiData``). If the annotation trial is not in the table
    or was not aligned, the ROIs are only cropped or padded, with a
    warning, as in MATLAB.

    Parameters
    ----------
    annotations : PathAnnotations
        The path's ROIs as read.
    fn_reg_ds : sequence of str
        The path's registered files, in trial order.
    warp : TrialWarp
        The plan's across-trial alignment.
    shape : tuple of int
        ``(rows, cols)`` of the consensus frame.

    Returns
    -------
    list of UserRoi
        New ROIs; the input is not changed.
    """
    if not annotations.rois:
        return []
    trial = annotation_trial(annotations.fn, fn_reg_ds)
    if trial is None:
        warnings.warn(
            f"annotation trial {annotations.fn!r} is not in the trial table; "
            "using its ROIs without cross-trial alignment"
        )
    elif not warp.aligned[trial]:
        warnings.warn(
            f"annotation trial {annotations.fn!r} was not aligned across "
            "trials; using its ROIs without cross-trial alignment"
        )
        trial = None
    full_map = trial is not None and not warp.is_shift(trial)
    shift = (
        np.zeros(2)
        if trial is None or full_map
        else warp.shift(trial)
    )
    out = []
    for roi in annotations.rois:
        moved = {}
        for name in ("center", "position"):
            xy = getattr(roi, name)
            if xy is None:
                continue
            xy = np.asarray(xy, dtype=np.float64)
            if full_map:
                moved[name] = _geometry_to_consensus(xy, warp, trial)
            else:
                moved[name] = xy - shift[::-1]
        mask = (
            mask_to_consensus(roi.mask, warp, trial, tuple(shape))
            if roi.mask.size
            else np.zeros(tuple(shape), bool)
        )
        out.append(replace(roi, mask=mask, **moved))
    return out


def soma_mask(rois: Sequence[UserRoi], shape: tuple) -> np.ndarray:
    """Pixels of the ROIs labelled as somata (SILo's ``somaMask``)."""
    out = np.zeros(tuple(shape), bool)
    for roi in rois:
        if roi.is_soma and roi.mask.size:
            h = min(shape[0], roi.mask.shape[0])
            w = min(shape[1], roi.mask.shape[1])
            out[:h, :w] |= roi.mask[:h, :w]
    return out


def session_annotations(
    path: Optional[Path], n_paths: int
) -> list[Optional[PathAnnotations]]:
    """Each path's annotations from a session's file, if there is one.

    Parameters
    ----------
    path : Path or None
        ``annotations.h5``; None or a missing file gives no ROIs.
    n_paths : int
        Imaging paths of the session.

    Returns
    -------
    list of (PathAnnotations or None)
        Index ``d`` holds the annotations of the file's ``Path<d + 1>``;
        None where the file has none for that path. (MATLAB's
        ``loadAnnotationsH5`` numbers the groups it finds in order, which
        is the same for the files its ``saveAnnotationsH5`` writes, with
        every path.)
    """
    out: list = [None] * n_paths
    if path is None or not Path(path).exists():
        return out
    for entry in read_annotations(path):
        if 1 <= entry.path <= n_paths:
            out[entry.path - 1] = entry
    return out
