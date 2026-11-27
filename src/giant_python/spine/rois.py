"""Windows that the spine-size readout measures in, in a consensus frame.

An :class:`RoiSet` holds weighted pixel sets, stored sparsely:

* **sources**: each source's mean footprint (``sources/spatial/profiles``),
  as GIAnT-MATLAB's spine-size scripts use it. A fixed footprint gave a
  less noisy size than the per-trial refit footprints;
* **user ROIs**: the run's ``user_rois/mask`` (weight 1 on the mask);
* **disks**: a disk of given radius around given points (marks, or ad hoc
  windows).

Identities of sources change from run to run, so ROIs of different runs and
marks are matched by position (:func:`nearest`).

Marks follow GIAnT-MATLAB's ``spine_marks.csv``: columns ``Path, Source,
Label, Note, Row0, Col0, Updated``, with zero-based consensus coordinates.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MARK_COLUMNS = ("Path", "Source", "Label", "Note", "Row0", "Col0", "Updated")


@dataclass
class RoiSet:
    """Weighted pixel sets in a ``(rows, cols)`` frame.

    Attributes
    ----------
    kind : str
        ``"sources"``, ``"user"`` or ``"disks"``.
    shape : tuple of int
        ``(rows, cols)`` of the frame.
    ids : ndarray of int, shape (R,)
        Zero-based index in the set's origin (source number, ROI number).
    labels : list of str
        One label per ROI.
    rows, cols : ndarray of float, shape (R,)
        Weighted centroids (zero-based).
    ptr : ndarray of int, shape (R + 1,)
        ROI ``k``'s pixels are ``index[ptr[k]:ptr[k + 1]]``.
    index : ndarray of int
        Flat (row-major) pixel indices.
    weight : ndarray of float
        The pixels' weights.
    """

    kind: str
    shape: tuple
    ids: np.ndarray
    labels: list
    rows: np.ndarray
    cols: np.ndarray
    ptr: np.ndarray
    index: np.ndarray
    weight: np.ndarray
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        """Number of ROIs."""
        return int(self.ids.size)

    def pixels(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """ROI ``k``'s flat pixel indices and weights."""
        s = slice(int(self.ptr[k]), int(self.ptr[k + 1]))
        return self.index[s], self.weight[s]

    def image(self, k: int) -> np.ndarray:
        """ROI ``k``'s weights as an image."""
        out = np.zeros(self.shape)
        idx, w = self.pixels(k)
        out.flat[idx] = w
        return out


def _from_images(
    kind: str, weights: np.ndarray, ids, labels, min_rel: float
) -> RoiSet:
    """An :class:`RoiSet` from dense weight images ``(R, rows, cols)``,
    keeping pixels above ``min_rel`` of each image's maximum."""
    weights = np.nan_to_num(np.asarray(weights, dtype=np.float64))
    n, h, w = weights.shape
    rr, cc = np.mgrid[0:h, 0:w]
    ptr = [0]
    index, weight, rows, cols = [], [], [], []
    for k in range(n):
        img = weights[k]
        top = img.max(initial=0.0)
        keep = img > (min_rel * top if top > 0 else np.inf)
        flat = np.flatnonzero(keep)
        vals = img.ravel()[flat]
        index.append(flat)
        weight.append(vals)
        ptr.append(ptr[-1] + flat.size)
        total = vals.sum()
        rows.append(
            (rr.ravel()[flat] * vals).sum() / total if total else np.nan
        )
        cols.append(
            (cc.ravel()[flat] * vals).sum() / total if total else np.nan
        )
    return RoiSet(
        kind=kind,
        shape=(h, w),
        ids=np.asarray(ids, dtype=np.int64),
        labels=list(labels),
        rows=np.asarray(rows, dtype=np.float64),
        cols=np.asarray(cols, dtype=np.float64),
        ptr=np.asarray(ptr, dtype=np.int64),
        index=(
            np.concatenate(index) if index else np.zeros(0, np.int64)
        ).astype(np.int64),
        weight=np.concatenate(weight) if weight else np.zeros(0),
    )


def sources_roiset(profiles: np.ndarray, min_rel: float = 0.0) -> RoiSet:
    """Sources' mean footprints as windows.

    Parameters
    ----------
    profiles : ndarray of shape (sources, rows, cols)
        ``sources/spatial/profiles`` (non-negative).
    min_rel : float
        Pixels at or below this share of a footprint's maximum are left out
        (0 keeps every pixel with a positive weight).
    """
    n = profiles.shape[0]
    labels = [f"source {k}" for k in range(n)]
    return _from_images("sources", profiles, np.arange(n), labels, min_rel)


def masks_roiset(masks: np.ndarray, labels: list) -> RoiSet:
    """User ROIs (boolean masks ``(rois, rows, cols)``) as windows of
    weight 1."""
    masks = np.asarray(masks, dtype=bool)
    if len(labels) != masks.shape[0]:
        raise ValueError(f"{masks.shape[0]} masks but {len(labels)} labels")
    return _from_images(
        "user", masks.astype(np.float64), np.arange(masks.shape[0]), labels, 0
    )


def disks_roiset(
    rows, cols, radius: float, shape: tuple, labels: list | None = None
) -> RoiSet:
    """Disks of ``radius`` pixels around points (weight 1)."""
    rows = np.asarray(rows, dtype=np.float64).ravel()
    cols = np.asarray(cols, dtype=np.float64).ravel()
    if labels is None:
        labels = [f"disk {k}" for k in range(rows.size)]
    rr, cc = np.mgrid[0 : shape[0], 0 : shape[1]]
    imgs = np.zeros((rows.size,) + tuple(shape))
    for k in range(rows.size):
        imgs[k] = ((rr - rows[k]) ** 2 + (cc - cols[k]) ** 2) <= radius**2
    out = _from_images("disks", imgs, np.arange(rows.size), labels, 0)
    out.meta["radius"] = float(radius)
    return out


def nearest(roiset: RoiSet, rows, cols, max_dist: float) -> np.ndarray:
    """For each point, the ROI whose centroid is nearest, within
    ``max_dist`` pixels (-1 where none is)."""
    rows = np.asarray(rows, dtype=np.float64).ravel()
    cols = np.asarray(cols, dtype=np.float64).ravel()
    out = np.full(rows.size, -1, np.int64)
    if len(roiset) == 0:
        return out
    for i in range(rows.size):
        d = np.hypot(roiset.rows - rows[i], roiset.cols - cols[i])
        d = np.where(np.isfinite(d), d, np.inf)
        k = int(np.argmin(d))
        if d[k] <= max_dist:
            out[i] = k
    return out


def read_marks(path) -> list:
    """Marks from a ``spine_marks.csv`` file, as dicts (``Path`` as int,
    ``Row0``/``Col0`` as float)."""
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            row = {k: row.get(k, "") for k in MARK_COLUMNS}
            row["Path"] = int(row["Path"])
            row["Row0"] = float(row["Row0"])
            row["Col0"] = float(row["Col0"])
            out.append(row)
    return out


def write_marks(path, marks: list) -> None:
    """Write marks in ``spine_marks.csv``'s format."""
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(MARK_COLUMNS))
        wr.writeheader()
        for m in marks:
            wr.writerow({k: m.get(k, "") for k in MARK_COLUMNS})
