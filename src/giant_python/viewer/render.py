"""SILo's fitted activity rendered in a trial's registered 80 Hz movie.

The rendering follows the F0 network's training-data exporter
(``exportF0TrainingData.m`` in slap2-F0-network), so that the movie minus
the rendered activity is what a training target made from the run would
be, with these changes:

* **Time.** An 80 Hz frame is the registration's weighted mean of the
  samples within ``max(3 dt, P)`` lines of its line, with weights
  ``exp(-|dline| / dt)``, ``dt = ceil(mean(diff(DSframes)))`` lines and
  ``P`` the recording's lines per cycle. The 200 Hz traces are averaged
  at SILo's frame lines with such weights, leaving out SILo's discarded
  frames (:func:`frame_weights`); an 80 Hz frame whose weight falls more
  than half on discarded 200 Hz frames is discarded. The exporter uses
  the movie's ``dt``; since the 200 Hz frames are already averages, the
  viewer's default narrows it to ``sqrt(dt**2 - dt200**2)``
  (:func:`~giant_python.viewer.trial.render_dt`).
* **Space.** Footprints live on SILo's consensus canvas; each trial's
  footprints are moved into the trial's registered frame through the
  trial's across-trial map (rigid, affine or patchwise), the one SILo read
  the trial's data with: the trial-frame footprint at pixel ``q`` is the
  canvas footprint at the map's inverse of ``q``, interpolated bilinearly
  (:class:`TrialMap`, :func:`place_footprints`). The exporter instead
  searches each trial's offset (integer offset of the mean images plus a
  sub-pixel search that maximizes the movie-activity correlation at the
  source peaks) and applies it to the canvas as a rigid shift.
* **Blur.** Optionally a Gaussian blur of the moved footprints (the
  exporter chooses one per trial by matching the movie's core and ring
  slopes; here it is a fixed setting, 0 by default).
* **Scale.** None: the activity is SILo's footprints (each sums to 1)
  times its denoised traces (``dF_denoised``, movie units), as the
  exporter writes it (``activityGain`` 1).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter, map_coordinates


def kernel_dt(ds_lines: np.ndarray, lines200: np.ndarray, ratio: float):
    """The kernel's time constant in lines (the exporter's ``dt``).

    ``ceil(mean(diff(DSframes)))``; with one registered frame, the 200 Hz
    frame spacing times ``ratio`` (analysis rate over movie rate).
    """
    ds_lines = np.asarray(ds_lines, dtype=np.float64).ravel()
    if ds_lines.size > 1:
        return int(math.ceil(np.mean(np.diff(ds_lines))))
    step = np.median(np.diff(np.asarray(lines200, dtype=np.float64)))
    return int(math.ceil(step * ratio))


def frame_weights(
    lines200: np.ndarray,
    discard200: np.ndarray,
    ds_lines: np.ndarray,
    dt: int,
    reach: float,
):
    """Weights that average 200 Hz frames into the movie's 80 Hz frames.

    Parameters
    ----------
    lines200 : ndarray, shape (n200,)
        SILo's frame lines of the trial (one-based, increasing).
    discard200 : ndarray of bool, shape (n200,)
        SILo's discarded frames.
    ds_lines : ndarray, shape (n80,)
        The registered frames' lines (``DSframes``).
    dt : int
        Kernel time constant in lines.
    reach : float
        Frames within this many lines count (``max(3 dt, P)``).

    Returns
    -------
    weights : scipy.sparse.csc_matrix, shape (n200, n80)
        Each 80 Hz frame's weights over the kept 200 Hz frames (columns sum
        to 1, or are empty).
    discard_share : ndarray, shape (n80,)
        Share of each frame's weight on discarded 200 Hz frames (1 where
        no 200 Hz frame is within reach).
    """
    lines200 = np.asarray(lines200, dtype=np.float64).ravel()
    disc = np.asarray(discard200, dtype=bool).ravel()
    ds = np.asarray(ds_lines, dtype=np.float64).ravel()
    lo = np.searchsorted(lines200, ds - reach, side="left")
    hi = np.searchsorted(lines200, ds + reach, side="right")
    ii, jj, ww = [], [], []
    share = np.ones(ds.size)
    for j in range(ds.size):
        idx = np.arange(lo[j], hi[j])
        if idx.size == 0:
            continue
        w = np.exp(-np.abs(lines200[idx] - ds[j]) / dt)
        share[j] = w[disc[idx]].sum() / max(w.sum(), np.finfo(float).eps)
        keep = ~disc[idx]
        if keep.any():
            ii.append(idx[keep])
            jj.append(np.full(int(keep.sum()), j))
            ww.append(w[keep] / w[keep].sum())
    if ii:
        ii, jj, ww = map(np.concatenate, (ii, jj, ww))
    m = sparse.csc_matrix((ww, (ii, jj)), shape=(lines200.size, ds.size))
    return m, share


def _bilinear(img: np.ndarray, r, c, mode: str = "nearest", cval=0.0):
    """Bilinear samples of ``img`` at zero-based ``(r, c)``."""
    return map_coordinates(
        img, [np.ravel(r), np.ravel(c)], order=1, mode=mode, cval=cval
    ).reshape(np.shape(r))


class TrialMap:
    """A trial's map from the consensus canvas to its registered frame.

    Zero-based pixel coordinates, as GIAnT-Python's summaries store them
    (``frame_info/trial_alignment_affine``: ``[row_t, col_t] = A @ [row,
    col, 1]``; ``frame_info/trial_alignment_displacement`` for patchwise
    maps: canvas pixel ``(r, c)`` lies at ``(r, c) + displacement`` in the
    trial). With a displacement field, the field (sampled on the canvas,
    interpolated bilinearly, edge values beyond it) is the map; without,
    the affine map.

    Parameters
    ----------
    affine : ndarray of shape (2, 3)
    field : ndarray of shape (2, rows, cols), optional
    """

    def __init__(self, affine: np.ndarray, field: np.ndarray | None = None):
        """Store the map."""
        self.affine = np.asarray(affine, dtype=np.float64).reshape(2, 3)
        self.field = None if field is None else np.asarray(field, np.float64)

    @property
    def valid(self) -> bool:
        """Whether the trial has a map (it was aligned)."""
        if self.field is not None:
            return bool(np.isfinite(self.field).all())
        return bool(np.isfinite(self.affine).all())

    def forward(self, r, c):
        """Trial-frame position of canvas positions ``(r, c)``."""
        r = np.asarray(r, dtype=np.float64)
        c = np.asarray(c, dtype=np.float64)
        if self.field is not None:
            return (
                r + _bilinear(self.field[0], r, c),
                c + _bilinear(self.field[1], r, c),
            )
        a = self.affine
        return (
            a[0, 0] * r + a[0, 1] * c + a[0, 2],
            a[1, 0] * r + a[1, 1] * c + a[1, 2],
        )

    def inverse(self, rt, ct, iters: int = 20, tol: float = 1e-6):
        """Canvas position of trial-frame positions ``(rt, ct)``.

        Exact for an affine map; for a displacement field the fixed point
        of ``p = q - d(p)``, started at the affine map's inverse.
        """
        rt = np.asarray(rt, dtype=np.float64)
        ct = np.asarray(ct, dtype=np.float64)
        a = self.affine
        if not np.isfinite(a).all():
            a = np.array([[1.0, 0, 0], [0, 1.0, 0]])
        m = np.linalg.inv(a[:, :2])
        dr, dc = rt - a[0, 2], ct - a[1, 2]
        r = m[0, 0] * dr + m[0, 1] * dc
        c = m[1, 0] * dr + m[1, 1] * dc
        if self.field is None:
            return r, c
        for _ in range(iters):
            r_new = rt - _bilinear(self.field[0], r, c)
            c_new = ct - _bilinear(self.field[1], r, c)
            step = max(
                float(np.max(np.abs(r_new - r), initial=0)),
                float(np.max(np.abs(c_new - c), initial=0)),
            )
            r, c = r_new, c_new
            if step < tol:
                break
        return r, c


def _patch(rows, cols, vals, margin: int):
    """A footprint's dense canvas patch with a zero margin; returns the
    patch and its origin ``(row, col)``."""
    r0, c0 = int(rows.min()) - margin, int(cols.min()) - margin
    h = int(rows.max()) - r0 + margin + 1
    w = int(cols.max()) - c0 + margin + 1
    patch = np.zeros((h, w))
    patch[rows - r0, cols - c0] = vals
    return patch, r0, c0


def _place_one(rows, cols, vals, tmap: TrialMap, shape, sigma: float):
    """One footprint in the trial frame: flat pixel indices and values."""
    margin = 2 + int(math.ceil(3 * sigma))
    patch, r0, c0 = _patch(rows, cols, vals, margin)
    h, w = patch.shape
    pr, pc = np.mgrid[r0 : r0 + h, c0 : c0 + w]
    tr, tc = tmap.forward(pr, pc)
    lo_r = max(0, int(math.floor(np.min(tr))))
    hi_r = min(shape[0] - 1, int(math.ceil(np.max(tr))))
    lo_c = max(0, int(math.floor(np.min(tc))))
    hi_c = min(shape[1] - 1, int(math.ceil(np.max(tc))))
    if hi_r < lo_r or hi_c < lo_c:
        return np.zeros(0, np.int64), np.zeros(0)
    qr, qc = np.mgrid[lo_r : hi_r + 1, lo_c : hi_c + 1]
    sr, sc = tmap.inverse(qr, qc)
    out = _bilinear(patch, sr - r0, sc - c0, mode="constant", cval=0.0)
    if sigma > 0:
        out = gaussian_filter(out, sigma, mode="constant", cval=0.0)
    peak = float(out.max(initial=0.0))
    ok = out > 1e-6 * peak if peak > 0 else np.zeros(out.shape, bool)
    return (qr[ok] * shape[1] + qc[ok]).astype(np.int64), out[ok]


def place_footprints(
    footprints: tuple,
    n_sources: int,
    tmap: TrialMap,
    shape: tuple,
    sigma: float = 0.0,
) -> sparse.csr_matrix:
    """A trial's footprints moved into its registered frame.

    Parameters
    ----------
    footprints : tuple of ndarray
        ``(rows, cols, sources, values)``: the trial's footprints on the
        canvas, one entry per nonzero pixel (zero-based).
    n_sources : int
        Number of sources (columns of the result).
    tmap : TrialMap
        The trial's map.
    shape : tuple of int
        ``(rows, cols)`` of the registered frame.
    sigma : float
        Gaussian blur (px) after the move; 0 for none.

    Returns
    -------
    scipy.sparse.csr_matrix, shape (rows * cols, n_sources)
        Row-major flat pixel index by source.
    """
    rows, cols, src, vals = (np.asarray(x) for x in footprints)
    order = np.argsort(src, kind="stable")
    rows, cols, src, vals = rows[order], cols[order], src[order], vals[order]
    edges = np.flatnonzero(np.diff(src)) + 1
    ii, jj, vv = [], [], []
    for k in np.split(np.arange(src.size), edges):
        if k.size == 0:
            continue
        idx, val = _place_one(rows[k], cols[k], vals[k], tmap, shape, sigma)
        ii.append(idx)
        jj.append(np.full(idx.size, int(src[k[0]])))
        vv.append(val)
    n_px = int(shape[0]) * int(shape[1])
    if not ii:
        return sparse.csr_matrix((n_px, n_sources))
    ii, jj, vv = map(np.concatenate, (ii, jj, vv))
    return sparse.csr_matrix((vv, (ii, jj)), shape=(n_px, n_sources))
