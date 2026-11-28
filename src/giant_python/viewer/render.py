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
  (:func:`~giant_python.viewer.trial.render_dt`). The ``exact`` kernel
  (:func:`exact_weights`) undoes the double smoothing exactly for
  densely sampled frames: ``r x200(line) + (1 - r)`` times the exporter's
  average, ``r = (dt200 / dt80)**2`` (:func:`kernel_weights` chooses).
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


def interp_weights(lines200: np.ndarray, keep: np.ndarray, ds_lines):
    """Linear interpolation of the kept 200 Hz frames at the 80 Hz lines.

    Each 80 Hz frame takes the two kept 200 Hz frames around its line
    (discarded frames are skipped, so a gap is bridged by its kept
    neighbours); before the first or after the last kept frame, the
    nearest kept frame.

    Parameters
    ----------
    lines200 : ndarray, shape (n200,)
        SILo's frame lines (increasing).
    keep : ndarray of bool, shape (n200,)
        Frames that may be used.
    ds_lines : ndarray, shape (n80,)
        The registered frames' lines.

    Returns
    -------
    scipy.sparse.csc_matrix, shape (n200, n80)
        Columns sum to 1 (all empty without a kept frame).
    """
    lines200 = np.asarray(lines200, dtype=np.float64).ravel()
    ds = np.asarray(ds_lines, dtype=np.float64).ravel()
    k = np.flatnonzero(np.asarray(keep, dtype=bool).ravel())
    shape = (lines200.size, ds.size)
    j = np.arange(ds.size)
    if k.size == 0:
        return sparse.csc_matrix(shape)
    if k.size == 1:
        return sparse.csc_matrix(
            (np.ones(ds.size), (np.full(ds.size, k[0]), j)), shape=shape
        )
    lk = lines200[k]
    i = np.clip(np.searchsorted(lk, ds, side="right") - 1, 0, k.size - 2)
    f = np.clip((ds - lk[i]) / (lk[i + 1] - lk[i]), 0.0, 1.0)
    return sparse.csc_matrix(
        (np.r_[1 - f, f], (np.r_[k[i], k[i + 1]], np.r_[j, j])), shape=shape
    )


def frame_dt200(lines200: np.ndarray, default: float) -> float:
    """The 200 Hz frames' own kernel ``dt`` in lines,
    ``ceil(mean(diff(lines)))`` (``default`` with fewer than 2 frames)."""
    lines200 = np.asarray(lines200, dtype=np.float64).ravel()
    if lines200.size < 2:
        return float(default)
    return float(math.ceil(float(np.mean(np.diff(lines200)))))


def exact_weights(
    lines200: np.ndarray,
    discard200: np.ndarray,
    ds_lines: np.ndarray,
    dt80: float,
    dt200: float,
    reach: float,
):
    """Weights that turn 200 Hz frames into the movie's 80 Hz frames
    without smoothing twice.

    Both frame rates are weighted means of the same samples with
    ``exp(-|dline| / dt)``, whose spectrum is ``1 / (1 + dt**2 w**2)``.
    The 80 Hz frame over the 200 Hz frame is therefore ``(1 + dt200**2
    w**2) / (1 + dt80**2 w**2) = r + (1 - r) / (1 + dt80**2 w**2)``,
    ``r = (dt200 / dt80)**2``: the 80 Hz frame is ``r`` times the 200 Hz
    frames interpolated at its line (:func:`interp_weights`) plus ``1 -
    r`` times their average with the movie's own kernel
    (:func:`frame_weights`, the exporter's weights). Exact for densely
    sampled pixels and untruncated kernels (on such data fast events keep
    0.99-1.01 of their 80 Hz peak); kernels cut at ``3 dt`` put them 1-2%
    high, and pixels sampled once per cycle deviate by about 1% either
    way (noiseless injected SLAP2 events: 0.99 and 1.01 on the two paths
    of the GFP copy). Discarded 200 Hz frames are left out of both terms.

    Parameters
    ----------
    lines200, discard200, ds_lines, reach
        As :func:`frame_weights`.
    dt80, dt200 : float
        The movie's and the 200 Hz frames' kernel ``dt`` (lines).

    Returns
    -------
    weights : scipy.sparse.csc_matrix, shape (n200, n80)
        Columns sum to 1, or are empty where no kept 200 Hz frame lies
        within reach.
    discard_share : ndarray, shape (n80,)
        As :func:`frame_weights` (of the movie's kernel).
    r : float
        The interpolated term's weight.
    """
    wk, share = frame_weights(lines200, discard200, ds_lines, dt80, reach)
    r = min(1.0, (float(dt200) / float(dt80)) ** 2)
    keep = ~np.asarray(discard200, dtype=bool).ravel()
    wi = interp_weights(lines200, keep, ds_lines)
    empty = np.asarray(wk.sum(0)).ravel() == 0
    if empty.any():  # nothing within reach: no activity, as frame_weights
        wi = wi @ sparse.diags((~empty).astype(np.float64))
    w = (r * wi + (1.0 - r) * wk).tocsc()
    w.eliminate_zeros()
    return w, share, r


KERNELS = ("exporter", "matched", "exact")


def kernel_weights(
    kernel: str,
    lines200: np.ndarray,
    discard200: np.ndarray,
    ds_lines: np.ndarray,
    dt80: float,
    lines_per_cycle: float = 0,
):
    """A time kernel's weights from a trial's 200 Hz frames to its 80 Hz
    frames.

    Parameters
    ----------
    kernel : {"exporter", "matched", "exact"}
        ``exporter``: ``exp(-|dline| / dt80)``, the training-data
        exporter's (smooths fast events twice); ``matched``: the same with
        ``dt = sqrt(dt80**2 - dt200**2)``; ``exact``:
        :func:`exact_weights`. ``dt200`` comes from the frame lines
        (:func:`frame_dt200`).
    lines200, discard200, ds_lines
        As :func:`frame_weights`.
    dt80 : float
        The movie's kernel ``dt`` (:func:`kernel_dt`).
    lines_per_cycle : float
        ``P``; the kernels reach ``max(3 dt, P)`` lines (0: unknown).

    Returns
    -------
    weights : scipy.sparse.csc_matrix, shape (n200, n80)
    discard_share : ndarray, shape (n80,)
        Share of each frame's weight on discarded 200 Hz frames under the
        movie's kernel (``dt80``) for ``exporter`` and ``exact``, under
        the narrower kernel for ``matched``.
    info : dict
        ``dt`` (the averaging kernel's), ``reach``, ``dt200`` and ``r``
        (the interpolated share; 0 except for ``exact``).
    """
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}, not {kernel!r}")
    dt200 = frame_dt200(lines200, dt80)
    if kernel == "matched":
        dt = math.sqrt(max(float(dt80) ** 2 - dt200**2, 1.0))
    else:
        dt = float(dt80)
    reach = max(3 * dt, float(lines_per_cycle))
    info = {"dt": dt, "reach": reach, "dt200": dt200, "r": 0.0}
    if kernel == "exact":
        w, share, r = exact_weights(
            lines200, discard200, ds_lines, dt80, dt200, reach
        )
        info["r"] = r
        return w, share, info
    w, share = frame_weights(lines200, discard200, ds_lines, dt, reach)
    return w, share, info


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


def _place_one(
    rows, cols, vals, tmap: TrialMap, shape, sigma: float, interp: int = 1
):
    """One footprint in the trial frame: flat pixel indices and values."""
    margin = 2 + int(math.ceil(3 * sigma)) + int(interp > 1)
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
    if interp == 1:
        out = _bilinear(patch, sr - r0, sc - c0, mode="constant", cval=0.0)
    else:  # spline of the zero-padded patch (small negative lobes kept)
        out = map_coordinates(
            patch,
            [np.ravel(sr - r0), np.ravel(sc - c0)],
            order=interp,
            mode="grid-constant",
            cval=0.0,
        ).reshape(sr.shape)
    if sigma > 0:
        out = gaussian_filter(out, sigma, mode="constant", cval=0.0)
    peak = float(out.max(initial=0.0))
    if peak > 0:
        ok = np.abs(out) > 1e-6 * peak
    else:
        ok = np.zeros(out.shape, bool)
    return (qr[ok] * shape[1] + qc[ok]).astype(np.int64), out[ok]


def place_footprints(
    footprints: tuple,
    n_sources: int,
    tmap: TrialMap,
    shape: tuple,
    sigma: float = 0.0,
    interp_order: int = 1,
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
    interp_order : int
        Interpolation of the canvas footprint at the map's inverse of each
        trial pixel: 1 bilinear (the viewer's; it blurs a footprint moved
        by a fraction of a pixel, by up to ~10% at the peak pixel of a
        1.3 px footprint), 3 cubic spline (keeps the shape; small negative
        lobes at the edge are kept so the flux is unchanged).

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
        idx, val = _place_one(
            rows[k], cols[k], vals[k], tmap, shape, sigma, interp_order
        )
        ii.append(idx)
        jj.append(np.full(idx.size, int(src[k[0]])))
        vv.append(val)
    n_px = int(shape[0]) * int(shape[1])
    if not ii:
        return sparse.csr_matrix((n_px, n_sources))
    ii, jj, vv = map(np.concatenate, (ii, jj, vv))
    return sparse.csr_matrix((vv, (ii, jj)), shape=(n_px, n_sources))
