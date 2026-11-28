"""Quality values of an F0 training trial: how the rendered activity
compares with the movie.

Ports of the QC in slap2-F0-network's ``exportF0TrainingData.m``
(``peakFit``, ``diskFit``, ``shapeFit``, the event-triggered residual) and
of its ``ringEventRatios.m`` (reference: ``slap2_f0.ringqc``). The values
are written as the trial file's attributes under the MATLAB names
(``peakSlope``, ``peakR``, ``diskSlope``, ``diskR``, ``activitySlope``,
``blurShapeRatio``, ``eventResidualFraction``, ``nEvents``,
``eventRatioCore``, ``eventRatio2px``, ``eventRatioSources``,
``eventRatioEvents``, ``eventRatioArtifactSources``), so that QC tables of
old and new exports line up.

Conventions: the movie ``Y`` is ``(frames, pixels)`` over the trial's
row-major flat pixels, NaN where a pixel is not imaged and at discarded
frames; footprints ``H`` are a sparse ``(pixels, sources)`` matrix and the
sources' 80 Hz traces ``X`` are ``(sources, frames)``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

from ..extraction.raster.localization import matlab_prctile

HPF_WIDTH = 81  # frames of the running median the fits are high-passed by


def movmedian_omitnan(x: np.ndarray, width: int = HPF_WIDTH) -> np.ndarray:
    """MATLAB ``movmedian(x, width, 2, 'omitnan')`` (endpoints shrink).

    Each row's running median over ``width`` frames (odd), NaN left out;
    NaN where a window holds no value.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    h = width // 2
    out = np.empty_like(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for a in range(0, x.shape[0], 32):
            blk = x[a : a + 32]
            pad = np.full((blk.shape[0], h), np.nan)
            blk = np.concatenate([pad, blk, pad], axis=1)
            win = np.lib.stride_tricks.sliding_window_view(blk, width, axis=1)
            out[a : a + 32] = np.nanmedian(win, axis=-1)
    return out


def hpf(x: np.ndarray) -> np.ndarray:
    """The rows minus their 81-frame running median (the fits' high
    pass)."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    return x - movmedian_omitnan(x)


def _slope(yh: np.ndarray, ah: np.ndarray) -> float:
    """Least-squares slope of ``yh`` on ``ah`` over frames where both are
    finite (NaN when ``ah`` is zero there)."""
    ok = np.isfinite(yh) & np.isfinite(ah)
    den = float(np.sum(ah[ok] ** 2))
    return float(np.sum(yh[ok] * ah[ok]) / den) if den > 0 else np.nan


def _corr(yh: np.ndarray, ah: np.ndarray, min_frames: int = 100) -> float:
    """Correlation over frames where both are finite (NaN with ``min_frames``
    or fewer, or a constant activity)."""
    ok = np.isfinite(yh) & np.isfinite(ah)
    if ok.sum() <= min_frames or np.std(ah[ok]) == 0:
        return np.nan
    return float(np.corrcoef(yh[ok], ah[ok])[0, 1])


def peak_pixels(H) -> np.ndarray:
    """Each source's peak pixel (flat index of its largest footprint
    value; -1 for a source without footprint pixels)."""
    H = H.tocsc()
    pk = np.full(H.shape[1], -1, np.int64)
    for s in range(H.shape[1]):
        a, b = H.indptr[s], H.indptr[s + 1]
        if b > a:
            pk[s] = H.indices[a + int(np.argmax(H.data[a:b]))]
    return pk


def peak_fit(H, X: np.ndarray, Y: np.ndarray, pk: np.ndarray):
    """Slope and correlation of the movie on the activity at each source's
    peak pixel (high-passed), with every source's activity at that pixel
    (``peakFit``). Returns two arrays of length ``sources`` (NaN for a
    source without pixels)."""
    n = pk.size
    sl, r = np.full(n, np.nan), np.full(n, np.nan)
    ok = pk >= 0
    if not ok.any():
        return sl, r
    A = np.asarray(H[pk[ok]] @ X)
    yh, ah = hpf(Y[:, pk[ok]].T), hpf(A)
    for i, s in enumerate(np.flatnonzero(ok)):
        sl[s], r[s] = _slope(yh[i], ah[i]), _corr(yh[i], ah[i])
    return sl, r


def _flux_fit(H, X, Y, idx):
    """Slope and correlation of the movie's flux over pixels ``idx`` on the
    activity's (high-passed; the movie's flux is NaN at a frame where any
    of the pixels is)."""
    fy = Y[:, idx].astype(np.float64).sum(1)
    fa = np.asarray(H[idx].sum(0)).ravel() @ X
    yh, ah = hpf(fy)[0], hpf(fa)[0]
    if not np.any(ah[np.isfinite(yh) & np.isfinite(ah)]):
        return np.nan, np.nan
    ok = np.isfinite(yh) & np.isfinite(ah)
    if ok.sum() <= 100:
        return np.nan, np.nan
    return _slope(yh, ah), _corr(yh, ah)


def disk_fit(H, X, Y, pk, shape, radius: float = 4.0):
    """Slope and correlation of the movie on the activity for the flux in
    the footprint-support pixels within ``radius`` px of each source's
    peak pixel (``diskFit``)."""
    n = pk.size
    sl, r = np.full(n, np.nan), np.full(n, np.nan)
    sup = np.flatnonzero(np.asarray(H.getnnz(axis=1)) > 0)
    sr, sc = np.divmod(sup, shape[1])
    for s in np.flatnonzero(pk >= 0):
        pr, pc = divmod(int(pk[s]), shape[1])
        idx = sup[np.hypot(sr - pr, sc - pc) <= radius]
        sl[s], r[s] = _flux_fit(H, X, Y, idx)
    return sl, r


def shape_fit(H, X, Y, valid: np.ndarray, pk: np.ndarray) -> np.ndarray:
    """Per source, the movie-on-activity slope of the flux in the core
    (valid pixels <= 1.5 px from the peak pixel) over that of the flux
    1.5-3.5 px out (``shapeFit``): above 1 when the activity is broader
    than the movie's events, below 1 when more concentrated; scale-free.
    NaN with fewer than 5 core or 12 outer pixels, or a slope <= 0."""
    R, C = valid.shape
    dr, dc = np.mgrid[-3:4, -3:4]
    d = np.hypot(dr, dc).ravel()
    dr, dc = dr.ravel(), dc.ravel()
    rho = np.full(pk.size, np.nan)
    for s in np.flatnonzero(pk >= 0):
        pr, pc = divmod(int(pk[s]), C)
        rr, cc = pr + dr, pc + dc
        inside = (d <= 3.5) & (rr >= 0) & (rr < R) & (cc >= 0) & (cc < C)
        rr, cc, ds = rr[inside], cc[inside], d[inside]
        ok = valid[rr, cc]
        idx, ds = (rr * C + cc)[ok], ds[ok]
        core, outer = idx[ds <= 1.5], idx[ds > 1.5]
        if core.size < 5 or outer.size < 12:
            continue
        a = _flux_fit(H, X, Y, core)[0]
        b = _flux_fit(H, X, Y, outer)[0]
        if a > 0 and b > 0:
            rho[s] = a / b
    return rho


def _events(x: np.ndarray) -> np.ndarray:
    """Frames of a trace's local maxima above its 99th percentile, with 8
    frames before and 16 after inside the trace."""
    thr = matlab_prctile(x, 99)
    m = x[1:-1]
    ev = np.flatnonzero((m > thr) & (m >= x[:-2]) & (m > x[2:])) + 1
    return ev[(ev >= 8) & (ev <= x.size - 17)]


def event_residual(H, X, Y, pk):
    """The event-triggered residual at the sources' peak pixels as a
    fraction of the event-triggered movie, events picked on each source's
    own 80 Hz trace (its 99th-percentile local maxima). Near 0 when the
    activity is removed. Returns ``(fraction, events)``."""
    eta_y, eta_r, n_ev = np.zeros(25), np.zeros(25), 0
    ok = np.flatnonzero(pk >= 0)
    if ok.size == 0:
        return np.nan, 0
    A = np.asarray(H[pk[ok]] @ X)
    for i, s in enumerate(ok):
        y = Y[:, pk[s]].astype(np.float64)
        for e in _events(np.asarray(X[s], np.float64)):
            wy = y[e - 8 : e + 17]
            if np.isnan(wy).any():
                continue
            wr = wy - A[i, e - 8 : e + 17]
            eta_y += wy - wy[:6].mean()
            eta_r += wr - wr[:6].mean()
            n_ev += 1
    if n_ev == 0:
        return np.nan, 0
    return float(eta_r[8:13].max() / eta_y[8:13].max()), n_ev


@dataclass
class RingEventRatios:
    """Activity / movie event amplitudes near sources, events picked on
    the movie 3-5 px away (``ringEventRatios.m``)."""

    core: float = np.nan
    near: float = np.nan
    n_sources: int = 0
    n_events: int = 0
    n_artifact_sources: int = 0


RING_RAD = 8
RING_EDGE = 160


def _ring_events(sel: np.ndarray, discard: np.ndarray) -> list:
    """Ring events: local maxima of the high-passed ring signal above its
    99th percentile, away from the trial's ends and discarded frames."""
    filled = np.nan_to_num(sel, nan=np.nanmedian(sel))
    sel = sel - median_filter(filled, size=81, mode="nearest")
    thr = np.nanpercentile(sel, 99)
    T = sel.size
    return [
        e
        for e in range(RING_EDGE, T - RING_EDGE)
        if sel[e] > thr
        and sel[e] >= sel[e - 1]
        and sel[e] > sel[e + 1]
        and not discard[e - 8 : e + 17].any()
    ]


def _ring_source(movie, act_at, valid, box, discard, sums):
    """One source's ring events, added to ``sums`` (movie and activity
    amplitude per distance bin); returns the number of events (0 when the
    source does not count)."""
    rows, cols, dist = box
    v = valid[rows, cols]
    ring = v & (dist >= 3) & (dist <= 5)
    core = v & (dist <= 1.5)
    if ring.sum() < 12 or core.sum() < 5:
        return 0
    mr = np.asarray(movie[:, rows[ring], cols[ring]], np.float64)
    mr[discard] = np.nan
    sel = np.where(np.isnan(mr).any(1), np.nan, mr.mean(1))
    if np.isnan(sel).all():
        return 0
    ev = _ring_events(sel, discard)
    if len(ev) < 3:
        return 0
    flat = rows * valid.shape[1] + cols
    Mb = np.asarray(movie[:, rows, cols], np.float64)
    Ab = np.asarray(act_at(flat.ravel()), np.float64).reshape(Mb.shape)
    for e in ev:
        am = Mb[e - 1 : e + 2].mean(0) - Mb[e - 8 : e - 2].mean(0)
        aa = Ab[e - 1 : e + 2].mean(0) - Ab[e - 8 : e - 2].mean(0)
        okp = v & np.isfinite(am)
        for i, (lo, hi) in enumerate(((0.0, 1.5), (1.5, 2.5))):
            q = okp & (dist >= lo) & (dist < hi)
            sums[0, i] += am[q].sum()
            sums[1, i] += aa[q].sum()
    return len(ev)


def ring_event_ratios(
    movie: np.ndarray,
    act_at,
    mean_act: np.ndarray,
    valid: np.ndarray,
    pr: np.ndarray,
    pc: np.ndarray,
    discard: np.ndarray,
) -> RingEventRatios:
    """The selection-safe event ratios of one trial (cropped arrays).

    Parameters
    ----------
    movie : ndarray (frames, rows, cols)
        NaN where not imaged and at discarded frames.
    act_at : callable
        ``act_at(flat)`` is the activity at flat pixel indices,
        ``(frames, n)``.
    mean_act : ndarray (rows, cols)
        The activity's time-mean.
    valid : ndarray of bool (rows, cols)
    pr, pc : ndarray of int
        The sources' peak pixels (zero-based; negative: none).
    discard : ndarray of bool (frames,)

    Returns
    -------
    RingEventRatios
        As ``slap2_f0.ringqc.ring_event_ratios`` on the written arrays.
    """
    R, C = valid.shape
    discard = np.asarray(discard, bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mm = np.nanmean(movie, axis=0)
    with np.errstate(invalid="ignore"):
        bad = mean_act > 3 * np.maximum(mm, 1)  # NaN compares False
    dr, dc = np.mgrid[-RING_RAD : RING_RAD + 1, -RING_RAD : RING_RAD + 1]
    dist = np.hypot(dr, dc)
    sums = np.zeros((2, 2))
    out = RingEventRatios()
    for r0, c0 in zip(np.asarray(pr, int), np.asarray(pc, int)):
        if not (
            RING_RAD <= r0 < R - RING_RAD and RING_RAD <= c0 < C - RING_RAD
        ):
            continue
        if not valid[r0, c0]:
            continue
        rows, cols = r0 + dr, c0 + dc
        if bad[rows, cols][dist <= 4].any():
            out.n_artifact_sources += 1
            continue
        n = _ring_source(
            movie, act_at, valid, (rows, cols, dist), discard, sums
        )
        if n:
            out.n_sources += 1
            out.n_events += n
    if out.n_events:
        out.core = float(sums[1, 0] / sums[0, 0])
        out.near = float(sums[1, 1] / sums[0, 1])
    return out
