"""SILo's kinetics test: sources whose transients are faster than the sensor.

Port of GIAnT-MATLAB's ``sourceKinetics.m`` and ``rejectFastSources.m``
(main 5a0673b). Each source's strongest transients in its least-squares
trace (activity channel, pooled over trials) are averaged, and the average
one decay time constant after the peak, relative to the peak, is divided by
the sensor's ``exp(-1)``: about 1 for the sensor's kinetics, about 0 for
one- or two-frame transients. Sources scoring below ``min_kinetic_score``
are removed after extraction.

An evidence test can remove, with them, sources whose extracted activity is
too weak to tell from noise (:func:`weak_sources`; not in MATLAB, off by
default): meant for permissive seeding (GIAnT-Python#51), whose extra seeds
the kinetics test cannot judge, since a source with fewer than
:data:`MIN_EVENTS` events gets no score and is kept.

Traces are ``(sources, frames)`` per trial, NaN at discarded frames, as
SILo's ``E{t}.dF.ls(:, :, 1)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.signal import lfilter, peak_prominences

THRESH = 5.0
"""Event threshold, robust SDs of the filtered trace."""

MIN_EVENTS = 5
"""A source needs this many events to be scored."""


def movmean_omitnan(x: np.ndarray, window: int) -> np.ndarray:
    """MATLAB ``movmean(x, window, 2, 'omitnan')`` along the last axis.

    The window holds ``floor(window / 2)`` samples before and
    ``ceil(window / 2) - 1`` after each sample (centred for odd windows),
    shrunk at the ends; NaN are left out, and a window with no number gives
    NaN.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[-1]
    before, after = window // 2, (window - 1) // 2
    ok = ~np.isnan(x)
    zero = np.zeros(x.shape[:-1] + (1,))
    csum = np.concatenate([zero, np.cumsum(np.where(ok, x, 0.0), -1)], -1)
    ccnt = np.concatenate([zero, np.cumsum(ok, -1, dtype=np.float64)], -1)
    idx = np.arange(n)
    lo = np.clip(idx - before, 0, n)
    hi = np.clip(idx + after + 1, 0, n)
    total = csum[..., hi] - csum[..., lo]
    count = ccnt[..., hi] - ccnt[..., lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def _plateaus(y: np.ndarray) -> list:
    """Local maxima of one row as MATLAB's ``islocalmax`` finds them.

    A run of equal values is a maximum if both its neighbours are lower.
    The first and last samples are never maxima. Returns ``(first, last,
    mark)`` of each maximum's run, where ``mark`` is its centre (the left
    one of the two centres of a run of even length).
    """
    n = y.size
    if n < 3:
        return []
    change = np.flatnonzero(np.diff(y) != 0) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [n]]) - 1
    out = []
    for s, e in zip(starts, ends):
        if s == 0 or e == n - 1:
            continue
        if y[s - 1] < y[s] and y[e + 1] < y[e]:
            out.append((int(s), int(e), int((s + e) // 2)))
    return out


def islocalmax(y: np.ndarray, min_separation: float = 0) -> np.ndarray:
    """MATLAB ``islocalmax(y, 2, 'MinSeparation', s)`` for each row.

    With a minimum separation, maxima are taken from left to right; one
    closer than ``min_separation`` samples to the last maximum kept (from
    the end of that maximum's run of equal values to the start of its
    own) replaces it if it is more prominent and is dropped otherwise.
    This reproduces MATLAB R2024b on rows with and without runs of equal
    values (checked against MATLAB in the tests). Prominences are those of
    :func:`scipy.signal.peak_prominences` at the runs' centres: the height
    above the higher of the lowest points between the maximum and the
    nearest higher sample on each side.

    Parameters
    ----------
    y : ndarray of shape (rows, n) or (n,)
        Rows to search; NaN-free.
    min_separation : float
        Minimum distance in samples between maxima.

    Returns
    -------
    ndarray of bool, the shape of ``y``
    """
    y = np.asarray(y, dtype=np.float64)
    rows = np.atleast_2d(y)
    out = np.zeros(rows.shape, bool)
    for r, row in enumerate(rows):
        runs = _plateaus(row)
        if not runs:
            continue
        marks = np.array([m for _, _, m in runs])
        if min_separation <= 0 or len(runs) == 1:
            out[r, marks] = True
            continue
        prom = peak_prominences(row, marks)[0]
        kept = []  # indices into runs
        for i, (first, _, _) in enumerate(runs):
            if kept and first - runs[kept[-1]][1] < min_separation:
                if prom[i] > prom[kept[-1]]:
                    kept[-1] = i
                continue
            kept.append(i)
        out[r, marks[kept]] = True
    return out.reshape(y.shape)


def source_kinetics(
    traces: Sequence[Optional[np.ndarray]], tau_s: float, analyze_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    """Kinetics score of each source (``sourceKinetics.m``).

    Each trial's trace is high-passed (moving mean over 1 s or 10 tau,
    whichever is longer) and filtered with the sensor's decay (an
    anti-causal exponential). Events are maxima of the filtered trace above
    5 robust SDs (median absolute deviation over all trials), at least 2
    tau apart, each aligned on the largest high-passed sample within 0.6
    tau (at most 3 frames) of the maximum. Discarded (NaN) frames count as
    0 in the filtering; events whose peak or decay frame was discarded are
    skipped.

    Parameters
    ----------
    traces : sequence of (ndarray of shape (sources, frames) or None)
        Each trial's least-squares activity-channel traces, NaN at the
        discarded frames; None for trials without traces.
    tau_s : float
        Decay time constant of the sensor, seconds.
    analyze_hz : float
        Frame rate of the traces.

    Returns
    -------
    score : ndarray of shape (sources,)
        ``(avg(L) / avg(0)) / exp(-L / tau)`` with ``L = round(tau)``
        frames; NaN for sources with fewer than 5 events or when ``L`` is
        under 3 frames.
    n_events : ndarray of int, shape (sources,)
        Events averaged per source.
    """
    present = [
        np.asarray(t, dtype=np.float64) for t in traces if t is not None
    ]
    present = [t for t in present if t.size]
    n_src = present[0].shape[0] if present else 0
    score = np.full(n_src, np.nan)
    n_events = np.zeros(n_src, dtype=np.int64)
    tau = tau_s * analyze_hz
    lag = int(np.floor(tau + 0.5))
    if n_src == 0 or lag < 3:
        return score, n_events
    sep = max(1, int(np.floor(2 * tau + 0.5)))
    w = min(3, max(1, int(np.floor(0.6 * tau + 0.5))))
    hp_win = max(3, int(np.floor(max(analyze_hz, 10 * tau) + 0.5)))
    a = np.exp(-1.0 / tau)

    xs, ys, vs = [], [], []
    for t in present:
        valid = ~np.isnan(t)
        x = t - movmean_omitnan(t, hp_win)
        x[~valid] = 0.0
        y = lfilter([1 - a], [1, -a], x[:, ::-1], axis=1)[:, ::-1]
        xs.append(x)
        ys.append(y)
        vs.append(valid)

    sd = np.full(n_src, np.nan)
    for s in range(n_src):
        v = np.concatenate([y[s, valid[s]] for y, valid in zip(ys, vs)])
        if v.size > 10:
            sd[s] = 1.4826 * np.median(np.abs(v - np.median(v)))

    sum0 = np.zeros(n_src)
    sum_l = np.zeros(n_src)
    for x, y, valid in zip(xs, ys, vs):
        n_t = y.shape[1]
        yd = np.where(valid, y, 0.0)
        with np.errstate(invalid="ignore"):
            above = yd > THRESH * sd[:, None]
        for s in np.flatnonzero(above.any(axis=1)):
            peaks = islocalmax(yd[s], sep) & above[s]
            for p in np.flatnonzero(peaks):
                lo, hi = p - w, p + w
                if lo < 0 or hi > n_t - 1:
                    continue
                q = lo + int(np.argmax(x[s, lo : hi + 1]))
                if (
                    q + lag > n_t - 1
                    or not valid[s, q]
                    or not valid[s, q + lag]
                ):
                    continue
                sum0[s] += x[s, q]
                sum_l[s] += x[s, q + lag]
                n_events[s] += 1
    ok = (n_events >= MIN_EVENTS) & (sum0 > 0)
    score[ok] = (sum_l[ok] / sum0[ok]) / np.exp(-lag / tau)
    return score, n_events


@dataclass
class KineticsTest:
    """Result of the kinetics test (``rejectFastSources``).

    Attributes
    ----------
    keep : ndarray of bool, shape (sources,)
        Sources kept.
    score, n_events : ndarray, shape (sources,)
        Every source's score and event count.
    weak : ndarray of bool, shape (sources,), optional
        Sources the evidence test removed (:func:`weak_sources`); None when
        it was not run.
    snr : ndarray, shape (sources,), optional
        Every source's SNR averaged over trials, when the evidence test ran.
    """

    keep: np.ndarray
    score: np.ndarray
    n_events: np.ndarray
    weak: Optional[np.ndarray] = None
    snr: Optional[np.ndarray] = None

    @property
    def kept_score(self) -> np.ndarray:
        """Scores of the kept sources (MATLAB's ``kinetics.score``)."""
        return self.score[self.keep]

    @property
    def kept_n_events(self) -> np.ndarray:
        """Event counts of the kept sources (``kinetics.nEvents``)."""
        return self.n_events[self.keep]

    @property
    def rejected(self) -> np.ndarray:
        """Indices of the removed sources."""
        return np.flatnonzero(~self.keep)


def reject_fast_sources(
    traces: Sequence[Optional[np.ndarray]],
    n_sources: int,
    tau_s: float,
    analyze_hz: float,
    min_kinetic_score: float,
) -> KineticsTest:
    """Which sources the kinetics test keeps (``rejectFastSources.m``).

    Sources scoring below ``min_kinetic_score`` are removed when it is
    above 0; sources that cannot be scored (NaN) are kept. The removed
    sources were fitted together with the kept ones, so the kept sources'
    traces are those of the joint fit.

    Parameters
    ----------
    traces : sequence of (ndarray or None)
        As for :func:`source_kinetics`.
    n_sources : int
        Sources of the path.
    tau_s, analyze_hz : float
        As for :func:`source_kinetics`.
    min_kinetic_score : float
        Threshold (0 keeps every source).

    Returns
    -------
    KineticsTest
    """
    score, n_events = source_kinetics(traces, tau_s, analyze_hz)
    if score.size != n_sources:
        # no trial returned traces: nothing to score
        score = np.full(n_sources, np.nan)
        n_events = np.zeros(n_sources, dtype=np.int64)
    keep = np.ones(n_sources, bool)
    if min_kinetic_score > 0:
        with np.errstate(invalid="ignore"):
            keep = ~(score < min_kinetic_score)
    return KineticsTest(keep=keep, score=score, n_events=n_events)


def mean_snr(
    snrs: Sequence[Optional[np.ndarray]], n_sources: int
) -> np.ndarray:
    """Each source's SNR averaged over the trials that have one.

    ``snrs`` holds each trial's per-source SNR (the solver's: the SD of the
    denoised activity over its noise, ``TrialActivity.snr``), or None for a
    trial without sources; NaN where no trial has one. Only NaN is left
    out, as in the summary's ``SNR`` (an infinite SNR stays infinite).
    """
    total = np.zeros(n_sources)
    count = np.zeros(n_sources)
    for s in snrs:
        if s is None or np.size(s) != n_sources:
            continue
        s = np.asarray(s, dtype=np.float64)
        ok = ~np.isnan(s)
        total[ok] += s[ok]
        count[ok] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def weak_sources(
    snr: np.ndarray,
    n_events: np.ndarray,
    min_snr: float = 0.0,
    min_events: int = 0,
) -> np.ndarray:
    """Sources whose extracted activity is too weak (the evidence test).

    A source is weak when its SNR averaged over trials (:func:`mean_snr`) is
    below ``min_snr`` or when the kinetics test found fewer than
    ``min_events`` events in its trace (:func:`source_kinetics`); 0 turns
    either criterion off. A source without an SNR (no extracted trial) is
    not judged by it; the caller turns the event criterion off where the
    kinetics test could not count events (see
    :func:`~giant_python.extraction.raster.summary.kinetics_test`).

    Returns
    -------
    ndarray of bool, shape (sources,)
    """
    if min_snr < 0 or min_events < 0:
        raise ValueError("min_snr and min_events must be nonnegative")
    weak = np.zeros(np.shape(snr), bool)
    if min_snr > 0:
        with np.errstate(invalid="ignore"):
            weak |= np.asarray(snr, dtype=np.float64) < min_snr
    if min_events > 0:
        weak |= np.asarray(n_events) < min_events
    return weak
