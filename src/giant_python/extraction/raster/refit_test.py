"""Ground-truth refit test of the subproblem solver on real subproblems.

Port of the core of GIAnT-MATLAB's ``utils/subproblemRefitTest.m``. For
each subproblem:

1. Fit it and take the fit as the truth: baseline B, footprints H and
   activity X = S * k (the activity channel only).
2. Draw new data from the truth under the fit's own noise model, with the
   subproblem's variance multiplier F:
   ``Z = F * Poisson(T / F)``, ``T = max(B + H X, 0)``, so that Z has mean
   T and variance F T.
3. Fit Z again and compare the refit with the truth.

Amplitudes are in the solver's normalized units at each source's peak
pixel. Events are the peaks of each source's true X (see
:func:`find_events`); an event is found when the refit's activity mass in
its window is above a quarter of the truth's.

Limits, as in MATLAB: the truth uses the fit's own footprints and kernel,
so the test cannot show a footprint-width or kinetics mismatch; noise is
drawn independently per sample. MATLAB's event injection is not ported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .solver import (
    SolverOptions,
    SubproblemResult,
    conv_same,
    solve_subproblems,
)
from .solver_setup import SolverParams, Subproblem

SIZE_EDGES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, math.inf)


def find_events(x0: np.ndarray, lag: int) -> np.ndarray:
    """Event peaks of one source's true activity (zero-based frames).

    Local maxima above 2% of the largest, with no other such maximum in
    the ``lag`` frames before them (a maximum that follows a skipped one
    closely is skipped too), not within 5 frames of the start or
    ``lag + 3`` of the end.
    """
    n = x0.size
    if n < 3 or not np.any(x0 > 0):
        return np.zeros(0, dtype=np.int64)
    mid = x0[1:-1]
    is_peak = (mid > 0.02 * x0.max()) & (mid >= x0[:-2]) & (mid > x0[2:])
    pk = np.nonzero(is_peak)[0] + 2  # MATLAB's one-based frames
    pk = pk[(pk > 5) & (pk < n - lag - 3)]
    keep = np.ones(pk.size, dtype=bool)
    keep[1:] = np.diff(pk) >= lag
    return pk[keep] - 1


@dataclass
class RefitTruth:
    """The truth of one subproblem's refit test and the redrawn data.

    Attributes
    ----------
    h : ndarray of shape (pixels, sources)
    x : ndarray of shape (sources, frames)
        True activity (events convolved with the kernel, scaled).
    b : ndarray of shape (pixels, frames)
    f0 : ndarray of shape (sources, frames)
    z : ndarray of shape (pixels, frames)
        The redrawn data.
    """

    h: np.ndarray
    x: np.ndarray
    b: np.ndarray
    f0: np.ndarray
    z: np.ndarray


def draw_refit_data(
    sub: Subproblem,
    truth: SubproblemResult,
    params: SolverParams,
    rng: np.random.Generator,
    truth_scale: float = 1.0,
) -> RefitTruth:
    """Redraw a subproblem's data from a fit of it (steps 1 and 2).

    ``truth`` must hold its baseline (``SolverOptions.keep_baseline``).
    ``truth_scale`` multiplies the true activity, to map the detection
    threshold.
    """
    if truth.baseline is None:
        raise ValueError(
            "the truth needs its baseline: fit it with "
            "SolverOptions(keep_baseline=True)"
        )
    f = np.asarray(sub.finv, dtype=np.float64)
    x = conv_same(truth_scale * truth.events[0], params.k)
    b = truth.baseline[0]
    t = np.maximum(b + truth.footprints @ x, 0)
    z = f * rng.poisson(t / f)
    return RefitTruth(h=truth.footprints, x=x, b=b, f0=truth.f0[0], z=z)


def score_refit(
    truth: RefitTruth,
    x1: np.ndarray,
    f01: np.ndarray,
    ls1: np.ndarray,
    tau_frames: float,
) -> tuple[list[dict], list[dict]]:
    """Compare a refit with the truth (step 3), per source and per event.

    Parameters
    ----------
    truth : RefitTruth
    x1, f01, ls1 : ndarray of shape (sources, frames)
        The refit's denoised activity, F0 and least-squares activity.
    tau_frames : float
        Decay time constant; event windows run from 2 frames before the
        peak to ``ceil(2.5 tau)`` after.

    Returns
    -------
    sources : list of dict
        ``true_mass``, ``recovered`` (refit / true activity mass),
        ``slope`` (of the refit on the truth, through 0), ``f0_ratio``,
        ``f0_ratio_active`` (where the truth is above 20% of its
        largest), ``ls_recovered``, ``peak_h`` and ``baseline`` (the true
        baseline under the footprint, per pixel and frame).
    events : list of dict
        ``source``, ``size`` (peak-pixel amplitude), ``true_mass``,
        ``refit_mass`` and ``found``.
    """
    lag = math.ceil(2.5 * tau_frames)
    peak_h = truth.h.max(0)
    b_mean = truth.b.mean(1)
    sources, events = [], []
    nan = float("nan")
    for s in range(truth.x.shape[0]):
        x0, xr = truth.x[s], x1[s]
        active = x0 > 0.2 * x0.max()
        mass = float(x0.sum())
        row = dict(
            source=s,
            true_mass=mass,
            recovered=nan,
            slope=nan,
            ls_recovered=nan,
            f0_ratio=float(f01[s].mean() / truth.f0[s].mean()),
            f0_ratio_active=nan,
            peak_h=float(peak_h[s]),
            baseline=float(truth.h[:, s] @ b_mean / truth.h[:, s].sum()),
        )
        if mass > 0:
            row["recovered"] = float(xr.sum() / mass)
            row["slope"] = float((xr * x0).sum() / (x0 * x0).sum())
            row["ls_recovered"] = float(ls1[s].sum() / mass)
        if active.any():
            row["f0_ratio_active"] = float(
                f01[s][active].mean() / truth.f0[s][active].mean()
            )
        sources.append(row)
        for e in find_events(x0, lag):
            win = slice(e - 2, e + lag + 1)
            m0, m1 = float(x0[win].sum()), float(xr[win].sum())
            events.append(
                dict(
                    source=s,
                    size=float(x0[e] * peak_h[s]),
                    true_mass=m0,
                    refit_mass=m1,
                    found=bool(m1 > 0.25 * m0),
                )
            )
    return sources, events


def by_size(events: list[dict], edges=SIZE_EDGES) -> list[dict]:
    """Recovery by true event size.

    One row per size bin with events: ``n``, ``share_found``,
    ``recovered_all`` (refit / true mass over the bin's events) and
    ``recovered_found`` (over its found events; NaN if none is found).
    """
    size = np.array([e["size"] for e in events])
    m0 = np.array([e["true_mass"] for e in events])
    m1 = np.array([e["refit_mass"] for e in events])
    found = np.array([e["found"] for e in events], dtype=bool)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (size >= lo) & (size < hi)
        if not m.any():
            continue
        f = m & found
        rows.append(
            dict(
                size_from=lo,
                size_to=hi,
                n=int(m.sum()),
                share_found=float(f.sum() / m.sum()),
                recovered_all=float(m1[m].sum() / m0[m].sum()),
                recovered_found=(
                    float(m1[f].sum() / m0[f].sum())
                    if f.any()
                    else float("nan")
                ),
            )
        )
    return rows


def summarize(sources: list[dict], events: list[dict]) -> dict:
    """Medians and quartiles over sources, and the pooled recovery."""

    def col(name):
        """A column of the sources' rows."""
        return np.array([s[name] for s in sources], dtype=np.float64)

    def q(name, pct):
        """A percentile of a column, NaN left out."""
        v = col(name)
        v = v[np.isfinite(v)]
        return float(np.percentile(v, pct)) if v.size else float("nan")

    m0 = sum(e["true_mass"] for e in events)
    m1 = sum(e["refit_mass"] for e in events)
    return dict(
        n_sources=len(sources),
        n_events=len(events),
        recovered=q("recovered", 50),
        recovered_q1=q("recovered", 25),
        recovered_q3=q("recovered", 75),
        pooled=float(m1 / m0) if m0 > 0 else float("nan"),
        slope=q("slope", 50),
        slope_q1=q("slope", 25),
        slope_q3=q("slope", 75),
        f0_ratio=q("f0_ratio", 50),
        f0_ratio_active=q("f0_ratio_active", 50),
        ls_recovered=q("ls_recovered", 50),
    )


@dataclass
class RefitResult:
    """Results of :func:`refit_test`.

    Attributes
    ----------
    sources, events : list of dict
        Rows of :func:`score_refit`, with the subproblem's index added.
    summary : dict
        :func:`summarize` of them.
    by_size : list of dict
        :func:`by_size` of the events.
    truths : list of RefitTruth
        Each subproblem's truth and redrawn data, if kept.
    """

    sources: list
    events: list
    summary: dict
    by_size: list
    truths: list


def refit_test(
    subs: list[Subproblem],
    params: SolverParams,
    options: SolverOptions | None = None,
    *,
    refit_params: SolverParams | None = None,
    truth_scale: float = 1.0,
    seed: int = 0,
    keep_truths: bool = False,
) -> RefitResult:
    """Run the refit test on subproblems.

    Parameters
    ----------
    subs : list of Subproblem
        Their activity channel is used; a second channel is left out.
    params : SolverParams
        Parameters of the truth fit.
    options : SolverOptions, optional
    refit_params : SolverParams, optional
        Parameters of the refit, if they differ from the truth's.
    truth_scale : float
        Multiplies the true activity.
    seed : int
        Subproblem i is redrawn with ``numpy.random.default_rng(seed + i)``.
    keep_truths : bool
        Return each subproblem's truth and redrawn data.
    """
    opt = replace(options or SolverOptions(), keep_baseline=True)
    refit_params = refit_params or params
    one = [replace(sb, y=sb.y[:1], _coords=None) for sb in subs]
    fits = solve_subproblems(one, params, opt)
    truths = [
        draw_refit_data(
            sb, fit, params, np.random.default_rng(seed + i), truth_scale
        )
        for i, (sb, fit) in enumerate(zip(one, fits))
    ]
    again = [
        replace(sb, y=tr.z[None], _coords=None) for sb, tr in zip(one, truths)
    ]
    refits = solve_subproblems(again, refit_params, opt)
    sources, events = [], []
    for i, (tr, fit) in enumerate(zip(truths, refits)):
        x1 = conv_same(fit.events[0], refit_params.k)
        src, ev = score_refit(tr, x1, fit.f0[0], fit.ls[0], params.tau_frames)
        sources += [dict(subproblem=i, **row) for row in src]
        events += [dict(subproblem=i, **row) for row in ev]
    return RefitResult(
        sources=sources,
        events=events,
        summary=summarize(sources, events),
        by_size=by_size(events),
        truths=truths if keep_truths else [],
    )
