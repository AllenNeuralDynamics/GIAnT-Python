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

**Baseline sensitivity** (the WP6 test of the SILo port). The refit can be
given an external baseline (``Subproblem.baseline``, for
``SolverParams.baseline_mode`` other than 'fit'): the true baseline with
an error of known type and size (``given_baseline``; :func:`scale_error`,
:func:`drift_error`, :func:`blur_error`, :func:`edge_error`), or the
fit's baseline while the truth carries a fast component shared by the
pixels that the given baseline lacks (``truth_baseline``:
:func:`shared_component`). That gives the event-amplitude and detection
error per unit of baseline error, for each type.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy import ndimage

from .solver import (
    SolverOptions,
    SubproblemResult,
    conv_same,
    solve_subproblems,
)
from .solver_baseline import source_f0
from .solver_setup import SolverParams, Subproblem, sel_coords

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
    b_fit : ndarray of shape (pixels, frames) or None
        The fit's baseline, when the truth's was changed from it
        (``truth_baseline``).
    """

    h: np.ndarray
    x: np.ndarray
    b: np.ndarray
    f0: np.ndarray
    z: np.ndarray
    b_fit: np.ndarray | None = None


def draw_refit_data(
    sub: Subproblem,
    truth: SubproblemResult,
    params: SolverParams,
    rng: np.random.Generator,
    truth_scale: float = 1.0,
    truth_baseline=None,
    baseline_rng: np.random.Generator | None = None,
) -> RefitTruth:
    """Redraw a subproblem's data from a fit of it (steps 1 and 2).

    ``truth`` must hold its baseline (``SolverOptions.keep_baseline``).
    ``truth_scale`` multiplies the true activity, to map the detection
    threshold. ``truth_baseline(b, sub, rng)``, if given, changes the true
    baseline before the data are drawn (with ``baseline_rng``, so that the
    data's draws stay those of ``rng``); the true F0 follows it.
    """
    if truth.baseline is None:
        raise ValueError(
            "the truth needs its baseline: fit it with "
            "SolverOptions(keep_baseline=True)"
        )
    f = np.asarray(sub.finv, dtype=np.float64)
    x = conv_same(truth_scale * truth.events[0], params.k)
    b = truth.baseline[0]
    f0 = truth.f0[0]
    b_fit = None
    if truth_baseline is not None:
        b_fit = b
        b = np.asarray(
            truth_baseline(b, sub, baseline_rng or np.random.default_rng()),
            dtype=np.float64,
        )
        if b.shape != b_fit.shape:
            raise ValueError("truth_baseline changed the baseline's shape")
        f0 = source_f0(truth.footprints, b)
    t = np.maximum(b + truth.footprints @ x, 0)
    z = f * rng.poisson(t / f)
    return RefitTruth(h=truth.footprints, x=x, b=b, f0=f0, z=z, b_fit=b_fit)


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
    fits: list[SubproblemResult] | None = None,
    truth_baseline=None,
    given_baseline=None,
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
    fits : list of SubproblemResult, optional
        The truth fits (with their baselines), if already made with
        ``params``; they are not fitted again.
    truth_baseline : callable, optional
        ``truth_baseline(b, sub, rng) -> b'``: a change of the true
        baseline before the data are drawn (:func:`shared_component`); its
        generator is ``default_rng((seed + i, 1))``.
    given_baseline : callable, optional
        ``given_baseline(truth, sub, rng) -> b``: the external baseline the
        refit gets (``Subproblem.baseline``, for a ``baseline_mode`` of
        ``refit_params`` that uses one), e.g. the true baseline with an
        error (:func:`scale_error` etc.); its generator is
        ``default_rng((seed + i, 2))``. Default with ``truth_baseline``:
        the fit's baseline, without the change.
    """
    opt = replace(options or SolverOptions(), keep_baseline=True)
    refit_params = refit_params or params
    one = [replace(sb, y=sb.y[:1], _coords=None) for sb in subs]
    if fits is None:
        fits = solve_subproblems(one, params, opt)
    elif len(fits) != len(one):
        raise ValueError(f"{len(fits)} fits for {len(one)} subproblems")
    truths = [
        draw_refit_data(
            sb,
            fit,
            params,
            np.random.default_rng(seed + i),
            truth_scale,
            truth_baseline,
            np.random.default_rng((seed + i, 1)),
        )
        for i, (sb, fit) in enumerate(zip(one, fits))
    ]
    if given_baseline is None and truth_baseline is not None:
        given_baseline = fit_baseline_given
    given = [None] * len(one)
    if given_baseline is not None:
        given = [
            np.asarray(
                given_baseline(tr, sb, np.random.default_rng((seed + i, 2))),
                dtype=np.float64,
            )
            for i, (sb, tr) in enumerate(zip(one, truths))
        ]
    again = [
        replace(sb, y=tr.z[None], baseline=g, _coords=None)
        for sb, tr, g in zip(one, truths, given)
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


def fit_baseline_given(truth: RefitTruth, sub: Subproblem, rng) -> np.ndarray:
    """``given_baseline``: the fit's baseline, before ``truth_baseline``
    changed it (the true baseline when nothing changed it)."""
    return truth.b if truth.b_fit is None else truth.b_fit


def scale_error(scale: float):
    """``given_baseline``: the true baseline times ``1 + scale``."""

    def given(truth, sub, rng):
        """The scaled true baseline."""
        return truth.b * (1 + scale)

    return given


def drift_error(drift: float):
    """``given_baseline``: the true baseline times a linear drift from
    ``1 - drift / 2`` at the first frame to ``1 + drift / 2`` at the
    last."""

    def given(truth, sub, rng):
        """The drifting true baseline."""
        n = truth.b.shape[1]
        ramp = np.linspace(-0.5, 0.5, n) if n > 1 else np.zeros(1)
        return truth.b * (1 + drift * ramp)[None, :]

    return given


def edge_error(error: float, frames: int):
    """``given_baseline``: the true baseline times ``1 + error`` in the
    first and last ``frames`` frames (a trial-edge error)."""

    def given(truth, sub, rng):
        """The true baseline with its ends scaled."""
        b = truth.b.copy()
        m = min(int(frames), b.shape[1])
        b[:, :m] *= 1 + error
        b[:, b.shape[1] - m :] *= 1 + error
        return b

    return given


def blur_error(sigma: float):
    """``given_baseline``: the true baseline blurred in space.

    A Gaussian of ``sigma`` pixels over the subproblem's crop, divided by
    the blurred map of the selected pixels, so that a pixel near the edge
    of the selection averages over selected pixels only.
    """

    def given(truth, sub, rng):
        """The blurred true baseline."""
        rows, cols = sel_coords(sub.sel)
        mask = np.zeros(sub.sel.shape)
        mask[rows, cols] = 1
        norm = ndimage.gaussian_filter(mask, sigma, mode="constant")
        norm = norm[rows, cols]
        img = np.zeros(sub.sel.shape + (truth.b.shape[1],))
        img[rows, cols] = truth.b
        blurred = ndimage.gaussian_filter(
            img, (sigma, sigma, 0), mode="constant"
        )
        return blurred[rows, cols] / norm[:, None]

    return given


def shared_component(rms: float, smooth_frames: int = 5):
    """``truth_baseline``: a fast change shared by all pixels.

    The true baseline times ``1 + rms * n(t)``, ``n`` white noise smoothed
    by a moving mean of ``smooth_frames`` frames and scaled to a root mean
    square of 1 (25 ms at 200 Hz by default: faster than the network's 80 Hz
    frames resolve).
    """

    def changed(b, sub, rng):
        """The baseline with the shared component."""
        n = b.shape[1]
        noise = rng.standard_normal(n + smooth_frames - 1)
        kern = np.ones(smooth_frames) / smooth_frames
        course = np.convolve(noise, kern, mode="valid")
        course /= np.sqrt(np.mean(course**2))
        return b * (1 + rms * course)[None, :]

    return changed
