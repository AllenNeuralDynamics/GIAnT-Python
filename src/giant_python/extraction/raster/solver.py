"""SiLo's subproblem solver: sources' footprints, activity and baseline.

Port of ``extractSources`` in GIAnT-MATLAB's ``extractTrial.m`` (main
5a0673b), MATLAB's algorithm unchanged: starting footprints and baseline,
then ``nmf_iter`` rounds of the activity fit, the baseline fit and the
footprint fit, the debias refit of the events found, least squares,
channel 2 on the same footprints, and each source's baseline F0.

The difference from MATLAB is in how far the iterative fits go. MATLAB's
fmincon runs stop at iteration caps; here the activity and footprint
fits, which are convex, are solved to convergence
(:mod:`~giant_python.extraction.raster.solver_fits`).
``SolverOptions.max_iter`` and ``fit_max_iter`` cap them, for diagnosing
differences.

Subproblems with the same number of frames are fitted together, padded to
a common size, on the CPU or a GPU.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .localization import matlab_prctile
from .solver_baseline import (
    activity_noise,
    compute_floor,
    fit_baseline,
    initial_baseline,
    source_f0,
    surround_operators,
)
from .solver_fits import TimeKernel, fit_activity, fit_footprints
from .solver_setup import (
    SolverParams,
    Subproblem,
    conv_operator,
    initial_footprints,
)

# events below this are dropped before the debias refit (MATLAB: 1e-1)
_EVENT_FLOOR = 0.1


@dataclass(frozen=True)
class SolverOptions:
    """How the iterative fits are run.

    Attributes
    ----------
    device : str
        Torch device of the fits ('cpu', 'cuda', ...).
    dtype : {'float64', 'float32'}
        Precision of the fits. float64 for parity tests; float32 is
        faster on a GPU, with the loss rounded at about 1e-7.
    max_iter : int
        Cap on the Newton iterations of each fit. The fits converge in
        about ten; a small cap imitates MATLAB's unconverged fits.
    fit_max_iter : dict
        Caps for single fits, by name: 'activity1', 'footprints1',
        'activity2', ... (one per outer iteration), 'debias',
        'activity_ch2', 'debias_ch2'. A cap of 0 leaves a fit at its
        start. These are Newton iterations, each far stronger than one of
        MATLAB's fmincon iterations, so MATLAB's caps (10 per outer
        iteration, 15 for channel 2) do not carry over.
    ftol : float or None
        Convergence tolerance of the fits (see
        :func:`~giant_python.extraction.raster.solver_fits.projected_newton`).
    cg_iter : int
        Cap on the conjugate-gradient iterations of a Newton step.
    batch_bytes : float
        Subproblems are fitted together while their padded data stay
        below this size.
    keep_baseline : bool
        Return each subproblem's baseline (pixels x frames per channel).
    """

    device: str = "cpu"
    dtype: str = "float64"
    max_iter: int = 100
    fit_max_iter: dict = field(default_factory=dict)
    ftol: float | None = None
    cg_iter: int = 100
    batch_bytes: float = 2e8
    keep_baseline: bool = False

    def fit_options(self, name: str) -> dict:
        """Keyword arguments of the fit called ``name``."""
        return dict(
            max_iter=self.fit_max_iter.get(name, self.max_iter),
            ftol=self.ftol,
            cg_iter=self.cg_iter,
        )


@dataclass
class SubproblemResult:
    """Outputs of one subproblem, in the solver's normalized units.

    Attributes
    ----------
    footprints : ndarray of shape (pixels, sources)
        H; each footprint sums to 1.
    events : ndarray of shape (channels, sources, frames)
        S; the denoised activity is its convolution with the kernel.
    ls : ndarray of shape (channels, sources, frames)
        Least-squares activity.
    f0 : ndarray of shape (channels, sources, frames)
        Footprint-weighted baseline of each source.
    snr : ndarray of shape (sources,)
    baseline : ndarray of shape (channels, pixels, frames) or None
    info : dict
        Per fit: iterations, loss, convergence and seconds.
    """

    footprints: np.ndarray
    events: np.ndarray
    ls: np.ndarray
    f0: np.ndarray
    snr: np.ndarray
    baseline: np.ndarray | None = None
    info: dict = field(default_factory=dict)


def estimate_photon_scale(y: np.ndarray, finv: np.ndarray) -> float:
    """``extractTrial``'s photon scale for recordings without one.

    From the freshness-weighted variance and mean of each pixel of the
    activity channel (pixels x frames): the 10th percentile, over pixels
    of middling brightness, of the variance above that of dim pixels per
    unit of mean.
    """
    y = np.asarray(y, dtype=np.float64)
    w = 1.0 / np.asarray(finv, dtype=np.float64)
    w = np.where(np.isnan(y), 0.0, w)
    y0 = np.where(np.isnan(y), 0.0, y)
    wsum = w.sum(1)
    mean = (w * y0).sum(1) / wsum
    var = (w * (y0 - mean[:, None]) ** 2).sum(1) / wsum
    vb = matlab_prctile(var, 5)
    bright = (mean > matlab_prctile(mean, 40)) & (
        mean < matlab_prctile(mean, 90)
    )
    return matlab_prctile((var[bright] - vb) / mean[bright], 10)


def _lstsq(h: np.ndarray, r: np.ndarray) -> np.ndarray:
    """MATLAB ``h \\ r`` for a tall full-rank h."""
    return np.linalg.lstsq(h, r, rcond=None)[0]


class _Batch:
    """Subproblems padded to a common size, as torch tensors."""

    def __init__(self, subs, params: SolverParams, opt: SolverOptions):
        """Sizes, the sources' mask and the weights 1 / F."""
        self.subs, self.params, self.opt = subs, params, opt
        self.n = len(subs)
        self.dev = torch.device(opt.device)
        self.dt = getattr(torch, opt.dtype)
        self.n_px = [s.n_pixels for s in subs]
        self.n_src = [s.n_sources for s in subs]
        self.p_max, self.s_max = max(self.n_px), max(self.n_src)
        self.n_t = subs[0].n_frames
        self.src_ok = torch.zeros(
            (self.n, self.s_max, 1), dtype=torch.bool, device=self.dev
        )
        for i, n in enumerate(self.n_src):
            self.src_ok[i, :n] = True
        self.w = self.pixels_frames(
            [1.0 / np.asarray(s.finv, np.float64) for s in subs], 0.0
        )

    def tensor(self, a: np.ndarray) -> torch.Tensor:
        """An array on the fits' device, in their precision."""
        return torch.as_tensor(a, dtype=self.dt, device=self.dev)

    def pixels_frames(self, arrays, fill: float) -> torch.Tensor:
        """(batch, pixels, frames) from per-member pixels x frames."""
        out = torch.full(
            (self.n, self.p_max, self.n_t),
            fill,
            dtype=self.dt,
            device=self.dev,
        )
        for i, a in enumerate(arrays):
            out[i, : a.shape[0]] = self.tensor(a)
        return out

    def pixels_sources(self, arrays, dtype=None) -> torch.Tensor:
        """(batch, pixels, sources) from per-member pixels x sources."""
        out = torch.zeros(
            (self.n, self.p_max, self.s_max),
            dtype=dtype or self.dt,
            device=self.dev,
        )
        for i, a in enumerate(arrays):
            out[i, : a.shape[0], : a.shape[1]] = torch.as_tensor(
                a, dtype=out.dtype, device=self.dev
            )
        return out

    def sources_frames(self, arrays) -> torch.Tensor:
        """(batch, sources, frames) from per-member sources x frames."""
        out = torch.zeros(
            (self.n, self.s_max, self.n_t), dtype=self.dt, device=self.dev
        )
        for i, a in enumerate(arrays):
            out[i, : a.shape[0]] = self.tensor(a)
        return out

    def member(self, t: torch.Tensor, i: int, rows: int, cols=None):
        """Member i of a padded tensor, unpadded, as float64 numpy."""
        a = t[i, :rows] if cols is None else t[i, :rows, :cols]
        return a.detach().to("cpu", torch.float64).numpy()


def _note(info, name, fit, seconds):
    """Record how a fit ended, per member."""
    for i, d in enumerate(info):
        d[name] = dict(
            iterations=int(fit.iterations[i]),
            loss=float(fit.loss[i]),
            optimality=float(fit.optimality[i]),
            converged=bool(fit.converged[i]),
            seconds=seconds / len(info),
        )


def _debias(bt, z, h, b, k, s, info, name):
    """The debias refit: the events found, refitted without their L1."""
    p = bt.params
    free = s >= _EVENT_FLOOR
    s = torch.where(free, s, torch.zeros_like(s))
    if not bool(free.any()):
        return s
    s0 = None
    if p.debias_variance == "fixed":
        s0 = h @ TimeKernel(k).conv(s) + b + 1
    t0 = time.perf_counter()
    s, fit = fit_activity(
        z,
        h,
        b,
        k,
        bt.w,
        p.lam * p.phi,
        s,
        s0=s0,
        free=free,
        **bt.opt.fit_options(name),
    )
    _note(info, name, fit, time.perf_counter() - t0)
    return s


def _solve_batch(subs, params: SolverParams, opt: SolverOptions):  # noqa: C901
    """``extractSources`` for subproblems with equal frames and channels."""
    bt = _Batch(subs, params, opt)
    p = params
    n_ch = subs[0].y.shape[0]
    info = [dict() for _ in subs]
    y1 = [np.asarray(s.y[0], np.float64) for s in subs]
    finv = [np.asarray(s.finv, np.float64) for s in subs]
    k = bt.tensor(p.k)
    kern = TimeKernel(k)

    # starting values
    t0 = time.perf_counter()
    init = [initial_footprints(s.sel, s.sources_rc, p.sigma_px) for s in subs]
    b_np = [initial_baseline(y, p) for y in y1]
    k_peak = p.k[p.k.size // 2]
    s_np = []
    for (_, h0, _), y, b0 in zip(init, y1, b_np):
        ls = _lstsq(h0, y - b0)
        prev = np.concatenate([np.zeros((ls.shape[0], 1)), ls[:, :-1]], 1)
        s_np.append(np.fmax(0, ls - prev) / k_peak)
    z = bt.pixels_frames(y1, 0.0)
    b = bt.pixels_frames(b_np, 1.0)
    hs = bt.pixels_sources([i[0] for i in init])
    h = bt.pixels_sources([i[1] for i in init])
    valid = bt.pixels_sources([i[2] for i in init], dtype=torch.bool)
    s = bt.sources_frames(s_np)
    g_np = [conv_operator(sb.sel, p.hfilter) for sb in subs]
    g = torch.zeros((bt.n, bt.p_max, bt.p_max), dtype=bt.dt, device=bt.dev)
    for i, gi in enumerate(g_np):
        g[i, : gi.shape[0], : gi.shape[0]] = bt.tensor(gi.toarray())
    operators = [surround_operators(sb.sel, p) for sb in subs]
    for d in info:
        d["init_seconds"] = (time.perf_counter() - t0) / bt.n

    snr = [None] * bt.n
    for outer in range(1, p.nmf_iter + 1):
        t0 = time.perf_counter()
        free = bt.src_ok.expand_as(s)
        name = f"activity{outer}"
        opts = opt.fit_options(name)
        s, fit = fit_activity(z, h, b, k, bt.w, p.lam, s, free=free, **opts)
        _note(info, name, fit, time.perf_counter() - t0)
        x = kern.conv(s)
        if outer == p.nmf_iter:
            break
        t0 = time.perf_counter()
        x_thr = []
        for i in range(bt.n):
            npx, nsrc = bt.n_px[i], bt.n_src[i]
            xi = bt.member(x, i, nsrc)
            hi = bt.member(h, i, npx, nsrc)
            _, noise, _ = activity_noise(
                y1[i], finv[i], b_np[i], hi, xi, p.tau_frames
            )
            floor = compute_floor(xi, p.denoise_window, p.baseline_window)
            xt = np.fmax(0, xi - floor - noise[:, None])
            b_np[i] = fit_baseline(
                b_np[i], y1[i], xt, hi, finv[i], subs[i].sel, p, operators[i]
            )
            x_thr.append(xt)
        b = bt.pixels_frames(b_np, 1.0)
        for d in info:
            d[f"baseline{outer}_seconds"] = (time.perf_counter() - t0) / bt.n
        t0 = time.perf_counter()
        name = f"footprints{outer}"
        opts = opt.fit_options(name)
        x_t = bt.sources_frames(x_thr)
        s0 = h @ x_t + b + 1 if p.footprint_variance == "fixed" else None
        hs, fit = fit_footprints(z, x_t, b, g, bt.w, hs, valid, s0=s0, **opts)
        _note(info, name, fit, time.perf_counter() - t0)
        h = g @ hs
        norm = h.sum(1, keepdim=True)
        norm = torch.where(norm > 0, norm, torch.ones_like(norm))
        h, hs = h / norm, hs / norm
        s = s * norm.transpose(1, 2)

    s = _debias(bt, z, h, b, k, s, info, "debias")
    x = kern.conv(s)

    results = []
    y2 = b2_np = ls2 = None
    if n_ch > 1:
        y2 = [np.asarray(sb.y[1], np.float64) for sb in subs]
        b2_np = [initial_baseline(y, p) for y in y2]
    h_np = [bt.member(h, i, bt.n_px[i], bt.n_src[i]) for i in range(bt.n)]
    ls1 = []
    for i in range(bt.n):
        xi = bt.member(x, i, bt.n_src[i])
        _, _, snr[i] = activity_noise(
            y1[i], finv[i], b_np[i], h_np[i], xi, p.tau_frames
        )
        snr[i] = np.real(snr[i])
        ls1.append(_lstsq(h_np[i], y1[i] - b_np[i]))

    s2 = None
    if n_ch > 1:
        # channel 2 on the same footprints, from its own starting baseline
        k2 = bt.tensor(p.k2)
        ls2 = [_lstsq(hh, y - bb) for hh, y, bb in zip(h_np, y2, b2_np)]
        z2 = bt.pixels_frames(y2, 0.0)
        b2 = bt.pixels_frames(b2_np, 1.0)
        start = bt.sources_frames(ls2)
        t0 = time.perf_counter()
        free2 = bt.src_ok.expand_as(start)
        opts = opt.fit_options("activity_ch2")
        s2, fit = fit_activity(
            z2, h, b2, k2, bt.w, p.lam, start, free=free2, **opts
        )
        _note(info, "activity_ch2", fit, time.perf_counter() - t0)
        s2 = _debias(bt, z2, h, b2, k2, s2, info, "debias_ch2")

    for i in range(bt.n):
        nsrc = bt.n_src[i]
        ev = [bt.member(s, i, nsrc)]
        ls = [ls1[i]]
        base = [b_np[i]]
        if n_ch > 1:
            ev.append(bt.member(s2, i, nsrc))
            ls.append(ls2[i])
            base.append(b2_np[i])
        results.append(
            SubproblemResult(
                footprints=h_np[i],
                events=np.stack(ev),
                ls=np.stack(ls),
                f0=np.stack([source_f0(h_np[i], bb) for bb in base]),
                snr=snr[i],
                baseline=np.stack(base) if opt.keep_baseline else None,
                info=info[i],
            )
        )
    return results


def solve_subproblems(
    subs: list[Subproblem],
    params: SolverParams,
    options: SolverOptions | None = None,
) -> list[SubproblemResult]:
    """Solve subproblems, batching those of equal length.

    Subproblems with the same number of frames and channels are padded to
    a common number of pixels and sources and fitted together, in batches
    of at most ``options.batch_bytes`` of padded data, formed in order of
    size so that little is padded.
    """
    opt = options or SolverOptions()
    itemsize = 8 if opt.dtype == "float64" else 4
    results: list = [None] * len(subs)
    groups: dict = {}
    for i, sb in enumerate(subs):
        groups.setdefault((sb.n_frames, sb.y.shape[0]), []).append(i)
    for idx in groups.values():
        idx = sorted(idx, key=lambda i: (subs[i].n_pixels, subs[i].n_sources))
        start = 0
        while start < len(idx):
            stop = start + 1
            while stop < len(idx):
                size = (
                    (stop - start + 1)
                    * subs[idx[stop]].n_pixels
                    * subs[idx[stop]].n_frames
                    * itemsize
                )
                if size > opt.batch_bytes:
                    break
                stop += 1
            chunk = idx[start:stop]
            out = _solve_batch([subs[i] for i in chunk], params, opt)
            for i, r in zip(chunk, out):
                results[i] = r
            start = stop
    return results


def solve_subproblem(
    sub: Subproblem,
    params: SolverParams,
    options: SolverOptions | None = None,
) -> SubproblemResult:
    """Solve one subproblem (MATLAB's ``extractSubproblem``)."""
    return solve_subproblems([sub], params, options)[0]


@dataclass
class TrialResult:
    """A trial's sources, assembled from its subproblems.

    In movie units (the solver's outputs times the photon scale), as
    MATLAB's ``assembleResults`` returns them. NaN where no subproblem
    covers a pixel or source.

    Attributes
    ----------
    footprints : ndarray of float32, shape (pixels, sources)
        Over the trial's selected pixels, in column-major order.
    events, ls, f0 : ndarray of shape (channels, sources, frames)
    snr : ndarray of shape (sources,)
    """

    footprints: np.ndarray
    events: np.ndarray
    ls: np.ndarray
    f0: np.ndarray
    snr: np.ndarray


def assemble_results(
    subs: list[Subproblem],
    results: list[SubproblemResult],
    n_pixels: int,
    n_sources: int,
    photon_scale: float,
) -> TrialResult:
    """Put a trial's subproblem results together (``assembleResults``).

    Each subproblem's ``pixels`` and ``sources`` say where its footprints
    and traces go. Footprints and SNR do not depend on the scale; events,
    least squares and F0 go back to movie units.
    """
    n_ch, _, n_t = results[0].events.shape
    h = np.full((n_pixels, n_sources), np.nan, dtype=np.float32)
    shape = (n_ch, n_sources, n_t)
    ev, ls, f0 = (np.full(shape, np.nan) for _ in range(3))
    snr = np.full(n_sources, np.nan)
    for sb, r in zip(subs, results):
        if sb.pixels is None or sb.sources is None:
            raise ValueError(
                "a subproblem needs its pixels and sources to be assembled"
            )
        h[np.ix_(sb.pixels, sb.sources)] = r.footprints
        ev[:, sb.sources] = r.events
        ls[:, sb.sources] = r.ls
        f0[:, sb.sources] = r.f0
        snr[sb.sources] = r.snr
    return TrialResult(
        footprints=h,
        events=ev * photon_scale,
        ls=ls * photon_scale,
        f0=f0 * photon_scale,
        snr=snr,
    )


@dataclass
class TrialActivity:
    """A trial's outputs as SILo stores them (MATLAB's ``processResult``).

    Attributes
    ----------
    footprints : ndarray of shape (pixels, sources)
    events, denoised, ls : ndarray of shape (channels, sources, frames)
        NaN at the discarded frames (the events of channel 2 are not, as
        in MATLAB).
    f0 : ndarray of shape (channels, sources, frames)
    snr : ndarray of shape (sources,)
    """

    footprints: np.ndarray
    events: np.ndarray
    denoised: np.ndarray
    ls: np.ndarray
    f0: np.ndarray
    snr: np.ndarray


def conv_same(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """MATLAB ``convn(x, k, 'same')`` along the last axis (k of odd length)."""
    x = np.asarray(x, dtype=np.float64)
    c, n = len(k) // 2, x.shape[-1]
    rows = x.reshape(-1, n)
    out = np.stack([np.convolve(row, k)[c : c + n] for row in rows])
    return out.reshape(x.shape)


def process_result(
    trial: TrialResult, params: SolverParams, discard: np.ndarray
) -> TrialActivity:
    """A trial's activity traces (``processResult``).

    The denoised activity is the events convolved with the kernel (channel
    2 with its own); events, denoised activity and least squares are NaN
    at the discarded frames. MATLAB blanks the events with
    ``events(:, discard) = nan``, which reaches channel 1 only; channel
    2's events keep their values here too, so that the outputs agree.
    """
    discard = np.asarray(discard, dtype=bool).ravel()
    denoised = np.empty_like(trial.events)
    denoised[0] = conv_same(trial.events[0], params.k)
    for ch in range(1, trial.events.shape[0]):
        denoised[ch] = conv_same(trial.events[ch], params.k2)
    events, ls = trial.events.copy(), trial.ls.copy()
    events[0][:, discard] = np.nan
    denoised[:, :, discard] = np.nan
    ls[:, :, discard] = np.nan
    return TrialActivity(
        footprints=trial.footprints,
        events=events,
        denoised=denoised,
        ls=ls,
        f0=trial.f0,
        snr=trial.snr,
    )
