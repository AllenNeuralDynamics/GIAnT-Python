"""Deterministic parts of SiLo's subproblem solver: baseline and noise.

Ports of the local functions of GIAnT-MATLAB's ``extractTrial.m`` that do
not iterate to a tolerance: ``splitFreq`` (the baseline's starting value),
``computeFloor``, ``getSurround``, ``bsplineBasis``, ``fitB`` (one bounded
quadratic program per pixel, solved here to machine precision) and the
noise and SNR estimate. Arrays are pixels x frames or sources x frames,
float64.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import linalg, ndimage, optimize, sparse

from .localization import movmean_nan
from .solver_setup import SolverParams, conv_operator


def smooth_lowess(a: np.ndarray, window: int) -> np.ndarray:
    """MATLAB ``smoothdata(a, 2, 'lowess', window, 'omitmissing')``.

    Each value is replaced by a local linear fit over its ``window``
    nearest samples (the window is shifted at the ends to keep its
    length), with tricube weights that reach zero at the farthest of
    them. NaN samples are left out of the fit. Where the weights leave
    fewer than two samples, the result is their weighted mean, or the
    mean of the window's samples if all weights are zero.
    """
    a = np.asarray(a, dtype=np.float64)
    n = a.shape[1]
    window = int(min(window, n))
    back = window // 2
    out = np.empty_like(a)
    ok = ~np.isnan(a)
    a0 = np.where(ok, a, 0.0)
    for i in range(n):
        lo = min(max(i - back, 0), n - window)
        x = np.arange(lo, lo + window, dtype=np.float64) - i
        h = np.abs(x).max()
        w = (1 - (np.abs(x) / h) ** 3) ** 3 if h > 0 else np.ones(1)
        wk = w[None, :] * ok[:, lo : lo + window]
        yk = a0[:, lo : lo + window]
        s0 = wk.sum(1)
        s1 = wk @ x
        s2 = wk @ (x * x)
        t0 = (wk * yk).sum(1)
        t1 = (wk * yk) @ x
        den = s0 * s2 - s1 * s1
        with np.errstate(invalid="ignore", divide="ignore"):
            fit = (s2 * t0 - s1 * t1) / den
            mean_w = t0 / s0
            cnt = ok[:, lo : lo + window].sum(1)
            mean_all = yk.sum(1) / cnt
        degenerate = den <= 1e-12 * s0 * s2
        res = np.where(degenerate, mean_w, fit)
        res = np.where(s0 > 0, res, mean_all)
        out[:, i] = res
    return out


def ordfilt_rows(
    a: np.ndarray, order: int, n: int, pad: str = "zeros"
) -> np.ndarray:
    """MATLAB ``ordfilt2(a, order, ones(1, n), pad)`` along the rows.

    The ``order``-th smallest (1-based) of each window of ``n`` samples;
    an even window reaches ``n/2 - 1`` samples back and ``n/2`` ahead.
    ``pad`` is 'zeros' or 'symmetric'.
    """
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    mode = {"zeros": "constant", "symmetric": "reflect"}[pad]
    origin = -1 if n % 2 == 0 else 0
    out = np.empty_like(a)
    for i, row in enumerate(a):
        out[i] = ndimage.rank_filter(
            row, order - 1, size=n, mode=mode, cval=0.0, origin=origin
        )
    return out


def interp1_extrap(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """MATLAB ``interp1(x, y', xq, 'linear', 'extrap')`` for rows of y."""
    j = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, x.size - 2)
    frac = (xq - x[j]) / (x[j + 1] - x[j])
    return y[:, j] + (y[:, j + 1] - y[:, j]) * frac


def split_freq_pages(ma: np.ndarray, lp_factor: int) -> np.ndarray:
    """Low-pass of the page means in ``splitFreq`` (pixels x pages).

    A lowess fit marks the pages that lie low (at or below a low order
    statistic of their window's residuals); the others are dropped and the
    rest averaged in a moving window, with up to three more passes to fill
    windows that held no low page.
    """
    ma = np.array(ma, dtype=np.float64)
    resid = ma - smooth_lowess(ma, lp_factor)
    order = max(2, math.ceil(0.15 * lp_factor))
    low = resid <= ordfilt_rows(resid, order, lp_factor, "zeros")
    ma[~low] = np.nan
    sma = movmean_nan(ma.T, lp_factor).T
    for _ in range(3):
        nans = np.isnan(sma)
        if not nans.any():
            break
        tmp = movmean_nan(sma.T, lp_factor).T
        sma[nans] = tmp[nans]
    return sma


def split_freq(
    a: np.ndarray, denoise_window: int, lp_factor: int
) -> tuple[np.ndarray, np.ndarray]:
    """MATLAB's ``splitFreq``: a slow lower envelope of ``a`` and the rest.

    ``a`` (pixels x frames) is averaged in pages of ``denoise_window``
    frames, the page means are low-passed over ``lp_factor`` pages keeping
    the low ones (:func:`split_freq_pages`), and the result is
    interpolated linearly back to the frames.

    Returns
    -------
    lp, hp : ndarray
        The low-pass component and ``a - lp``.
    """
    a = np.asarray(a, dtype=np.float64)
    n_px, n_t = a.shape
    n_pages = n_t // denoise_window
    if n_pages < 2:
        with np.errstate(invalid="ignore"):
            lp = np.repeat(np.nanmean(a, axis=1)[:, None], n_t, axis=1)
        return lp, a - lp
    pages = a[:, : n_pages * denoise_window].reshape(
        n_px, n_pages, denoise_window
    )
    with np.errstate(invalid="ignore"):
        ma = np.nanmean(pages, axis=2)
    sma = split_freq_pages(ma, lp_factor)
    t_ds = (denoise_window + 1) / 2 + denoise_window * np.arange(n_pages)
    lp = interp1_extrap(t_ds, sma, np.arange(1, n_t + 1, dtype=np.float64))
    return lp, a - lp


def initial_baseline(y: np.ndarray, params: SolverParams) -> np.ndarray:
    """Starting baseline of a channel: ``splitFreq``'s low-pass, bounded.

    Where the low-pass is NaN (a pixel with no low page for longer than
    the fill passes of :func:`split_freq_pages` reach), the baseline is
    ``params.min_baseline``, as MATLAB's ``max`` returns it.
    """
    lp, _ = split_freq(y, params.denoise_window, params.lp_factor)
    return np.fmax(params.min_baseline, lp)


def compute_floor(
    x: np.ndarray, denoise_window: float, baseline_window: float
) -> np.ndarray:
    """MATLAB's ``computeFloor`` of the activity (sources x frames).

    A running median over ``2 ceil(denoise_window) + 1`` frames, then the
    10th-percentile order statistic over a baseline window, then a moving
    mean over a baseline window (symmetric padding for the first two).
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    order = math.ceil(0.1 * baseline_window)
    n_med = 2 * math.ceil(denoise_window) + 1
    n_base = math.ceil(baseline_window)
    med = ordfilt_rows(x, (n_med + 1) // 2, n_med, "symmetric")
    low = ordfilt_rows(med, order, n_base, "symmetric")
    return movmean_nan(low.T, n_base).T


def activity_noise(
    y: np.ndarray,
    finv: np.ndarray,
    b: np.ndarray,
    h: np.ndarray,
    x: np.ndarray,
    tau: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Noise and SNR of the activity, as ``extractSources`` estimates them.

    The variance of each pixel's residual is taken from its negative
    residuals, weighted by the freshness; the activity's noise follows
    from the footprints by weighted least squares, over ``tau`` frames.

    Returns
    -------
    resid_var : ndarray of shape (pixels,)
    x_noise, x_snr : ndarray of shape (sources,)
    """
    resid = y - (b + h @ x)
    w = np.where(resid >= 0, 0.0, 1.0 / finv)
    with np.errstate(invalid="ignore", divide="ignore"):
        resid_var = (resid * resid * w).sum(1) / w.sum(1)
        cov = np.linalg.inv(h.T @ (h / resid_var[:, None]))
        x_noise = np.sqrt(np.diag(cov) / tau)
        x_snr = np.std(x, axis=1, ddof=1) / x_noise
    return resid_var, x_noise, x_snr


def _bspline_parts(n_t: int, delta: float):
    """Values and first column of the four splines at each frame."""
    t = np.arange(n_t, dtype=np.float64)
    n = max(1, int(math.floor((n_t - 1) / delta + 0.5)))
    delta = max(n_t - 1, 1) / n
    u = t / delta
    i = np.minimum(np.floor(u), n - 1)
    f = u - i
    vals = (
        np.stack(
            [
                (1 - f) ** 3,
                3 * f**3 - 6 * f**2 + 4,
                -3 * f**3 + 3 * f**2 + 3 * f + 1,
                f**3,
            ],
            axis=1,
        )
        / 6
    )
    return vals, i.astype(np.int64), n + 3


def bspline_basis(n_t: int, delta: float) -> sparse.csr_matrix:
    """MATLAB's ``bsplineBasis``: uniform cubic B-splines on 0..n_t-1.

    The knot spacing is adjusted from ``delta`` so that whole knot
    intervals tile the trial. Sparse, frames x (intervals + 3).
    """
    vals, first, n_coef = _bspline_parts(n_t, delta)
    cols = first[:, None] + np.arange(4)[None, :]
    rows = np.broadcast_to(np.arange(n_t)[:, None], cols.shape)
    return sparse.csr_matrix(
        (vals.ravel(), (rows.ravel(), cols.ravel())), shape=(n_t, n_coef)
    )


def _bspline_products(n_t: int, delta: float) -> sparse.csr_matrix:
    """Products of the splines at each frame: frames x (coef * coef)."""
    vals, first, n_coef = _bspline_parts(n_t, delta)
    cols = first[:, None] + np.arange(4)[None, :]
    prod = vals[:, :, None] * vals[:, None, :]
    idx = cols[:, :, None] * n_coef + cols[:, None, :]
    rows = np.broadcast_to(np.arange(n_t)[:, None, None], idx.shape)
    return sparse.csr_matrix(
        (prod.ravel(), (rows.ravel(), idx.ravel())),
        shape=(n_t, n_coef * n_coef),
    )


def curvature_penalty(n_coef: int) -> np.ndarray:
    """Squared second differences of the spline coefficients, per interval."""
    d = np.zeros((n_coef - 2, n_coef))
    i = np.arange(n_coef - 2)
    d[i, i], d[i, i + 1], d[i, i + 2] = 1, -2, 1
    return d.T @ d / (n_coef - 3)


def _bvls_fallback(g, c, lb, ub):
    """One bounded QP by scipy's bounded-variable least squares.

    Variables whose bounds coincide are set to them first (the least
    squares need a lower bound below the upper one), and the others
    solved with their contribution moved into the linear term.
    """
    fixed = lb >= ub
    x = np.where(fixed, lb, 0.0)
    if fixed.all():
        return x
    free = ~fixed
    gf = g[np.ix_(free, free)]
    cf = c[free] - g[np.ix_(free, fixed)] @ x[fixed]
    n = gf.shape[0]
    ridge = 1e-13 * max(np.trace(gf) / n, np.finfo(float).tiny)
    r = linalg.cholesky(gf + ridge * np.eye(n), lower=False)
    rhs = linalg.solve_triangular(r, cf, trans="T")
    res = optimize.lsq_linear(
        r,
        rhs,
        bounds=(lb[free], ub[free]),
        method="bvls",
        tol=1e-14,
        max_iter=50 * n,
    )
    x[free] = res.x
    return x


def _qp_objective(g, c, x):
    """``x'Gx/2 - c'x`` of each problem of a stack."""
    return 0.5 * np.einsum("mi,mij,mj->m", x, g, x) - (c * x).sum(1)


def _qp_backtrack(g, c, x, target, grad, lo, hi):
    """Projected steps from x towards target that lower the objective.

    The step of each problem is halved until its objective decreases
    enough; a problem that finds no such step stays at x.
    """
    f0 = _qp_objective(g, c, x)
    alpha = np.ones(x.shape[0])
    new = x.copy()
    accepted = np.zeros(x.shape[0], dtype=bool)
    for _ in range(40):
        trial = np.clip(x + alpha[:, None] * (target - x), lo, hi)
        ft = _qp_objective(g, c, trial)
        better = ~accepted & (ft <= f0 + 1e-4 * (grad * (trial - x)).sum(1))
        new[better] = trial[better]
        accepted |= better
        if accepted.all():
            break
        alpha = np.where(accepted, alpha, alpha / 2)
    return new


def solve_box_qp(
    g: np.ndarray,
    c: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> tuple[np.ndarray, dict]:
    """Minimize ``x'Gx/2 - c'x`` within bounds, for a stack of problems.

    A projected Newton method on the free variables (exact for a
    quadratic once the active bounds are identified), run for all problems
    at once; a problem it does not finish goes to scipy's bounded-variable
    least squares. A variable whose row of G is zero enters the objective
    through ``c`` alone: it goes to its upper bound where ``c`` is
    positive, to its lower bound where ``c`` is negative, and to its
    bound nearest zero where ``c`` is zero.

    Parameters
    ----------
    g : ndarray of shape (m, n, n)
        Symmetric positive semidefinite matrices.
    c : ndarray of shape (m, n)
    lb, ub : ndarray of shape (m, n)

    Returns
    -------
    x : ndarray of shape (m, n)
    info : dict
        ``converged`` (per problem; False where the fallback was used) and
        ``kkt``, the largest gradient component among the variables that
        no bound holds, relative to the largest of ``c`` (both with G
        scaled to a unit diagonal).
    """
    m, n = c.shape
    diag = np.einsum("mii->mi", g)
    dead = diag <= 0
    scale = np.where(dead, 1.0, 1.0 / np.sqrt(np.where(dead, 1.0, diag)))
    # work in x / scale: unit diagonal
    gs = g * scale[:, :, None] * scale[:, None, :]
    cs = c * scale
    lo, hi = lb / scale, ub / scale
    eye = np.eye(n)

    def newton_point(gk, x, grad, active):
        """Minimum over the free variables, the active ones held."""
        free = ~active
        gm = gk * (free[:, :, None] & free[:, None, :])
        gm = gm + eye[None] * active[:, :, None]
        rhs = np.where(free, -grad, 0.0)[..., None]
        try:
            step = np.linalg.solve(gm, rhs)
        except np.linalg.LinAlgError:
            # collinear predictors: a ridge far below the unit diagonal
            step = np.linalg.solve(gm + 1e-12 * eye[None], rhs)
        return x + step[..., 0]

    def kkt_residual(x):
        """Gradient, active bounds and the largest free gradient."""
        grad = np.einsum("mij,mj->mi", gs, x) - cs
        active = ((x <= lo) & (grad > 0)) | ((x >= hi) & (grad < 0)) | dead
        return grad, active, np.where(active, 0.0, np.abs(grad)).max(1)

    x = np.clip(np.zeros((m, n)), lo, hi)
    done = np.zeros(m, dtype=bool)
    grad_scale = np.maximum(np.abs(cs).max(1), 1e-300)
    for _ in range(max_iter):
        grad, active, kkt = kkt_residual(x)
        done = kkt <= tol * grad_scale
        if done.all():
            break
        todo = ~done
        gk, xt, gt = gs[todo], x[todo], grad[todo]
        target = newton_point(gk, xt, gt, active[todo])
        x[todo] = _qp_backtrack(
            gk, cs[todo], xt, target, gt, lo[todo], hi[todo]
        )
    for i in np.nonzero(~done)[0]:
        keep = ~dead[i]
        xi = np.zeros(n)
        xi[keep] = _bvls_fallback(
            gs[i][np.ix_(keep, keep)], cs[i][keep], lo[i][keep], hi[i][keep]
        )
        x[i] = xi
    kkt = kkt_residual(x)[2] / grad_scale
    # back to the given scale; the bounds are met exactly
    x = np.clip(x * scale, lb, ub)
    linear = np.where(c > 0, ub, np.where(c < 0, lb, np.clip(0.0, lb, ub)))
    return np.where(dead, linear, x), {"converged": done, "kkt": kkt}


def surround_operators(sel: np.ndarray, params: SolverParams) -> list:
    """The three surround filters as pixels x pixels matrices.

    Dense where the filter reaches a good share of the pixels (a small
    subproblem), sparse otherwise.
    """
    ops = [conv_operator(sel, f) for f in params.bfilter]
    return [
        op.toarray() if op.nnz > 0.05 * op.shape[0] * op.shape[1] else op
        for op in ops
    ]


def get_surround(hp: np.ndarray, operators: list) -> np.ndarray:
    """MATLAB's ``getSurround``: ``(filters, pixels, frames)``."""
    return np.stack([op @ hp for op in operators])


def baseline_programs(
    b0: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    h: np.ndarray,
    finv: np.ndarray,
    sel: np.ndarray,
    params: SolverParams,
    operators: list | None = None,
) -> dict:
    """The per-pixel quadratic programs of ``fitB``.

    Each pixel's coefficients are those of its sources' contributions,
    the three surround predictors and the baseline spline, in that order.

    Returns
    -------
    dict
        ``g`` (pixels x n x n) and ``c`` (pixels x n): the normal
        equations, with the curvature penalty on the spline block;
        ``lb``, ``ub``: the bounds; ``surround`` (3 x pixels x frames) and
        ``spline`` (frames x coefficients, sparse): the baseline's
        predictors; ``n_sources``.
    """
    n_px, n_t = y.shape
    n_src = h.shape[1]
    hx = h @ x
    _, hp = split_freq(y - hx, 2 * params.denoise_window, params.lp_factor)
    hp[np.isnan(hp)] = 0
    if operators is None:
        operators = surround_operators(sel, params)
    sur = get_surround(hp, operators)
    n_sur = sur.shape[0]
    delta = params.baseline_knot_factor * params.baseline_window
    bs = bspline_basis(n_t, delta)
    n_k = bs.shape[1]
    d = n_src + n_sur
    n = d + n_k

    scale = np.fmax(1e-3, y.mean(1))
    w = (1.0 / finv) * (b0 / (b0 + hx)) ** 2
    wy = w * y

    g = np.zeros((n_px, n, n))
    c = np.zeros((n_px, n))
    # sources x sources, for the pairs that share a pixel
    live = (h != 0).astype(np.float64)
    jj, ll = np.nonzero(np.triu(live.T @ live > 0))
    gss = (w @ (x[jj] * x[ll]).T) * h[:, jj] * h[:, ll]
    g[:, jj, ll] = gss
    g[:, ll, jj] = gss
    # sources and spline x surround, surround x surround
    wsur = w[None] * sur
    for f in range(n_sur):
        gsf = (wsur[f] @ x.T) * h
        g[:, :n_src, n_src + f] = gsf
        g[:, n_src + f, :n_src] = gsf
        for f2 in range(f, n_sur):
            v = (wsur[f] * sur[f2]).sum(1)
            g[:, n_src + f, n_src + f2] = v
            g[:, n_src + f2, n_src + f] = v
        c[:, n_src + f] = (wsur[f] * y).sum(1)
        gbf = (bs.T @ wsur[f].T).T
        g[:, d:, n_src + f] = gbf
        g[:, n_src + f, d:] = gbf
    # spline x sources
    for j in range(n_src):
        gbj = (bs.T @ (w * x[j]).T).T * h[:, j : j + 1]
        g[:, d:, j] = gbj
        g[:, j, d:] = gbj
    c[:, :n_src] = (wy @ x.T) * h
    c[:, d:] = (bs.T @ wy.T).T
    # spline x spline, with the curvature penalty
    bb = _bspline_products(n_t, delta)
    penalty = params.baseline_penalty * w.sum(1)
    g[:, d:, d:] = (bb.T @ w.T).T.reshape(n_px, n_k, n_k) + (
        penalty[:, None, None] * curvature_penalty(n_k)[None]
    )
    g = (g + g.transpose(0, 2, 1)) / 2

    wide = 10 * scale[:, None] * np.ones((1, n))
    lb = -wide
    lb[:, :n_src] = 0
    return dict(
        g=g, c=c, lb=lb, ub=wide, surround=sur, spline=bs, n_sources=n_src
    )


def baseline_from_coef(
    coef: np.ndarray, programs: dict, min_baseline: float
) -> np.ndarray:
    """The baseline of ``fitB`` from its coefficients: surround + spline."""
    n_src = programs["n_sources"]
    sur, bs = programs["surround"], programs["spline"]
    d = n_src + sur.shape[0]
    fast = np.einsum("pf,fpt->pt", coef[:, n_src:d], sur)
    slow = (bs @ coef[:, d:].T).T
    return np.fmax(min_baseline, fast + slow)


def fit_baseline(
    b0: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    h: np.ndarray,
    finv: np.ndarray,
    sel: np.ndarray,
    params: SolverParams,
    operators: list | None = None,
    return_info: bool = False,
):
    """MATLAB's ``fitB``: the baseline given the activity, per pixel.

    For each pixel, a bounded weighted least-squares fit of ``y`` to the
    sources' contributions (>= 0), three fast predictors from the
    surrounding pixels and a slow cubic B-spline with a curvature penalty.
    The baseline is the fit without the sources' part, bounded below by
    ``params.min_baseline``.

    Parameters
    ----------
    b0 : ndarray of shape (pixels, frames)
        Current baseline; it sets the weights.
    y, finv : ndarray of shape (pixels, frames)
        Data and variance multiplier.
    x : ndarray of shape (sources, frames)
        Activity (thresholded, as the solver passes it).
    h : ndarray of shape (pixels, sources)
        Footprints.
    sel : ndarray of bool
        The crop of selected pixels.
    operators : list, optional
        :func:`surround_operators` of ``sel``, to reuse them.

    Returns
    -------
    b : ndarray of shape (pixels, frames)
    info : dict, if ``return_info``
        ``coef`` (pixels x (sources + 3 + spline coefficients)), and
        ``converged`` and ``kkt`` of :func:`solve_box_qp`.
    """
    programs = baseline_programs(b0, y, x, h, finv, sel, params, operators)
    coef, info = solve_box_qp(
        programs["g"], programs["c"], programs["lb"], programs["ub"]
    )
    b = baseline_from_coef(coef, programs, params.min_baseline)
    if return_info:
        return b, dict(coef=coef, **info)
    return b


def source_f0(h: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Footprint-weighted baseline of each source (sources x frames)."""
    return (h.T @ b) / (h * h).sum(0)[:, None]
