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
import warnings

import numpy as np
from scipy import linalg, ndimage, optimize, sparse

from .localization import movmean_nan
from .solver_setup import SolverParams, conv_operator, sel_coords


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
    """One bounded QP by scipy, with the objective as given.

    Variables whose bounds coincide are set to them first, and the others
    solved with their contribution moved into the linear term: by
    bounded-variable least squares on the Cholesky factor where G is
    positive definite, and otherwise (G singular, so that the linear term
    may act alone along some direction) by L-BFGS-B on the objective
    itself. No ridge is added.
    """
    fixed = lb >= ub
    x = np.where(fixed, lb, 0.0)
    if fixed.all():
        return x
    free = ~fixed
    gf = g[np.ix_(free, free)]
    cf = c[free] - g[np.ix_(free, fixed)] @ x[fixed]
    lo, hi = lb[free], ub[free]
    try:
        r = linalg.cholesky(gf, lower=False)
        rhs = linalg.solve_triangular(r, cf, trans="T")
        x[free] = optimize.lsq_linear(
            r,
            rhs,
            bounds=(lo, hi),
            method="bvls",
            tol=1e-14,
            max_iter=50 * gf.shape[0],
        ).x
    except linalg.LinAlgError:
        # in units of the linear term, so that the tolerance means the same
        # whatever the program's scale
        scale = np.abs(cf).max()
        if scale == 0:
            scale = 1.0  # no linear term: G's unit diagonal sets the scale

        def objective(v):
            """The objective and its gradient, over the linear term's scale."""
            gv = gf @ v
            return (0.5 * v @ gv - cf @ v) / scale, (gv - cf) / scale

        x[free] = optimize.minimize(
            objective,
            np.clip(0.0, lo, hi),
            jac=True,
            method="L-BFGS-B",
            bounds=list(zip(lo, hi)),
            options=dict(maxiter=20000, ftol=0.0, gtol=1e-12),
        ).x
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
        scaled to a unit diagonal). It is computed on the objective as
        given, also for the fallback's solutions; a warning is given if
        one of those is above 1e-6.
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
    if np.any(kkt[~done] > 1e-6):
        warnings.warn(
            f"{int(np.sum(kkt[~done] > 1e-6))} bounded quadratic programs "
            "were not solved to tolerance (largest relative gradient "
            f"{kkt[~done].max():.1e})",
            stacklevel=2,
        )
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
    offset: np.ndarray | None = None,
) -> dict:
    """The per-pixel quadratic programs of ``fitB``.

    Each pixel's coefficients are those of its sources' contributions,
    the three surround predictors and the baseline spline, in that order.
    With ``offset`` (pixels x frames), the programs fit ``y - offset``:
    the baseline is then the offset plus the surround and spline parts
    (``SolverParams.baseline_mode`` 'fit_from'). The coefficients' bounds
    still scale with the mean of ``y``.

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
    target = y if offset is None else y - offset
    _, hp = split_freq(
        target - hx, 2 * params.denoise_window, params.lp_factor
    )
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
    wy = w * target

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
        c[:, n_src + f] = (wsur[f] * target).sum(1)
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
    coef: np.ndarray,
    programs: dict,
    min_baseline: float,
    offset: np.ndarray | None = None,
) -> np.ndarray:
    """The baseline of ``fitB`` from its coefficients: surround + spline
    (plus the offset the programs were formed with, if any)."""
    n_src = programs["n_sources"]
    sur, bs = programs["surround"], programs["spline"]
    d = n_src + sur.shape[0]
    fast = np.einsum("pf,fpt->pt", coef[:, n_src:d], sur)
    slow = (bs @ coef[:, d:].T).T
    if offset is not None:
        return np.fmax(min_baseline, offset + fast + slow)
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
    offset: np.ndarray | None = None,
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
    return_info : bool
        Also return the coefficients and the programs' convergence.
    offset : ndarray of shape (pixels, frames), optional
        A baseline to fit the correction to (not a MATLAB option): the
        result is the offset plus the fitted surround and spline parts
        (see :func:`baseline_programs`).

    Returns
    -------
    b : ndarray of shape (pixels, frames)
    info : dict, if ``return_info``
        ``coef`` (pixels x (sources + 3 + spline coefficients)), and
        ``converged`` and ``kkt`` of :func:`solve_box_qp`.
    """
    programs = baseline_programs(
        b0, y, x, h, finv, sel, params, operators, offset
    )
    coef, info = solve_box_qp(
        programs["g"], programs["c"], programs["lb"], programs["ub"]
    )
    b = baseline_from_coef(coef, programs, params.min_baseline, offset)
    if return_info:
        return b, dict(coef=coef, **info)
    return b


# upper bound of a pixel's gain on an external baseline (the 'gain' modes)
GAIN_MAX = 10.0
# fewest pixels outside the footprints that a shared time course is
# estimated from ('gain_shared'); with fewer it is held at 1
MIN_OUTSIDE = 10
# fewest pixels in each half of the outside pixels for the shared
# variance ('gain_shared'); with fewer it is 0
MIN_HALF = 5
# frames whose shared factor's sampling variance exceeds this multiple of
# the median do not enter the estimate of the shared variance
RELIABLE_FACTOR = 4.0
# samples whose variance multiplier exceeds this multiple of the median are
# filled in (the loader gives them 1000 x the mean)
FILLED_FACTOR = 100.0


def gain_programs(
    b0: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    h: np.ndarray,
    finv: np.ndarray,
    predictor: np.ndarray,
) -> dict:
    """Per-pixel programs of the gain fit, ``fitB``'s with one predictor.

    Each pixel's data are fitted by its sources' contributions (each with
    a coefficient >= 0, as in :func:`baseline_programs`) and a gain times
    ``predictor`` (pixels x frames; an external baseline), with ``fitB``'s
    weights ``(1 / F) (b0 / (b0 + H x))^2`` and no penalty. The sources'
    coefficients are bounded as in ``fitB``, the gain to
    ``[0, GAIN_MAX]``.

    Returns
    -------
    dict
        ``g`` (pixels x n x n), ``c`` (pixels x n), ``lb``, ``ub`` with
        ``n = sources + 1`` (the gain last), and ``n_sources``.
    """
    n_px, _ = y.shape
    n_src = h.shape[1]
    hx = h @ x
    w = (1.0 / finv) * (b0 / (b0 + hx)) ** 2
    n = n_src + 1
    g = np.zeros((n_px, n, n))
    c = np.zeros((n_px, n))
    live = (h != 0).astype(np.float64)
    jj, ll = np.nonzero(np.triu(live.T @ live > 0))
    gss = (w @ (x[jj] * x[ll]).T) * h[:, jj] * h[:, ll]
    g[:, jj, ll] = gss
    g[:, ll, jj] = gss
    wq = w * predictor
    gq = (wq @ x.T) * h
    g[:, :n_src, n_src] = gq
    g[:, n_src, :n_src] = gq
    g[:, n_src, n_src] = (wq * predictor).sum(1)
    c[:, :n_src] = ((w * y) @ x.T) * h
    c[:, n_src] = (wq * y).sum(1)
    g = (g + g.transpose(0, 2, 1)) / 2
    scale = np.fmax(1e-3, y.mean(1))
    lb = np.zeros((n_px, n))
    ub = 10 * scale[:, None] * np.ones((1, n))
    ub[:, n_src] = GAIN_MAX
    return dict(g=g, c=c, lb=lb, ub=ub, n_sources=n_src)


def fit_gain(
    b0: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    h: np.ndarray,
    finv: np.ndarray,
    predictor: np.ndarray,
    min_baseline: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The baseline ``max(min_baseline, gain * predictor)`` per pixel.

    The gains come from :func:`gain_programs`, solved by
    :func:`solve_box_qp`.

    Returns
    -------
    b : ndarray of shape (pixels, frames)
    coef : ndarray of shape (pixels, sources + 1)
        The sources' coefficients and the gain (last).
    """
    programs = gain_programs(b0, y, x, h, finv, predictor)
    coef, _ = solve_box_qp(
        programs["g"], programs["c"], programs["lb"], programs["ub"]
    )
    b = np.fmax(min_baseline, coef[:, -1:] * predictor)
    return b, coef


def shared_time_course(
    y: np.ndarray,
    q: np.ndarray,
    src: np.ndarray,
    finv: np.ndarray,
    halves: tuple | None = None,
) -> tuple[np.ndarray, dict]:
    """A factor per frame shared by pixels, and how it was shrunk.

    From pixels (outside the footprints) with data ``y``, predicted
    baseline ``q`` and sources' part ``src`` (all pixels x frames): per
    frame, the weighted least-squares factor of ``q`` in ``y - src``,

        c_t = sum_p w r q / sum_p w q^2,  r = y - src,
        w = 1 / (F (q + src + 1))  (the solver's loss weights),

    whose sampling variance is ``alpha / sum_p w q^2``, with ``alpha`` the
    mean weighted squared residual of the observed samples (not filled in)
    of the reliable frames (1 if the loss's variance model holds).
    Each frame's factor is then shrunk towards their precision-weighted
    mean ``m`` by the empirical-Bayes weight ``s2 / (s2 + var_t)``, where
    ``s2`` is the variance of the factors about ``m`` beyond their
    sampling variance (at least 0), estimated from the reliable frames
    (sampling variance at most ``RELIABLE_FACTOR`` times its median, or
    times ten times its 10th percentile if that is lower; this leaves out
    filled-in frames, also where they are most of the trial). Without a
    shared change the factor is ``m`` throughout.

    How ``s2`` is estimated:

    - with ``halves`` (two boolean masks over the pixels, spatially apart:
      the solver splits the outside pixels at their median column or row
      with a gap): ``s2`` is the covariance over the reliable frames of
      the factors estimated from each half alone. A change shared by the
      pixels appears in both halves; noise shared only by neighbouring
      pixels (the 200 Hz frames are resampled, so neighbours share
      photons) does not. The sampling variance of each frame is then the
      factors' remaining variance, ``var(c) - s2``, apportioned by the
      frame's precision. With fewer than ``MIN_HALF`` pixels in a half,
      ``s2`` is 0.
    - without: ``s2`` is the factors' variance beyond the sampling
      variance that independent pixel noise would give. This holds only
      for independent noise (it over-estimates ``s2`` on resampled
      frames).

    Returns
    -------
    c : ndarray of shape (frames,)
    info : dict
        ``raw`` (the unshrunk factors), ``mean`` (m), ``alpha``, ``s2``.
    """
    y, q, src = (np.asarray(a, np.float64) for a in (y, q, src))
    w = 1.0 / (np.asarray(finv, np.float64) * (q + src + 1))
    r = y - src
    den = (w * q * q).sum(0)
    num = (w * r * q).sum(0)
    ok = den > 0
    n_t = y.shape[1]
    raw = np.ones(n_t)
    raw[ok] = num[ok] / den[ok]
    if not ok.any():
        return raw, dict(raw=raw, mean=1.0, alpha=float("nan"), s2=0.0)
    m = float(num[ok].sum() / den[ok].sum())
    raw[~ok] = m
    # reliable frames: sampling variance (proportional to 1 / den) at most
    # RELIABLE_FACTOR times its median. Filled-in frames (discarded,
    # variance multiplier 1000 x) would otherwise inflate the mean sampling
    # variance without adding spread, and count in alpha's degrees of
    # freedom without adding residual. The median is capped at 10 times
    # the 10th percentile, for trials where most frames are filled in
    inv = np.full(n_t, np.inf)
    inv[ok] = 1.0 / den[ok]
    rel = reliable_frames(den)
    # observed samples: not filled in (variance multiplier far above the
    # typical); filled ones add no residual and would bias alpha down
    finv = np.asarray(finv, np.float64)
    seen = (finv <= FILLED_FACTOR * np.median(finv)) & rel[None, :]
    dof = max(int(seen.sum()) - int(rel.sum()), 1)
    alpha = float((w * (r - raw * q) ** 2)[seen].sum() / dof)
    var_t = alpha * inv
    spread = float(((raw[rel] - m) ** 2).sum() / max(int(rel.sum()) - 1, 1))
    record = dict(raw=raw, mean=m, alpha=alpha, var_raw=spread)
    if halves is None:
        s2 = max(0.0, spread - float(var_t[rel].mean()))
    else:
        s2, cov = _shared_variance_halves(w, r, q, rel, halves)
        record["cov_halves"] = cov
        # the rest of the factors' variance is their sampling variance,
        # apportioned to the frames by their precision; centred on the
        # arithmetic mean, as the halves' covariance
        sample_var = float(np.var(raw[rel], ddof=1)) if rel.sum() > 1 else 0.0
        noise = max(sample_var - s2, 0.0)
        var_t = noise * inv / float(inv[rel].mean())
    lam = np.zeros(n_t)
    if s2 > 0:
        lam[ok] = s2 / (s2 + var_t[ok])
    c = m + lam * (raw - m)
    record["s2"] = s2
    return c, record


def _shared_variance_halves(w, r, q, rel, halves):
    """Covariance of two halves' factors over the reliable frames (>= 0).

    Returns ``(s2, cov)``; ``s2`` is 0 when a half has fewer than
    ``MIN_HALF`` pixels or no reliable frame has a factor in both.
    """
    a, b = (np.asarray(x, dtype=bool) for x in halves)
    if a.sum() < MIN_HALF or b.sum() < MIN_HALF:
        return 0.0, float("nan")
    fac = []
    both = rel.copy()
    for m in (a, b):
        den = (w[m] * q[m] * q[m]).sum(0)
        num = (w[m] * r[m] * q[m]).sum(0)
        fac.append((num, den))
        # each half's frames must be reliable too (a half can be filled
        # in where the whole is not)
        both &= reliable_frames(den)
    if both.sum() < 2:
        return 0.0, float("nan")
    ca = fac[0][0][both] / fac[0][1][both]
    cb = fac[1][0][both] / fac[1][1][both]
    # sample covariance, normalized as the factors' variance
    cov = float(((ca - ca.mean()) * (cb - cb.mean())).sum() / (both.sum() - 1))
    return max(cov, 0.0), cov


def reliable_frames(den: np.ndarray) -> np.ndarray:
    """Frames whose factor is reliable, from its precision ``den``.

    A frame is reliable if ``den`` is positive and its sampling variance
    (proportional to ``1 / den``) is at most ``RELIABLE_FACTOR`` times its
    median over those frames, or times ten times its 10th percentile if
    that is lower (for trials where most frames are filled in).
    """
    den = np.asarray(den, dtype=np.float64)
    ok = den > 0
    out = np.zeros(den.shape, dtype=bool)
    if not ok.any():
        return out
    inv = 1.0 / den[ok]
    ref = min(np.median(inv), 10 * np.percentile(inv, 10))
    out[ok] = inv <= RELIABLE_FACTOR * ref
    return out


def outside_halves(sel: np.ndarray, outside: np.ndarray, gap: float = 1.0):
    """Two spatially separate halves of the outside pixels.

    The outside pixels (a boolean mask over the subproblem's pixels, in
    ``sel``'s column-major order) are split at their median column, or
    row if they extend further in rows; pixels within ``gap`` of the
    median are left out, so that the halves share no neighbours.

    Returns
    -------
    a, b : ndarray of bool, shape (outside pixels,)
        Masks over the outside pixels.
    """
    rows, cols = sel_coords(sel)
    rows, cols = rows[outside], cols[outside]
    axis = cols if np.ptp(cols) >= np.ptp(rows) else rows
    med = float(np.median(axis))
    return axis < med - gap, axis > med + gap


def fit_network_gain(
    b0: np.ndarray,
    b_ext: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    h: np.ndarray,
    finv: np.ndarray,
    params: SolverParams,
    shared: bool = False,
    sel: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """The baseline step of the 'gain' and 'gain_shared' modes.

    'gain': ``B = g_p * b_ext`` with one gain per pixel
    (:func:`fit_gain`). 'gain_shared' (``shared``): ``B = g_p * b_ext *
    c_t``. The gains are fitted first with ``c = 1``; ``c`` is then
    estimated from the pixels where every footprint is below
    ``params.outside_footprint`` of its largest value
    (:func:`shared_time_course`; held at 1 with fewer than
    ``MIN_OUTSIDE`` such pixels), and the gains are fitted again with
    ``b_ext * c`` as the predictor.

    Parameters
    ----------
    b0 : ndarray of shape (pixels, frames)
        Current baseline (sets ``fitB``'s weights).
    b_ext : ndarray of shape (pixels, frames)
        The external baseline.
    y, finv : ndarray of shape (pixels, frames)
    x : ndarray of shape (sources, frames)
        Activity (thresholded, as for ``fitB``).
    h : ndarray of shape (pixels, sources)
    params : SolverParams
    shared : bool
        Also fit the shared time course.
    sel : ndarray of bool, optional
        The subproblem's crop: its outside pixels are split into two
        spatially separate halves (:func:`outside_halves`) for the shared
        variance. Without it the shared variance assumes independent
        pixel noise (see :func:`shared_time_course`).

    Returns
    -------
    b : ndarray of shape (pixels, frames)
    info : dict
        ``gain`` (pixels); with ``shared``: ``shared`` (frames, c),
        ``n_outside``, and the :func:`shared_time_course` record.
    """
    b, coef = fit_gain(b0, y, x, h, finv, b_ext, params.min_baseline)
    info = dict(gain=coef[:, -1].copy())
    if not shared:
        return b, info
    n_src = h.shape[1]
    peak = h.max(0)
    live = peak > 0  # a source without a footprint excludes no pixel
    outside = (h[:, live] < params.outside_footprint * peak[None, live]).all(1)
    info["n_outside"] = int(outside.sum())
    if outside.sum() < MIN_OUTSIDE:
        info["shared"] = np.ones(y.shape[1])
        return b, info
    src = (coef[:, :n_src] * h) @ x
    q = coef[:, -1:] * b_ext
    halves = None if sel is None else outside_halves(sel, outside)
    c, record = shared_time_course(
        y[outside], q[outside], src[outside], finv[outside], halves
    )
    b, coef = fit_gain(
        b0, y, x, h, finv, b_ext * c[None, :], params.min_baseline
    )
    info.update(gain=coef[:, -1].copy(), shared=c, **record)
    return b, info


def source_f0(h: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Footprint-weighted baseline of each source (sources x frames)."""
    return (h.T @ b) / (h * h).sum(0)[:, None]
