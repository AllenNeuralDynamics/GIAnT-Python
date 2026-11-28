"""Source seeds in a multi-ROI (raster) SiLo activity image.

Port of GIAnT-MATLAB's ``getActImPeaks.m`` and its ``gaussianPeaksIntegrated``
(the 2-D raster version; the band-scan port in
:mod:`giant_python.numerics.peaks` is a different algorithm). Candidate peaks
are local maxima whose second-largest 3x3 neighbour exceeds the threshold
(median plus ``peak_th`` MAD-scaled SDs of the whole image). Isotropic
Gaussians, each integrated over its pixel, are fitted jointly to the image
around them; residual maxima above the threshold are then added one at a time
(refitting the peaks of their connected region), and fitted peaks whose
amplitude is too small for the threshold are dropped.

MATLAB fits with ``lsqcurvefit`` (trust-region-reflective, finite-difference
Jacobian, at most 5000 function evaluations). This port uses
:func:`scipy.optimize.least_squares` (``trf``) with the analytical Jacobian and
caps its evaluations at the number of iterations those 5000 evaluations allow
(one plus one per parameter each), so fits of many peaks stop early as in
MATLAB. The optimizers' paths differ, so positions agree to a small fraction
of a pixel rather than exactly.

``fit='local'`` is a faster variant of the same search (not in MATLAB):
each Gaussian is evaluated only near its start pixel, every fit runs to
convergence, and a residual peak refits only its neighbours. ``merge_px``
(either fit) also rejects a residual peak that ends next to another peak:
an activity image's peaks are often split in two that way.

Coordinates follow MATLAB internally (1-based pixel centres); the public
function returns 0-based rows and columns.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndimage
import scipy.sparse as sparse
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import least_squares
from scipy.special import erf

_AMP_SCALE = 1.0 / 0.75
_SIGMA0 = 0.35  # starting width of every fitted peak
_MAD_SCALE = (
    0.6741891400433162  # MATLAB's getActImPeaks divides mad(x, 1) by this
)
_MAX_FUN_EVALS = 5000
_SQRT_PI_2 = np.sqrt(np.pi / 2)
_SQRT2 = np.sqrt(2.0)

#: Seed fits: MATLAB's (``"matlab"``) and the local variant (``"local"``).
SEED_FITS = ("matlab", "local")
# the local fit: pixels (Chebyshev) around its start pixel a Gaussian is
# evaluated at (its centre stays within 1.5 px of it, and activity-image
# peaks are narrower than a pixel); peaks within this distance of a new
# residual peak are refitted with it; tolerances of each fit; largest fit
# (parameters) solved with a dense Jacobian rather than iteratively
_LOCAL_WINDOW = 10
_LOCAL_REFIT_RADIUS = 6.0
_LOCAL_TOL = 1e-4
_LOCAL_DENSE_MAX = 400


def ordfilt3x3(a: np.ndarray, order: int) -> np.ndarray:
    """MATLAB ``ordfilt2(a, order, ones(3))`` for a 2-D float image.

    The ``order``-th smallest value (9 = maximum, 8 = second largest) of each
    3x3 neighbourhood, with zero padding at the border, and NaN wherever the
    neighbourhood contains a NaN (as MATLAB's ``ordfilt2`` returns).
    """
    nan = np.isnan(a)
    padded = np.pad(np.where(nan, 0.0, a), 1, constant_values=0.0)
    win = sliding_window_view(padded, (3, 3)).reshape(a.shape + (9,))
    out = np.partition(win, order - 1, axis=-1)[..., order - 1]
    nan_win = ndimage.maximum_filter(nan, size=3, mode="constant", cval=False)
    return np.where(nan_win, np.nan, out)


def _edges(u: np.ndarray) -> np.ndarray:
    """Pixel edges from sorted unique centres, as gaussianPeaksIntegrated.

    The pixel size is the median spacing of the unique centres; interior edges
    are midpoints between neighbouring centres (so pixels next to a gap in the
    selection are wider, as in MATLAB).
    """
    d = float(np.median(np.diff(u))) if u.size > 1 else 1.0
    return np.concatenate(
        [[u[0] - d / 2], (u[:-1] + u[1:]) / 2, [u[-1] + d / 2]]
    )


class _IntegratedGaussians:
    """Integrated isotropic Gaussians at a fixed set of pixels.

    ``theta`` is flattened column-major like MATLAB's ``N x 4`` matrix
    ``[A, mu_y, mu_x, sigma]``: all amplitudes, then all ``mu_y``, ...
    """

    def __init__(self, y: np.ndarray, x: np.ndarray) -> None:
        """Precompute the pixel-edge grid of the selected pixels."""
        yu, yi = np.unique(y, return_inverse=True)
        xu, xi = np.unique(x, return_inverse=True)
        ye, xe = _edges(yu), _edges(xu)
        self.y_lo, self.y_hi = ye[:-1][yi], ye[1:][yi]
        self.x_lo, self.x_hi = xe[:-1][xi], xe[1:][xi]

    @staticmethod
    def _axis(lo, hi, mu, s, derivs=True):
        """Integral over [lo, hi] and (with ``derivs``) its derivatives in mu
        and sigma."""
        u_hi = (hi[:, None] - mu[None, :]) / (_SQRT2 * s[None, :])
        u_lo = (lo[:, None] - mu[None, :]) / (_SQRT2 * s[None, :])
        val = _SQRT_PI_2 * s[None, :] * (erf(u_hi) - erf(u_lo))
        if not derivs:
            return val
        e_hi, e_lo = np.exp(-(u_hi**2)), np.exp(-(u_lo**2))
        d_mu = -(e_hi - e_lo)
        d_s = val / s[None, :] - _SQRT2 * (u_hi * e_hi - u_lo * e_lo)
        return val, d_mu, d_s

    def parts(self, theta: np.ndarray, derivs: bool = True):
        """Per-axis integrals (pixels x peaks), with their derivatives
        unless ``derivs`` is False."""
        n = theta.size // 4
        a, my, mx = theta[:n], theta[n : 2 * n], theta[2 * n : 3 * n]
        s = np.maximum(theta[3 * n :], np.finfo(float).eps)
        iy = self._axis(self.y_lo, self.y_hi, my, s, derivs)
        ix = self._axis(self.x_lo, self.x_hi, mx, s, derivs)
        return a, iy, ix

    def value(self, theta: np.ndarray) -> np.ndarray:
        """Model value at each selected pixel."""
        a, vy, vx = self.parts(theta, derivs=False)
        return (vy * vx) @ a

    def jacobian(self, theta: np.ndarray) -> np.ndarray:
        """Jacobian (pixels x parameters), parameters column-major."""
        a, (vy, dmy, dsy), (vx, dmx, dsx) = self.parts(theta)
        return np.hstack(
            [vy * vx, a * dmy * vx, a * vy * dmx, a * (dsy * vx + vy * dsx)]
        )


def _eval_budget(n_par: int) -> int:
    """``max_nfev`` that gives ``least_squares`` MATLAB's iterations.

    ``lsqcurvefit`` with ``MaxFunEvals`` 5000 and finite differences spends
    one evaluation on the start and ``n_par + 1`` per iteration (Jacobian
    and step), so ``(5000 - 1) // (n_par + 1)`` iterations. With the
    analytical Jacobian each iteration costs one evaluation here, plus one
    for the start.
    """
    return 1 + (_MAX_FUN_EVALS - 1) // (n_par + 1)


def _fit(theta0, y, x, data, lb, ub):
    """lsqcurvefit-like bounded least squares of integrated Gaussians."""
    model = _IntegratedGaussians(y, x)
    lb_f, ub_f = lb.ravel(order="F"), ub.ravel(order="F")
    x0 = np.clip(theta0.ravel(order="F").astype(float), lb_f, ub_f)
    max_iter = _eval_budget(x0.size)
    kw = dict(
        jac=model.jacobian,
        bounds=(lb_f, ub_f),
        method="trf",
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
        max_nfev=max_iter,
    )
    try:
        res = least_squares(lambda t: model.value(t) - data, x0, **kw)
    except np.linalg.LinAlgError:
        # LAPACK's SVD now and then fails to converge on a fit that succeeds
        # when rerun (once in a run of 24 simulated trials under full load);
        # the lsmr trust-region solver needs no SVD
        res = least_squares(
            lambda t: model.value(t) - data, x0, tr_solver="lsmr", **kw
        )
    return res.x.reshape(theta0.shape, order="F")


def _bounds(locs: np.ndarray, shape) -> tuple[np.ndarray, np.ndarray]:
    """getActImPeaks' bounds: positions within 1.5 px of the start."""
    n = locs.shape[0]
    ub = np.column_stack(
        [
            np.full(n, np.inf),
            np.minimum(shape[0] + 0.5, locs[:, 0] + 1.5),
            np.minimum(shape[1] + 0.5, locs[:, 1] + 1.5),
            np.full(n, 5.0),
        ]
    )
    lb = np.column_stack(
        [
            np.zeros(n),
            np.maximum(0.5, locs[:, 0] - 1.5),
            np.maximum(0.5, locs[:, 1] - 1.5),
            np.zeros(n),
        ]
    )
    return lb, ub


def _find_f(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """1-based (row, col) of true pixels in MATLAB's column-major order."""
    c, r = np.nonzero(mask.T)
    return r + 1.0, c + 1.0


def _round(x):
    """MATLAB's ``round`` (halves away from zero; numpy's go to even)."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def _rounded_index(theta: np.ndarray, shape) -> tuple[np.ndarray, np.ndarray]:
    """0-based pixel of each fitted peak (rounded and clamped)."""
    iy = np.clip(_round(theta[:, 1]), 1, shape[0]).astype(int) - 1
    ix = np.clip(_round(theta[:, 2]), 1, shape[1]).astype(int) - 1
    return iy, ix


def _explore(res_im, buffer_mask, exclusion, reject, fit_support, sel, th):
    """getActImPeaks' search image for the next residual peak and its peak."""
    explored = res_im * ~buffer_mask * ~exclusion * ~reject * fit_support * sel
    if fit_support.any() and np.isfinite(explored).any():
        top = np.nanmax(explored)
        if top > th:
            return explored == top
    return np.zeros(explored.shape, bool)


def get_act_im_peaks(  # noqa: C901
    act_im: np.ndarray,
    peak_th: float,
    exclusion_mask: np.ndarray | None = None,
    min_peak_distance: int = 1,
    fit: str = "matlab",
    merge_px: float = 0.0,
    background: tuple[float, float] | None = None,
) -> np.ndarray:
    """Seed sources in a 2-D activity image (port of ``getActImPeaks.m``).

    Parameters
    ----------
    act_im : ndarray of shape (H, W)
        Activity image (NaN outside the imaged pixels).
    peak_th : float
        Threshold in robust SDs (MAD over all finite pixels) above the median.
    exclusion_mask : ndarray of bool, optional
        Pixels where no peak may be placed (e.g. somata). Unlike MATLAB, a
        peak whose fitted centre ends up in the mask is also dropped.
    min_peak_distance : int
        Minimum Chebyshev distance between a residual peak and fitted peaks.
    fit : {'matlab', 'local'}
        ``'matlab'``: MATLAB's fits (all candidates jointly, then each
        residual peak with its whole connected region, every fit stopped
        after MATLAB's evaluation budget). ``'local'``: each Gaussian only
        within 10 px of its start pixel, every fit run to convergence, a
        residual peak refitted with the peaks within 6 px of it (the others
        held), and a rejected residual peak leaves its neighbours' previous
        fit in place. Much faster on large images; the same search
        otherwise.
    merge_px : float
        Also reject a residual peak whose fitted centre ends closer than
        this to another peak's (0: only MATLAB's rule, which rejects a peak
        that does not move). An activity image's peak that falls between
        pixels is otherwise often fitted as two peaks less than a pixel
        apart.
    background : (float, float), optional
        The centre and scale that ``peak_th`` counts from, in place of the
        image's median and robust SD (MATLAB's); e.g. those of a null image
        (:func:`~giant_python.extraction.raster.seeding.null_threshold`).
        The centre is also the level the Gaussians are fitted above.

    Returns
    -------
    ndarray of shape (N, 4)
        ``[amplitude, row, col, sigma]`` of each seed, rows and columns
        0-based (subpixel).
    """
    if fit not in SEED_FITS:
        raise ValueError(f"fit must be one of {SEED_FITS}, not {fit!r}")
    if not merge_px >= 0:
        raise ValueError("merge_px must be nonnegative")
    act = np.asarray(act_im, dtype=float)
    shape = act.shape
    excl = (
        np.zeros(shape, bool)
        if exclusion_mask is None
        else np.asarray(exclusion_mask, bool)
    )
    min_peak_distance = max(1, int(min_peak_distance))
    buffer_se = np.ones((2 * min_peak_distance - 1,) * 2, bool)
    finite = ~np.isnan(act)
    if background is None:
        vals = act[finite]
        mu_bg = float(np.median(vals))
        sigma_bg = float(np.median(np.abs(vals - mu_bg))) / _MAD_SCALE
    else:
        mu_bg, sigma_bg = (float(v) for v in background)
        if not (np.isfinite(mu_bg) and sigma_bg > 0):
            raise ValueError(
                "background must be a finite centre and scale > 0"
            )
    peak_thresh = mu_bg + peak_th * sigma_bg

    explored = act * ~excl
    with np.errstate(invalid="ignore"):
        p_tmp = (ordfilt3x3(explored, 8) > peak_thresh) & (
            explored == ordfilt3x3(explored, 9)
        )
    if not p_tmp.any():
        return np.zeros((0, 4))
    if fit == "local":
        local = _LocalFit(act, (mu_bg, sigma_bg), peak_th, excl, buffer_se)
        theta = local.run(p_tmp, merge_px)
        return _finish(theta, peak_thresh, excl, exclusion_mask is not None)

    p_y, p_x = _find_f(p_tmp)
    amp = act[(p_y - 1).astype(int), (p_x - 1).astype(int)] * _AMP_SCALE
    theta = np.column_stack([amp, p_y, p_x, np.full(p_y.size, 0.35)])
    locs = np.column_stack([p_y, p_x])
    sel = ndimage.binary_dilation(p_tmp, np.ones((9, 9), bool)) & finite
    sy, sx = _find_f(sel)
    lb, ub = _bounds(locs, shape)
    data = act[(sy - 1).astype(int), (sx - 1).astype(int)] - mu_bg
    theta = _fit(theta, sy, sx, data, lb, ub)

    def fit_image(theta_now, ys, xs, fit_im):
        """Write the model of all peaks at the given pixels into fit_im."""
        fit_im[(ys - 1).astype(int), (xs - 1).astype(int)] = (
            _IntegratedGaussians(ys, xs).value(theta_now.ravel(order="F"))
        )
        return fit_im

    def peak_image(theta_now):
        """Mask of the fitted peaks' rounded positions."""
        p_im = np.zeros(shape, bool)
        iy, ix = _rounded_index(theta_now, shape)
        p_im[iy, ix] = True
        return p_im

    p_im = peak_image(theta)
    buffer_mask = ndimage.binary_dilation(p_im, buffer_se)
    fit_im = fit_image(theta, sy, sx, np.zeros(shape))
    res_im = (act - fit_im - mu_bg) / sigma_bg
    fit_support = fit_im > 1e-3
    reject = np.zeros(shape, bool)
    with np.errstate(invalid="ignore"):
        p_tmp = _explore(
            res_im, buffer_mask, excl, reject, fit_support, sel, peak_th
        )

    struct8 = np.ones((3, 3), bool)
    while p_tmp.any():
        c_new, r_new = np.nonzero(p_tmp.T)
        r_new, c_new = int(r_new[0]), int(c_new[0])
        theta = np.vstack(
            [
                theta,
                [act[r_new, c_new] * _AMP_SCALE, r_new + 1, c_new + 1, 0.35],
            ]
        )
        locs = np.vstack([locs, [r_new + 1, c_new + 1]])
        new_idx = theta.shape[0] - 1

        labels, _ = ndimage.label(sel, structure=struct8)
        lab = labels[r_new, c_new]
        if lab == 0:
            raise RuntimeError(
                f"peak at row {r_new + 1}, col {c_new + 1} is not in any "
                "connected component of the fitted pixels"
            )
        sel = labels == lab
        sy, sx = _find_f(sel)
        iy, ix = _rounded_index(theta, shape)
        to_fit = sel[iy, ix]
        data = act[(sy - 1).astype(int), (sx - 1).astype(int)] - mu_bg
        lb, ub = _bounds(locs[to_fit], shape)
        theta[to_fit] = _fit(theta[to_fit], sy, sx, data, lb, ub)

        my_new, mx_new = theta[new_idx, 1], theta[new_idx, 2]
        if (
            abs(my_new - _round(my_new)) < 1e-3
            and abs(mx_new - _round(mx_new)) < 1e-3
        ) or _next_to_another(theta, new_idx, merge_px):
            # the new peak did not move (or ends next to another peak):
            # refit the others without it, reject
            others = to_fit.copy()
            others[new_idx] = False
            if others.any():
                lb, ub = _bounds(locs[others], shape)
                theta[others] = _fit(theta[others], sy, sx, data, lb, ub)
            theta = np.delete(theta, new_idx, axis=0)
            locs = np.delete(locs, new_idx, axis=0)
            reject[r_new, c_new] = True

        p_im = peak_image(theta)
        buffer_mask = ndimage.binary_dilation(p_im, buffer_se)
        fit_im = fit_image(theta, sy, sx, fit_im)
        res_im = (act - fit_im - mu_bg) / sigma_bg
        sel = ndimage.binary_dilation(p_im, np.ones((9, 9), bool)) & finite
        fit_support = fit_im > 1e-3
        with np.errstate(invalid="ignore"):
            p_tmp = _explore(
                res_im, buffer_mask, excl, reject, fit_support, sel, peak_th
            )

    return _finish(theta, peak_thresh, excl, exclusion_mask is not None)


def _next_to_another(theta, idx, merge_px):
    """Whether peak ``idx``'s centre is within ``merge_px`` of another's."""
    if merge_px <= 0 or theta.shape[0] < 2:
        return False
    d = np.hypot(theta[:, 1] - theta[idx, 1], theta[:, 2] - theta[idx, 2])
    d[idx] = np.inf
    return bool(d.min() < merge_px)


def _finish(theta, peak_thresh, excl, masked):
    """Drop peaks too small for the threshold and (if ``masked``) peaks
    whose centre ended in the exclusion mask; 0-based ``N x 4``."""
    s = theta[:, 3]
    adj_thresh = peak_thresh / (
        np.pi / 2 * s**2 * erf(1.0 / (_SQRT2 * s)) ** 2
    )
    theta = theta[~(theta[:, 0] < adj_thresh)]
    if masked and theta.shape[0]:
        # a fit may move a centre up to 1.5 px into the mask (not in MATLAB)
        iy, ix = _rounded_index(theta, excl.shape)
        theta = theta[~excl[iy, ix]]
    out = theta.copy()
    out[:, 1:3] -= 1.0
    return out


def _integral(lo, hi, mu, s, derivs=False):
    """Integral of a unit Gaussian over [lo, hi], elementwise, and (with
    ``derivs``) its derivatives in ``mu`` and ``s``."""
    u_hi = (hi - mu) / (_SQRT2 * s)
    u_lo = (lo - mu) / (_SQRT2 * s)
    val = _SQRT_PI_2 * s * (erf(u_hi) - erf(u_lo))
    if not derivs:
        return val
    e_hi, e_lo = np.exp(-(u_hi**2)), np.exp(-(u_lo**2))
    d_mu = -(e_hi - e_lo)
    d_s = val / s - _SQRT2 * (u_hi * e_hi - u_lo * e_lo)
    return val, d_mu, d_s


class _WindowedGaussians:
    """Integrated Gaussians, each only within a window of its start pixel.

    The local fit's model: peak ``k`` contributes to the data pixels within
    ``window`` pixels (Chebyshev) of its start pixel ``starts[k]`` and
    nowhere else, so the Jacobian is sparse. Positions are parameters as
    offsets from the start pixel: ``x`` holds all amplitudes, then all row
    offsets, column offsets and widths.
    """

    def __init__(self, idx_im, edges, starts, window, rows):
        """Pair each peak with the data pixels in its window.

        ``idx_im`` maps a pixel to its index in the data (-1 elsewhere),
        ``edges`` holds every data pixel's ``(y_lo, y_hi, x_lo, x_hi)``,
        ``starts`` the peaks' 0-based start pixels, ``rows`` the data
        indices the model is evaluated at.
        """
        h, w = idx_im.shape
        local = np.full(edges[0].size, -1, np.int64)
        local[rows] = np.arange(rows.size)
        p_parts, k_parts = [], []
        for k, (cy, cx) in enumerate(np.asarray(starts, int)):
            win = idx_im[
                max(cy - window, 0) : min(cy + window + 1, h),
                max(cx - window, 0) : min(cx + window + 1, w),
            ].ravel()
            win = local[win[win >= 0]]
            win = win[win >= 0]
            p_parts.append(win)
            k_parts.append(np.full(win.size, k))
        self.p = np.concatenate(p_parts) if p_parts else np.zeros(0, int)
        self.k = np.concatenate(k_parts) if k_parts else np.zeros(0, int)
        self.m, self.n = rows.size, len(starts)
        src = rows[self.p]
        self.lo_y, self.hi_y = edges[0][src], edges[1][src]
        self.lo_x, self.hi_x = edges[2][src], edges[3][src]
        st = np.asarray(starts, float).reshape(-1, 2)
        self.y0, self.x0 = st[:, 0] + 1.0, st[:, 1] + 1.0

    def _pairs(self, x):
        """Amplitude, centre and width of the peak of every pixel pair."""
        n, k = self.n, self.k
        s = np.maximum(x[3 * n :], np.finfo(float).eps)
        return (
            x[:n][k],
            (self.y0 + x[n : 2 * n])[k],
            (self.x0 + x[2 * n : 3 * n])[k],
            s[k],
        )

    def value(self, x):
        """Model value at each of the rows."""
        a, my, mx, s = self._pairs(x)
        vy = _integral(self.lo_y, self.hi_y, my, s)
        vx = _integral(self.lo_x, self.hi_x, mx, s)
        return np.bincount(self.p, a * vy * vx, minlength=self.m)

    def jacobian(self, x, dense):
        """Jacobian (rows x parameters), dense or sparse (CSR)."""
        a, my, mx, s = self._pairs(x)
        vy, dmy, dsy = _integral(self.lo_y, self.hi_y, my, s, True)
        vx, dmx, dsx = _integral(self.lo_x, self.hi_x, mx, s, True)
        parts = (
            vy * vx,
            a * dmy * vx,
            a * vy * dmx,
            a * (dsy * vx + vy * dsx),
        )
        n = self.n
        cols = [self.k + j * n for j in range(4)]
        if dense:
            jac = np.zeros((self.m, 4 * n))
            for c, v in zip(cols, parts):
                jac[self.p, c] = v  # a pixel is in a peak's window once
            return jac
        return sparse.csr_matrix(
            (
                np.concatenate(parts),
                (np.tile(self.p, 4), np.concatenate(cols)),
            ),
            shape=(self.m, 4 * n),
        )


def _to_offsets(theta, starts):
    """``N x 4`` [A, mu_y, mu_x, sigma] (1-based) as the local fit's x."""
    return np.concatenate(
        [
            theta[:, 0],
            theta[:, 1] - (starts[:, 0] + 1),
            theta[:, 2] - (starts[:, 1] + 1),
            theta[:, 3],
        ]
    )


def _from_offsets(x, starts):
    """The local fit's x as ``N x 4`` [A, mu_y, mu_x, sigma] (1-based)."""
    n = starts.shape[0]
    return np.column_stack(
        [
            x[:n],
            x[n : 2 * n] + starts[:, 0] + 1,
            x[2 * n : 3 * n] + starts[:, 1] + 1,
            x[3 * n :],
        ]
    )


def _offset_bounds(starts, shape):
    """getActImPeaks' bounds as offsets: within 1.5 px of the start pixel
    and inside the image; widths in [0, 5]."""
    n = starts.shape[0]
    y0, x0 = starts[:, 0] + 1.0, starts[:, 1] + 1.0
    lb = np.concatenate(
        [
            np.zeros(n),
            np.maximum(0.5 - y0, -1.5),
            np.maximum(0.5 - x0, -1.5),
            np.zeros(n),
        ]
    )
    ub = np.concatenate(
        [
            np.full(n, np.inf),
            np.minimum(shape[0] + 0.5 - y0, 1.5),
            np.minimum(shape[1] + 0.5 - x0, 1.5),
            np.full(n, 5.0),
        ]
    )
    return lb, ub


def _cover(cover, pix, sign):
    """Add ``sign`` to the 9 x 9 squares around pixels (``cover`` is padded
    by 4 on every side)."""
    for r, c in pix:
        cover[r : r + 9, c : c + 9] += sign


class _LocalFit:
    """getActImPeaks' search with windowed, converged, local fits.

    See :func:`get_act_im_peaks` (``fit='local'``). Peaks are kept as
    1-based ``theta`` rows with their 0-based start pixels.
    """

    def __init__(self, act, background, peak_th, excl, buffer_se):
        """Index the image's finite pixels and their pixel edges."""
        self.act, self.shape = act, act.shape
        self.mu_bg, self.sigma_bg = background
        self.peak_th, self.excl, self.buffer_se = peak_th, excl, buffer_se
        self.finite = ~np.isnan(act)
        cols, rows = np.nonzero(self.finite.T)  # column-major, as MATLAB
        self.ry, self.rx = rows, cols
        self.idx_im = np.full(self.shape, -1, np.int64)
        self.idx_im[rows, cols] = np.arange(rows.size)
        yu, yi = np.unique(rows + 1.0, return_inverse=True)
        xu, xi = np.unique(cols + 1.0, return_inverse=True)
        ye, xe = _edges(yu), _edges(xu)
        self.edges = (ye[:-1][yi], ye[1:][yi], xe[:-1][xi], xe[1:][xi])
        self.data = act[rows, cols] - self.mu_bg

    def rows_of(self, mask):
        """Sorted data indices of the pixels of a mask."""
        r = self.idx_im[mask]
        return np.sort(r[r >= 0])

    def solve(self, theta, starts, rows, fixed=None):
        """Fit peaks to the data at ``rows`` (minus ``fixed``)."""
        model = _WindowedGaussians(
            self.idx_im, self.edges, starts, _LOCAL_WINDOW, rows
        )
        target = self.data[rows]
        if fixed is not None:
            target = target - fixed
        lb, ub = _offset_bounds(starts, self.shape)
        start = np.clip(_to_offsets(theta, starts), lb, ub)

        def run(dense):
            """least_squares with a dense (SVD) or sparse (lsmr) solver."""
            return least_squares(
                lambda x: model.value(x) - target,
                start,
                jac=lambda x: model.jacobian(x, dense),
                bounds=(lb, ub),
                method="trf",
                tr_solver="exact" if dense else "lsmr",
                x_scale="jac",
                ftol=_LOCAL_TOL,
                xtol=_LOCAL_TOL,
                gtol=_LOCAL_TOL,
                max_nfev=100 * start.size,
            )

        try:
            res = run(start.size <= _LOCAL_DENSE_MAX)
        except np.linalg.LinAlgError:
            # LAPACK's SVD now and then fails to converge (see _fit); the
            # sparse solver needs none
            res = run(False)
        return _from_offsets(res.x, starts)

    def model_at(self, theta, starts, rows):
        """All peaks' model at ``rows`` (peaks whose window reaches them)."""
        if rows.size == 0 or theta.shape[0] == 0:
            return np.zeros(rows.size)
        ry, rx = self.ry[rows], self.rx[rows]
        w = _LOCAL_WINDOW
        near = (
            (starts[:, 0] >= ry.min() - w)
            & (starts[:, 0] <= ry.max() + w)
            & (starts[:, 1] >= rx.min() - w)
            & (starts[:, 1] <= rx.max() + w)
        )
        if not near.any():
            return np.zeros(rows.size)
        model = _WindowedGaussians(
            self.idx_im, self.edges, starts[near], w, rows
        )
        return model.value(_to_offsets(theta[near], starts[near]))

    def explore(self, fit_im, pix, cover, reject):
        """The next residual peak (mask) and the fitted peaks' region."""
        sel = (cover[4:-4, 4:-4] > 0) & self.finite
        p_im = np.zeros(self.shape, bool)
        p_im[pix[:, 0], pix[:, 1]] = True
        buffer_mask = (
            p_im
            if self.buffer_se.shape == (1, 1)
            else ndimage.binary_dilation(p_im, self.buffer_se)
        )
        res_im = (self.act - fit_im - self.mu_bg) / self.sigma_bg
        with np.errstate(invalid="ignore"):
            new = _explore(
                res_im,
                buffer_mask,
                self.excl,
                reject,
                fit_im > 1e-3,
                sel,
                self.peak_th,
            )
        return new, sel

    def run(self, cand, merge_px):
        """Fit the candidates, then add residual peaks; ``N x 4`` theta."""
        c, r = np.nonzero(cand.T)
        starts = np.column_stack([r, c]).astype(int)
        theta = np.column_stack(
            [
                self.act[r, c] * _AMP_SCALE,
                starts + 1.0,
                np.full(len(starts), _SIGMA0),
            ]
        )
        sel0 = ndimage.binary_dilation(cand, np.ones((9, 9), bool))
        rows = self.rows_of(sel0 & self.finite)
        theta = self.solve(theta, starts, rows)
        # the model everywhere the peaks' windows reach
        side = 2 * _LOCAL_WINDOW + 1
        reach = self.rows_of(
            ndimage.binary_dilation(cand, np.ones((side, side), bool))
        )
        fit_im = np.zeros(self.shape)
        fit_im[self.ry[reach], self.rx[reach]] = self.model_at(
            theta, starts, reach
        )
        pix = np.column_stack(_rounded_index(theta, self.shape))
        # how many peaks' 9 x 9 squares cover each pixel (padded by 4)
        cover = np.zeros((self.shape[0] + 8, self.shape[1] + 8), np.int32)
        _cover(cover, pix, 1)
        reject = np.zeros(self.shape, bool)
        new, sel = self.explore(fit_im, pix, cover, reject)
        while new.any():
            c_n, r_n = np.nonzero(new.T)
            rc = (int(r_n[0]), int(c_n[0]))
            added = self._add(theta, starts, pix, sel, fit_im, rc, merge_px)
            if added is None:
                reject[rc] = True
            else:
                theta, starts, changed, old = added
                _cover(cover, old, -1)
                pix = np.column_stack(_rounded_index(theta, self.shape))
                _cover(cover, pix[changed], 1)
            new, sel = self.explore(fit_im, pix, cover, reject)
        return theta

    def _add(self, theta, starts, pix, sel, fit_im, rc, merge_px):
        """Fit a residual peak with its neighbours; None if rejected.

        Updates ``fit_im`` in place when the peak is kept and returns the
        new theta and starts, the indices of the peaks that changed and the
        pixels those had before.
        """
        r_n, c_n = rc
        labels, _ = ndimage.label(sel, structure=np.ones((3, 3), bool))
        comp = labels == labels[r_n, c_n]
        in_comp = comp[pix[:, 0], pix[:, 1]]
        dist = np.hypot(theta[:, 1] - (r_n + 1), theta[:, 2] - (c_n + 1))
        free = in_comp & (dist <= _LOCAL_REFIT_RADIUS)
        f_starts = np.vstack([starts[free], [[r_n, c_n]]])
        f_theta = np.vstack(
            [
                theta[free],
                [[self.act[r_n, c_n] * _AMP_SCALE, r_n + 1, c_n + 1, _SIGMA0]],
            ]
        )
        region = np.zeros(self.shape, bool)
        w = _LOCAL_WINDOW
        for r, c in f_starts:
            region[max(r - w, 0) : r + w + 1, max(c - w, 0) : c + w + 1] = 1
        rows = self.rows_of(region & comp)
        # every other peak whose window reaches these pixels is held,
        # those of neighbouring regions too (model_at leaves out the rest)
        held = np.flatnonzero(~free)
        fixed = None
        if held.size:
            fixed = self.model_at(theta[held], starts[held], rows)
        fitted = self.solve(f_theta, f_starts, rows, fixed)
        my, mx = fitted[-1, 1], fitted[-1, 2]
        if abs(my - _round(my)) < 1e-3 and abs(mx - _round(mx)) < 1e-3:
            return None  # MATLAB's rule: the new peak did not move
        others_y = np.concatenate([theta[~free, 1], fitted[:-1, 1]])
        others_x = np.concatenate([theta[~free, 2], fitted[:-1, 2]])
        if merge_px > 0 and others_y.size:
            if np.hypot(others_y - my, others_x - mx).min() < merge_px:
                return None
        old = pix[free]
        theta = theta.copy()
        theta[free] = fitted[:-1]
        theta = np.vstack([theta, fitted[-1:]])
        starts = np.vstack([starts, [[r_n, c_n]]])
        changed = np.append(np.flatnonzero(free), theta.shape[0] - 1)
        # refresh the model wherever the refitted peaks' windows reach,
        # beyond their region too
        reach = self.rows_of(region)
        fit_im[self.ry[reach], self.rx[reach]] = self.model_at(
            theta, starts, reach
        )
        return theta, starts, changed, old
