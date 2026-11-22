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

Coordinates follow MATLAB internally (1-based pixel centres); the public
function returns 0-based rows and columns.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndimage
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import least_squares
from scipy.special import erf

_AMP_SCALE = 1.0 / 0.75
_MAD_SCALE = (
    0.6741891400433162  # MATLAB's getActImPeaks divides mad(x, 1) by this
)
_MAX_FUN_EVALS = 5000
_SQRT_PI_2 = np.sqrt(np.pi / 2)
_SQRT2 = np.sqrt(2.0)


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
    def _axis(lo, hi, mu, s):
        """Integral over [lo, hi] and its derivatives in mu and sigma."""
        u_hi = (hi[:, None] - mu[None, :]) / (_SQRT2 * s[None, :])
        u_lo = (lo[:, None] - mu[None, :]) / (_SQRT2 * s[None, :])
        e_hi, e_lo = np.exp(-(u_hi**2)), np.exp(-(u_lo**2))
        val = _SQRT_PI_2 * s[None, :] * (erf(u_hi) - erf(u_lo))
        d_mu = -(e_hi - e_lo)
        d_s = val / s[None, :] - _SQRT2 * (u_hi * e_hi - u_lo * e_lo)
        return val, d_mu, d_s

    def parts(self, theta: np.ndarray):
        """Per-axis integrals (pixels x peaks) and their derivatives."""
        n = theta.size // 4
        a, my, mx = theta[:n], theta[n : 2 * n], theta[2 * n : 3 * n]
        s = np.maximum(theta[3 * n :], np.finfo(float).eps)
        iy = self._axis(self.y_lo, self.y_hi, my, s)
        ix = self._axis(self.x_lo, self.x_hi, mx, s)
        return a, iy, ix

    def value(self, theta: np.ndarray) -> np.ndarray:
        """Model value at each selected pixel."""
        a, (vy, _, _), (vx, _, _) = self.parts(theta)
        return (vy * vx) @ a

    def jacobian(self, theta: np.ndarray) -> np.ndarray:
        """Jacobian (pixels x parameters), parameters column-major."""
        a, (vy, dmy, dsy), (vx, dmx, dsx) = self.parts(theta)
        return np.hstack(
            [vy * vx, a * dmy * vx, a * vy * dmx, a * (dsy * vx + vy * dsx)]
        )


def _fit(theta0, y, x, data, lb, ub):
    """lsqcurvefit-like bounded least squares of integrated Gaussians."""
    model = _IntegratedGaussians(y, x)
    lb_f, ub_f = lb.ravel(order="F"), ub.ravel(order="F")
    x0 = np.clip(theta0.ravel(order="F").astype(float), lb_f, ub_f)
    n_par = x0.size
    max_iter = max(1, (_MAX_FUN_EVALS - (n_par + 1)) // (n_par + 1))
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


def _rounded_index(theta: np.ndarray, shape) -> tuple[np.ndarray, np.ndarray]:
    """0-based pixel of each fitted peak (rounded and clamped)."""
    iy = np.clip(np.round(theta[:, 1]), 1, shape[0]).astype(int) - 1
    ix = np.clip(np.round(theta[:, 2]), 1, shape[1]).astype(int) - 1
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

    Returns
    -------
    ndarray of shape (N, 4)
        ``[amplitude, row, col, sigma]`` of each seed, rows and columns
        0-based (subpixel).
    """
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
    vals = act[finite]
    mu_bg = float(np.median(vals))
    sigma_bg = float(np.median(np.abs(vals - mu_bg))) / _MAD_SCALE
    peak_thresh = mu_bg + peak_th * sigma_bg

    explored = act * ~excl
    with np.errstate(invalid="ignore"):
        p_tmp = (ordfilt3x3(explored, 8) > peak_thresh) & (
            explored == ordfilt3x3(explored, 9)
        )
    if not p_tmp.any():
        return np.zeros((0, 4))

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
            abs(my_new - np.round(my_new)) < 1e-3
            and abs(mx_new - np.round(mx_new)) < 1e-3
        ):
            # the new peak did not move: refit the others without it, reject
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

    s = theta[:, 3]
    adj_thresh = peak_thresh / (
        np.pi / 2 * s**2 * erf(1.0 / (_SQRT2 * s)) ** 2
    )
    theta = theta[~(theta[:, 0] < adj_thresh)]
    if exclusion_mask is not None and theta.shape[0]:
        # a fit may move a centre up to 1.5 px into the mask (not in MATLAB)
        iy, ix = _rounded_index(theta, shape)
        theta = theta[~excl[iy, ix]]
    out = theta.copy()
    out[:, 1:3] -= 1.0
    return out
