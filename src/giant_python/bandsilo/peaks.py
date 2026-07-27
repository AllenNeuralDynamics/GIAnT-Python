"""Gaussian peak detection in the activity image.

Ported from the peak-detection kernels of
``extractSLAP2IntegrationSources.py`` (ref 307-712, originally
``getActImPeaks.m``). :func:`get_act_im_peaks` finds source seeds in the 3-D
(Z, H, W) activity image produced by
:mod:`giant_python.bandsilo.summary_images`; those seeds initialize the Phase-6
NMF localization.

Detection fits integrated isotropic 2-D Gaussians (each Gaussian integrated
over unit pixels) to the activity image with a bounded Levenberg-Marquardt
solver (:func:`_lsq_curvefit`, using the analytical Jacobian from
:func:`_gaussian_peaks_integrated_val_jac`). Per plane, an initial set of local
maxima is fit jointly, then residual peaks are added one per connected
component per round until none exceed the threshold, and finally
under-amplitude peaks are removed. An optional exclusion mask (e.g. user soma
ROIs) suppresses detection in masked regions.

The reference's nested helpers (``_make_bounds``/``_peak_mask``/
``_buffer_mask`` and the per-component new-peak step) are lifted to
module-level functions here so each is independently testable.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.ndimage as ndimage
from scipy.special import erf

# Amplitude rescale applied to raw activity-image peak values when seeding the
# Gaussian fit (matches the reference ``AMP_SCALE = 1 / 0.75``).
_AMP_SCALE = 1.0 / 0.75

# MAD -> Gaussian-sigma scale factor (median absolute deviation / 0.6745).
_MAD_TO_SIGMA = 0.6741891400433162


def gaussian_peaks_integrated(
    theta: np.ndarray, yxdata: np.ndarray
) -> np.ndarray:
    """Evaluate integrated isotropic 2-D Gaussians at selected pixels.

    Each Gaussian is integrated over unit pixels on a regular grid (port of
    ``gaussianPeaksIntegrated``). Only the ``M`` requested pixels are evaluated
    (the full grid is never materialized).

    Parameters
    ----------
    theta : ndarray of shape (N, 4)
        Per-Gaussian ``[amp, mu_y, mu_x, sigma]``.
    yxdata : ndarray of shape (M, 2)
        Pixel-center ``[y, x]`` coordinates.

    Returns
    -------
    ndarray of shape (M,)
        Predicted value at each pixel: ``sum_n A_n * Iy[y, n] * Ix[x, n]``.
    """
    x_int = yxdata[:, 1].astype(np.intp)
    y_int = yxdata[:, 0].astype(np.intp)
    x_min = int(x_int.min())
    x_max = int(x_int.max())
    y_min = int(y_int.min())
    y_max = int(y_int.max())
    x_idx = x_int - x_min
    y_idx = y_int - y_min

    amp = theta[:, 0]
    my = theta[:, 1]
    mx = theta[:, 2]
    s = np.maximum(theta[:, 3], np.finfo(float).eps)

    c = np.sqrt(np.pi / 2)
    rt2 = np.sqrt(2.0)

    xc = np.arange(x_min, x_max + 1, dtype=float)
    xl = (xc - 0.5)[:, np.newaxis]
    xr = (xc + 0.5)[:, np.newaxis]
    ix = (
        c
        * s[np.newaxis, :]
        * (
            erf((xr - mx[np.newaxis, :]) / (rt2 * s[np.newaxis, :]))
            - erf((xl - mx[np.newaxis, :]) / (rt2 * s[np.newaxis, :]))
        )
    )

    yc = np.arange(y_min, y_max + 1, dtype=float)
    yb = (yc - 0.5)[:, np.newaxis]
    yt = (yc + 0.5)[:, np.newaxis]
    iy = (
        c
        * s[np.newaxis, :]
        * (
            erf((yt - my[np.newaxis, :]) / (rt2 * s[np.newaxis, :]))
            - erf((yb - my[np.newaxis, :]) / (rt2 * s[np.newaxis, :]))
        )
    )

    return (iy[y_idx, :] * ix[x_idx, :]) @ amp


def _gaussian_peaks_integrated_val_jac(
    theta: np.ndarray, yxdata: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrated Gaussians with their analytical Jacobian.

    Shares the integrated-profile intermediates between the forward value and
    the Jacobian so the LM solver needs one evaluation per iteration instead of
    ``4*N`` finite differences.

    Parameters
    ----------
    theta : ndarray of shape (N, 4)
        Per-Gaussian ``[amp, mu_y, mu_x, sigma]``.
    yxdata : ndarray of shape (M, 2)
        Pixel-center ``[y, x]`` coordinates.

    Returns
    -------
    val : ndarray of shape (M,)
        Forward value at each pixel.
    jac : ndarray of shape (M, 4*N)
        Column-major-per-Gaussian Jacobian (columns ``0::4`` are ``d/dA``,
        ``1::4`` are ``d/dmu_y``, ``2::4`` are ``d/dmu_x``, ``3::4`` are
        ``d/dsigma``).
    """
    x_int = yxdata[:, 1].astype(np.intp)
    y_int = yxdata[:, 0].astype(np.intp)
    m = len(x_int)
    n = theta.shape[0]

    x_min = int(x_int.min())
    x_max = int(x_int.max())
    y_min = int(y_int.min())
    y_max = int(y_int.max())
    x_idx = x_int - x_min
    y_idx = y_int - y_min

    amp = theta[:, 0]
    my = theta[:, 1]
    mx = theta[:, 2]
    s = np.maximum(theta[:, 3], np.finfo(float).eps)

    c = np.sqrt(np.pi / 2)
    rt2 = np.sqrt(2.0)
    inv_rt2s = 1.0 / (rt2 * s[np.newaxis, :])

    xc = np.arange(x_min, x_max + 1, dtype=float)
    xl = (xc - 0.5)[:, np.newaxis]
    xr = (xc + 0.5)[:, np.newaxis]
    ux_l = (xl - mx[np.newaxis, :]) * inv_rt2s
    ux_r = (xr - mx[np.newaxis, :]) * inv_rt2s
    erf_ux_l = erf(ux_l)
    erf_ux_r = erf(ux_r)
    ix = c * s[np.newaxis, :] * (erf_ux_r - erf_ux_l)

    yc = np.arange(y_min, y_max + 1, dtype=float)
    yb = (yc - 0.5)[:, np.newaxis]
    yt = (yc + 0.5)[:, np.newaxis]
    uy_b = (yb - my[np.newaxis, :]) * inv_rt2s
    uy_t = (yt - my[np.newaxis, :]) * inv_rt2s
    erf_uy_b = erf(uy_b)
    erf_uy_t = erf(uy_t)
    iy = c * s[np.newaxis, :] * (erf_uy_t - erf_uy_b)

    iy_m = iy[y_idx, :]
    ix_m = ix[x_idx, :]

    dval_da = iy_m * ix_m
    val = dval_da @ amp

    exp_uyb2 = np.exp(-(uy_b**2))
    exp_uyt2 = np.exp(-(uy_t**2))
    d_iy_dmy_m = (exp_uyb2 - exp_uyt2)[y_idx, :]
    dval_dmy = amp[np.newaxis, :] * d_iy_dmy_m * ix_m

    exp_uxl2 = np.exp(-(ux_l**2))
    exp_uxr2 = np.exp(-(ux_r**2))
    d_ix_dmx_m = (exp_uxl2 - exp_uxr2)[x_idx, :]
    dval_dmx = amp[np.newaxis, :] * iy_m * d_ix_dmx_m

    sqrt2 = rt2
    d_iy_ds_m = (
        iy / s[np.newaxis, :] + sqrt2 * (uy_b * exp_uyb2 - uy_t * exp_uyt2)
    )[y_idx, :]
    d_ix_ds_m = (
        ix / s[np.newaxis, :] + sqrt2 * (ux_l * exp_uxl2 - ux_r * exp_uxr2)
    )[x_idx, :]
    dval_ds = amp[np.newaxis, :] * (d_iy_ds_m * ix_m + iy_m * d_ix_ds_m)

    jac = np.empty((m, 4 * n), dtype=float)
    jac[:, 0::4] = dval_da
    jac[:, 1::4] = dval_dmy
    jac[:, 2::4] = dval_dmx
    jac[:, 3::4] = dval_ds

    return val, jac


def _lsq_curvefit(
    theta0: np.ndarray,
    xdata: np.ndarray,
    ydata: np.ndarray,
    lb_flat: np.ndarray,
    ub_flat: np.ndarray,
    max_nfev: int = 5000,
) -> np.ndarray:
    """Bounded Levenberg-Marquardt fit of integrated Gaussians.

    Reimplements the subset of MATLAB ``lsqcurvefit`` (trust-region-reflective
    defaults) used by ``getActImPeaks.m``, but with the analytical Jacobian so
    each iteration is a single value+Jacobian evaluation.

    Parameters
    ----------
    theta0 : ndarray of shape (N, 4)
        Initial ``[amp, mu_y, mu_x, sigma]`` per Gaussian.
    xdata : ndarray of shape (M, 2)
        Pixel-center ``[y, x]`` coordinates.
    ydata : ndarray of shape (M,)
        Observed values (``act_im(sel) - mu_bg``).
    lb_flat, ub_flat : ndarray of shape (4*N,)
        Row-major-flattened lower/upper bounds.
    max_nfev : int
        Cap on value+Jacobian evaluations.

    Returns
    -------
    ndarray of shape (N, 4)
        Optimized parameters.
    """
    x = np.clip(theta0.ravel().copy(), lb_flat, ub_flat)

    val, jac = _gaussian_peaks_integrated_val_jac(x.reshape(-1, 4), xdata)
    r = val - ydata
    cost = np.dot(r, r)
    nfev = 1

    lam = 1e-2  # initial damping (MATLAB InitDamping default)

    for _ in range(400):  # MATLAB MaxIter default
        if nfev >= max_nfev:
            break

        jtj = jac.T @ jac
        jtr = jac.T @ r
        diag_jtj = np.maximum(np.diag(jtj), 1e-8)

        delta = np.linalg.solve(jtj + lam * np.diag(diag_jtj), -jtr)
        x_new = np.clip(x + delta, lb_flat, ub_flat)

        val_new, jac_new = _gaussian_peaks_integrated_val_jac(
            x_new.reshape(-1, 4), xdata
        )
        r_new = val_new - ydata
        cost_new = np.dot(r_new, r_new)
        nfev += 1

        if cost_new < cost:
            step_norm = np.max(np.abs(x_new - x))
            rel_cost_drop = (cost - cost_new) / max(cost, 1.0)

            x = x_new
            r = r_new
            jac = jac_new
            cost = cost_new
            lam = max(lam * 0.1, 1e-10)

            if step_norm < 1e-6 or rel_cost_drop < 1e-6:
                break
        else:
            lam = min(lam * 10.0, 1e10)

    return x.reshape(-1, 4)


def _make_bounds(
    plocs: np.ndarray, height: int, width: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return flattened LM bounds for peaks located at ``plocs``.

    Amplitude is unbounded above; means are constrained to +/- 1.5 px of the
    seed (clipped to the image); sigma to ``[0.35, 5.0]``.

    Parameters
    ----------
    plocs : ndarray of shape (N, 2)
        Seed ``[y, x]`` locations.
    height, width : int
        Image extents (for clipping the mean bounds).

    Returns
    -------
    lb_flat, ub_flat : ndarray of shape (4*N,)
        Row-major-flattened lower/upper bounds.
    """
    n = plocs.shape[0]
    lb = np.column_stack(
        [
            np.zeros(n),
            np.maximum(0, plocs[:, 0] - 1.5),
            np.maximum(0, plocs[:, 1] - 1.5),
            np.ones(n) * 0.35,
        ]
    )
    ub = np.column_stack(
        [
            np.full(n, np.inf),
            np.minimum(height - 1, plocs[:, 0] + 1.5),
            np.minimum(width - 1, plocs[:, 1] + 1.5),
            np.full(n, 5.0),
        ]
    )
    return lb.ravel(), ub.ravel()


def _peak_mask(tf: np.ndarray, height: int, width: int) -> np.ndarray:
    """Return a boolean image marking each Gaussian's rounded center."""
    pim = np.zeros((height, width), dtype=bool)
    if tf.shape[0] > 0:
        iy = np.clip(np.round(tf[:, 1]).astype(int), 0, height - 1)
        ix = np.clip(np.round(tf[:, 2]).astype(int), 0, width - 1)
        pim[iy, ix] = True
    return pim


def _buffer_mask(pim: np.ndarray, buffer_size: int) -> np.ndarray:
    """Dilate the peak mask by a ``buffer_size`` square (identity if <= 0)."""
    if buffer_size > 0:
        return ndimage.binary_dilation(
            pim, structure=np.ones((buffer_size, buffer_size))
        )
    return pim


def _process_cc_new_peak(
    new_peak,
    thetaf: np.ndarray,
    p_locs: np.ndarray,
    labeled_full: np.ndarray,
    act_im_2d: np.ndarray,
    mu_bg: float,
    reject_mask: np.ndarray,
    height: int,
    width: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Add and fit one new residual peak within its connected component.

    Appends the candidate to the parameter set, refits every Gaussian whose
    center lands in the component, and — if the new peak barely moved (its
    center stayed near-integer) — rejects it (refitting the others without it
    and marking its pixel so it is not re-proposed).

    Parameters
    ----------
    new_peak : tuple
        ``((pY, pX), cc_label)`` for the candidate.
    thetaf : ndarray of shape (N, 4)
        Current Gaussian parameters.
    p_locs : ndarray of shape (N, 2)
        Current seed locations.
    labeled_full : ndarray
        Connected-component labels of the current support.
    act_im_2d : ndarray
        The plane's activity image.
    mu_bg : float
        Background median subtracted from the fit target.
    reject_mask : ndarray of bool
        Mutated in place: the rejected peak's pixel is set True.
    height, width : int
        Image extents.

    Returns
    -------
    thetaf, p_locs, cc_mask, cc_yx
        Updated parameters/locations plus the component mask and its pixel
        coordinates (for the caller's per-round fit-image update).
    """
    (py_new, px_new), cc_label = new_peak
    amp_new = act_im_2d[py_new, px_new] * _AMP_SCALE

    n_before = thetaf.shape[0]
    thetaf = np.vstack([thetaf, [amp_new, float(py_new), float(px_new), 0.5]])
    p_locs = np.vstack([p_locs, [float(py_new), float(px_new)]])
    new_idx = n_before

    cc_mask = labeled_full == cc_label
    cc_yx = np.column_stack(np.where(cc_mask)).astype(float)
    cc_vals = act_im_2d[cc_mask] - mu_bg

    iy = np.clip(np.round(thetaf[:, 1]).astype(int), 0, height - 1)
    ix = np.clip(np.round(thetaf[:, 2]).astype(int), 0, width - 1)
    in_cc = cc_mask[iy, ix]

    lb_cc, ub_cc = _make_bounds(p_locs[in_cc], height, width)
    thetaf[in_cc] = _lsq_curvefit(
        thetaf[in_cc], cc_yx, cc_vals, lb_cc, ub_cc, max_nfev=5000
    )

    mu_y_new = thetaf[new_idx, 1]
    mu_x_new = thetaf[new_idx, 2]
    if (
        abs(mu_y_new - round(mu_y_new)) < 1e-3
        and abs(mu_x_new - round(mu_x_new)) < 1e-3
    ):
        refit_mask = in_cc.copy()
        refit_mask[new_idx] = False
        if np.any(refit_mask):
            lb_rf, ub_rf = _make_bounds(p_locs[refit_mask], height, width)
            thetaf[refit_mask] = _lsq_curvefit(
                thetaf[refit_mask], cc_yx, cc_vals, lb_rf, ub_rf, max_nfev=5000
            )
        thetaf = np.delete(thetaf, new_idx, axis=0)
        p_locs = np.delete(p_locs, new_idx, axis=0)
        reject_mask[py_new, px_new] = True

    return thetaf, p_locs, cc_mask, cc_yx


def _refine_residual_peaks(
    thetaf: np.ndarray,
    p_locs: np.ndarray,
    act_im_2d: np.ndarray,
    exclusion_mask: np.ndarray,
    mu_bg: float,
    sigma_bg: float,
    peak_th: float,
    buffer_size: int,
    height: int,
    width: int,
    buffer_mask: np.ndarray,
    res_im: np.ndarray,
    fit_support: np.ndarray,
    act_sel_pix: np.ndarray,
    fit_im: np.ndarray,
) -> np.ndarray:
    """Iteratively add residual peaks, one per connected component per round.

    Each round finds the strongest normalized-residual pixel in every component
    and, for those above ``peak_th``, adds and fits a new Gaussian (see
    :func:`_process_cc_new_peak`). The fit image, residual, support, and buffer
    are rebuilt once per round. Terminates when no component yields a new peak.

    Parameters
    ----------
    thetaf, p_locs : ndarray
        Current Gaussian parameters and seed locations.
    act_im_2d, exclusion_mask : ndarray
        The plane's activity image and its exclusion mask.
    mu_bg, sigma_bg : float
        Background median and MAD-scaled sigma.
    peak_th : float
        Residual threshold (in sigma) for accepting a new peak.
    buffer_size : int
        Peak-buffer dilation size.
    height, width : int
        Image extents.
    buffer_mask, res_im, fit_support, act_sel_pix, fit_im : ndarray
        Initial per-round state from the first fit.

    Returns
    -------
    ndarray of shape (N, 4)
        The refined Gaussian parameters.
    """
    reject_mask = np.zeros((height, width), dtype=bool)
    labeled_full, _ = ndimage.label(act_sel_pix)

    while True:
        e = res_im.copy()
        e[buffer_mask | exclusion_mask | reject_mask | ~fit_support] = -np.inf
        e[np.isnan(e)] = -np.inf

        n_labels = int(labeled_full.max())
        if n_labels == 0:
            break

        label_ids = list(range(1, n_labels + 1))
        max_vals = ndimage.maximum(e, labeled_full, label_ids)
        max_pos = ndimage.maximum_position(e, labeled_full, label_ids)

        new_peaks = [
            (max_pos[i], label_ids[i])
            for i in range(n_labels)
            if max_vals[i] > peak_th
        ]
        if not new_peaks:
            break

        modified_ccs = []
        for new_peak in new_peaks:
            thetaf, p_locs, cc_mask, cc_yx = _process_cc_new_peak(
                new_peak,
                thetaf,
                p_locs,
                labeled_full,
                act_im_2d,
                mu_bg,
                reject_mask,
                height,
                width,
            )
            modified_ccs.append((cc_mask, cc_yx))

        p_im = _peak_mask(thetaf, height, width)
        buffer_mask = _buffer_mask(p_im, buffer_size)
        for cc_mask, cc_yx in modified_ccs:
            fit_im[cc_mask] = gaussian_peaks_integrated(thetaf, cc_yx)
        res_im = (act_im_2d - fit_im - mu_bg) / sigma_bg
        fit_support = fit_im > 1e-3

        act_sel_pix = ndimage.binary_dilation(p_im, structure=np.ones((9, 9)))
        act_sel_pix &= ~np.isnan(act_im_2d)
        labeled_full, _ = ndimage.label(act_sel_pix)

    return thetaf


def detect_peaks_2d(
    act_im_2d: np.ndarray,
    exclusion_mask: np.ndarray,
    mu_bg: float,
    sigma_bg: float,
    peak_thresh: float,
    peak_th: float,
    buffer_size: int = 0,
) -> np.ndarray:
    """Detect Gaussian peaks in one activity-image plane.

    Finds initial local maxima (above ``peak_thresh``), fits them jointly, then
    iteratively adds residual peaks (:func:`_refine_residual_peaks`), and
    finally drops peaks whose amplitude falls below a sigma-adjusted threshold.

    Parameters
    ----------
    act_im_2d : ndarray of shape (H, W)
        One activity-image plane (may contain NaNs).
    exclusion_mask : ndarray of bool, shape (H, W)
        Pixels to exclude from detection.
    mu_bg, sigma_bg : float
        Global background median and MAD-scaled sigma.
    peak_thresh : float
        Absolute detection threshold ``mu_bg + peak_th * sigma_bg``.
    peak_th : float
        Threshold in sigma for residual peaks.
    buffer_size : int
        If > 0, detected peaks are buffered by a ``buffer_size`` square when
        suppressing nearby residual peaks.

    Returns
    -------
    ndarray of shape (N, 4)
        ``[amp, mu_y, mu_x, sigma]`` per peak, or ``(0, 4)`` if none.
    """
    height, width = act_im_2d.shape
    empty = np.zeros((0, 4))

    explored = act_im_2d.copy()
    explored[exclusion_mask | np.isnan(explored)] = -np.inf

    rank8 = ndimage.rank_filter(explored, rank=7, size=3)
    rank9 = ndimage.maximum_filter(explored, size=3)
    p_tmp = (rank8 > peak_thresh) & (explored == rank9)

    if not np.any(p_tmp):
        return empty

    py, px = np.where(p_tmp)
    amp = act_im_2d[py, px] * _AMP_SCALE
    n_peaks = len(py)

    act_sel_pix = ndimage.binary_dilation(p_tmp, structure=np.ones((9, 9)))
    act_sel_pix &= ~np.isnan(act_im_2d)

    thetaf = np.column_stack(
        [amp, py.astype(float), px.astype(float), 0.5 * np.ones(n_peaks)]
    )
    p_locs = np.column_stack([py.astype(float), px.astype(float)])

    sel_yx = np.column_stack(np.where(act_sel_pix)).astype(float)
    sel_vals = act_im_2d[act_sel_pix] - mu_bg

    lb_f, ub_f = _make_bounds(p_locs, height, width)
    thetaf = _lsq_curvefit(thetaf, sel_yx, sel_vals, lb_f, ub_f, max_nfev=5000)

    p_im = _peak_mask(thetaf, height, width)
    buffer_mask = _buffer_mask(p_im, buffer_size)

    fit_im = np.zeros((height, width), dtype=float)
    fit_im[act_sel_pix] = gaussian_peaks_integrated(thetaf, sel_yx)
    res_im = (act_im_2d - fit_im - mu_bg) / sigma_bg
    fit_support = fit_im > 1e-3

    thetaf = _refine_residual_peaks(
        thetaf,
        p_locs,
        act_im_2d,
        exclusion_mask,
        mu_bg,
        sigma_bg,
        peak_th,
        buffer_size,
        height,
        width,
        buffer_mask,
        res_im,
        fit_support,
        act_sel_pix,
        fit_im,
    )

    if thetaf.shape[0] > 0:
        s = thetaf[:, 3]
        adj_thresh = peak_thresh / (
            np.pi / 2 * s**2 * erf(1 / (np.sqrt(2) * s)) ** 2
        )
        thetaf = thetaf[thetaf[:, 0] >= adj_thresh]

    return thetaf


def get_act_im_peaks(
    act_im: np.ndarray,
    peak_th: float = 3.0,
    exclusion_mask: np.ndarray = None,
    buffer_size: int = 0,
) -> np.ndarray:
    """Find Gaussian source seeds in a 3-D (Z, H, W) activity image.

    Background statistics (median and MAD-scaled sigma) and the detection
    threshold are computed once across all planes for uniform sensitivity, then
    each plane is detected independently (:func:`detect_peaks_2d`).

    Parameters
    ----------
    act_im : ndarray of shape (Z, H, W)
        Activity image (may contain NaNs).
    peak_th : float
        Threshold in MAD-normalized standard deviations.
    exclusion_mask : None or ndarray
        ``None``, a 2-D ``(H, W)`` mask applied to every plane, or a 3-D
        ``(Z, H, W)`` per-plane mask (e.g. user soma ROIs).
    buffer_size : int
        If > 0, detected peaks are buffered by a ``buffer_size`` square when
        suppressing nearby residual peaks.

    Returns
    -------
    ndarray of shape (N, 3)
        Source seeds ``[z, mu_y, mu_x]``, or ``(0, 3)`` if none detected.
    """
    n_z, height, width = act_im.shape
    empty = np.zeros((0, 3))

    if exclusion_mask is None:
        excl_planes = [np.zeros((height, width), dtype=bool)] * n_z
    elif exclusion_mask.ndim == 2:
        excl_planes = [exclusion_mask.astype(bool)] * n_z
    else:
        excl_planes = [exclusion_mask[z].astype(bool) for z in range(n_z)]

    valid_vals = act_im[~np.isnan(act_im)]
    if valid_vals.size == 0:
        return empty

    mu_bg = float(np.nanmedian(act_im))
    sigma_bg = (
        float(np.median(np.abs(valid_vals - np.median(valid_vals))))
        / _MAD_TO_SIGMA
    )
    if sigma_bg <= 0:
        return empty

    peak_thresh = mu_bg + peak_th * sigma_bg

    source_seeds_list = []
    for z in range(n_z):
        thetaf_z = detect_peaks_2d(
            act_im[z],
            excl_planes[z],
            mu_bg,
            sigma_bg,
            peak_thresh,
            peak_th,
            buffer_size=buffer_size,
        )
        if thetaf_z.shape[0] > 0:
            z_col = np.full((thetaf_z.shape[0], 1), z, dtype=float)
            source_seeds_list.append(
                np.column_stack([z_col, thetaf_z[:, 1], thetaf_z[:, 2]])
            )

    return np.vstack(source_seeds_list) if source_seeds_list else empty
