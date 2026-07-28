"""Baseline (F0) estimation and dF/F assembly.

Ported from ``extractSLAP2IntegrationSources.py`` (``_movmean_nan`` /
``computeF0`` ref 217-305, and the dF/F assembly ref 2972-2978).
:func:`compute_f0` estimates a slowly-varying fluorescence baseline via a
rolling median plus a decimated-grid rolling-min ("convex-hull"-like)
envelope, interpolated back to the full time grid with PCHIP.
:func:`assemble_dff` combines the per-trial
least-squares traces into ``F``, re-estimates ``F0``, and forms ``dF``/``dFF``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


def _movmean_nan(x: np.ndarray, window: int) -> np.ndarray:
    """Centered, NaN-ignoring moving mean (like MATLAB ``smoothdata movmean``).

    Uses partial windows at the edges and returns NaN where a window contains
    no valid samples.

    Parameters
    ----------
    x : ndarray
        1-D input (NaNs allowed).
    window : int
        Window length in samples (clamped to >= 1).

    Returns
    -------
    ndarray of float
        The smoothed signal.
    """
    window = max(int(window), 1)
    if window == 1:
        return x.astype(float, copy=True)

    kernel = np.ones(window, dtype=float)
    valid = ~np.isnan(x)
    x_filled = np.where(valid, x, 0.0)

    num = np.convolve(x_filled, kernel, mode="same")
    den = np.convolve(valid.astype(float), kernel, mode="same")

    return num / np.where(den == 0.0, np.nan, den)


def _compute_f0_column(
    f0_col: np.ndarray,
    sample_times: np.ndarray,
    n_samps_in_hull: int,
    total_frames: int,
) -> np.ndarray:
    """Estimate the baseline for one already-median-filtered trace column.

    Builds a set of decimated-grid interpolations of the trace, takes their
    per-sample minimum (the rolling-min envelope), discards samples with too
    few contributions, smooths/fills the gaps, and PCHIP-interpolates the
    envelope back onto the full ``total_frames`` grid.
    """
    f00 = np.full((sample_times.shape[0], n_samps_in_hull), np.nan)
    for dix in range(n_samps_in_hull, 0, -1):
        start = dix - 1
        xi = sample_times[start::n_samps_in_hull]
        f00[:, dix - 1] = np.interp(
            sample_times, xi, f0_col[xi], left=np.nan, right=np.nan
        )
    ff = np.nanmin(f00, axis=1)

    doubt = np.sum(~np.isnan(f00), axis=1) < int(np.ceil(n_samps_in_hull / 2))
    if np.sum(~doubt) > 2:
        ff[doubt] = np.nan

    win = 2 * int(np.ceil(n_samps_in_hull / 2.0)) + 1
    fill = _movmean_nan(ff, win)
    nan_mask = np.isnan(ff)
    ff[nan_mask] = fill[nan_mask]
    ff = _movmean_nan(ff, win)

    nan_mask = np.isnan(ff)
    if np.any(nan_mask):
        ff = np.interp(sample_times, sample_times[~nan_mask], ff[~nan_mask])

    pchip = PchipInterpolator(sample_times, ff, extrapolate=True)
    return pchip(np.arange(total_frames))


def compute_f0(
    f_in: np.ndarray, denoise_window: int, hull_window: int
) -> np.ndarray:
    """Estimate a slowly-varying fluorescence baseline ``F0``.

    Port of ``computeF0``: (1) a centered rolling-median denoise, (2) a
    rolling convex-hull-like min envelope on decimated grids, (3) discard of
    doubtful samples near NaNs, smoothing/filling, and PCHIP back to the full
    time grid.

    Parameters
    ----------
    f_in : ndarray of shape (T, ...)
        Fluorescence with time along axis 0 (NaNs allowed).
    denoise_window : int
        Rolling-median window (time samples).
    hull_window : int
        Window controlling the convex-hull-like envelope.

    Returns
    -------
    ndarray
        The baseline ``F0``, same shape as ``f_in``.
    """
    f = np.asarray(f_in)
    orig_shape = f.shape
    if f.ndim == 1:
        f = f[:, None]
    else:
        f = f.reshape(f.shape[0], -1)

    total_frames = f.shape[0]
    if total_frames < 4:
        return np.ones_like(f_in, dtype=float) * np.nanmean(
            f_in, axis=0, keepdims=True
        )

    hull_window = int(min(hull_window, total_frames // 4))
    delta_des = max(4.0, denoise_window / 6.0)

    sample_times = np.rint(
        np.linspace(
            0, total_frames - 1, num=int(np.ceil(total_frames / delta_des) + 1)
        )
    ).astype(int)
    n_samps_in_hull = int(np.ceil(hull_window / delta_des))

    f0 = (
        pd.DataFrame(f)
        .rolling(window=denoise_window, center=True, min_periods=1)
        .median()
        .to_numpy()
    )

    f0 = f0.reshape(f0.shape[0], -1)
    for cix in range(f0.shape[1]):
        if np.all(np.isnan(f0[:, cix])):
            continue
        f0[:, cix] = _compute_f0_column(
            f0[:, cix], sample_times, n_samps_in_hull, total_frames
        )

    return f0.reshape(orig_shape)


def assemble_dff(
    d_f_ls: np.ndarray,
    f0_ls: np.ndarray,
    denoise_window: int,
    hull_window: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble ``F``, re-estimate ``F0``, and form ``dF``/``dFF``.

    Reconstructs total fluorescence ``F = dF_ls + F0_ls`` from the per-motion
    least-squares source and background traces, re-estimates the baseline with
    :func:`compute_f0`, and computes ``dF = F - F0`` and ``dFF = dF /
    clip(F0, 1e-4, inf)``.

    Parameters
    ----------
    d_f_ls : ndarray of shape (T, n_sources)
        Concatenated per-source least-squares dF traces (``phi``).
    f0_ls : ndarray of shape (T, n_sources)
        Concatenated per-source least-squares background traces.
    denoise_window, hull_window : int
        Windows passed to :func:`compute_f0`.

    Returns
    -------
    f, f0, d_f, d_ff : ndarray
        Total fluorescence, baseline, baseline-subtracted, and dF/F.
    """
    f = d_f_ls + f0_ls
    f0 = compute_f0(f, denoise_window, hull_window)
    d_f = f - f0
    d_ff = d_f / np.clip(f0, 1e-4, None)
    return f, f0, d_f, d_ff
