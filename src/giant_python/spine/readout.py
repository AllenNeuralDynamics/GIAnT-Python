"""Project per-trial images on fixed windows: the spine-size readout.

For window weights ``h`` (a source's mean footprint) restricted to the
pixels valid in every trial and normalized to sum 1, and a trial's image
``x``::

    F = (h . x) / (h . h)

as GIAnT-MATLAB's ``spineTrends.m``. Restricting ``h`` to pixels valid in
all trials keeps the window identical across trials, so a pixel that drops
out of one trial cannot change the readout of the others.

User ROIs (masks) are read out as the mean over the mask's pixels, either
restricted to the same valid pixels as the footprints (identical windows in
every trial and image), or as GIAnT-MATLAB's ``user_rois/per_trial_F0``
does: every mask pixel that has a value in that trial, with the share of
such pixels as the coverage.

:func:`norm_log` is ``spineTrends.m``'s bleaching normalization: the log of
the size, minus each trial's median over the sources taken into account,
minus each source's own mean.
"""

from __future__ import annotations

import numpy as np

from .rois import RoiSet


def common_valid(stacks, trials=None) -> np.ndarray:
    """Pixels with a value in every given trial of every stack.

    Parameters
    ----------
    stacks : sequence of ndarray of shape (trials, rows, cols)
        Image stacks in one frame.
    trials : array_like of int, optional
        Trials to require (default: those with any value in the first
        stack).

    Returns
    -------
    ndarray of bool, shape (rows, cols)
        No pixel is valid without trials (nothing to compare).
    """
    stacks = [np.asarray(s) for s in stacks]
    if trials is None:
        first = stacks[0]
        trials = np.flatnonzero(
            np.isfinite(first.reshape(first.shape[0], -1)).any(axis=1)
        )
    trials = np.asarray(trials, dtype=np.int64)
    if trials.size == 0:
        return np.zeros(stacks[0].shape[1:], bool)
    ok = np.ones(stacks[0].shape[1:], bool)
    for s in stacks:
        ok &= np.isfinite(s[trials]).all(axis=0)
    return ok


def project(images: np.ndarray, rois: RoiSet, valid: np.ndarray):
    """Footprint-weighted readout of each ROI in each trial.

    Parameters
    ----------
    images : ndarray of shape (trials, rows, cols)
        Per-trial images.
    rois : RoiSet
        The windows (in the images' frame).
    valid : ndarray of bool, shape (rows, cols)
        Pixels the windows are restricted to (:func:`common_valid`).

    Returns
    -------
    values : ndarray of float64, shape (trials, rois)
        ``(h . x) / (h . h)`` with ``h`` the ROI's weights on ``valid``,
        normalized to sum 1; NaN for an ROI without valid pixels.
    n_pixels : ndarray of int, shape (rois,)
        Valid pixels per ROI.
    """
    images = np.asarray(images)
    t = images.shape[0]
    flat = images.reshape(t, -1)
    vflat = np.asarray(valid, dtype=bool).ravel()
    out = np.full((t, len(rois)), np.nan)
    n_pix = np.zeros(len(rois), np.int64)
    for k in range(len(rois)):
        idx, w = rois.pixels(k)
        keep = vflat[idx] & (w > 0)
        idx, w = idx[keep], w[keep]
        n_pix[k] = idx.size
        if idx.size == 0 or w.sum() <= 0:
            continue
        h = w / w.sum()
        out[:, k] = flat[:, idx].astype(np.float64) @ h / (h @ h)
    return out, n_pix


def mask_mean(images: np.ndarray, rois: RoiSet, valid=None):
    """Mean of each ROI's pixels with a value, per trial.

    Without ``valid`` this is GIAnT-MATLAB's ``user_rois/per_trial_F0``:
    every pixel with a value in that trial, so the window can change
    between trials and images. With ``valid`` (:func:`common_valid`) the
    ROI's valid pixels are averaged, the same windows in every trial and
    image: a trial in which the image has no value at one of them gives
    NaN, as :func:`project` does.

    Returns
    -------
    values : ndarray of float64, shape (trials, rois)
    coverage : ndarray of float64, shape (trials, rois)
        Share of the ROI's pixels averaged.
    """
    images = np.asarray(images)
    t = images.shape[0]
    flat = images.reshape(t, -1)
    vflat = None if valid is None else np.asarray(valid, bool).ravel()
    values = np.full((t, len(rois)), np.nan)
    coverage = np.zeros((t, len(rois)))
    for k in range(len(rois)):
        idx, _ = rois.pixels(k)
        if idx.size == 0:
            continue
        n_all = idx.size
        if vflat is not None:
            idx = idx[vflat[idx]]
            if idx.size == 0:
                continue
        x = flat[:, idx].astype(np.float64)
        ok = np.isfinite(x)
        n = ok.sum(axis=1)
        coverage[:, k] = n / n_all
        # with fixed windows, only complete ones
        need = idx.size if vflat is not None else 1
        with np.errstate(invalid="ignore", divide="ignore"):
            values[:, k] = np.where(
                n >= need,
                np.where(ok, x, 0).sum(axis=1) / np.maximum(n, 1),
                np.nan,
            )
    return values, coverage


def norm_log(values: np.ndarray, ok: np.ndarray | None = None) -> np.ndarray:
    """``spineTrends.m``'s bleaching normalization.

    ``log F``, minus each trial's median over the ROIs in ``ok``, minus
    each ROI's mean over trials.

    Parameters
    ----------
    values : ndarray of shape (trials, rois)
        Readouts ``F`` (positive).
    ok : ndarray of bool, shape (rois,), optional
        ROIs that enter the per-trial median (default: those with a finite,
        positive value in every trial).

    Returns
    -------
    ndarray of float64, shape (trials, rois)
    """
    values = np.asarray(values, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        lg = np.log(values)
    if ok is None:
        ok = np.isfinite(lg).all(axis=0)
    ok = np.asarray(ok, dtype=bool)
    if not ok.any():
        return np.full_like(lg, np.nan)
    trend = np.median(lg[:, ok], axis=1, keepdims=True)
    centred = lg - trend
    return centred - np.mean(centred, axis=0, keepdims=True)
