"""Synthetic trials for the across-trial alignment tests (a port of the
scenario of GIAnT-MATLAB's ``TestTrialAlignment.m``, with numpy's random
numbers): a labeled neuron (dendrites with spines, an unscanned band and a
hole) seen through known affine maps and, optionally, smooth bends, with
shot-like noise in two halves per trial."""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndimage

from giant_python.extraction.raster.alignment import interp2_cubic

SIZE = (180, 260)
N_TRIALS = 6


def neuron(rng: np.random.Generator, size=SIZE):
    """Mean image of a labeled neuron and its scanned pixels (NaN outside
    them)."""
    h, w = size
    rr, cc = np.mgrid[0:h, 0:w].astype(np.float64)
    truth = np.full(size, 5.0)
    for _ in range(6):
        t = np.linspace(0, 1, 400)[:, None]
        p0 = rng.random(2) * np.array(size)
        ang = rng.random() * np.pi
        length = 150 + 100 * rng.random()
        along = np.array([np.sin(ang), np.cos(ang)])
        across = np.array([np.cos(ang), -np.sin(ang)])
        wiggle = 8 * np.sin(
            2 * np.pi * (t * (1 + rng.random()) + rng.random())
        )
        path = p0 + t * length * along + wiggle * across
        for q in range(0, path.shape[0], 20):
            truth += 30 * np.exp(
                -((rr - path[q, 0]) ** 2 + (cc - path[q, 1]) ** 2)
                / (2 * 1.5**2)
            )
            sp = path[q] + (3 + 2 * rng.random()) * across * np.sign(
                rng.standard_normal()
            )
            truth += (40 + 80 * rng.random()) * np.exp(
                -((rr - sp[0]) ** 2 + (cc - sp[1]) ** 2) / (2 * 1.1**2)
            )
    scan = np.ones(size, bool)
    scan[:, :12] = False
    scan[79:100, 119:150] = False
    truth[~scan] = np.nan
    return truth, scan


def trials(
    truth,
    scan,
    bends: bool,
    rng: np.random.Generator,
    n_trials=N_TRIALS,
    bend_px: float = 0.25,
    affine_sd: float = 0.003,
):
    """Trial mean images ``(trials, rows, cols)``, their halves ``(trials, 2,
    rows, cols)``, the true fields ``(trials, 2, rows, cols)`` (consensus
    pixel ``x`` is at ``x + U(x)`` in the trial) and the rigid starting maps
    ``(trials, 2, 3)``."""
    h, w = truth.shape
    rr, cc = np.mgrid[0:h, 0:w].astype(np.float64)
    filled = np.where(np.isnan(truth), 5.0, truth)
    m = np.full((n_trials, h, w), np.nan)
    mh = np.full((n_trials, 2, h, w), np.nan)
    u = np.zeros((n_trials, 2, h, w))
    a0 = np.zeros((n_trials, 2, 3))
    for t in range(n_trials):
        lin = np.eye(2) + affine_sd * rng.standard_normal((2, 2))
        tr = 0.6 * rng.standard_normal(2)
        u_r = (lin[0, 0] - 1) * rr + lin[0, 1] * cc + tr[0]
        u_c = lin[1, 0] * rr + (lin[1, 1] - 1) * cc + tr[1]
        if bends:
            for comp in (0, 1):
                b = ndimage.gaussian_filter(
                    rng.standard_normal((h, w)), 40, mode="nearest", truncate=5
                )
                b = bend_px * b / b.std()
                if comp == 0:
                    u_r = u_r + b
                else:
                    u_c = u_c + b
        u[t, 0], u[t, 1] = u_r, u_c
        # the consensus point each trial pixel shows: y = x - U(y)
        y_r, y_c = rr.ravel().copy(), cc.ravel().copy()
        for _ in range(8):
            coords = np.vstack([y_r, y_c])
            ur = ndimage.map_coordinates(u_r, coords, order=1, mode="nearest")
            uc = ndimage.map_coordinates(u_c, coords, order=1, mode="nearest")
            y_r, y_c = rr.ravel() - ur, cc.ravel() - uc
        img = interp2_cubic(filled, y_r, y_c)
        img = np.where(np.isnan(img), 5.0, img).reshape(h, w)
        for half in range(2):
            x = img + 0.2 * np.sqrt(np.maximum(img, 0)) * rng.standard_normal(
                (h, w)
            )
            x[~scan] = np.nan  # the scan mask is fixed in the trial's frame
            mh[t, half] = x
        m[t] = mh[t].mean(axis=0)
        c0 = (h // 2 - 1, w // 2 - 1)  # MATLAB's round(sz/2), zero-based
        a0[t] = [[1, 0, np.round(u_r[c0])], [0, 1, np.round(u_c[c0])]]
    return m, mh, u, a0


def field_error(coords, u, a0, truth, scan):
    """RMS error of fitted maps relative to the trials' mean map, weighted by
    image structure; and the same for the rigid start. ``coords(t, r, c)``
    gives the fitted positions."""
    n_trials = u.shape[0]
    sel = ndimage.binary_erosion(scan, np.ones((5, 5)), border_value=0)
    rows, cols = np.nonzero(sel)
    est = np.empty((n_trials, rows.size, 2))
    tru = np.empty_like(est)
    rig = np.empty_like(est)
    for t in range(n_trials):
        r_t, c_t = coords(t, rows.astype(float), cols.astype(float))
        est[t] = np.column_stack([r_t - rows, c_t - cols])
        tru[t] = np.column_stack([u[t, 0][sel], u[t, 1][sel]])
        rig[t] = a0[t, :, 2]
    sq = np.sqrt(np.fmax(truth, 0))
    g_r, g_c = np.gradient(sq, axis=0), np.gradient(sq, axis=1)
    wgt = g_r[sel] ** 2 + g_c[sel] ** 2
    wgt[np.isnan(wgt)] = 0

    def rel(x):
        """Relative to the trials' mean."""
        return x - x.mean(axis=0)

    def wrms(e):
        """Structure-weighted RMS over trials and pixels."""
        return np.sqrt(
            np.sum((e**2).sum(axis=2).mean(axis=0) * wgt) / wgt.sum()
        )

    return wrms(rel(est) - rel(tru)), wrms(rel(rig) - rel(tru))
