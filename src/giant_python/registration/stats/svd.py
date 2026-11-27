"""Shared modes of a trial: SVD of the binned movie in photon-noise units.

The movie is binned in 8-frame time bins (a pixel's bin is the mean of its
valid frames, missing with fewer than 6) and 2 x 2 pixel bins (the mean of
the four pixels' bins, missing unless all four are present), starting at
frame 0 and pixel (0, 0); a trailing partial time bin and odd row or column
are dropped. Spatial bins without a missing time bin enter the SVD. Each is
centred on its mean over time and divided by its photon-noise SD,

    sigma_bin^2 = kappa * sum_p sigma8(p)^2 / 16,

where ``sigma8(p)`` is pixel ``p``'s photon SD of an 8-frame mean and
``kappa`` the trial's mean factor by which bilinear interpolation's shared
raw pixels raise the variance of a 2 x 2 sum (see
``geometry.spatial_factor``), so that photon noise has unit variance.

The top :data:`N_COMPONENTS` components of ``Y = U S V^T`` are kept, each
signed so that its spatial vector sums to a non-negative value (a mode that
brightens the trial on balance has a time course that rises). Every pixel's
weight on component ``j`` is the coefficient of ``u_j`` in a least-squares
fit of its own 8-frame bin means (movie units, its valid bins, with an
intercept) on the time courses: ``weight`` in movie units per unit of the
unit-norm time course, ``weight_ph`` divided by ``sigma8``.
"""

from __future__ import annotations

import numpy as np

N_COMPONENTS = 3
"""Components kept."""
MIN_BINS = 64
"""Valid time bins a pixel needs for its weights."""

RANK_TOL = 1e-6
"""Singular values below this share of the largest count as 0 (Gram
eigenvalues carry rounding of about 1e-8 of it in singular values)."""

MAX_CONDITION = 1e8
"""Largest condition number of a pixel's regression (normal equations)."""


def _stack_bins(bins: list, n_idx: int) -> np.ndarray:
    """``(channels, n_bins, n_idx)`` float64, NaN-padded per block."""
    parts = []
    for b in bins:
        b = b.numpy().astype(np.float64)
        pad = np.full(b.shape[:2] + (n_idx - b.shape[2],), np.nan)
        parts.append(np.concatenate([b, pad], 2))
    if not parts:
        return np.zeros((0, 0, n_idx))
    return np.concatenate(parts, 1)


def bin_members(idx, shape) -> np.ndarray:
    """``(n_space, 4)`` compact positions of the four pixels of every
    2 x 2 bin whose pixels are all in the index (bins in raster order).

    Parameters
    ----------
    idx : ndarray of int
        Flat registered-grid index of the compact pixels.
    shape : (int, int)
        Registered grid.
    """
    h, w = shape
    h2, w2 = h // 2, w // 2
    pos = np.full(h * w, -1, dtype=np.int64)
    pos[idx] = np.arange(idx.size)
    grid = pos.reshape(h, w)[: 2 * h2, : 2 * w2]
    four = grid.reshape(h2, 2, w2, 2).transpose(0, 2, 1, 3).reshape(-1, 4)
    return four[(four >= 0).all(1)]


def _binned(full: np.ndarray, members: np.ndarray) -> np.ndarray:
    """2 x 2 bin means ``(n_bins, n_space)`` of compact time bins
    ``(n_bins, n_idx)`` (NaN unless all four pixels are present)."""
    return full[:, members].mean(-1)


def _bin_sigma(bin_var: np.ndarray, members: np.ndarray):
    """Photon-noise SD of each 2 x 2 bin's mean ``(n_space,)`` from its
    pixels' shares (:func:`.maps.bin_var8`)."""
    return np.sqrt(bin_var[members].sum(-1) / 16)


def _components(y: np.ndarray, k: int):
    """Top ``k`` signed components of ``y`` ``(n_bins, n_space)``.

    From the eigenvectors of the smaller Gram matrix: ``y y^T`` (time bins
    x time bins; its eigenvectors are the left singular vectors) or, for a
    trial with more time bins than spatial bins, ``y^T y`` (then ``u = y v
    / s``). Memory and time go with the smaller side squared and cubed.

    Returns
    -------
    u : ndarray ``(n_bins, n)``
    s : ndarray ``(n,)``
        The ``n <= k`` components whose singular value is above
        :data:`RANK_TOL` of the largest.
    total : float
        ``sum(y^2)`` (the sum of all squared singular values).
    """
    tall = y.shape[0] > y.shape[1]
    gram = y.T @ y if tall else y @ y.T
    evals, evecs = np.linalg.eigh(gram)
    order = np.argsort(evals)[::-1][: min(k, evals.size)]
    s = np.sqrt(np.maximum(evals[order], 0.0))
    # a rank below k: no further components (relative to rounding)
    keep = s > (s[0] if s.size else 0.0) * RANK_TOL
    s, u = s[keep], evecs[:, order[keep]]
    if tall:
        u = (y @ u) / s
    sign = np.sign((y.T @ u).sum(0))
    sign[sign == 0] = 1
    return u * sign, s, float(np.trace(gram))


def _descriptors(u: np.ndarray):
    """Skewness and lag-1 autocorrelation of each time course."""
    c = u - u.mean(0)
    m2 = (c**2).mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        skew = (c**3).mean(0) / m2**1.5
        lag1 = (c[1:] * c[:-1]).sum(0) / (c**2).sum(0)
    return skew, lag1


def regression_weights(y: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Least-squares coefficients of ``u`` for every pixel.

    Parameters
    ----------
    y : ndarray ``(n_bins, n_px)``
        Pixels' bin means, NaN where missing.
    u : ndarray ``(n_bins, k)``
        Time courses.

    Returns
    -------
    ndarray ``(n_px, k)``
        Coefficients (with an intercept in the fit); NaN for pixels with
        fewer than :data:`MIN_BINS` bins, or whose bins do not determine
        the fit (a time course that is flat over them: condition number of
        the normal equations above :data:`MAX_CONDITION`).
    """
    nb, k = u.shape
    out = np.full((y.shape[1], k), np.nan)
    if k == 0 or not np.isfinite(u).all():
        return out
    d = np.concatenate([np.ones((nb, 1)), u], 1)
    m = np.isfinite(y).astype(np.float64)
    prod = (d[:, :, None] * d[:, None, :]).reshape(nb, -1)
    gram = (m.T @ prod).reshape(-1, k + 1, k + 1)
    rhs = np.where(m > 0, y, 0.0).T @ d
    ok = m.sum(0) >= MIN_BINS
    if ok.any():
        sv = np.linalg.svd(gram[ok], compute_uv=False)
        ok[ok] = sv[:, -1] > sv[:, 0] / MAX_CONDITION
    if ok.any():
        sol = np.linalg.solve(gram[ok], rhs[ok][:, :, None])[:, :, 0]
        out[ok] = sol[:, 1:]
    return out


def _empty(c: int, nb: int, n_idx: int, k: int) -> dict:
    """Outputs of a trial without enough data (NaN)."""
    nan = np.full((c, k), np.nan)
    return {
        "time_courses": np.full((c, k, nb), np.nan),
        "singular_values": nan.copy(),
        "var_share": nan.copy(),
        "skew": nan.copy(),
        "lag1": nan.copy(),
        "total_var": np.full(c, np.nan),
        "n_space": np.zeros(c, dtype=np.int64),
        "weight": np.full((c, n_idx, k), np.nan),
        "weight_ph": np.full((c, n_idx, k), np.nan),
    }


def _channel(out, ch, full, idx, shape, sigma8, bin_var, k) -> None:
    """Fill channel ``ch`` of ``out``."""
    members = bin_members(idx, shape)
    b = _binned(full, members)
    sd = _bin_sigma(bin_var, members)
    use = np.isfinite(b).all(0) & np.isfinite(sd) & (sd > 0)
    if use.sum() < k or b.shape[0] <= k:
        return
    y = (b[:, use] - b[:, use].mean(0)) / sd[use]
    u, s, total = _components(y, k)
    n = s.size
    skew, lag1 = _descriptors(u)
    out["time_courses"][ch, :n] = u.T
    out["singular_values"][ch, :n] = s
    out["var_share"][ch, :n] = s**2 / total
    out["skew"][ch, :n] = skew
    out["lag1"][ch, :n] = lag1
    out["total_var"][ch] = total
    out["n_space"][ch] = int(use.sum())
    wts = regression_weights(full, u)
    out["weight"][ch, :, :n] = wts
    with np.errstate(invalid="ignore", divide="ignore"):
        out["weight_ph"][ch, :, :n] = wts / sigma8[:, None]


def shared_modes(bins, idx, shape, sigma8, bin_var, k=N_COMPONENTS) -> dict:
    """The trial's top shared modes per channel.

    Parameters
    ----------
    bins : list of torch.Tensor
        Per block ``(channels, n_bins, n_idx_then)`` 8-frame bin means.
    idx : ndarray of int
        Flat registered-grid index of the compact pixels.
    shape : (int, int)
        Registered grid.
    sigma8 : ndarray ``(channels, n_idx)``
        Photon SD of each pixel's 8-frame mean (for ``weight_ph``).
    bin_var : ndarray ``(channels, n_idx)``
        Each pixel's share of the photon variance of a 2 x 2 sum of
        8-frame means (:func:`.maps.bin_var8`).
    k : int
        Components.

    Returns
    -------
    dict
        ``time_courses`` ``(channels, k, n_bins)``, ``singular_values``,
        ``var_share``, ``skew``, ``lag1`` ``(channels, k)``, ``total_var``,
        ``n_space`` ``(channels,)``, ``weight`` and ``weight_ph``
        ``(channels, n_idx, k)``.
    """
    full = _stack_bins(bins, idx.size)
    c = sigma8.shape[0]
    out = _empty(c, full.shape[1], idx.size, k)
    out["n_bins"] = np.int64(full.shape[1])
    if full.shape[0] == 0:
        return out
    for ch in range(c):
        _channel(out, ch, full[ch], idx, shape, sigma8[ch], bin_var[ch], k)
    return out
