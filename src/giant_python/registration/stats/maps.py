"""The statistic maps of a trial, from the accumulated per-pixel sums.

Every map is per channel, ``(channels, rows, cols)`` float32 on the
registered grid, NaN where a pixel has too few valid frames. For each
second moment ``m`` the file holds the moment; ``m_q`` and ``m_qa``, its
photon-noise expectation per unit of ``g2 mu`` from the samples
themselves and per unit of ``2 c2 mu`` from pairs of consecutive samples
(properties of frame formation and registration only); ``m_ph = mu (g2
m_q + 2 c2 m_qa)``; and a normalised map that is 0 under photon noise. The
fast third moment likewise has ``fast_q3``, ``fast_q3a`` and ``fast_q3b``
(per unit of ``g3 mu``, ``3 c3a mu``, ``3 c3b mu``). See
``docs/source/reg_stats.rst`` for the definitions.

For two frames that share samples, the pair terms are those of the frames'
own sums scaled by the pixel's ratio ``r = sum_t QA2(t) / sum_t A2(t, t)``
(the pairs lie on the same lines as the pixel's samples, so their kernel
overlap between frames is that of the samples).
"""

from __future__ import annotations

import numpy as np
import torch

from .svd import shared_modes

ZERO_NU = np.geomspace(1e-4, 2.0, 33)
"""Non-zero rates per sample at which the photon-noise share of zeros is
accumulated (interpolated between them at each pixel's rate)."""
MIN_FRAMES = 64
"""Valid frames (or fast-part centres) a pixel needs for its maps."""
MIN_BLOCKS = 8
"""Valid 64-frame blocks a pixel needs for its slow map."""
SUMS = (
    "n", "sx", "sx2", "zeros", "s0", "s1", "sq",
    "nb", "sb", "sb2", "vb", "vbl", "vbq", "cb",
    "v8", "v8l", "v8q", "n8",
    "nf", "sz", "sz2", "sz3", "q2", "q2l", "qa", "q3", "q3a", "q3b",
)  # fmt: skip
"""Per-pixel sums of the accumulator (zeros if a trial had none)."""


def _np(t: torch.Tensor) -> np.ndarray:
    """A tensor as a float64 NumPy array."""
    return t.detach().double().cpu().numpy()


def _ratio(num, den, ok):
    """``num / den`` where ``ok`` and ``den > 0``, else NaN."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ok & (den > 0), num / den, np.nan)


def _pair_ratio(a: dict) -> np.ndarray:
    """``r``: the pixel's pair sums relative to its own sums."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(a["s0"] > 0, a["sq"] / a["s0"], 0.0)


def single_frame_maps(a: dict, mu, g: dict) -> dict:
    """Mean-related maps: valid frames, variance, share of zeros.

    ``var = mean(x^2) - mean(x)^2`` over a pixel's valid frames, and its
    photon-noise expectation ``var_q = S0 / n - (S0 + 2 S1) / n^2`` with
    ``S0 = sum_t A2(t, t)`` and ``S1 = sum_t sum_l A2(t, t + l)`` over pairs
    of valid frames (the second term is the variance of the mean);
    ``var_qa`` likewise from ``SQ = sum_t QA2(t)`` and ``r S1``.
    """
    n = a["n"]
    ok = n >= MIN_FRAMES
    nn = np.maximum(n, 1)
    r = _pair_ratio(a)
    mean = a["sx"] / nn
    var = np.where(ok, a["sx2"] / nn - mean**2, np.nan)
    var_q = np.where(
        ok, a["s0"] / nn - (a["s0"] + 2 * a["s1"]) / nn**2, np.nan
    )
    var_qa = np.where(
        ok, a["sq"] / nn - (a["sq"] + 2 * r * a["s1"]) / nn**2, np.nan
    )
    var_ph = mu * (g["g2"] * var_q + 2 * g["c2"] * var_qa)
    return {
        "var": var,
        "var_q": var_q,
        "var_qa": var_qa,
        "var_ph": var_ph,
        "var_rel": _ratio(var, var_ph, ok) - 1,
        "zero_share": np.where(n > 0, a["zeros"] / nn, np.nan),
    }


def slow_maps(a: dict, mu, g: dict) -> dict:
    """Variance of 64-frame block means and its photon-noise expectation.

    ``slow_var = sum_j (B_j - mean B)^2 / (J - 1)`` over the pixel's ``J``
    valid blocks; ``slow_q = (sum_j V_j - (sum_j V_j + 2 sum_j C_j) / J) /
    (J - 1)`` with ``V_j`` the photon variance of block mean ``j`` and
    ``C_j`` its covariance with the previous block's mean (kernel overlap);
    ``slow_qa`` likewise with the pair terms.
    """
    nb = a["nb"]
    ok = nb >= MIN_BLOCKS
    jj = np.maximum(nb, 2)
    r = _pair_ratio(a)
    var = np.where(ok, (a["sb2"] - a["sb"] ** 2 / jj) / (jj - 1), np.nan)

    def expect(vb, cb):
        """Expectation of the block-mean variance from its sums."""
        return np.where(ok, (vb - (vb + 2 * cb) / jj) / (jj - 1), np.nan)

    q = expect(a["vb"], a["cb"])
    qa = expect(a["vbq"] + r * a["vbl"], r * a["cb"])
    ph = mu * (g["g2"] * q + 2 * g["c2"] * qa)
    return {
        "n_blocks": nb,
        "slow_var": var,
        "slow_q": q,
        "slow_qa": qa,
        "slow_var_ph": ph,
        "slow_rel": _ratio(var, ph, ok) - 1,
    }


def fast_maps(a: dict, mu, g: dict) -> dict:
    """Second and third central moments of the fast part and their
    photon-noise expectations (means over the pixel's centres)."""
    nf = a["nf"]
    ok = nf >= MIN_FRAMES
    nn = np.maximum(nf, 1)
    r = _pair_ratio(a)
    m1 = a["sz"] / nn
    m2 = a["sz2"] / nn
    var = np.where(ok, m2 - m1**2, np.nan)
    m3 = np.where(ok, a["sz3"] / nn - 3 * m1 * m2 + 2 * m1**3, np.nan)
    q = {
        k: np.where(ok, a[k] / nn, np.nan) for k in ("q2", "q3", "q3a", "q3b")
    }
    qa = np.where(ok, (a["qa"] + r * a["q2l"]) / nn, np.nan)
    ph2 = mu * (g["g2"] * q["q2"] + 2 * g["c2"] * qa)
    ph3 = mu * (
        g["g3"] * q["q3"] + 3 * (g["c3a"] * q["q3a"] + g["c3b"] * q["q3b"])
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        skew = np.where(ok & (ph2 > 0), (m3 - ph3) / ph2**1.5, np.nan)
    return {
        "n_fast": nf,
        "fast_var": var,
        "fast_q": q["q2"],
        "fast_qa": qa,
        "fast_var_ph": ph2,
        "fast_rel": _ratio(var, ph2, ok) - 1,
        "fast_m3": m3,
        "fast_q3": q["q3"],
        "fast_q3a": q["q3a"],
        "fast_q3b": q["q3b"],
        "fast_m3_ph": ph3,
        "fast_skew_rel": skew,
    }


def zero_maps(grid: np.ndarray, n: np.ndarray, nu: np.ndarray, share):
    """Photon-noise share of exact zeros.

    A frame is exactly 0 when all of its contributing samples are
    (samples are non-negative). With ``N`` such samples, ``M`` of them in
    pairs of consecutive samples, a rate ``nu`` of non-zero events per
    sample and a share ``phi`` of a pair's events in both samples, that has
    probability ``exp(-nu (N - phi M))``; ``grid`` holds the sum of it over
    the pixel's valid frames at the rates :data:`ZERO_NU`, and the
    expectation interpolates ``log(grid / n)`` linearly in ``nu`` (beyond
    the grid: the nearest interval's slope).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.log(np.maximum(grid, 1e-300) / np.maximum(n, 1)[..., None])
    j = np.clip(np.searchsorted(ZERO_NU, nu) - 1, 0, ZERO_NU.size - 2)
    j = np.where(np.isfinite(nu), j, 0)
    x0, x1 = ZERO_NU[j], ZERO_NU[j + 1]
    y0 = np.take_along_axis(lg, j[..., None], -1)[..., 0]
    y1 = np.take_along_axis(lg, (j + 1)[..., None], -1)[..., 0]
    zph = np.exp(y0 + (y1 - y0) * (nu - x0) / (x1 - x0))
    zph = np.where(nu <= ZERO_NU[0], np.exp(lg[..., 0] * nu / ZERO_NU[0]), zph)
    zph = np.where((n > 0) & np.isfinite(nu), np.minimum(zph, 1.0), np.nan)
    return {"zero_ph": zph, "zero_excess": share - zph}


def sigma8_map(a: dict, mu, g: dict) -> np.ndarray:
    """Photon SD of each pixel's 8-frame mean (mean over its valid bins)."""
    r = _pair_ratio(a)
    n8 = np.maximum(a["n8"], 1)
    v = g["g2"] * a["v8"] + 2 * g["c2"] * (a["v8q"] + r * a["v8l"])
    with np.errstate(invalid="ignore"):
        return np.where(a["n8"] > 0, np.sqrt(mu * v / n8), np.nan)


def bin_var8(a: dict, mu, g: dict, kappa: float, lam: float) -> np.ndarray:
    """Each pixel's share of the photon variance of a 2 x 2 sum of 8-frame
    means: ``mu v8 (g2 kappa + 2 c2 lam) / n8``, with ``kappa`` the
    trial's :func:`.geometry.spatial_factor` and ``lam`` its
    :func:`.geometry.pair_bin_factor` (all pairs inside the sum, each
    pixel's own among them)."""
    n8 = np.maximum(a["n8"], 1)
    v = a["v8"] * (g["g2"] * kappa + 2 * g["c2"] * lam)
    with np.errstate(invalid="ignore"):
        return np.where(a["n8"] > 0, mu * v / n8, np.nan)


def finish_maps(o) -> dict:
    """All statistics of a finished accumulator.

    Returns
    -------
    dict
        ``maps`` (name -> ``(channels, rows, cols)`` float32), ``svd``,
        ``gain`` (per channel curves) and ``attrs`` (scalars).
    """
    h, w = o.shape
    mean32 = o.sum32 / torch.clamp(o.count32, min=1)
    mean32 = mean32.masked_fill(o.count32 < 1, float("nan"))
    idx = o.index.cpu().numpy()
    a = {k: _np(o.acc[k]) for k in SUMS}
    mu = _np(mean32.reshape(o.c, -1))[:, idx]
    at = [cv.at(mu[ch]) for ch, cv in enumerate(o.gains)]
    g = {k: np.stack([x[k] for x in at]) for k in at[0]}
    compact = {"n_valid": a["n"]}
    compact.update(
        {f"gain_{k}": g[k] for k in ("g2", "g3", "c2", "c3a", "c3b")}
    )
    compact.update(single_frame_maps(a, mu, g))
    compact.update(slow_maps(a, mu, g))
    compact.update(fast_maps(a, mu, g))
    n_idx = idx.size
    grid = (
        _np(o.hist[:, :n_idx])
        if o.hist is not None
        else np.ones(mu.shape + (ZERO_NU.size,))
    )
    compact.update(zero_maps(grid, a["n"], g["nu"], compact["zero_share"]))
    compact["sigma8_ph"] = sigma8_map(a, mu, g)
    kappa = o.kappa[0] / o.kappa[1] if o.kappa[1] else float("nan")
    lam = o.pair_bin / o.kappa[1] if o.kappa[1] else float("nan")
    var8 = bin_var8(a, mu, g, kappa, lam)
    svd = shared_modes(o.bins, idx, (h, w), compact["sigma8_ph"], var8)

    def dense(v):
        """Compact ``(channels, n_idx)`` values on the registered grid."""
        out = np.full((o.c, h * w), np.nan, dtype=np.float32)
        out[:, idx] = v
        return out.reshape(o.c, h, w)

    maps = {"mean": mean32.cpu().numpy().astype(np.float32)}
    maps.update({k: dense(v) for k, v in compact.items()})
    for key in ("weight", "weight_ph"):
        svd[key] = np.stack(
            [dense(svd[key][:, :, j]) for j in range(svd[key].shape[2])], 1
        )
    attrs = {
        "n_frames": o.n_frames,
        "n_lags": o.n_lags,
        "spatial_factor": kappa,
        "pair_bin_factor": lam,
        "n_index": int(idx.size),
        "zero_shared": [float(v) for v in o.shared_zero],
        "validity_differs": [int(v) for v in o.validity_differs],
    }
    curves = [cv.describe() for cv in o.gains]
    gain = {k: _stack_curves([c[k] for c in curves]) for k in curves[0]}
    return {"maps": maps, "svd": svd, "gain": gain, "attrs": attrs}


def _stack_curves(curves: list) -> np.ndarray:
    """Per-channel curves of possibly different lengths, NaN-padded."""
    k = max((c.size for c in curves), default=0)
    out = np.full((len(curves), k), np.nan)
    for i, c in enumerate(curves):
        out[i, : c.size] = c
    return out
