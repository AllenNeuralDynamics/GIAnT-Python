"""The statistic maps of a trial, from the accumulated per-pixel sums.

Every map is per channel, ``(channels, rows, cols)`` float32 on the
registered grid, NaN where a pixel has too few valid frames. For each
moment ``m`` the file holds the moment, ``m_q`` (its photon-noise
expectation per unit of gain times mean: a property of frame formation and
registration only), ``m_ph = gain * mean * m_q`` and a normalised map that
is 0 under photon noise. See ``docs/source/reg_stats.rst`` for the
definitions.
"""

from __future__ import annotations

import numpy as np
import torch

from .svd import shared_modes

MIN_FRAMES = 64
"""Valid frames (or fast-part centres) a pixel needs for its maps."""
MIN_BLOCKS = 8
"""Valid 64-frame blocks a pixel needs for its slow map."""
SUMS = (
    "n", "sx", "sx2", "zeros", "s0", "s1",
    "nb", "sb", "sb2", "vb", "cb",
    "v8", "n8",
    "nf", "sz", "sz2", "sz3", "q2", "q3",
)  # fmt: skip
"""Per-pixel sums of the accumulator (zeros if a trial had none)."""


def _np(t: torch.Tensor) -> np.ndarray:
    """A tensor as a float64 NumPy array."""
    return t.detach().double().cpu().numpy()


def _ratio(num, den, ok):
    """``num / den`` where ``ok`` and ``den > 0``, else NaN."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ok & (den > 0), num / den, np.nan)


def single_frame_maps(a: dict, mu, g2) -> dict:
    """Mean-related maps: valid frames, variance, share of zeros.

    ``var = mean(x^2) - mean(x)^2`` over a pixel's valid frames, and its
    photon-noise expectation ``var_q = S0 / n - (S0 + 2 S1) / n^2`` with
    ``S0 = sum_t A2(t, t)`` and ``S1 = sum_t sum_l A2(t, t + l)`` over pairs
    of valid frames (the second term is the variance of the mean).
    """
    n = a["n"]
    ok = n >= MIN_FRAMES
    nn = np.maximum(n, 1)
    mean = a["sx"] / nn
    var = np.where(ok, a["sx2"] / nn - mean**2, np.nan)
    var_q = np.where(
        ok, a["s0"] / nn - (a["s0"] + 2 * a["s1"]) / nn**2, np.nan
    )
    var_ph = g2 * mu * var_q
    return {
        "var": var,
        "var_q": var_q,
        "var_ph": var_ph,
        "var_rel": _ratio(var, var_ph, ok) - 1,
        "zero_share": np.where(n > 0, a["zeros"] / nn, np.nan),
    }


def slow_maps(a: dict, mu, g2) -> dict:
    """Variance of 64-frame block means and its photon-noise expectation.

    ``slow_var = sum_j (B_j - mean B)^2 / (J - 1)`` over the pixel's ``J``
    valid blocks; ``slow_q = (sum_j V_j - (sum_j V_j + 2 sum_j C_j) / J) /
    (J - 1)`` with ``V_j`` the photon variance of block mean ``j`` and
    ``C_j`` its covariance with the previous block's mean (kernel overlap).
    """
    nb = a["nb"]
    ok = nb >= MIN_BLOCKS
    jj = np.maximum(nb, 2)
    var = np.where(ok, (a["sb2"] - a["sb"] ** 2 / jj) / (jj - 1), np.nan)
    q = np.where(
        ok, (a["vb"] - (a["vb"] + 2 * a["cb"]) / jj) / (jj - 1), np.nan
    )
    ph = g2 * mu * q
    return {
        "n_blocks": nb,
        "slow_var": var,
        "slow_q": q,
        "slow_var_ph": ph,
        "slow_rel": _ratio(var, ph, ok) - 1,
    }


def fast_maps(a: dict, mu, g2, g3) -> dict:
    """Second and third central moments of the fast part and their
    photon-noise expectations (means over the pixel's centres)."""
    nf = a["nf"]
    ok = nf >= MIN_FRAMES
    nn = np.maximum(nf, 1)
    m1 = a["sz"] / nn
    m2 = a["sz2"] / nn
    var = np.where(ok, m2 - m1**2, np.nan)
    m3 = np.where(ok, a["sz3"] / nn - 3 * m1 * m2 + 2 * m1**3, np.nan)
    q2 = np.where(ok, a["q2"] / nn, np.nan)
    q3 = np.where(ok, a["q3"] / nn, np.nan)
    ph2 = g2 * mu * q2
    ph3 = g3 * mu * q3
    with np.errstate(invalid="ignore", divide="ignore"):
        skew = np.where(ok & (ph2 > 0), (m3 - ph3) / ph2**1.5, np.nan)
    return {
        "n_fast": nf,
        "fast_var": var,
        "fast_q": q2,
        "fast_var_ph": ph2,
        "fast_rel": _ratio(var, ph2, ok) - 1,
        "fast_m3": m3,
        "fast_q3": q3,
        "fast_m3_ph": ph3,
        "fast_skew_rel": skew,
    }


def zero_maps(hist: np.ndarray, n: np.ndarray, nu: np.ndarray, share):
    """Photon-noise share of exact zeros.

    A frame is exactly 0 when all of its ``N`` contributing samples are
    (samples are non-negative); each is with probability ``exp(-nu)``, so
    the expectation is ``sum_N h(N) exp(-N nu) / n`` over the histogram
    ``h`` of the pixel's valid frames.
    """
    k = np.arange(hist.shape[-1])
    with np.errstate(over="ignore", invalid="ignore"):
        p = np.exp(-nu[..., None] * k)
    zph = np.where(n > 0, (hist * p).sum(-1) / np.maximum(n, 1), np.nan)
    return {"zero_ph": zph, "zero_excess": share - zph}


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
    gains = [g.at(mu[ch]) for ch, g in enumerate(o.gains)]
    g2, g3, nu = (np.stack([g[i] for g in gains]) for i in range(3))
    compact = {"n_valid": a["n"], "gain_g2": g2, "gain_g3": g3}
    compact.update(single_frame_maps(a, mu, g2))
    compact.update(slow_maps(a, mu, g2))
    compact.update(fast_maps(a, mu, g2, g3))
    hist = _np(o.hist) if o.hist is not None else np.zeros(mu.shape + (1,))
    compact.update(zero_maps(hist, a["n"], nu, compact["zero_share"]))
    with np.errstate(invalid="ignore", divide="ignore"):
        v8 = np.where(a["n8"] > 0, a["v8"] / np.maximum(a["n8"], 1), np.nan)
    compact["sigma8_ph"] = np.sqrt(g2 * mu * v8)
    kappa = o.kappa[0] / o.kappa[1] if o.kappa[1] else float("nan")
    svd = shared_modes(o.bins, idx, (h, w), compact["sigma8_ph"], kappa)

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
        "n_index": int(idx.size),
    }
    gain = {
        k: _stack_curves([g.describe()[k] for g in o.gains])
        for k in ("mu", "g2", "g3", "nu_per_mu", "n_slots")
    }
    return {"maps": maps, "svd": svd, "gain": gain, "attrs": attrs}


def _stack_curves(curves: list) -> np.ndarray:
    """Per-channel curves of possibly different lengths, NaN-padded."""
    k = max((c.size for c in curves), default=0)
    out = np.full((len(curves), k), np.nan)
    for i, c in enumerate(curves):
        out[i, : c.size] = c
    return out
