"""Per-brightness null of SiLo's activity image, from a parametric bootstrap.

SiLo's activity image sums squared spatio-temporal maxima of a Z-scored,
filtered movie. At SLAP2's photon counts (0.03-1 photons per pixel and frame on
most pixels) the filtered statistic is far from Gaussian: its upper tail is
many times heavier, and its lower tail nearly empty, so a sign-flip null misses
the excess, and one threshold means a different false-positive rate on dim and
bright pixels. Here each trial's activity image is calibrated against noise
movies drawn with that trial's baseline, gain, noise correlations and skew, put
through the same filters and score:

* :func:`noise_model` measures the gain, the correlations at 1 and 2 frames and
  1 row and column, and the skew, from the residual (movie - baseline) of the
  dimmer half of the valid pixels;
* :func:`null_movie` draws filtered compound Poisson noise with that model,
  ``y = b + s (k * (N - lambda))``, ``N ~ Poisson(lambda)`` per voxel and ``k``
  a separable kernel with the measured correlations; its mean is the baseline,
  its variance ``gain * b * v`` (scaled per pixel where the rate varies across
  the kernel), and its skew ``q`` times that of filtered
  Poisson noise at that gain (larger photon events, as in
  ``slap2_f0.simulate``);
* :func:`null_calibration` scores the null movies like the trial and returns
  ``(act_im - m0) / s0``, with ``m0`` and ``s0`` the null image's centre and
  scale among pixels of similar brightness (12 equal-count bins of log photons
  per frame). By default (``null_scale="tail"``) they are the median and the
  distance from it to the 99.5th percentile in Gaussian units, so every
  brightness has the same upper tail: at 0.03-0.3 photons the null image is
  so heavy-tailed that a bin's SD hangs on a few pixels. ``"sd"`` uses the
  mean and SD.
"""

from __future__ import annotations

import time

import numpy as np

_N_BINS = 12
_TAIL_Q = 0.995
_TAIL_Z = 2.5758293035489004  # standard normal quantile at 0.995


def _pooled_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled correlation of centred values (sum of products over norms)."""
    ok = ~np.isnan(a) & ~np.isnan(b)
    a = np.where(ok, a, 0.0)
    b = np.where(ok, b, 0.0)
    den = np.sqrt((a**2).sum() * (b**2).sum())
    r = float((a * b).sum() / den) if den > 0 else 0.0
    return r if np.isfinite(r) else 0.0


def spatial_kernel(rho: float) -> np.ndarray:
    """``[a, 1, a] / (1 + 2a)``, whose neighbour correlation is ``rho``."""
    rho = float(np.clip(rho, 0.0, 0.7))
    a = 0.0 if rho == 0 else (1 - np.sqrt(1 - 2 * rho**2)) / (2 * rho)
    return np.array([a, 1.0, a]) / (1 + 2 * a)


def temporal_kernel(t1: float, t2: float) -> np.ndarray:
    """Seven symmetric taps fitted to the correlations at 1 and 2 frames.

    As ``slap2_f0.simulate.noise_kernels`` without its long-average term; short
    kernels are preferred where the two lags leave taps free. Sums to 1; only
    taps whose raw sum is at least 0.5 are considered, so it stays finite.
    """
    t1 = float(np.clip(t1, 0.0, 0.9))
    t2 = float(np.clip(t2, -0.2, 0.8))
    k1, k2, k3 = np.meshgrid(
        np.linspace(0, 1.5, 61),
        np.linspace(-0.5, 1.5, 81),
        np.linspace(-0.5, 1.5, 81),
        indexing="ij",
        sparse=True,
    )
    r0 = 1 + 2 * (k1**2 + k2**2 + k3**2)
    r1 = 2 * (k1 + k1 * k2 + k2 * k3)
    r2 = 2 * k2 + k1**2 + 2 * k1 * k3
    err = (
        (r1 / r0 - t1) ** 2
        + (r2 / r0 - t2) ** 2
        + 1e-6 * (k1**2 + k2**2 + k3**2)
    )
    # only kernels that can be normalised: a sum near 0 (negative lag-2
    # correlations) would blow the taps up
    err = np.where(1 + 2 * (k1 + k2 + k3) >= 0.5, err, np.inf)
    i = np.unravel_index(np.argmin(err), err.shape)
    a1, a2, a3 = k1.flat[i[0]], k2.flat[i[1]], k3.flat[i[2]]
    k = np.array([a3, a2, a1, 1.0, a1, a2, a3])
    return k / k.sum()


def noise_model(
    resid: np.ndarray,
    baseline: np.ndarray,
    var_factor: np.ndarray,
    vidx: np.ndarray,
    shape2d: tuple[int, int],
    frames: np.ndarray | None = None,
) -> dict:
    """Measure a trial's photon-noise model.

    Parameters
    ----------
    resid, baseline, var_factor : ndarray of shape (frames, n_valid)
        Movie minus baseline, the baseline (counts) and the variance factor
        at the valid pixels (NaN where not imaged).
    vidx : ndarray of int
        Flat index of each valid pixel in the frame (``shape2d``).
    shape2d : tuple
        Frame shape (rows, cols).
    frames : ndarray of int, optional
        Frames to measure on (default all).

    Returns
    -------
    dict
        ``gain`` (single-frame variance / (baseline x variance factor)),
        ``rho`` (correlations at 1, 2 frames, 1 row, 1 column), kernels ``kt``,
        ``kr``, ``kc``, their ``W2 = sum k^2``, ``q`` (photon event size
        against Poisson, 1-3) and the medians ``skew_measured`` and
        ``skew_poisson``.
    """
    sel = slice(None) if frames is None else frames
    r = resid[sel].astype(np.float64)
    b = baseline[sel].astype(np.float64)
    v = var_factor[sel].astype(np.float64)
    ok = ~np.isnan(r) & ~np.isnan(b) & ~np.isnan(v)
    r[~ok] = np.nan
    n = ok.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        b_mean = np.where(ok, b, 0).sum(0) / n
        bv_mean = np.where(ok, b * v, 0).sum(0) / n
        r -= np.where(ok, r, 0).sum(0) / n
        var = np.where(ok, r**2, 0).sum(0) / n
        m3 = np.where(ok, r**3, 0).sum(0) / n
        use = (n > 50) & (bv_mean > 0) & (var > 0)
    if use.sum() >= 10:
        use &= b_mean <= np.median(b_mean[use])
    if use.sum() < 10:
        raise ValueError("too few imaged pixels to measure the noise model")
    gain = float(np.median(var[use] / bv_mean[use]))

    # lags in recorded frames: each measured frame is paired with the one k
    # frames later if that was measured too, also across discarded frames
    t_idx = np.arange(resid.shape[0])[sel]
    row_of = np.full(resid.shape[0] + 3, -1)
    row_of[t_idx] = np.arange(t_idx.size)
    rho = []
    for k in (1, 2):
        j = row_of[t_idx + k]
        i = np.flatnonzero(j >= 0)
        rho.append(_pooled_corr(r[i][:, use], r[j[i]][:, use]))
    col_of = np.full(shape2d[0] * shape2d[1], -1)
    col_of[vidx[use]] = np.flatnonzero(use)
    col_of = col_of.reshape(shape2d)
    for a, c in (
        (col_of[:-1, :], col_of[1:, :]),
        (col_of[:, :-1], col_of[:, 1:]),
    ):
        pair = (a >= 0) & (c >= 0)
        rho.append(_pooled_corr(r[:, a[pair]], r[:, c[pair]]))
    kt = temporal_kernel(rho[0], rho[1])
    kr, kc = spatial_kernel(rho[2]), spatial_kernel(rho[3])
    w2 = float((kt**2).sum() * (kr**2).sum() * (kc**2).sum())
    w3 = float((kt**3).sum() * (kr**3).sum() * (kc**3).sum())

    skew_m = m3[use] / var[use] ** 1.5
    lam = b_mean[use] ** 2 * w2 / (gain * bv_mean[use])  # events per voxel
    skew_p = w3 / w2**1.5 / np.sqrt(lam)
    skew_measured = float(np.median(skew_m))
    skew_poisson = float(np.median(skew_p))
    q = skew_measured / skew_poisson if skew_poisson > 0 else 1.0
    q = float(np.clip(q, 1.0, 3.0)) if np.isfinite(q) else 1.0
    return dict(
        gain=gain,
        rho=np.array(rho),
        kt=kt,
        kr=kr,
        kc=kc,
        W2=w2,
        q=q,
        skew_measured=skew_measured,
        skew_poisson=skew_poisson,
    )


def _filter_rows_cols(d: np.ndarray, kr: np.ndarray, kc: np.ndarray):
    """Zero-padded 'same' convolution of each frame with ``kr`` and ``kc``."""
    from scipy import ndimage

    out = ndimage.correlate1d(d, kr, axis=1, mode="constant", cval=0.0)
    return ndimage.correlate1d(out, kc, axis=2, mode="constant", cval=0.0)


def _fill_unobserved(lam: np.ndarray) -> np.ndarray:
    """Rates at non-finite voxels from the nearest earlier observed frame of
    the same pixel (the next later one before the first); 0 if never."""
    ok = np.isfinite(lam)
    if ok.all():
        return lam
    n_t = lam.shape[0]
    t = np.arange(n_t, dtype=np.int32)[:, None]
    src = np.maximum.accumulate(np.where(ok, t, -1), axis=0)
    nxt = np.minimum.accumulate(np.where(ok, t, n_t)[::-1], axis=0)[::-1]
    src = np.where(src >= 0, src, nxt)
    never = src >= n_t
    out = np.take_along_axis(lam, np.minimum(src, n_t - 1), axis=0)
    out[never] = 0.0
    return out


def _mixing_correction(lam, vidx, rows, cols, kr, kc) -> np.ndarray:
    """Per-pixel factor that gives each pixel the variance ``gain * b * v``.

    The spatial kernel mixes the neighbours' Poisson fluctuations into a
    pixel, so where the rate changes across the kernel (next to bright
    structures) the variance would follow the neighbours' rates. This scales
    each pixel by ``sqrt(sum(k^2) lam / (k^2 * lam))`` of the time-mean rates
    (1 where they are uniform).
    """
    from scipy import ndimage

    lam_mean = np.zeros(rows * cols)
    lam_mean[vidx] = lam.mean(0)
    mixed = ndimage.correlate1d(
        lam_mean.reshape(rows, cols), kr**2, axis=0, mode="constant"
    )
    mixed = ndimage.correlate1d(mixed, kc**2, axis=1, mode="constant")
    mixed = mixed.ravel()[vidx]
    own = float((kr**2).sum() * (kc**2).sum()) * lam_mean[vidx]
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.sqrt(own / mixed)
    f[~np.isfinite(f) | ~(own > 0)] = 1.0
    return f.astype(np.float32)


def null_movie(
    baseline: np.ndarray,
    var_factor: np.ndarray,
    std: np.ndarray,
    nans: np.ndarray,
    vidx: np.ndarray,
    noise: dict,
    seed: int,
    block: int = 256,
) -> np.ndarray:
    """A Z-scored noise movie with the trial's baseline and noise model.

    Parameters
    ----------
    baseline, var_factor, std : ndarray of shape (frames, n_valid)
        Baseline (counts), variance factor and the Z-score's SD at the valid
        pixels.
    nans : ndarray of bool, shape (frames, rows, cols)
        Voxels not imaged; NaN in the output, as in the trial.
    vidx : ndarray of int
        Flat index of each valid pixel in the frame.
    noise : dict
        From :func:`noise_model`.
    seed : int
        Seed of the draw.
    block : int
        Frames drawn and filtered at a time.

    Returns
    -------
    ndarray of float32, shape of ``nans``
        ``(y - baseline) / std``: what the localization sees after its
        high-pass if the trial held only noise.
    """
    rng = np.random.default_rng(seed)
    n_t, rows, cols = nans.shape
    kt, kr, kc = noise["kt"], noise["kr"], noise["kc"]
    h = (kt.size - 1) // 2
    s = noise["q"] * noise["gain"] / noise["W2"]
    lam_scale = noise["W2"] / (noise["q"] ** 2 * noise["gain"])
    with np.errstate(invalid="ignore", divide="ignore"):
        lam_all = baseline.astype(np.float64) * lam_scale / var_factor
        gain_z = (s * var_factor / std).astype(np.float32)
    # unobserved voxels (discarded frames) keep their latent noise, which
    # reaches the observed frames next to them through the temporal kernel;
    # the observation mask is applied after the filters
    lam_all = _fill_unobserved(lam_all)
    lam_all[~(lam_all > 0)] = 0.0
    gain_z *= _mixing_correction(lam_all, vidx, rows, cols, kr, kc)
    out = np.full(nans.shape, np.nan, dtype=np.float32)
    flat_out = out.reshape(n_t, -1)
    drawn = np.zeros((0, rows * cols), dtype=np.float32)
    d0 = 0  # frame index of drawn[0]
    for a in range(0, n_t, block):
        b = min(n_t, a + block)
        hi = min(n_t, b + h)
        start = d0 + drawn.shape[0]
        if hi > start:
            lam = lam_all[start:hi]
            fresh = np.zeros((hi - start, rows * cols), dtype=np.float32)
            fresh[:, vidx] = (rng.poisson(lam) - lam).astype(np.float32)
            drawn = np.concatenate([drawn, fresh])
        drop = max(0, a - h - d0)
        if drop:
            drawn = drawn[drop:]
            d0 += drop
        f = _filter_rows_cols(drawn.reshape(-1, rows, cols), kr, kc)
        f = np.concatenate(
            [
                np.zeros((max(0, h - a), rows, cols), np.float32),
                f,
                np.zeros((max(0, b + h - n_t), rows, cols), np.float32),
            ]
        )
        n_out = f.shape[0] - kt.size + 1  # 'valid' in time: frames a .. b-1
        g = np.zeros((n_out, rows, cols), np.float32)
        for j, w in enumerate(kt[::-1]):
            g += np.float32(w) * f[j : j + n_out]
        f = g
        flat_out[a:b, vidx] = f.reshape(b - a, -1)[:, vidx] * gain_z[a:b]
    out[nans] = np.nan
    return out


def null_calibration(
    act_im: np.ndarray,
    baseline: np.ndarray,
    var_factor: np.ndarray,
    std: np.ndarray,
    nans: np.ndarray,
    valid: np.ndarray,
    vidx: np.ndarray,
    noise: dict,
    params,
    sd_u: float | None,
) -> tuple[np.ndarray, dict]:
    """Calibrate a trial's activity image per pixel brightness.

    Runs ``params.null_draws`` null movies (:func:`null_movie`) through the
    trial's activity score (with the trial's ``sd_u``) and returns
    ``(act_im - m0) / s0``: ``m0`` and ``s0`` are the null image's centre and
    scale among pixels of similar brightness (12 equal-count bins of the log
    mean photons per frame, linear in between, constant beyond the outer bins'
    centres): with ``params.null_scale`` "tail" the median and
    ``(q99.5 - median) / 2.576``, with "sd" the mean and SD. Also returns
    the noise model, bin brightness (photons per frame), ``m0``, ``s0`` and
    the seconds taken.
    """
    from .localization import activity_score

    t0 = time.time()
    with np.errstate(invalid="ignore", divide="ignore"):
        ph = np.nanmean(
            baseline.astype(np.float64)
            / (noise["gain"] * var_factor.astype(np.float64)),
            axis=0,
        )
    photons = np.full(valid.size, np.nan)
    photons[vidx] = ph
    photons = photons.reshape(valid.shape)

    null_ims = []
    for k in range(params.null_draws):
        z0 = null_movie(
            baseline, var_factor, std, nans, vidx, noise, params.null_seed + k
        )
        sk0, _ = activity_score(z0, nans, params, sd_u)
        sk0[~valid] = np.nan
        null_ims.append(sk0)
        del z0
    null_ims = np.stack(null_ims)

    # pixels that rarely have a full 3x3 neighbourhood (the edges of the
    # imaged area) can never be maxima there; they do not set the scale
    from scipy import ndimage

    edge = (
        ndimage.maximum_filter(
            nans, size=(1, 3, 3), mode="constant", cval=1
        ).mean(0)
        > 0.5
    )
    ok = valid & ~edge & (photons > 0) & np.isfinite(null_ims).all(0)
    if ok.sum() < 2:
        raise ValueError("too few imaged pixels for the null calibration")
    lp = np.log(photons[ok])
    edges = np.unique(np.quantile(lp, np.linspace(0, 1, _N_BINS + 1)))
    if edges.size == 1:  # a single brightness: one bin
        edges = np.repeat(edges, 2)
    bins = np.clip(np.searchsorted(edges, lp, side="right") - 1, 0, None)
    bins = np.minimum(bins, edges.size - 2)
    vals = null_ims[:, ok]
    centre, m0, s0 = [], [], []
    for k in range(edges.size - 1):
        in_bin = bins == k
        x = vals[:, in_bin].ravel()
        if x.size < 2:
            continue
        if params.null_scale == "tail":
            mid_x = float(np.median(x))
            sd = (float(np.quantile(x, _TAIL_Q)) - mid_x) / _TAIL_Z
        else:
            mid_x, sd = float(np.mean(x)), float(np.std(x, ddof=1))
        if sd > 0:
            centre.append(float(np.median(lp[in_bin])))
            m0.append(mid_x)
            s0.append(sd)
    if not centre:
        raise ValueError("the null activity image has no spread")
    centre, m0, s0 = np.array(centre), np.array(m0), np.array(s0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lp_all = np.log(photons)
    mid = (centre.size - 1) // 2
    m_map = np.interp(lp_all, centre, m0)
    s_map = np.interp(lp_all, centre, s0)
    bad = ~np.isfinite(lp_all)
    m_map[bad], s_map[bad] = m0[mid], s0[mid]
    out = (act_im - m_map) / s_map
    info = dict(
        noise=noise,
        photons=np.exp(centre),
        m0=m0,
        s0=s0,
        draws=params.null_draws,
        seconds=time.time() - t0,
    )
    return out, info
