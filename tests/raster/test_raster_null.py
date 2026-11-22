"""Tests of the bootstrap null of the raster SiLo activity image."""

import numpy as np

from giant_python.extraction.raster.localization import (
    RasterLocalizationParams,
    localize_sources,
)
from giant_python.extraction.raster.null import (
    noise_model,
    null_movie,
    spatial_kernel,
    temporal_kernel,
)


def _gradient_movie(rng, t=1500, rows=40, cols=90, gain=8.0, margin=0):
    """Pure photon noise with brightness 0.03-30 photons across the columns,
    filtered as image formation does (correlations in time and space); with a
    margin of unimaged (NaN) pixels around it, as SLAP2 frames have."""
    photons = np.exp(np.linspace(np.log(0.03), np.log(30), cols))
    b = np.broadcast_to(photons[None, :] * gain, (rows, cols))
    base = np.broadcast_to(b[None], (t, rows, cols)).astype(np.float32)
    noise = dict(
        gain=gain,
        kt=temporal_kernel(0.6, 0.25),
        kr=spatial_kernel(0.23),
        kc=spatial_kernel(0.37),
        q=1.0,
    )
    noise["W2"] = float(
        (noise["kt"] ** 2).sum()
        * (noise["kr"] ** 2).sum()
        * (noise["kc"] ** 2).sum()
    )
    vidx = np.arange(rows * cols)
    nans = np.zeros((t, rows, cols), bool)
    flat = base.reshape(t, -1)
    ones = np.ones_like(flat)
    z = null_movie(flat, ones, ones, nans, vidx, noise, seed=3)
    movie = base + z  # std = 1: z is the noise in counts
    if margin:
        pad = ((0, 0), (margin, margin), (margin, margin))
        movie = np.pad(movie, pad, constant_values=np.nan)
        base = np.pad(base, pad, constant_values=np.nan)
    return movie.astype(np.float32), base, noise, photons


def test_kernels_give_the_correlations():
    """The fitted kernels' autocorrelations are the requested ones."""
    k = temporal_kernel(0.6, 0.25)
    ac = np.correlate(k, k, "full")[k.size - 1 :] / (k**2).sum()
    assert abs(ac[1] - 0.6) < 0.01 and abs(ac[2] - 0.25) < 0.01
    s = spatial_kernel(0.36)
    assert abs((s[0] * s[1] + s[1] * s[2]) / (s**2).sum() - 0.36) < 1e-9
    assert np.isclose(k.sum(), 1) and np.isclose(s.sum(), 1)


def test_noise_model_recovers_gain_and_correlations():
    """A null movie's measured model is the model it was drawn with."""
    rng = np.random.default_rng(0)
    movie, base, noise, _ = _gradient_movie(rng)
    t = movie.shape[0]
    resid = (movie - base).reshape(t, -1)
    m = noise_model(
        resid,
        base.reshape(t, -1),
        np.ones_like(resid),
        np.arange(resid.shape[1]),
        movie.shape[1:],
    )
    assert abs(m["gain"] / noise["gain"] - 1) < 0.05
    np.testing.assert_allclose(m["rho"], [0.6, 0.25, 0.23, 0.37], atol=0.03)
    assert 1.0 <= m["q"] < 1.3


def test_null_mean_is_the_baseline_and_skew_is_poisson():
    """Null noise has the baseline's mean, gain x b variance, Poisson skew."""
    rng = np.random.default_rng(1)
    movie, base, noise, photons = _gradient_movie(rng, t=3000, cols=6)
    r = movie - base
    sd_frame = np.sqrt(noise["gain"] * base[0, 0])
    assert np.all(np.abs(r.mean((0, 1))) < 0.1 * sd_frame)
    var = r.var((0, 1))
    np.testing.assert_allclose(var / (noise["gain"] * base[0, 0]), 1, rtol=0.1)
    skew = (r**3).mean((0, 1)) / var**1.5
    w2 = noise["W2"]
    w3 = (
        (noise["kt"] ** 3).sum()
        * (noise["kr"] ** 3).sum()
        * (noise["kc"] ** 3).sum()
    )
    want = w3 / w2**1.5 / np.sqrt(photons * w2)
    np.testing.assert_allclose(skew[:3], want[:3], rtol=0.25)


def _tail(x):
    """Median and (99.5th percentile - median) of an image's pixels."""
    x = x[np.isfinite(x)]
    m = np.median(x)
    return m, np.quantile(x, 0.995) - m


def test_bootstrap_calibrates_dim_and_bright_pixels():
    """On a trial of pure photon noise, the calibrated image has the same
    centre and upper tail at every brightness; uncalibrated, dim pixels'
    tail is heavier."""
    rng = np.random.default_rng(2)
    movie, base, _, photons = _gradient_movie(rng, rows=80, margin=6)
    p_raw = RasterLocalizationParams(variance_floor=False)
    p_cal = RasterLocalizationParams(
        variance_floor=False, null="bootstrap", null_seed=11, null_draws=2
    )
    raw = localize_sources(movie, None, p_raw, base, find_peaks=False)
    cal = localize_sources(movie, None, p_cal, base, find_peaks=False)
    inner = (slice(6 + 2, -6 - 2), slice(6 + 2, -6 - 2))
    a_raw, a_cal = raw.act_im[inner], cal.act_im[inner]
    cols = photons[2:-2]
    dim, bright = cols < 0.3, cols > 3
    assert _tail(a_raw[:, dim])[1] > 1.3 * _tail(a_raw[:, bright])[1]
    for sel in (dim, bright):
        m, t = _tail(a_cal[:, sel])
        assert abs(m) < 0.3
        assert 0.7 * 2.576 < t < 1.3 * 2.576
    info = cal.info["null"]
    assert info["s0"][0] > 1.3 * info["s0"][-1]
