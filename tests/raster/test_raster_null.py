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


def _gradient_movie(rng, t=1500, rows=40, cols=90, gain=8.0):
    """Pure photon noise with brightness 0.03-30 photons across the columns,
    filtered as image formation does (correlations in time and space)."""
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
    np.testing.assert_allclose(r.mean((0, 1)), 0, atol=0.05 * base[0, 0])
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


def test_bootstrap_calibrates_dim_and_bright_pixels():
    """On a trial of pure photon noise, the calibrated image has mean 0 and
    SD 1 at every brightness; uncalibrated, dim pixels are inflated."""
    rng = np.random.default_rng(2)
    movie, base, _, photons = _gradient_movie(rng)
    p_raw = RasterLocalizationParams(variance_floor=False, nan_thresh=0.33)
    p_cal = RasterLocalizationParams(
        variance_floor=False, null="bootstrap", null_seed=11
    )
    raw = localize_sources(movie, None, p_raw, base, find_peaks=False)
    cal = localize_sources(movie, None, p_cal, base, find_peaks=False)
    inner = (slice(4, -4), slice(4, -4))
    a_raw, a_cal = raw.act_im[inner], cal.act_im[inner]
    cols = photons[4:-4]
    dim, bright = cols < 0.3, cols > 3
    assert a_raw[:, dim].std() > 1.3 * a_raw[:, bright].std()
    for sel in (dim, bright):
        assert abs(a_cal[:, sel].mean()) < 0.25
        assert 0.75 < a_cal[:, sel].std() < 1.3
    info = cal.info["null"]
    assert info["s0"][0] > 1.3 * info["s0"][-1]
