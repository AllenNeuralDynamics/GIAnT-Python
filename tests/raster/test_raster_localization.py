"""Tests of the multi-ROI (raster) SiLo localization port and its options."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from giant_python.extraction.raster.localization import (
    RasterLocalizationParams,
    activity_score,
    gauss_filter,
    localize_sources,
    matlab_prctile,
    movmean_nan,
    movmedian_nan,
)

FIXTURE = Path(__file__).parent / "data" / "matlab_localization.h5"


def _fixture():
    """The synthetic trial and GIAnT-MATLAB's results on it.

    HDF5 holds MATLAB's (rows, cols, frames) arrays as (frames, cols, rows).
    """
    with h5py.File(FIXTURE, "r") as f:
        movie = f["movie"][()].transpose(0, 2, 1)
        vf = f["var_factor"][()].transpose(0, 2, 1)
        base = f["baseline"][()].transpose(0, 2, 1)
        ref = {}
        for kind in ("movmedian", "external"):
            g = f["matlab"][kind]
            ref[kind] = dict(
                act_im=g["act_im"][()].T,
                summary=g["summary"][()].T,
                peaks={
                    th: np.atleast_2d(g[f"peaks_{th}"][()].T)
                    for th in (3, 5, 8)
                },
            )
    return movie, vf, base, ref


def _seed_offsets(py, ml):
    """Distance from each MATLAB seed (1-based) to the nearest Python seed."""
    if ml.size == 0 or py.size == 0:
        return np.array([])
    d = np.hypot(
        ml[:, 1:2] - 1 - py[None, :, 1], ml[:, 2:3] - 1 - py[None, :, 2]
    )
    return d.min(1)


@pytest.mark.parametrize("kind", ["movmedian", "external"])
def test_matches_matlab(kind):
    """Activity image, summary and seeds agree with GIAnT-MATLAB."""
    movie, vf, base, ref = _fixture()
    r = localize_sources(
        movie,
        vf,
        RasterLocalizationParams(),
        base if kind == "external" else None,
    )
    want = ref[kind]
    assert np.array_equal(np.isnan(r.act_im), np.isnan(want["act_im"]))
    ok = np.isfinite(want["act_im"])
    scale = np.max(np.abs(want["act_im"][ok]))
    assert np.max(np.abs(r.act_im[ok] - want["act_im"][ok])) < 1e-3 * scale
    ok = np.isfinite(want["summary"])
    assert np.corrcoef(r.summary[ok], want["summary"][ok])[0, 1] > 0.99999
    from giant_python.extraction.raster.peaks import get_act_im_peaks

    # the optimizers differ (lsqcurvefit vs least_squares), so marginal seeds
    # among many can differ; the strong ones agree
    py = get_act_im_peaks(r.summary, 8)
    ml = want["peaks"][8]
    assert abs(py.shape[0] - ml.shape[0]) <= 1
    assert np.mean(_seed_offsets(py, ml) < 0.1) >= 0.9


def test_peak_port_on_matlab_summary():
    """Seeds from MATLAB's own summary image agree with MATLAB's seeds."""
    from giant_python.extraction.raster.peaks import get_act_im_peaks

    _, _, _, ref = _fixture()
    for kind in ("movmedian", "external"):
        for th, near in ((5, 0.8), (8, 0.9)):
            ml = ref[kind]["peaks"][th]
            py = get_act_im_peaks(ref[kind]["summary"], th)
            assert abs(py.shape[0] - ml.shape[0]) <= max(1, 0.1 * ml.shape[0])
            assert np.mean(_seed_offsets(py, ml) < 0.25) >= near


def test_moving_windows_match_matlab():
    """smoothdata movmean/movmedian with 'omitnan' and an even window."""
    x = np.array([1, np.nan, 3, 4, np.nan, 6, 7, 8, 9, 10], np.float32)
    mean = movmean_nan(x[:, None], 4).ravel()
    med = movmedian_nan(x[:, None], 4).ravel()
    np.testing.assert_allclose(
        mean, [1, 2, 8 / 3, 3.5, 13 / 3, 17 / 3, 7, 7.5, 8.5, 9], rtol=1e-6
    )
    np.testing.assert_allclose(med, [1, 2, 3, 3.5, 4, 6, 7, 7.5, 8.5, 9])
    assert matlab_prctile(np.array([3, 1, 4, 1, 5, 9, 2, 6.0]), 90) == (
        pytest.approx(8.1)
    )


def test_no_surround_is_the_gaussian_alone():
    """surround=False: the spatial filter is the Gaussian, without the DoG."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((40, 30, 32)).astype(np.float32)
    nans = np.zeros(z.shape, bool)
    p0 = RasterLocalizationParams(surround=False)
    sk, _ = activity_score(z.copy(), nans, p0)
    # the same score by hand: exponential filter, Gaussian, maxima squared
    from giant_python.extraction.raster.localization import temporal_filter

    g = gauss_filter(temporal_filter(z.copy(), p0.tau_frames), p0.sigma_px)
    sk_with, _ = activity_score(z.copy(), nans, RasterLocalizationParams())
    assert not np.allclose(sk, sk_with)
    last = g.shape[0] - int(np.ceil(1.5 * p0.tau_frames))
    from scipy import ndimage

    want = np.zeros(g.shape[1:])
    for fr in range(1, last):
        mx = ndimage.maximum_filter(g[fr], 3, mode="constant", cval=0)
        is_max = (g[fr] == mx) & (g[fr] > g[fr - 1]) & (g[fr] >= g[fr + 1])
        want[is_max] += g[fr][is_max].astype(float) ** 2
    want /= 300 + (last - 1)
    # the 3x3 maximum at the border sees MATLAB's zero padding in both
    np.testing.assert_allclose(sk, want, rtol=1e-5, atol=1e-9)


def test_soft_threshold_ignores_noise_maxima():
    """Maxima below soft_th add nothing; a strong event still scores."""
    rng = np.random.default_rng(1)
    z = rng.standard_normal((300, 24, 24)).astype(np.float32)
    nans = np.zeros(z.shape, bool)
    p = RasterLocalizationParams(score="soft_threshold", soft_th=2.0)
    sk_noise, sd = activity_score(z.copy(), nans, p)
    sk_sq, _ = activity_score(z.copy(), nans, RasterLocalizationParams())
    assert sd > 0
    assert np.mean(sk_noise == 0) > 0.2 and np.mean(sk_sq == 0) < 0.05
    z[150:153, 12, 12] += 40
    sk_ev, _ = activity_score(z.copy(), nans, p, sd)
    assert np.unravel_index(np.argmax(sk_ev), sk_ev.shape) == (12, 12)


def test_params_validation():
    """Unknown option values are rejected."""
    with pytest.raises(ValueError):
        RasterLocalizationParams(null="signflip")
    with pytest.raises(ValueError):
        RasterLocalizationParams(score="energy")
    with pytest.raises(ValueError):
        RasterLocalizationParams(null_draws=0)
    with pytest.raises(ValueError):
        RasterLocalizationParams(soft_th=-1)


def test_external_baseline_shape_checked():
    """An external baseline must have the movie's shape."""
    movie = np.ones((50, 10, 10), np.float32)
    with pytest.raises(ValueError):
        localize_sources(movie, None, baseline=np.ones((50, 10, 9)))


def test_no_variance_floor_needs_an_external_baseline():
    """Without the floor, the moving median's low bias at low counts would
    blow up dim pixels' Z-scores, so an external baseline is required."""
    movie = np.ones((50, 10, 10), np.float32)
    p = RasterLocalizationParams(variance_floor=False)
    with pytest.raises(ValueError, match="external baseline"):
        localize_sources(movie, None, p)


def test_moving_mean_keeps_precision_on_long_recordings():
    """Prefix sums in float64: no drift over a long float32 recording."""
    x = np.full((24000, 2), 1.1, np.float32)
    out = movmean_nan(x, 8)
    np.testing.assert_allclose(out[-50:], np.float32(1.1), rtol=1e-6)


def test_no_floor_with_a_uniform_external_baseline():
    """A spatially uniform baseline (no pixel above the 90th percentile of
    brightness) still gives a finite activity image without the floor."""
    rng = np.random.default_rng(3)
    base = np.full((600, 20, 20), 9.0, np.float32)
    movie = (base + 3 * rng.standard_normal(base.shape)).astype(np.float32)
    p = RasterLocalizationParams(variance_floor=False)
    loc = localize_sources(movie, None, p, base, find_peaks=False)
    assert np.isfinite(loc.act_im[loc.valid]).all()
    assert loc.info["Vk"] > 0


def test_no_valid_pixels_keeps_find_peaks_false():
    """The early return for a trial without valid pixels honours
    ``find_peaks``: peaks are None when seeding was not requested."""
    movie = np.full((50, 10, 10), np.nan, np.float32)
    with pytest.warns(UserWarning):
        loc = localize_sources(movie, find_peaks=False)
    assert loc.peaks is None
    with pytest.warns(UserWarning):
        assert localize_sources(movie).peaks.shape == (0, 4)


def test_pixels_mostly_missing_leave_the_variance_model():
    """As MATLAB (``varIM(nanFrac > 0.4) = nan``): valid pixels missing
    more than 40% of frames do not change Vb."""
    rng = np.random.default_rng(5)
    t = 600
    movie = (20 + 3 * rng.standard_normal((t, 20, 20))).astype(np.float32)
    gone = rng.random(t) < 0.45
    p = RasterLocalizationParams(nan_thresh=0.5)
    vb = []
    for scale in (0.1, 30.0):  # those pixels' noise: tiny, or huge
        m = movie.copy()
        block = 20 + scale * rng.standard_normal((t, 5, 5))
        m[:, :5, :5] = block.astype(np.float32)
        m[gone, :5, :5] = np.nan
        loc = localize_sources(m, None, p, find_peaks=False)
        assert loc.valid[:5, :5].all()
        vb.append(loc.info["Vb"])
    assert vb[0] == vb[1]
