"""Tests of the solver's baseline modes (the variants R0, N0-N3 of the
SILo port's baseline test): an external baseline held, scaled per pixel,
scaled by a shared time course, or corrected by ``fitB``."""

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from giant_python.extraction.raster.solver import (
    SolverOptions,
    conv_same,
    solve_subproblem,
)
from giant_python.extraction.raster.solver_baseline import (
    fit_baseline,
    fit_gain,
    fit_network_gain,
    gain_programs,
    outside_halves,
    shared_time_course,
)
from giant_python.extraction.raster.solver_setup import (
    SolverParams,
    Subproblem,
)

SUBPROBLEM = Path(__file__).parent / "data" / "matlab_solver_subproblem.h5"
PARAMS = SolverParams()
KEEP = SolverOptions(keep_baseline=True)


def _corr(a, b):
    """Pearson correlation of two traces."""
    return float(np.corrcoef(np.ravel(a), np.ravel(b))[0, 1])


@pytest.fixture(scope="module")
def real():
    """A real two-source subproblem (activity channel only) and the
    default solver's result on it, with its baseline."""
    with h5py.File(SUBPROBLEM, "r") as f:
        sub = Subproblem(
            y=f["in/y"][()][:1].astype(np.float64),
            finv=f["in/finv"][()].astype(np.float64),
            sel=f["in/sel"][()].astype(bool),
            sources_rc=f["in/sources_rc"][()],
        )
    return sub, solve_subproblem(sub, PARAMS, KEEP)


def _synthetic(seed=0, n_t=1600, shared=None, scale=1.0):
    """A one-source subproblem drawn from a known baseline and events.

    The baseline is a bright stripe times a slow drift (and ``shared``, a
    time course common to all pixels, if given); the data are ``F``
    times Poisson counts of mean ``T / F``, with the variance multiplier
    ``F`` 0.2. Returns the subproblem (with the true
    baseline times ``scale`` as its external baseline, without the shared
    course), the true baseline and the true events.
    """
    rng = np.random.default_rng(seed)
    sel = np.ones((13, 13), bool)
    rows, cols = np.nonzero(sel.T)[::-1]
    t = np.arange(n_t)
    slow = 1 + 0.1 * np.sin(2 * np.pi * t / n_t)
    space = 2 + 3 * np.exp(-((cols - 6.0) ** 2) / 8)
    b = space[:, None] * slow[None, :]
    if shared is not None:
        b_true = b * shared[None, :]
    else:
        b_true = b
    h = np.exp(-((rows - 6.0) ** 2 + (cols - 6.0) ** 2) / (2 * 1.33**2))
    h /= h.sum()
    s = np.zeros(n_t)
    s[rng.choice(np.arange(20, n_t - 40), 25, replace=False)] = 20 / 0.19
    x = conv_same(s[None], PARAMS.k)[0]
    finv = np.full(b.shape, 0.2)
    z = finv * rng.poisson((b_true + np.outer(h, x)) / finv)
    sub = Subproblem(
        y=z[None],
        finv=finv,
        sel=sel,
        sources_rc=np.array([[6, 6]]),
        baseline=b * scale,
    )
    return sub, b_true, s


def test_params_and_subproblem_check_the_baseline():
    """Unknown modes, a misfitting or non-finite baseline are refused."""
    with pytest.raises(ValueError, match="baseline_mode"):
        SolverParams(baseline_mode="network")
    for bad in (0.0, 1.0):
        with pytest.raises(ValueError, match="outside_footprint"):
            SolverParams(outside_footprint=bad)
    sub, _, _ = _synthetic()
    with pytest.raises(ValueError, match="pixels and frames"):
        replace(sub, baseline=sub.baseline[:, :-1], _coords=None)
    bad = sub.baseline.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        replace(sub, baseline=bad, _coords=None)


def test_modes_other_than_fit_need_a_baseline(real):
    """Without Subproblem.baseline only the default mode runs."""
    sub, _ = real
    for mode in ("fixed", "gain", "gain_shared", "fit_from"):
        with pytest.raises(ValueError, match="Subproblem.baseline"):
            solve_subproblem(sub, replace(PARAMS, baseline_mode=mode))


def test_fit_mode_ignores_an_external_baseline(real):
    """R0 gives the same results with and without an external baseline."""
    sub, r0 = real
    with_b = replace(sub, baseline=2 * r0.baseline[0], _coords=None)
    again = solve_subproblem(with_b, PARAMS, KEEP)
    for name in ("footprints", "events", "ls", "f0", "baseline"):
        np.testing.assert_array_equal(getattr(again, name), getattr(r0, name))


def test_fixed_mode_holds_the_baseline(real):
    """N0 keeps the given baseline (bounded below) and, given R0's final
    baseline, finds R0's sources."""
    sub, r0 = real
    b = r0.baseline[0]
    given = replace(sub, baseline=b, _coords=None)
    n0 = solve_subproblem(given, replace(PARAMS, baseline_mode="fixed"), KEEP)
    np.testing.assert_array_equal(
        n0.baseline[0], np.fmax(PARAMS.min_baseline, b)
    )
    for j in range(2):
        cos = n0.footprints[:, j] @ r0.footprints[:, j]
        cos /= np.linalg.norm(n0.footprints[:, j])
        cos /= np.linalg.norm(r0.footprints[:, j])
        assert cos > 0.99
        x0 = conv_same(r0.events[0, j], PARAMS.k)
        xn = conv_same(n0.events[0, j], PARAMS.k)
        assert _corr(x0, xn) > 0.9
        assert 0.9 < xn.sum() / x0.sum() < 1.1
    assert "baseline_step" not in n0.info


def _mean_error(b, truth):
    """Mean absolute error of the pixels' time means, over the truth's
    mean."""
    return float(np.abs(b.mean(1) - truth.mean(1)).mean() / truth.mean())


def test_fit_from_mode_corrects_little_from_the_right_baseline():
    """N3 given the true baseline stays near it, as R0 comes near it."""
    sub, b_true, _ = _synthetic(seed=7)
    n3 = solve_subproblem(sub, replace(PARAMS, baseline_mode="fit_from"), KEEP)
    r0 = solve_subproblem(sub, PARAMS, KEEP)
    assert _mean_error(n3.baseline[0], b_true) < 0.01
    assert _mean_error(r0.baseline[0], b_true) < 0.01


def test_gain_mode_undoes_a_scale_error():
    """N1 given the true baseline times 1.1 finds gains near 1 / 1.1; N0
    keeps the error."""
    sub, b_true, _ = _synthetic(seed=7, scale=1.1)
    n1 = solve_subproblem(sub, replace(PARAMS, baseline_mode="gain"), KEEP)
    gain = n1.info["baseline_step"]["gain"]
    assert gain.shape == (sub.n_pixels,)
    assert abs(np.median(gain) - 1 / 1.1) < 0.01
    np.testing.assert_allclose(
        n1.baseline[0],
        np.fmax(PARAMS.min_baseline, gain[:, None] * sub.baseline),
    )
    assert _mean_error(n1.baseline[0], b_true) < 0.01
    n0 = solve_subproblem(sub, replace(PARAMS, baseline_mode="fixed"), KEEP)
    assert abs(_mean_error(n0.baseline[0], b_true) - 0.1) < 1e-9


def test_gain_programs_without_sources_are_weighted_ratios():
    """With no activity the gain is the weighted ratio of data and
    predictor."""
    rng = np.random.default_rng(3)
    y = rng.random((4, 50)) + 1
    q = rng.random((4, 50)) + 0.5
    finv = rng.random((4, 50)) + 0.5
    x = np.zeros((1, 50))
    h = np.full((4, 1), 0.25)
    b, coef = fit_gain(q, y, x, h, finv, q, 0.01)
    w = 1 / finv
    want = (w * y * q).sum(1) / (w * q * q).sum(1)
    np.testing.assert_allclose(coef[:, -1], want, rtol=1e-9)
    np.testing.assert_allclose(b, want[:, None] * q, rtol=1e-9)
    prog = gain_programs(q, y, x, h, finv, q)
    assert prog["g"].shape == (4, 2, 2) and prog["n_sources"] == 1
    assert (prog["lb"] == 0).all()


def test_shared_time_course_shrinks_noise_to_its_mean():
    """Without a shared change the factor is flat; with one it follows."""
    rng = np.random.default_rng(4)
    n_px, n_t = 200, 2000
    q = np.full((n_px, n_t), 3.0)
    src = np.zeros((n_px, n_t))
    finv = np.ones((n_px, n_t))
    y = rng.poisson(q).astype(np.float64)
    c, rec = shared_time_course(y, q, src, finv)
    assert rec["s2"] < 1e-4
    assert np.ptp(c) < 0.01
    assert abs(rec["mean"] - 1) < 0.01
    # Poisson of mean 3 against the loss's variance 3 + 1
    assert abs(rec["alpha"] - 0.75) < 0.03
    true = 1 + 0.15 * np.sin(np.arange(n_t) / 7.0)
    y = rng.poisson(q * true[None]).astype(np.float64)
    c, rec = shared_time_course(y, q, src, finv)
    assert rec["s2"] > 0
    assert _corr(c, true) > 0.85
    raw_err = np.abs(rec["raw"] - true).mean()
    assert np.abs(c - true).mean() < raw_err


def test_gain_shared_mode_finds_a_shared_change():
    """N2 recovers a fast change shared by all pixels that the external
    baseline lacks, and N1 does not."""
    n_t = 1600
    shared = 1 + 0.08 * np.sign(np.sin(np.arange(n_t) / 3.0))
    sub, b_true, _ = _synthetic(seed=5, n_t=n_t, shared=shared)
    n2 = solve_subproblem(
        sub, replace(PARAMS, baseline_mode="gain_shared"), KEEP
    )
    step = n2.info["baseline_step"]
    assert step["n_outside"] > 50
    assert _corr(step["shared"], shared) > 0.8
    n1 = solve_subproblem(sub, replace(PARAMS, baseline_mode="gain"), KEEP)
    err2 = np.abs(n2.baseline[0] - b_true).mean()
    err1 = np.abs(n1.baseline[0] - b_true).mean()
    assert err2 < 0.7 * err1


def test_gain_shared_without_outside_pixels_is_gain():
    """With too few pixels outside the footprints the factor is 1."""
    sub, _, _ = _synthetic(seed=6)
    params = replace(PARAMS, baseline_mode="gain_shared")
    rows, cols = np.nonzero(sub.sel)
    b = sub.baseline
    rng = np.random.default_rng(0)
    h = rng.random((sub.n_pixels, 1)) + 1  # flat footprint: nothing outside
    h /= h.sum()
    x = np.zeros((1, sub.n_frames))
    bb, info = fit_network_gain(
        b, b, sub.y[0], x, h, sub.finv, params, shared=True
    )
    assert info["n_outside"] == 0
    np.testing.assert_array_equal(info["shared"], 1.0)
    b1, info1 = fit_network_gain(
        b, b, sub.y[0], x, h, sub.finv, params, shared=False
    )
    np.testing.assert_array_equal(bb, b1)
    assert rows.size == cols.size == sub.n_pixels


def test_fit_baseline_offset():
    """A zero offset changes nothing; with the true baseline as offset the
    fitted correction is small."""
    sub, b_true, s = _synthetic(seed=7)
    y = sub.y[0]
    x = conv_same(s[None], PARAMS.k)
    rows, cols = np.nonzero(sub.sel.T)[::-1]
    h = np.exp(-((rows - 6.0) ** 2 + (cols - 6.0) ** 2) / (2 * 1.33**2))
    h = (h / h.sum())[:, None]
    b0 = sub.baseline
    plain = fit_baseline(b0, y, x, h, sub.finv, sub.sel, PARAMS)
    zero = fit_baseline(
        b0, y, x, h, sub.finv, sub.sel, PARAMS, offset=np.zeros_like(y)
    )
    np.testing.assert_allclose(zero, plain, rtol=1e-12, atol=1e-12)
    corr = fit_baseline(b0, y, x, h, sub.finv, sub.sel, PARAMS, offset=b0)
    err_from = np.abs(corr - b_true).mean() / b_true.mean()
    assert err_from < 0.03


def test_shared_variance_ignores_filled_frames():
    """Filled-in frames (variance multiplier 1000 x) do not hide a shared
    change: they are left out of the shared variance and shrunk alone."""
    rng = np.random.default_rng(8)
    n_px, n_t = 200, 2000
    q = np.full((n_px, n_t), 3.0)
    src = np.zeros((n_px, n_t))
    finv = np.ones((n_px, n_t))
    filled = np.zeros(n_t, bool)
    filled[100:160] = True
    finv[:, filled] = 1000.0
    true = 1 + 0.15 * np.sin(np.arange(n_t) / 7.0)
    y = rng.poisson(q * true[None]).astype(np.float64)
    y[:, filled] = 3.0  # filled with a smooth value
    c, rec = shared_time_course(y, q, src, finv)
    assert rec["s2"] > 0
    assert _corr(c[~filled], true[~filled]) > 0.85
    # the filled frames stay near the mean
    assert np.abs(c[filled] - rec["mean"]).max() < 0.01


def test_outside_pixels_ignore_a_source_without_footprint():
    """A source whose footprint is all zero leaves the outside pixels to
    the other sources."""
    sub, _, _ = _synthetic(seed=9)
    params = replace(PARAMS, baseline_mode="gain_shared")
    rows, cols = np.nonzero(sub.sel.T)[::-1]
    h1 = np.exp(-((rows - 6.0) ** 2 + (cols - 6.0) ** 2) / (2 * 1.33**2))
    h = np.column_stack([h1 / h1.sum(), np.zeros(sub.n_pixels)])
    x = np.zeros((2, sub.n_frames))
    b = sub.baseline
    _, info = fit_network_gain(
        b, b, sub.y[0], x, h, sub.finv, params, shared=True
    )
    assert info["n_outside"] > 100


def test_filled_frames_do_not_fake_a_shared_change():
    """With a constant baseline, many filled-in frames leave the residual
    scale unbiased, so no shared change is found."""
    rng = np.random.default_rng(10)
    n_px, n_t = 200, 2000
    q = np.full((n_px, n_t), 3.0)
    finv = np.ones((n_px, n_t))
    filled = rng.random(n_t) < 0.3
    finv[:, filled] = 1000.0
    y = rng.poisson(q).astype(np.float64)
    y[:, filled] = 3.0
    c, rec = shared_time_course(y, q, np.zeros_like(q), finv)
    assert abs(rec["alpha"] - 0.75) < 0.03
    assert np.ptp(c) < 0.01


def test_filled_samples_do_not_count_in_alpha():
    """Pixels filled in at some frames do not bias the residual scale."""
    rng = np.random.default_rng(13)
    n_px, n_t = 200, 2000
    q = np.full((n_px, n_t), 3.0)
    finv = np.ones((n_px, n_t))
    holes = rng.random((n_px, n_t)) < 0.3
    finv[holes] = 1000.0
    y = rng.poisson(q).astype(np.float64)
    y[holes] = 3.0
    c, rec = shared_time_course(y, q, np.zeros_like(q), finv)
    assert abs(rec["alpha"] - 0.75) < 0.03
    assert np.ptp(c) < 0.01


def test_shared_change_found_when_most_frames_are_filled():
    """With 70% of the frames filled in, the shared change is still found
    on the others."""
    rng = np.random.default_rng(12)
    n_px, n_t = 200, 3000
    q = np.full((n_px, n_t), 3.0)
    finv = np.ones((n_px, n_t))
    filled = rng.random(n_t) < 0.7
    finv[:, filled] = 1000.0
    true = 1 + 0.15 * np.sin(np.arange(n_t) / 7.0)
    y = rng.poisson(q * true[None]).astype(np.float64)
    y[:, filled] = 3.0
    c, rec = shared_time_course(y, q, np.zeros_like(q), finv)
    assert rec["s2"] > 0
    assert _corr(c[~filled], true[~filled]) > 0.85


def _paired_noise(rng, q, true):
    """Poisson data whose noise neighbouring pixel pairs share (as on
    resampled frames): each pair's pixels are the mean of two draws."""
    n_px, n_t = q.shape
    draws = rng.poisson(q * true[None]).astype(np.float64)
    pair = draws.reshape(n_px // 2, 2, n_t).mean(1)
    return np.repeat(pair, 2, axis=0)


def test_halves_ignore_noise_shared_by_neighbours():
    """Noise shared by neighbouring pixels is no shared change when the
    halves are apart; the independent-noise estimate mistakes it for one."""
    rng = np.random.default_rng(14)
    n_px, n_t = 200, 2000
    q = np.full((n_px, n_t), 3.0)
    finv = np.ones((n_px, n_t))
    src = np.zeros_like(q)
    y = _paired_noise(rng, q, np.ones(n_t))
    half_a = np.arange(n_px) < 90
    half_b = np.arange(n_px) >= 110
    c1, r1 = shared_time_course(y, q, src, finv)
    c2, r2 = shared_time_course(y, q, src, finv, (half_a, half_b))
    assert r1["s2"] > 0 and np.ptp(c1) > 0.05
    assert r2["s2"] < 0.1 * r2["var_raw"]
    assert np.std(c2) < 0.3 * np.std(c1)
    true = 1 + 0.15 * np.sin(np.arange(n_t) / 7.0)
    y = _paired_noise(rng, q, true)
    c3, r3 = shared_time_course(y, q, src, finv, (half_a, half_b))
    assert r3["s2"] > 0
    assert _corr(c3, true) > 0.85
    small = np.zeros(n_px, bool)
    small[:3] = True
    _, r4 = shared_time_course(y, q, src, finv, (small, half_b))
    assert r4["s2"] == 0


def test_outside_halves_are_apart():
    """The halves split the outside pixels along their longer extent,
    with a gap."""
    sel = np.ones((5, 12), bool)
    outside = np.ones(60, bool)
    a, b = outside_halves(sel, outside)
    rows, cols = np.nonzero(sel.T)[::-1]
    assert cols[a].max() < cols[b].min() - 2
    assert a.sum() > 15 and b.sum() > 15


def test_halves_skip_frames_where_one_half_is_filled():
    """Frames where one half is filled in (and the whole is not) do not
    enter the halves' covariance."""
    rng = np.random.default_rng(15)
    n_px, n_t = 200, 3000
    q = np.full((n_px, n_t), 3.0)
    finv = np.ones((n_px, n_t))
    half_a = np.arange(n_px) < 90
    half_b = np.arange(n_px) >= 110
    holes = rng.random(n_t) < 0.4
    finv[np.ix_(half_a, holes)] = 1000.0
    true = 1 + 0.15 * np.sin(np.arange(n_t) / 7.0)
    y = rng.poisson(q * true[None]).astype(np.float64)
    y[np.ix_(half_a, holes)] = 3.0
    c, rec = shared_time_course(y, q, np.zeros_like(q), finv, (half_a, half_b))
    assert rec["s2"] == pytest.approx(np.var(true), rel=0.25)
    assert _corr(c, true) > 0.85


def test_halves_do_not_shrink_noiseless_factors():
    """Identical noiseless factors in both halves are kept as they are,
    also with few frames."""
    q = np.ones((20, 2))
    y = q * np.array([0.8, 1.2])[None]
    halves = (np.arange(20) < 8, np.arange(20) >= 12)
    for finv in (np.ones_like(q), np.ones_like(q) * np.array([1.0, 4.0])):
        c, rec = shared_time_course(y, q, np.zeros_like(q), finv, halves)
        np.testing.assert_allclose(c, [0.8, 1.2])


def test_halves_use_the_same_frames_for_both_variances():
    """When one half is filled in during the quiet part of a shared
    change, the shared variance does not exceed the factors' variance on
    the same frames, and the change is followed."""
    rng = np.random.default_rng(16)
    n_px, n_t = 200, 3000
    q = np.full((n_px, n_t), 3.0)
    finv = np.ones((n_px, n_t))
    half_a = np.arange(n_px) < 90
    half_b = np.arange(n_px) >= 110
    true = 1 + 0.15 * np.sin(np.arange(n_t) / 7.0)
    quiet = true < 1
    finv[np.ix_(half_a, quiet)] = 1000.0
    y = rng.poisson(q * true[None]).astype(np.float64)
    y[np.ix_(half_a, quiet)] = 3.0
    c, rec = shared_time_course(y, q, np.zeros_like(q), finv, (half_a, half_b))
    assert 0 < rec["s2"] <= rec["var_used"]
    # on these frames the change is half a sine, a smaller variance
    assert rec["s2"] == pytest.approx(np.var(true[~quiet]), rel=0.4)
    assert _corr(c[~quiet], true[~quiet]) > 0.6
