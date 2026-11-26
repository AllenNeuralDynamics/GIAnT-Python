"""Tests of the subproblem solver end to end, of the assembly of a trial's
results and of the trial's activity traces."""

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from giant_python.extraction.raster.solver import (
    SolverOptions,
    SubproblemResult,
    assemble_results,
    conv_same,
    estimate_photon_scale,
    process_result,
    solve_subproblem,
    solve_subproblems,
)
from giant_python.extraction.raster.solver_setup import (
    SolverParams,
    Subproblem,
    conv_operator,
    initial_footprints,
    sel_coords,
)

DATA = Path(__file__).parent / "data"
SUBPROBLEM = DATA / "matlab_solver_subproblem.h5"
HELPERS = DATA / "matlab_solver_helpers.h5"
PARAMS = SolverParams()


def _corr(a, b):
    """Pearson correlation of two traces."""
    return float(np.corrcoef(np.ravel(a), np.ravel(b))[0, 1])


@pytest.fixture(scope="module")
def fixture():
    """A real two-source subproblem and MATLAB's outputs for it."""
    with h5py.File(SUBPROBLEM, "r") as f:
        sub = Subproblem(
            y=f["in/y"][()].astype(np.float64),
            finv=f["in/finv"][()].astype(np.float64),
            sel=f["in/sel"][()].astype(bool),
            sources_rc=f["in/sources_rc"][()],
        )
        out = {k: v[()] for k, v in f["out"].items()}
    return sub, out


@pytest.fixture(scope="module")
def solved(fixture):
    """The solver's result on the fixture, with its baseline."""
    sub, _ = fixture
    return solve_subproblem(sub, PARAMS, SolverOptions(keep_baseline=True))


def test_agrees_with_matlab(fixture, solved):
    """The converged solver's outputs are close to MATLAB's, whose fits
    stop early: same footprints and least squares, similar activity."""
    sub, out = fixture
    r = solved
    assert r.footprints.shape == (sub.n_pixels, 2)
    assert r.events.shape == r.ls.shape == r.f0.shape == (2, 2, 1000)
    assert r.baseline.shape == (2, sub.n_pixels, 1000)
    np.testing.assert_allclose(r.footprints.sum(0), 1.0, atol=1e-12)
    assert r.events.min() >= 0
    for j in range(2):
        hp, hm = r.footprints[:, j], out["H"][:, j]
        cos = hp @ hm / np.sqrt((hp @ hp) * (hm @ hm))
        assert cos > 0.97
        assert _corr(r.ls[0, j], out["LS"][0, j]) > 0.97
        x = conv_same(r.events[0, j], PARAMS.k)
        xm = conv_same(out["S"][0, j], PARAMS.k)
        assert _corr(x, xm) > 0.9
        assert 0.8 < x.sum() / xm.sum() < 1.25
        ratio = r.f0[:, j].mean(1) / out["F0"][:, j].mean(1)
        assert np.all(np.abs(ratio - 1) < 0.1)
    assert np.all(np.isfinite(r.snr)) and np.all(r.snr > 0)
    assert np.all(np.abs(r.snr / out["SNR"] - 1) < 0.3)
    # channel 2's least squares do not depend on the activity fits
    for j in range(2):
        assert _corr(r.ls[1, j], out["LS"][1, j]) > 0.97


def test_fits_converge_and_are_recorded(solved):
    """Every fit ends on its tolerance, and its record is kept."""
    fits = {k: v for k, v in solved.info.items() if isinstance(v, dict)}
    assert set(fits) == {
        "activity1",
        "footprints1",
        "activity2",
        "debias",
        "activity_ch2",
        "debias_ch2",
    }
    for name, fit in fits.items():
        assert fit["converged"], name
        assert fit["iterations"] <= 20 and np.isfinite(fit["loss"])
    assert solved.info["baseline1_seconds"] > 0


def test_events_below_the_floor_are_dropped(solved):
    """The debias refit keeps the events of 0.1 or more only, and does not
    move the others' places."""
    ev = solved.events
    assert not np.any((ev > 0) & (ev < 1e-3))
    assert 0.01 < (ev[0] > 0).mean() < 0.5


def test_batch_matches_single_solves(fixture, solved):
    """Subproblems of different sizes solved together get the results they
    get alone, to the fits' tolerance."""
    sub, _ = fixture
    # a smaller subproblem: the first source and the pixels near it
    rows, cols = sel_coords(sub.sel)
    r0, c0 = sub.sources_rc[0]
    keep = np.hypot(rows - r0, cols - c0) <= 4.2
    sel = np.zeros_like(sub.sel)
    sel[rows[keep], cols[keep]] = True
    small = Subproblem(
        y=sub.y[:, keep], finv=sub.finv[keep], sel=sel, sources_rc=[[r0, c0]]
    )
    alone = solve_subproblem(small, PARAMS)
    both = solve_subproblems(
        [small, sub], PARAMS, SolverOptions(batch_bytes=1e9)
    )
    assert both[0].footprints.shape == (int(keep.sum()), 1)
    for got, ref in ((both[0], alone), (both[1], solved)):
        np.testing.assert_allclose(got.footprints, ref.footprints, atol=1e-6)
        np.testing.assert_allclose(got.snr, ref.snr, rtol=1e-4)
        np.testing.assert_allclose(got.f0, ref.f0, rtol=1e-5)
        scale = ref.events.max()
        np.testing.assert_allclose(got.events, ref.events, atol=1e-3 * scale)
        assert got.baseline is None


def test_trials_of_different_length_are_grouped(fixture, solved):
    """Subproblems with other numbers of frames or channels are solved in
    batches of their own, and returned in the order given."""
    sub, _ = fixture
    short = replace(
        sub, y=sub.y[:1, :, :600].copy(), finv=sub.finv[:, :600].copy()
    )
    out = solve_subproblems([short, sub], PARAMS, SolverOptions(batch_bytes=1))
    assert out[0].events.shape == (1, 2, 600)
    assert out[1].events.shape == (2, 2, 1000)
    np.testing.assert_allclose(out[1].footprints, solved.footprints, atol=1e-6)


def test_more_than_two_channels_are_refused(fixture):
    """The solver fits the activity channel and one more, as MATLAB's; a
    third channel is refused, not dropped."""
    sub, _ = fixture
    three = replace(sub, y=np.concatenate([sub.y, sub.y[:1]]))
    with pytest.raises(ValueError, match="3 channels"):
        solve_subproblem(three, PARAMS)


def test_float32_fits_are_close(fixture, solved):
    """Fits in float32 give nearly the float64 result."""
    sub, _ = fixture
    r = solve_subproblem(sub, PARAMS, SolverOptions(dtype="float32"))
    for j in range(2):
        assert _corr(r.footprints[:, j], solved.footprints[:, j]) > 0.999
        x = conv_same(r.events[0, j], PARAMS.k)
        ref = conv_same(solved.events[0, j], PARAMS.k)
        assert _corr(x, ref) > 0.995
    np.testing.assert_allclose(r.f0, solved.f0, rtol=1e-2)


def test_iteration_cap_is_reported(fixture, solved):
    """With one Newton iteration per fit, the fits are marked as not
    converged and reach higher losses."""
    sub, _ = fixture
    r = solve_subproblem(sub, PARAMS, SolverOptions(max_iter=1))
    assert not r.info["activity1"]["converged"]
    assert r.info["activity1"]["iterations"] == 1
    assert r.info["activity1"]["loss"] > solved.info["activity1"]["loss"]


def test_single_fits_can_be_capped(fixture, solved):
    """A cap on one fit leaves the others to converge; a cap of 0 on the
    footprint fit keeps the starting footprints' shape."""
    sub, _ = fixture
    one = replace(sub, y=sub.y[:1])
    opt = SolverOptions(fit_max_iter={"footprints1": 0})
    r = solve_subproblem(one, PARAMS, opt)
    assert r.info["footprints1"]["iterations"] == 0
    assert not r.info["footprints1"]["converged"]
    assert (
        r.info["activity1"]["converged"] and r.info["activity2"]["converged"]
    )
    assert r.info["activity1"]["loss"] == pytest.approx(
        solved.info["activity1"]["loss"], rel=1e-9
    )
    # the starting Hs through the footprint filter, normalized
    start = initial_footprints(sub.sel, sub.sources_rc, PARAMS.sigma_px)[0]
    h = conv_operator(sub.sel, PARAMS.hfilter) @ start
    np.testing.assert_allclose(r.footprints, h / h.sum(0), atol=1e-12)
    assert np.abs(r.footprints - solved.footprints).max() > 1e-3


def test_fixed_variance_footprint_fit_is_more_compact(fixture, solved):
    """With the variance held at its value at the fit's start, the footprint
    fit no longer gains from raising the model, and the footprints come
    out more compact than with MATLAB's loss; the fit still converges."""
    sub, _ = fixture
    one = replace(sub, y=sub.y[:1])
    r = solve_subproblem(one, replace(PARAMS, footprint_variance="fixed"))
    assert r.info["footprints1"]["converged"]
    np.testing.assert_allclose(r.footprints.sum(0), 1.0, atol=1e-12)
    compact = (r.footprints**2).sum(0)
    assert np.all(compact > 1.02 * (solved.footprints**2).sum(0))
    for j in range(2):
        assert _corr(r.footprints[:, j], solved.footprints[:, j]) > 0.98
    with pytest.raises(ValueError, match="footprint_variance"):
        replace(PARAMS, footprint_variance="other")


def test_debias_variance_model_gives_larger_events(fixture, solved):
    """The model-variance debias refit rewards larger activity (MATLAB's
    behaviour before the fixed variance); the rest is unchanged."""
    sub, _ = fixture
    one = replace(sub, y=sub.y[:1])
    r = solve_subproblem(one, replace(PARAMS, debias_variance="model"))
    assert r.events.shape == (1, 2, 1000)
    np.testing.assert_allclose(r.footprints, solved.footprints, atol=1e-6)
    np.testing.assert_allclose(r.ls[0], solved.ls[0], atol=1e-6)
    assert r.events.sum() > 1.1 * solved.events[0].sum()
    # both refit the events that the L1 fits found
    same = (r.events[0] > 0) == (solved.events[0] > 0)
    assert same.mean() > 0.99


def test_one_outer_iteration(fixture):
    """With nmf_iter 1 the footprints stay at their start and no baseline
    or footprint fit runs."""
    sub, _ = fixture
    one = replace(sub, y=sub.y[:1])
    r = solve_subproblem(one, replace(PARAMS, nmf_iter=1))
    assert "footprints1" not in r.info and "activity1" in r.info
    assert r.events.shape == (1, 2, 1000)


def test_photon_scale_matches_matlab():
    """extractTrial's photon scale estimate."""
    with h5py.File(HELPERS, "r") as f:
        finv = f["photon/Finv"][()]
        y = 3 * f["photon/counts"][()] * np.sqrt(finv)
        ref = float(f["photon/scale"][()])
    assert estimate_photon_scale(y, finv) == pytest.approx(ref, rel=1e-12)


def _result(n_px, n_src, n_t, value, n_ch=2):
    """A subproblem result filled with a constant."""
    return SubproblemResult(
        footprints=np.full((n_px, n_src), value),
        events=np.full((n_ch, n_src, n_t), value),
        ls=np.full((n_ch, n_src, n_t), 2 * value),
        f0=np.full((n_ch, n_src, n_t), 3 * value),
        snr=np.full(n_src, 10 * value),
    )


def _placed(pixels, sources, n_t):
    """A subproblem that only says where its results go."""
    n_px = len(pixels)
    return Subproblem(
        y=np.zeros((2, n_px, n_t), np.float32),
        finv=np.ones((n_px, n_t), np.float32),
        sel=np.ones((1, n_px), dtype=bool),
        sources_rc=[[0, 0]] * len(sources),
        pixels=np.array(pixels),
        sources=np.array(sources),
    )


def test_assemble_results():
    """Results go to their pixels and sources; events, least squares and
    F0 return to movie units; what no subproblem covers is NaN."""
    subs = [_placed([0, 1, 4], [2], 6), _placed([2, 4], [0, 3], 6)]
    results = [_result(3, 1, 6, 1.0), _result(2, 2, 6, 2.0)]
    t = assemble_results(subs, results, 6, 5, photon_scale=25.0)
    assert t.footprints.dtype == np.float32
    assert t.footprints.shape == (6, 5)
    assert np.all(t.footprints[[0, 1, 4], 2] == 1.0)
    assert np.all(t.footprints[[2, 4]][:, [0, 3]] == 2.0)
    assert np.isnan(t.footprints[0, 0]) and np.isnan(t.footprints[5]).all()
    assert t.events.shape == t.ls.shape == t.f0.shape == (2, 5, 6)
    assert np.all(t.events[:, 2] == 25.0) and np.all(t.events[:, 3] == 50.0)
    assert np.all(t.ls[:, 0] == 100.0) and np.all(t.f0[:, 2] == 75.0)
    assert np.isnan(t.events[:, [1, 4]]).all()
    assert np.array_equal(t.snr[[0, 2, 3]], [20.0, 10.0, 20.0])
    assert np.isnan(t.snr[[1, 4]]).all()
    with pytest.raises(ValueError, match="pixels and sources"):
        assemble_results(
            [replace(subs[0], pixels=None)], results[:1], 6, 5, 25.0
        )
    # indices that would put results in the wrong place are refused
    for bad, message in (
        (dict(pixels=np.array([0, 1, 6])), "pixels indices must lie"),
        (dict(sources=np.array([5])), "sources indices must lie"),
        (dict(pixels=np.array([0, 1])), "3 pixels and 2 indices for"),
        (dict(pixels=np.array([0, 1, 1])), "pixels indices repeat"),
    ):
        with pytest.raises(ValueError, match=message):
            assemble_results(
                [replace(subs[0], **bad)], results[:1], 6, 5, 25.0
            )
    with pytest.raises(ValueError, match="negative index"):
        replace(subs[0], pixels=np.array([0, -1, 4]))


def test_process_result():
    """The denoised activity is the events convolved with each channel's
    kernel; discarded frames are NaN in the traces, as MATLAB leaves them
    (channel 2's events keep their values)."""
    n_t = 40
    rng = np.random.default_rng(0)
    subs = [_placed([0, 1], [0, 1], n_t)]
    res = _result(2, 2, n_t, 1.0)
    res.events = (rng.random((2, 2, n_t)) < 0.2) * rng.random((2, 2, n_t))
    trial = assemble_results(subs, [res], 2, 2, photon_scale=2.0)
    discard = np.zeros(n_t, dtype=bool)
    discard[[3, 4, 30]] = True
    params = SolverParams(tau_s=0.01, tau2_s=0.02)
    act = process_result(trial, params, discard)
    keep = ~discard
    np.testing.assert_allclose(
        act.denoised[0][:, keep],
        conv_same(trial.events[0], params.k)[:, keep],
    )
    np.testing.assert_allclose(
        act.denoised[1][:, keep],
        conv_same(trial.events[1], params.k2)[:, keep],
    )
    assert np.isnan(act.denoised[:, :, discard]).all()
    assert np.isnan(act.ls[:, :, discard]).all()
    assert np.isnan(act.events[0][:, discard]).all()
    assert np.array_equal(act.events[1], trial.events[1])
    assert np.array_equal(act.events[0][:, keep], trial.events[0][:, keep])
    assert act.f0 is trial.f0 and act.snr is trial.snr
    # the trial's own arrays are left as they were
    assert not np.isnan(trial.events).any()


def test_conv_same_is_matlabs():
    """'same' keeps the first argument's length, also below the kernel's."""
    k = np.array([0.0, 0.0, 1.0, 0.5, 0.25])
    x = np.array([[1.0, 0.0, 0.0, 2.0]])
    assert np.array_equal(conv_same(x, k), [[1.0, 0.5, 0.25, 2.0]])
    assert np.array_equal(conv_same(x[:, :2], k), [[1.0, 0.5]])
