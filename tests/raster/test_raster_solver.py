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
# MATLAB's 3 x 3 support: the outputs are compared with MATLAB's (the
# default is 5 x 5; test_valid_kernel_lets_a_wide_footprint_be_fit sets it)
PARAMS = SolverParams(valid_kernel=3)
# MATLAB's loss for the footprint fit (the default holds its variance)
MATLAB_LOSS = replace(PARAMS, footprint_variance="model")


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


@pytest.fixture(scope="module")
def solved_matlab_loss(fixture):
    """The result with MATLAB's loss for the footprint fit."""
    sub, _ = fixture
    opt = SolverOptions(keep_baseline=True)
    return solve_subproblem(sub, MATLAB_LOSS, opt)


def test_agrees_with_matlab(fixture, solved_matlab_loss):
    """With MATLAB's loss, the converged solver's outputs are close to
    MATLAB's, whose fits stop early: same footprints and least squares,
    similar activity."""
    sub, out = fixture
    r = solved_matlab_loss
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


def test_batches_count_the_footprint_filter(fixture, monkeypatch):
    """The size of a batch counts each member's data and its dense
    pixels x pixels filter arrays."""
    import giant_python.extraction.raster.solver as solver

    sub, _ = fixture
    sizes = []
    monkeypatch.setattr(
        solver,
        "_solve_batch",
        lambda subs, p, o: sizes.append(len(subs)) or subs,
    )
    n_px, n_t = sub.n_pixels, sub.n_frames
    member = (n_px * n_t + 4 * n_px**2) * 8
    solve_subproblems([sub] * 5, PARAMS, SolverOptions(batch_bytes=2 * member))
    assert sizes == [2, 2, 1]
    sizes.clear()
    # the data alone would let three through
    solve_subproblems(
        [sub] * 3, PARAMS, SolverOptions(batch_bytes=3 * n_px * n_t * 8)
    )
    assert sizes == [2, 1]
    sizes.clear()
    solve_subproblems([sub] * 3, PARAMS)  # the CPU's default: one at a time
    assert sizes == [1, 1, 1]


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


def test_data_the_fits_cannot_weigh_are_refused(fixture):
    """NaN samples and variance multipliers that are not positive and
    finite are input errors, not NaN results."""
    sub, _ = fixture
    y = sub.y[:1].copy()
    y[0, 3, 7] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        solve_subproblem(replace(sub, y=y), PARAMS)
    for bad in (0.0, -1.0, np.nan, np.inf):
        finv = sub.finv.copy()
        finv[5, 9] = bad
        with pytest.raises(ValueError, match="variance multiplier"):
            solve_subproblem(replace(sub, y=sub.y[:1], finv=finv), PARAMS)
    empty = replace(sub, y=sub.y[:1], sources_rc=np.zeros((0, 2), int))
    with pytest.raises(ValueError, match="no source"):
        solve_subproblem(empty, PARAMS)
    assert solve_subproblems([], PARAMS) == []


def test_options_are_checked():
    """Options the fits cannot run with are refused when they are made."""
    with pytest.raises(ValueError, match="dtype"):
        SolverOptions(dtype="float16")
    with pytest.raises(ValueError, match="max_iter"):
        SolverOptions(max_iter=-1)
    with pytest.raises(ValueError, match="names no fit"):
        SolverOptions(fit_max_iter={"footprint1": 0})
    with pytest.raises(ValueError, match="whole number"):
        SolverOptions(fit_max_iter={"footprints1": -1})
    with pytest.raises(ValueError, match="batch_bytes"):
        SolverOptions(batch_bytes=-1.0)
    caps = {"activity2": 3, "debias": 0, "activity_ch2": 1, "debias_ch2": 2}
    opt = SolverOptions(fit_max_iter=caps, max_iter=7)
    assert opt.fit_options("activity2")["max_iter"] == 3
    assert opt.fit_options("activity1")["max_iter"] == 7


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


def test_default_footprint_fit_holds_its_variance(solved, solved_matlab_loss):
    """By default the variance of the footprint fit is held at its value
    at the fit's start, so the fit does not gain from raising the model:
    the footprints come out more compact than with MATLAB's loss, and F0,
    which scales with their spread, lower. Both fits converge."""
    assert PARAMS.footprint_variance == "fixed"
    fixed, model = solved, solved_matlab_loss
    assert fixed.info["footprints1"]["converged"]
    assert model.info["footprints1"]["converged"]
    np.testing.assert_allclose(fixed.footprints.sum(0), 1.0, atol=1e-12)
    compact = (fixed.footprints**2).sum(0)
    assert np.all(compact > 1.02 * (model.footprints**2).sum(0))
    assert np.all(fixed.f0[0].mean(1) < model.f0[0].mean(1))
    for j in range(2):
        assert _corr(fixed.footprints[:, j], model.footprints[:, j]) > 0.98
    # what precedes the footprint fit does not depend on the choice
    assert fixed.info["activity1"]["loss"] == pytest.approx(
        model.info["activity1"]["loss"], rel=1e-12
    )
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
        sources_rc=[[0, j] for j in range(len(sources))],
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
    # one result per subproblem, each of the trial's channels and frames
    with pytest.raises(ValueError, match="2 subproblems and 1 results"):
        assemble_results(subs, results[:1], 6, 5, 25.0)
    with pytest.raises(ValueError, match="at least one"):
        assemble_results([], [], 6, 5, 25.0)
    one_channel = _result(2, 2, 6, 2.0, n_ch=1)
    with pytest.raises(ValueError, match="events are"):
        assemble_results(subs, [results[0], one_channel], 6, 5, 25.0)
    short = _result(2, 2, 5, 2.0)
    with pytest.raises(ValueError, match="events are"):
        assemble_results(subs, [results[0], short], 6, 5, 25.0)
    wrong_snr = _result(2, 2, 6, 2.0)
    wrong_snr.snr = np.ones(3)
    with pytest.raises(ValueError, match="one SNR per source"):
        assemble_results(subs, [results[0], wrong_snr], 6, 5, 25.0)


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
    with pytest.raises(ValueError, match="discard has 39 frames"):
        process_result(trial, params, discard[:-1])
    # the trial's own arrays are left as they were
    assert not np.isnan(trial.events).any()


def test_conv_same_is_matlabs():
    """'same' keeps the first argument's length, also below the kernel's."""
    k = np.array([0.0, 0.0, 1.0, 0.5, 0.25])
    x = np.array([[1.0, 0.0, 0.0, 2.0]])
    assert np.array_equal(conv_same(x, k), [[1.0, 0.5, 0.25, 2.0]])
    assert np.array_equal(conv_same(x[:, :2], k), [[1.0, 0.5]])


def _rms_width(h, rows, cols):
    """RMS width per axis of a footprint about its centroid."""
    h = np.clip(h, 0, None) / np.clip(h, 0, None).sum()
    m = np.array([h @ rows, h @ cols])
    return float(np.sqrt((h @ ((rows - m[0]) ** 2 + (cols - m[1]) ** 2)) / 2))


def test_valid_kernel_lets_a_wide_footprint_be_fit():
    """A sigma-1.8 source on its true baseline: with the 3 x 3 support the
    fitted footprint stops near its ceiling (1.56 px), with 5 x 5 it gets
    close to the truth (1.8 px)."""
    rng = np.random.default_rng(3)
    sel = np.ones((13, 13), dtype=bool)
    rows, cols = sel_coords(sel)
    rows, cols = rows.astype(float), cols.astype(float)
    h = np.exp(-((rows - 6) ** 2 + (cols - 6) ** 2) / (2 * 1.8**2))
    h /= h.sum()
    n = 8000
    ev = np.where(rng.random(n) < 0.01, rng.exponential(150.0, n), 0.0)
    x = np.convolve(ev, PARAMS.k, mode="same")
    b = np.full((h.size, n), 2.0)
    y = rng.poisson(b + h[:, None] * x[None]).astype(np.float64)[None]
    sub = Subproblem(
        y=y,
        finv=np.ones(y.shape[1:]),
        sel=sel,
        sources_rc=np.array([[6, 6]]),
        baseline=b,
    )
    true_w = _rms_width(h, rows, cols)
    widths = {}
    for vk in (3, 5):
        p = replace(PARAMS, baseline_mode="fixed", valid_kernel=vk)
        res = solve_subproblem(sub, p)
        widths[vk] = _rms_width(res.footprints[:, 0], rows, cols)
    assert widths[3] < 1.57
    assert abs(widths[5] - true_w) < 0.1
    assert abs(widths[5] - true_w) < abs(widths[3] - true_w) / 2
