"""Tests of the deterministic parts of the subproblem solver: the baseline
(splitFreq, fitB and its helpers), the floor and the noise estimate,
against GIAnT-MATLAB's values."""

import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy import optimize

from giant_python.extraction.raster import solver_baseline as sb
from giant_python.extraction.raster.localization import movmean_nan
from giant_python.extraction.raster.solver_setup import (
    SolverParams,
    conv_operator,
    sel_coords,
)

DATA = Path(__file__).parent / "data"
HELPERS = DATA / "matlab_solver_helpers.h5"
SUBPROBLEM = DATA / "matlab_solver_subproblem.h5"
PARAMS = SolverParams()  # SILo's defaults for SLAP2, as the subproblem's
# parameters of the small fitB and getSurround references
SMALL = SolverParams(
    sigma_px=1.33,
    sel_radius=2,
    tau_full=3,
    tau2_s=0.05,
    analyze_hz=100,
    baseline_window_s=0.6,
    denoise_window_s=0.1,
)


def _close(a, b, tol=1e-12):
    """Equal NaN patterns, and differences below tol of the largest value."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    assert a.shape == b.shape
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.nanmax(np.abs(a - b)) <= tol * np.nanmax(np.abs(b))


def _conv_same(x, k):
    """MATLAB convn(x, k, 'same') along the rows."""
    c = len(k) // 2
    return np.stack([np.convolve(r, k)[c : c + x.shape[1]] for r in x])


def _read_tree(g):
    """An HDF5 group as nested dicts of arrays."""
    return {
        k: _read_tree(v) if isinstance(v, h5py.Group) else v[()]
        for k, v in g.items()
    }


@pytest.fixture(scope="module")
def helpers():
    """MATLAB's values for the helper functions."""
    with h5py.File(HELPERS, "r") as f:
        return _read_tree(f)


@pytest.fixture(scope="module")
def sub():
    """The subproblem fixture: inputs as float64, MATLAB's intermediates."""
    with h5py.File(SUBPROBLEM, "r") as f:
        d = _read_tree(f)
        d["every"] = int(f.attrs["baseline_every"])
    d["y"] = d["in"]["y"].astype(np.float64)
    d["finv"] = d["in"]["finv"].astype(np.float64)
    d["sel"] = d["in"]["sel"].astype(bool)
    return d


@pytest.mark.parametrize("window", [4, 7, 20, 50])
def test_lowess_and_movmean_match_matlab(helpers, window):
    """smoothdata's lowess and movmean, for even, odd and too long
    windows."""
    s = helpers["smooth"]
    _close(sb.smooth_lowess(s["A"], window), s[f"lowess_{window}"])
    _close(movmean_nan(s["A"].T, window).T, s[f"movmean_{window}"])


@pytest.mark.parametrize("window", [7, 20])
def test_lowess_omits_nan_as_matlab(helpers, window):
    """NaN samples are left out of the local fits."""
    s = helpers["smooth"]
    assert np.isnan(s["An"]).sum() == 10
    _close(sb.smooth_lowess(s["An"], window), s[f"lowess_nan_{window}"])


def test_lowess_degenerate_windows():
    """A window of one sample returns the data; a window whose only
    weighted sample is the centre returns it; all-NaN windows give NaN."""
    a = np.array([[1.0, 4.0, 2.0, np.nan, np.nan, np.nan, 7.0]])
    assert np.array_equal(sb.smooth_lowess(a[:, :3], 1), a[:, :3])
    out = sb.smooth_lowess(a, 3)
    assert np.array_equal(out[0, :3], a[0, :3])
    assert np.isnan(out[0, 4])
    # the NaN next to a value takes it (zero weight, so the window's mean)
    assert out[0, 3] == 2.0 and out[0, 5] == 7.0


@pytest.mark.parametrize(
    "name, order, n, pad",
    [
        ("zeros_3_20", 3, 20, "zeros"),
        ("zeros_2_5", 2, 5, "zeros"),
        ("zeros_1_4", 1, 4, "zeros"),
        ("sym_5_46", 5, 46, "symmetric"),
        ("sym_3_7", 3, 7, "symmetric"),
        ("sym_2_6", 2, 6, "symmetric"),
        ("med_9", 5, 9, "symmetric"),
    ],
)
def test_order_filters_match_matlab(helpers, name, order, n, pad):
    """ordfilt2 and medfilt2 along rows, with both paddings; the window of
    46 is longer than the 41 samples."""
    o = helpers["ord"]
    assert np.array_equal(sb.ordfilt_rows(o["a"], order, n, pad), o[name])


def test_split_freq_matches_matlab(helpers):
    """splitFreq, also for a trial too short to interpolate."""
    s = helpers["split"]
    lp, hp = sb.split_freq(s["A"], 10, 6)
    _close(lp, s["LP"])
    assert np.array_equal(hp, s["A"] - lp)
    short, _ = sb.split_freq(s["A"][:, :15], 10, 6)
    _close(short, s["LP_short"])
    _close(sb.split_freq(s["D"], 10, 6)[0], s["LP_nan"])


def test_initial_baseline_takes_min_where_nan(monkeypatch):
    """Where splitFreq's low-pass is NaN the baseline is min_baseline, as
    MATLAB's max(minBaseline, NaN) gives; elsewhere it is bounded below."""
    lp = np.array([[np.nan, 0.5, -1.0], [2.0, np.nan, 0.001]])
    monkeypatch.setattr(sb, "split_freq", lambda *a: (lp, None))
    b = sb.initial_baseline(np.zeros((2, 3)), PARAMS)
    assert np.array_equal(b, [[0.01, 0.5, 0.01], [2.0, 0.01, 0.01]])


def test_compute_floor_matches_matlab(helpers):
    """computeFloor, for integer and fractional windows."""
    f = helpers["floor"]
    _close(sb.compute_floor(f["X"], 7, 45.5), f["out"])
    _close(sb.compute_floor(f["X"], 6, 40), f["out_int"])


def test_bspline_basis_matches_matlab(helpers):
    """bsplineBasis: whole knot intervals tile the trial; one interval when
    the spacing exceeds it. The products are those of the basis."""
    s = helpers["spline"]
    b = sb.bspline_basis(101, 17.3)
    _close(b.toarray(), s["B"], 1e-14)
    _close(sb.bspline_basis(40, 100).toarray(), s["B1"], 1e-14)
    assert b.shape == (101, 9) and s["B1"].shape == (40, 4)
    np.testing.assert_allclose(b.sum(1), 1.0, atol=1e-14)
    dense = b.toarray()
    prod = sb._bspline_products(101, 17.3).toarray().reshape(101, 9, 9)
    np.testing.assert_allclose(
        prod, dense[:, :, None] * dense[:, None, :], atol=1e-15
    )


def test_curvature_penalty():
    """Squared second differences per knot interval: constants and lines
    cost nothing."""
    pc = sb.curvature_penalty(7)
    c = np.arange(7.0)
    assert np.allclose(pc @ np.ones(7), 0) and np.allclose(pc @ c, 0)
    quad = c**2
    assert quad @ pc @ quad == pytest.approx(5 * 2.0**2 / 4)


def test_surround_matches_matlab(helpers):
    """getSurround on a map with a hole, dense and sparse operators."""
    sel = helpers["foot"]["selPix"].astype(bool)
    s = helpers["surround"]
    _close(SMALL.bfilter, s["Bfilter"].transpose(2, 0, 1))
    ops = sb.surround_operators(sel, SMALL)
    assert all(isinstance(op, np.ndarray) for op in ops)  # small: dense
    ref = s["out"].transpose(2, 0, 1)
    _close(sb.get_surround(s["HP"], ops), ref)
    sparse_ops = [conv_operator(sel, f) for f in SMALL.bfilter]
    _close(sb.get_surround(s["HP"], sparse_ops), ref)


def test_fit_baseline_matches_matlab(helpers):
    """fitB on a small synthetic problem. MATLAB's quadprog stops at its
    tolerance, so the baselines agree to about 1e-8; Python's programs are
    solved to machine precision."""
    sel = helpers["foot"]["selPix"].astype(bool)
    f = helpers["fitB"]
    y = f["Finv"] * f["counts"]
    b0 = np.full(y.shape, 1.4)
    b, info = sb.fit_baseline(
        b0, y, f["X"], f["H"], f["Finv"], sel, SMALL, return_info=True
    )
    _close(b, f["B"], 1e-7)
    assert info["converged"].all() and info["kkt"].max() < 1e-9
    assert info["coef"].shape == (y.shape[0], 2 + 3 + 13)
    # the sources' coefficients are not negative
    assert info["coef"][:, :2].min() >= 0


def _random_qp(rng, m, n, rank=None):
    """A stack of random bounded quadratic programs."""
    a = rng.standard_normal((m, rank or n + 3, n))
    g = np.einsum("mki,mkj->mij", a, a)
    c = rng.standard_normal((m, n)) * 3
    lb = -rng.random((m, n))
    ub = rng.random((m, n))
    return g, c, lb, ub


def test_box_qp_matches_scipy():
    """The batched solver finds the minimum that a bounded least-squares
    solver finds, with many bounds active."""
    rng = np.random.default_rng(3)
    g, c, lb, ub = _random_qp(rng, 40, 6)
    x, info = sb.solve_box_qp(g, c, lb, ub)
    assert info["converged"].all() and info["kkt"].max() < 1e-9
    assert (x >= lb).all() and (x <= ub).all()
    assert ((x == lb) | (x == ub)).mean() > 0.2
    for i in range(40):
        # min x'Gx/2 - c'x  ==  min |R x - R^-T c|^2 with G = R'R
        r = np.linalg.cholesky(g[i]).T
        rhs = np.linalg.solve(r.T, c[i])
        ref = optimize.lsq_linear(
            r, rhs, bounds=(lb[i], ub[i]), method="bvls", tol=1e-13
        ).x
        np.testing.assert_allclose(x[i], ref, atol=1e-8)


def test_box_qp_zero_curvature_follows_the_linear_term():
    """A variable without curvature goes to the bound that its linear
    term favours."""
    g = np.zeros((3, 1, 1))
    c = np.array([[1.0], [-1.0], [0.0]])
    lb, ub = np.full((3, 1), -3.0), np.full((3, 1), 2.0)
    x, info = sb.solve_box_qp(g, c, lb, ub)
    assert x.ravel().tolist() == [2.0, -3.0, 0.0]
    assert info["converged"].all()
    # beside variables with curvature, which do not depend on it
    g = np.array([[[2.0, 0.0], [0.0, 0.0]]])
    x, _ = sb.solve_box_qp(
        g, np.array([[1.0, 5.0]]), np.full((1, 2), -4.0), np.full((1, 2), 3.0)
    )
    assert x.ravel().tolist() == [0.5, 3.0]


def test_box_qp_fixed_variables():
    """Bounds that coincide fix a variable, in the Newton iterations and in
    the fallback; the others are solved around it."""
    one = np.ones((1, 1))
    for max_iter in (100, 0):
        x, _ = sb.solve_box_qp(one[None], 0 * one, one, one, max_iter=max_iter)
        assert x.tolist() == [[1.0]]
    rng = np.random.default_rng(9)
    g, c, lb, ub = _random_qp(rng, 8, 5)
    lb[:, 1] = ub[:, 1] = 0.3
    lb[:, 4] = ub[:, 4] = -0.2
    x, info = sb.solve_box_qp(g, c, lb, ub)
    assert info["converged"].all()
    assert np.all(x[:, 1] == 0.3) and np.all(x[:, 4] == -0.2)
    fallback, info = sb.solve_box_qp(g, c, lb, ub, max_iter=0)
    assert not info["converged"].any()
    np.testing.assert_allclose(fallback, x, atol=1e-7)
    # the free variables solve the program with the fixed ones moved into
    # its linear term
    keep = [0, 2, 3]
    shift = g[:, keep][:, :, [1, 4]] @ np.array([0.3, -0.2])
    ref, _ = sb.solve_box_qp(
        g[:, keep][:, :, keep], c[:, keep] - shift, lb[:, keep], ub[:, keep]
    )
    np.testing.assert_allclose(x[:, keep], ref, atol=1e-9)


def test_box_qp_fallback_keeps_the_objective():
    """The fallback solves the program as given: a singular G whose linear
    term acts along the null direction ends on the bounds, without a
    warning, as the Newton iterations do."""
    g = np.array([[[1.0, 1.0], [1.0, 1.0]]])
    lb, ub = np.full((1, 2), -2.0), np.full((1, 2), 2.0)
    for c in ([[1e-16, -1e-16]], [[0.5, -0.5]]):
        c = np.array(c)
        for max_iter in (100, 0):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                x, info = sb.solve_box_qp(g, c, lb, ub, max_iter=max_iter)
            assert x.tolist() == [[2.0, -2.0]], (c, max_iter)
            assert info["kkt"].max() < 1e-9
    # no linear term at all: the quadratic alone, pushed to the bounds
    ones = np.ones((1, 4, 4))
    zero = np.zeros((1, 4))
    x, info = sb.solve_box_qp(
        ones, zero, np.full((1, 4), 1.0), np.full((1, 4), 2.0), max_iter=0
    )
    assert x.tolist() == [[1.0] * 4] and np.isfinite(info["kkt"]).all()
    # a positive semidefinite program with its minimum inside the bounds
    c = np.array([[1.0, 1.0]])
    x, info = sb.solve_box_qp(g, c, lb, ub, max_iter=0)
    assert x.sum() == pytest.approx(1.0, abs=1e-9) and info["kkt"].max() < 1e-9


def test_box_qp_zero_rows_and_fallback():
    """A variable with a zero row is left at its bound nearest zero; a
    problem the Newton iterations do not finish goes to the fallback."""
    rng = np.random.default_rng(4)
    g, c, lb, ub = _random_qp(rng, 5, 4)
    g[:, 2, :] = g[:, :, 2] = 0
    c[:, 2] = 0
    lb[:, 2], ub[:, 2] = 0.25, 3.0
    x, info = sb.solve_box_qp(g, c, lb, ub)
    assert np.all(x[:, 2] == 0.25) and info["converged"].all()
    ref, _ = sb.solve_box_qp(
        np.delete(np.delete(g, 2, 1), 2, 2),
        np.delete(c, 2, 1),
        np.delete(lb, 2, 1),
        np.delete(ub, 2, 1),
    )
    np.testing.assert_allclose(np.delete(x, 2, 1), ref, atol=1e-10)
    # no iterations: every problem is solved by the fallback
    g, c, lb, ub = _random_qp(rng, 6, 5)
    good, _ = sb.solve_box_qp(g, c, lb, ub)
    x, info = sb.solve_box_qp(g, c, lb, ub, max_iter=0)
    assert not info["converged"].any()
    np.testing.assert_allclose(x, good, atol=1e-7)


def test_subproblem_start_matches_matlab(sub):
    """Starting baseline, least squares and events of a real subproblem."""
    y, every = sub["y"][0], sub["every"]
    b = sb.initial_baseline(y, PARAMS)
    _close(b[:, ::every], sub["init"]["B"])
    ls = np.linalg.lstsq(sub["init"]["H"], y - b, rcond=None)[0]
    _close(ls, sub["init"]["dFls"])
    prev = np.concatenate([np.zeros((2, 1)), ls[:, :-1]], axis=1)
    k_peak = PARAMS.k[PARAMS.k.size // 2]
    _close(np.fmax(0, ls - prev) / k_peak, sub["init"]["S"])
    b2 = sb.initial_baseline(sub["y"][1], PARAMS)
    _close(b2[:, ::every], sub["ch2"]["B"])


def test_subproblem_noise_and_floor_match_matlab(sub):
    """Noise, SNR and floor from MATLAB's first activity fit."""
    y, it1 = sub["y"][0], sub["it1"]
    b = sb.initial_baseline(y, PARAMS)
    x = _conv_same(it1["S_out"], PARAMS.k)
    var, noise, snr = sb.activity_noise(
        y, sub["finv"], b, sub["init"]["H"], x, PARAMS.tau_frames
    )
    _close(var, it1["residVar"])
    _close(noise, it1["Xnoise"])
    _close(snr, it1["Xsnr"])
    floor = sb.compute_floor(x, PARAMS.denoise_window, PARAMS.baseline_window)
    _close(floor, it1["Xfloor"])
    _close(np.fmax(0, x - floor - noise[:, None]), it1["X_thr"])


def test_subproblem_baseline_fit_matches_matlab(sub):
    """fitB on a real subproblem (its trial is shorter than the lowess
    window of the high-pass split), to quadprog's tolerance."""
    y, it1 = sub["y"][0], sub["it1"]
    b0 = sb.initial_baseline(y, PARAMS)
    b, info = sb.fit_baseline(
        b0,
        y,
        it1["X_thr"],
        sub["init"]["H"],
        sub["finv"],
        sub["sel"],
        PARAMS,
        return_info=True,
    )
    _close(b[:, :: sub["every"]], it1["B_out"], 1e-5)
    assert info["converged"].all() and info["kkt"].max() < 1e-9


def test_subproblem_footprint_update_matches_matlab(sub):
    """H from MATLAB's fitted Hs through the footprint filter, and the
    normalization."""
    it1 = sub["it1"]
    h = conv_operator(sub["sel"], PARAMS.hfilter) @ it1["Hs_out"]
    norm = h.sum(0)
    _close(norm, it1["normFac"])
    _close(h / norm, it1["H"])
    _close(it1["Hs_out"] / norm, it1["Hs"])
    _close(it1["S_out"] * norm[:, None], it1["S"])
    rows, _ = sel_coords(sub["sel"])
    assert h.shape == (rows.size, 2)


def test_subproblem_outputs_match_matlab(sub):
    """SNR, least squares and F0 from MATLAB's final activity."""
    y, out, it1 = sub["y"], sub["out"], sub["it1"]
    h = out["H"]
    b1 = sb.fit_baseline(
        sb.initial_baseline(y[0], PARAMS),
        y[0],
        it1["X_thr"],
        sub["init"]["H"],
        sub["finv"],
        sub["sel"],
        PARAMS,
    )
    x = _conv_same(out["S"][0], PARAMS.k)
    _, _, snr = sb.activity_noise(
        y[0], sub["finv"], b1, h, x, PARAMS.tau_frames
    )
    _close(snr, out["SNR"], 1e-5)
    _close(np.linalg.lstsq(h, y[0] - b1, rcond=None)[0], out["LS"][0], 1e-5)
    _close(sb.source_f0(h, b1), out["F0"][0], 1e-5)
    b2 = sb.initial_baseline(y[1], PARAMS)
    _close(np.linalg.lstsq(h, y[1] - b2, rcond=None)[0], out["LS"][1])
    _close(sb.source_f0(h, b2), out["F0"][1])


def test_interp1_extrapolates():
    """Linear interpolation of rows with linear extrapolation."""
    x = np.array([1.0, 2.0, 4.0])
    y = np.array([[0.0, 1.0, 5.0]])
    out = sb.interp1_extrap(x, y, np.array([0.0, 1.5, 3.0, 5.0]))
    np.testing.assert_allclose(out, [[-1.0, 0.5, 3.0, 7.0]])
