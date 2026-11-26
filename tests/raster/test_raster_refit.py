"""Tests of the ground-truth refit test of the subproblem solver."""

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from giant_python.extraction.raster.refit_test import (
    RefitTruth,
    blur_error,
    by_size,
    draw_refit_data,
    drift_error,
    edge_error,
    find_events,
    refit_test,
    scale_error,
    score_refit,
    shared_component,
    summarize,
)
from giant_python.extraction.raster.solver import (
    SolverOptions,
    solve_subproblem,
)
from giant_python.extraction.raster.solver_baseline import source_f0
from giant_python.extraction.raster.solver_setup import (
    SolverParams,
    Subproblem,
)

SUBPROBLEM = Path(__file__).parent / "data" / "matlab_solver_subproblem.h5"
PARAMS = SolverParams()


def test_find_events():
    """Peaks above 2% of the largest, at least the lag after the previous
    peak (kept or not), away from the trial's ends."""
    x = np.zeros(60)
    x[[2, 10, 14, 17, 30, 31, 44, 55]] = [5, 4, 3, 6, 0.05, 2, 1, 9]
    # frame 2 is within 5 of the start; 14 and 17 follow a peak within the
    # lag of 5; 30 is below 2% of 9, and not a maximum next to 31; 55 is
    # within lag + 3 of the end
    assert find_events(x, 5).tolist() == [10, 31, 44]
    assert find_events(x, 3).tolist() == [10, 14, 17, 31, 44]
    assert find_events(np.zeros(30), 5).size == 0
    # a plateau counts once, at its last frame
    y = np.zeros(40)
    y[[20, 21]] = 3.0
    assert find_events(y, 5).tolist() == [21]


def _truth(x, h=None):
    """A truth with the given activity, flat baseline and F0."""
    n_src, n_t = x.shape
    if h is None:
        h = np.array([[0.5], [0.3], [0.2]])[:, :n_src]
    return RefitTruth(
        h=h,
        x=x,
        b=np.full((h.shape[0], n_t), 2.0),
        f0=np.full((n_src, n_t), 4.0),
        z=np.zeros((h.shape[0], n_t)),
    )


def test_score_refit_of_a_perfect_and_a_scaled_refit():
    """A refit equal to the truth recovers everything; a refit at half the
    size and with one event missing is scored as such."""
    k = np.exp(-np.arange(12) / 4.8)
    x = np.zeros((1, 120))
    for t, a in ((20, 2.0), (60, 8.0), (95, 1.0)):
        x[0, t : t + 12] += a * k
    truth = _truth(x)
    f0, ls = truth.f0, x.copy()
    src, ev = score_refit(truth, x, f0, ls, 4.8)
    assert src[0]["recovered"] == pytest.approx(1.0)
    assert src[0]["slope"] == pytest.approx(1.0)
    assert src[0]["f0_ratio"] == src[0]["f0_ratio_active"] == 1.0
    assert src[0]["baseline"] == pytest.approx(2.0)
    assert src[0]["peak_h"] == 0.5
    assert [e["found"] for e in ev] == [True] * 3
    assert [e["size"] for e in ev] == pytest.approx([1.0, 4.0, 0.5])
    for e in ev:
        assert e["refit_mass"] == pytest.approx(e["true_mass"])

    half = 0.5 * x
    half[0, 90:] = 0  # the last event is missed
    src, ev = score_refit(truth, half, 1.1 * f0, 2 * ls, 4.8)
    assert src[0]["f0_ratio"] == pytest.approx(1.1)
    assert src[0]["ls_recovered"] == pytest.approx(2.0)
    assert [e["found"] for e in ev] == [True, True, False]
    assert ev[0]["refit_mass"] == pytest.approx(0.5 * ev[0]["true_mass"])
    rows = by_size(ev)
    assert [(r["size_from"], r["n"]) for r in rows] == [
        (0.5, 1),
        (1.0, 1),
        (4.0, 1),
    ]
    assert rows[0]["share_found"] == 0 and np.isnan(rows[0]["recovered_found"])
    assert rows[1]["recovered_found"] == pytest.approx(0.5)
    summary = summarize(src, ev)
    assert summary["n_sources"] == 1 and summary["n_events"] == 3
    assert 0.3 < summary["pooled"] < 0.5


def test_score_refit_of_a_silent_source():
    """A source without true activity has no events and NaN recovery."""
    truth = _truth(np.zeros((1, 50)))
    src, ev = score_refit(truth, np.zeros((1, 50)), truth.f0, truth.x, 4.8)
    assert ev == [] and np.isnan(src[0]["recovered"])
    assert np.isnan(src[0]["f0_ratio_active"])
    assert np.isnan(summarize(src, ev)["pooled"])


def test_refit_test_on_a_real_subproblem():
    """The refit of data redrawn from a real subproblem's fit (a dim one,
    5 s long, most events below one unit) recovers the events it finds;
    the same seed gives the same data."""
    with h5py.File(SUBPROBLEM, "r") as f:
        sub = Subproblem(
            y=f["in/y"][()].astype(np.float64),
            finv=f["in/finv"][()].astype(np.float64),
            sel=f["in/sel"][()].astype(bool),
            sources_rc=f["in/sources_rc"][()],
        )
    r = refit_test([sub], PARAMS, SolverOptions(), seed=3, keep_truths=True)
    s = r.summary
    assert s["n_sources"] == 2 and s["n_events"] >= 20
    assert len(r.sources) == 2 and len(r.events) == s["n_events"]
    assert all(row["subproblem"] == 0 for row in r.sources + r.events)
    assert 0.3 < s["recovered"] < 1.2 and 0.3 < s["pooled"] < 1.2
    assert 1.0 < s["f0_ratio"] < 1.3
    found = np.mean([e["found"] for e in r.events])
    assert found > 0.25
    many = [row for row in r.by_size if row["n"] >= 10]
    assert many and all(0.6 < row["recovered_found"] < 1.2 for row in many)
    truth = r.truths[0]
    assert truth.z.shape == sub.finv.shape and truth.z.min() >= 0
    # the redrawn data have the truth's mean
    mean = truth.b + truth.h @ truth.x
    assert truth.z.mean() == pytest.approx(mean.mean(), rel=0.02)
    again = refit_test([sub], PARAMS, seed=3, keep_truths=True)
    assert np.array_equal(again.truths[0].z, truth.z)
    assert refit_test([sub], PARAMS, seed=4).truths == []
    # the truth of a redraw needs its baseline
    fit = solve_subproblem(sub, PARAMS)
    with pytest.raises(ValueError, match="keep_baseline"):
        draw_refit_data(sub, fit, PARAMS, np.random.default_rng(0))


def _crop_truth(n_t=40):
    """A truth on a 4 x 5 crop with a baseline that varies over pixels."""
    sel = np.ones((4, 5), bool)
    sub = Subproblem(
        y=np.ones((1, 20, n_t)),
        finv=np.ones((20, n_t)),
        sel=sel,
        sources_rc=np.array([[1, 2]]),
    )
    b = np.outer(np.arange(1.0, 21.0), np.ones(n_t))
    truth = RefitTruth(
        h=np.full((20, 1), 0.05),
        x=np.zeros((1, n_t)),
        b=b,
        f0=np.full((1, n_t), 4.0),
        z=np.zeros((20, n_t)),
    )
    return sub, truth


def test_baseline_errors():
    """Scale, drift, trial-edge and blur errors of the true baseline."""
    sub, truth = _crop_truth()
    rng = np.random.default_rng(0)
    b = truth.b
    np.testing.assert_allclose(scale_error(-0.05)(truth, sub, rng), 0.95 * b)
    d = drift_error(0.1)(truth, sub, rng)
    np.testing.assert_allclose(d[:, 0], 0.95 * b[:, 0])
    np.testing.assert_allclose(d[:, -1], 1.05 * b[:, -1])
    np.testing.assert_allclose(d.mean(1), b.mean(1))
    e = edge_error(0.03, 5)(truth, sub, rng)
    np.testing.assert_allclose(e[:, :5], 1.03 * b[:, :5])
    np.testing.assert_allclose(e[:, -5:], 1.03 * b[:, -5:])
    np.testing.assert_allclose(e[:, 5:-5], b[:, 5:-5])
    # a flat baseline stays flat when blurred, edges included
    flat = RefitTruth(truth.h, truth.x, np.full_like(b, 3.0), truth.f0, b)
    np.testing.assert_allclose(blur_error(1.0)(flat, sub, rng), 3.0)
    blurred = blur_error(1.0)(truth, sub, rng)
    assert blurred.shape == b.shape
    assert np.ptp(blurred[:, 0]) < np.ptp(b[:, 0])
    np.testing.assert_allclose(blurred.mean(), b.mean(), rtol=0.05)


def test_shared_component():
    """A shared fast change of the given root mean square, the same for
    every pixel, reproducible from its generator."""
    sub, truth = _crop_truth(n_t=4000)
    change = shared_component(0.02)
    b1 = change(truth.b, sub, np.random.default_rng(5))
    course = b1 / truth.b - 1
    np.testing.assert_allclose(course, course[:1].repeat(20, 0))
    assert np.sqrt(np.mean(course[0] ** 2)) == pytest.approx(0.02)
    # smoothed: neighbouring frames are correlated
    assert np.corrcoef(course[0, :-1], course[0, 1:])[0, 1] > 0.6
    b2 = change(truth.b, sub, np.random.default_rng(5))
    np.testing.assert_array_equal(b1, b2)


def test_refit_test_with_a_given_baseline():
    """The refit can get the true baseline, or one with an error; the
    truth's baseline can carry a shared change that the given one lacks;
    the truth fits can be passed in."""
    with h5py.File(SUBPROBLEM, "r") as f:
        sub = Subproblem(
            y=f["in/y"][()][:1].astype(np.float64),
            finv=f["in/finv"][()].astype(np.float64),
            sel=f["in/sel"][()].astype(bool),
            sources_rc=f["in/sources_rc"][()],
        )
    opt = SolverOptions(keep_baseline=True)
    fit = solve_subproblem(sub, PARAMS, opt)
    fixed = replace(PARAMS, baseline_mode="fixed")
    exact = refit_test(
        [sub],
        PARAMS,
        opt,
        refit_params=fixed,
        seed=2,
        fits=[fit],
        given_baseline=scale_error(0.0),
        keep_truths=True,
    )
    plain = refit_test([sub], PARAMS, opt, seed=2, keep_truths=True)
    # the same draws whatever the refit gets
    np.testing.assert_array_equal(exact.truths[0].z, plain.truths[0].z)
    high = refit_test(
        [sub],
        PARAMS,
        opt,
        refit_params=fixed,
        seed=2,
        fits=[fit],
        given_baseline=scale_error(0.1),
    )
    assert high.summary["pooled"] < exact.summary["pooled"]
    assert high.summary["f0_ratio"] > exact.summary["f0_ratio"] + 0.01
    shared = refit_test(
        [sub],
        PARAMS,
        opt,
        refit_params=fixed,
        seed=2,
        fits=[fit],
        truth_baseline=shared_component(0.05),
        keep_truths=True,
    )
    tr = shared.truths[0]
    np.testing.assert_array_equal(tr.b_fit, fit.baseline[0])
    assert not np.allclose(tr.b, tr.b_fit)
    np.testing.assert_allclose(tr.f0, source_f0(tr.h, tr.b))
    with pytest.raises(ValueError, match="fits for"):
        refit_test([sub], PARAMS, opt, fits=[fit, fit])
