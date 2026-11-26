"""Tests of the ground-truth refit test of the subproblem solver."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from giant_python.extraction.raster.refit_test import (
    RefitTruth,
    by_size,
    find_events,
    refit_test,
    score_refit,
    summarize,
)
from giant_python.extraction.raster.solver import SolverOptions
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
