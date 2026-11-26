"""Tests of the kinetics test (sourceKinetics, rejectFastSources).

``data/matlab_kinetics.h5`` holds GIAnT-MATLAB's results (main 5a0673b,
MATLAB R2024b) on the inputs of ``data/make_matlab_kinetics.m``: rows with
and without runs of equal values for ``islocalmax``, and three trials of
ten sources (four with the sensor's kinetics, three with one-frame
transients, three in between; discarded frames; a source without data in
one trial; a trial without traces).
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

from giant_python.extraction.raster.kinetics import (
    islocalmax,
    movmean_omitnan,
    reject_fast_sources,
    source_kinetics,
)

DATA = Path(__file__).parent / "data" / "matlab_kinetics.h5"


@pytest.fixture(scope="module")
def ref():
    """MATLAB's inputs and results."""
    with h5py.File(DATA, "r") as f:
        return {k: f[k][()] for k in f}


def test_islocalmax_matches_matlab(ref):
    """Maxima, runs of equal values and MinSeparation as MATLAB's."""
    for i, sep in enumerate(ref["seps"]):
        got = islocalmax(ref["Y"], sep)
        np.testing.assert_array_equal(got, ref["M"][i].astype(bool))


def test_islocalmax_rules():
    """Ends are never maxima; a run's centre is marked; a closer, more
    prominent maximum replaces the last one kept."""
    y = np.array([5.0, 1, 3, 3, 3, 3, 1, 0, 4, 0, 2, 9])
    assert np.flatnonzero(islocalmax(y)).tolist() == [3, 8]
    # 3 (run 2..5) and 8 are 3 apart from the run's end; 8 is more prominent
    assert np.flatnonzero(islocalmax(y, 4)).tolist() == [8]
    assert np.flatnonzero(islocalmax(y, 3)).tolist() == [3, 8]
    assert not islocalmax(np.array([1.0, 2.0])).any()
    two = np.stack([y, y[::-1]])
    np.testing.assert_array_equal(islocalmax(two)[1], islocalmax(y[::-1]))


def test_kinetics_scores_match_matlab(ref):
    """Scores and event counts of sourceKinetics; the sources kept."""
    traces = list(ref["LS"]) + [None]
    score, n_events = source_kinetics(traces, 0.024, 200.0)
    np.testing.assert_allclose(score, ref["score"], rtol=1e-12)
    np.testing.assert_array_equal(n_events, ref["nEvents"])
    test = reject_fast_sources(traces, 10, 0.024, 200.0, 0.2)
    np.testing.assert_array_equal(test.keep, ref["keep"].astype(bool))
    assert test.kept_score.size == test.keep.sum()
    np.testing.assert_array_equal(test.rejected, np.flatnonzero(~test.keep))


def test_movmean_omitnan():
    """Window placement of MATLAB's movmean (even and odd), NaN left out."""
    x = np.array([[1.0, np.nan, 3, 4, 5, 6]])
    np.testing.assert_allclose(
        movmean_omitnan(x, 3)[0], [1, 2, 3.5, 4, 5, 5.5]
    )
    # even window: two before, one after
    np.testing.assert_allclose(
        movmean_omitnan(x, 4)[0], [1, 2, 8 / 3, 4, 4.5, 5]
    )
    assert np.isnan(movmean_omitnan(np.full((1, 4), np.nan), 3)).all()


def test_kinetics_edge_cases():
    """No traces, a short time constant and threshold 0 keep everything."""
    score, n = source_kinetics([None], 0.024, 200.0)
    assert score.size == 0 and n.size == 0
    rng = np.random.default_rng(0)
    trace = rng.standard_normal((3, 400))
    score, n = source_kinetics([trace], 0.005, 200.0)  # tau 1 frame
    assert np.isnan(score).all() and (n == 0).all()
    test = reject_fast_sources([None], 4, 0.024, 200.0, 0.2)
    assert test.keep.all() and np.isnan(test.score).all()
    test = reject_fast_sources([trace], 3, 0.024, 200.0, 0.0)
    assert test.keep.all()
