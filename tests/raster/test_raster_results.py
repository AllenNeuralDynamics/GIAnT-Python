"""Tests of results files: solved subproblems of a trial-path, written by
the solve stage and assembled by the collect stage."""

import numpy as np
import pytest

from giant_python.extraction.raster.results import (
    TrialRecord,
    read_results,
    subproblem_results,
    trial_outcome,
    write_results,
)
from giant_python.extraction.raster.solver import (
    SubproblemResult,
    assemble_results,
    process_result,
)
from giant_python.extraction.raster.solver_setup import SolverParams

N_PX, N_SRC, N_T = 9, 4, 50


def record(seed=0):
    """A trial's record."""
    rng = np.random.default_rng(seed)
    return TrialRecord(
        frame_lines=np.arange(1, N_T + 1) * 40,
        discard=rng.random(N_T) < 0.1,
        global_f=rng.random((2, N_T)),
        roi_f=rng.random((3, 2, N_T)),
        roi_fsvd=rng.random((3, 2, N_T)),
    )


def solved(seed=1):
    """Three subproblems' placements and results."""
    rng = np.random.default_rng(seed)
    placements = [
        (np.array([0, 1, 2, 3]), np.array([0])),
        (np.array([3, 4, 5]), np.array([1, 2])),
        (np.array([6, 7, 8]), np.array([3])),
    ]
    out = {}
    for k, (pix, src) in enumerate(placements):
        n_s = src.size
        out[k] = (
            pix,
            src,
            SubproblemResult(
                footprints=rng.random((pix.size, n_s)),
                events=rng.random((2, n_s, N_T)),
                ls=rng.random((2, n_s, N_T)),
                f0=rng.random((2, n_s, N_T)),
                snr=rng.random(n_s),
                info={"activity1": {"iterations": k, "seconds": 0.1}},
            ),
        )
    return out


def attrs(**kw):
    """A results file's attributes."""
    a = dict(
        settings_hash="h",
        container="path1_trial0001.h5",
        container_hash="c",
        path=1,
        trial=1,
        n_subproblems=3,
        n_pixels=N_PX,
        n_sources=N_SRC,
        photon_scale=25.0,
        extracted=1,
    )
    a.update(kw)
    return a


def test_round_trip(tmp_path):
    """The record, every result and the attributes come back."""
    rec, res = record(), solved()
    write_results(tmp_path / "r.h5", rec, res, attrs())
    back = read_results(tmp_path / "r.h5")
    for name in ("frame_lines", "discard", "global_f", "roi_f", "roi_fsvd"):
        np.testing.assert_array_equal(
            getattr(back.record, name), getattr(rec, name)
        )
    assert sorted(back.results) == [0, 1, 2]
    for k, (pix, src, r) in res.items():
        bp, bs, br = back.results[k]
        np.testing.assert_array_equal(bp, pix)
        np.testing.assert_array_equal(bs, src)
        for name in ("footprints", "events", "ls", "f0", "snr"):
            np.testing.assert_array_equal(getattr(br, name), getattr(r, name))
        assert br.info == r.info
    assert back.attrs["container"] == "path1_trial0001.h5"
    assert back.attrs["n_subproblems"] == 3
    light = read_results(tmp_path / "r.h5", data=False)
    assert light.results == {0: None, 1: None, 2: None}
    assert not list(tmp_path.glob("*.partial"))


def test_assembly_does_not_depend_on_the_split(tmp_path):
    """One file or one per subproblem: the same trial, equal to
    assembleResults and processResult of the results."""
    rec, res = record(), solved()
    p = SolverParams()
    write_results(tmp_path / "all.h5", rec, res, attrs())
    whole = trial_outcome([read_results(tmp_path / "all.h5")], p)
    parts = []
    for k in res:
        f = tmp_path / f"s{k}.h5"
        write_results(f, rec, {k: res[k]}, attrs())
        parts.append(read_results(f))
    split = trial_outcome(parts[::-1], p)

    class Place:
        """What assemble_results reads of a subproblem."""

        def __init__(self, pix, src):
            """Placement only."""
            self.pixels, self.sources = pix, src

    direct = process_result(
        assemble_results(
            [Place(pix, src) for pix, src, _ in res.values()],
            [r for _, _, r in res.values()],
            N_PX,
            N_SRC,
            25.0,
        ),
        p,
        rec.discard,
    )
    for got in (whole, split):
        for name in ("footprints", "events", "denoised", "ls", "f0", "snr"):
            np.testing.assert_array_equal(
                getattr(got.activity, name), getattr(direct, name)
            )
        np.testing.assert_array_equal(got.record.global_f, rec.global_f)


def test_incomplete_or_conflicting_results_raise(tmp_path):
    """A missing subproblem, two different results of one, and files of
    two containers are errors."""
    rec, res = record(), solved()
    p = SolverParams()
    write_results(tmp_path / "a.h5", rec, {0: res[0], 1: res[1]}, attrs())
    with pytest.raises(ValueError, match="no results"):
        trial_outcome([read_results(tmp_path / "a.h5")], p)
    other = solved(seed=5)
    write_results(tmp_path / "b.h5", rec, {1: other[1], 2: res[2]}, attrs())
    with pytest.raises(ValueError, match="two different results"):
        trial_outcome(
            [read_results(tmp_path / "a.h5"), read_results(tmp_path / "b.h5")],
            p,
        )
    write_results(
        tmp_path / "c.h5", rec, {2: res[2]}, attrs(container_hash="other")
    )
    with pytest.raises(ValueError, match="different containers"):
        trial_outcome(
            [read_results(tmp_path / "a.h5"), read_results(tmp_path / "c.h5")],
            p,
        )
    with pytest.raises(ValueError):
        trial_outcome([], p)


def test_trial_not_extracted(tmp_path):
    """A trial too short to extract has its record and no sources."""
    rec = record()
    write_results(tmp_path / "s.h5", rec, {}, attrs(extracted=0))
    out = trial_outcome([read_results(tmp_path / "s.h5")], SolverParams())
    assert out.activity is None
    assert out.record.n_frames == N_T


def test_subproblem_results():
    """Placements of solved subproblems, with their container indices."""

    class Sub:
        """A subproblem's placement."""

        def __init__(self, k):
            """Pixels and sources of subproblem ``k``."""
            self.pixels, self.sources = np.array([k]), np.array([k])

    res = subproblem_results([Sub(4), Sub(7)], ["a", "b"], [4, 7])
    assert sorted(res) == [4, 7] and res[7][2] == "b"
    assert sorted(subproblem_results([Sub(0)], ["a"])) == [0]
    with pytest.raises(ValueError):
        subproblem_results([Sub(0)], ["a", "b"])
