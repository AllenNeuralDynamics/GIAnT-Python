"""Tests of the gain-rescaled F0 (:mod:`giant_python.spine.gains`)."""

import json

import h5py
import numpy as np
import pytest

from giant_python.spine.gains import (
    gain_maps,
    results_files,
    selected_pixels,
    trial_gains,
)
from giant_python.spine.pipeline import make_spine_summary
from giant_python.spine.product import read_product

from . import synthetic as syn


def test_gain_maps_average_overlapping_subproblems(tmp_path):
    """A pixel in two subproblems gets the mean of its gains, and their
    spread is reported."""
    work = syn.make_gain_work(tmp_path, lambda t: 1.0 + 0.1 * t)
    rows, cols, shape = selected_pixels(work / "plan" / "path1.h5")
    gain, spread, n_files = gain_maps(work, 1, syn.N_TRIALS)
    assert gain.shape == (syn.N_TRIALS,) + shape
    assert (n_files == 1).all()
    n = rows.size
    overlap = n // 2
    g1 = 1.1  # trial 2
    np.testing.assert_allclose(gain[1, rows[0], cols[0]], g1)
    np.testing.assert_allclose(gain[1, rows[-1], cols[-1]], 2 * g1)
    np.testing.assert_allclose(
        gain[1, rows[overlap], cols[overlap]], 1.5 * g1, rtol=1e-6
    )
    np.testing.assert_allclose(
        spread[1, rows[overlap], cols[overlap]], g1, rtol=1e-6
    )
    assert spread[1, rows[0], cols[0]] == 0
    assert np.isnan(gain[1, 0, 0])


def test_trial_gains_needs_a_gain_record(tmp_path):
    """Results of a mode without gains are refused."""
    path = tmp_path / "r.h5"
    with h5py.File(path, "w") as f:
        f["subproblems/index"] = np.array([0])
        g = f.create_group("subproblems/0")
        g["pixels"] = np.arange(3)
        g.attrs["info"] = json.dumps({"iterations": 3})
    with pytest.raises(ValueError, match="no gain record"):
        trial_gains([path])


def test_readout_adds_the_gain_image(tmp_path):
    """``f0_gain`` is the gain times the network's image, at the selected
    pixels."""
    s = syn.make_session(tmp_path)
    work = syn.make_gain_work(tmp_path, lambda t: 1.0, s["baselines"])
    rep = make_spine_summary(
        s["run"],
        s["table"],
        tmp_path / "out",
        baseline_dir=s["baselines"],
        moco_dir=s["moco"],
        trim_s=0.1,
        gain_work_dir=work,
    )
    p = read_product(rep["output"])["paths"][1]
    gain, _, _ = gain_maps(work, 1, syn.N_TRIALS)
    ok = np.isfinite(p["images"]["f0_gain"])
    assert ok.any()
    np.testing.assert_allclose(
        p["images"]["f0_gain"][ok],
        (gain * p["images"]["f0_net"])[ok],
        rtol=1e-6,
    )
    assert not np.isfinite(p["images"]["f0_gain"][:, 0, 0]).any()
    assert rep["paths"][1]["gain"]["median"] > 1.0
    # every image is read in the same windows: with gains of 1, the user
    # ROIs' f0_gain and f0_net readouts agree
    user = p["readout"]["user"]
    ok = np.isfinite(user["f0_gain"])
    assert ok.any()
    np.testing.assert_allclose(
        user["f0_gain"][ok], user["f0_net"][ok], rtol=1e-6
    )
    with pytest.raises(ValueError, match="needs --baseline-dir"):
        make_spine_summary(s["run"], s["table"], tmp_path, gain_work_dir=work)


def test_results_files_select_the_plan_and_the_exact_trial(tmp_path):
    """Results of another plan are left out, a longer trial number with the
    same prefix is not this trial, and two solves of the plan are refused."""
    work = syn.make_gain_work(tmp_path, lambda t: 1.0)
    res = work / "results"
    info = {"baseline_step": {"gain": [9.0]}}
    syn.write_results(
        res / "path1_trial0001_s9.h5", {0: ([0], info)}, plan_hash="old-plan"
    )
    assert [f.name for f in results_files(work, 1, 0)] == [
        "path1_trial0001.h5"
    ]
    syn.write_results(res / "path1_trial1000.h5", {0: ([0], info)})
    syn.write_results(res / "path1_trial10000.h5", {0: ([0], info)})
    assert [f.name for f in results_files(work, 1, 999)] == [
        "path1_trial1000.h5"
    ]
    syn.write_results(
        res / "path1_trial0002_s1.h5",
        {0: ([0], info)},
        solve_science="science-b",
        container_hash="c1-1",
    )
    with pytest.raises(ValueError, match="more than one solve"):
        results_files(work, 1, 1)


def test_trial_gains_counts_a_copied_subproblem_once(tmp_path):
    """A subproblem in two files (identical copies) is one subproblem; two
    different copies are refused."""
    work = syn.make_gain_work(tmp_path, lambda t: 1.0)
    rows, cols, _ = selected_pixels(work / "plan" / "path1.h5")
    n = rows.size
    first = np.arange(0, n // 2 + 1)
    copy = {0: (first, {"baseline_step": {"gain": [1.0] * first.size}})}
    syn.write_results(
        work / "results" / "path1_trial0001_s1.h5", copy, container_hash="c1-0"
    )
    gain, _, _ = gain_maps(work, 1, syn.N_TRIALS)
    k = n // 2  # the pixel in both subproblems: gains 1 and 2
    np.testing.assert_allclose(gain[0, rows[k], cols[k]], 1.5, rtol=1e-6)
    other = {0: (first, {"baseline_step": {"gain": [3.0] * first.size}})}
    syn.write_results(
        work / "results" / "path1_trial0001_s1.h5",
        other,
        container_hash="c1-0",
    )
    with pytest.raises(ValueError, match="two different results"):
        gain_maps(work, 1, syn.N_TRIALS)


def test_trial_gains_refuses_a_shared_time_course(tmp_path):
    """'gain_shared' subproblems are read only while their shared time
    course is 1 (then the baseline is still gain x network baseline)."""
    path = tmp_path / "r.h5"
    step = {"gain": [2.0, 2.0], "shared": [1.0, 1.0, 1.0]}
    syn.write_results(path, {0: ([0, 1], {"baseline_step": step})})
    pix, g = trial_gains([path])
    assert pix.tolist() == [0, 1] and g.tolist() == [2.0, 2.0]
    step["shared"] = [1.0, 1.1, 0.9]
    syn.write_results(path, {0: ([0, 1], {"baseline_step": step})})
    with pytest.raises(ValueError, match="shared baseline time course"):
        trial_gains([path])


def test_summary_key_follows_the_gain_plan(tmp_path):
    """A changed plan gives a new summary, not the old one reused."""
    s = syn.make_session(tmp_path)
    work = syn.make_gain_work(tmp_path, lambda t: 1.0, s["baselines"])
    kw = dict(
        baseline_dir=s["baselines"],
        moco_dir=s["moco"],
        trim_s=0.1,
        gain_work_dir=work,
    )
    rep = make_spine_summary(s["run"], s["table"], tmp_path / "out", **kw)
    with h5py.File(work / "plan" / "path1.h5", "a") as f:
        f.attrs["note"] = "replanned"
    rep2 = make_spine_summary(s["run"], s["table"], tmp_path / "out", **kw)
    assert rep2["key"] != rep["key"] and not rep2["reused"]


def test_gains_need_the_images_baseline_files(tmp_path):
    """Gains are used only with the baseline files they were fitted
    against: a baseline predicted again after the solve, or results that
    record no network baseline, are refused."""
    import os

    s = syn.make_session(tmp_path)
    work = syn.make_gain_work(tmp_path, lambda t: 1.0, s["baselines"])
    kw = dict(
        baseline_dir=s["baselines"],
        moco_dir=s["moco"],
        trim_s=0.1,
        gain_work_dir=work,
    )
    make_spine_summary(s["run"], s["table"], tmp_path / "out", **kw)
    b = s["baselines"] / "E1T2DMD1_BASELINE-80Hz.h5"
    st = b.stat()
    os.utime(b, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    with pytest.raises(ValueError, match="Path1 trial 2: the gains were"):
        make_spine_summary(s["run"], s["table"], tmp_path / "out", **kw)
    os.utime(b, ns=(st.st_atime_ns, st.st_mtime_ns))
    plain = syn.make_gain_work(tmp_path / "plain", lambda t: 1.0)
    kw["gain_work_dir"] = plain
    with pytest.raises(ValueError, match="record no network baseline"):
        make_spine_summary(s["run"], s["table"], tmp_path / "out", **kw)
