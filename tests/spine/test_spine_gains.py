"""Tests of the gain-rescaled F0 (:mod:`giant_python.spine.gains`)."""

import json

import h5py
import numpy as np
import pytest

from giant_python.spine.gains import gain_maps, selected_pixels, trial_gains
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
    work = syn.make_gain_work(tmp_path, lambda t: 1.0)
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
    with pytest.raises(ValueError, match="needs --baseline-dir"):
        make_spine_summary(s["run"], s["table"], tmp_path, gain_work_dir=work)
