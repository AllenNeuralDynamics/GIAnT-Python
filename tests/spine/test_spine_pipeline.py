"""End-to-end tests of ``giant spine`` on a synthetic session."""

import json

import numpy as np
import pytest

from giant_python.cli import main
from giant_python.spine.pipeline import (
    _stem,
    make_spine_summary,
    table_trials,
)
from giant_python.spine.product import read_product
from giant_python.spine.readout import norm_log
from giant_python.spine.summary_io import RunSummary, TrialMap

from . import synthetic as syn


@pytest.fixture
def session(tmp_path):
    """The synthetic session's folders."""
    return syn.make_session(tmp_path)


def test_run_summary_reads_matlab_orientation(session):
    """Arrays come back in MATLAB's dimension order and orientation."""
    run = RunSummary(session["run"])
    assert run.paths == [1, 2]
    assert run.shape(1) == (syn.H, syn.W)
    assert run.n_channels(1) == 2
    np.testing.assert_allclose(run.profiles(1), syn.footprints(), atol=1e-7)
    stack = run.per_trial_images(1, "per_trial_mean_im", 0)
    expect = 1.2 * syn.consensus_trial(1, 2)
    np.testing.assert_allclose(stack[2], expect)
    labels, masks = run.user_rois(1)
    assert labels == ["roi A", "roi B"] and masks.shape == (2, syn.H, syn.W)
    act = run.activity(1)
    np.testing.assert_allclose(act["mean_dF"][:, 0], 10 * np.arange(4) + 4.5)


def test_trial_map_needs_a_base_only_for_affine_maps(session):
    """Shift maps need no index base; true affine ones do."""
    run = RunSummary(session["run"])
    tm = run.trial_map(1)
    assert isinstance(tm, TrialMap) and tm.displacement is None
    r, c = tm.grid_coords(1, (2, 2))
    np.testing.assert_allclose(r, [[1, 1], [2, 2]])


def test_table_trials(session):
    """Stems from the registered movies, numbers from the raw files."""
    stems, acq = table_trials(session["table"], 2)
    assert stems[0] == "E1T1DMD2"
    np.testing.assert_array_equal(acq, [2, 3, 4, 5])


def test_readout_writes_a_product(session, tmp_path):
    """The readout's images, windows and normalized sizes are consistent,
    and a rerun reuses the cache and the output."""
    out = tmp_path / "out"
    rep = make_spine_summary(
        session["run"],
        session["table"],
        out,
        baseline_dir=session["baselines"],
        moco_dir=session["moco"],
        trim_s=0.1,
    )
    assert not rep["reused"]
    prod = read_product(rep["output"])
    p = prod["paths"][1]
    assert set(p["images"]) == {
        "mean_ch1",
        "mean_ch2",
        "f0_silo_ch1",
        "f0_silo_ch2",
        "f0_net",
    }
    # the network image equals the run's own F0 image (same scene)
    ok = p["valid"]
    np.testing.assert_allclose(
        p["images"]["f0_net"][:, ok],
        p["images"]["f0_silo_ch1"][:, ok],
        rtol=1e-3,
    )
    # a scene scaled per trial: every source's normalized size is 0
    norm = p["normalized"]["sources"]
    assert norm["ok"].all()
    np.testing.assert_allclose(norm["f0_net"], 0, atol=2e-3)
    np.testing.assert_allclose(
        norm["f0_silo_ch1"],
        norm_log(p["readout"]["sources"]["f0_silo_ch1"]),
    )
    user, user_any = p["readout"]["user"], p["readout"]["user_any"]
    assert (user["n_pixels"] > 0).all()
    assert "coverage_f0_net" in user_any and "coverage_f0_net" not in user
    # the MATLAB-style readout reaches pixels outside the common windows
    assert (user_any["coverage_f0_net"] >= 0).all()
    assert p["trials"]["acquisition"].tolist() == [2, 3, 4, 5]
    assert p["rois"]["sources"].labels[0] == "source 0"
    rep2 = make_spine_summary(
        session["run"],
        session["table"],
        out,
        baseline_dir=session["baselines"],
        moco_dir=session["moco"],
        trim_s=0.1,
    )
    assert rep2["reused"] and rep2["output"] == rep["output"]
    rep3 = make_spine_summary(
        session["run"],
        session["table"],
        out,
        baseline_dir=session["baselines"],
        moco_dir=session["moco"],
        trim_s=0.2,
    )
    assert rep3["output"] != rep["output"]
    assert not rep3["paths"][1]["f0_reused"]


def test_cli_readout_and_images(session, tmp_path, capsys):
    """``giant spine images`` fills the cache that ``readout`` reuses."""
    out = tmp_path / "cli"
    common = [
        str(session["run"]),
        "--table",
        str(session["table"]),
        "--baseline-dir",
        str(session["baselines"]),
        "--moco-dir",
        str(session["moco"]),
        "--out-dir",
        str(out),
        "--trim-s",
        "0.1",
        "--quiet",
    ]
    assert main(["spine", "images"] + common) == 0
    rep = json.loads(capsys.readouterr().out)
    assert not rep["paths"]["1"]["f0_reused"]
    assert main(["spine", "readout"] + common + ["--paths", "1"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["paths"]["1"]["f0_reused"] and list(rep["paths"]) == ["1"]


def test_readout_without_baselines(tmp_path):
    """Without baselines the run's own images are read out."""
    s = syn.make_session(tmp_path, with_f0_silo=False)
    rep = make_spine_summary(s["run"], s["table"], tmp_path / "o")
    p = read_product(rep["output"])["paths"][2]
    assert set(p["images"]) == {"mean_ch1", "mean_ch2"}


def test_baselines_need_the_moco_folder(session, tmp_path):
    """A baseline folder without the motion-correction folder is refused."""
    with pytest.raises(ValueError, match="both"):
        make_spine_summary(
            session["run"],
            session["table"],
            tmp_path,
            baseline_dir=session["baselines"],
        )


def test_activity_uses_the_activity_channel(tmp_path):
    """Activity comes from ``params/activityChannel``; a trial without
    traces is NaN, not inactive."""
    s = syn.make_session(tmp_path, activity_channel=2, missing_trial=1)
    run = RunSummary(s["run"])
    assert run.activity_channel() == 2
    act = run.activity(1)
    np.testing.assert_allclose(
        act["mean_dF"][[0, 2, 3], 0], 1000 + 10 * np.array([0, 2, 3]) + 4.5
    )
    np.testing.assert_allclose(act["event_frames"][[0, 2, 3], 0], 10)
    assert np.isnan(act["mean_dF"][1]).all()
    assert np.isnan(act["event_frames"][1]).all()


def test_summary_key_follows_the_paths(tmp_path):
    """Different path selections give different summaries."""
    s = syn.make_session(tmp_path)
    keys = {
        tuple(sel or ()): make_spine_summary(
            s["run"], s["table"], tmp_path / "out", paths=sel
        )["key"]
        for sel in (None, [1], [2])
    }
    assert len(set(keys.values())) == 3


def test_network_images_take_a_trial_warp(session, tmp_path):
    """The extraction plan's TrialWarp works as the trial map."""
    from giant_python.extraction.raster.alignment import TrialWarp
    from giant_python.spine.images import network_f0_stack

    run = RunSummary(session["run"])
    tmap = run.trial_map(1)
    stems, _ = table_trials(session["table"], 1)
    args = (session["baselines"], session["moco"], stems)
    a = network_f0_stack(*args, tmap, run.shape(1), trim_s=0.1)
    b = network_f0_stack(
        *args, TrialWarp(tmap.affine), run.shape(1), trim_s=0.1
    )
    np.testing.assert_allclose(a.images, b.images, equal_nan=True)
    assert np.isfinite(a.images).any()


def test_write_atomic_gives_each_writer_its_own_file(tmp_path):
    """A writer that starts while another is writing the same file does not
    share its temporary file; both finish, and nothing is left behind. A
    failed write leaves no file."""
    from giant_python.spine.pipeline import _write_atomic

    dest = tmp_path / "out" / "x.txt"

    def outer(tmp):
        """Write, with a second writer of the same file in between."""
        tmp.write_text("outer")
        _write_atomic(dest, lambda t: t.write_text("inner"))
        assert tmp.read_text() == "outer"

    _write_atomic(dest, outer)
    assert dest.read_text() == "outer"
    assert [p.name for p in dest.parent.iterdir()] == ["x.txt"]

    def fail(tmp):
        """A write that fails."""
        tmp.write_text("partial")
        raise RuntimeError("stop")

    other = tmp_path / "out" / "y.txt"
    with pytest.raises(RuntimeError):
        _write_atomic(other, fail)
    assert [p.name for p in dest.parent.iterdir()] == ["x.txt"]


def test_stem_strips_windows_folders():
    """A table written on Windows gives the bare stem on any system."""
    name = "E1T1DMD1_REGISTERED_DOWNSAMPLED-80Hz.tif"
    assert _stem("C:\\moco\\" + name) == "E1T1DMD1"
    assert _stem("/data/moco/" + name) == "E1T1DMD1"
    assert _stem("E1T1DMD1.tif") is None


def test_network_images_follow_the_runs_discards(tmp_path):
    """Without overrides the network images leave out the frames the run
    discarded (``params/discardInitial_s``); an explicit value wins."""
    s = syn.make_session(tmp_path, discard_initial_s=0.5)
    kw = dict(baseline_dir=s["baselines"], moco_dir=s["moco"], trim_s=0.1)
    rep = make_spine_summary(s["run"], s["table"], tmp_path / "a", **kw)
    prod = read_product(rep["output"])
    assert prod["attrs"]["settings"]["discard_initial_s"] == 0.5
    n_run = prod["paths"][1]["image_attrs"]["f0_net"]["n_frames"]
    # 40 discarded frames at the start, 8 trimmed at the end
    assert (n_run == syn.N_FRAMES - 48).all()
    rep0 = make_spine_summary(
        s["run"], s["table"], tmp_path / "b", discard_initial_s=0.0, **kw
    )
    n0 = read_product(rep0["output"])["paths"][1]["image_attrs"]["f0_net"]
    assert (n0["n_frames"] == syn.N_FRAMES - 16).all()
