"""End-to-end tests of ``giant spine`` on a synthetic session."""

import json

import numpy as np
import pytest

from giant_python.cli import main
from giant_python.spine.pipeline import (
    _stem,
    _trials_and_window,
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


def test_affine_maps_convert_from_one_based(tmp_path):
    """A true affine map written one-based (GIAnT-MATLAB) gives the same
    zero-based positions as the zero-based map (GIAnT-Python)."""
    lin = np.array([[1.01, 0.02], [-0.015, 0.99]])
    b0 = np.array([1.5, -2.25])
    a0 = np.zeros((syn.N_TRIALS, 2, 3))
    a0[:, :, :2] = lin
    a0[:, :, 2] = b0
    # one-based: r1 = A r1' + b1 with r1 = r0 + 1 on both sides
    a1 = a0.copy()
    a1[:, :, 2] = b0 - lin[:, 0] - lin[:, 1] + 1
    shape = (3, 4)
    rr, cc = np.meshgrid(np.arange(3.0), np.arange(4.0), indexing="ij")
    expect_r = lin[0, 0] * rr + lin[0, 1] * cc + b0[0]
    expect_c = lin[1, 0] * rr + lin[1, 1] * cc + b0[1]
    for base, a in ((0, a0), (1, a1)):
        s = syn.make_session(tmp_path / f"base{base}", affine=a)
        run = RunSummary(s["run"])
        with pytest.raises(ValueError, match="index base"):
            run.trial_map(1)
        r, c = run.trial_map(1, affine_base=base).grid_coords(2, shape)
        np.testing.assert_allclose(r, expect_r, atol=1e-12)
        np.testing.assert_allclose(c, expect_c, atol=1e-12)


def test_product_records_the_activity_channel(tmp_path):
    """The product says which file channel is the run's activity."""
    s = syn.make_session(tmp_path, activity_channel=2)
    rep = make_spine_summary(s["run"], s["table"], tmp_path / "out")
    prod = read_product(rep["output"])
    assert prod["paths"][1]["activity_channel"] == 2
    assert {"mean_ch1", "mean_ch2"} <= set(prod["paths"][1]["images"])


def test_displacement_maps_end_to_end(tmp_path):
    """A run with a displacement field (patchwise alignment) puts the
    network's trials in place through the field, not the affine maps."""
    kw = dict(trim_s=0.1)
    s = syn.make_session(tmp_path / "d", displacement=True)
    run = RunSummary(s["run"])
    assert run.trial_map(1).displacement is not None
    rep = make_spine_summary(
        s["run"],
        s["table"],
        tmp_path / "d" / "out",
        baseline_dir=s["baselines"],
        moco_dir=s["moco"],
        **kw,
    )
    p = read_product(rep["output"])["paths"][1]
    ok = p["valid"]
    assert ok.sum() > 0.5 * ok.size
    np.testing.assert_allclose(
        p["images"]["f0_net"][:, ok],
        p["images"]["f0_silo_ch1"][:, ok],
        rtol=1e-3,
    )
    np.testing.assert_allclose(
        p["normalized"]["sources"]["f0_net"], 0, atol=2e-3
    )
    # the same maps as affine identity alone leave the trials displaced
    a = np.zeros((syn.N_TRIALS, 2, 3))
    a[:, :, :2] = np.eye(2)
    s0 = syn.make_session(tmp_path / "a", affine=a)
    rep0 = make_spine_summary(
        s0["run"],
        s0["table"],
        tmp_path / "a" / "out",
        baseline_dir=s0["baselines"],
        moco_dir=s0["moco"],
        **kw,
    )
    p0 = read_product(rep0["output"])["paths"][1]
    ok0 = p0["valid"]
    net = p0["images"]["f0_net"][1:, ok0]
    assert np.abs(net / p0["images"]["f0_silo_ch1"][1:, ok0] - 1).max() > 1e-2


def test_fitted_images_do_not_shrink_the_windows(tmp_path):
    """SILo's own F0 images, with values only at the pixels it fitted, are
    read out in the windows of the field images: the sources (inside the
    fitted pixels) read as before, a user ROI partly outside keeps its
    readouts of the other images and gets NaN in f0_silo, and the report
    counts it."""
    kw = dict(trim_s=0.1)
    full = syn.make_session(tmp_path / "full")
    box = syn.make_session(
        tmp_path / "box", f0_silo_box=(slice(4, 20), slice(5, 25))
    )
    reps, prods = {}, {}
    for name, s in (("full", full), ("box", box)):
        reps[name] = make_spine_summary(
            s["run"],
            s["table"],
            tmp_path / name / "out",
            baseline_dir=s["baselines"],
            moco_dir=s["moco"],
            **kw,
        )
        prods[name] = read_product(reps[name]["output"])["paths"][1]
    a, b = prods["full"], prods["box"]
    np.testing.assert_array_equal(a["valid"], b["valid"])
    for k in a["readout"]["sources"]:
        np.testing.assert_allclose(
            a["readout"]["sources"][k], b["readout"]["sources"][k]
        )
    for k in a["normalized"]["sources"]:
        np.testing.assert_allclose(
            a["normalized"]["sources"][k], b["normalized"]["sources"][k]
        )
    ua, ub = a["readout"]["user"], b["readout"]["user"]
    np.testing.assert_array_equal(ua["n_pixels"], ub["n_pixels"])
    np.testing.assert_allclose(ua["f0_net"], ub["f0_net"])
    assert np.isnan(ub["f0_silo_ch1"][:, 0]).all()  # roi A: partly outside
    np.testing.assert_allclose(
        ub["f0_silo_ch1"][:, 1], ua["f0_silo_ch1"][:, 1]
    )
    assert reps["box"]["paths"][1]["not_covered"] == {
        "user": {"f0_silo_ch1": 1, "f0_silo_ch2": 1}
    }
    assert "not_covered" not in reps["full"]["paths"][1]
    assert b["image_attrs"]["f0_silo_ch1"]["covers"] == "fitted"
    assert b["image_attrs"]["f0_net"]["covers"] == "field"
    # a reused file gives the same report and the same notes again
    notes = []
    again = make_spine_summary(
        box["run"],
        box["table"],
        tmp_path / "box" / "out",
        baseline_dir=box["baselines"],
        moco_dir=box["moco"],
        progress=notes.append,
        **kw,
    )
    assert again["reused"]
    for k in ("not_covered", "window_images", "trials_used", "valid_pixels"):
        assert again["paths"][1][k] == reps["box"]["paths"][1][k]
    assert sum("does not cover" in m for m in notes) == 4  # 2 images x 2 paths


def test_no_shared_trial_reads_nothing():
    """Field images without a trial in common leave no trial and no valid
    pixel, with a note, rather than windows over the whole frame."""
    from giant_python.spine.readout import common_valid

    a = np.ones((4, 3, 3))
    b = np.ones((4, 3, 3))
    a[2:] = np.nan
    b[:2] = np.nan
    images = {
        "mean_ch1": (a, {"covers": "field"}),
        "f0_net": (b, {"covers": "field"}),
    }
    notes = []
    used, window = _trials_and_window(1, images, np.full(4, 10), notes.append)
    assert used.size == 0 and window == ["mean_ch1", "f0_net"]
    assert any("nothing is read out" in m for m in notes)
    assert not common_valid([a, b], used).any()


def test_acquisition_numbers_from_the_file_name(tmp_path):
    """The acquisition number is read from the raw file's own name (not a
    folder that also says TRIAL, with either separator), and a table whose
    raw and registered lists differ in length is refused."""
    from giant_python.io.hdf5 import save_struct_h5

    reg = np.array(
        [[f"E1T{t}DMD1_REGISTERED_DOWNSAMPLED-80Hz.tif" for t in (1, 2)]],
        dtype=object,
    )
    raw = np.array(
        [
            [
                "C:\\TRIAL09\\activity_x_DMD1-TRIAL000002.dat",
                "/data/TRIAL09/activity_x_DMD1-TRIAL000003.dat",
            ]
        ],
        dtype=object,
    )
    table = tmp_path / "trial_table.h5"
    save_struct_h5(
        table,
        {
            "datadr": str(tmp_path),
            "savedr": str(tmp_path),
            "filename": raw,
            "motion_correction": {"fn_reg_ds": reg},
        },
    )
    stems, acq = table_trials(table, 1)
    assert stems == ["E1T1DMD1", "E1T2DMD1"]
    np.testing.assert_array_equal(acq, [2, 3])
    save_struct_h5(
        table,
        {
            "datadr": str(tmp_path),
            "savedr": str(tmp_path),
            "filename": raw[:, :1],
            "motion_correction": {"fn_reg_ds": reg},
        },
    )
    with pytest.raises(ValueError, match="1 raw files but 2"):
        table_trials(table, 1)


def test_untrimmed_trials_are_reported(session, tmp_path):
    """A trial too short for the trim is averaged whole, as GIAnT-MATLAB
    does, and the product, the report and a note say so."""
    notes = []
    kw = dict(baseline_dir=session["baselines"], moco_dir=session["moco"])
    # 160 frames: a 1 s trim at 80 Hz leaves nothing
    rep = make_spine_summary(
        session["run"],
        session["table"],
        tmp_path / "long",
        trim_s=1.0,
        progress=notes.append,
        **kw,
    )
    p = read_product(rep["output"])["paths"][1]
    attrs = p["image_attrs"]["f0_net"]
    assert attrs["untrimmed"].astype(bool).all()
    assert (attrs["n_frames"] == syn.N_FRAMES).all()
    assert rep["paths"][1]["f0_untrimmed"] == [1, 2, 3, 4]
    assert sum("too short for the trim" in m for m in notes) == 2
    rep = make_spine_summary(
        session["run"], session["table"], tmp_path / "short", trim_s=0.1, **kw
    )
    p = read_product(rep["output"])["paths"][1]
    assert not p["image_attrs"]["f0_net"]["untrimmed"].any()
    assert "f0_untrimmed" not in rep["paths"][1]


def test_readout_runs_the_trend_tests(session, tmp_path):
    """Every normalized image gets the trend tests over its trials, with
    trial positions as trial-table indices; the test settings change the
    output's key but not the network images' cache."""
    from giant_python.spine.trends import trend_tests

    kw = dict(
        baseline_dir=session["baselines"], moco_dir=session["moco"], trim_s=0.1
    )
    rep = make_spine_summary(
        session["run"], session["table"], tmp_path / "o", n_perm=200, **kw
    )
    p = read_product(rep["output"])["paths"][1]
    norm = p["normalized"]["sources"]
    tr = p["trends"]["sources"]
    assert set(tr) == set(norm) - {"ok"}
    rec = tr["f0_net"]
    rows = rec["trials"]
    np.testing.assert_array_equal(rows, np.arange(syn.N_TRIALS))
    direct = trend_tests(
        norm["f0_net"][rows],
        norm["ok"],
        p["trials"]["acquisition"][rows],
        n_perm=200,
    )
    tested = direct["tested"]
    assert tested.any()
    np.testing.assert_array_equal(rec["tested"], tested)
    for k in ("step_t", "step_p", "brief_t", "resid_p", "shared_r"):
        np.testing.assert_array_equal(rec[k], direct[k])
    # positions as trial-table indices where a split or a run was found,
    # -1 elsewhere
    step = tested & (direct["step_k"] > 0)
    assert step.any()
    np.testing.assert_array_equal(
        rec["step_after"][step], rows[direct["step_k"][step] - 1]
    )
    assert (rec["step_after"][~step] == -1).all()
    run = tested & (direct["brief_start"] >= 0)
    np.testing.assert_array_equal(
        rec["brief_first"][run], rows[direct["brief_start"][run]]
    )
    assert (rec["brief_first"][~run] == -1).all()
    np.testing.assert_array_equal(
        rec["brief_len"][tested & ~run], 0
    )
    assert rec["attrs"]["n_perm"] == 200
    assert rec["attrs"]["var_share"] == pytest.approx(direct["var_share"])
    summ = rep["paths"][1]["trends"]["f0_net"]
    assert summ["tested"] == int(tested.sum())
    assert summ["counts"] == direct["counts"]
    rep2 = make_spine_summary(
        session["run"], session["table"], tmp_path / "o", n_perm=0, **kw
    )
    assert rep2["output"] != rep["output"] and not rep2["reused"]
    assert rep2["paths"][1]["f0_reused"]
    p2 = read_product(rep2["output"])["paths"][1]
    assert np.isnan(p2["trends"]["sources"]["f0_net"]["step_p"]).all()


def test_readout_refuses_bad_trend_settings(session, tmp_path):
    """Invalid trend settings are refused before anything is read or
    reused."""
    with pytest.raises(ValueError, match="fdr_q"):
        make_spine_summary(
            session["run"], session["table"], tmp_path / "o", fdr_q=2.0
        )
    assert not (tmp_path / "o").exists()


def test_trials_without_numbers_are_noted(session, tmp_path, monkeypatch):
    """A trial whose raw file has no acquisition number is left out of the
    brief test's runs, and the summary's notes say so."""
    import giant_python.spine.pipeline as pl

    real = pl.table_trials

    def no_number(table, path):
        """The table's trials with trial 2 unnumbered."""
        stems, acq = real(table, path)
        acq = acq.copy()
        acq[1] = -1
        return stems, acq

    monkeypatch.setattr(pl, "table_trials", no_number)
    notes = []
    rep = make_spine_summary(
        session["run"],
        session["table"],
        tmp_path / "o",
        n_perm=0,
        progress=notes.append,
    )
    assert sum("trial(s) 2 have no acquisition number" in m for m in notes)
    p = read_product(rep["output"])["paths"][1]
    blocks = p["trends"]["sources"]["mean_ch1"]["blocks"]
    assert blocks[1] != blocks[0] and blocks[1] != blocks[2]
