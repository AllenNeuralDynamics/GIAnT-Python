"""SILo with the F0 network's baseline (``baseline_source='network'``).

The synthetic session of :mod:`test_raster_silo_stages`, with baseline
files as ``slap2-f0 predict-registered`` writes them: the network's
baseline is mapped into the containers and the streamed run's subproblems
alike, so that streamed, staged and sharded runs agree; each stage refuses
files made with another baseline; the prediction step calls slap2-f0 for
the missing files only.
"""

import json
import os
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from giant_python.cli import build_parser, main
from giant_python.cli_silo import params_from_args
from giant_python.extraction.raster import trial as trial_mod
from giant_python.extraction.raster import trial_data as trial_data_mod
from giant_python.extraction.raster.baseline_map import (
    map_baseline,
    read_network_baseline,
    read_trial_baseline,
    subproblem_baseline,
)
from giant_python.extraction.raster.container import (
    read_container,
    read_subproblem,
    write_container,
)
from giant_python.extraction.raster.solver_setup import SolverParams
from giant_python.models.params import SiloExecution, SiloParams
from giant_python.pipeline import silo, silo_baseline, silo_collect, silo_solve
from giant_python.pipeline.silo_session import (
    SiloSession,
    file_id,
    recording_ids,
    stage_hash,
)

from . import test_raster_silo_stages as stages
from .test_raster_silo_stages import (
    FRAMES,
    N_PATHS,
    N_TRIALS,
    H,
    W,
    _all_results,
    _equal,
    _seeds,
    _summary,
    fake_load,
    make_session,
    seeds_file,
)


@pytest.fixture
def fake(monkeypatch):
    """Synthetic localization and loading (as in the stages' tests)."""
    monkeypatch.setattr(trial_mod, "localize_trial", stages.fake_localize)
    monkeypatch.setattr(trial_data_mod, "load_trial", fake_load)


EX = SiloExecution(workers=1, solver_workers=1, solver_threads=1)
NET = SiloParams(baseline_source="network", min_kinetic_score=0.0)
MEDIAN = SiloParams(min_kinetic_score=0.0)


def _baseline_stack(d, t, n_frames, level=50.0):
    """A smooth baseline in movie counts, (frames, rows, cols), with a
    corner without values (NaN, as outside the valid pixels)."""
    r, c = np.mgrid[:H, :W]
    img = level * (1 + 0.2 * np.sin(r / 5.0 + d) * np.cos(c / 7.0 + t))
    ramp = 1 + 0.05 * np.linspace(-1, 1, n_frames)
    stack = img[None] * ramp[:, None, None]
    stack[:, :3, :3] = np.nan
    return stack.astype(np.float32)


def write_baselines(root: Path, folder: Path, level=50.0) -> Path:
    """Baseline files of every trial-path, laid out as slap2-f0 writes
    them: ``/baseline`` (frame, col, row) float16, with its attributes."""
    folder.mkdir(parents=True, exist_ok=True)
    moco = root / "motion_correction"
    for d in range(N_PATHS):
        for t in range(N_TRIALS):
            stem = f"E1T{t + 1}DMD{d + 1}"
            with h5py.File(moco / f"{stem}_ALIGNMENTDATA.h5", "r") as f:
                n = np.asarray(f["DSframes"]).size
            stack = _baseline_stack(d, t, n, level)
            with h5py.File(folder / f"{stem}_BASELINE-80Hz.h5", "w") as f:
                f["baseline"] = stack.transpose(0, 2, 1).astype(np.float16)
                f.attrs["checkpoint"] = "ckpt/s6_test.pt"
                f.attrs["step"] = 1700
                f.attrs["gain"] = 130.0
                f.attrs["n_frames"] = n
                f.attrs["source"] = f"{stem}_REGISTERED_DOWNSAMPLED-80Hz.tif"
    return folder


def test_params_baseline_source():
    """'network' sets the localization settings and N1; each can still be
    set; MATLAB's algorithm keeps its settings, hashes and params struct."""
    m = SiloParams()
    assert (m.peak_th, m.localization_baseline, m.baseline_mode) == (
        3.0,
        "median",
        "fit",
    )
    assert m.localization_surround and m.localization_null == "none"
    assert not m.needs_network_baseline
    n = SiloParams(baseline_source="network")
    assert n.peak_th == 8.0 and n.localization_baseline == "network"
    assert not n.localization_surround and n.localization_null == "bootstrap"
    assert n.localization_variance_floor
    assert n.baseline_mode == "gain" == n.solver_params().baseline_mode
    assert n.needs_network_baseline
    o = SiloParams(baseline_source="network", peak_th=5, baseline_mode="fixed")
    assert o.peak_th == 5 and o.baseline_mode == "fixed"
    assert SiloParams(**asdict(n)) == n
    assert SiloParams(**json.loads(json.dumps(asdict(n)))) == n
    # MATLAB's algorithm: no new fields in the hashes or the params struct
    assert "baseline_mode" not in m.stage_settings("solve")
    assert n.stage_settings("solve")["baseline_mode"] == "gain"
    pm = m.to_matlab(2, 80.0)
    assert "baselineSource" not in pm and "baselineMode" not in pm
    assert pm["peakth"] == 3.0
    pn = n.to_matlab(2, 80.0)
    assert pn["baselineSource"] == "network" and pn["baselineMode"] == "gain"
    assert pn["peakth"] == 8.0 and pn["localizationNull"] == "bootstrap"
    with pytest.raises(ValueError, match="baseline_source"):
        SiloParams(baseline_source="spline")
    with pytest.raises(ValueError, match="baseline_mode"):
        SiloParams(baseline_mode="guess")


def test_trial_baseline_is_map_baseline(tmp_path):
    """The pipeline's two steps (space and units at the registration's
    frames, then time and the fallback) give map_baseline's values."""
    rng = np.random.default_rng(3)
    n_t, n_r, n_c = 9, 20, 16
    stack = (40 + 10 * rng.random((n_t, n_r, n_c))).astype(np.float32)
    stack[:, :4, :] = np.nan  # rows without values
    stack[3, 10, 10] = np.nan  # one missing sample
    path = tmp_path / "E1T1DMD1_BASELINE-80Hz.h5"
    with h5py.File(path, "w") as f:
        f["baseline"] = stack.transpose(0, 2, 1).astype(np.float16)
        f.attrs["checkpoint"] = "x.pt"
    full, _ = read_network_baseline(path)
    rc = np.column_stack(
        [rng.uniform(-0.8, n_r - 0.2, 60), rng.uniform(2, 9, 60)]
    )
    rc[:5, 0] = rng.uniform(0, 2, 5)  # no network value: fallback
    ds_lines = np.linspace(1, 400, n_t)
    frame_lines = np.arange(1, 401, 3)
    data = 1.5 + rng.random((rc.shape[0], frame_lines.size))
    ref = map_baseline(
        full, rc, ds_lines, frame_lines, 25.0, fallback_data=data
    )
    tb = read_trial_baseline(path, rc, ds_lines, 25.0)
    assert tb.values.dtype == np.float32
    assert tb.identity["missing_pixels"] == int(ref.missing.sum()) > 0
    assert tb.identity["network"]["checkpoint"] == "x.pt"
    got = subproblem_baseline(tb.values, tb.lines, frame_lines, data)
    np.testing.assert_allclose(got, ref.values, rtol=2e-6)
    np.testing.assert_array_equal(tb.weight, ref.weight)
    # any subset of pixels: the same rows (the fallback is per pixel)
    rows = np.array([1, 4, 7, 30, 59])
    np.testing.assert_array_equal(
        subproblem_baseline(
            tb.values[rows], tb.lines, frame_lines, data[rows]
        ),
        got[rows],
    )


def test_container_holds_the_baseline(tmp_path):
    """A container stores the baseline at the registration's frames; its
    subproblems get it for a mode that uses it, from memory or the file."""
    trial = fake_load(
        "x_DMD1-TRIAL000001.dat",
        1,
        2000,
        None,
        np.ones((H, W), bool),
        None,
        None,
        None,
    )
    n_px = trial.y.shape[1]
    path = tmp_path / "E1T1DMD1_BASELINE-80Hz.h5"
    stack = _baseline_stack(0, 0, 6)
    with h5py.File(path, "w") as f:
        f["baseline"] = stack.transpose(0, 2, 1).astype(np.float16)
    ds_lines = np.linspace(1, FRAMES[0] * 8, 6)
    tb = read_trial_baseline(path, trial.trial_rc, ds_lines, 25.0)
    groups = [np.arange(0, 50), np.arange(40, n_px)]
    out = write_container(
        tmp_path / "c.h5",
        trial,
        np.array([[5, 1], [10, 10]]),
        groups,
        [[0], [1]],
        (H, W),
        baseline=tb,
    )
    sp = SolverParams(baseline_mode="gain")
    c = read_container(out)
    assert c.baseline_filled and c.baseline_values.dtype == np.float32
    np.testing.assert_array_equal(c.baseline_values, tb.values)
    for k, pix in enumerate(groups):
        expect = subproblem_baseline(
            tb.values[pix], tb.lines, trial.frame_lines, trial.y[0, pix], sp
        )
        a = c.subproblem(k, sp)
        b = read_subproblem(out, k, sp)
        np.testing.assert_array_equal(a.baseline, expect)
        np.testing.assert_array_equal(b.baseline, expect)
        assert np.isfinite(expect).all()  # NaN corner: the fallback
        # MATLAB's mode, or no solver parameters: no baseline is attached
        assert c.subproblem(k).baseline is None
        assert read_subproblem(out, k, SolverParams()).baseline is None
    # a container without the baseline cannot feed a mode that needs it
    bare = write_container(
        tmp_path / "bare.h5",
        trial,
        np.array([[5, 1]]),
        [groups[0]],
        [[0]],
        (H, W),
    )
    assert not read_container(bare).baseline_filled
    with pytest.raises(ValueError, match="no network baseline"):
        read_subproblem(bare, 0, sp)
    with pytest.raises(ValueError, match="no network baseline"):
        read_container(bare).subproblem(0, sp)


def test_network_streamed_and_staged_runs_agree(tmp_path, fake):
    """With the network's baseline (N1), the streamed run, one job and
    three jobs give the same results and summaries; the summaries record
    the baseline; the baseline changes the results."""
    root = tmp_path / "streamed"
    table = make_session(root)
    bdir = write_baselines(root, tmp_path / "baselines")
    seeds = seeds_file(tmp_path)
    silo.run_silo(table, NET, EX, seeds_file=seeds, baseline_dir=bdir)
    s1 = SiloSession.open(table)
    exp = s1.out_dir / "experiment_summary.h5"
    streamed = _all_results(s1)
    with h5py.File(exp, "r") as f:
        assert f.attrs["baseline_source"] == "network"
        assert f.attrs["baseline_mode"] == "gain"
        used = json.loads(f.attrs["network_baseline"])
    assert len(used) == N_PATHS * N_TRIALS
    first = used["path1_trial0001"]
    assert first["file"]["name"] == "E1T1DMD1_BASELINE-80Hz.h5"
    assert first["network"]["checkpoint"] == "ckpt/s6_test.pt"
    assert first["network"]["step"] == 1700
    with h5py.File(s1.localize_file(0, 0), "r") as f:
        assert f.attrs["localization_baseline"] == "network"
    with h5py.File(s1.plan_file(0), "r") as f:
        assert f.attrs["baseline_source"] == "network"

    for machines in (1, 3):
        r2 = tmp_path / f"staged{machines}"
        s2 = SiloSession.open(make_session(r2))
        silo.localize(s2, NET, EX, baseline_dir=bdir)
        silo.plan(s2, NET, EX, seeds=_seeds(s2, tmp_path))
        silo.prepare(s2, NET, EX, baseline_dir=bdir)
        c = read_container(s2.container_file(0, 0), data=False)
        assert c.baseline_filled and c.params["baseline_source"] == "network"
        for job in silo_solve.dispatch(s2, NET, machines, EX):
            silo_solve.solve_job(job, execution=EX)
        silo_collect.collect(s2, NET, EX, baseline_dir=bdir)
        staged = _all_results(s2)
        assert staged.keys() == streamed.keys()
        for key in staged:
            for name, value in staged[key].items():
                np.testing.assert_array_equal(value, streamed[key][name])
        _equal(_summary(s2.out_dir / "experiment_summary.h5"), _summary(exp))

    # the same run with MATLAB's baseline fit gives other results
    r3 = tmp_path / "median"
    silo.run_silo(make_session(r3), MEDIAN, EX, seeds_file=seeds)
    s3 = SiloSession.open(r3 / "trial_table.h5")
    median = _all_results(s3)
    assert median.keys() == streamed.keys()
    key = next(iter(median))
    assert not np.array_equal(median[key]["f0"], streamed[key]["f0"])
    with h5py.File(s3.out_dir / "experiment_summary.h5", "r") as f:
        assert "baseline_source" not in f.attrs  # as MATLAB writes it


def test_baseline_file_change_redoes_its_containers(tmp_path, fake):
    """The settings hash of a container covers its baseline file; MATLAB's
    algorithm keeps the hash it had before the network was wired in."""
    root = tmp_path / "s"
    s = SiloSession.open(make_session(root))
    bdir = write_baselines(
        root, root / "source_extraction" / "silo" / "baselines"
    )
    # localized with the moving median, extracted with the network: a new
    # baseline file redoes its container only (with network localization it
    # redoes the localization, hence the plan, first)
    ext = replace(
        NET,
        localization_baseline="median",
        localization_surround=True,
        localization_null="none",
    )
    silo.localize(s, ext, EX)
    silo.plan(s, ext, EX, seeds=_seeds(s, tmp_path))
    assert len(silo.prepare(s, ext, EX)) == N_PATHS * N_TRIALS
    assert silo.prepare(s, ext, EX) == []  # the default baselines folder
    f = bdir / "E1T2DMD1_BASELINE-80Hz.h5"
    with h5py.File(f, "a") as h:
        h["baseline"][0, 0, 0] = 1.0
    redone = silo.prepare(s, ext, EX)
    assert [r["unit"] for r in redone] == [(0, 1)]
    # MATLAB's algorithm: the hash of the stage files before this change
    ph = "abc"
    old = stage_hash(
        {
            "stage": "prepare",
            "settings": MEDIAN.stage_settings("prepare"),
            "plan": ph,
            "unit": [0, 0],
            "raw": recording_ids(s.raw(0, 0)),
            "lines": [
                float(s.first_line[0, 0]),
                float(s.last_line[0, 0]),
            ],
            "alignment": file_id(s.alignment(0, 0)),
        }
    )
    assert silo._prepare_hash(s, MEDIAN, ph, 0, 0) == old
    assert silo._prepare_hash(s, NET, ph, 0, 0, f) != old


def test_stages_refuse_another_baseline(tmp_path, fake):
    """Each stage checks that the stage before it used the same baseline."""
    root = tmp_path / "s"
    s = SiloSession.open(make_session(root))
    bdir = write_baselines(root, tmp_path / "b")
    silo.localize(s, NET, EX, baseline_dir=bdir)
    with pytest.raises(ValueError, match="localization_baseline='network'"):
        silo.plan(s, MEDIAN, EX, seeds=_seeds(s, tmp_path))
    silo.plan(s, NET, EX, seeds=_seeds(s, tmp_path))
    with pytest.raises(ValueError, match="baseline_source='network'"):
        silo.prepare(s, MEDIAN, EX)
    # network localization and plan, MATLAB's baseline fit: no baseline
    fit = replace(NET, baseline_mode="fit")
    silo.prepare(s, fit, EX, baseline_dir=bdir)
    assert not read_container(
        s.container_file(0, 0), data=False
    ).baseline_filled
    with pytest.raises(ValueError, match="has no network baseline"):
        silo_solve.dispatch(s, NET, 2, EX)
    with pytest.raises(ValueError, match="baseline_source='network'"):
        silo_solve.dispatch(s, MEDIAN, 2, EX)
    with pytest.raises(ValueError, match="baseline_source='network'"):
        silo_collect.collect(s, MEDIAN, EX)
    # a missing baseline file names itself and the ways to get it
    (bdir / "E1T1DMD2_BASELINE-80Hz.h5").unlink()
    with pytest.raises(FileNotFoundError, match="--f0-ckpt"):
        silo.prepare(s, NET, EX, baseline_dir=bdir)


def _args(*argv):
    """Parsed arguments of a ``giant`` command line."""
    return build_parser().parse_args(list(argv))


def test_cli_baseline_default():
    """--baseline-dir or --f0-ckpt make the network the default; --baseline
    chooses; otherwise MATLAB's moving median."""
    p = params_from_args(_args("silo", "t.h5"))
    assert p.baseline_source == "median" and p.peak_th == 3.0
    p = params_from_args(_args("silo", "t.h5", "--baseline-dir", "b"))
    assert p.baseline_source == "network" and p.peak_th == 8.0
    p = params_from_args(_args("silo", "t.h5", "--f0-ckpt", "x.pt"))
    assert p.baseline_source == "network"
    p = params_from_args(
        _args("silo", "t.h5", "--baseline-dir", "b", "--baseline", "median")
    )
    assert p.baseline_source == "median"
    p = params_from_args(_args("silo-plan", "t.h5", "--baseline", "network"))
    assert p.baseline_source == "network" and p.baseline_mode == "gain"
    p = params_from_args(
        _args("silo-prepare", "t.h5", "--baseline-dir", "b", "--peak-th", "6")
    )
    assert p.baseline_source == "network" and p.peak_th == 6.0
    p = params_from_args(
        _args(
            "silo-collect",
            "t.h5",
            "--set",
            "baseline_source=median",
            "--baseline-dir",
            "b",
        )
    )
    assert p.baseline_source == "median"
    a = _args("silo", "t.h5", "--f0-ckpt", "x.pt", "--f0-device", "cpu")
    assert a.f0_device == "cpu" and a.f0_command is None


def test_run_silo_default_source(tmp_path, fake, monkeypatch):
    """run_silo: a dict or no params take the network with its baselines;
    a SiloParams is used as it is (with a warning that they are unused)."""
    root = tmp_path / "s"
    table = make_session(root)
    bdir = write_baselines(root, tmp_path / "b")
    seen = []

    def stop(session, params, *args, **kw):
        """Record the params and stop the run."""
        seen.append(params)
        raise RuntimeError("stop")

    monkeypatch.setattr(silo, "_ready_paths", stop)
    for params in (None, {"min_kinetic_score": 0.0}):
        with pytest.raises(RuntimeError, match="stop"):
            silo.run_silo(table, params, EX, baseline_dir=bdir)
        assert seen[-1].baseline_source == "network"
    with pytest.warns(UserWarning, match="not used"):
        with pytest.raises(RuntimeError, match="stop"):
            silo.run_silo(table, MEDIAN, EX, baseline_dir=bdir)
    assert seen[-1].baseline_source == "median"
    assert silo.default_baseline_source(None, False) is None


def test_predict_baselines(tmp_path, monkeypatch):
    """--f0-ckpt runs slap2-f0 predict-registered once, for the trial-paths
    without a baseline file, with the run's trial preparation settings."""
    root = tmp_path / "s"
    s = SiloSession.open(make_session(root))
    bdir = tmp_path / "b"
    write_baselines(root, bdir)
    for t in (1, 2):
        (bdir / f"E1T{t + 1}DMD2_BASELINE-80Hz.h5").unlink()
    ckpt = tmp_path / "s6_test.pt"
    ckpt.write_bytes(b"weights")
    calls = []
    monkeypatch.setattr(
        silo_baseline.shutil, "which", lambda c: f"/bin/{Path(c).name}"
    )

    def fake_run(args, **kw):
        """slap2-f0 stood in for: write the files asked for."""
        calls.append(args)
        keys = args[args.index("--files") + 1].split(",")
        for k in keys:
            stem = k[: -len("_REGISTERED")]
            src = bdir / "E1T1DMD1_BASELINE-80Hz.h5"
            (bdir / f"{stem}_BASELINE-80Hz.h5").write_bytes(src.read_bytes())
            with h5py.File(bdir / f"{stem}_BASELINE-80Hz.h5", "a") as f:
                f.attrs["source"] = f"{stem}_REGISTERED_DOWNSAMPLED-80Hz.tif"
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(silo_baseline.subprocess, "run", fake_run)
    params = SiloParams(baseline_source="network", motion_thresh=1.5)
    rec = silo_baseline.predict_baselines(s, params, bdir, ckpt, device="cpu")
    assert rec["predicted"] == [[2, 2], [2, 3]]
    assert rec["checkpoint"]["name"] == "s6_test.pt"
    rec_ckpt = rec["checkpoint"]
    assert len(rec["checkpoint"]["sha256"]) == 64
    (args,) = calls
    assert args[:2] == ["/bin/slap2-f0", "predict-registered"]
    assert args[args.index("--ckpt") + 1] == str(ckpt)
    assert args[args.index("--out") + 1] == str(bdir)
    assert args[args.index("--files") + 1] == (
        "E1T2DMD2_REGISTERED,E1T3DMD2_REGISTERED"
    )
    assert args[args.index("--motion_thresh") + 1] == "1.5"
    assert args[args.index("--channel") + 1] == "1"
    assert args[args.index("--device") + 1] == "cpu"
    # nothing missing: nothing run
    calls.clear()
    assert (
        silo_baseline.predict_baselines(s, params, bdir, ckpt)["predicted"]
        == []
    )
    assert not calls

    # the predicted files have records: the prediction's settings, the
    # inputs and the checkpoint as it was then
    base = bdir / "E1T2DMD2_BASELINE-80Hz.h5"
    rec = json.loads(silo_baseline.record_path(base).read_text())
    assert rec["settings"]["motion_thresh"] == 1.5
    assert rec["checkpoint"] == rec_ckpt
    assert rec["movie"]["name"].startswith("E1T2DMD2_REGISTERED")
    imported = bdir / "E1T1DMD1_BASELINE-80Hz.h5"
    assert not silo_baseline.record_path(imported).exists()
    ckpt.write_bytes(b"other weights")  # overwritten after the prediction
    trial = fake_load(
        "x_DMD2-TRIAL000002.dat",
        1,
        2000,
        None,
        np.ones((H, W), bool),
        None,
        None,
        None,
    )
    tb = silo.load_trial_baseline(s, params, trial, base, 1, 1)
    assert tb.identity["checkpoint"] == rec_ckpt
    assert tb.identity["prediction"]["activity_channel"] == 1
    tb = silo.load_trial_baseline(s, params, trial, imported, 0, 0)
    assert "checkpoint" not in tb.identity  # not verified: path and step only
    assert tb.identity["network"]["step"] == 1700
    # a baseline predicted with other settings is refused, not reused
    other = replace(params, activity_channel=2)
    with pytest.raises(ValueError, match="was predicted with"):
        silo_baseline.predict_baselines(s, other, bdir, ckpt)
    with pytest.raises(ValueError, match="was predicted with"):
        silo.localize(s, other, EX, baseline_dir=bdir)
    # a record of a file that was replaced since
    with h5py.File(base, "a") as f:
        f.attrs["step"] = 1800
    with pytest.raises(ValueError, match="does not describe"):
        silo.localize(s, params, EX, baseline_dir=bdir)
    silo_baseline.record_path(base).unlink()
    ckpt.write_bytes(b"weights")

    # a failed or incomplete prediction is an error
    (bdir / "E1T3DMD1_BASELINE-80Hz.h5").unlink()
    # a trial with both movies, the newer not SILo's: refused before the GPU
    moco = root / "motion_correction"
    tif = moco / "E1T3DMD1_REGISTERED_DOWNSAMPLED-80Hz.tif"
    h5 = tif.with_suffix(".h5")
    h5.write_bytes(b"")
    os.utime(h5, (tif.stat().st_mtime + 10, tif.stat().st_mtime + 10))
    with pytest.raises(ValueError, match="remove one of them"):
        silo_baseline.predict_baselines(s, params, bdir, ckpt)
    assert silo_baseline.f0_movie(moco, "E1T3DMD1") == h5
    h5.unlink()
    monkeypatch.setattr(
        silo_baseline.subprocess,
        "run",
        lambda args, **kw: subprocess.CompletedProcess(args, 3),
    )
    with pytest.raises(RuntimeError, match="exited with 3"):
        silo_baseline.predict_baselines(s, params, bdir, ckpt)

    # a run that writes a file and then fails: the file has its record, so
    # a retry with other settings does not reuse it unchecked
    def fail_after_one(args, **kw):
        """Write the files, then fail."""
        fake_run(args, **kw)
        return subprocess.CompletedProcess(args, 1)

    monkeypatch.setattr(silo_baseline.subprocess, "run", fail_after_one)
    with pytest.raises(RuntimeError, match="exited with 1"):
        silo_baseline.predict_baselines(s, params, bdir, ckpt)
    written = bdir / "E1T3DMD1_BASELINE-80Hz.h5"
    rec = json.loads(silo_baseline.record_path(written).read_text())
    assert rec["baseline"]["name"] == written.name
    with pytest.raises(ValueError, match="was predicted with"):
        silo_baseline.predict_baselines(s, other, bdir, ckpt)
    # stopped before the record got the file's identity: its settings hold
    rec["baseline"] = None
    silo_baseline.record_path(written).write_text(json.dumps(rec))
    args = (written, s.movie(0, 2), s.alignment(0, 2))
    assert silo_baseline.check_baseline_record(*args, params) is not None
    with pytest.raises(ValueError, match="was predicted with"):
        silo_baseline.check_baseline_record(*args, other)
    written.unlink()
    monkeypatch.setattr(
        silo_baseline.subprocess,
        "run",
        lambda args, **kw: subprocess.CompletedProcess(args, 0),
    )
    with pytest.raises(RuntimeError, match="E1T3DMD1"):
        silo_baseline.predict_baselines(s, params, bdir, ckpt)
    monkeypatch.setattr(silo_baseline.shutil, "which", lambda c: None)
    with pytest.raises(FileNotFoundError, match="--f0-command"):
        silo_baseline.predict_baselines(s, params, bdir, ckpt)
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        silo_baseline.predict_baselines(s, params, bdir, tmp_path / "x.pt")


def test_source_alone_redoes_plan_and_containers(tmp_path, fake):
    """baseline_source enters the plan and prepare hashes (when it is not
    MATLAB's): changing it alone redoes those stages rather than reusing
    files that the next stage would refuse."""
    alt = SiloParams(
        baseline_source="median",
        localization_baseline="network",
        localization_surround=False,
        localization_null="bootstrap",
        peak_th=8.0,
        baseline_mode="gain",
        min_kinetic_score=0.0,
    )
    for stage in ("localize", "solve", "collect"):
        assert alt.stage_settings(stage) == NET.stage_settings(stage)
    for stage in ("plan", "prepare"):
        assert alt.stage_settings(stage) != NET.stage_settings(stage)
        assert "baseline_source" not in MEDIAN.stage_settings(stage)
    root = tmp_path / "s"
    s = SiloSession.open(make_session(root))
    bdir = write_baselines(root, tmp_path / "b")
    silo.localize(s, NET, EX, baseline_dir=bdir)
    silo.plan(s, NET, EX, seeds=_seeds(s, tmp_path))
    silo.prepare(s, NET, EX, baseline_dir=bdir)
    silo.plan(s, alt, EX, seeds=_seeds(s, tmp_path))  # redone, not reused
    with h5py.File(s.plan_file(0), "r") as f:
        assert f.attrs["baseline_source"] == "median"
    assert len(silo.prepare(s, alt, EX, baseline_dir=bdir)) == 6
    with pytest.raises(ValueError, match="baseline_source='median'"):
        silo.prepare(s, NET, EX, baseline_dir=bdir)


def test_unrecorded_network_localization_is_redone(tmp_path, fake):
    """A localization file without the record of its baseline (made before
    it was recorded) counts as the moving median's: with the network's
    settings it is localized again, with MATLAB's it is kept."""
    root = tmp_path / "s"
    s = SiloSession.open(make_session(root))
    bdir = write_baselines(root, tmp_path / "b")
    silo.localize(s, NET, EX, baseline_dir=bdir)
    assert silo.localize_jobs(s, NET, baseline_dir=bdir) == []
    with h5py.File(s.localize_file(1, 2), "a") as f:
        del f.attrs["localization_baseline"]
    jobs = silo.localize_jobs(s, NET, baseline_dir=bdir)
    assert [j["unit"] for j in jobs] == [(1, 2)]
    m = SiloSession.open(make_session(tmp_path / "m"))
    silo.localize(m, MEDIAN, EX)
    with h5py.File(m.localize_file(0, 1), "a") as f:
        del f.attrs["localization_baseline"]
    assert silo.localize_jobs(m, MEDIAN) == []


def test_summary_records_the_baseline_of_the_results_used(tmp_path, fake):
    """The summary's network_baseline comes from the results files that
    are assembled, not from earlier runs' files left in the folder."""
    root = tmp_path / "s"
    s = SiloSession.open(make_session(root))
    bdir = write_baselines(root, tmp_path / "b")
    silo.localize(s, NET, EX, baseline_dir=bdir)
    silo.plan(s, NET, EX, seeds=_seeds(s, tmp_path))
    silo.prepare(s, NET, EX, baseline_dir=bdir)
    for job in silo_solve.dispatch(s, NET, 3, EX):
        silo_solve.solve_job(job, execution=EX)
    shard = sorted(s.results_dir.glob("*_s*.h5"))[0]
    stem = shard.name.split("_s")[0]
    stale = s.results_dir / f"{stem}.h5"  # sorts before the shards
    assert not stale.exists()
    stale.write_bytes(shard.read_bytes())
    with h5py.File(stale, "a") as f:
        f.attrs["plan_hash"] = "an earlier plan"
        f.attrs["network_baseline"] = json.dumps({"file": {"name": "old"}})
    with pytest.warns(UserWarning, match="left out"):
        record = silo_collect.collect(s, NET, EX, baseline_dir=bdir)
    used = record["network_baseline"][stem]
    assert used["file"]["name"].endswith("_BASELINE-80Hz.h5")
    with h5py.File(s.out_dir / "experiment_summary.h5", "r") as f:
        assert json.loads(f.attrs["network_baseline"])[stem] == used


def test_baseline_from_another_movie_is_refused(tmp_path, fake):
    """A baseline file predicted from another movie than SILo reads (its
    'source') is refused."""
    root = tmp_path / "s"
    s = SiloSession.open(make_session(root))
    bdir = write_baselines(root, tmp_path / "b")
    with h5py.File(bdir / "E1T2DMD1_BASELINE-80Hz.h5", "a") as f:
        f.attrs["source"] = "E1T2DMD1_REGISTERED_DOWNSAMPLED-80Hz.h5"
    with pytest.raises(ValueError, match="was predicted from"):
        silo.localize(s, NET, EX, baseline_dir=bdir)


def test_collect_checks_baselines_without_movies(tmp_path, fake):
    """A collect machine without the registered movies still checks the
    results against the baseline files (named after the trial table's
    movies): a replaced baseline leaves the results stale."""
    root = tmp_path / "s"
    table = make_session(root)
    bdir = write_baselines(root, tmp_path / "b")
    silo.run_silo(
        table, NET, EX, seeds_file=seeds_file(tmp_path), baseline_dir=bdir
    )
    for movie in (root / "motion_correction").glob("*_REGISTERED_*"):
        movie.unlink()
    s = SiloSession.open(table, inputs="collect")
    assert s.movie(0, 0) is None
    silo_collect.collect(s, NET, EX, baseline_dir=bdir)
    with h5py.File(bdir / "E1T1DMD2_BASELINE-80Hz.h5", "a") as f:
        f["baseline"][0, 5, 5] = 1.0
    with pytest.warns(UserWarning, match="left out"):
        with pytest.raises(FileNotFoundError, match="path 2, trial 1"):
            silo_collect.collect(s, NET, EX, baseline_dir=bdir)


def test_localize_predicts_for_extraction_too(tmp_path, fake, monkeypatch):
    """silo-localize --f0-ckpt predicts whenever a stage uses the network's
    baselines, also when localization alone uses the moving median."""
    table = make_session(tmp_path / "s")
    calls = []
    monkeypatch.setattr(
        silo_baseline,
        "predict_baselines",
        lambda *a, **k: calls.append(a[1]) or {"predicted": []},
    )
    monkeypatch.setattr(silo, "localize", lambda *a, **k: [])
    argv = ["silo-localize", str(table), "--f0-ckpt", "x.pt"]
    assert main(argv + ["--localization-baseline", "median"]) == 0
    assert len(calls) == 1 and calls[0].baseline_mode == "gain"
    assert main(argv + ["--baseline", "median"]) == 0
    assert len(calls) == 1


def test_extraction_uses_the_localization_baseline(tmp_path, fake):
    """Extraction refuses another baseline file than the trial was
    localized with (stages given different folders)."""
    import shutil

    root = tmp_path / "s"
    s = SiloSession.open(make_session(root))
    a = write_baselines(root, tmp_path / "a")
    b = tmp_path / "b"
    # the same baselines, other files (new modification times)
    shutil.copytree(a, b, copy_function=shutil.copy)
    silo.localize(s, NET, EX, baseline_dir=a)
    silo.plan(s, NET, EX, seeds=_seeds(s, tmp_path))
    with pytest.raises(ValueError, match="localized with another baseline"):
        silo.prepare(s, NET, EX, baseline_dir=b)
    assert len(silo.prepare(s, NET, EX, baseline_dir=a)) == 6


def test_predict_units_may_be_a_generator(tmp_path, monkeypatch):
    """The trial-paths to predict may be given as a generator."""
    root = tmp_path / "s"
    s = SiloSession.open(make_session(root))
    bdir = write_baselines(root, tmp_path / "b")
    (bdir / "E1T2DMD2_BASELINE-80Hz.h5").unlink()
    ckpt = tmp_path / "s6_test.pt"
    ckpt.write_bytes(b"weights")
    monkeypatch.setattr(
        silo_baseline.shutil, "which", lambda c: f"/bin/{Path(c).name}"
    )

    rec_args = []

    def fake_run(args, **kw):
        """Write the one file asked for."""
        rec_args.append(args)
        src = bdir / "E1T1DMD1_BASELINE-80Hz.h5"
        dest = bdir / "E1T2DMD2_BASELINE-80Hz.h5"
        dest.write_bytes(src.read_bytes())
        with h5py.File(dest, "a") as f:
            f.attrs["source"] = "E1T2DMD2_REGISTERED_DOWNSAMPLED-80Hz.tif"
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(silo_baseline.subprocess, "run", fake_run)
    units = (u for u in [(0, 0), (1, 1)])
    rec = silo_baseline.predict_baselines(
        s, NET, bdir, ckpt, units, out_frames=512
    )
    assert rec["predicted"] == [[2, 2]]
    args = rec_args[-1]
    assert args[args.index("--out_frames") + 1] == "512"
    assert _args("silo", "t.h5", "--f0-out-frames", "256").f0_out_frames == 256


def test_summary_records_the_localization_baseline(tmp_path, fake):
    """Localized with the network's baseline and extracted with MATLAB's
    fit: the summary records the baseline that decided the seeds."""
    root = tmp_path / "s"
    table = make_session(root)
    bdir = write_baselines(root, tmp_path / "b")
    loc_only = replace(NET, baseline_mode="fit")
    silo.run_silo(
        table, loc_only, EX, seeds_file=seeds_file(tmp_path), baseline_dir=bdir
    )
    s = SiloSession.open(table)
    with h5py.File(s.out_dir / "experiment_summary.h5", "r") as f:
        assert f.attrs["baseline_mode"] == "fit"
        assert "network_baseline" not in f.attrs  # not extracted with it
        loc = json.loads(f.attrs["localization_network_baseline"])
    assert len(loc) == N_PATHS * N_TRIALS
    first = loc["path1_trial0001"]
    assert first["file"]["name"] == "E1T1DMD1_BASELINE-80Hz.h5"
    assert first["network"] == {"checkpoint": "ckpt/s6_test.pt", "step": 1700}


def test_localized_again_after_the_plan(tmp_path, fake):
    """Trials localized again (here with other baseline files) after their
    plan was made: prepare and collect refuse the stale plan and results
    instead of reading the new localization's peaks and baselines."""
    import shutil

    root = tmp_path / "s"
    s = SiloSession.open(make_session(root))
    a = write_baselines(root, tmp_path / "a")
    b = tmp_path / "b"
    shutil.copytree(a, b, copy_function=shutil.copy)
    silo.localize(s, NET, EX, baseline_dir=a)
    silo.plan(s, NET, EX, seeds=_seeds(s, tmp_path))
    silo.prepare(s, NET, EX, baseline_dir=a)
    for job in silo_solve.dispatch(s, NET, 1, EX):
        silo_solve.solve_job(job, execution=EX)
    silo_collect.collect(s, NET, EX, baseline_dir=a)
    silo.localize(s, NET, EX, baseline_dir=b)  # every trial again
    with pytest.raises(ValueError, match="localized again since its plan"):
        silo_collect.collect(s, NET, EX, baseline_dir=a)
    with pytest.raises(ValueError, match="localized again since its plan"):
        silo.prepare(s, NET, EX, baseline_dir=b)
    silo.plan(s, NET, EX, seeds=_seeds(s, tmp_path))  # made again: fine
    assert len(silo.prepare(s, NET, EX, baseline_dir=b)) == 6


def test_one_baseline_mode_per_run(tmp_path, fake):
    """Containers prepared with the network's baseline are not solved with
    MATLAB's fit (their hash covers a baseline the fit does not use, so
    collect would take the results for stale); results name the baseline
    only when the solver used it."""
    root = tmp_path / "s"
    s = SiloSession.open(make_session(root))
    bdir = write_baselines(root, tmp_path / "b")
    silo.localize(s, NET, EX, baseline_dir=bdir)
    silo.plan(s, NET, EX, seeds=_seeds(s, tmp_path))
    silo.prepare(s, NET, EX, baseline_dir=bdir)
    fit = replace(NET, baseline_mode="fit")
    with pytest.raises(ValueError, match="baseline_mode 'fit'"):
        silo_solve.dispatch(s, fit, 1, EX)
    job = silo_solve.dispatch(s, NET, 1, EX)[0]
    with pytest.raises(ValueError, match="baseline_mode 'fit'"):
        silo_solve.solve_container(
            s.container_file(0, 0),
            s.results_file(0, 0),
            fit,
            EX.solver_options(),
        )
    silo_solve.solve_job(job, execution=EX)
    with h5py.File(s.results_file(0, 0), "r") as h:
        assert "network_baseline" in h.attrs
    c = read_container(s.container_file(0, 0), data=False)
    attrs = silo_solve.results_attrs(
        c, "c.h5", "h", (0, 1), 1, True, "r", fit, EX.solver_options()
    )
    assert "network_baseline" not in attrs


def test_collect_checks_the_localization_baseline(tmp_path, fake):
    """collect refuses settings whose localization baseline is not the one
    the trials were localized with (the summary would mislabel them)."""
    root = tmp_path / "s"
    table = make_session(root)
    bdir = write_baselines(root, tmp_path / "b")
    silo.run_silo(
        table, NET, EX, seeds_file=seeds_file(tmp_path), baseline_dir=bdir
    )
    s = SiloSession.open(table)
    median_loc = replace(
        NET,
        localization_baseline="median",
        localization_surround=True,
        localization_null="none",
    )
    assert median_loc.stage_settings("solve") == NET.stage_settings("solve")
    with pytest.raises(ValueError, match="localization_baseline='network'"):
        silo_collect.collect(s, median_loc, EX, baseline_dir=bdir)


def test_one_checkpoint_per_run(tmp_path, monkeypatch):
    """Baselines kept from an earlier prediction must have been made with
    the given checkpoint (its SHA-256, not only its name)."""
    root = tmp_path / "s"
    s = SiloSession.open(make_session(root))
    bdir = write_baselines(root, tmp_path / "b")
    (bdir / "E1T2DMD2_BASELINE-80Hz.h5").unlink()
    ckpt = tmp_path / "s6_test.pt"
    ckpt.write_bytes(b"weights")
    monkeypatch.setattr(
        silo_baseline.shutil, "which", lambda c: f"/bin/{Path(c).name}"
    )

    def fake_run(args, **kw):
        """Write the one file asked for."""
        src = bdir / "E1T1DMD1_BASELINE-80Hz.h5"
        dest = bdir / "E1T2DMD2_BASELINE-80Hz.h5"
        dest.write_bytes(src.read_bytes())
        with h5py.File(dest, "a") as f:
            f.attrs["source"] = "E1T2DMD2_REGISTERED_DOWNSAMPLED-80Hz.tif"
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(silo_baseline.subprocess, "run", fake_run)
    silo_baseline.predict_baselines(s, NET, bdir, ckpt)
    assert (
        silo_baseline.predict_baselines(s, NET, bdir, ckpt)["predicted"] == []
    )
    ckpt.write_bytes(b"retrained weights")  # same name, other model
    with pytest.raises(ValueError, match="another checkpoint"):
        silo_baseline.predict_baselines(s, NET, bdir, ckpt)
