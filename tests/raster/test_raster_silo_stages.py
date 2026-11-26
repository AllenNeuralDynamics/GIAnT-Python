"""SILo's stages and driver on a small synthetic session.

The registered movies and the raw data are stood in for: localization and
trial loading are replaced by functions that return synthetic images and
200 Hz data (with known sources), so that the stages' plumbing (files,
settings hashes, skipping, shards, assembly, summaries) is tested with the
real plan stage, solver, kinetics test and writers.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

from giant_python.cli import build_parser, main
from giant_python.cli_silo import params_from_args, parse_numbers
from giant_python.extraction.raster import trial as trial_mod
from giant_python.extraction.raster import trial_data as trial_data_mod
from giant_python.extraction.raster.localization import RasterLocalization
from giant_python.extraction.raster.results import read_results
from giant_python.extraction.raster.trial import RegisteredTrial
from giant_python.extraction.raster.trial_data import TrialData
from giant_python.io.hdf5 import load_struct_h5, save_struct_h5
from giant_python.models.params import SiloExecution, SiloParams
from giant_python.pipeline import silo, silo_collect, silo_solve
from giant_python.pipeline.silo_session import SiloSession

H, W = 24, 30
N_PATHS, N_TRIALS = 2, 3
SOURCES = np.array([[6, 7], [8, 9], [15, 20], [18, 8]])
FRAMES = {0: 260, 1: 240, 2: 60}  # trial 3 is too short to extract
SHIFTS = {0: (0, 0), 1: (1, -1), 2: (0, 1)}


def _blobs(amplitude, sigma=1.3, shift=(0, 0)):
    """Gaussian blobs at the sources (moved into a trial's frame)."""
    r, c = np.mgrid[:H, :W]
    img = np.zeros((H, W))
    for (sr, sc), a in zip(SOURCES, amplitude):
        img += a * np.exp(
            -((r - sr - shift[0]) ** 2 + (c - sc - shift[1]) ** 2)
            / (2 * sigma**2)
        )
    return img


def fake_localize(moco_dir, fn, num_channels, params=None, **kw):
    """Images of a localized trial (instead of reading its movie)."""
    t = int(fn.split("T")[1][0]) - 1
    rng = np.random.default_rng(t)
    shift = SHIFTS[t]
    act = _blobs([30, 25, 40, 20], shift=shift) + rng.random((H, W))
    mean = np.stack(
        [_blobs([5, 5, 5, 5], 3, shift) + 10, np.full((H, W), 4.0)], 2
    )
    loc = RasterLocalization(
        act_im=act,
        summary=act,
        peaks=None,
        valid=np.ones((H, W), bool),
    )
    reg = RegisteredTrial(
        movie=np.zeros((1, H, W), np.float32),
        var_factor=None,
        mean_im=mean.astype(np.float32),
        discard=np.zeros(50, bool),
        align_hz=80.0,
    )
    return loc, reg


def fake_load(
    dat_path,
    first_line,
    last_line,
    alignment,
    sel_pix,
    warp,
    discard_frames,
    mean_image,
    roi_masks=(),
    params=None,
    device="cpu",
    dtype=np.float32,
):
    """A trial at the selected pixels: baseline plus sources with events."""
    name = Path(dat_path).name
    t = int(name.split("TRIAL")[1][:6]) - 1
    d = int(name.split("DMD")[1][0]) - 1
    n = FRAMES[t]
    rng = np.random.default_rng(100 * d + t)
    cols, rows = np.nonzero(np.asarray(sel_pix).T)
    y = np.full((2, rows.size, n), 2.0)
    k = np.exp(-np.arange(30) / 4.8)
    for sr, sc in SOURCES:
        fp = np.exp(-((rows - sr) ** 2 + (cols - sc) ** 2) / (2 * 1.3**2))
        ev = (rng.random(n) < 0.03) * rng.uniform(2, 6, n)
        trace = np.convolve(ev, k)[:n]
        y[0] += fp[:, None] * trace[None]
    y = rng.poisson(y * 25) / 25.0
    return TrialData(
        y=y.astype(dtype),
        finv=np.ones((rows.size, n), dtype),
        frame_lines=np.arange(n) * 8 + 1 + 1000 * t,
        motion_r=np.zeros(n),
        motion_c=np.zeros(n),
        discard=np.arange(n) % 50 == 0,
        global_f=rng.random((2, n)),
        roi_f=rng.random((len(roi_masks), 2, n)),
        roi_fsvd=rng.random((len(roi_masks), 2, n)),
        pixel_rc=np.stack([rows, cols], 1),
        trial_rc=np.stack([rows, cols], 1).astype(float),
        photon_scale=25.0,
    )


def make_session(root: Path) -> Path:
    """Trial table, alignment files and empty movies and raw files."""
    moco, raw = root / "motion_correction", root / "raw"
    moco.mkdir(parents=True)
    raw.mkdir()
    fn = np.empty((N_PATHS, N_TRIALS), object)
    reg = np.empty_like(fn)
    adata = np.empty_like(fn)
    for d in range(N_PATHS):
        for t in range(N_TRIALS):
            fn[d, t] = f"x_DMD{d + 1}-TRIAL{t + 1:06d}.dat"
            stem = f"E1T{t + 1}DMD{d + 1}"
            reg[d, t] = f"{stem}_REGISTERED_DOWNSAMPLED-80Hz.tif"
            adata[d, t] = f"{stem}_ALIGNMENTDATA.h5"
            (raw / fn[d, t]).write_bytes(b"")
            (moco / reg[d, t]).write_bytes(b"")
            n_ds = 6
            save_struct_h5(
                moco / adata[d, t],
                {
                    "numChannels": 2.0,
                    "alignHz": 80.0,
                    "DSframes": np.linspace(1, FRAMES[t] * 8, n_ds) + 1000 * t,
                    "motionDSr": np.linspace(0, 0.5, n_ds),
                    "motionDSc": np.linspace(0.2, 0, n_ds),
                    "slap2": {
                        "Z_depths": np.float32(-20.0),
                        "onlineMotionXshift": np.arange(n_ds, dtype=float),
                        "onlineMotionYshift": np.zeros(n_ds),
                        "onlineMotionZshift": np.zeros(n_ds),
                    },
                },
            )
    save_struct_h5(
        root / "trial_table.h5",
        {
            "datadr": str(raw),
            "savedr": str(root),
            "filename": fn,
            "slap2_info": {
                "first_line": np.ones((N_PATHS, N_TRIALS)),
                "last_line": np.full((N_PATHS, N_TRIALS), 2000.0),
            },
            "motion_correction": {
                "fn_reg_ds": reg,
                "fn_adata": adata,
                "registration_failed": np.zeros((N_PATHS, N_TRIALS), bool),
            },
        },
    )
    return root / "trial_table.h5"


@pytest.fixture
def fake(monkeypatch):
    """Synthetic localization and loading, in this process."""
    monkeypatch.setattr(trial_mod, "localize_trial", fake_localize)
    monkeypatch.setattr(trial_data_mod, "load_trial", fake_load)


EX = SiloExecution(workers=1, solver_workers=1, solver_threads=1)
PARAMS = SiloParams(min_kinetic_score=0.0)


def seeds_file(root: Path) -> Path:
    """The sources as seeds of both paths (zero-based); the plan's own
    seeding is tested with the plan stage."""
    from giant_python.extraction.raster.stage_files import write_seeds

    seeds = np.column_stack(
        [np.full(len(SOURCES), 10.0), SOURCES, np.full(len(SOURCES), 1.3)]
    ).astype(float)
    path = root / "seeds.h5"
    write_seeds(path, {1: seeds, 2: seeds}, index_base=0)
    return path


def _seeds(session, root):
    """The seeds of :func:`seeds_file` per path."""
    from giant_python.extraction.raster.stage_files import read_seeds

    return read_seeds(seeds_file(root), session.n_paths)


def _data(path):
    """The solved arrays of a results file, by subproblem."""
    r = read_results(path)
    return {
        k: {
            n: getattr(v[2], n)
            for n in ("footprints", "events", "ls", "f0", "snr")
        }
        for k, v in r.results.items()
    }


def _all_results(session):
    """Every subproblem's arrays, keyed by (path, trial, subproblem)."""
    out = {}
    for f in sorted(session.results_dir.glob("*.h5")):
        r = read_results(f)
        for k, v in _data(f).items():
            out[(r.attrs["path"], r.attrs["trial"], k)] = v
    return out


def _summary(path):
    """A summary file's datasets (params' times left out)."""
    s = load_struct_h5(path)
    s.pop("params", None)
    return s


def _equal(a, b):
    """Nested dicts of arrays are equal (NaN equal)."""
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], dict):
            _equal(a[k], b[k])
        else:
            np.testing.assert_array_equal(a[k], b[k])


def test_streamed_and_staged_runs_agree(tmp_path, fake):
    """The streaming driver, one job and three jobs give the same results
    and summaries; reruns keep finished units."""
    table = make_session(tmp_path / "streamed")
    seeds = seeds_file(tmp_path)
    report = silo.run_silo(table, PARAMS, EX, seeds_file=seeds)
    s1 = SiloSession.open(table)
    assert s1.keep.all() and s1.num_channels == 2 and s1.align_hz == 80.0
    exp = s1.out_dir / "experiment_summary.h5"
    assert exp.exists() and (s1.out_dir / "per_trial_summary.h5").exists()
    assert report["total"] > 0
    # timings per trial-path: localization, and each trial's solve span
    assert len(report["localize_trials"]) == N_PATHS * N_TRIALS
    assert all(r["seconds"] > 0 for r in report["localize_trials"])
    loaded = [r for r in report["trials"] if "load" in r]
    assert loaded and all(
        0 <= r["submitted"] <= r["solved"] <= report["total"] for r in loaded
    )
    streamed = _all_results(s1)
    # trial 3 is short: its results file has its record and no subproblems
    short = read_results(s1.results_file(0, 2))
    assert short.attrs["extracted"] == 0 and not short.results
    table_after = load_struct_h5(table)
    assert table_after["source_extraction"]["fn_raw"][0, 0].endswith(".dat")
    assert table_after["source_extraction"]["analysis_params"]["peakth"] == 3

    # rerun: everything is up to date
    mtime = s1.results_file(0, 0).stat().st_mtime_ns
    again = silo.run_silo(table, PARAMS, EX, seeds_file=seeds)
    assert again["localize_units"] == 0
    assert all("skipped" in t for t in again["trials"])
    assert s1.results_file(0, 0).stat().st_mtime_ns == mtime

    for machines in (1, 3):
        root = tmp_path / f"staged{machines}"
        s2 = SiloSession.open(make_session(root))
        silo.localize(s2, PARAMS, EX)
        silo.plan(s2, PARAMS, EX, seeds=_seeds(s2, tmp_path))
        prepared = silo.prepare(s2, PARAMS, EX)
        assert len(prepared) == N_PATHS * N_TRIALS
        jobs = silo_solve.dispatch(s2, PARAMS, machines, EX)
        assert len(jobs) == machines
        for job in jobs:
            silo_solve.solve_job(job, execution=EX)
        silo_collect.collect(s2, PARAMS, EX)
        staged = _all_results(s2)
        assert staged.keys() == streamed.keys()
        for key in staged:
            for name, value in staged[key].items():
                np.testing.assert_array_equal(value, streamed[key][name])
        _equal(_summary(s2.out_dir / "experiment_summary.h5"), _summary(exp))
        _equal(
            _summary(s2.out_dir / "per_trial_summary.h5"),
            _summary(s1.out_dir / "per_trial_summary.h5"),
        )
    many = list(
        (tmp_path / "staged3" / "source_extraction" / "silo" / "results").glob(
            "*_s*.h5"
        )
    )
    assert many, "three jobs split a container into ranges"


def test_settings_change_redoes_units(tmp_path, fake):
    """A changed solver setting redoes the solve stage only."""
    s = SiloSession.open(make_session(tmp_path))
    silo.localize(s, PARAMS, EX)
    silo.plan(s, PARAMS, EX, seeds=_seeds(s, tmp_path))
    silo.prepare(s, PARAMS, EX)
    silo_solve.solve_containers(s, PARAMS, EX)
    assert silo.localize(s, PARAMS, EX) == []
    assert silo.prepare(s, PARAMS, EX) == []
    done = silo_solve.solve_containers(s, PARAMS, EX)
    assert all(v is None for v in done.values())
    other = SiloParams(min_kinetic_score=0.0, lam=0.5)
    assert silo.prepare(s, other, EX) == []
    done = silo_solve.solve_containers(s, other, EX)
    assert all(v is not None for v in done.values())


def test_inputs_and_stale_results(tmp_path, fake, monkeypatch):
    """A changed raw file redoes prepare; collect uses only results of the
    current plan and solve settings; a trial found without data loses its
    earlier container and results."""
    import os

    from giant_python.pipeline.silo_session import NO_DATA

    s = SiloSession.open(make_session(tmp_path))
    silo.localize(s, PARAMS, EX)
    silo.plan(s, PARAMS, EX, seeds=_seeds(s, tmp_path))
    silo.prepare(s, PARAMS, EX)
    assert silo.prepare(s, PARAMS, EX) == []
    raw = s.raw(0, 0)
    os.utime(raw, ns=(raw.stat().st_atime_ns, raw.stat().st_mtime_ns + 10**9))
    assert [u["unit"] for u in silo.prepare(s, PARAMS, EX)] == [(0, 0)]

    # results of three jobs with lam 0.9, then whole files with lam 0.5
    for job in silo_solve.dispatch(s, PARAMS, 3, EX):
        silo_solve.solve_job(job, execution=EX)
    split = list(s.results_dir.glob("*_s*.h5"))
    assert split
    other = SiloParams(min_kinetic_score=0.0, lam=0.5)
    silo_solve.solve_containers(s, other, EX)
    # the three jobs' range files are from other settings: left out
    with pytest.warns(UserWarning, match="left out"):
        silo_collect.collect(s, other, EX)
    third = SiloParams(min_kinetic_score=0.0, lam=0.3)
    with pytest.raises(FileNotFoundError, match="current"):
        with pytest.warns(UserWarning):
            silo_collect.collect(s, third, EX)

    # the trial has no data any more: its container and results go
    def no_data(dat_path, *args, **kw):
        """Path 1, trial 1 has no data now."""
        if "DMD1-TRIAL000001" in str(dat_path):
            raise ValueError(NO_DATA)
        return fake_load(dat_path, *args, **kw)

    monkeypatch.setattr(trial_data_mod, "load_trial", no_data)
    ex = SiloExecution(
        workers=1, solver_workers=1, solver_threads=1, overwrite=True
    )
    silo.prepare(s, other, ex, units=[(0, 0)])
    assert s.skip_file(0, 0).exists()
    assert not s.container_file(0, 0).exists()
    assert not list(s.results_dir.glob("path1_trial0001*.h5"))
    assert all(u != (0, 0) for u, _ in silo_solve._containers(s))
    silo_collect.collect(s, other, EX)
    e = load_struct_h5(s.out_dir / "experiment_summary.h5")
    frames = e["Path1"]["frame_info"]["trial_num_frames"].ravel()
    assert frames.tolist() == [0, FRAMES[1], FRAMES[2]]

    # a raw file changed after solving: its results are stale
    raw = s.raw(1, 1)
    os.utime(raw, ns=(raw.stat().st_atime_ns, raw.stat().st_mtime_ns + 10**9))
    with pytest.raises(FileNotFoundError, match="current"):
        with pytest.warns(UserWarning, match="left out"):
            silo_collect.collect(s, other, EX)


def test_collect_and_dispatch_without_raw_data(tmp_path, fake):
    """Dispatch and collect need neither the raw data nor the movies; a
    short trial is one item of no work."""
    import json
    import shutil

    table = make_session(tmp_path)
    s = SiloSession.open(table)
    silo.localize(s, PARAMS, EX)
    silo.plan(s, PARAMS, EX, seeds=_seeds(s, tmp_path))
    silo.prepare(s, PARAMS, EX)
    shutil.rmtree(tmp_path / "raw")
    for movie in (tmp_path / "motion_correction").glob("*.tif"):
        movie.unlink()
    with pytest.raises(ValueError):
        SiloSession.open(table)
    c = SiloSession.open(table, inputs="collect")
    jobs = silo_solve.dispatch(c, PARAMS, 3, EX)
    items = [
        it
        for j in jobs
        for it in json.loads(j.read_text(encoding="utf-8"))["items"]
    ]
    short = [it for it in items if it["container"] == "path1_trial0003.h5"]
    assert len(short) == 1 and short[0]["work"] == 0
    for job in jobs:
        silo_solve.solve_job(job, execution=EX)
    silo_collect.collect(c, PARAMS, EX)
    e = load_struct_h5(c.out_dir / "experiment_summary.h5")
    frames = e["Path1"]["frame_info"]["trial_num_frames"].ravel()
    assert frames.tolist() == [FRAMES[0], FRAMES[1], FRAMES[2]]


def test_localize_fits_missing_peaks(tmp_path, fake):
    """Localization files without their peaks (a streaming run stopped
    before its background fits) get them from silo-localize."""
    from giant_python.extraction.raster.stage_files import localization_attrs

    s = SiloSession.open(make_session(tmp_path))
    jobs = silo.localize_jobs(s, PARAMS, fit_peaks=False)
    for job in jobs:
        silo._localize_job(job)
    f = s.localize_file(0, 0)
    assert localization_attrs(f)["peaks_fitted"] == 0
    done = silo.localize(s, PARAMS, EX)
    assert len(done) == N_PATHS * N_TRIALS  # peak fits only
    assert localization_attrs(f)["peaks_fitted"] == 1
    assert silo.localize(s, PARAMS, EX) == []


def test_job_needs_its_container(tmp_path, fake):
    """A job refuses a container other than the one it was made for."""
    s = SiloSession.open(make_session(tmp_path))
    silo.localize(s, PARAMS, EX)
    silo.plan(s, PARAMS, EX, seeds=_seeds(s, tmp_path))
    silo.prepare(s, PARAMS, EX)
    job = silo_solve.dispatch(s, PARAMS, 1, EX)[0]
    with h5py.File(s.container_file(0, 0), "a") as f:
        f.attrs["settings_hash"] = "other"
    with pytest.raises(ValueError, match="not the container"):
        silo_solve.solve_job(job, execution=EX)


def test_trial_without_data_is_skipped(tmp_path, fake, monkeypatch):
    """A trial whose selected pixels have no sample is marked and left out
    of the summaries, streamed and staged alike."""
    from giant_python.pipeline.silo_session import NO_DATA

    def no_data(dat_path, *args, **kw):
        """Path 2, trial 2 has no data."""
        if "DMD2-TRIAL000002" in str(dat_path):
            raise ValueError(NO_DATA)
        return fake_load(dat_path, *args, **kw)

    monkeypatch.setattr(trial_data_mod, "load_trial", no_data)
    for staged in (False, True):
        root = tmp_path / str(staged)
        table = make_session(root)
        s = SiloSession.open(table)
        if staged:
            silo.localize(s, PARAMS, EX)
            silo.plan(s, PARAMS, EX, seeds=_seeds(s, tmp_path))
            silo.prepare(s, PARAMS, EX)
            assert not s.container_file(1, 1).exists()
            silo_solve.solve_containers(s, PARAMS, EX)
            silo_collect.collect(s, PARAMS, EX)
        else:
            silo.run_silo(table, PARAMS, EX, seeds_file=seeds_file(tmp_path))
        assert s.skip_file(1, 1).exists()
        e = load_struct_h5(s.out_dir / "experiment_summary.h5")
        frames = e["Path2"]["frame_info"]["trial_num_frames"].ravel()
        assert frames.tolist() == [FRAMES[0], 0, FRAMES[2]]
        # a rerun does not try the trial again
        assert silo.prepare(s, PARAMS, EX, units=[(1, 1)]) == []


def test_seeds_file_replaces_seeding(tmp_path, fake):
    """Seeds from a file (one-based, MATLAB's thetaf) give the plan's
    sources."""
    from giant_python.extraction.raster.plan import read_plan
    from giant_python.extraction.raster.stage_files import (
        read_seeds,
        write_seeds,
    )

    s = SiloSession.open(make_session(tmp_path))
    silo.localize(s, PARAMS, EX)
    seeds = np.array([[5.0, 7.0, 8.0, 1.0], [4.0, 16.0, 21.0, 1.0]])
    write_seeds(tmp_path / "seeds.h5", {1: seeds}, index_base=1)
    got = read_seeds(tmp_path / "seeds.h5", 2)
    assert got[1] is None
    np.testing.assert_array_equal(got[0][:, 1:3], seeds[:, 1:3] - 1)
    silo.plan(s, PARAMS, EX, paths=[0], seeds=got)
    p = read_plan(s.plan_file(0))
    np.testing.assert_array_equal(p.source_rows, [6, 15])
    np.testing.assert_array_equal(p.source_cols, [7, 20])


def test_cli(tmp_path, fake, capsys):
    """The commands parse their options and run the stages."""
    assert parse_numbers("1,3-4") == [1, 3, 4]
    assert parse_numbers(None) is None
    with pytest.raises(ValueError):
        parse_numbers("0")
    (tmp_path / "p.json").write_text('{"lam": 0.5}')
    args = build_parser().parse_args(
        [
            "silo",
            "t.h5",
            "--params",
            str(tmp_path / "p.json"),
            "--set",
            "nmf_iter=3",
            "--set",
            "footprint_variance=fixed",
            "--peak-th",
            "4",
            "--no-trial-peaks",
        ]
    )
    p = params_from_args(args)
    assert (p.lam, p.nmf_iter, p.footprint_variance, p.peak_th) == (
        0.5,
        3,
        "fixed",
        4.0,
    )
    assert p.trial_peaks is False
    with pytest.raises(ValueError):
        params_from_args(
            build_parser().parse_args(["silo", "t", "--set", "x=1"])
        )
    table = make_session(tmp_path / "s")
    opts = ["--workers", "1", "--set", "min_kinetic_score=0"]
    assert main(["silo-localize", str(table), "--paths", "1", *opts]) == 0
    assert main(["silo-localize", str(table), *opts]) == 0
    seeds = str(seeds_file(tmp_path))
    assert main(["silo-plan", str(table), "--seeds", seeds, *opts]) == 0
    assert main(["silo-prepare", str(table), "--trials", "1-3", *opts]) == 0
    assert (
        main(
            [
                "silo-dispatch",
                str(table),
                "--machines",
                "2",
                "--solver-workers",
                "1",
                *opts,
            ]
        )
        == 0
    )
    jobs = sorted(
        (tmp_path / "s" / "source_extraction" / "silo" / "jobs").glob("*.json")
    )
    for job in jobs:
        assert main(["silo-solve", str(job), "--solver-workers", "1"]) == 0
    assert main(["silo-collect", str(table), *opts]) == 0
    assert (
        tmp_path / "s" / "source_extraction" / "experiment_summary.h5"
    ).exists()
    capsys.readouterr()


def test_solve_span_is_the_last_subproblem_solved(monkeypatch):
    """A trial's solve span ends when its last subproblem is done, also
    when the trial finishes while the previous one is still collected and
    is itself collected later; it starts when it was submitted. Uses a
    controlled clock and futures completed in a set order."""
    import concurrent.futures as cf
    import types

    now = [0.0]
    monkeypatch.setattr(
        silo, "time", types.SimpleNamespace(perf_counter=lambda: now[0])
    )
    futures = [cf.Future(), cf.Future()]
    done = silo._track(futures)  # tracked as submitted, at 0.0
    now[0] = 0.05
    futures[0].set_result(None)
    now[0] = 0.2
    futures[1].set_result(None)  # the last subproblem
    now[0] = 0.6  # collected later
    p = silo._Pending(0, 0, None, [], futures, [], {}, Path("x"), 0.01, done)
    assert silo._solve_span(p, 0.0) == (0.01, 0.2)
    # solved here (no futures): done when the solve returned
    now[0] = 0.7
    q = silo._Pending(
        0, 1, None, [], [], [1], {}, Path("y"), 0.5, silo._track([])
    )
    now[0] = 0.9
    assert silo._solve_span(q, 0.0) == (0.5, 0.7)
