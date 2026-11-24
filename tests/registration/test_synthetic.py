"""Reader, frame formation and registration on a synthetic recording."""

import dataclasses
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from giant_python.io.hdf5 import load_struct_h5, save_struct_h5
from giant_python.io.slap2 import is_cycle_split
from giant_python.io.slap2_raster import read_line_offsets, read_raster_samples
from giant_python.io.trial_table import read_trial_table
from giant_python.registration import multiroi
from giant_python.registration.frames import FrameFormer

from .synthetic import HEADER_WORDS, make_layout, write_recording


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    """Layout, data file and true motion of a 6-second recording."""
    path = tmp_path_factory.mktemp("rec") / "synthetic_DMD1-TRIAL000001.dat"
    layout = make_layout(path, n_cycles=1400)
    truth = write_recording(layout, invalid=0.002)
    return layout, truth


def frames_reference(layout, samples, times, dt):
    """Per-pixel weighted means by the definition (slow loops)."""
    p, total = layout.lines_per_cycle, layout.total_lines
    pix = np.unique(layout.slot_pixel)
    dt_read = max(3 * dt, p)
    num = np.zeros((layout.num_channels, len(times), pix.size))
    den = np.zeros_like(num)
    for s in range(layout.slot_line.size):
        j = np.searchsorted(pix, layout.slot_pixel[s])
        for f, t in enumerate(times):
            lines = np.arange(max(1, t - dt_read), min(total, t + dt_read) + 1)
            lines = lines[(lines - 1) % p + 1 == layout.slot_line[s]]
            for ln in lines:
                w = np.exp(-abs(t - ln) / dt)
                for ch in range(layout.num_channels):
                    x = samples[ch, s, (ln - 1) // p]
                    if x != -32768:
                        num[ch, f, j] += w * x
                        den[ch, f, j] += w
    return num, den


def test_reader_and_line_headers(recording):
    """Samples land at their slots; header offsets decode."""
    layout, _ = recording
    s = read_raster_samples(layout, (10, 20), chunk_cycles=3)
    whole = read_raster_samples(layout, (0, layout.num_cycles))
    np.testing.assert_array_equal(s, whole[:, :, 10:20])
    p = layout.lines_per_cycle
    lines = np.array([1, p, p + 3, 5 * p + 2])
    x, y, z = read_line_offsets(layout, lines)
    cyc = (lines - 1) // p
    np.testing.assert_array_equal(x, cyc % 3)
    np.testing.assert_array_equal(y, -(cyc % 2))
    np.testing.assert_allclose(z, 0.25 * cyc)


def test_frame_former_matches_definition(recording):
    """Exp-kernel sums equal the definition, invalid samples masked."""
    layout, _ = recording
    samples = read_raster_samples(layout, (0, layout.num_cycles))
    times = np.array([1, 50, 333, 4000])
    former = FrameFormer(layout, samples, 0)
    num, den = former.sums(times, 20)
    want_num, want_den = frames_reference(layout, samples, times, 20)
    np.testing.assert_allclose(num.numpy(), want_num, rtol=1e-12)
    np.testing.assert_allclose(den.numpy(), want_den, rtol=1e-12)
    part = FrameFormer(layout, samples[:, :, 40:200], 40)
    np.testing.assert_allclose(
        part.sums([700], 20)[0].numpy(), former.sums([700], 20)[0].numpy()
    )
    with pytest.raises(ValueError):
        part.sums([20], 20)


def test_register_trial_recovers_motion(recording, tmp_path, monkeypatch):
    """Recovered motion follows the known motion (up to a constant)."""
    layout, truth = recording
    monkeypatch.setattr(multiroi, "raster_layout", lambda path: layout)
    params = multiroi.MultiRoiRegistrationParams(maxshift=6)
    n_lines = 9000
    res = multiroi.register_trial(
        layout.files[0],
        1,
        n_lines,
        tmp_path / "m.tif",
        tmp_path / "a.h5",
        params,
    )
    dt = 1600 / 80
    kernel = np.exp(-np.abs(np.arange(-3 * dt, 3 * dt + 1)) / dt)
    kernel /= kernel.sum()
    true = np.stack(
        [
            np.convolve(truth[:, i], kernel, "same")[res.ds_frames - 1]
            for i in (0, 1)
        ]
    )
    est = np.stack([res.motion_r, res.motion_c])
    err = est - true
    err -= np.median(err, axis=1, keepdims=True)
    assert np.sqrt(np.mean(err**2)) < 0.1
    a = load_struct_h5(tmp_path / "a.h5")
    assert a["slap2"]["varFacDS"].shape == (
        a["meanIM"].shape[1],
        a["meanIM"].shape[2],
        res.ds_frames.size,
    )
    np.testing.assert_array_equal(a["DSframes"].ravel(), res.ds_frames)
    assert not (tmp_path / "a.h5.partial").exists()


def digests(*paths):
    """SHA-256 of each file."""
    return [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths]


@pytest.mark.parametrize("save_tif", [True, False])
def test_failed_rerun_keeps_the_previous_outputs(
    recording, tmp_path, monkeypatch, save_tif
):
    """A run that fails while writing leaves an existing pair as it was."""
    layout, _ = recording
    monkeypatch.setattr(multiroi, "raster_layout", lambda path: layout)
    params = multiroi.MultiRoiRegistrationParams(maxshift=5, save_tif=save_tif)
    movie = tmp_path / ("m.tif" if save_tif else "m.h5")
    adata = tmp_path / "a.h5"
    args = (layout.files[0], 1, 3000, movie, adata, params)
    multiroi.register_trial(*args)
    before = digests(movie, adata)
    add = multiroi.OutputSink.add

    def fail_midway(self, k, regs, var_factor):
        """Write 50 frames, then fail."""
        if k == 50:
            raise OSError("disk full")
        return add(self, k, regs, var_factor)

    def fail_at_the_end(*args, **kwargs):
        """Fail after the movie is complete."""
        raise OSError("disk full")

    for target, name, fail in (
        (multiroi.OutputSink, "add", fail_midway),
        (multiroi, "alignment_fields", fail_at_the_end),
    ):
        with monkeypatch.context() as m:
            m.setattr(target, name, fail)
            with pytest.raises(OSError):
                multiroi.register_trial(*args)
        assert digests(movie, adata) == before
        assert sorted(tmp_path.iterdir()) == sorted([movie, adata])


def test_interrupted_publish_leaves_no_alignment_file(
    recording, tmp_path, monkeypatch
):
    """A pair that cannot be moved into place leaves no alignment file.

    The movie may then be new, but without an alignment file the trial is
    not taken for registered.
    """
    layout, _ = recording
    monkeypatch.setattr(multiroi, "raster_layout", lambda path: layout)
    params = multiroi.MultiRoiRegistrationParams(maxshift=5, save_tif=False)
    movie, adata = tmp_path / "m.h5", tmp_path / "a.h5"
    args = (layout.files[0], 1, 3000, movie, adata, params)
    multiroi.register_trial(*args)
    replace = os.replace

    def refuse_alignment_file(src, dst):
        """Fail to move the alignment file into place."""
        if Path(dst) == adata:
            raise PermissionError("in use")
        replace(src, dst)

    monkeypatch.setattr(os, "replace", refuse_alignment_file)
    with pytest.raises(PermissionError):
        multiroi.register_trial(*args)
    assert sorted(tmp_path.iterdir()) == [movie]


def blank_frames(monkeypatch, blank, half=None):
    """Frames formed at times where ``blank(times)`` holds have no samples;
    where ``half(times)`` holds, the lower half of the raster has none."""
    from giant_python.registration.frames import FrameFormer

    frames = FrameFormer.frames

    def blanked(self, times, dt, channels=None):
        """The frames, with the selected ones emptied."""
        vals, fresh = frames(self, times, dt, channels)
        t = np.asarray(times).ravel()
        lower = self.rows > np.median(self.rows)
        for i in np.flatnonzero(half(t) if half else []):
            vals[:, i, lower] = float("nan")
            fresh[i, lower] = 0
        hit = torch.as_tensor(blank(t), device=vals.device)
        vals[:, hit] = float("nan")
        fresh[hit] = 0
        return vals, fresh

    monkeypatch.setattr(FrameFormer, "frames", blanked)


@pytest.mark.parametrize("schedule", ["sequential", "grow"])
def test_frame_without_samples_is_missing(
    recording, tmp_path, monkeypatch, schedule
):
    """A frame without valid samples in mid-trial gets NaN motion and an
    all-NaN movie frame; the other frames are registered exactly as in a
    run without that frame, and the trial does not fail."""
    from giant_python.registration.growth import MISSING

    layout, _ = recording
    monkeypatch.setattr(multiroi, "raster_layout", lambda path: layout)
    params = multiroi.MultiRoiRegistrationParams(
        maxshift=5, save_tif=False, schedule=schedule
    )
    dt = (1 / layout.line_period_s) / params.align_hz
    init, ds = multiroi.frame_times(1, 3000, dt, params.init_frames)
    k = 70
    assert ds[k] not in init
    times = multiroi.frame_times
    out = {}
    for name in ("blank", "without"):
        with monkeypatch.context() as m:
            if name == "blank":
                blank_frames(m, lambda t: t == ds[k])
            else:
                m.setattr(
                    multiroi,
                    "frame_times",
                    lambda *a: (times(*a)[0], np.delete(times(*a)[1], k)),
                )
            out[name] = multiroi.register_trial(
                layout.files[0],
                1,
                3000,
                tmp_path / f"{name}.h5",
                tmp_path / f"{name}_a.h5",
                params,
            )
    res, ref = out["blank"], out["without"]
    assert not res.registration_failed
    assert np.isnan([res.motion_r[k], res.motion_c[k], res.peak[k]]).all()
    others = np.arange(ds.size) != k
    for f in ("motion_r", "motion_c", "peak"):
        np.testing.assert_array_equal(getattr(res, f)[others], getattr(ref, f))
    n_ch = layout.num_channels
    with (
        h5py.File(tmp_path / "blank.h5") as a,
        h5py.File(tmp_path / "without.h5") as b,
    ):
        pages = a["data"][()]
        mine = np.s_[k * n_ch : (k + 1) * n_ch]
        assert np.isnan(pages[mine]).all()
        np.testing.assert_array_equal(
            np.delete(pages, mine, axis=0), b["data"][()]
        )
    adata = load_struct_h5(tmp_path / "blank_a.h5")
    want = load_struct_h5(tmp_path / "without_a.h5")
    assert adata["registrationFailed"].item() == 0
    assert np.isnan(adata["motionDSr"].ravel()[k])
    assert np.isnan(adata["slap2"]["varFacDS"][..., k]).all()
    np.testing.assert_array_equal(adata["meanIM"], want["meanIM"])
    if schedule == "grow":
        assert adata["growthQC"]["kind"].ravel()[k] == MISSING


@pytest.mark.parametrize("schedule", ["sequential", "grow"])
def test_trial_without_samples_at_its_start_or_at_all(
    recording, tmp_path, monkeypatch, schedule
):
    """A recording that starts late is registered from its first frames
    with samples, in the usual window even if the first of them covers
    only part of the raster; a trial without any samples completes and is
    flagged."""
    from giant_python.registration.sources import make_source

    layout, _ = recording
    monkeypatch.setattr(multiroi, "raster_layout", lambda path: layout)
    params = multiroi.MultiRoiRegistrationParams(
        maxshift=5, save_tif=False, schedule=schedule
    )
    args = (layout.files[0], 1, 3000)
    lay, former, init, ds, dt = multiroi.open_trial(
        *args, params, torch.device("cpu")
    )
    window = make_source(lay, former, 1, 3000, init, ds, dt, False).window
    with monkeypatch.context() as m:
        # the first initial frames have no samples, the next one only half
        assert np.sum((init >= 1200) & (init < 1300)) == 1
        blank_frames(m, lambda t: t < 1200, lambda t: t < 1300)
        late = multiroi.register_trial(
            *args, tmp_path / "late.h5", tmp_path / "late_a.h5", params
        )
    early = late.ds_frames < 1200
    assert np.isnan(late.motion_r[early]).all()
    assert np.isfinite(late.motion_r[late.ds_frames >= 1200]).all()
    assert not late.registration_failed
    trim = load_struct_h5(tmp_path / "late_a.h5")["slap2"]
    np.testing.assert_array_equal(trim["trimRows"].ravel(), window.trim_rows)
    np.testing.assert_array_equal(trim["trimCols"].ravel(), window.trim_cols)
    with monkeypatch.context() as m:
        blank_frames(m, lambda t: np.ones(t.shape, dtype=bool))
        empty = multiroi.register_trial(
            *args, tmp_path / "empty.h5", tmp_path / "empty_a.h5", params
        )
    assert np.isnan(empty.motion_r).all() and empty.registration_failed
    with h5py.File(tmp_path / "empty.h5") as f:
        assert np.isnan(f["data"][()]).all()


def test_register_trial_keeps_fractional_lines(
    recording, tmp_path, monkeypatch
):
    """Fractional trial edges set the frame times, as in MATLAB.

    Trial tables of continuous recordings cut trials at ``linspace`` edges;
    MultiRoiRegistration uses them unrounded: ``ceil(first:dt:last)``.
    """
    layout, _ = recording
    monkeypatch.setattr(multiroi, "raster_layout", lambda path: layout)
    params = multiroi.MultiRoiRegistrationParams(maxshift=6)
    first, last = 1000.4, 4000.7
    res = multiroi.register_trial(
        layout.files[0],
        first,
        last,
        tmp_path / "m.tif",
        tmp_path / "a.h5",
        params,
    )
    dt = 1600 / 80
    n = int(np.floor((last - first) / dt)) + 1
    np.testing.assert_array_equal(
        res.ds_frames, np.ceil(first + dt * np.arange(n))
    )
    assert res.ds_frames[0] == 1001


def test_register_multiroi_updates_trial_table(
    recording, tmp_path, monkeypatch
):
    """The pipeline writes MATLAB-named outputs and motion_correction."""
    from giant_python.pipeline.register import register_multiroi

    layout, _ = recording
    monkeypatch.setattr(multiroi, "raster_layout", lambda path: layout)
    dat = Path(layout.files[0])
    name = dat.name
    table = tmp_path / "trial_table.h5"
    save_struct_h5(
        table,
        {
            "datadr": str(dat.parent),
            "savedr": str(tmp_path),
            "filename": np.array([[name]], dtype=object),
            "epoch": np.array([[3.0]]),
            "true_trial_ix": np.array([[1.0]]),
            "slap2_info": {
                "first_line": np.array([[1.0]]),
                "last_line": np.array([[3000.0]]),
            },
            "motion_correction": {"keep_me": 7.0},
        },
    )
    fields = register_multiroi(
        table,
        {"maxshift": 5, "save_tif": False},
        execution={"device": "cpu", "max_workers": 1},
    )
    assert fields["fn_adata"][0, 0] == "E3T1DMD1_ALIGNMENTDATA.h5"
    movie = tmp_path / "motion_correction" / fields["fn_reg_ds"][0, 0]
    assert movie.name == "E3T1DMD1_REGISTERED_DOWNSAMPLED-80Hz.h5"
    with h5py.File(movie) as f:
        assert f["data"].shape[0] == 2 * 150
    tt = read_trial_table(table)
    mc = tt.motion_correction
    assert mc["keep_me"].item() == 7.0
    assert mc["fn_adata"] == "E3T1DMD1_ALIGNMENTDATA.h5"
    assert mc["align_params"]["alignHz"].item() == 80.0
    assert mc["registration_failed"].dtype == np.int8


def one_trial_table(layout, tmp_path):
    """A MATLAB-style trial table with one 3000-line trial."""
    dat = Path(layout.files[0])
    table = tmp_path / "trial_table.h5"
    save_struct_h5(
        table,
        {
            "datadr": str(dat.parent),
            "savedr": str(tmp_path),
            "filename": np.array([[dat.name]], dtype=object),
            "epoch": np.array([[3.0]]),
            "true_trial_ix": np.array([[1.0]]),
            "slap2_info": {
                "first_line": np.array([[1.0]]),
                "last_line": np.array([[3000.0]]),
            },
        },
    )
    return table


def test_rerun_keeps_the_saved_failure_flag(recording, tmp_path, monkeypatch):
    """Reused outputs report the registrationFailed they were saved with."""
    from giant_python.pipeline.register import register_multiroi

    layout, _ = recording
    monkeypatch.setattr(multiroi, "raster_layout", lambda path: layout)
    table = one_trial_table(layout, tmp_path)
    run = dict(
        params={"maxshift": 5, "save_tif": False},
        execution={"device": "cpu", "max_workers": 1},
    )
    fields = register_multiroi(table, **run)
    assert not fields["registration_failed"][0, 0]
    adata = tmp_path / "motion_correction" / fields["fn_adata"][0, 0]
    with h5py.File(adata, "r+") as f:  # as if the motion had been too large
        f["registrationFailed"][...] = 1
    before = digests(adata)
    fields = register_multiroi(table, **run)
    assert digests(adata) == before  # reused, not registered again
    assert fields["registration_failed"][0, 0]
    assert read_trial_table(table).motion_correction["registration_failed"]


def test_rerun_with_other_settings_registers_again(
    recording, tmp_path, monkeypatch
):
    """Outputs are reused only if made with the requested settings."""
    from giant_python.pipeline.register import register_multiroi
    from giant_python.registration.outputs import SETTINGS_ATTR

    layout, _ = recording
    monkeypatch.setattr(multiroi, "raster_layout", lambda path: layout)
    table = one_trial_table(layout, tmp_path)
    execution = {"device": "cpu", "max_workers": 1}
    sequential = {"maxshift": 5, "save_tif": False}
    grow = dict(sequential, schedule="grow")
    fields = register_multiroi(table, sequential, execution=execution)
    adata = tmp_path / "motion_correction" / fields["fn_adata"][0, 0]
    first = digests(adata)
    assert "growthQC" not in load_struct_h5(adata)

    fields = register_multiroi(table, grow, execution=execution)
    assert digests(adata) != first  # registered again
    assert "growthQC" in load_struct_h5(adata)
    mc = read_trial_table(table).motion_correction
    assert mc["align_params"]["schedule"] == "grow"
    with h5py.File(adata) as f:
        saved = json.loads(f.attrs[SETTINGS_ATTR])
    assert saved["schedule"] == "grow" and saved["maxshift"] == 5
    assert "overwriteExisting" not in saved

    grown = digests(adata)
    register_multiroi(table, grow, execution=execution)
    assert digests(adata) == grown  # same settings: reused

    with h5py.File(adata, "r+") as f:  # outputs that do not record them
        del f.attrs[SETTINGS_ATTR]
    register_multiroi(table, grow, execution=execution)
    with h5py.File(adata) as f:
        assert SETTINGS_ATTR in f.attrs


def test_failed_rerun_does_not_name_the_old_outputs(
    recording, tmp_path, monkeypatch
):
    """A re-registration that raises leaves the old files on disk but
    clears their names, so the table never pairs them with new settings."""
    from giant_python.pipeline.register import register_multiroi

    layout, _ = recording
    monkeypatch.setattr(multiroi, "raster_layout", lambda path: layout)
    table = one_trial_table(layout, tmp_path)
    execution = {"device": "cpu", "max_workers": 1}
    sequential = {"maxshift": 5, "save_tif": False}
    fields = register_multiroi(table, sequential, execution=execution)
    moco = tmp_path / "motion_correction"
    files = [moco / fields["fn_reg_ds"][0, 0], moco / fields["fn_adata"][0, 0]]
    before = digests(*files)

    def fail(*args, **kwargs):
        """Fail before anything is published."""
        raise OSError("disk full")

    monkeypatch.setattr(multiroi, "register_trial", fail)
    fields = register_multiroi(
        table, dict(sequential, schedule="grow"), execution=execution
    )
    assert fields["fn_reg_ds"][0, 0] == fields["fn_adata"][0, 0] == ""
    assert fields["registration_failed"][0, 0]
    assert digests(*files) == before
    mc = read_trial_table(table).motion_correction
    assert mc["fn_reg_ds"] == mc["fn_adata"] == ""
    assert mc["registration_failed"]
    assert mc["align_params"]["schedule"] == "grow"


@pytest.mark.parametrize("case", ["row_major", "channels", "device"])
def test_rejected_request_writes_nothing(
    recording, tmp_path, monkeypatch, case
):
    """A request that fails validation (a row-major table, an unsupported
    recording, an unavailable device) leaves outputs and table untouched."""
    from giant_python.pipeline.register import register_multiroi

    layout, _ = recording
    lay = layout
    if case == "channels":
        lay = dataclasses.replace(layout, num_channels=3)
    monkeypatch.setattr(multiroi, "raster_layout", lambda path: lay)
    table = one_trial_table(layout, tmp_path)
    if case == "row_major":
        with h5py.File(table, "r+") as f:
            f["row_major"][...] = 1
    moco = tmp_path / "motion_correction"
    moco.mkdir()
    old = [
        moco / "E3T1DMD1_REGISTERED_DOWNSAMPLED-80Hz.h5",
        moco / "E3T1DMD1_ALIGNMENTDATA.h5",
    ]
    for p in old:
        p.write_bytes(b"from an earlier run")
    before = digests(table, *old)
    with pytest.raises(ValueError):
        register_multiroi(
            table,
            {"maxshift": 5, "save_tif": False, "overwrite_existing": True},
            execution={
                "device": "cuda:99" if case == "device" else "cpu",
                "max_workers": 1,
            },
        )
    assert digests(table, *old) == before
    assert sorted(moco.iterdir()) == sorted(old)


def test_ragged_table_skips_empty_cells(recording, tmp_path, monkeypatch):
    """A cell without a recording (one path has fewer trials) is skipped
    and flagged, with empty output names; the other cells register."""
    from giant_python.pipeline.register import register_multiroi

    layout, _ = recording
    monkeypatch.setattr(multiroi, "raster_layout", lambda path: layout)
    dat = Path(layout.files[0])
    table = tmp_path / "trial_table.h5"
    save_struct_h5(
        table,
        {
            "datadr": str(dat.parent),
            "savedr": str(tmp_path),
            "filename": np.array(
                [[dat.name, dat.name], [dat.name, ""]], dtype=object
            ),
            "epoch": np.array([[3.0, 3.0], [3.0, np.nan]]),
            "true_trial_ix": np.array([[1.0, 2.0]]),
            "slap2_info": {
                "first_line": np.array([[1.0, 3001.0], [1.0, np.nan]]),
                "last_line": np.array([[3000.0, 6000.0], [3000.0, np.nan]]),
            },
        },
    )
    fields = register_multiroi(
        table,
        {"maxshift": 5, "save_tif": False},
        execution={"device": "cpu", "max_workers": 1},
    )
    assert fields["fn_reg_ds"][1, 1] == fields["fn_adata"][1, 1] == ""
    np.testing.assert_array_equal(
        fields["registration_failed"], [[False, False], [False, True]]
    )
    names = [
        fields[k][d, t]
        for k in ("fn_reg_ds", "fn_adata")
        for d, t in ((0, 0), (0, 1), (1, 0))
    ]
    assert all(names)
    moco = tmp_path / "motion_correction"
    assert sorted(p.name for p in moco.iterdir()) == sorted(names)
    mc = read_trial_table(table).motion_correction
    assert mc["fn_reg_ds"][1, 1] == mc["fn_adata"][1, 1] == ""
    assert mc["registration_failed"][1, 1] == 1


def test_trial_table_update_keeps_other_entries(tmp_path, monkeypatch):
    """Only the registration fields are replaced, once all are written.

    Other entries of the group and their attributes are left untouched.
    """
    from giant_python.pipeline import register

    table = tmp_path / "trial_table.h5"
    save_struct_h5(
        table,
        {
            "datadr": "d",
            "motion_correction": {
                "keep_me": 7.0,
                "fn_adata": "old.h5",
                "align_params": {"old_only": 1.0},
            },
        },
    )
    with h5py.File(table, "r+") as f:
        mc = f["motion_correction"]
        mc.attrs["note"] = "kept"
        mc["keep_me"].attrs["unit"] = "px"
        mc.create_dataset("empty", shape=(0, 0), dtype=np.float64)
        nested = mc.create_group("nested")
        nested.create_dataset("row_major", data=np.zeros((1, 1), np.uint8))
    fields = {
        "fn_adata": np.array([["new.h5"]], dtype=object),
        "registration_failed": np.array([[True]]),
        "align_params": {"alignHz": 80.0},
    }
    write = register.write_matlab_value

    def fail_on_align_params(grp, key, value):
        """Fail while writing the last field."""
        if key.startswith("align_params"):
            raise OSError("disk full")
        write(grp, key, value)

    def entries():
        """Every entry under motion_correction."""
        names = []
        with h5py.File(table) as f:
            f["motion_correction"].visit(names.append)
        return set(names)

    kept = {"keep_me", "empty", "nested", "nested/row_major"}
    with monkeypatch.context() as m:
        m.setattr(register, "write_matlab_value", fail_on_align_params)
        with pytest.raises(OSError):
            register._update_trial_table(table, fields)
    old = {"fn_adata", "align_params", "align_params/old_only"}
    assert entries() == kept | old
    assert load_struct_h5(table)["motion_correction"]["fn_adata"] == "old.h5"
    register._update_trial_table(table, fields)
    assert entries() == kept | {
        "fn_adata",
        "registration_failed",
        "align_params",
        "align_params/alignHz",
    }
    with h5py.File(table) as f:
        mc = f["motion_correction"]
        assert mc.attrs["note"] == "kept"
        assert mc["keep_me"].attrs["unit"] == "px"
        assert mc["empty"].shape == (0, 0)
    mc = load_struct_h5(table)["motion_correction"]
    assert mc["fn_adata"] == "new.h5"
    assert mc["registration_failed"].item() == 1
    assert mc["align_params"]["alignHz"].item() == 80.0


def test_register_multiroi_keeps_fractional_lines(
    recording, tmp_path, monkeypatch
):
    """The pipeline passes the table's fractional trial edges on unrounded."""
    from giant_python.io.hdf5 import load_struct_h5
    from giant_python.pipeline.register import register_multiroi

    layout, _ = recording
    monkeypatch.setattr(multiroi, "raster_layout", lambda path: layout)
    dat = Path(layout.files[0])
    table = tmp_path / "trial_table.h5"
    first, last = 1000.4, 4000.7
    save_struct_h5(
        table,
        {
            "datadr": str(dat.parent),
            "savedr": str(tmp_path),
            "filename": np.array([[dat.name]], dtype=object),
            "epoch": np.array([[3.0]]),
            "true_trial_ix": np.array([[1.0]]),
            "slap2_info": {
                "first_line": np.array([[first]]),
                "last_line": np.array([[last]]),
            },
        },
    )
    fields = register_multiroi(
        table,
        {"maxshift": 5, "save_tif": False},
        execution={"device": "cpu", "max_workers": 1},
    )
    adata = load_struct_h5(
        tmp_path / "motion_correction" / fields["fn_adata"][0, 0]
    )
    adata = adata.get("aData", adata)
    dt = 1600 / 80
    n = int(np.floor((last - first) / dt)) + 1
    np.testing.assert_array_equal(
        np.ravel(adata["DSframes"]), np.ceil(first + dt * np.arange(n))
    )


def test_unsupported_options_raise():
    """MATLAB options that are not ported are rejected."""
    with pytest.raises(NotImplementedError):
        multiroi.MultiRoiRegistrationParams(ref_stack_template=True)
    with pytest.raises(TypeError):
        from giant_python.models.params import resolve_registration_options

        resolve_registration_options({"compensateOnline": True})


def test_save_struct_h5_is_column_major(tmp_path):
    """Values come back in MATLAB orientation and MATLAB classes."""
    path = tmp_path / "s.h5"
    save_struct_h5(
        path,
        {
            "a": np.arange(6.0).reshape(2, 3),
            "v": np.arange(4.0),
            "flag": True,
            "n": 3,
            "s": "text",
            "cells": np.array([["x"], ["yy"]], dtype=object),
            "g": {"single": np.float32(2.5)},
        },
    )
    with h5py.File(path) as f:
        assert f["a"].shape == (3, 2) and f["v"].shape == (4, 1)
        assert f["flag"].dtype == np.int8 and f["n"].dtype == np.float64
        assert f["row_major"][()] == 0
    back = load_struct_h5(path)
    np.testing.assert_array_equal(back["a"], np.arange(6.0).reshape(2, 3))
    assert back["v"].shape == (1, 4)
    assert back["s"] == "text"
    assert list(back["cells"].ravel()) == ["x", "yy"]
    assert back["g"]["single"].dtype == np.float32


def test_registration_runs_on_cuda_like_cpu(recording, tmp_path, monkeypatch):
    """CUDA and CPU give the same motion (skipped without a GPU)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    layout, _ = recording
    monkeypatch.setattr(multiroi, "raster_layout", lambda path: layout)
    params = multiroi.MultiRoiRegistrationParams(maxshift=6)
    out = [
        multiroi.register_trial(
            layout.files[0],
            1,
            3000,
            tmp_path / f"{d}.tif",
            tmp_path / f"{d}.h5",
            params,
            device=d,
        )
        for d in ("cpu", "cuda")
    ]
    diff = np.hypot(
        out[0].motion_r - out[1].motion_r, out[0].motion_c - out[1].motion_c
    )
    # float32 sums differ in order; a near-tie peak may resolve differently
    assert np.median(diff) < 1e-3 and diff.max() < 0.1


def test_var_factor_deflate_keeps_values(recording, tmp_path, monkeypatch):
    """Deflated varFacDS (default) holds the same values as uncompressed."""
    layout, _ = recording
    monkeypatch.setattr(multiroi, "raster_layout", lambda path: layout)
    for level in (0, 1):
        params = multiroi.MultiRoiRegistrationParams(
            maxshift=5, var_factor_deflate=level, save_tif=False
        )
        multiroi.register_trial(
            layout.files[0],
            1,
            3000,
            tmp_path / f"m{level}.h5",
            tmp_path / f"a{level}.h5",
            params,
        )
    with (
        h5py.File(tmp_path / "a0.h5") as a,
        h5py.File(tmp_path / "a1.h5") as b,
    ):
        assert a["slap2/varFacDS"].compression is None
        assert b["slap2/varFacDS"].compression == "gzip"
        np.testing.assert_array_equal(
            a["slap2/varFacDS"][()], b["slap2/varFacDS"][()]
        )
    with h5py.File(tmp_path / "m1.h5") as m:
        assert m["data"].compression == "gzip"
        assert m["data"].compression_opts == 1


def test_cycle_split_names():
    """Files of CYCLE-split recordings are recognised by name only."""
    assert is_cycle_split("acq_DMD1-TRIAL000001-CYCLE-000000.dat")
    assert is_cycle_split("d/acq_DMD2-CYCLE-318000.dat")
    assert is_cycle_split("acq_DMD1-CYCLE000000.dat")
    assert not is_cycle_split("acq_DMD1-TRIAL000001.dat")
    assert not is_cycle_split("CYCLE-1/acq_DMD1-TRIAL000001.dat")


def split_recording(layout, k, tmp_path):
    """The recording written as two files, split before cycle ``k``."""
    raw = np.fromfile(layout.files[0], dtype=np.int16)
    head = raw[:HEADER_WORDS]
    body = raw[HEADER_WORDS:].reshape(layout.num_cycles, -1)
    files = []
    for name, part in (("a", body[:k]), ("b", body[k:])):
        path = tmp_path / f"split_{name}.dat"
        with open(path, "wb") as f:
            f.write(head.tobytes())
            f.write(part.tobytes())
        files.append(str(path))
    return dataclasses.replace(
        layout,
        files=tuple(files),
        file_cycles=np.array([k, layout.num_cycles - k]),
        file_offset_words=np.array([HEADER_WORDS, HEADER_WORDS]),
    )


def test_split_recording_reads_and_registers_alike(
    recording, tmp_path, monkeypatch
):
    """Reads and registration across two files equal those of one file."""
    layout, _ = recording
    k = 500
    split = split_recording(layout, k, tmp_path)
    for cycles in ((k - 7, k + 9), (0, layout.num_cycles), (k + 3, k + 40)):
        np.testing.assert_array_equal(
            read_raster_samples(split, cycles, chunk_cycles=5),
            read_raster_samples(layout, cycles),
        )
    p = layout.lines_per_cycle
    lines = np.arange((k - 3) * p + 1, (k + 3) * p + 1)
    for a, b in zip(
        read_line_offsets(split, lines), read_line_offsets(layout, lines)
    ):
        np.testing.assert_array_equal(a, b)
    params = multiroi.MultiRoiRegistrationParams(maxshift=5, save_tif=False)
    first, last = (k - 150) * p + 1, (k + 150) * p
    out = {}
    for name, lay in (("one", layout), ("two", split)):
        monkeypatch.setattr(
            multiroi, "raster_layout", lambda path, lay=lay: lay
        )
        out[name] = multiroi.register_trial(
            lay.files[0],
            first,
            last,
            tmp_path / f"m_{name}.h5",
            tmp_path / f"a_{name}.h5",
            params,
        )
    for f in ("ds_frames", "motion_r", "motion_c", "peak"):
        np.testing.assert_array_equal(
            getattr(out["one"], f), getattr(out["two"], f)
        )
