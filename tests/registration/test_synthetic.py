"""Reader, frame formation and registration on a synthetic recording."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from giant_python.io.hdf5 import load_struct_h5, save_struct_h5
from giant_python.io.slap2_raster import read_line_offsets, read_raster_samples
from giant_python.io.trial_table import read_trial_table
from giant_python.registration import multiroi
from giant_python.registration.frames import FrameFormer

from .synthetic import make_layout, write_recording


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
