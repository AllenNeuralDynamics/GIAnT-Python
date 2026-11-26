"""Tests of the files SILo's stages hand to each other: per-trial
localization files and seeds files."""

import h5py
import numpy as np
import pytest

from giant_python.extraction.raster.stage_files import (
    add_peaks,
    localization_attrs,
    read_localization,
    read_seeds,
    write_localization,
    write_seeds,
)


def test_localization_round_trip(tmp_path):
    """Images, discards and peaks come back; peaks can be added later."""
    rng = np.random.default_rng(0)
    mean_im = rng.random((6, 7, 2)).astype(np.float32)
    act_im = rng.random((6, 7))
    act_im[0, 0] = np.nan
    discard = rng.random(40) < 0.2
    path = tmp_path / "loc.h5"
    write_localization(
        path, mean_im, act_im, act_im * 2, discard, None, {"trial": 3}
    )
    rec = read_localization(path)
    np.testing.assert_array_equal(rec.mean_im, mean_im)
    assert rec.mean_im.dtype == np.float32
    np.testing.assert_array_equal(rec.act_im, act_im)
    np.testing.assert_array_equal(rec.summary, act_im * 2)
    np.testing.assert_array_equal(rec.discard, discard)
    assert rec.peaks is None and rec.attrs["peaks_fitted"] == 0
    assert rec.attrs["trial"] == 3
    peaks = np.array([[1.0, 2.0, 3.0, 0.5]])
    side = add_peaks(path, peaks)
    assert side.name == "loc.peaks.h5"  # the localization file is not written
    rec = read_localization(path)
    np.testing.assert_array_equal(rec.peaks, peaks)
    assert localization_attrs(path)["peaks_fitted"] == 1
    assert read_localization(path, peaks=False).peaks is None
    images = rec.trial_images()
    np.testing.assert_array_equal(images.act_im, act_im)
    np.testing.assert_array_equal(images.peaks, peaks)
    assert not list(tmp_path.glob("*.partial"))
    # a new localization replaces the earlier one's peaks
    write_localization(
        path, mean_im, act_im, act_im, discard, None, {"trial": 3}
    )
    assert not side.exists() and read_localization(path).peaks is None


def test_localization_format_checks(tmp_path):
    """Other files are refused; a missing file has no attributes."""
    with h5py.File(tmp_path / "other.h5", "w") as f:
        f.attrs["format"] = "other"
    with pytest.raises(ValueError):
        read_localization(tmp_path / "other.h5")
    assert localization_attrs(tmp_path / "other.h5") is None
    assert localization_attrs(tmp_path / "missing.h5") is None
    path = tmp_path / "loc.h5"
    write_localization(
        path,
        np.zeros((2, 2, 1)),
        np.zeros((2, 2)),
        np.zeros((2, 2)),
        np.zeros(3, bool),
        None,
        {},
    )
    with h5py.File(path, "a") as f:
        f.attrs["format_version"] = 99
    with pytest.raises(ValueError, match="99"):
        read_localization(path)


def test_seeds(tmp_path):
    """Seeds per path, one-based (MATLAB's thetaf) or zero-based."""
    seeds = np.array([[2.0, 10.0, 20.0, 0.5], [1.0, 3.0, 4.0, 0.4]])
    write_seeds(tmp_path / "s1.h5", {2: seeds}, index_base=1)
    got = read_seeds(tmp_path / "s1.h5", 3)
    assert got[0] is None and got[2] is None
    np.testing.assert_array_equal(got[1][:, 1:3], seeds[:, 1:3] - 1)
    np.testing.assert_array_equal(got[1][:, [0, 3]], seeds[:, [0, 3]])
    write_seeds(tmp_path / "s0.h5", {1: seeds})
    np.testing.assert_array_equal(read_seeds(tmp_path / "s0.h5", 1)[0], seeds)
    with pytest.raises(ValueError):
        write_seeds(tmp_path / "bad.h5", {1: seeds}, index_base=2)


def test_recording_ids(tmp_path):
    """A recording's files: CYCLE segments and .meta; a missing one raises."""
    from giant_python.pipeline.silo_session import recording_ids

    for name in (
        "rec_DMD1-CYCLE-000000.dat",
        "rec_DMD1-CYCLE-000100.dat",
        "rec_DMD2-CYCLE-000000.dat",
        "rec_DMD1.meta",
        "other.meta",
    ):
        (tmp_path / name).write_bytes(b"x")
    ids = recording_ids(tmp_path / "rec_DMD1-CYCLE-000100.dat")
    assert [i["name"] for i in ids] == [
        "rec_DMD1-CYCLE-000000.dat",
        "rec_DMD1-CYCLE-000100.dat",
        "rec_DMD1.meta",
    ]
    with pytest.raises(FileNotFoundError):
        recording_ids(tmp_path / "rec_DMD3-CYCLE-000000.dat")
