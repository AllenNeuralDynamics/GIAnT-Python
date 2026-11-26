"""Tests of the container: one trial-path's inputs of source extraction."""

import h5py
import numpy as np
import pytest

from giant_python.extraction.raster.container import (
    FORMAT_VERSION,
    crop_of,
    read_container,
    read_subproblem,
    settings_hash,
    write_container,
)
from giant_python.extraction.raster.trial_data import (
    TrialData,
    TrialLoadParams,
)

SHAPE = (12, 15)  # consensus frame


def make_trial(n_frames=40, seed=0):
    """A small trial with two blobs of selected pixels and three sources.

    Returns the trial, the sources and each subproblem's pixels and
    sources: the two blobs' subproblems share the pixels of column 7.
    """
    rng = np.random.default_rng(seed)
    sel = np.zeros(SHAPE, dtype=bool)
    sel[2:7, 3:8] = True
    sel[4:10, 7:12] = True
    cols, rows = np.nonzero(sel.T)  # column-major, as find(selPix)
    pixel_rc = np.stack([rows, cols], 1)
    n_px = rows.size
    trial = TrialData(
        y=rng.standard_normal((2, n_px, n_frames)).astype(np.float32),
        finv=(1 + rng.random((n_px, n_frames))).astype(np.float32),
        frame_lines=np.ceil(np.arange(n_frames) * 53.4 + 1).astype(np.int64),
        motion_r=rng.standard_normal(n_frames),
        motion_c=rng.standard_normal(n_frames),
        discard=rng.random(n_frames) < 0.1,
        global_f=rng.random((2, n_frames)),
        roi_f=rng.random((3, 2, n_frames)),
        roi_fsvd=rng.random((3, 2, n_frames)),
        pixel_rc=pixel_rc,
        trial_rc=pixel_rc + np.array([0.25, -0.5]),
        photon_scale=25.0,
    )
    trial.roi_f[1] = np.nan  # a ROI without data
    sources_rc = np.array([[4, 5], [6, 9], [8, 10]])
    in_a = (rows < 7) & (cols < 8)
    in_b = (rows >= 4) & (cols >= 7)
    pixels = [np.flatnonzero(in_a), np.flatnonzero(in_b)]
    return trial, sources_rc, pixels, [np.array([0]), np.array([1, 2])], sel


@pytest.mark.parametrize("compression", ["lzf", "gzip", None])
def test_round_trip_is_exact(tmp_path, compression):
    """Everything written is read back unchanged, whole or by subproblem."""
    trial, sources_rc, pixels, sources, sel = make_trial()
    params = {"loader": TrialLoadParams(), "lambda": 0.9, "k": np.arange(3.0)}
    path = write_container(
        tmp_path / "trial.h5",
        trial,
        sources_rc,
        pixels,
        sources,
        SHAPE,
        params,
        {"dat": "a_DMD1-TRIAL000002.dat", "path": 1, "trial": 2},
        compression=compression,
        chunk_pixels=4,
    )
    assert path.exists() and not list(tmp_path.glob("*.partial"))
    c = read_container(path)
    for name in (
        "y",
        "finv",
        "frame_lines",
        "discard",
        "motion_r",
        "motion_c",
        "pixel_rc",
        "trial_rc",
        "global_f",
        "roi_f",
        "roi_fsvd",
    ):
        got, want = getattr(c, name), getattr(trial, name)
        np.testing.assert_array_equal(got, want)
        assert got.dtype == want.dtype or name == "pixel_rc"
    np.testing.assert_array_equal(c.sources_rc, sources_rc)
    assert c.consensus_shape == SHAPE and c.photon_scale == 25.0
    assert c.n_subproblems == 2 and not c.baseline_filled
    assert c.params["lambda"] == 0.9 and c.params["k"] == [0.0, 1.0, 2.0]
    assert c.params["loader"]["analyze_hz"] == 200.0
    assert c.source == {"dat": "a_DMD1-TRIAL000002.dat", "path": 1, "trial": 2}
    assert c.settings_hash == settings_hash(params) and c.code_version
    shell = read_container(path, data=False)
    assert shell.y is None and shell.finv is None
    with pytest.raises(ValueError, match="without its data"):
        shell.subproblem(0)
    with h5py.File(path, "r") as f:
        from_file = [read_subproblem(f, k) for k in range(2)]
    for k in range(2):
        for sp in (c.subproblem(k), read_subproblem(path, k), from_file[k]):
            np.testing.assert_array_equal(sp.y, trial.y[:, pixels[k]])
            np.testing.assert_array_equal(sp.finv, trial.finv[pixels[k]])
            assert sp.y.dtype == np.float32 and sp.y.shape[0] == 2
            np.testing.assert_array_equal(sp.pixels, pixels[k])
            np.testing.assert_array_equal(sp.sources, sources[k])
            # the crop is the bounding box of the pixels; sel's true
            # elements in column-major order are the rows of y
            rc = trial.pixel_rc[pixels[k]]
            assert (sp.row0, sp.col0) == tuple(rc.min(0))
            box = sel[
                sp.row0 : sp.row0 + sp.sel.shape[0],
                sp.col0 : sp.col0 + sp.sel.shape[1],
            ]
            assert not (sp.sel & ~box).any()  # within the selected map
            cols, rows = np.nonzero(sp.sel.T)
            np.testing.assert_array_equal(
                np.stack([rows + sp.row0, cols + sp.col0], 1), rc
            )
            np.testing.assert_array_equal(
                sp.sources_rc, sources_rc[sources[k]] - [sp.row0, sp.col0]
            )
    # overlapping subproblems share pixels: the data are stored once
    shared = np.intersect1d(pixels[0], pixels[1])
    assert shared.size == 3
    with h5py.File(path, "r") as f:
        assert f["data/y"].shape == trial.y.shape
        assert f["data/y"].chunks == (1, 4, trial.y.shape[2])
        assert f["data/y"].compression == compression
        assert f.attrs["format_version"] == FORMAT_VERSION


def test_baseline_is_reserved_and_can_be_filled(tmp_path):
    """The F0 network's baseline has empty, resizable datasets."""
    trial, sources_rc, pixels, sources, _ = make_trial()
    path = write_container(
        tmp_path / "trial.h5", trial, sources_rc, pixels, sources, SHAPE
    )
    n_px = trial.y.shape[1]
    with h5py.File(path, "r+") as f:
        g = f["baseline"]
        assert g["values"].shape == (n_px, 0)
        assert g["values"].maxshape == (n_px, None)
        assert g["frame_lines"].shape == (0,) and g.attrs["filled"] == 0
        assert g["values"].id.get_storage_size() == 0  # nothing stored
        g["values"].resize((n_px, 16))
        g["frame_lines"].resize((16,))
        assert np.isnan(g["values"][()]).all()  # unwritten: NaN
        g["values"][...] = 1.5
        g["frame_lines"][...] = np.arange(16) * 133.0 + 1
        g.attrs["filled"] = 1
    c = read_container(path, data=False)
    assert c.baseline_filled
    np.testing.assert_array_equal(  # the data are untouched
        read_subproblem(path, 1).y, trial.y[:, pixels[1]]
    )


def test_settings_hash_ignores_key_order_only():
    """The same settings hash alike however they are ordered or typed."""
    a = {"lambda": 0.9, "loader": TrialLoadParams(), "k": np.array([1.0, 2.0])}
    b = {
        "k": [1.0, 2.0],
        "loader": {
            "block_frames": 600,
            "nan_thresh": 0.33,
            "baseline_window_s": 4.0,
            "photon_scale": np.float64(25.0),
            "activity_channel": np.int64(1),
            "analyze_hz": 200.0,
        },
        "lambda": 0.9,
    }
    assert settings_hash(a) == settings_hash(b)
    assert len(settings_hash(a)) == 64
    assert settings_hash(a) != settings_hash({**a, "lambda": 1.0})
    assert settings_hash({}) != settings_hash({"lambda": 0.9})


def test_subproblem_lists_are_checked(tmp_path):
    """Index lists must match, increase and stay inside their lists."""
    trial, sources_rc, pixels, sources, _ = make_trial()
    path = tmp_path / "trial.h5"
    cases = [
        (pixels[:1], sources, "one source list"),
        ([pixels[0][::-1], pixels[1]], sources, "pixels must increase"),
        ([pixels[0], pixels[1] + 1000], sources, "pixels outside"),
        (pixels, [np.array([0]), np.array([1, 3])], "sources outside"),
        (pixels, [np.array([0]), np.array([2, 1])], "sources must increase"),
    ]
    for pix, src, message in cases:
        with pytest.raises(ValueError, match=message):
            write_container(path, trial, sources_rc, pix, src, SHAPE)
        assert not path.exists()
    write_container(path, trial, sources_rc, pixels, sources, SHAPE)
    # the same indices are refused in memory and from the file
    whole = read_container(path)
    for k in (2, -1):
        for source in (path, whole):
            with pytest.raises(IndexError, match="not one of 2"):
                read_subproblem(source, k)


def test_empty_grouping_and_foreign_files(tmp_path):
    """A container may hold no subproblems, or an empty one; other HDF5
    files and newer formats are refused."""
    trial, sources_rc, pixels, sources, _ = make_trial()
    path = write_container(
        tmp_path / "none.h5", trial, sources_rc[:0], [], [], SHAPE
    )
    c = read_container(path)
    assert c.n_subproblems == 0 and c.crops.shape == (0, 4)
    assert c.sources_rc.shape == (0, 2)
    empty = [np.zeros(0, int), pixels[1]]
    path = write_container(
        tmp_path / "empty.h5", trial, sources_rc, empty, [[], [1, 2]], SHAPE
    )
    sp = read_subproblem(path, 0)
    assert sp.y.shape == (2, 0, trial.y.shape[2]) and sp.sel.shape == (0, 0)
    assert sp.sources_rc.shape == (0, 2)
    np.testing.assert_array_equal(crop_of(trial.pixel_rc[:0]), [0, 0, 0, 0])
    np.testing.assert_array_equal(
        read_subproblem(path, 1).y, trial.y[:, pixels[1]]
    )
    other = tmp_path / "other.h5"
    with h5py.File(other, "w") as f:
        f["data"] = np.arange(3)
    with pytest.raises(ValueError, match="not a SILo container"):
        read_container(other)
    with h5py.File(path, "r+") as f:
        f.attrs["format_version"] = FORMAT_VERSION + 1
    with pytest.raises(ValueError, match="container format"):
        read_subproblem(path, 0)


@pytest.mark.parametrize("compression", ["lzf", None])
def test_trial_without_selected_pixels_round_trips(tmp_path, compression):
    """A path without sources has no selected pixels: its container holds
    the frames and traces, and empty data."""
    full, *_ = make_trial()
    n_f = full.frame_lines.size
    trial = TrialData(
        y=np.zeros((2, 0, n_f), np.float32),
        finv=np.zeros((0, n_f), np.float32),
        frame_lines=full.frame_lines,
        motion_r=full.motion_r,
        motion_c=full.motion_c,
        discard=full.discard,
        global_f=full.global_f,
        roi_f=np.zeros((0, 2, n_f)),
        roi_fsvd=np.zeros((0, 2, n_f)),
        pixel_rc=np.zeros((0, 2), np.int64),
        trial_rc=np.zeros((0, 2)),
        photon_scale=25.0,
    )
    path = write_container(
        tmp_path / "empty.h5",
        trial,
        np.zeros((0, 2), int),
        [],
        [],
        SHAPE,
        compression=compression,
    )
    c = read_container(path)
    assert c.y.shape == (2, 0, n_f) and c.finv.shape == (0, n_f)
    assert c.pixel_rc.shape == (0, 2) and c.roi_f.shape == (0, 2, n_f)
    assert c.n_subproblems == 0 and not c.baseline_filled
    np.testing.assert_array_equal(c.global_f, full.global_f)
    np.testing.assert_array_equal(c.frame_lines, full.frame_lines)
    with h5py.File(path, "r") as f:
        assert f["baseline/values"].shape == (0, 0)
    with pytest.raises(IndexError, match="not one of 0"):
        read_subproblem(path, 0)


def test_interrupted_write_leaves_the_old_container(tmp_path, monkeypatch):
    """A failed write keeps an existing container and no partial file under
    its name."""
    trial, sources_rc, pixels, sources, _ = make_trial()
    path = write_container(
        tmp_path / "trial.h5", trial, sources_rc, pixels, sources, SHAPE
    )
    before = path.read_bytes()
    from giant_python.extraction.raster import container

    def fail(*args, **kwargs):
        """Stand in for a failure late in the write."""
        raise OSError("disk full")

    monkeypatch.setattr(container, "_reserve_baseline", fail)
    other, *rest = make_trial(seed=1)
    with pytest.raises(OSError, match="disk full"):
        write_container(path, other, *rest[:3], SHAPE)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.partial"))
    np.testing.assert_array_equal(read_container(path).y, trial.y)
    # the write completes but the file cannot take its name (a folder has
    # it): the temporary file is removed too
    monkeypatch.undo()
    taken = tmp_path / "taken.h5"
    taken.mkdir()
    with pytest.raises(OSError):
        write_container(taken, other, *rest[:3], SHAPE)
    assert taken.is_dir() and not list(tmp_path.glob("*.partial"))
