"""Tests of loading a registered SLAP2 trial for raster localization."""

import h5py
import numpy as np
import pytest

from giant_python.extraction.raster.trial import (
    discard_frames,
    load_registered_trial,
)


def test_discard_frames_around_a_motion_event():
    """A jump of recNegErr discards it and neighbours above half the
    threshold (MATLAB: dilate by ~50 ms, keep above half, close)."""
    rne = -0.5 + 0.01 * np.random.default_rng(0).standard_normal(800)
    rne[400:402] += 0.2  # registration error jumps for two frames
    d = discard_frames(rne, 80.0, motion_thresh=2.0)
    assert d[400:402].all()
    assert d.sum() <= 8 and not d[:395].any() and not d[408:].any()
    d0 = discard_frames(rne, 80.0, discard_initial_s=0.5)
    assert d0[:40].all() and not d0[40:395].any()


def test_load_registered_trial(tmp_path):
    """Channels are deinterleaved, pages transposed to MATLAB's rows and
    columns, discards set to NaN, the variance factor read."""
    tifffile = pytest.importorskip("tifffile")
    n, rows, cols = 400, 6, 9  # MATLAB rows x cols; n > the 4 s median
    rng = np.random.default_rng(0)
    act = rng.poisson(5, (n, rows, cols)).astype(np.float32)
    other = np.full((n, rows, cols), 100, np.float32)
    pages = np.empty((2 * n, cols, rows), np.float32)  # pages: (col, row)
    pages[0::2] = act.transpose(0, 2, 1)
    pages[1::2] = other.transpose(0, 2, 1)
    stem = "E1T1DMD1"
    tifffile.imwrite(
        tmp_path / f"{stem}_REGISTERED_DOWNSAMPLED-80Hz.tif", pages
    )
    rne = -0.5 + 0.01 * rng.standard_normal(n)
    rne[200] += 0.2
    vf = rng.uniform(1, 2, (n, cols, rows)).astype(np.float32)
    with h5py.File(tmp_path / f"{stem}_ALIGNMENTDATA.h5", "w") as f:
        f["alignHz"] = np.array([[80.0]])
        f["recNegErr"] = rne[None, :]
        f["slap2/varFacDS"] = vf  # MATLAB rows x cols x frames, reversed
    tr = load_registered_trial(
        tmp_path, f"{stem}_REGISTERED_DOWNSAMPLED-80Hz.tif", 2
    )
    assert tr.movie.shape == (n, rows, cols) and tr.align_hz == 80.0
    keep = ~tr.discard
    assert tr.discard[200] and keep.sum() > n - 8
    np.testing.assert_array_equal(tr.movie[keep], act[keep])
    assert np.isnan(tr.movie[tr.discard]).all()
    np.testing.assert_allclose(tr.mean_im[..., 1], 100)
    np.testing.assert_allclose(tr.var_factor, vf.transpose(0, 2, 1))
