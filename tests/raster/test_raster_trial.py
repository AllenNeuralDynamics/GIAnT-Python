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


@pytest.mark.parametrize("layout", ["one_series", "page_by_page"])
def test_load_registered_trial(tmp_path, layout):
    """Channels are deinterleaved, pages transposed to MATLAB's rows and
    columns, discards set to NaN, the variance factor read. The TIFF may
    hold one series of all pages (MATLAB's Fast_BigTiff_Write) or one per
    page (GIAnT-Python's registration writes each page on its own)."""
    tifffile = pytest.importorskip("tifffile")
    n, rows, cols = 400, 6, 9  # MATLAB rows x cols; n > the 4 s median
    rng = np.random.default_rng(0)
    act = rng.poisson(5, (n, rows, cols)).astype(np.float32)
    other = np.full((n, rows, cols), 100, np.float32)
    pages = np.empty((2 * n, cols, rows), np.float32)  # pages: (col, row)
    pages[0::2] = act.transpose(0, 2, 1)
    pages[1::2] = other.transpose(0, 2, 1)
    stem = "E1T1DMD1"
    tif = tmp_path / f"{stem}_REGISTERED_DOWNSAMPLED-80Hz.tif"
    if layout == "one_series":
        tifffile.imwrite(tif, pages)
    else:  # as giant_python.registration.outputs writes it
        with tifffile.TiffWriter(tif, bigtiff=True) as tw:
            for page in pages:
                tw.write(page, photometric="minisblack")
        with tifffile.TiffFile(tif) as tf:
            assert len(tf.series) == 2 * n  # imread would read one page
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


def test_localize_trial_retries_numerical_failures(monkeypatch):
    """Any failure of the localization itself is retried with the outer
    rows and columns NaN (MATLAB's catch); invalid inputs are not."""
    import giant_python.extraction.raster.trial as trial_mod
    from giant_python.extraction.raster.localization import (
        RasterLocalizationParams,
    )

    movie = np.ones((20, 6, 7), np.float32)
    fake = trial_mod.RegisteredTrial(
        movie, None, np.ones((6, 7, 1)), np.zeros(20, bool), 80.0
    )
    monkeypatch.setattr(trial_mod, "load_registered_trial", lambda *a: fake)
    seen = []

    def flaky(mov, *args):
        """Fails like a numerical error unless the edges are NaN."""
        seen.append(bool(np.isnan(mov[:, 0, :]).all()))
        if not seen[-1]:
            raise ValueError("non-finite residuals")
        return "ok"

    monkeypatch.setattr(trial_mod, "localize_sources", flaky)
    loc, _ = trial_mod.localize_trial("dir", "x_REGISTERED.tif", 1)
    assert loc == "ok" and seen == [False, True]
    seen.clear()
    p = RasterLocalizationParams(variance_floor=False)
    with pytest.raises(ValueError, match="external baseline"):
        trial_mod.localize_trial("dir", "x_REGISTERED.tif", 1, p)
    assert seen == []


@pytest.mark.parametrize(
    "num_channels, activity_channel",
    [(2, 0), (2, 3), (0, 1), (2.0, 1), (2, True), (1, -1)],
)
def test_channel_arguments_checked(tmp_path, num_channels, activity_channel):
    """Channel arguments outside 1..num_channels (0 would select the last
    channel), or not integers, raise before anything is read."""
    with pytest.raises(ValueError, match="channel"):
        load_registered_trial(
            tmp_path,
            "E1T1DMD1_REGISTERED_DOWNSAMPLED-80Hz.tif",
            num_channels,
            activity_channel,
        )


def test_page_count_not_a_multiple_of_channels(tmp_path):
    """An incomplete last frame is left out, with a warning."""
    tifffile = pytest.importorskip("tifffile")
    n, rows, cols = 400, 4, 5
    pages = np.ones((2 * n + 1, cols, rows), np.float32)
    stem = "E1T1DMD1"
    tifffile.imwrite(
        tmp_path / f"{stem}_REGISTERED_DOWNSAMPLED-80Hz.tif",
        pages,
        photometric="minisblack",
    )
    with h5py.File(tmp_path / f"{stem}_ALIGNMENTDATA.h5", "w") as f:
        f["alignHz"] = np.array([[80.0]])
        f["recNegErr"] = np.full((1, n), -0.5)
    with pytest.warns(UserWarning, match="not a multiple"):
        tr = load_registered_trial(
            tmp_path, f"{stem}_REGISTERED_DOWNSAMPLED-80Hz.tif", 2
        )
    assert tr.movie.shape == (n, rows, cols)
