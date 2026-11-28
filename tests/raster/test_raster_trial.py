"""Tests of loading a registered SLAP2 trial for raster localization."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from giant_python.extraction.raster.trial import (
    discard_frames,
    half_mean_images,
    load_registered_trial,
    trial_halves,
)

PLAN_FIXTURE = Path(__file__).parent / "data" / "matlab_plan.h5"


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


@pytest.mark.parametrize("save_tif", [False, True])
def test_load_registration_movie(tmp_path, save_tif):
    """Movies written by GIAnT-Python's registration, HDF5 (the default) or
    TIFF, load as the same frames in MATLAB's rows and columns."""
    if save_tif:
        pytest.importorskip("tifffile")
    from giant_python.registration.outputs import MovieWriter

    n, rows, cols = 50, 6, 9
    rng = np.random.default_rng(1)
    act = rng.poisson(5, (n, rows, cols)).astype(np.float32)
    act[3, 2, 4] = np.nan
    other = np.full((n, rows, cols), 100, np.float32)
    stem = "E1T1DMD1"
    name = f"{stem}_REGISTERED_DOWNSAMPLED-80Hz" + (
        ".tif" if save_tif else ".h5"
    )
    movie = MovieWriter(tmp_path / name, (rows, cols), 2 * n, save_tif)
    for a, o in zip(act, other):
        movie.write(a)
        movie.write(o)
    movie.close()
    movie.tmp.rename(movie.path)
    with h5py.File(tmp_path / f"{stem}_ALIGNMENTDATA.h5", "w") as f:
        f["alignHz"] = np.array([[80.0]])
        f["recNegErr"] = np.full((1, n), -0.5)
    tr = load_registered_trial(tmp_path, name, 2)
    np.testing.assert_array_equal(tr.movie, act)
    np.testing.assert_allclose(tr.mean_im[..., 1], 100)


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

    def flaky(mov, *args, **kwargs):
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


def test_trial_halves_match_matlab():
    """trialHalves.m of GIAnT-MATLAB: alternating blocks, then the larger
    half's last frames move over until the halves are equal to a frame."""
    cases = {
        (10, 3): [1, 1, 1, 2, 2, 2, 1, 1, 2, 2],
        (7, 2): [1, 1, 2, 2, 1, 1, 2],
        (12, 4): [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2],
        (9, 40): [1, 1, 1, 1, 1, 2, 2, 2, 2],
        (5, 1): [1, 2, 1, 2, 1],
        (1, 3): [1],
        (23, 5): [1] * 5 + [2] * 5 + [1] * 5 + [2] * 5 + [1, 1, 2],
    }
    for (n, block), matlab in cases.items():
        half = trial_halves(n, block)
        np.testing.assert_array_equal(half + 1, matlab)
        assert abs(int((half == 0).sum()) - int((half == 1).sum())) <= 1


def test_half_mean_images_match_matlab():
    """The half-trial mean images of MATLAB's loadAndProcessTrialAsync on
    a small movie with missing pixels (exactly: sums of whole counts)."""
    with h5py.File(PLAN_FIXTURE, "r") as f:
        g = f["halves"]
        stack, half, ref = g["stack"][()], g["half"][()], g["half_ims"][()]
        nan_thresh, block = g.attrs["nanThresh"], int(g.attrs["block_len"])
    np.testing.assert_array_equal(trial_halves(stack.shape[0], block), half)
    out = half_mean_images(stack, half, nan_thresh)
    assert out.dtype == np.float32 and np.isnan(ref).any()
    np.testing.assert_array_equal(out, ref)


def test_load_registered_trial_half_images(tmp_path):
    """Half images are returned only when asked for; their mean is the
    trial's mean image where nothing is missing."""
    n, rows, cols = 120, 5, 6
    rng = np.random.default_rng(3)
    data = rng.poisson(20, (2 * n, cols, rows)).astype(np.float32)
    stem = "E1T1DMD1"
    name = f"{stem}_REGISTERED_DOWNSAMPLED-80Hz.h5"
    with h5py.File(tmp_path / name, "w") as f:
        f["data"] = data
    with h5py.File(tmp_path / f"{stem}_ALIGNMENTDATA.h5", "w") as f:
        f["alignHz"] = np.array([[80.0]])
        f["recNegErr"] = np.full((1, n), -0.5)
    assert load_registered_trial(tmp_path, name, 2).half_ims is None
    tr = load_registered_trial(tmp_path, name, 2, half_ims=True)
    assert tr.half_ims.shape == (2, rows, cols, 2)
    # 40-frame blocks of 120 frames: the halves hold 60 frames each
    np.testing.assert_allclose(tr.half_ims.mean(axis=0), tr.mean_im, rtol=1e-6)
    stack = data.transpose(0, 2, 1).reshape(n, 2, rows, cols)
    first = np.r_[0:40, 80:100]
    np.testing.assert_allclose(
        tr.half_ims[0, :, :, 1], stack[first, 1].mean(axis=0), rtol=1e-6
    )


def test_channel_means_equal_the_joint_nanmean():
    """The mean image and the missing pixels, one channel at a time, are
    those of ``nanmean`` and ``isnan(...).mean`` over all channels at once,
    bit for bit (pixels without samples included)."""
    from giant_python.extraction.raster.trial import _channel_means

    rng = np.random.default_rng(3)
    stack = rng.gamma(2.0, 30.0, (97, 3, 11, 13)).astype(np.float32)
    stack[rng.random(stack.shape) < 0.3] = np.nan
    stack[:, :, 2, 5] = np.nan  # no sample in any channel
    stack[:, 1, 4, 4] = np.nan  # none in one channel
    with np.errstate(invalid="ignore"), pytest.warns(RuntimeWarning):
        want = np.nanmean(stack, axis=0).transpose(1, 2, 0)
    gaps = np.isnan(stack).mean(axis=(0, 1))
    with pytest.warns(RuntimeWarning):
        got, missing = _channel_means(stack, 0.33)
    assert got.dtype == want.dtype
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(missing, gaps > 0.33)
