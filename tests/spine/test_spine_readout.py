"""Tests of the spine-size windows and readout (:mod:`giant_python.spine.rois`
and :mod:`giant_python.spine.readout`)."""

import numpy as np
import pytest

from giant_python.spine.readout import (
    common_valid,
    mask_mean,
    norm_log,
    project,
)
from giant_python.spine.rois import (
    disks_roiset,
    masks_roiset,
    nearest,
    read_marks,
    sources_roiset,
    write_marks,
)

from . import synthetic as syn


def test_sources_roiset_is_sparse_and_centred():
    """Footprints are stored as their positive pixels, centroids weighted."""
    fp = syn.footprints()
    rs = sources_roiset(fp)
    assert len(rs) == len(syn.SOURCES)
    for k, (r0, c0) in enumerate(syn.SOURCES):
        np.testing.assert_allclose(rs.image(k), fp[k])
        assert abs(rs.rows[k] - r0) < 0.05 and abs(rs.cols[k] - c0) < 0.05
        idx, w = rs.pixels(k)
        assert (w > 0).all() and idx.size == (fp[k] > 0).sum()


def test_masks_roiset_needs_one_label_per_mask():
    """Labels and masks must agree in number."""
    with pytest.raises(ValueError, match="labels"):
        masks_roiset(np.zeros((2, 4, 4), bool), ["a"])


def test_project_is_the_matlab_formula():
    """``F = (h . x) / (h . h)`` with h normalized over the valid pixels."""
    rng = np.random.default_rng(2)
    fp = syn.footprints()
    imgs = 100 + rng.random((5, syn.H, syn.W))
    valid = np.ones((syn.H, syn.W), bool)
    valid[8, 9] = False  # the first source's peak pixel is left out
    rs = sources_roiset(fp)
    got, n_pix = project(imgs, rs, valid)
    for k in range(len(rs)):
        h = np.where(valid, fp[k], 0).ravel()
        h = h / h.sum()
        expect = imgs.reshape(5, -1) @ h / (h @ h)
        np.testing.assert_allclose(got[:, k], expect)
    assert n_pix[0] == (fp[0] > 0).sum() - 1


def test_project_is_scale_equivariant():
    """A uniform brightness change scales every readout by the same
    factor, so it cancels in the normalization."""
    fp = syn.footprints()
    img = syn.scene(1)[None]
    rs = sources_roiset(fp)
    valid = np.ones(img.shape[1:], bool)
    a, _ = project(img, rs, valid)
    b, _ = project(0.8 * img, rs, valid)
    np.testing.assert_allclose(b, 0.8 * a)


def test_common_valid_requires_every_trial_and_stack():
    """A pixel missing in one trial of one stack is not valid."""
    a = np.ones((3, 4, 5))
    b = np.ones((3, 4, 5))
    a[1, 0, 0] = np.nan
    b[2, 3, 4] = np.nan
    ok = common_valid([a, b], [0, 1, 2])
    assert not ok[0, 0] and not ok[3, 4] and ok.sum() == 18
    assert common_valid([a, b], [0])[0, 0]


def test_mask_mean_and_coverage():
    """Mean over the mask's pixels with a value; coverage is their share."""
    masks = np.zeros((1, 4, 4), bool)
    masks[0, :2, :2] = True
    rs = masks_roiset(masks, ["r"])
    imgs = np.ones((2, 4, 4)) * np.array([1.0, 3.0])[:, None, None]
    imgs[1, 0, 0] = np.nan
    v, cov = mask_mean(imgs, rs)
    np.testing.assert_allclose(v[:, 0], [1.0, 3.0])
    np.testing.assert_allclose(cov[:, 0], [1.0, 0.75])


def test_norm_log_matches_spinetrends():
    """log F, minus each trial's median over the ok ROIs, minus each ROI's
    mean (``normLog`` of ``spineTrends.m``)."""
    rng = np.random.default_rng(3)
    f = np.exp(rng.normal(size=(7, 9)))
    ok = np.ones(9, bool)
    ok[4] = False
    lg = np.log(f)
    expect = lg - np.median(lg[:, ok], axis=1, keepdims=True)
    expect = expect - expect.mean(axis=0, keepdims=True)
    np.testing.assert_allclose(norm_log(f, ok), expect)


def test_norm_log_removes_a_shared_trend():
    """A brightness factor shared by all ROIs (bleaching) leaves zeros."""
    sizes = np.array([1.0, 2.0, 3.0])
    trend = np.array([1.0, 0.9, 0.8, 0.7])
    out = norm_log(trend[:, None] * sizes[None])
    np.testing.assert_allclose(out, 0, atol=1e-12)


def test_nearest_and_disks():
    """Points find the nearest ROI within a distance."""
    rs = disks_roiset([5.0, 10.0], [5.0, 20.0], 2.0, (20, 30))
    assert rs.meta["radius"] == 2.0
    np.testing.assert_array_equal(
        nearest(rs, [5.5, 10.0, 0.0], [5.0, 21.0, 0.0], 2.5), [0, 1, -1]
    )


def test_marks_round_trip(tmp_path):
    """Marks keep GIAnT-MATLAB's spine_marks.csv columns."""
    marks = [
        {
            "Path": 1,
            "Source": 59,
            "Label": "Growing",
            "Note": "",
            "Row0": 72.5,
            "Col0": 585.45,
            "Updated": "2026-10-30 15:54:36",
        }
    ]
    path = tmp_path / "spine_marks.csv"
    write_marks(path, marks)
    back = read_marks(path)
    assert back[0]["Path"] == 1 and back[0]["Label"] == "Growing"
    assert back[0]["Row0"] == pytest.approx(72.5)
