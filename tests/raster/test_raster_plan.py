"""Tests of SiLo's path-level plan (alignment, sources, pixels, groups)."""

from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.special import erf

from giant_python.extraction.raster.plan import (
    ExtractionPlan,
    PlanParams,
    TrialImages,
    combined_activity_image,
    group_sources,
    plan_path,
    read_plan,
    write_plan,
)
from giant_python.extraction.raster.seeding import disk_strel

FIXTURE = Path(__file__).parent / "data" / "matlab_plan.h5"
NAN = np.nan

# sources of the synthetic path (scene coordinates): four apart, and a pair
# four pixels from each other
SOURCES = np.array(
    [[14, 16], [14, 60], [40, 30], [52, 70], [30, 50], [30, 54]], dtype=float
)
SHIFTS = np.array([[0, 0], [1, -1], [-1, 2], [2, 1], [0, -2]], dtype=float)


def _blobs(shape, centres, amps, sigma, shift):
    """Gaussian blobs at ``centres + shift``."""
    rr, cc = np.mgrid[: shape[0], : shape[1]].astype(float)
    im = np.zeros(shape)
    for (r0, c0), a in zip(centres, amps):
        im += a * np.exp(
            -((rr - r0 - shift[0]) ** 2 + (cc - c0 - shift[1]) ** 2)
            / (2 * sigma**2)
        )
    return im


def _peaks(shape, centres, amp, sigma, shift):
    """Narrow peaks at ``centres + shift``: Gaussians integrated over each
    pixel, the model that seeding fits (activity images are sharper than
    the PSF, and a sampled Gaussian would leave residual peaks)."""

    def axis(n, mu):
        """Integral of the Gaussian over each of n pixels."""
        edges = (np.arange(n + 1) - 0.5 - mu) / (np.sqrt(2) * sigma)
        return np.sqrt(np.pi / 2) * sigma * np.diff(erf(edges))

    im = np.zeros(shape)
    for r0, c0 in centres:
        im += amp * np.outer(
            axis(shape[0], r0 + shift[0]), axis(shape[1], c0 + shift[1])
        )
    return im


def _synthetic_trials(shape=(64, 84), seed=0):
    """Five trials of one scene at SHIFTS, with an imaged region, two
    channels, and activity images that peak at SOURCES."""
    rng = np.random.default_rng(seed)
    scene_c = rng.uniform([6, 6], [shape[0] - 6, shape[1] - 6], (40, 2))
    scene_a = rng.uniform(50, 150, 40)
    imaged = np.zeros(shape, bool)
    imaged[4:60, 5:80] = True
    trials = []
    for shift in SHIFTS:
        mean = 20 + _blobs(shape, scene_c, scene_a, 2.0, shift)
        act = _peaks(shape, SOURCES, 0.05, 0.6, shift)
        act += 0.0005 * np.abs(rng.normal(size=shape))
        mean[~imaged] = NAN
        act[~imaged] = NAN
        trials.append(
            TrialImages(
                np.stack([mean, 0.5 * mean], axis=2),
                act,
                peaks=np.array([[1.0, 2.0, 3.0, 0.4]]),
                discard=np.array([False, True, False]),
            )
        )
    return trials


@pytest.fixture(scope="module")
def synthetic_plan():
    """The plan of the synthetic path, and its intermediates."""
    details = {}
    plan = plan_path(
        _synthetic_trials(), params=PlanParams(peak_th=12.0), details=details
    )
    return plan, details


def test_plan_params():
    """Radii follow MATLAB (selRadius = ceil(2 dXY); zones of
    ceil(1.5 sigma + 1)); options are validated."""
    p = PlanParams()
    assert p.sel_radius == 6 and p.zone_radius == 3
    assert PlanParams(d_xy=2.2, sigma_px=2.0).sel_radius == 5
    assert PlanParams(sigma_px=2.0).zone_radius == 4
    with pytest.raises(ValueError, match="inter_trial_alignment"):
        PlanParams(inter_trial_alignment="elastic")
    with pytest.raises(ValueError, match="peak_method"):
        PlanParams(peak_method="blobs")
    with pytest.raises(ValueError, match="d_xy"):
        PlanParams(d_xy=0)


def test_group_sources_zones_and_order():
    """Sources whose zones (5 x 5 squares for radius 3) touch form a group;
    groups come in MATLAB's order, by the first pixel of their zone column
    by column; pixels are the group's disks within the selected pixels."""
    sel = np.ones((40, 60), bool)
    sel[:, 30] = False
    # (row, col): b and c touch through adjacent zones (5 columns apart);
    # a is 6 columns from b: its own group; d lies in an earlier column
    rows = np.array([20, 20, 20, 8])
    cols = np.array([20, 26, 31, 10])
    groups = group_sources(rows, cols, sel, zone_radius=3, sel_radius=6)
    assert [g.sources.tolist() for g in groups] == [[3], [0], [1, 2]]
    disk = disk_strel(6)
    first = groups[0]
    assert (first.row0, first.col0) == (3, 5)
    np.testing.assert_array_equal(first.sel, disk)
    np.testing.assert_array_equal(first.sources_rc, [[5, 5]])
    pair = groups[2]
    assert (pair.row0, pair.col0) == (15, 21)
    assert pair.sel.shape == (11, 16)
    np.testing.assert_array_equal(pair.sources_rc, [[5, 5], [5, 10]])
    assert not pair.sel[:, 30 - 21].any()  # the unselected column
    # the pixel lists address the selected-pixel list (column-major)
    cc, rr = np.nonzero(sel.T)
    for g in groups:
        assert (np.diff(g.pixels) > 0).all()
        crop = np.zeros(g.sel.shape, bool)
        crop[rr[g.pixels] - g.row0, cc[g.pixels] - g.col0] = True
        np.testing.assert_array_equal(crop, g.sel)
    assert group_sources(rows[:0], cols[:0], sel, 3, 6) == []
    with pytest.raises(ValueError, match="no selected pixels"):
        group_sources(rows, cols, np.zeros_like(sel), 3, 6)


def test_combined_activity_image():
    """The trials' mean (NaN where any trial is NaN), scaled to a maximum
    of 10 to 100, minus its 5 x 5 NaN-median."""
    rng = np.random.default_rng(1)
    act = rng.uniform(0.001, 0.002, (3, 12, 14))
    act[:, 6, 7] = 0.04
    act[1, 0, 0] = NAN
    im, med, nan_frac = combined_activity_image(act, 0.33)
    assert np.isnan(im[0, 0]) and nan_frac[0, 0] == pytest.approx(1 / 3)
    scaled = act.mean(axis=0) / 1e-3
    np.testing.assert_allclose(im[6, 7] + med[6, 7], 40.0)
    np.testing.assert_allclose(
        med[3, 3], np.median(scaled[1:6, 1:6]), rtol=1e-12
    )
    empty = combined_activity_image(act[:0], 0.33)
    assert all(np.isnan(x).all() for x in empty)


def test_plan_of_a_synthetic_path(synthetic_plan):
    """Shifts, valid trials, sources, selected pixels and groups of a path
    with known sources."""
    plan, details = synthetic_plan
    assert plan.shape == (64, 84) and plan.n_trials == 5
    np.testing.assert_array_equal(plan.valid_trials, np.arange(5))
    np.testing.assert_array_equal(plan.aligned_trials, np.arange(5))
    assert (plan.corr > 0.99).all() and plan.corr_thresh <= 0.9
    # shifts are relative to the template's reference trial
    np.testing.assert_allclose(
        plan.shifts - plan.shifts[0], SHIFTS - SHIFTS[0], atol=0.1
    )
    assert plan.n_sources == 6
    # each source is where it was put, in every trial's own frame
    for t in range(5):
        r_t, c_t = plan.warp.coords(t, plan.source_rows, plan.source_cols)
        found = np.column_stack([r_t, c_t])
        expect = SOURCES + SHIFTS[t]
        apart = np.abs(found[:, None] - expect[None]).max(axis=2)
        # a source is the nearest pixel of its seed (half a pixel off)
        assert (apart.min(axis=0) < 0.7).all()
        assert (apart.min(axis=1) < 0.7).all()
    # selected pixels: valid pixels within the disks; the union is sel
    assert plan.sel.sum() == plan.n_pixels
    assert not (plan.sel & ~plan.px_always_valid).any()
    union = np.zeros(plan.shape, bool)
    for k in range(plan.n_sources):
        mask = plan.source_mask(k)
        assert mask[plan.source_rows[k], plan.source_cols[k]]
        assert mask.sum() == plan.source_pixels[k].size
        union |= mask
    np.testing.assert_array_equal(union, plan.sel)
    rows, cols = plan.selected_pixels()
    assert (np.diff(rows + cols * plan.shape[0]) > 0).all()
    # five groups: the pair shares one; every source is in exactly one
    assert sorted(g.sources.size for g in plan.groups) == [1, 1, 1, 1, 2]
    np.testing.assert_array_equal(
        np.sort(np.concatenate([g.sources for g in plan.groups])),
        np.arange(6),
    )
    for g in plan.groups:
        members = np.zeros(plan.shape, bool)
        for k in g.sources:
            members |= plan.source_mask(k)
        crop = members[
            g.row0 : g.row0 + g.sel.shape[0], g.col0 : g.col0 + g.sel.shape[1]
        ]
        np.testing.assert_array_equal(crop, g.sel)
        assert members.sum() == g.sel.sum() == g.pixels.size
        np.testing.assert_array_equal(
            g.sources_rc,
            np.column_stack(
                [
                    plan.source_rows[g.sources] - g.row0,
                    plan.source_cols[g.sources] - g.col0,
                ]
            ),
        )
    # images: the mean of the aligned trials, NaN ignored
    assert plan.mean_im.shape == (64, 84, 2)
    np.testing.assert_allclose(
        plan.mean_im[30, 40], np.nanmean(plan.mean_aligned[:, 30, 40], axis=0)
    )
    assert details["keep_sources"].all() and details["rows"].size == 6
    assert details["med_im"].shape == plan.shape


def test_plan_leaves_out_bad_and_missing_trials():
    """A trial that was not localized, one that is not kept, one without an
    activity image and one that does not match the others are not valid;
    sources are seeded from the rest."""
    trials = _synthetic_trials()
    shape = trials[0].act_im.shape
    noise = np.random.default_rng(7).uniform(10, 30, shape)
    noise[np.isnan(trials[0].act_im)] = NAN  # nothing like the scene
    odd = TrialImages(np.stack([noise, noise], axis=2), trials[0].act_im)
    blank = TrialImages(trials[1].mean_im, np.full(shape, NAN))
    everything = trials + [None, odd, blank, trials[2]]
    keep = np.ones(len(everything), bool)
    keep[8] = False
    plan = plan_path(
        everything, params=PlanParams(peak_th=12.0), keep_trials=keep
    )
    np.testing.assert_array_equal(plan.keep_trials, keep & (np.arange(9) != 5))
    np.testing.assert_array_equal(plan.valid_trials, np.arange(5))
    np.testing.assert_array_equal(plan.aligned_trials, [0, 1, 2, 3, 4, 6])
    assert plan.corr[6] < plan.corr_thresh < 0.9
    assert np.isnan(plan.shifts[[5, 7, 8]]).all()
    assert np.isnan(plan.act_aligned[[5, 7, 8]]).all()
    assert plan.n_sources == 6
    assert plan.trial_peaks[5] is None and plan.trial_peaks[0] is not None
    with pytest.raises(ValueError, match="no trials"):
        plan_path([None, None])


def test_plan_soma_mask_and_given_seeds():
    """No source is seeded in the soma mask (smaller masks sit top left);
    given seeds replace the seeding; modes that are not ported are
    refused."""
    trials = _synthetic_trials()
    soma = np.zeros((40, 45), bool)
    soma[:25, :30] = True  # covers the source at (14, 16)
    plan = plan_path(trials, soma, PlanParams(peak_th=12.0))
    assert plan.n_sources == 5
    assert plan.soma_mask.shape == plan.shape and plan.soma_mask.sum() == 750
    assert not plan.soma_mask[plan.source_rows, plan.source_cols].any()
    seeds = np.array(
        [[3.0, 20.2, 30.6, 0.5], [2.0, 20.4, 31.4, 0.5], [1.0, 2.0, 2.0, 0.5]]
    )
    given = plan_path(trials, seeds=seeds)
    # the first two round to one pixel; the last is outside the imaged area
    np.testing.assert_array_equal(given.source_rows, [20])
    np.testing.assert_array_equal(given.source_cols, [31])
    np.testing.assert_array_equal(given.seeds, seeds)
    # the same mask drawn on trial 3, whose image is the scene moved by
    # SHIFTS[3]: it lands on the same sources of the consensus frame
    drawn = np.zeros(plan.shape, bool)
    rr, cc = np.nonzero(plan.soma_mask)
    move = np.round(plan.shifts[3]).astype(int)
    drawn[rr + move[0], cc + move[1]] = True
    moved = plan_path(
        trials, drawn, PlanParams(peak_th=12.0), soma_mask_trial=3
    )
    np.testing.assert_array_equal(moved.soma_mask, plan.soma_mask)
    np.testing.assert_array_equal(moved.source_rows, plan.source_rows)
    with pytest.warns(UserWarning, match="not aligned"):
        loose = plan_path(
            trials + [None], drawn, PlanParams(peak_th=12.0), soma_mask_trial=5
        )
    np.testing.assert_array_equal(loose.soma_mask, drawn)
    with pytest.raises(ValueError, match="soma_mask"):
        plan_path(trials, np.zeros((100, 100), bool))
    with pytest.raises(NotImplementedError, match="patchwise"):
        plan_path(trials, params=PlanParams(inter_trial_alignment="patchwise"))
    with pytest.raises(NotImplementedError, match="WMF"):
        plan_path(trials, params=PlanParams(peak_method="wmf"))


def test_plan_of_trials_of_different_sizes(synthetic_plan):
    """Trials of different sizes share one canvas (top left, NaN beyond),
    as in MATLAB; their mean images must have the same channels."""
    plan, _ = synthetic_plan
    trials = _synthetic_trials()
    # nothing is imaged in the rows and columns cut off here
    trials[4] = TrialImages(
        trials[4].mean_im[:60, :80], trials[4].act_im[:60, :80]
    )
    small = plan_path(trials, params=PlanParams(peak_th=12.0))
    assert small.shape == plan.shape
    np.testing.assert_array_equal(small.shifts, plan.shifts)
    np.testing.assert_array_equal(small.act_im, plan.act_im)
    np.testing.assert_array_equal(small.source_rows, plan.source_rows)
    np.testing.assert_array_equal(small.sel, plan.sel)
    trials[1] = TrialImages(trials[1].mean_im[:, :, :1], trials[1].act_im)
    with pytest.raises(ValueError, match="channels"):
        plan_path(trials)


def test_plan_with_a_trial_without_images(synthetic_plan):
    """A kept trial whose images are all NaN (nothing was imaged often
    enough) does not enter the template, so the other trials are planned
    as without it; in MATLAB it would leave the path without sources."""
    plan, _ = synthetic_plan
    trials = _synthetic_trials()
    shape = trials[0].act_im.shape
    blank = TrialImages(np.full(shape + (2,), NAN), np.full(shape, NAN))
    with_blank = plan_path(
        trials[:2] + [blank] + trials[2:], params=PlanParams(peak_th=12.0)
    )
    rest = [0, 1, 3, 4, 5]
    assert with_blank.keep_trials.all()
    np.testing.assert_array_equal(with_blank.template_trials, rest)
    np.testing.assert_array_equal(with_blank.valid_trials, rest)
    np.testing.assert_array_equal(with_blank.shifts[rest], plan.shifts)
    assert np.isnan(with_blank.shifts[2]).all()
    np.testing.assert_array_equal(with_blank.act_im, plan.act_im)
    np.testing.assert_array_equal(with_blank.source_rows, plan.source_rows)
    np.testing.assert_array_equal(with_blank.source_cols, plan.source_cols)
    with pytest.raises(ValueError, match="mean image"):
        plan_path([blank, blank])


def test_plan_without_sources():
    """A path whose activity images hold nothing above threshold has no
    sources, pixels or groups, and still writes and reads."""
    trials = _synthetic_trials()
    plan = plan_path(trials, params=PlanParams(peak_th=1e6))
    assert plan.n_sources == 0 and plan.n_pixels == 0 and plan.groups == []
    assert plan.seeds.shape == (0, 4)


def _assert_plans_equal(a: ExtractionPlan, b: ExtractionPlan) -> None:
    """Every field of two plans is equal (NaN equal to NaN)."""
    assert a.params == b.params
    assert a.corr_thresh == b.corr_thresh
    for name in (
        "keep_trials",
        "template",
        "template_trials",
        "shifts",
        "corr",
        "act_valid_pix",
        "valid_trials",
        "mean_ims",
        "act_ims",
        "mean_aligned",
        "act_aligned",
        "mean_im",
        "act_im",
        "soma_mask",
        "px_always_valid",
        "seeds",
        "source_rows",
        "source_cols",
        "source_values",
        "sel",
    ):
        x, y = getattr(a, name), getattr(b, name)
        assert x.dtype.kind == y.dtype.kind, name
        np.testing.assert_array_equal(x, y, err_msg=name)
    np.testing.assert_array_equal(a.warp.affine, b.warp.affine)
    np.testing.assert_array_equal(a.warp.nodes, b.warp.nodes)
    np.testing.assert_array_equal(a.warp.disp, b.warp.disp)
    assert len(a.source_pixels) == len(b.source_pixels)
    for x, y in zip(a.source_pixels, b.source_pixels):
        np.testing.assert_array_equal(x, y)
    assert len(a.groups) == len(b.groups)
    for x, y in zip(a.groups, b.groups):
        assert (x.row0, x.col0) == (y.row0, y.col0)
        for name in ("sources", "pixels", "sel", "sources_rc"):
            np.testing.assert_array_equal(getattr(x, name), getattr(y, name))
    for x, y in zip(
        a.trial_peaks + a.trial_discard, b.trial_peaks + b.trial_discard
    ):
        assert (x is None) == (y is None)
        if x is not None:
            np.testing.assert_array_equal(x, y)


def test_write_and_read_plan(tmp_path, synthetic_plan):
    """A plan survives the file; several paths share a file by group; a
    file that is not a plan, or a newer version, is refused."""
    plan, _ = synthetic_plan
    fn = tmp_path / "extraction_plan.h5"
    write_plan(fn, plan)
    back = read_plan(fn)
    _assert_plans_equal(plan, back)
    assert back.trial_discard[0].dtype == bool
    # an empty plan, written next to it as a second imaging path
    empty = plan_path(
        _synthetic_trials()[:2] + [None], params=PlanParams(peak_th=1e6)
    )
    write_plan(fn, plan, group="DMD1")
    write_plan(fn, empty, group="DMD2")
    write_plan(fn, empty, group="DMD2")  # replaces the group
    _assert_plans_equal(read_plan(fn, "DMD1"), plan)
    _assert_plans_equal(read_plan(fn, "DMD2"), empty)
    with h5py.File(fn, "r+") as f:
        f["DMD2"].attrs["version"] = 99
        f.create_group("other")
    with pytest.raises(ValueError, match="version 99"):
        read_plan(fn, "DMD2")
    with pytest.raises(ValueError, match="does not hold"):
        read_plan(fn, "other")


def _matlab_case(case):
    """The inputs of one fixture case and MATLAB's results on them.

    The fixture holds a 72 x 96 window of five trials of one imaging path
    (780125) and what ``SILo.m`` and ``extractTrial.m`` (GIAnT-MATLAB main
    5a0673b) compute from it, in Python's orientation and zero-based.
    """
    with h5py.File(FIXTURE, "r") as f:
        mean = f["inputs/mean_ims"][()].astype(np.float64)
        act = f["inputs/act_ims"][()]
        g = f["matlab"][case]
        ref = {k: g[k][()] for k in g}
        ref.update({k: g.attrs[k] for k in g.attrs})
        a = f["inputs"].attrs
        params = PlanParams(
            nan_thresh=float(a["nanThresh"]),
            peak_th=float(ref["peakth"]),
            min_peak_distance=int(a["minPeakDistance"]),
            d_xy=float(a["dXY"]),
            sigma_px=float(a["sigma_px"]),
        )
    keep = ref["keep_trials"].astype(bool)
    trials = [
        TrialImages(mean[t], act[t]) if keep[t] else None
        for t in range(keep.size)
    ]
    return trials, params, ref


def _lists(ref, name):
    """Index lists of the fixture (``<name>_index`` and ``<name>_ptr``)."""
    index, ptr = ref[f"{name}_index"], ref[f"{name}_ptr"]
    return [index[a:b] for a, b in zip(ptr[:-1], ptr[1:])]


@pytest.mark.parametrize("case", ["all_trials", "one_trial_out"])
def test_plan_matches_matlab(case):
    """Every step but the Gaussian fit of the seeds reproduces MATLAB: the
    template, shifts, valid trials, aligned and combined images to rounding
    error; given MATLAB's seeds, the sources, their pixels and the
    subproblems exactly."""
    trials, params, ref = _matlab_case(case)
    details = {}
    plan = plan_path(
        trials, params=params, seeds=ref["seeds"], details=details
    )
    close = dict(rtol=0, atol=1e-9)
    np.testing.assert_allclose(plan.template, ref["template"], **close)
    np.testing.assert_array_equal(plan.template_trials, ref["template_trials"])
    np.testing.assert_allclose(plan.shifts, ref["shifts"], **close)
    np.testing.assert_allclose(plan.corr, ref["corr"], **close)
    assert plan.corr_thresh == pytest.approx(ref["corr_thresh"], abs=1e-12)
    np.testing.assert_allclose(
        plan.act_valid_pix, ref["act_valid_pix"], **close
    )
    np.testing.assert_array_equal(plan.valid_trials, ref["valid_trials"])
    np.testing.assert_allclose(plan.act_aligned, ref["act_aligned"], **close)
    np.testing.assert_allclose(plan.act_im, ref["act_im"], **close)
    if "mean_aligned" in ref:
        np.testing.assert_allclose(
            plan.mean_aligned, ref["mean_aligned"], rtol=1e-12, atol=1e-9
        )
        np.testing.assert_allclose(
            plan.mean_im, ref["mean_im"], rtol=1e-12, atol=1e-9
        )
        np.testing.assert_allclose(details["med_im"], ref["med_im"], **close)
    # from MATLAB's seeds on: exact
    np.testing.assert_array_equal(
        details["keep_sources"], ref["keep_sources"].astype(bool)
    )
    np.testing.assert_array_equal(plan.source_rows, ref["source_rows"])
    np.testing.assert_array_equal(plan.source_cols, ref["source_cols"])
    np.testing.assert_array_equal(plan.source_values, ref["source_values"])
    np.testing.assert_array_equal(
        plan.px_always_valid, ref["px_always_valid"].astype(bool)
    )
    np.testing.assert_array_equal(plan.sel, ref["sel"].astype(bool))
    for mine, matlab in zip(plan.source_pixels, _lists(ref, "source_pixel")):
        np.testing.assert_array_equal(mine, matlab)
    assert len(plan.groups) == ref["group_rows"].shape[0]
    for g, src, pix, rows, cols in zip(
        plan.groups,
        _lists(ref, "group_source"),
        _lists(ref, "group_pixel"),
        ref["group_rows"],
        ref["group_cols"],
    ):
        np.testing.assert_array_equal(g.sources, src)
        np.testing.assert_array_equal(g.pixels, pix)
        assert (g.row0, g.row0 + g.sel.shape[0] - 1) == tuple(rows)
        assert (g.col0, g.col0 + g.sel.shape[1] - 1) == tuple(cols)


@pytest.mark.parametrize(
    "case, same_pixel", [("all_trials", 0.85), ("one_trial_out", 1.0)]
)
def test_own_seeds_are_close_to_matlabs(case, same_pixel):
    """Seeding MATLAB's combined activity image with this package's fit
    gives nearly MATLAB's sources. They are not identical: the Gaussian fit
    is ill-conditioned for peaks narrower than a pixel and MATLAB's
    lsqcurvefit stops on a tolerance or on its evaluation limit, so where
    it ends depends on the optimizer (at a threshold of 3 SDs MATLAB finds
    21 sources in this window and this package 20)."""
    trials, params, ref = _matlab_case(case)
    plan = plan_path(trials, params=params)
    matlab = np.column_stack([ref["source_rows"], ref["source_cols"]])
    mine = np.column_stack([plan.source_rows, plan.source_cols])
    apart = np.abs(matlab[:, None] - mine[None]).max(axis=2)
    assert abs(len(mine) - len(matlab)) <= 1
    assert (apart.min(axis=1) == 0).mean() >= same_pixel
    assert (apart.min(axis=1) <= 1).all() and (apart.min(axis=0) <= 1).all()
