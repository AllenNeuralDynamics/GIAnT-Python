"""Tests of the across-trial alignment of SiLo's plan stage."""

import numpy as np
import pytest

from giant_python.extraction.raster.alignment import (
    TrialWarp,
    align_trials,
    dftregistration_clipped,
    interp2_linear,
    make_consensus_template,
    mask_to_consensus,
    matlab_round,
    template_samples,
    warp_trial_image,
    xcorr2_nans,
)

NAN = np.nan


def _scene(shape=(60, 80), n_blobs=25, seed=0):
    """A smooth image of Gaussian blobs, with a function to shift it."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform([8, 8], [shape[0] - 8, shape[1] - 8], (n_blobs, 2))
    amps = rng.uniform(1, 3, n_blobs)

    def render(shift=(0.0, 0.0)):
        """The scene moved by ``shift``: pixel p shows scene point
        p - shift."""
        rr, cc = np.mgrid[: shape[0], : shape[1]].astype(float)
        im = np.full(shape, 0.2)
        for (r0, c0), a in zip(centres, amps):
            im += a * np.exp(
                -((rr - r0 - shift[0]) ** 2 + (cc - c0 - shift[1]) ** 2)
                / (2 * 2.0**2)
            )
        return im

    return render


def test_matlab_round_halves_away_from_zero():
    """MATLAB rounds 2.5 to 3 and -2.5 to -3; the largest double below a
    half rounds down."""
    np.testing.assert_array_equal(
        matlab_round([2.5, -2.5, 3.5, 0.5, 0.49999999999999994, -0.2]),
        [3, -3, 4, 1, 0, 0],
    )


def test_interp2_linear_is_matlabs():
    """Values from MATLAB R2024b's interp2 on the same image: a point on a
    grid row or column ignores the next one (so a NaN there does not
    spread), fractional points next to a NaN are NaN, outside is NaN."""
    v = np.array(
        [[1, 2, 3], [4, NAN, 6], [7, 8, 9], [10, 11, 12]], dtype=float
    )
    rr, cc = np.mgrid[:4, :3].astype(float)
    np.testing.assert_array_equal(interp2_linear(v, rr, cc), v)
    np.testing.assert_array_equal(
        interp2_linear(v, rr + 0.25, cc),
        [
            [1.75, NAN, 3.75],
            [4.75, NAN, 6.75],
            [7.75, 8.75, 9.75],
            [NAN, NAN, NAN],
        ],
    )
    np.testing.assert_array_equal(
        interp2_linear(v, rr, cc + 0.5),
        [
            [1.5, 2.5, NAN],
            [NAN, NAN, NAN],
            [7.5, 8.5, NAN],
            [10.5, 11.5, NAN],
        ],
    )
    # the last row and column are inside; just beyond is not
    np.testing.assert_array_equal(
        interp2_linear(v, [3, 0, 0, 3.0000001], [2, 2, 2.0000001, 0]),
        [12, 3, NAN, NAN],
    )
    assert np.isnan(interp2_linear(v, NAN, 1.0))


def test_xcorr2_nans_finds_shifts():
    """A frame moved by a known amount against the template: the motion is
    the frame's shift, to a fraction of a pixel, and the correlation is
    near 1. NaN borders and a NaN hole do not matter."""
    render = _scene()
    template = render()
    template[:3] = NAN
    template[20:23, 30:34] = NAN
    for shift in [(2.0, -3.0), (0.4, 1.7), (-1.3, 0.25)]:
        frame = render(shift)
        frame[:, -2:] = NAN
        centre = matlab_round(shift)
        motion, r = xcorr2_nans(frame, template, centre, 4)
        np.testing.assert_allclose(motion, shift, atol=0.08)
        assert r > 0.99
    # a search that starts away from the shift still finds it
    motion, _ = xcorr2_nans(render((2.0, -3.0)), template, (0, 0), 4)
    np.testing.assert_allclose(motion, (2.0, -3.0), atol=0.05)


def test_xcorr2_nans_edge_of_search_and_errors():
    """A maximum on the edge of the search is returned whole (no parabola);
    shapes must agree."""
    render = _scene()
    motion, _ = xcorr2_nans(render((3.0, 0.0)), render(), (0, 0), 2)
    np.testing.assert_array_equal(motion, [2.0, 0.0])
    with pytest.raises(ValueError, match="frame is"):
        xcorr2_nans(np.zeros((4, 5)), np.zeros((5, 4)), (0, 0), 1)


def test_xcorr2_nans_wrapping_matches_the_rolled_masks():
    """Where shifted pixels leave the image the mask is rolled as MATLAB's
    circshift (the general path); it must agree with the index lookup on
    an image where nothing wraps, and run where it does."""
    render = _scene((30, 40), 12, seed=3)
    template, frame = render(), render((1.0, -1.0))
    motion, r = xcorr2_nans(frame, template, (1, -1), 2)
    np.testing.assert_allclose(motion, (1.0, -1.0), atol=0.05)
    # a centre beyond the margin makes the valid pixels wrap around
    wrapped, _ = xcorr2_nans(frame, template, (9, 0), 2)
    assert np.isfinite(wrapped).all()


def test_dftregistration_clipped_whole_shifts():
    """A circularly shifted image is registered by minus its shift, and a
    shift outside the clip window is not found."""
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(64, 48))
    moved = np.roll(ref, (5, -7), axis=(0, 1))
    out = dftregistration_clipped(np.fft.fft2(ref), np.fft.fft2(moved), 80)
    assert out[2:] == (-5, 7)
    assert out[0] < 1e-6  # identical up to the shift
    far = np.roll(ref, (20, 0), axis=(0, 1))
    out = dftregistration_clipped(np.fft.fft2(ref), np.fft.fft2(far), 10)
    assert abs(out[2]) <= 5 and abs(out[3]) <= 5


def test_dftregistration_clipped_odd_sizes_as_matlab():
    """On an odd axis MATLAB's midpoint test (``rloc > fix(m/2)`` on the
    one-based index) returns a shift of half the axis as its negative alias:
    values of dftregistration_clipped.m (R2024b) for a 5 x 7 and a 6 x 8
    image moved circularly."""
    rng = np.random.default_rng(4)
    matlab = {
        (5, 7): {
            (-2, 0): (-3, 0),  # the alias of +2 on 5 rows
            (2, 0): (-2, 0),
            (0, 3): (0, -3),
            (0, -3): (0, -4),  # the alias of +3 on 7 columns
            (1, -2): (-1, 2),
        },
        (6, 8): {
            (-3, 0): (-3, 0),
            (3, 0): (-3, 0),
            (0, 4): (0, -4),
            (0, -4): (0, -4),
        },
    }
    for shape, cases in matlab.items():
        ref = rng.normal(size=shape)
        for moved_by, expect in cases.items():
            moved = np.roll(ref, moved_by, axis=(0, 1))
            out = dftregistration_clipped(
                np.fft.fft2(ref), np.fft.fft2(moved), 80
            )
            assert out[2:] == expect, (shape, moved_by)


def test_template_samples():
    """Up to 20 kept trials, evenly spaced as MATLAB's
    round(linspace(1, n, 20))."""
    keep = np.array([True, False, True, True])
    np.testing.assert_array_equal(template_samples(keep), [0, 2, 3])
    many = template_samples(np.ones(50, bool))
    assert many.size == 20 and many[0] == 0 and many[-1] == 49
    # MATLAB: round(linspace(1, 50, 20)) starts 1 4 6 9 11
    np.testing.assert_array_equal(many[:5] + 1, [1, 4, 6, 9, 11])


def test_consensus_template_of_shifted_copies():
    """Copies of one scene are averaged on the reference; one image gives
    itself on the larger canvas."""
    render = _scene()
    shifts = [(0, 0), (1, 0), (0, -2), (1, 1), (-1, 2)]
    stack = np.stack([render(s) for s in shifts])
    stack[:, :2] = NAN
    template = make_consensus_template(stack, 5)
    assert template.shape == (70, 90)
    # the reference is one of the copies: the template shows the scene
    # moved by that copy's shift (the parabola fit of a whole shift is off
    # by a few thousandths of a pixel)
    best = min(
        np.nanmax(np.abs(template[5:-5, 5:-5] - render(s))) for s in shifts
    )
    assert best < 5e-3
    alone = make_consensus_template(stack[:1], 3)
    np.testing.assert_array_equal(alone[3:-3, 3:-3], stack[0])
    assert np.isnan(alone[:3]).all() and np.isnan(alone[:, -3:]).all()


def test_consensus_template_without_overlap_is_nan():
    """Images that share no pixels give no template."""
    a = np.full((20, 20), NAN)
    b = a.copy()
    a[:8] = 1.0
    b[12:] = 1.0
    assert np.isnan(make_consensus_template(np.stack([a, b]), 2)).all()


def test_trial_warp_coordinates():
    """Shifts and affine maps; MATLAB's one-based affine round-trips."""
    warp = TrialWarp.from_shifts([[0.5, -1.25], [NAN, NAN]])
    np.testing.assert_array_equal(warp.aligned, [True, False])
    assert warp.n_trials == 2 and warp.is_shift(0)
    np.testing.assert_array_equal(warp.shift(0), [0.5, -1.25])
    r_t, c_t = warp.coords(0, np.array([0.0, 3.0]), np.array([2.0, 4.0]))
    np.testing.assert_array_equal(r_t, [0.5, 3.5])
    np.testing.assert_array_equal(c_t, [0.75, 2.75])
    # MATLAB: [rT; cT] = A * [r; c; 1] in one-based coordinates
    a1 = np.array([[[1.01, 0.02, -0.4], [-0.03, 0.98, 0.7]]])
    affine = TrialWarp.from_matlab(a1)
    assert not affine.is_shift(0)
    with pytest.raises(ValueError, match="not a shift"):
        affine.shift(0)
    np.testing.assert_allclose(affine.to_matlab(), a1, atol=1e-15)
    r_t, c_t = affine.coords(0, 4.0, 9.0)
    expect = a1[0] @ [5.0, 10.0, 1.0] - 1
    np.testing.assert_allclose([r_t, c_t], expect, atol=1e-12)
    # a shift is the same in both conventions
    np.testing.assert_array_equal(
        TrialWarp.from_shifts([[0.5, -1.25]]).to_matlab()[0, :, 2],
        [0.5, -1.25],
    )


def test_patchwise_warp_is_not_ported():
    """A warp with patch centres is refused, not silently treated as its
    affine part."""
    warp = TrialWarp(np.zeros((1, 2, 3)), nodes=np.zeros((3, 2)))
    assert not warp.is_shift(0)
    with pytest.raises(NotImplementedError, match="patchwise"):
        warp.coords(0, 0.0, 0.0)


def test_warp_trial_image():
    """A whole shift moves the image exactly (NaN where it leaves the
    trial's image); channels are resampled alike; fractional shifts
    interpolate."""
    rng = np.random.default_rng(2)
    im = rng.normal(size=(6, 7))
    warp = TrialWarp.from_shifts([[1.0, -2.0], [0.5, 0.0]])
    out = warp_trial_image(im, warp, 0)
    np.testing.assert_array_equal(out[:5, 2:], im[1:, :5])
    assert np.isnan(out[5]).all() and np.isnan(out[:, :2]).all()
    two = warp_trial_image(np.stack([im, 2 * im], axis=2), warp, 0)
    np.testing.assert_array_equal(two[:, :, 0], out)
    np.testing.assert_array_equal(two[:, :, 1], 2 * out)
    half = warp_trial_image(im, warp, 1)
    np.testing.assert_allclose(half[:5], (im[:5] + im[1:]) / 2)
    with pytest.raises(NotImplementedError, match="cubic"):
        warp_trial_image(im, warp, 0, method="cubic")


def test_align_trials_recovers_shifts():
    """Trials that show one scene at known offsets are aligned to within a
    tenth of a pixel of each other; trials that are not kept, or without an
    activity image, are left out."""
    render = _scene((70, 90), 30, seed=5)
    true = np.array(
        [[0.0, 0.0], [1.3, -0.6], [-2.2, 1.9], [0.4, 3.1], [0, 0], [0, 0]]
    )
    m = np.stack([render(s) for s in true])
    m[:, :, :4] = NAN
    act = np.ones(m.shape)
    act[5] = NAN  # no activity image: not aligned
    keep = np.array([True, True, True, True, False, True])
    res = align_trials(m, act, keep, max_shift=5)
    assert res.template.shape == (80, 100)
    np.testing.assert_array_equal(res.samples, [0, 1, 2, 3, 5])
    assert np.isnan(res.shifts[4:]).all() and np.isnan(res.corr[4:]).all()
    # shifts are relative to the template's reference trial
    rel = res.shifts[:4] - res.shifts[0]
    np.testing.assert_allclose(rel, true[:4] - true[0], atol=0.1)
    assert (res.corr[:4] > 0.99).all()
    with pytest.raises(ValueError, match="no trials"):
        align_trials(m, act, np.zeros(6, bool))


def test_mask_to_consensus():
    """A mask drawn on a trial moves by minus the trial's shift, rounded as
    MATLAB's shiftRoiData (halves away from zero); pixels leaving the frame
    are dropped; an unknown or unaligned trial only crops or pads; an
    affine map takes the nearest trial pixel (roisToConsensusFrame)."""
    mask = np.zeros((6, 8), bool)
    mask[2, 3] = mask[5, 7] = True
    warp = TrialWarp.from_shifts(
        [[1.4, -2.5], [NAN, NAN], [0.0, 0.0], [-0.5, 0.5]]
    )
    out = mask_to_consensus(mask, warp, 0, (6, 8))
    # consensus (r, c) is trial (r + 1.4, c - 2.5): the mask moves by
    # round(-1.4) = -1 rows and round(2.5) = 3 columns
    assert out.sum() == 1 and out[1, 6]
    half = mask_to_consensus(mask, warp, 3, (6, 8))
    assert half.sum() == 1 and half[3, 2]  # +1 row, -1 column
    np.testing.assert_array_equal(
        mask_to_consensus(mask, warp, 2, (6, 8)), mask
    )
    for trial in (None, 1):
        padded = mask_to_consensus(mask, warp, trial, (7, 7))
        assert padded.shape == (7, 7) and padded.sum() == 1 and padded[2, 3]
    # an affine map that is nearly the shift (1, -2) of trial 0
    affine = TrialWarp([[[1.0, 1e-9, 1.0], [0.0, 1.0, -2.0]]])
    assert not affine.is_shift(0)
    moved = mask_to_consensus(mask, affine, 0, (6, 8))
    assert moved.sum() == 1 and moved[1, 5]
