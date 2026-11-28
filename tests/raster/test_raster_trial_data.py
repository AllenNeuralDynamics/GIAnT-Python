"""Tests of the SLAP2 trial loader (SILo's loadTrial and its helpers).

The helpers are checked against values computed by MATLAB R2024b
(``data/matlab_trial_data.mat``, made by ``data/make_matlab_trial_data.m``),
the frames against slow loops written from the definition, and the whole
loader on small synthetic recordings of a static scene under known motion.
"""

from pathlib import Path

import numpy as np
import pytest
import scipy.io
import torch
from scipy.ndimage import map_coordinates

from giant_python.extraction.raster import trial_data
from giant_python.extraction.raster.trial_data import (
    OnlineCompensation,
    RigidShift,
    TrialAlignment,
    TrialLoadParams,
    block_compensation,
    block_edges,
    compensation_pad,
    finish_pixels,
    frame_lines,
    frames_at,
    labeled_pixels,
    load_trial,
    medfilt3,
    movmean,
    nansvd,
    online_compensation,
    read_alignment,
    roi_svd_trace,
    upsample_discard,
    upsample_motion,
)
from giant_python.io.slap2_raster import read_raster_samples
from giant_python.registration import multiroi
from giant_python.registration.compensation import MAPPING, line_offsets
from giant_python.registration.frames import FrameFormer, PixelFormer
from tests.registration.synthetic import (
    make_layout,
    online_offsets,
    scene,
    write_recording,
)

DATA = Path(__file__).parent / "data" / "matlab_trial_data.mat"
MARGIN = 3  # rows and columns of the registered frame around the raster


@pytest.fixture(scope="module")
def ref():
    """Values computed by MATLAB."""
    return scipy.io.loadmat(DATA)


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    """Layout, samples and true motion of a short recording."""
    path = tmp_path_factory.mktemp("rec") / "synthetic_DMD1-TRIAL000001.dat"
    layout = make_layout(path, n_cycles=500, seed=3)
    truth = write_recording(layout, seed=4, invalid=0.02)
    samples = read_raster_samples(layout, (0, layout.num_cycles))
    return layout, samples, truth


@pytest.fixture(scope="module")
def online_recording(tmp_path_factory):
    """A recording whose online offsets move the DMD pattern."""
    path = tmp_path_factory.mktemp("rec") / "online_DMD1-TRIAL000001.dat"
    layout = make_layout(path, n_cycles=1400, seed=5)
    online = online_offsets(layout.total_lines)
    truth = write_recording(layout, seed=6, online=online)
    samples = read_raster_samples(layout, (0, layout.num_cycles))
    return layout, samples, truth, online


class Noiseless:
    """Stands in for the generator of ``write_recording``: samples are 20
    times their expected count instead of Poisson draws."""

    def __init__(self, seed):
        """Keep a real generator for the other draws."""
        self._rng = np.random.Generator(np.random.PCG64(seed))

    def poisson(self, lam):
        """The expected count, scaled to keep whole-number rounding small."""
        return np.round(20 * lam)

    def random(self, n):
        """Uniform draws (which samples are invalid)."""
        return self._rng.random(n)


@pytest.fixture(scope="module")
def quiet_online_recording(tmp_path_factory):
    """A noise-free recording whose online offsets move the DMD pattern:
    what varies in its motion-corrected samples is motion-correction
    error (as GIAnT-MATLAB's TestCompensatedExtraction)."""
    from tests.registration import synthetic

    path = tmp_path_factory.mktemp("rec") / "quiet_DMD1-TRIAL000001.dat"
    layout = make_layout(path, n_cycles=900, seed=5)
    online = online_offsets(layout.total_lines)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(synthetic.np.random, "default_rng", Noiseless)
        write_recording(layout, seed=6, online=online)
    return layout


# ---------------------------------------------------------------- MATLAB


def test_movmean_matches_smoothdata(ref):
    """Even, odd and fractional windows are placed as smoothdata's."""
    x = ref["mm_x"]
    for w, want in zip(ref["mm_w"].ravel(), ref["mm_y"]):
        np.testing.assert_allclose(movmean(x, w)[0], want, rtol=1e-13)
    full = np.arange(12.0)[None, :] ** 2  # the branch without gaps
    gap = full.copy()
    gap[0, 3] = np.nan
    want = movmean(gap, 4)[0]
    keep = np.r_[0:2, 6:12]  # windows that do not hold the gap
    np.testing.assert_allclose(movmean(full, 4)[0][keep], want[keep])
    assert np.isnan(movmean(np.full((1, 5), np.nan), 3)).all()


def test_medfilt3_and_labeled_pixels(ref):
    """medfilt2 is NaN wherever its window holds a NaN; zero padding."""
    got = medfilt3(ref["med_in"])
    np.testing.assert_array_equal(got, ref["med_out"])
    np.testing.assert_array_equal(
        labeled_pixels(ref["med_in"]), ref["labeled"].astype(bool)
    )
    assert not labeled_pixels(np.full((4, 4), np.nan)).any()


def test_upsample_motion_matches_interp1_pchip(ref):
    """pchip with extrapolation, as interp1; frames without motion left
    out; one frame's motion used throughout."""
    x, y = ref["pchip_x"].ravel(), ref["pchip_y"].ravel()
    q = ref["pchip_q"].ravel()
    got = upsample_motion(x, y, q)
    np.testing.assert_allclose(got, ref["pchip_v"].ravel(), rtol=1e-12)
    y_gap = y.copy()
    y_gap[3] = np.nan  # a registered frame without samples
    keep = np.arange(y.size) != 3
    np.testing.assert_array_equal(
        upsample_motion(x, y_gap, q), upsample_motion(x[keep], y[keep], q)
    )
    one = upsample_motion(x[:1], y[:1], q)
    np.testing.assert_array_equal(one, np.full(q.size, y[0]))
    with pytest.raises(ValueError, match="no motion"):
        upsample_motion(x, np.full(x.size, np.nan), q)


def test_upsample_discard_and_block_edges(ref):
    """A frame is discarded if either registered frame around it is."""
    d = ref["disc_in"].ravel() > 0
    want = ref["disc_out"].ravel() > 0
    np.testing.assert_array_equal(upsample_discard(d, want.size), want)
    assert upsample_discard(np.array([True]), 4).all()
    assert not upsample_discard(np.array([False]), 4).any()
    np.testing.assert_array_equal(
        block_edges(5633, 600), ref["blocks"].ravel() - 1
    )
    np.testing.assert_array_equal(block_edges(10, 600), [0, 10])


def test_frame_lines_are_ceil_of_matlab_colon():
    """``ceil(first:dt:last)``, with fractional line ranges."""
    lines = frame_lines(1.0, 300974.0, 53.43333333333334)
    assert lines.size == 5633 and lines[0] == 1
    assert lines[-1] <= 300974 < lines[-1] + 54
    np.testing.assert_array_equal(frame_lines(2.5, 20.0, 4.0)[:3], [3, 7, 11])


def test_roi_svd_trace_matches_matlab(ref):
    """nansvd and the component choice of loadTrial."""
    d = ref["svd_d"]
    u, s, v, bg = nansvd(d, 3, 10, 0.33)
    np.testing.assert_allclose(s, ref["svd_s"].ravel(), rtol=1e-8)
    np.testing.assert_allclose(bg, ref["svd_bg"], rtol=1e-12)
    np.testing.assert_allclose(
        (u * s) @ v.T, ref["svd_recon"], rtol=1e-6, atol=1e-8
    )
    np.testing.assert_allclose(
        roi_svd_trace(d, 0.33), ref["svd_trace"].ravel(), rtol=1e-8
    )
    # a one-pixel ROI has one component; the others are padded
    np.testing.assert_allclose(
        roi_svd_trace(d[4:5], 0.33), ref["svd1_trace"].ravel(), rtol=1e-8
    )
    u1, s1, _, _ = nansvd(d[4:5], 3, 10, 0.33)
    assert np.isnan(u1[:, 1:]).all() and np.all(s1[1:] == 0)
    # nothing usable: every pixel misses too many frames, or no data
    assert np.isnan(roi_svd_trace(np.full((3, 20), np.nan), 0.33)).all()
    assert np.isnan(roi_svd_trace(np.zeros((0, 20)), 0.33)).all()
    sparse = np.full((3, 20), np.nan)
    sparse[:, ::2] = 1.0
    assert np.isnan(roi_svd_trace(sparse, 0.33)).all()


def test_nansvd_selects_until_the_choice_is_stable():
    """Rows and columns are left out until the choice no longer changes:
    leaving a column out can bring a row back, and the rows and columns
    kept each miss less than the threshold among the other kept ones."""
    rng = np.random.default_rng(3)
    d = rng.standard_normal((8, 30)) + 5
    d[0, :12] = np.nan  # misses 40%: out, until ...
    d[:5, :6] = np.nan  # ... columns 0-5, missing in most rows, go
    u, s, v, bg = nansvd(d, 3, 10, 0.33)
    rows, cols = ~np.isnan(u[:, 0]), ~np.isnan(v[:, 0])
    assert not cols[:6].any() and cols[6:].all()
    gaps = np.isnan(d)
    assert (gaps[np.ix_(rows, cols)].mean(1) < 0.33).all()
    assert (gaps[np.ix_(rows, cols)].mean(0) < 0.33).all()
    assert (gaps[~rows][:, cols].mean(1) >= 0.33).all() or rows.all()
    assert np.isfinite(s).all() and np.isnan(bg[:, :6]).all()


@pytest.mark.parametrize(
    "dtype, rtol", [(np.float64, 1e-12), (np.float32, 2e-6)]
)
@pytest.mark.parametrize("rows", [2, 512])
def test_finish_pixels_matches_matlab(ref, dtype, rtol, rows):
    """Discards, outliers and gap filling as at the end of loadTrial: a
    pixel without samples, gaps longer than the window (15 passes over
    everything), and the variance multiplier of filled samples."""
    y = ref["fin_y_in"].transpose(1, 0, 2).astype(dtype)  # channels first
    finv = ref["fin_finv_in"].astype(dtype)
    discard = upsample_discard(ref["fin_discard_in"].ravel() > 0, y.shape[2])
    np.testing.assert_array_equal(discard, ref["fin_discard"].ravel() > 0)
    assert int(ref["fin_passes"].ravel()[0]) == 15
    finish_pixels(y, finv, discard, float(ref["fin_window"].ravel()[0]), rows)
    np.testing.assert_allclose(
        y, ref["fin_y"].transpose(1, 0, 2), rtol=rtol, atol=rtol
    )
    np.testing.assert_allclose(finv, ref["fin_finv"], rtol=rtol)
    assert not np.isnan(y).any() and not np.isnan(finv).any()


def test_finish_pixels_without_gaps_changes_nothing():
    """No discards and no missing samples: data and multiplier are kept."""
    rng = np.random.default_rng(0)
    y = 5 + rng.random((2, 7, 40))
    finv = 1 + rng.random((7, 40))
    y0, f0 = y.copy(), finv.copy()
    finish_pixels(y, finv, np.zeros(40, bool), 8.0)
    np.testing.assert_array_equal(y, y0)
    np.testing.assert_array_equal(finv, f0)


def test_finish_pixels_rejects_what_it_cannot_fill():
    """A window that reaches no frame ahead never fills a gap at the start
    (MATLAB's loop would not end), and without any sample there is no
    variance multiplier to give: both are errors. No pixels: nothing to
    do."""
    rng = np.random.default_rng(0)
    y = 5 + rng.random((2, 4, 30))
    finv = 1 + rng.random((4, 30))
    discard = np.zeros(30, dtype=bool)
    discard[0] = True
    with pytest.raises(ValueError, match="cannot fill"):
        finish_pixels(y.copy(), finv.copy(), discard, 2.0)
    with pytest.raises(ValueError, match="cannot fill"):
        finish_pixels(y.copy(), finv.copy(), discard, 0.5)
    finish_pixels(y, finv, discard, 3.0)  # one frame ahead is enough
    assert np.isfinite(y).all()
    gap = np.full((1, 2, 30), np.nan)
    with pytest.raises(ValueError, match="none of the selected pixels"):
        finish_pixels(gap, np.full((2, 30), np.nan), ~discard, 8.0)
    empty = np.zeros((2, 0, 30))
    finish_pixels(empty, np.zeros((0, 30)), discard, 2.0)
    assert empty.shape == (2, 0, 30)


# ------------------------------------------------- frames, by definition


def raw_reference(layout, samples, times, dt, channels, shift=None):
    """Num and Den images by the definition (slow loops).

    getImages' kernel without the lines exactly ``max(3 dt, P)`` from the
    frame, invalid samples left out; ``shift(line) -> (dr, dc)`` moves a
    line's samples. Returns ``(channels, frames, rows, cols)`` arrays.
    """
    p, total = layout.lines_per_cycle, layout.total_lines
    reach = max(3 * dt, p)
    shape = (len(channels), len(times), layout.n_rows, layout.n_cols)
    num, den = np.zeros(shape), np.zeros(shape)
    pr = (layout.slot_pixel - 1) // layout.n_cols
    pc = (layout.slot_pixel - 1) % layout.n_cols
    for f, t in enumerate(times):
        for ln in range(max(1, t - reach + 1), min(total, t + reach - 1) + 1):
            wt = np.exp(-abs(t - ln) / dt)
            sel = np.flatnonzero(layout.slot_line == (ln - 1) % p + 1)
            dr, dc = shift(ln) if shift else (0, 0)
            r, c = pr[sel] + dr, pc[sel] + dc
            ok = (
                (r >= 0) & (r < layout.n_rows) & (c >= 0) & (c < layout.n_cols)
            )
            for k, ch in enumerate(channels):
                x = samples[ch, sel, (ln - 1) // p].astype(float)
                use = ok & (x != -32768)
                np.add.at(num[k, f], (r[use], c[use]), wt * x[use])
                np.add.at(den[k, f], (r[use], c[use]), wt)
    return num, den


def frames_reference(num, den, trim_rows, trim_cols, r_d, c_d, m_r, m_c):
    """slap2FramesAt's interpolation by the definition.

    ``num`` and ``den`` are whole-DMD sums. Returns ``(channels, points,
    frames)`` values and ``(points, frames)`` variance multipliers.
    """
    rows = slice(trim_rows[0] - 1, trim_rows[-1])
    cols = slice(trim_cols[0] - 1, trim_cols[-1])
    with np.errstate(invalid="ignore", divide="ignore"):
        v = (num / den)[:, :, rows, cols]
    w = den[0][:, rows, cols]
    n1, n2 = w.shape[1:]
    n_ch, n_f = v.shape[:2]
    y = np.zeros((n_ch, r_d.size, n_f))
    fq = np.zeros((r_d.size, n_f))
    for f in range(n_f):
        rr, cc = r_d + m_r[f], c_d + m_c[f]
        rf, cf = np.floor(rr), np.floor(cc)
        fr, fc = rr - rf, cc - cf
        acc_n, acc_d = np.zeros((n_ch, r_d.size)), np.zeros(r_d.size)
        for rk, ck, wt in (
            (rf, cf, (1 - fc) * (1 - fr)),
            (rf, np.ceil(cc), fc * (1 - fr)),
            (np.ceil(rr), cf, (1 - fc) * fr),
            (np.ceil(rr), np.ceil(cc), fc * fr),
        ):
            ok = (rk >= 1) & (rk <= n1) & (ck >= 1) & (ck <= n2)
            ri = np.where(ok, rk, 1).astype(int) - 1
            ci = np.where(ok, ck, 1).astype(int) - 1
            fw = np.where(ok, w[f, ri, ci], np.nan)
            acc_d += wt * fw
            acc_n += wt * fw * np.where(ok, v[:, f, ri, ci], np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            y[:, :, f] = acc_n / acc_d
            inv = 1 / acc_d
        inv[np.isinf(inv)] = np.nan
        fq[:, f] = inv
    return y, fq


def collect(parts):
    """All parts of :func:`frames_at` as NumPy arrays."""
    ys, fqs = [], []
    for _, y, fq in parts:
        ys.append(y.numpy())
        fqs.append(fq.numpy())
    return np.concatenate(ys, 2), np.concatenate(fqs, 1)


def raster_window(layout):
    """One-based DMD rows and columns of the raster's bounding box."""
    rows = (layout.slot_pixel - 1) // layout.n_cols + 1
    cols = (layout.slot_pixel - 1) % layout.n_cols + 1
    return (
        np.arange(rows.min(), rows.max() + 1),
        np.arange(cols.min(), cols.max() + 1),
    )


def test_exclusive_boundary_and_pixel_sums(recording):
    """The exclusive kernel leaves out the lines exactly max(3 dt, P) from
    a frame; PixelFormer's sums are FrameFormer's, pixel by pixel."""
    layout, samples, _ = recording
    times, dt = np.array([30, 500, 1777, 3480]), 8
    num, den = raw_reference(layout, samples, times, dt, [1, 0])
    former = FrameFormer(layout, samples, 0, exclusive=True)
    s_num, s_den = former.sums(times, dt, [1, 0])
    r, c = former.rows, former.cols
    np.testing.assert_allclose(s_num.numpy(), num[:, :, r, c], rtol=1e-12)
    np.testing.assert_allclose(s_den.numpy(), den[:, :, r, c], rtol=1e-12)
    inclusive = FrameFormer(layout, samples, 0).sums(times, dt, [1, 0])[1]
    assert (inclusive.numpy() > s_den.numpy()).any()
    for band in (2, 16):
        pixel = PixelFormer(layout, samples, 0, exclusive=True, band=band)
        ((shift, frames, p_num, p_den),) = pixel.pixel_sums(times, dt, [1, 0])
        assert shift == (0, 0) and list(frames) == [0, 1, 2, 3]
        np.testing.assert_allclose(
            p_num.numpy(), s_num.numpy().transpose(0, 2, 1), rtol=1e-12
        )
        np.testing.assert_allclose(
            p_den.numpy(), s_den.numpy().transpose(0, 2, 1), rtol=1e-12
        )
    assert (samples == -32768).any()  # some sums leave samples out


def test_pixel_sums_with_shifts_match_shifted_sums(online_recording):
    """Grouped by displacement, as FrameFormer.shifted_sums."""
    layout, samples, _, _ = online_recording
    offsets = line_offsets(layout, (1, layout.total_lines))
    comp = OnlineCompensation(offsets, offsets.median_xy, MAPPING)
    times, dt = np.arange(900, 1500, 97), 8
    args = (times, dt, comp.displacement)
    want = list(
        FrameFormer(layout, samples, 0, exclusive=True).shifted_sums(*args)
    )
    got = list(
        PixelFormer(layout, samples, 0, exclusive=True).pixel_sums(
            times, dt, None, comp.displacement
        )
    )
    assert len(got) == len(want) > 1
    for (s0, f0, n0, d0), (s1, f1, n1, d1) in zip(want, got):
        assert s0 == s1
        np.testing.assert_array_equal(f0, f1)
        np.testing.assert_allclose(
            n1.numpy(), n0.numpy().transpose(0, 2, 1), rtol=1e-12
        )
        np.testing.assert_allclose(
            d1.numpy(), d0.numpy().transpose(0, 2, 1), rtol=1e-12
        )


def view_points(layout, step=1):
    """Whole-pixel points of a registered frame around the raster."""
    trim_rows, trim_cols = raster_window(layout)
    r, c = np.meshgrid(
        np.arange(1 - MARGIN, trim_rows.size + MARGIN + 1, step),
        np.arange(1 - MARGIN, trim_cols.size + MARGIN + 1, step),
        indexing="ij",
    )
    return (
        trim_rows,
        trim_cols,
        r.ravel().astype(float),
        c.ravel().astype(float),
    )


def test_frames_at_matches_definition(recording):
    """Frames at whole-pixel and at fractional points, with motion that
    crosses whole pixels and takes points off the image."""
    layout, samples, _ = recording
    trim_rows, trim_cols, r_d, c_d = view_points(layout, 2)
    times, dt = np.arange(200, 3300, 61), 8
    rng = np.random.default_rng(1)
    m_r = 1.5 * np.sin(np.arange(times.size) / 4) + 0.2
    m_c = rng.uniform(-2.2, 2.2, times.size)
    m_r[3], m_c[3] = 1.0, -2.0  # whole-pixel motion: ceil equals floor
    m_r[5] = 1 - 1e-12  # within rounding of a whole pixel
    num, den = raw_reference(layout, samples, times, dt, [0, 1])
    args = (layout, [0, 1], times, dt, trim_rows, trim_cols)
    for dr, dc in ((0.0, 0.0), (0.37, -0.21)):
        rq, cq = r_d + dr, c_d + dc
        want = frames_reference(
            num, den, trim_rows, trim_cols, rq, cq, m_r, m_c
        )
        for whole in (True, False):
            got = collect(
                frames_at(*args, rq, cq, m_r, m_c, chunk=7, whole_pixels=whole)
            )
            for g, w in zip(got, want):
                np.testing.assert_array_equal(np.isnan(g), np.isnan(w))
                np.testing.assert_allclose(g, w, rtol=1e-10, atol=1e-12)
    assert np.isnan(want[0]).any() and not np.isnan(want[0]).all()


def test_frames_at_points_off_the_raster(recording):
    """Points whose neighbours are all outside the image need no sample:
    every value is missing."""
    layout, _, _ = recording
    trim_rows, trim_cols = raster_window(layout)
    r_d = np.array([-40.0, -41.0, trim_rows.size + 30.0])
    c_d = np.array([5.0, -7.0, 3.0])
    times = np.arange(300, 900, 50)
    motion = np.linspace(-1.5, 1.5, times.size)
    y, fq = collect(
        frames_at(
            layout,
            [0],
            times,
            8,
            trim_rows,
            trim_cols,
            r_d,
            c_d,
            motion,
            motion,
        )
    )
    assert y.shape == (1, 3, times.size) and fq.shape == (3, times.size)
    assert np.isnan(y).all() and np.isnan(fq).all()
    none = np.zeros(0)
    for whole in (True, False):  # no points at all
        y, fq = collect(
            frames_at(
                layout,
                [0, 1],
                times,
                8,
                trim_rows,
                trim_cols,
                none,
                none,
                motion,
                motion,
                whole_pixels=whole,
            )
        )
        assert y.shape == (2, 0, times.size) and fq.shape == (0, times.size)


def test_frames_at_compensated_matches_definition(online_recording):
    """With compensation, each line's samples move by its online offset
    relative to the block's reference, also past the trimmed raster."""
    layout, samples, _, _ = online_recording
    offsets = line_offsets(layout, (1, layout.total_lines))
    comp = OnlineCompensation(offsets, offsets.median_xy, MAPPING)
    trim_rows, trim_cols, r_d, c_d = view_points(layout, 2)
    times, dt = np.arange(3000, 4500, 53), 8
    m_r = 0.8 * np.cos(np.arange(times.size) / 3)
    m_c = 0.6 * np.sin(np.arange(times.size) / 5) - 0.3
    comp_b, d_b = block_compensation(comp, times)
    assert np.ptp(offsets.x[times - 1]) > 0  # offsets change in the block
    pad = compensation_pad(comp_b, layout, times, dt)
    assert pad >= 1 and compensation_pad(None, layout, times, dt) == 0

    def shift(line):
        """A line's displacement around the block's reference."""
        return (
            int(offsets.y_eff[line - 1] - comp_b.ref_offset[1]),
            int(offsets.x[line - 1] - comp_b.ref_offset[0]),
        )

    num, den = raw_reference(layout, samples, times, dt, [0, 1], shift)
    rows = np.arange(trim_rows[0] - pad, trim_rows[-1] + pad + 1)
    cols = np.arange(trim_cols[0] - pad, trim_cols[-1] + pad + 1)
    want = frames_reference(
        num, den, rows, cols, r_d + pad, c_d + pad, m_r - d_b[0], m_c - d_b[1]
    )
    got = collect(
        frames_at(
            layout,
            [0, 1],
            times,
            dt,
            trim_rows,
            trim_cols,
            r_d,
            c_d,
            m_r - d_b[0],
            m_c - d_b[1],
            comp_b,
            chunk=9,
        )
    )
    for g, w in zip(got, want):
        np.testing.assert_array_equal(np.isnan(g), np.isnan(w))
        np.testing.assert_allclose(g, w, rtol=1e-10, atol=1e-12)
    # the block's reference only moves the frame: around the trial's
    # reference, the same points give the same samples
    other = collect(
        frames_at(
            layout,
            [0, 1],
            times,
            dt,
            trim_rows,
            trim_cols,
            r_d,
            c_d,
            m_r,
            m_c,
            comp,
            chunk=64,
        )
    )
    inner = ~np.isnan(got[0]) & ~np.isnan(other[0])
    assert inner.mean() > 0.5
    np.testing.assert_allclose(got[0][inner], other[0][inner], rtol=1e-9)


def test_compensation_helpers(online_recording):
    """onlineCompensation: the motion to interpolate is the saved motion
    plus each registered frame's offset; blocks take their median."""
    layout, _, _, (x, y, y_dmd) = online_recording
    ds = np.arange(100.0, 9000.0, 20.0)
    ref = np.array([2.0, 0.0])
    fx = x[ds.astype(int) - 1].astype(float)
    fy = y_dmd[ds.astype(int) - 1].astype(float)
    alignment = TrialAlignment(
        ds_frames=ds,
        motion_r=np.linspace(0, 1, ds.size),
        motion_c=np.linspace(1, 0, ds.size),
        align_hz=80.0,
        view_r=np.arange(1.0, 5.0),
        view_c=np.arange(1.0, 5.0),
        trim_rows=np.arange(9, 13),
        trim_cols=np.arange(11, 15),
        compensation={
            "refOffset": ref,
            "frameRefX": fx,
            "frameRefY": fy,
            "mapping": MAPPING,
            "yLead": np.array([2.0, 6.0]),
        },
    )
    comp, mot_r, mot_c = online_compensation(
        alignment, layout, (100, 9000), 20
    )
    np.testing.assert_allclose(mot_r, alignment.motion_r + fy - ref[1])
    np.testing.assert_allclose(mot_c, alignment.motion_c + fx - ref[0])
    np.testing.assert_array_equal(comp.offsets.y_eff[99:9000], y_dmd[99:9000])
    dr, dc = comp.displacement(np.array([[150, 4000], [8000, 20_000_000]]))
    assert dr.shape == (2, 2) and dr.dtype == np.int64
    assert dr[0, 1] == y_dmd[3999] - 0 and dc[0, 1] == x[3999] - 2
    lines = np.arange(4000, 4400)
    comp_b, shift = block_compensation(comp, lines)
    med = [np.median(x[lines - 1]), np.median(y_dmd[lines - 1])]
    np.testing.assert_array_equal(comp_b.ref_offset, np.round(med))
    assert shift == (comp_b.ref_offset[1] - 0.0, comp_b.ref_offset[0] - 2.0)
    assert block_compensation(None, lines) == (None, (0.0, 0.0))
    # without the record: the saved motion, and no compensation
    alignment.compensation = None
    none, mot_r, _ = online_compensation(alignment, layout, (100, 9000), 20)
    assert none is None
    np.testing.assert_array_equal(mot_r, alignment.motion_r)
    bad = OnlineCompensation(comp.offsets, ref, 0.5 * MAPPING)
    with pytest.raises(ValueError, match="integer"):
        bad.displacement(lines)


# ------------------------------------------------------- the whole loader


def loader_inputs(layout, truth, shift=(0.0, 0.0), ds_step=20):
    """Alignment, maps and masks for loading a synthetic recording.

    The registered frame is the raster's bounding box with MARGIN pixels
    around it; registration's motion is the true motion at its frames.
    Returns also the scene where the trial's ``shift`` reads each pixel of
    the consensus canvas (``truth``); the mean image is its square / 20
    (and half that in channel 2), so that it has dim and bright parts as a
    recording's, and differs from the data it normalizes.
    """
    trim_rows, trim_cols = raster_window(layout)
    shape = (trim_rows.size + 2 * MARGIN, trim_cols.size + 2 * MARGIN)
    ds = np.arange(float(ds_step), layout.total_lines - ds_step, ds_step)
    alignment = TrialAlignment(
        ds_frames=ds,
        motion_r=truth[ds.astype(int) - 1, 0],
        motion_c=truth[ds.astype(int) - 1, 1],
        align_hz=80.0,
        view_r=np.arange(1.0 - MARGIN, trim_rows.size + MARGIN + 1),
        view_c=np.arange(1.0 - MARGIN, trim_cols.size + MARGIN + 1),
        trim_rows=trim_rows,
        trim_cols=trim_cols,
    )
    # consensus canvas: two rows and a column larger than this trial's frame
    canvas = (shape[0] + 2, shape[1] + 1)
    sel = np.zeros(canvas, dtype=bool)
    sel[8:15, 9:21] = True
    sel[18:21, 25:31] = True
    sel[1, 2] = True  # in the margin: never sampled
    sel[shape[0] + 1, 5] = True  # outside this trial's frame
    rows = trim_rows[0] - 1 - MARGIN + np.arange(canvas[0])
    cols = trim_cols[0] - 1 - MARGIN + np.arange(canvas[1])
    rr, cc = np.meshgrid(rows + shift[0], cols + shift[1], indexing="ij")
    inside = (
        (rr >= trim_rows[0] - 1)
        & (rr <= trim_rows[-1] - 1)
        & (cc >= trim_cols[0] - 1)
        & (cc <= trim_cols[-1] - 1)
    )
    image = np.full(canvas, np.nan)
    image[inside] = map_coordinates(
        scene(layout), [rr[inside], cc[inside]], order=1
    )
    mean = np.stack([image**2 / 20, image**2 / 40], 2)
    roi_a = np.zeros(canvas, dtype=bool)
    roi_a[10:14, 12:17] = True
    roi_b = np.zeros(canvas, dtype=bool)
    roi_b[0:6, 0:8] = True  # partly in the margin
    roi_c = np.zeros(canvas, dtype=bool)
    roi_c[shape[0] :, :] = True  # outside this trial's frame
    discard = np.zeros(ds.size, dtype=bool)
    discard[40:44] = True
    return alignment, sel, mean, [roi_a, roi_b, roi_c], discard, image


def test_load_trial_follows_the_scene(recording, monkeypatch):
    """Motion-corrected samples of a static scene are steady and equal the
    scene at each pixel's position; traces, discards and filled samples
    are as loadTrial makes them."""
    layout, _, truth = recording
    monkeypatch.setattr(trial_data, "raster_layout", lambda path: layout)
    shift = (0.3, -0.4)
    alignment, sel, mean, rois, discard, image = loader_inputs(
        layout, truth, shift
    )
    params = TrialLoadParams(block_frames=37, baseline_window_s=0.2)
    first, last = 10.0, layout.total_lines - 30.5
    td = load_trial(
        layout.files[0],
        first,
        last,
        alignment,
        sel,
        RigidShift(*shift),
        discard,
        mean,
        rois,
        params,
    )
    n_f = td.frame_lines.size
    assert td.y.shape == (2, sel.sum(), n_f) and td.y.dtype == np.float32
    assert td.finv.shape == (sel.sum(), n_f) and td.photon_scale == 25.0
    np.testing.assert_array_equal(
        td.frame_lines, frame_lines(first, last, 8.0)
    )
    np.testing.assert_array_equal(td.discard, upsample_discard(discard, n_f))
    assert 0 < td.discard.sum() < 20
    # pixels in MATLAB's column-major order of the map
    np.testing.assert_array_equal(td.pixel_rc, np.argwhere(sel.T)[:, ::-1])
    np.testing.assert_allclose(td.trial_rc, td.pixel_rc + np.array(shift))
    want_r = upsample_motion(
        alignment.ds_frames, alignment.motion_r, td.frame_lines
    )
    np.testing.assert_allclose(td.motion_r, want_r + shift[0])
    assert np.isfinite(td.y).all() and np.isfinite(td.finv).all()

    never = (td.pixel_rc[:, 0] == 1) | (td.pixel_rc[:, 0] == sel.shape[0] - 1)
    assert never.sum() == 2
    assert np.all(td.y[:, never] == 0)  # no sample: filled from 0
    filled = td.finv > 100 * np.median(td.finv)
    assert filled[never].all() and filled[:, td.discard].all()
    assert not filled[~never][:, ~td.discard].any()
    # 1000 times the mean over all samples, those of discarded frames too
    assert np.ptp(td.finv[filled]) == 0
    np.testing.assert_allclose(
        td.finv[filled][0], 1000 * td.finv[~filled].mean(), rtol=5e-3
    )

    keep = ~td.discard
    got = td.y[:, ~never][:, :, keep].astype(np.float64) * 25.0
    want = image[tuple(td.pixel_rc[~never].T)]
    mean_0, mean_1 = got[0].mean(1), got[1].mean(1)
    assert np.corrcoef(mean_0, want)[0, 1] > 0.995
    assert np.median(np.abs(mean_0 / want - 1)) < 0.03
    assert np.median(np.abs(mean_1 / (0.5 * want) - 1)) < 0.05
    # steady: temporal SD near Poisson's, far below the spatial SD
    assert np.median(got[0].std(1)) < 0.35 * want.std()

    # traces: NaN at discards. Global: the labeled pixels' summed data
    # over their summed mean image, times the whole mean image's sum. ROI:
    # the same over the ROI's pixels, which is their summed data
    shape = alignment.view_shape
    view, mean_v = image[: shape[0], : shape[1]], mean[: shape[0], : shape[1]]
    lab = labeled_pixels(mean_v[..., 0])
    assert lab.sum() > 20
    assert np.isnan(td.global_f[:, td.discard]).all()
    assert np.isfinite(td.global_f[:, keep]).all()
    assert td.roi_f.shape == td.roi_fsvd.shape == (3, 2, n_f)
    assert np.isnan(td.roi_f[2]).all() and np.isnan(td.roi_fsvd[2]).all()
    assert np.isfinite(td.roi_f[1][:, keep]).all()
    for k, gain in enumerate((1.0, 0.5)):
        m = mean_v[..., k]
        want_g = gain * view[lab].sum() / m[lab].sum() * np.nansum(m)
        np.testing.assert_allclose(
            td.global_f[k, keep].mean(), want_g, rtol=0.05
        )
        want_r = (
            gain
            * view[: shape[0], : shape[1]][
                rois[0][: shape[0], : shape[1]]
            ].sum()
        )
        np.testing.assert_allclose(
            td.roi_f[0, k, keep].mean(), want_r, rtol=0.03
        )
        np.testing.assert_allclose(
            td.roi_fsvd[0, k, keep].mean(), want_r / rois[0].sum(), rtol=0.03
        )

    # the blocks do not show: one block of all frames gives the same, and
    # so does double precision
    one = load_trial(
        layout.files[0],
        first,
        last,
        alignment,
        sel,
        RigidShift(*shift),
        discard,
        mean,
        rois,
        TrialLoadParams(block_frames=100000, baseline_window_s=0.2),
        dtype=np.float64,
    )
    np.testing.assert_allclose(td.y, one.y, rtol=2e-6, atol=1e-6)
    np.testing.assert_allclose(td.finv, one.finv, rtol=2e-6)
    np.testing.assert_allclose(
        td.global_f, one.global_f, rtol=1e-9, equal_nan=True
    )
    np.testing.assert_allclose(
        td.roi_fsvd, one.roi_fsvd, rtol=1e-7, equal_nan=True
    )


def test_load_trial_with_a_general_warp(recording, monkeypatch):
    """A callable warp is evaluated at every pixel: the same shift given as
    a function gives the same data, and an affine map samples the scene at
    the mapped positions."""
    layout, _, truth = recording
    monkeypatch.setattr(trial_data, "raster_layout", lambda path: layout)
    alignment, sel, mean, rois, discard, image = loader_inputs(layout, truth)
    trim_rows, trim_cols = raster_window(layout)
    origin = (trim_rows[0] - 1 - MARGIN, trim_cols[0] - 1 - MARGIN)
    args = (layout.files[0], 10.0, 3000.0, alignment, sel)
    rest = (discard, mean, rois, TrialLoadParams(baseline_window_s=0.2))
    shift = (0.3, -0.4)
    rigid = load_trial(*args, RigidShift(*shift), *rest, dtype=np.float64)

    def as_function(r, c):
        """The shift as a general coordinate map."""
        return r + shift[0], c + shift[1]

    general = load_trial(*args, as_function, *rest, dtype=np.float64)
    np.testing.assert_allclose(general.y, rigid.y, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(general.finv, rigid.finv, rtol=1e-8)
    np.testing.assert_allclose(general.motion_r, rigid.motion_r, atol=1e-12)
    np.testing.assert_allclose(general.trial_rc, rigid.trial_rc)

    a = np.array([[1.02, 0.015], [-0.01, 0.98]])
    x0 = np.array([12.0, 17.0])

    def affine(r, c):
        """A stretch and shear about the field's centre, plus a shift."""
        p = a @ (np.stack([r, c]) - x0[:, None]) + x0[:, None]
        return p[0] + 0.25, p[1] - 0.15

    warped = load_trial(*args, affine, *rest, dtype=np.float64)
    r_t, c_t = affine(*warped.pixel_rc.T.astype(float))
    np.testing.assert_allclose(warped.trial_rc, np.stack([r_t, c_t], 1))
    img = scene(layout)
    want = map_coordinates(img, [origin[0] + r_t, origin[1] + c_t], order=1)
    used = np.median(warped.finv, 1) < 100 * np.median(warped.finv)
    got = warped.y[0][:, ~warped.discard].mean(1) * 25.0
    assert used.sum() > 80
    assert np.corrcoef(got[used], want[used])[0, 1] > 0.995
    ignored = image[tuple(warped.pixel_rc.T)]  # as if the map did nothing
    err = np.abs(got[used] - want[used]).mean()
    assert err < 0.5 * np.abs(got[used] - ignored[used]).mean()


def test_load_trial_reads_selected_pixels_beyond_the_frame(
    recording, monkeypatch
):
    """The consensus frame may be larger than a trial's registered frame. A
    selected pixel beyond the frame is read where the warp puts it: with a
    shift of nine rows up it stays on the raster under the motion, and has
    data."""
    layout, _, truth = recording
    monkeypatch.setattr(trial_data, "raster_layout", lambda path: layout)
    shift = (-9.3, 0.2)
    alignment, sel, mean, rois, discard, image = loader_inputs(
        layout, truth, shift
    )
    beyond = (alignment.view_shape[0] + 1, 5)
    assert sel[beyond] and np.isfinite(image[beyond])
    args = (layout.files[0], 10.0, 3000.0, alignment, sel)
    rest = (discard, mean, rois[:1], TrialLoadParams(baseline_window_s=0.2))
    td = load_trial(*args, RigidShift(*shift), *rest, dtype=np.float64)
    k = np.flatnonzero((td.pixel_rc == beyond).all(1))[0]
    np.testing.assert_allclose(td.trial_rc[k], np.add(beyond, shift))
    keep = ~td.discard
    assert (td.finv[k, keep] < 100 * np.median(td.finv[k])).all()
    np.testing.assert_allclose(
        td.y[0, k, keep].mean() * 25.0, image[beyond], rtol=0.1
    )

    def as_function(r, c):
        """The shift as a general coordinate map."""
        return r + shift[0], c + shift[1]

    general = load_trial(*args, as_function, *rest, dtype=np.float64)
    np.testing.assert_allclose(general.y, td.y, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(general.finv, td.finv, rtol=1e-8)


def test_recording_layout_is_read_once_per_file(
    recording, monkeypatch, tmp_path
):
    """Trials of one recording file (a continuous recording's line
    ranges) read its layout once, also when loaded at the same time; a
    changed file, another file or a missing one is read again; the loads
    are the same as with a layout read every time."""
    import os
    import threading
    from concurrent.futures import ThreadPoolExecutor

    layout, _, truth = recording
    calls = []
    lock = threading.Lock()

    def counted(path):
        """The fake layout, counting reads."""
        with lock:
            calls.append(str(path))
        return layout

    monkeypatch.setattr(trial_data, "raster_layout", counted)
    alignment, sel, mean, rois, discard, _ = loader_inputs(layout, truth)
    rest = (discard, mean, rois, TrialLoadParams(baseline_window_s=0.2))
    path = layout.files[0]

    def load(first, last):
        """One line range of the recording."""
        return load_trial(path, first, last, alignment, sel, RigidShift(),
                          *rest, dtype=np.float64)  # fmt: skip

    a = load(10.0, 2000.0)
    b = load(1500.0, 3000.0)
    assert calls == [str(path)]
    assert a.timings["layout"] <= a.timings["setup"]
    # concurrent loads of a new key read it once
    other = tmp_path / "copy_DMD1-TRIAL000001.dat"
    other.write_bytes(Path(path).read_bytes())
    calls.clear()
    with ThreadPoolExecutor(3) as pool:
        got = list(pool.map(lambda k: trial_data.recording_layout(other),
                            range(3)))  # fmt: skip
    assert calls == [str(other)] and all(g is layout for g in got)
    # a changed file is read again; a missing one every time
    stat = other.stat()
    os.utime(other, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    trial_data.recording_layout(other)
    assert calls == [str(other)] * 2
    calls.clear()
    for _ in range(2):
        trial_data.recording_layout(tmp_path / "missing.dat")
    assert len(calls) == 2
    # the same loads as reading the layout every time
    monkeypatch.setattr(trial_data, "recording_layout", lambda p: layout)
    for got, first, last in ((a, 10.0, 2000.0), (b, 1500.0, 3000.0)):
        ref = load(first, last)
        np.testing.assert_array_equal(got.y, ref.y)
        np.testing.assert_array_equal(got.finv, ref.finv)


def test_recording_layout_follows_segments_and_metadata(
    recording, monkeypatch, tmp_path
):
    """A CYCLE-split recording's layout is read again when a segment is
    added or removed or its metadata changes, and the path as given is
    part of the key (its name decides how the reader opens it)."""
    import os

    layout = recording[0]
    calls = []
    monkeypatch.setattr(
        trial_data, "raster_layout", lambda path: calls.append(path) or layout
    )
    folder = tmp_path / "split"
    folder.mkdir()
    first = folder / "rec_DMD1-TRIAL000001-CYCLE-000000.dat"
    first.write_bytes(b"x")
    meta = folder / "rec_DMD1.meta"
    meta.write_bytes(b"m")
    (folder / "rec_DMD2-TRIAL000001-CYCLE-000000.dat").write_bytes(b"y")
    read = trial_data.recording_layout
    read(first), read(first)
    assert len(calls) == 1
    names = [n for n, _, _ in trial_data._layout_inputs(first)]
    assert names == [first.name, meta.name]  # not the other DMD's
    second = folder / "rec_DMD1-TRIAL000001-CYCLE-000100.dat"
    second.write_bytes(b"z")  # a segment added
    read(first), read(first)
    assert len(calls) == 2
    st = meta.stat()  # the metadata changed
    os.utime(meta, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    read(first)
    assert len(calls) == 3
    second.unlink()  # a segment removed
    read(first)
    assert len(calls) == 4
    # the same absolute path is the same key; a link to the file is not
    (folder / "sub").mkdir()
    read(str(folder / "sub" / ".." / first.name))
    assert len(calls) == 4
    link = folder / "link_DMD1-TRIAL000001.dat"
    try:
        os.symlink(first, link)
    except (OSError, NotImplementedError):
        return  # no symbolic links here
    read(link)
    assert len(calls) == 5


def test_load_trial_without_selected_pixels(recording, monkeypatch):
    """No selected pixel, no ROI and no labeled pixel: empty data, and the
    traces are missing. A selection that is never sampled is an error."""
    layout, _, truth = recording
    monkeypatch.setattr(trial_data, "raster_layout", lambda path: layout)
    alignment, sel, mean, _, discard, _ = loader_inputs(layout, truth)
    args = (layout.files[0], 10.0, 2000.0, alignment)
    dark = np.full(mean.shape, np.nan)
    td = load_trial(*args, np.zeros_like(sel), RigidShift(), discard, dark)
    n_f = td.frame_lines.size
    assert td.y.shape == (2, 0, n_f) and td.finv.shape == (0, n_f)
    assert td.pixel_rc.shape == td.trial_rc.shape == (0, 2)
    assert np.isnan(td.global_f).all() and td.roi_f.shape == (0, 2, n_f)
    margin = np.zeros_like(sel)
    margin[0, :3] = True  # never on the raster
    with pytest.raises(ValueError, match="none of the selected pixels"):
        load_trial(*args, margin, RigidShift(), discard, mean)


def test_load_trial_rejects_bad_inputs(recording, monkeypatch):
    """The activity channel and the mean image must fit the recording."""
    layout, _, truth = recording
    monkeypatch.setattr(trial_data, "raster_layout", lambda path: layout)
    alignment, sel, mean, rois, discard, _ = loader_inputs(layout, truth)
    args = (layout.files[0], 10.0, 3000.0, alignment, sel, RigidShift())
    with pytest.raises(ValueError, match="activity_channel"):
        load_trial(
            *args, discard, mean, rois, TrialLoadParams(activity_channel=3)
        )
    with pytest.raises(ValueError, match="2 channels"):
        load_trial(*args, discard, mean[..., :1], rois)
    for bad in (2.9, 1.0, True, "1"):  # not cut to a channel below
        with pytest.raises(ValueError, match="must be an integer"):
            TrialLoadParams(activity_channel=bad).ordered_channels(2)
    assert TrialLoadParams(activity_channel=np.int64(2)).ordered_channels(
        2
    ) == [1, 0]


def test_load_trial_second_activity_channel(recording, monkeypatch):
    """With activityChannel 2, channel 0 of the data is the file's second
    channel; the global trace stays in the file's order."""
    layout, _, truth = recording
    monkeypatch.setattr(trial_data, "raster_layout", lambda path: layout)
    alignment, sel, mean, rois, discard, _ = loader_inputs(layout, truth)
    args = (layout.files[0], 10.0, 2000.0, alignment, sel, RigidShift())
    one = load_trial(*args, discard, mean, rois[:1], dtype=np.float64)
    two = load_trial(
        *args,
        discard,
        mean,
        rois[:1],
        TrialLoadParams(activity_channel=2),
        dtype=np.float64,
    )
    keep = ~one.discard
    np.testing.assert_allclose(
        two.global_f[:, keep], one.global_f[:, keep], rtol=0.02
    )
    # (the channels' invalid samples differ, and with them the shares of
    # the neighbours: equal on average, not sample by sample)
    np.testing.assert_allclose(
        two.roi_f[0, ::-1][:, keep].mean(1),
        one.roi_f[0][:, keep].mean(1),
        rtol=0.01,
    )
    used = np.median(one.finv, 1) < 100 * np.median(one.finv)
    ratio = two.y[0][used][:, keep].mean() / one.y[0][used][:, keep].mean()
    assert 0.45 < ratio < 0.55  # channel 2 sees half the scene


def test_load_trial_after_compensated_registration(
    quiet_online_recording, tmp_path, monkeypatch
):
    """Registered with compensateOnline, the loader undoes the online
    correction too: samples of the static, noise-free scene vary less than
    after an uncompensated registration, none is filled in, and the block
    size does not show."""
    layout = quiet_online_recording
    monkeypatch.setattr(multiroi, "raster_layout", lambda path: layout)
    monkeypatch.setattr(trial_data, "raster_layout", lambda path: layout)
    spread, loaded = {}, {}
    for comp in (False, True):
        params = multiroi.MultiRoiRegistrationParams(
            maxshift=6, compensate_online=comp
        )
        adata = tmp_path / f"{comp}_ALIGNMENTDATA.h5"
        multiroi.register_trial(
            layout.files[0], 1, 6000, tmp_path / f"{comp}.tif", adata, params
        )
        alignment = read_alignment(adata)
        assert (alignment.compensation is not None) == comp
        assert alignment.view_shape == (20 + 12, 30 + 12)
        sel = np.zeros(alignment.view_shape, dtype=bool)
        sel[12:20, 14:28] = True
        mean = np.ones(alignment.view_shape + (2,))
        discard = np.zeros(alignment.ds_frames.size, dtype=bool)
        args = (layout.files[0], 1, 6000, alignment, sel, RigidShift())
        td = load_trial(
            *args, discard, mean, (), TrialLoadParams(block_frames=60)
        )
        y = td.y[0].astype(np.float64)
        spread[comp] = np.median(y.std(1)) / y.mean(1).std()
        loaded[comp] = (td, args, discard, mean)
        assert np.isfinite(td.y).all()
        assert (td.finv < 100 * np.median(td.finv)).all()  # nothing filled
    # temporal SD of a pixel over the spatial SD of the pixels' means
    # (measured: 0.044 compensated, 0.074 uncompensated; part of it is
    # interpolation error common to both)
    assert spread[True] < 0.8 * spread[False]
    assert spread[True] < 0.06
    td, args, discard, mean = loaded[True]
    other = load_trial(
        *args, discard, mean, (), TrialLoadParams(block_frames=100000)
    )
    np.testing.assert_allclose(td.y, other.y, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(td.finv, other.finv, rtol=1e-5)


def test_read_alignment_orientation(tmp_path):
    """viewR and viewC are read in MATLAB's orientation from files written
    column-major (MATLAB, GIAnT-Python) or row-major."""
    import h5py

    view_r = np.repeat(np.arange(-2.0, 5.0)[:, None], 4, 1)  # 7 x 4
    view_c = np.repeat(np.arange(-1.0, 3.0)[None, :], 7, 0)
    for row_major in (False, True):
        path = tmp_path / f"{row_major}_ALIGNMENTDATA.h5"
        with h5py.File(path, "w") as f:
            f["DSframes"] = np.arange(5.0)[None, :]
            f["motionDSr"] = np.zeros((1, 5))
            f["motionDSc"] = np.ones((5, 1))
            f["alignHz"] = np.array([[80.0]])
            f["slap2/viewR"] = view_r if row_major else view_r.T
            f["slap2/viewC"] = view_c if row_major else view_c.T
            f["slap2/trimRows"] = np.arange(3.0, 6.0)[:, None]
            f["slap2/trimCols"] = np.arange(8.0, 10.0)[:, None]
            f["slap2/onlineCompensation/enabled"] = np.zeros((1, 1), np.int8)
            f["row_major"] = np.array([[int(row_major)]], np.uint8)
        a = read_alignment(path)
        np.testing.assert_array_equal(a.view_r, view_r[:, 0])
        np.testing.assert_array_equal(a.view_c, view_c[0])
        assert a.view_shape == (7, 4) and a.compensation is None
        np.testing.assert_array_equal(a.trim_rows, [3, 4, 5])
        assert a.align_hz == 80.0 and a.motion_c.shape == (5,)


def test_cuda_matches_cpu(recording):
    """Frames agree on CUDA and CPU (skipped without a GPU)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    layout, _, _ = recording
    trim_rows, trim_cols, r_d, c_d = view_points(layout, 2)
    times = np.arange(200, 3300, 61)
    m_r = np.sin(np.arange(times.size) / 4)
    out = []
    for dev in ("cpu", "cuda"):
        parts = frames_at(
            layout,
            [0, 1],
            times,
            8,
            trim_rows,
            trim_cols,
            r_d,
            c_d,
            m_r,
            m_r,
            device=dev,
        )
        out.append([(y.cpu().numpy(), f.cpu().numpy()) for _, y, f in parts])
    for (y0, f0), (y1, f1) in zip(*out):
        np.testing.assert_allclose(y0, y1, rtol=1e-9, equal_nan=True)
        np.testing.assert_allclose(f0, f1, rtol=1e-9, equal_nan=True)
