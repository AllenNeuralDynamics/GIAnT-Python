"""Registration kernels against direct ports of the GIAnT-MATLAB code."""

import warnings

import numpy as np
import pytest
import torch

from giant_python.registration.interpolation import interp_frame
from giant_python.registration.matching import (
    RegistrationError,
    correlation_map,
    match,
    match_maps,
    shift_tables,
)
from giant_python.registration.multiroi import matlab_colon
from giant_python.registration.outputs import (
    downsample2,
    matlab_round,
    rec_neg_err,
)
from giant_python.registration.template import TemplateState


def xcorr_reference(frame, fresh, template, s):
    """Loop port of ``xcorr2_nans_weighted`` (correlation map only)."""
    h, w = template.shape
    rows, cols = np.nonzero(~np.isnan(frame))
    out = np.full((2 * s + 1, 2 * s + 1), np.nan)
    for i, dr in enumerate(range(-s, s + 1)):
        for j, dc in enumerate(range(-s, s + 1)):
            r, c = rows + dr, cols + dc
            ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
            t = np.full(rows.size, np.nan)
            t[ok] = template[r[ok], c[ok]]
            sel = ~np.isnan(t)
            f, t, wt = frame[rows, cols][sel], t[sel], fresh[rows, cols][sel]
            sw = wt.sum()
            df = f - (wt * f).sum() / sw
            dt = t - (wt * t).sum() / sw
            out[i, j] = (wt * df * dt).sum() / np.sqrt(
                (wt * dt**2).sum() * (wt * df**2).sum()
            )
    return out


def interp_reference(m1, view_c, view_r, fresh):
    """Gather port of ``interpFrame`` (one-based view coordinates)."""
    h, w = m1.shape

    def corner(rfun, cfun):
        """Frame and freshness at one bilinear corner, NaN outside."""
        rr = rfun(view_r).astype(int)
        cc = cfun(view_c).astype(int)
        bad_r = (rr < 1) | (rr > h)
        bad_c = (cc < 1) | (cc > w)
        rr[bad_r], cc[bad_c] = 1, 1
        a = m1[np.ix_(rr - 1, cc - 1)].astype(np.float64)
        f = fresh[np.ix_(rr - 1, cc - 1)].astype(np.float64)
        for x in (a, f):
            x[bad_r, :] = np.nan
            x[:, bad_c] = np.nan
        return a, f

    fc, fr = np.mod(view_c[0], 1), np.mod(view_r[0], 1)
    coef = [(1 - fc) * (1 - fr), fc * (1 - fr), (1 - fc) * fr, fc * fr]
    funcs = [
        (np.floor, np.floor),
        (np.floor, np.ceil),
        (np.ceil, np.floor),
        (np.ceil, np.ceil),
    ]
    num = den = 0
    for cf, (rf, cfun) in zip(coef, funcs):
        a, f = corner(rf, cfun)
        num = num + cf * f * a
        den = den + cf * f
    with np.errstate(divide="ignore", invalid="ignore"):
        vim = 1 / den
    vim[np.isinf(vim)] = np.nan
    return num / den, vim


def random_frame(rng, h=18, w=23, frac=0.4):
    """Smooth frame with a NaN pattern, its freshness and a template."""
    base = rng.random((h + 8, w + 8))
    img = (base[4:-4, 4:-4] + base[5:-3, 4:-4] + base[4:-4, 5:-3]) * 10
    frame = img + rng.normal(0, 0.5, img.shape)
    fresh = rng.random(img.shape) + 0.2
    frame[rng.random(img.shape) < frac] = np.nan
    fresh[np.isnan(frame)] = 0
    template = np.roll(img, (1, -2), (0, 1))
    template[rng.random(img.shape) < 0.2] = np.nan
    return frame, fresh, template


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_correlation_map_matches_matlab_loop(dtype):
    """Weighted NCC equals the MATLAB loop at every shift."""
    rng = np.random.default_rng(0)
    frame, fresh, template = random_frame(rng)
    s = 4
    rows, cols = np.nonzero(~np.isnan(frame))
    f = torch.from_numpy(frame[rows, cols])[None]
    w = torch.from_numpy(fresh[rows, cols])[None]
    tables = shift_tables(
        torch.from_numpy(template),
        torch.from_numpy(rows),
        torch.from_numpy(cols),
        s,
        dtype,
    )
    got = correlation_map(f, w, tables, s)[0].double().numpy()
    want = xcorr_reference(frame, fresh, template, s)
    tol = 1e-4 if dtype == torch.float32 else 1e-10
    np.testing.assert_allclose(got, want, atol=tol)


def test_match_recovers_shift_with_subpixel_sign():
    """A template shifted by (+1, -2) gives motion (-1, +2), MATLAB's sign."""
    rng = np.random.default_rng(1)
    frame, fresh, template = random_frame(rng, frac=0.0)
    frame = np.roll(template, (-1, 2), (0, 1))
    frame[np.isnan(frame)] = 5.0
    rows, cols = np.nonzero(~np.isnan(frame))
    mot, peak = match(
        torch.from_numpy(frame[rows, cols])[None],
        torch.ones(1, rows.size, dtype=torch.float64),
        torch.from_numpy(template),
        torch.from_numpy(rows),
        torch.from_numpy(cols),
        4,
        torch.float64,
    )
    np.testing.assert_allclose(mot[0].numpy(), [-1, 2], atol=0.05)
    assert peak[0] > 0.9


def test_frame_without_valid_samples_has_nan_motion():
    """An undefined correlation map gives NaN motion, not a search edge.

    The other frame of the batch is unaffected, and ``match`` raises.
    """
    rng = np.random.default_rng(1)
    _, _, template = random_frame(rng, frac=0.0)
    good = np.roll(template, (-1, 2), (0, 1))
    good[np.isnan(good)] = 5.0
    rows, cols = np.nonzero(~np.isnan(good))
    frames = torch.from_numpy(np.stack([good[rows, cols]] * 2))
    weights = torch.ones(2, rows.size, dtype=torch.float64)
    frames[1] = float("nan")  # no valid samples
    weights[1] = 0
    args = (
        torch.from_numpy(template),
        torch.from_numpy(rows),
        torch.from_numpy(cols),
        4,
        torch.float64,
    )
    mot, peak, corr = match_maps(frames, weights, *args)
    assert torch.isnan(corr[1]).all()
    assert torch.isnan(peak[1]) and torch.isnan(mot[1]).all()
    want, want_peak = match(frames[:1], weights[:1], *args)
    np.testing.assert_allclose(mot[0].numpy(), want[0].numpy(), rtol=1e-12)
    np.testing.assert_allclose(peak[0].item(), want_peak[0].item())
    with pytest.raises(RegistrationError):
        match(frames, weights, *args)


@pytest.mark.parametrize(
    "mr,mc", [(0.0, 0.0), (0.3, -0.7), (-1.25, 2.5), (3.0, -0.4), (30, 2)]
)
def test_interp_frame_matches_matlab(mr, mc):
    """Resampling equals interpFrame, including NaN and edge behaviour."""
    rng = np.random.default_rng(2)
    frame, fresh, _ = random_frame(rng)
    frame = frame.astype(np.float32)
    fresh = fresh.astype(np.float32)
    m = 3
    h, w = frame.shape
    view_r = np.arange(1, h + 2 * m + 1) - m + mr
    view_c = np.arange(1, w + 2 * m + 1) - m + mc
    want_im, want_v = interp_reference(frame, view_c, view_r, fresh)
    im, v = interp_frame(
        torch.from_numpy(frame), torch.from_numpy(fresh), mr, mc, m
    )
    np.testing.assert_array_equal(np.isnan(im.numpy()), np.isnan(want_im))
    np.testing.assert_allclose(im.numpy(), want_im, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(v.numpy(), want_v, rtol=1e-5)


def test_matlab_helpers():
    """MATLAB rounding, colon and 2 x 2 downsampling semantics."""
    np.testing.assert_array_equal(
        matlab_round([-2.5, -0.5, 0.5, 1.5, 2.4]), [-3, -1, 1, 2, 2]
    )
    ds = matlab_colon(1, 133.6, 1000)
    assert ds[0] == 1 and ds.size == 8 and ds[-1] <= 1000
    x = np.arange(30, dtype=np.float32).reshape(5, 6)
    x[0, 0] = np.nan
    y = downsample2(x)
    assert y.shape == (2, 3) and np.isnan(y[0, 0])
    assert y[1, 2] == x[2, 4] + x[2, 5] + x[3, 4] + x[3, 5]


def test_rec_neg_err_matches_definition():
    """Template excess over frames, with MATLAB's NaN handling."""
    rng = np.random.default_rng(3)
    movie = rng.random((30, 6, 7)).astype(np.float32) * 20
    movie[5, 2, 3] = np.nan
    movie[:, 0, 0] = np.nan
    err = rec_neg_err(movie, align_hz=80)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        tmpl = np.nanmedian(movie, 0)
    tmpl[np.isnan(movie).mean(0) > 0.2] = np.nan
    gam = np.sqrt(np.maximum(0, tmpl) + 10)
    for k in (0, 5):
        ex = (tmpl - movie[k]) / gam
        num = np.nansum(np.maximum(0, ex) ** 2) / ex.size
        den_v = (tmpl / gam) ** 2 + movie[k] * 0
        den = np.nanmean(den_v)
        np.testing.assert_allclose(err[k], np.sqrt(num / den), rtol=1e-5)


def test_template_state_matches_matlab_rules():
    """sqrt of the mean, NaN below min_samples, and the T0 average."""
    total = torch.tensor([[4.0, 9.0], [0.0, 1.0]], dtype=torch.float64)
    count = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
    t0 = torch.tensor([[1.0, float("nan")], [3.0, float("nan")]])
    state = TemplateState(total, count, t0.double(), min_samples=1)
    np.testing.assert_allclose(
        state.template.numpy(), [[2, 3], [np.nan, 1]], equal_nan=True
    )
    np.testing.assert_allclose(
        state.matching_template().numpy(), [[1.5, 3], [3, 1]]
    )
    state.add(torch.tensor([[float("nan"), 7.0], [4.0, 3.0]]))
    np.testing.assert_allclose(
        state.template.numpy(), [[2, np.sqrt(8)], [2, np.sqrt(2)]]
    )
