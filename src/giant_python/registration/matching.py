"""Freshness-weighted normalized cross-correlation over integer shifts.

Equivalent to GIAnT-MATLAB ``xcorr2_nans_weighted``: for each shift
``(dr, dc)`` in ``[-S, S]^2``, the Pearson correlation between the frame's
valid pixels and the template at the shifted positions (inside the template
and not NaN), with weighted means, variances and covariance, each pixel
weighted by its freshness. The maximum is refined by a parabola fit on each
axis, unless it lies on the edge of the search range.

The correlation for all shifts of a batch of frames is six matrix products
``(frames x pixels) @ (pixels x shifts)``.
"""

from typing import Tuple

import torch


class RegistrationError(RuntimeError):
    """Raised when a frame cannot be matched (e.g. NaN motion)."""


def shift_tables(
    template: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    search: int,
    dtype=torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Template values at each pixel for every shift.

    Shifts are ordered as MATLAB's column-major ``C(drix, dcix)``: the row
    shift varies fastest.

    Parameters
    ----------
    template : torch.Tensor
        ``(H, W)`` template, NaN where invalid.
    rows, cols : torch.Tensor of int
        Zero-based positions of the frame's pixels in the template.
    search : int
        Largest shift ``S`` on each axis.
    dtype : torch.dtype
        Precision of the tables.

    Returns
    -------
    valid, value, value2 : torch.Tensor
        ``(pixels, shifts)``: whether the shifted template pixel is usable,
        its value (centred by the template's mean, 0 where unusable) and the
        value squared.
    """
    h, w = template.shape
    s = int(search)
    n = 2 * s + 1
    finite = template[~torch.isnan(template)]
    centre = finite.mean() if finite.numel() else template.new_zeros(())
    # NaN padding makes shifts that leave the template unusable
    wp = w + 2 * s
    padded = torch.full(
        (h + 2 * s, wp), float("nan"), dtype=dtype, device=template.device
    )
    padded[s : s + h, s : s + w] = (template - centre).to(dtype)
    sh = torch.arange(-s, s + 1, device=template.device)
    offset = sh.repeat(n) * wp + sh.repeat_interleave(n)  # dr fastest
    base = (rows + s) * wp + (cols + s)
    v = padded.reshape(-1)[base[:, None] + offset[None, :]]
    valid = ~torch.isnan(v)
    value = torch.nan_to_num(v, nan=0.0)
    return valid.to(dtype), value, value * value


def correlation_map(
    frames: torch.Tensor,
    weights: torch.Tensor,
    tables: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    search: int,
) -> torch.Tensor:
    """Weighted Pearson correlation for every shift.

    Parameters
    ----------
    frames : torch.Tensor
        ``(B, pixels)`` frame values; ignored where ``weights`` is 0.
    weights : torch.Tensor
        ``(B, pixels)`` freshness, 0 at invalid pixels.
    tables : tuple of torch.Tensor
        Output of :func:`shift_tables`.
    search : int
        Largest shift ``S``.

    Returns
    -------
    torch.Tensor
        ``(B, 2S + 1, 2S + 1)`` correlations indexed ``[b, dr, dc]``, NaN
        where undefined.
    """
    valid, value, value2 = tables
    dtype = value.dtype
    f = frames.to(dtype)
    w = torch.where(torch.isnan(f), 0.0, weights.to(dtype))
    b = f.shape[0]
    sw0 = w.sum(1, keepdim=True)
    mean0 = (w * torch.where(w > 0, f, 0.0)).sum(1, keepdim=True) / sw0
    fc = torch.where(w > 0, f - mean0, 0.0)
    wf = w * fc
    # three products instead of six: stack the frame-side vectors
    on_valid = torch.cat([w, wf, wf * fc]) @ valid
    on_value = torch.cat([w, wf]) @ value
    sw, s_f, s_ff = on_valid[:b], on_valid[b : 2 * b], on_valid[2 * b :]
    s_t, s_ft = on_value[:b], on_value[b:]
    s_tt = w @ value2
    m_f = s_f / sw
    m_t = s_t / sw
    cov = s_ft - sw * m_f * m_t
    v_f = s_ff - sw * m_f * m_f
    v_t = s_tt - sw * m_t * m_t
    corr = cov / torch.sqrt(v_f * v_t)
    n = 2 * search + 1
    return corr.reshape(-1, n, n).transpose(1, 2)


def peak_motion(
    corr: torch.Tensor, search: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Motion and correlation at the maximum of correlation maps.

    Ties go to the first maximum in MATLAB's column-major order. The motion
    is the negated (sub-pixel) shift, as ``xcorr2_nans_weighted`` returns.

    Parameters
    ----------
    corr : torch.Tensor
        ``(B, n, n)`` maps from :func:`correlation_map`.
    search : int
        Largest shift ``S``.

    Returns
    -------
    motion : torch.Tensor of float64
        ``(B, 2)`` [row, col] motion; NaN where the peak is NaN.
    peak : torch.Tensor of float64
        ``(B,)`` maximum correlation (NaN if every shift is undefined).
    """
    b = corr.shape[0]
    n = 2 * search + 1
    c = corr.to(torch.float64)
    rr, cc, peak = peak_index(c)
    shifts = torch.arange(
        -search, search + 1, device=c.device, dtype=torch.float64
    )
    mot = torch.stack([shifts[rr], shifts[cc]], 1)
    inner = (rr > 0) & (rr < n - 1) & (cc > 0) & (cc < n - 1)
    bi = torch.arange(b, device=c.device)
    r1 = rr.clamp(1, n - 2)
    c1 = cc.clamp(1, n - 2)
    c0 = c[bi, r1, c1]
    ratio_r = torch.clamp(
        (c0 - c[bi, r1 - 1, c1]) / (c0 - c[bi, r1 + 1, c1]), max=1e6
    )
    ratio_c = torch.clamp(
        (c0 - c[bi, r1, c1 - 1]) / (c0 - c[bi, r1, c1 + 1]), max=1e6
    )
    d_r = (1 - ratio_r) / (1 + ratio_r) / 2
    d_c = (1 - ratio_c) / (1 + ratio_c) / 2
    sub = torch.stack([mot[:, 0] - d_r, mot[:, 1] - d_c], 1)
    mot = torch.where(inner[:, None], sub, mot)
    # an undefined map (e.g. a frame without valid samples) has no peak;
    # its argmax would be the first shift, a finite but arbitrary motion
    mot = mot.masked_fill(torch.isnan(peak)[:, None], float("nan"))
    return -mot, peak


def peak_index(corr: torch.Tensor):
    """Integer position and value of each map's maximum.

    Ties go to the first maximum in MATLAB's column-major order; NaNs are
    ignored.

    Returns
    -------
    rr, cc : torch.Tensor of int
        Row- and column-shift indices of the maximum.
    peak : torch.Tensor
        The maximum (NaN if the whole map is undefined).
    """
    b, n_r, n_c = corr.shape
    flat = corr.transpose(1, 2).reshape(b, -1)
    idx = torch.argmax(torch.nan_to_num(flat, nan=-torch.inf), dim=1)
    return idx % n_r, idx // n_r, flat.gather(1, idx[:, None])[:, 0]


def peak_margin(corr: torch.Tensor, exclude: int = 2) -> torch.Tensor:
    """Maximum minus the largest value farther than ``exclude`` shifts.

    Parameters
    ----------
    corr : torch.Tensor
        ``(B, n, n)`` correlation maps.
    exclude : int
        Chebyshev radius around the maximum left out of the comparison.

    Returns
    -------
    torch.Tensor
        ``(B,)`` margins (NaN when nothing lies outside the radius).
    """
    b, n_r, n_c = corr.shape
    rr, cc, peak = peak_index(corr)
    ir = torch.arange(n_r, device=corr.device)
    ic = torch.arange(n_c, device=corr.device)
    near = ((ir[None, :, None] - rr[:, None, None]).abs() <= exclude) & (
        (ic[None, None, :] - cc[:, None, None]).abs() <= exclude
    )
    rest = torch.nan_to_num(corr, nan=-torch.inf).masked_fill(near, -torch.inf)
    second = rest.reshape(b, -1).max(1).values
    margin = peak - second
    return margin.masked_fill(torch.isinf(second), float("nan"))


def match_maps(
    frames: torch.Tensor,
    weights: torch.Tensor,
    template: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    search: int,
    dtype=torch.float32,
):
    """Like :func:`match`, also returning the correlation maps.

    Returns
    -------
    motion : torch.Tensor of float64
        ``(B, 2)`` [row, col] motion (NaN where undefined; not raised).
    peak : torch.Tensor of float64
        ``(B,)`` correlation at the maximum.
    corr : torch.Tensor
        ``(B, 2S + 1, 2S + 1)`` maps indexed ``[b, dr, dc]``.
    """
    search = int(round(search))
    tables = shift_tables(template, rows, cols, search, dtype)
    corr = correlation_map(frames, weights, tables, search)
    motion, peak = peak_motion(corr, search)
    return motion, peak, corr


def wide_correlation(
    frames: torch.Tensor, weights: torch.Tensor, template: torch.Tensor
) -> torch.Tensor:
    """Weighted Pearson correlation over every shift within a padded template.

    The same statistic as :func:`correlation_map`, evaluated with FFTs for
    all shifts at once: frame pixel ``(r, c)`` is compared with template
    pixel ``(r + m + dr, c + m + dc)``, where the template is the frame
    window padded by ``m`` on each side, for ``|dr|, |dc| <= m``. Unlike
    MATLAB's search, the template is not cropped around a starting shift.

    Parameters
    ----------
    frames, weights : torch.Tensor
        ``(B, H, W)`` frame images (NaN where invalid) and freshness.
    template : torch.Tensor
        ``(H + 2m, W + 2m)`` template, NaN where invalid.

    Returns
    -------
    torch.Tensor of float64
        ``(B, 2m + 1, 2m + 1)`` maps indexed ``[b, dr + m, dc + m]``.
    """
    dt = torch.float64
    ho, wo = template.shape
    h, w = frames.shape[-2:]
    reach = (ho - h) // 2
    if ho - h != 2 * reach or wo - w != 2 * reach:
        raise ValueError("the template must pad the frames equally")
    f = frames.to(dt)
    wt = torch.where(torch.isnan(f), 0.0, weights.to(dt))
    mean0 = (wt * torch.nan_to_num(f)).sum((1, 2), keepdim=True) / wt.sum(
        (1, 2), keepdim=True
    )
    fc = torch.where(wt > 0, f - mean0, 0.0)
    t = template.to(dt)
    tv = ~torch.isnan(t)
    tz = torch.where(tv, t - t[tv].mean(), 0.0)
    size = (ho, wo)
    fa = [torch.fft.rfft2(a, s=size) for a in (wt, wt * fc, wt * fc * fc)]
    fb = [torch.fft.rfft2(b) for b in (tv.to(dt), tz, tz * tz)]
    n = 2 * reach + 1

    def xc(a, b):
        """Correlation of frame-side ``a`` with template-side ``b``."""
        out = torch.fft.irfft2(torch.conj(a) * b[None], s=size)
        return out[:, :n, :n]

    sw, s_f, s_ff = xc(fa[0], fb[0]), xc(fa[1], fb[0]), xc(fa[2], fb[0])
    s_t, s_tt, s_ft = xc(fa[0], fb[1]), xc(fa[0], fb[2]), xc(fa[1], fb[1])
    m_f = s_f / sw
    m_t = s_t / sw
    cov = s_ft - sw * m_f * m_t
    v_f = s_ff - sw * m_f * m_f
    v_t = s_tt - sw * m_t * m_t
    corr = cov / torch.sqrt(v_f * v_t)
    return corr.masked_fill(
        sw < 1e-9 * sw.amax((1, 2), keepdim=True), float("nan")
    )


def match(
    frames: torch.Tensor,
    weights: torch.Tensor,
    template: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    search: int,
    dtype=torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Register frames to a template (``xcorr2_nans_weighted``).

    Parameters
    ----------
    frames, weights : torch.Tensor
        ``(B, pixels)`` frame values and freshness (0 where invalid).
    template : torch.Tensor
        ``(H, W)`` template, NaN where invalid.
    rows, cols : torch.Tensor of int
        Zero-based pixel positions in the template.
    search : int
        Largest shift on each axis.
    dtype : torch.dtype
        Precision of the correlation sums.

    Returns
    -------
    motion : torch.Tensor of float64
        ``(B, 2)`` [row, col] motion.
    peak : torch.Tensor of float64
        ``(B,)`` correlation at the maximum.

    Raises
    ------
    RegistrationError
        If a motion is NaN, as ``xcorr2_nans_weighted`` does; this includes
        frames whose correlation is undefined at every shift (e.g. frames
        without valid samples).
    """
    search = int(round(search))
    tables = shift_tables(template, rows, cols, search, dtype)
    motion, peak = peak_motion(
        correlation_map(frames, weights, tables, search), search
    )
    if torch.isnan(motion).any():
        raise RegistrationError("Computed motion contains NaN.")
    return motion, peak
