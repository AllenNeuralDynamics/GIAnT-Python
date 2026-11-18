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
        ``(B, 2)`` [row, col] motion.
    peak : torch.Tensor of float64
        ``(B,)`` maximum correlation (NaN if every shift is undefined).
    """
    b = corr.shape[0]
    n = 2 * search + 1
    c = corr.to(torch.float64)
    flat = c.transpose(1, 2).reshape(b, -1)
    filled = torch.nan_to_num(flat, nan=-torch.inf)
    idx = torch.argmax(filled, dim=1)
    peak = flat.gather(1, idx[:, None])[:, 0]
    rr = idx % n
    cc = idx // n
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
    return -mot, peak


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
        If a motion is NaN, as ``xcorr2_nans_weighted`` does.
    """
    search = int(round(search))
    tables = shift_tables(template, rows, cols, search, dtype)
    motion, peak = peak_motion(
        correlation_map(frames, weights, tables, search), search
    )
    if torch.isnan(motion).any():
        raise RegistrationError("Computed motion contains NaN.")
    return motion, peak
