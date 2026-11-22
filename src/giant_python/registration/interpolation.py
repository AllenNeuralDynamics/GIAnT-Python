"""Freshness-weighted bilinear resampling of a frame by a rigid shift.

Equivalent to GIAnT-MATLAB ``interpFrame``: the output pixel ``(i, j)``
(one-based, ``i = 1 .. H + 2m``) samples the frame at row ``i - m + r`` and
column ``j - m + c``. Each of the four neighbours is weighted by its
bilinear coefficient times its freshness; a neighbour outside the frame makes
the output NaN, as does a NaN neighbour (even one with zero weight). The
variance factor is the reciprocal of the summed weights.
"""

import math
from typing import Tuple

import numpy as np
import torch


def _corner(x: torch.Tensor, ri: torch.Tensor, ci: torch.Tensor):
    """Values at rows ``ri`` and columns ``ci``, NaN outside ``x``."""
    h, w = x.shape
    out = x.index_select(0, ri.clamp(0, h - 1)).index_select(
        1, ci.clamp(0, w - 1)
    )
    bad_r = (ri < 0) | (ri >= h)
    bad_c = (ci < 0) | (ci >= w)
    out = out.masked_fill(bad_r[:, None], float("nan"))
    return out.masked_fill(bad_c[None, :], float("nan"))


def interp_frame(
    frame: torch.Tensor,
    freshness: torch.Tensor,
    motion_r: float,
    motion_c: float,
    margin: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resample one frame onto the padded output grid.

    Parameters
    ----------
    frame : torch.Tensor
        ``(H, W)`` float32 frame, NaN where unsampled.
    freshness : torch.Tensor
        ``(H, W)`` float32 freshness, 0 where unsampled.
    motion_r, motion_c : float
        Row and column offsets added to the output grid.
    margin : int
        Padding ``m`` of the output grid on each side.

    Returns
    -------
    image : torch.Tensor
        ``(H + 2m, W + 2m)`` resampled frame.
    var_factor : torch.Tensor
        Same shape: ``1 / sum(coefficient * freshness)``, NaN where that sum
        is 0 or undefined.
    """
    h, w = frame.shape
    # one-based sample positions, as MATLAB's viewR(:,1) + motion; index
    # arithmetic stays on the CPU (no device synchronization)
    view_r = np.arange(1, h + 2 * margin + 1) - margin + float(motion_r)
    view_c = np.arange(1, w + 2 * margin + 1) - margin + float(motion_c)
    first_r = 1 - margin + float(motion_r)
    first_c = 1 - margin + float(motion_c)
    fr = first_r - math.floor(first_r)
    fc = first_c - math.floor(first_c)
    idx = (
        np.floor(view_r).astype(np.int64) - 1,
        np.ceil(view_r).astype(np.int64) - 1,
        np.floor(view_c).astype(np.int64) - 1,
        np.ceil(view_c).astype(np.int64) - 1,
    )
    coef = [
        float(np.float32(v)) if frame.dtype == torch.float32 else v
        for v in (
            (1 - fc) * (1 - fr),
            fc * (1 - fr),
            (1 - fc) * fr,
            fc * fr,
        )
    ]
    if all(np.all(np.diff(i) == 1) for i in idx):
        # every output row/column reads a run of input rows/columns: slice
        # NaN-padded copies instead of gathering
        pr = margin + abs(math.floor(motion_r)) + 2
        pc = margin + abs(math.floor(motion_c)) + 2
        pads = [
            torch.nn.functional.pad(
                x[None], (pc, pc, pr, pr), value=float("nan")
            )[0]
            for x in (frame, freshness)
        ]

        def corner(which, ri, ci):
            """Slice of the padded frame (0) or freshness (1)."""
            r0, c0 = int(ri[0]) + pr, int(ci[0]) + pc
            return pads[which][r0 : r0 + ri.size, c0 : c0 + ci.size]

    else:
        dev = frame.device

        def corner(which, ri, ci):
            """Gathered frame (0) or freshness (1), NaN outside."""
            return _corner(
                (frame, freshness)[which],
                torch.from_numpy(ri).to(dev),
                torch.from_numpy(ci).to(dev),
            )

    r_lo, r_hi, c_lo, c_hi = idx
    num = None
    den = None
    for k, (ri, ci) in enumerate(
        ((r_lo, c_lo), (r_lo, c_hi), (r_hi, c_lo), (r_hi, c_hi))
    ):
        cf = coef[k] * corner(1, ri, ci)
        term = cf * corner(0, ri, ci)
        num = term if num is None else num + term
        den = cf if den is None else den + cf
    image = num / den
    var_factor = 1.0 / den
    var_factor = var_factor.masked_fill(torch.isinf(var_factor), float("nan"))
    return image, var_factor
