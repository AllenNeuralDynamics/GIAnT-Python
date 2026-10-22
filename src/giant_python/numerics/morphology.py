"""Shared binary morphology on trailing image axes."""

from __future__ import annotations

from typing import Optional

import numpy as np


def fast_dilation(
    mask: np.ndarray,
    kernel: Optional[np.ndarray] = None,
    iterations: int = 1,
) -> np.ndarray:
    """Binary-dilate the trailing two axes of ``mask``.

    The common 3x3 / single-iteration case is handled with a pure-numpy shift
    accumulation (fast, no OpenCV); other cases fall back to per-slice
    ``cv2.dilate`` over all leading axes.

    Parameters
    ----------
    mask : ndarray
        Boolean (or 0/1) array; the last two axes are the image plane.
    kernel : ndarray, optional
        Structuring element; defaults to a 7x7 block.
    iterations : int
        Number of dilation iterations (generic path only).

    Returns
    -------
    ndarray of bool
        The dilated mask.
    """
    if kernel is None:
        kernel = np.ones((7, 7), np.uint8)

    if iterations == 1 and kernel.shape == (3, 3) and np.all(kernel):
        m = mask.astype(bool, copy=False)
        pad = [(0, 0)] * (m.ndim - 2) + [(1, 1), (1, 1)]
        p = np.pad(m, pad, mode="constant", constant_values=False)
        out = np.zeros_like(m, dtype=bool)
        h, w = m.shape[-2], m.shape[-1]
        for dr in range(3):
            r_end = dr + h
            for dc in range(3):
                c_end = dc + w
                out |= p[..., dr:r_end, dc:c_end]
        return out

    import cv2

    out = np.empty_like(mask, dtype=bool)
    for idx in np.ndindex(mask.shape[:-2]):
        out[idx] = cv2.dilate(
            mask[idx].astype(np.uint8, copy=False),
            kernel,
            iterations=iterations,
        ).astype(bool, copy=False)
    return out
