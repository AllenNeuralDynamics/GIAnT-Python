"""Source seeds and their selected pixels in SiLo's consensus frame.

Port of the seeding part of GIAnT-MATLAB's ``SILo.m`` (main at 5a0673b):

1. seeds in the combined activity image (:func:`seed_sources`; the default
   method is :func:`~giant_python.extraction.raster.peaks.get_act_im_peaks`,
   and ``"wmf"`` is reserved for the port of ``findSourcesWMF``);
2. sources at the seeds' rounded positions, duplicates removed
   (:func:`sources_from_seeds`);
3. each source's selected pixels: a disk of ``selRadius`` around it
   (MATLAB's ``strel('disk', r)``, :func:`disk_strel`) within the pixels
   imaged in enough trials, and the pruning of sources that this clips
   (:func:`select_source_pixels`).

Positions are zero-based ``(row, col)`` in MATLAB's orientation. Pixel lists
are linear indices in MATLAB's column-major order, ``row + col * rows``, so
that a sorted list is in the order of MATLAB's ``find``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import convolve2d

from .alignment import matlab_round
from .peaks import get_act_im_peaks

SEEDING_METHODS = ("getactimpeaks", "wmf")

# offsets of the periodic lines of MATLAB's disk decomposition
_DISK_LINES = {
    4: ((1, 0), (1, 1), (0, 1), (-1, 1)),
    6: ((1, 0), (1, 2), (2, 1), (0, 1), (-1, 2), (-2, 1)),
    8: (
        (1, 0),
        (2, 1),
        (1, 1),
        (1, 2),
        (0, 1),
        (-1, 2),
        (-1, 1),
        (-2, 1),
    ),
}


def _dilate_full(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dilation of one binary shape by another, grown to hold the result
    (MATLAB's ``imdilate(..., 'full')``)."""
    return convolve2d(a.astype(np.int64), b.astype(np.int64)) > 0


def _periodic_line(reps: int, offset: tuple[int, int]) -> np.ndarray:
    """MATLAB's ``strel('periodicline', reps, offset)``: the points
    ``k * offset`` for ``k`` in ``-reps .. reps``."""
    dr, dc = offset
    line = np.zeros((2 * reps * abs(dr) + 1, 2 * reps * abs(dc) + 1), bool)
    for k in range(-reps, reps + 1):
        line[reps * abs(dr) + k * dr, reps * abs(dc) + k * dc] = True
    return line


def disk_strel(radius: int, n: int = 4) -> np.ndarray:
    """Neighbourhood of MATLAB's ``strel('disk', radius, n)``.

    MATLAB approximates the disk by dilating ``n`` periodic lines (Adams
    1993; Jones and Soille 1996) and then lines along the rows and columns
    that make up the missing radius. The result is an octagon for ``n = 4``
    and is smaller than the Euclidean disk: 11 x 11 for radius 6, and a full
    5 x 5 square for radius 3. Radii under 3 (or ``n = 0``) give the
    Euclidean disk.

    Parameters
    ----------
    radius : int
        The disk's radius (non-negative).
    n : int
        Number of periodic lines: 0, 4 (MATLAB's default), 6 or 8.

    Returns
    -------
    ndarray of bool
        The square, symmetric neighbourhood with its origin in the centre.
    """
    radius = int(radius)
    if radius < 0:
        raise ValueError("radius must not be negative")
    if n not in (0, 4, 6, 8):
        raise ValueError(f"n must be 0, 4, 6 or 8, not {n}")
    if radius < 3 or n == 0:
        ax = np.arange(-radius, radius + 1)
        return ax[:, None] ** 2 + ax[None, :] ** 2 <= radius**2
    theta = math.pi / (2 * n)
    k = 2 * radius / (1 / math.tan(theta) + 1 / math.sin(theta))
    nhood = np.ones((1, 1), bool)
    for offset in _DISK_LINES[n]:
        reps = math.floor(k / math.hypot(*offset))
        nhood = _dilate_full(nhood, _periodic_line(reps, offset))
    reach = (nhood.shape[0] - 1) // 2  # the shape is symmetric
    length = 2 * (radius - reach - 1) + 1
    if length >= 3:
        nhood = _dilate_full(nhood, np.ones((1, length), bool))
        nhood = _dilate_full(nhood, np.ones((length, 1), bool))
    return nhood


def seed_sources(
    act_im: np.ndarray,
    peak_th: float,
    soma_mask: np.ndarray | None = None,
    min_peak_distance: int = 1,
    method: str = "getactimpeaks",
) -> np.ndarray:
    """Seed sources in the combined activity image.

    Parameters
    ----------
    act_im : ndarray of shape (rows, cols)
        The combined activity image (NaN where not imaged in every valid
        trial).
    peak_th : float
        Threshold in robust SDs of the image (``peakth``).
    soma_mask : ndarray of bool, optional
        Pixels where no source may be seeded.
    min_peak_distance : int
        ``minPeakDistance``.
    method : str
        ``"getactimpeaks"`` (maxima fitted with Gaussians). ``"wmf"``
        (MATLAB's ``findSourcesWMF``) is not ported yet.

    Returns
    -------
    ndarray of shape (n, 4)
        ``[amplitude, row, col, sigma]`` of each seed (zero-based, subpixel;
        MATLAB's ``thetaf``).
    """
    method = str(method).lower()
    if method not in SEEDING_METHODS:
        raise ValueError(
            f"method must be one of {SEEDING_METHODS}, not {method!r}"
        )
    if method == "wmf":
        raise NotImplementedError(
            "WMF seeding (findSourcesWMF) is not ported yet"
        )
    act_im = np.asarray(act_im, dtype=np.float64)
    mask = None
    if soma_mask is not None and np.any(soma_mask):
        mask = np.asarray(soma_mask, dtype=bool)
    usable = ~np.isnan(act_im) if mask is None else ~np.isnan(act_im) & ~mask
    if not usable.any():  # SILo.m: no sources without a pixel to seed in
        return np.zeros((0, 4))
    return get_act_im_peaks(act_im, peak_th, mask, min_peak_distance)


def sources_from_seeds(
    seeds: np.ndarray, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sources at the seeds' rounded positions, without duplicates.

    Positions are rounded as MATLAB's ``round`` (halves up) and the first
    seed of each pixel is kept, in the seeds' order (``unique(..., 'rows',
    'stable')``). Positions are clamped to the image (a fit that ends on the
    image's last half pixel rounds to the row or column after it, which
    MATLAB would index out of bounds).

    Parameters
    ----------
    seeds : ndarray of shape (n, 4)
        ``[amplitude, row, col, sigma]`` (zero-based).
    shape : tuple of int
        ``(rows, cols)`` of the consensus frame.

    Returns
    -------
    rows, cols : ndarray of int
        The sources' pixels.
    values : ndarray of float
        The seeds' amplitudes (MATLAB's ``sources.V``).
    """
    seeds = np.asarray(seeds, dtype=np.float64).reshape(-1, 4)
    # round in MATLAB's one-based coordinates, where every position is > 0
    rows = matlab_round(seeds[:, 1] + 1).astype(np.int64) - 1
    cols = matlab_round(seeds[:, 2] + 1).astype(np.int64) - 1
    rows = np.clip(rows, 0, shape[0] - 1)
    cols = np.clip(cols, 0, shape[1] - 1)
    _, first = np.unique(rows + cols * shape[0], return_index=True)
    first.sort()
    return rows[first], cols[first], seeds[first, 0]


@dataclass
class SourceSelection:
    """Result of :func:`select_source_pixels`, before pruning.

    Attributes
    ----------
    pixels : list of ndarray of int
        For each source, its selected pixels as sorted column-major linear
        indices (``row + col * rows``) into the consensus frame.
    keep : ndarray of bool
        Sources that keep more than ``min_pixels`` pixels and whose own
        pixel is always valid (MATLAB's ``keepSources``).
    """

    pixels: list
    keep: np.ndarray


def select_source_pixels(
    rows: np.ndarray,
    cols: np.ndarray,
    px_always_valid: np.ndarray,
    sel_radius: int,
    min_pixels: int = 5,
) -> SourceSelection:
    """Each source's selected pixels and which sources survive.

    A source's pixels are MATLAB's ``strel('disk', sel_radius)`` around it,
    within the image and within ``px_always_valid``. A source is kept if it
    has more than ``min_pixels`` pixels and its own pixel is valid.

    Parameters
    ----------
    rows, cols : ndarray of int
        The sources' pixels (zero-based).
    px_always_valid : ndarray of bool, shape (rows, cols)
        Pixels imaged in enough of the valid trials.
    sel_radius : int
        ``selRadius`` (``ceil(2 * dXY)``).
    min_pixels : int
        A source needs more selected pixels than this.

    Returns
    -------
    SourceSelection
    """
    valid = np.asarray(px_always_valid, dtype=bool)
    h, w = valid.shape
    nhood = disk_strel(sel_radius)
    half = (nhood.shape[0] - 1) // 2
    d_col, d_row = np.nonzero(nhood.T)  # column-major: sorted linear indices
    d_row, d_col = d_row - half, d_col - half
    pixels = []
    keep = np.zeros(len(rows), bool)
    for k, (r, c) in enumerate(zip(rows, cols)):
        rr, cc = r + d_row, c + d_col
        ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        rr, cc = rr[ok], cc[ok]
        ok = valid[rr, cc]
        pixels.append((rr[ok] + cc[ok] * h).astype(np.int64))
        keep[k] = pixels[k].size > min_pixels and valid[r, c]
    return SourceSelection(pixels, keep)
