"""Source seeds and their selected pixels in SiLo's consensus frame.

Port of the seeding part of GIAnT-MATLAB's ``SILo.m`` (main at 5a0673b):

1. seeds in the combined activity image (:func:`seed_sources`; the default
   method is :func:`~giant_python.extraction.raster.peaks.get_act_im_peaks`,
   and ``"wmf"`` is reserved for the port of ``findSourcesWMF``); the
   threshold counts in robust SDs of the image itself (MATLAB's), or, with
   ``acceptance="null"`` (not in MATLAB), is set from the null of the
   combined image so that it admits a chosen expected number of false seeds
   per field (:func:`null_threshold`);
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
from .peaks import _MAD_SCALE, get_act_im_peaks, ordfilt3x3

SEEDING_METHODS = ("getactimpeaks", "wmf")
#: How a seed's threshold is set: robust SDs of the image (``"image"``,
#: MATLAB's), false seeds per field under the image's null (``"null"``), or
#: robust SDs of the image's null (``"null_sd"``).
SEED_ACCEPTANCE = ("image", "null", "null_sd")
# largest candidates of the null (per field) that its exponential tail is
# fitted to
_TAIL_K = 50

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


def candidate_statistic(
    act_im: np.ndarray, exclusion: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """The seeding's candidate statistic at an image's local maxima.

    A pixel is a candidate of
    :func:`~giant_python.extraction.raster.peaks.get_act_im_peaks` when it
    is the maximum of its 3 x 3 neighbourhood and the second-largest value
    there exceeds the threshold; this returns, for every 3 x 3 maximum
    outside ``exclusion`` (and away from NaN), that second-largest value.

    Returns
    -------
    rc : ndarray of int, shape (n, 2)
        The maxima's (row, col).
    value : ndarray of shape (n,)
        Their second-largest 3 x 3 value.
    """
    act = np.asarray(act_im, dtype=np.float64)
    excl = np.zeros(act.shape, bool)
    if exclusion is not None:
        excl = np.asarray(exclusion, bool)
        act = act * ~excl  # as get_act_im_peaks: excluded pixels are 0
    second = ordfilt3x3(act, 8)
    with np.errstate(invalid="ignore"):
        is_max = (act == ordfilt3x3(act, 9)) & np.isfinite(second) & ~excl
    rc = np.argwhere(is_max)
    return rc, second[is_max]


def null_background(null_ims: np.ndarray) -> tuple[float, float]:
    """Median and robust SD (MATLAB's MAD scale) of null images' finite
    pixels."""
    vals = np.asarray(null_ims, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size < 10:
        raise ValueError("the null image has too few pixels")
    mu = float(np.median(vals))
    sigma = float(np.median(np.abs(vals - mu))) / _MAD_SCALE
    if not sigma > 0:
        raise ValueError("the null image has no spread")
    return mu, sigma


def null_threshold(
    null_ims: np.ndarray,
    false_per_field: float,
    exclusion: np.ndarray | None = None,
    tail_k: int = _TAIL_K,
) -> tuple[tuple[float, float], float, dict]:
    """Seed threshold from the null of a combined activity image.

    ``null_ims`` holds one or more images of what the combined activity
    image would be if every trial held only noise (the bootstrap null's
    images, combined as the data:
    :func:`~giant_python.extraction.raster.plan.combined_null_image`). The
    threshold's centre and scale are the null's median and robust SD (in
    place of the image's own, which rise with the density of real
    sources), and its level is where the null's candidates
    (:func:`candidate_statistic`) exceed it ``false_per_field`` times per
    image: read off the null where it has at least ``tail_k`` candidates
    per image above it, else extrapolated with an exponential tail fitted
    to the ``tail_k`` largest candidates per image.

    Parameters
    ----------
    null_ims : ndarray of shape (draws, rows, cols) or (rows, cols)
        Null combined images (NaN outside the image).
    false_per_field : float
        Expected false seeds per field (per combined image), > 0.
    exclusion : ndarray of bool, optional
        Pixels where no seed may be placed (the soma mask).
    tail_k : int
        Candidates per image the tail is fitted to.

    Returns
    -------
    background : (float, float)
        The null's median and robust SD (MATLAB's MAD scale).
    peak_th : float
        The threshold in units of ``background``.
    info : dict
        ``draws``, ``candidates`` (per image), ``tail_start`` and
        ``tail_scale`` (in units of the background) and ``extrapolated``.
    """
    if not false_per_field > 0:
        raise ValueError("false_per_field must be positive")
    null = np.asarray(null_ims, dtype=np.float64)
    if null.ndim == 2:
        null = null[None]
    mu, sigma = null_background(null)
    z = np.concatenate(
        [(candidate_statistic(im, exclusion)[1] - mu) / sigma for im in null]
    )
    z = np.sort(z)[::-1]
    draws = null.shape[0]
    k = int(min(tail_k * draws, z.size))
    if k < 2:
        raise ValueError("the null image has too few local maxima")
    want = false_per_field * draws  # exceedances over all draws
    u = float(z[k - 1])
    beta = float(np.mean(z[: k - 1] - u))
    if want >= k:  # within the null's own candidates: read it off
        th, extrapolated = float(z[min(int(np.ceil(want)), z.size) - 1]), False
    else:
        th = u + max(beta, 1e-12) * float(np.log(k / want))
        extrapolated = True
    info = dict(
        draws=draws,
        candidates=z.size / draws,
        tail_start=u,
        tail_scale=beta,
        extrapolated=extrapolated,
    )
    return (mu, sigma), th, info


def seed_sources(
    act_im: np.ndarray,
    peak_th: float,
    soma_mask: np.ndarray | None = None,
    min_peak_distance: int = 1,
    method: str = "getactimpeaks",
    fit: str = "matlab",
    merge_px: float = 0.0,
    *,
    acceptance: str = "image",
    null_im: np.ndarray | None = None,
    false_per_field: float = 1.0,
    info: dict | None = None,
    amplitude_rule: str = "matlab",
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
    fit : {'matlab', 'local'}
        The Gaussian fits (:func:`~giant_python.extraction.raster.peaks.
        get_act_im_peaks`): MATLAB's, or the faster local ones.
    merge_px : float
        Reject a residual peak that ends closer than this to another peak
        (0: MATLAB's rule only).
    acceptance : {'image', 'null', 'null_sd'}
        ``'image'``: ``peak_th`` robust SDs above the image's median
        (MATLAB's). ``'null'``: the threshold that ``null_im`` crosses
        ``false_per_field`` times per field (:func:`null_threshold`);
        ``peak_th`` is then not used. ``'null_sd'``: ``peak_th`` robust SDs
        above the median of ``null_im`` (a scale that does not grow with
        the density of real sources; no tail is extrapolated).
    null_im : ndarray of shape (draws, rows, cols), optional
        The combined image's null (needed with ``acceptance='null'``).
    false_per_field : float
        Expected false seeds per field with ``acceptance='null'``.
    amplitude_rule : {'matlab', 'pixel'}
        The final test of every fitted peak (see
        :func:`~giant_python.extraction.raster.peaks.get_act_im_peaks`).
    info : dict, optional
        Receives the threshold used (``peak_th``, ``background``: None for
        the image's own) and, with ``acceptance='null'``, the null's tail
        (:func:`null_threshold`).

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
    _check_acceptance(acceptance, null_im)
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
    background, peak_th = _threshold(
        acceptance, peak_th, null_im, false_per_field, mask, info
    )
    return get_act_im_peaks(
        act_im,
        peak_th,
        mask,
        min_peak_distance,
        fit,
        merge_px,
        background,
        amplitude_rule,
    )


def _check_acceptance(acceptance: str, null_im) -> None:
    """Reject an unknown acceptance, or a null one without the null."""
    if acceptance not in SEED_ACCEPTANCE:
        raise ValueError(
            f"acceptance must be one of {SEED_ACCEPTANCE}, "
            f"not {acceptance!r}"
        )
    if acceptance != "image" and null_im is None:
        raise ValueError(
            f"acceptance={acceptance!r} needs the null image (null_im)"
        )


def _threshold(acceptance, peak_th, null_im, false_per_field, mask, info):
    """The background (None: the image's own) and threshold of a seeding;
    recorded in ``info`` when given."""
    background = None
    if acceptance == "null":
        background, peak_th, tail = null_threshold(
            null_im, false_per_field, mask
        )
        if info is not None:
            info.update(tail)
    elif acceptance == "null_sd":
        background = null_background(null_im)
    if info is not None:
        info.update(peak_th=float(peak_th), background=background)
    return background, peak_th


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
