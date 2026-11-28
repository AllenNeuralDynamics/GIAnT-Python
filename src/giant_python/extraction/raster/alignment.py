"""Across-trial alignment of one imaging path (SiLo's consensus frame).

Port of the alignment part of GIAnT-MATLAB's ``SILo.m`` (main at 5a0673b)
and its helpers ``makeConsensusTemplate``, ``dftregistration_clipped``
(the whole-pixel branch SiLo uses), ``xcorr2_nans``, ``trialWarpCoords`` and
``warpTrialImage`` / ``warpImageAffine``:

1. a consensus template from up to 20 trials' mean images: every pair is
   registered, the trial that correlates best with the others is the
   reference, and the trials that match it at least as well as the median
   are averaged around it on a canvas ``max_shift`` larger on every side;
2. each trial's mean image is registered to the template: a whole-pixel
   estimate by FFT cross-correlation, refined by a normalized
   cross-correlation over ``+-max_shift`` pixels with a parabola fit;
3. the shifts are the trials' maps from the consensus frame to their
   registered images (:class:`TrialWarp`), which resample trial images into
   the consensus frame (:func:`warp_trial_image`).

The shifts can be refined groupwise (``interTrialAlignment``), as GIAnT-MATLAB
main does after the rigid fit:

- ``'affine'`` (:func:`align_trials_affine`, port of
  ``alignTrialsAffine.m``): one affine map per trial;
- ``'patchwise'`` (:func:`align_trials_patchwise`, port of
  ``alignTrialsPatchwise.m``, ``patchFieldMatrix.m``, ``patchFieldApply.m``):
  a smooth displacement field per trial, set by its values at the centres of
  patches of imaged pixels, on top of the field's plane (an affine map).

:class:`TrialWarp` holds any of the three maps; :meth:`TrialWarp.coords`
(``trialWarpCoords.m``) and :meth:`TrialWarp.inverse`
(``trialWarpInverse.m``) carry points between the consensus frame and a
trial's registered image, and :func:`warp_trial_image`
(``warpTrialImage.m``) resamples images.

Images are ``(rows, cols)`` in MATLAB's orientation and coordinates are
zero-based: a shift of ``(dr, dc)`` means that consensus pixel ``(r, c)`` is
at ``(r + dr, c + dc)`` in the trial's registered image. Patch centres are
zero-based too; displacements are the same in either base.

Against MATLAB: the k-means clustering that makes the patches
(:func:`make_patches`) draws from numpy's generator, not MATLAB's, so the
patches differ; given MATLAB's patches (``patches=``) the fits follow
MATLAB's arithmetic, with numpy's least-squares and linear solvers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.ndimage as ndimage
import scipy.sparse as sparse
from scipy.spatial import cKDTree

_EPS = float(np.finfo(np.float64).eps)


def matlab_round(x) -> np.ndarray:
    """MATLAB's ``round``: halves away from zero (numpy's go to even)."""
    x = np.asarray(x, dtype=np.float64)
    whole = np.trunc(x)
    # x - whole is exact, unlike floor(|x| + 0.5) just below a half
    return whole + np.sign(x) * (np.abs(x - whole) >= 0.5)


def interp2_linear(im: np.ndarray, rows, cols) -> np.ndarray:
    """MATLAB ``interp2(im, cols, rows, 'linear')`` at zero-based points.

    Bilinear interpolation, NaN outside the image and wherever a pixel that
    enters with a non-zero weight is NaN. As in MATLAB, a point exactly on a
    row (or column) of the grid does not use the next row (or column), so a
    pixel keeps its value next to a NaN.

    Parameters
    ----------
    im : ndarray of shape (rows, cols)
        The image.
    rows, cols : array_like
        Zero-based query coordinates (the same shape).

    Returns
    -------
    ndarray
        The interpolated values, in the shape of ``rows``.
    """
    im = np.asarray(im, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.float64)
    cols = np.asarray(cols, dtype=np.float64)
    rows, cols = np.broadcast_arrays(rows, cols)
    h, w = im.shape
    with np.errstate(invalid="ignore"):
        inside = (rows >= 0) & (rows <= h - 1) & (cols >= 0) & (cols <= w - 1)
    r = np.where(inside, rows, 0.0)
    c = np.where(inside, cols, 0.0)
    r0 = np.floor(r).astype(np.intp)
    c0 = np.floor(c).astype(np.intp)
    fr, fc = r - r0, c - c0
    r1 = np.minimum(r0 + 1, h - 1)
    c1 = np.minimum(c0 + 1, w - 1)
    out = np.zeros(rows.shape)
    for wgt, rr, cc in (
        ((1 - fr) * (1 - fc), r0, c0),
        (fr * (1 - fc), r1, c0),
        ((1 - fr) * fc, r0, c1),
        (fr * fc, r1, c1),
    ):
        used = wgt != 0
        out[used] += wgt[used] * im[rr[used], cc[used]]
    out[~inside] = np.nan
    return out


def _keys_weights(s: np.ndarray) -> tuple:
    """Weights of Keys' cubic convolution (a = -1/2) at offsets -1, 0, 1, 2
    from the grid point below a query, ``s`` its distance from it."""
    s2 = s * s
    s3 = s2 * s
    return (
        (-s3 + 2 * s2 - s) / 2,
        (3 * s3 - 5 * s2 + 2) / 2,
        (-3 * s3 + 4 * s2 + s) / 2,
        (s3 - s2) / 2,
    )


class CubicInterpolant:
    """MATLAB's ``interp2(im, cols, rows, 'cubic')`` at zero-based points.

    Keys' cubic convolution on the pixel grid, with MATLAB's boundary
    condition (one row and column outside each edge, ``3 f(0) - 3 f(1) +
    f(2)``). NaN outside the image and wherever any of the 16 pixels around
    a point is NaN, also those with zero weight (as MATLAB; checked against
    R2024b). The extended image is built once, for the many evaluations of
    a Gauss-Newton fit.

    Parameters
    ----------
    im : ndarray of shape (rows, cols)
        The image (at least 3 x 3).
    """

    def __init__(self, im: np.ndarray) -> None:
        """Extend the image by MATLAB's boundary rows and columns."""
        im = np.asarray(im, dtype=np.float64)
        h, w = im.shape
        if h < 3 or w < 3:
            raise ValueError(
                f"cubic interpolation needs 3 x 3, not {im.shape}"
            )
        e = np.empty((h + 2, w))
        e[1:-1] = im
        e[0] = 3 * im[0] - 3 * im[1] + im[2]
        e[-1] = 3 * im[-1] - 3 * im[-2] + im[-3]
        ext = np.empty((h + 2, w + 2))
        ext[:, 1:-1] = e
        ext[:, 0] = 3 * e[:, 0] - 3 * e[:, 1] + e[:, 2]
        ext[:, -1] = 3 * e[:, -1] - 3 * e[:, -2] + e[:, -3]
        self.shape = (h, w)
        self._flat = ext.ravel()
        self._w = w + 2

    def __call__(self, rows, cols) -> np.ndarray:
        """Values at zero-based ``rows``, ``cols`` (equal shapes)."""
        h, w = self.shape
        r = np.asarray(rows, dtype=np.float64)
        c = np.asarray(cols, dtype=np.float64)
        r, c = np.broadcast_arrays(r, c)
        with np.errstate(invalid="ignore"):
            inside = (r >= 0) & (r <= h - 1) & (c >= 0) & (c <= w - 1)
        r = np.where(inside, r, 0.0)
        c = np.where(inside, c, 0.0)
        i = np.minimum(np.floor(r).astype(np.intp), h - 2)
        j = np.minimum(np.floor(c).astype(np.intp), w - 2)
        wr = _keys_weights(r - i)
        wc = _keys_weights(c - j)
        base = i * self._w + j  # ext[i + a, j + b] is im[i + a - 1, j + b - 1]
        out = np.zeros(r.shape)
        for a in range(4):
            row = base + a * self._w
            part = np.zeros(r.shape)
            for b in range(4):
                part += wc[b] * self._flat[row + b]
            out += wr[a] * part
        out[~inside] = np.nan
        return out


def interp2_cubic(im: np.ndarray, rows, cols) -> np.ndarray:
    """MATLAB ``interp2(im, cols, rows, 'cubic')`` at zero-based points (see
    :class:`CubicInterpolant`)."""
    return CubicInterpolant(im)(rows, cols)


def matlab_gradient(im: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """MATLAB's ``[gc, gr] = gradient(im)`` as ``(gr, gc)``: central
    differences inside, one-sided at the edges, NaN carried."""
    im = np.asarray(im, dtype=np.float64)
    return np.gradient(im, axis=0), np.gradient(im, axis=1)


def _colmajor_nonzero(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rows and columns of a mask's pixels in MATLAB's ``find`` order
    (column by column)."""
    cols, rows = np.nonzero(np.asarray(mask, dtype=bool).T)
    return rows, cols


def _sqrt_nan(m: np.ndarray) -> np.ndarray:
    """``sqrt(max(M, 0))`` with NaN kept (roughly variance-stabilized for
    shot noise; unscanned pixels stay NaN, not sharp zero edges)."""
    m = np.asarray(m, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        s = np.sqrt(np.fmax(m, 0.0))
    s[np.isnan(m)] = np.nan
    return s


def _first_nanmax(values: np.ndarray) -> int:
    """Index of MATLAB's ``max``: the first largest value, NaN ignored (the
    first element when all are NaN)."""
    if np.isnan(values).all():
        return 0
    return int(np.nanargmax(values))


def _parabola_offset(c: np.ndarray, rr: int, cc: int) -> tuple[float, float]:
    """xcorr2_nans' sub-pixel offset of the correlation maximum."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_r = np.fmin(
            1e6, (c[rr, cc] - c[rr - 1, cc]) / (c[rr, cc] - c[rr + 1, cc])
        )
        ratio_c = np.fmin(
            1e6, (c[rr, cc] - c[rr, cc - 1]) / (c[rr, cc] - c[rr, cc + 1])
        )
        d_r = (1 - ratio_r) / (1 + ratio_r) / 2
        d_c = (1 - ratio_c) / (1 + ratio_c) / 2
    return float(d_r), float(d_c)


def _shifted_template_values(
    template: np.ndarray, t_valid: np.ndarray, d: int
):
    """The template at ``t_valid`` moved by every shift of ``[-d, d]^2``.

    Yields ``(i, j, values)`` with the values in MATLAB's column-major order
    of ``template(circshift(tValid, -[dr dc]))``. While no shifted pixel
    leaves the image this is a lookup at shifted indices; otherwise the mask
    is rolled as MATLAB does (pixels wrap around, and their order with them).
    """
    h, w = template.shape
    cols, rows = np.nonzero(t_valid.T)  # column-major order
    shifts = range(-d, d + 1)
    wraps = rows.size > 0 and (
        rows.min() - d < 0
        or rows.max() + d >= h
        or cols.min() - d < 0
        or cols.max() + d >= w
    )
    if not wraps:
        flat = template.ravel()
        lin = rows * w + cols
        for i, dr in enumerate(shifts):
            for j, dc in enumerate(shifts):
                yield i, j, flat[lin - dr * w - dc]
        return
    for i, dr in enumerate(shifts):
        for j, dc in enumerate(shifts):
            moved = np.roll(t_valid, (-dr, -dc), axis=(0, 1))
            yield i, j, template.T[moved.T]


def xcorr2_nans(
    frame: np.ndarray, template: np.ndarray, shifts_center, d_shift: int
) -> tuple[np.ndarray, float]:
    """Local normalized cross-correlation of images with NaN.

    Port of ``xcorr2_nans.m``. The frame's pixels that are not NaN, and whose
    template pixels are not NaN for any shift of the search, are correlated
    with the template at every whole shift within ``d_shift`` of
    ``shifts_center``; the maximum is refined by a parabola on each axis
    unless it lies on the edge of the search.

    Parameters
    ----------
    frame, template : ndarray of shape (rows, cols)
        The image to align and the template (the same shape).
    shifts_center : array_like of 2 int
        Centre of the search ``(rows, cols)``.
    d_shift : int
        Largest shift from the centre on each axis.

    Returns
    -------
    motion : ndarray of shape (2,)
        The shift ``(rows, cols)``: frame pixel ``p`` matches template pixel
        ``p - motion``.
    r : float
        The correlation at the best whole shift.

    Raises
    ------
    ValueError
        If the shapes differ or the motion is NaN.
    """
    frame = np.asarray(frame, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)
    if frame.shape != template.shape:
        raise ValueError(
            f"frame is {frame.shape}, the template {template.shape}"
        )
    d = int(matlab_round(np.ravel(d_shift)[0]))
    sc = np.asarray(shifts_center, dtype=np.float64).ravel().astype(int)
    h, w = frame.shape
    near_nan = ndimage.binary_dilation(
        np.isnan(template), structure=np.ones((2 * d + 1,) * 2, bool)
    )
    f_valid = ~np.isnan(frame) & np.roll(~near_nan, sc, axis=(0, 1))
    f_valid[:d] = False
    f_valid[h - d :] = False
    f_valid[:, :d] = False
    f_valid[:, w - d :] = False
    t_valid = np.roll(f_valid, -sc, axis=(0, 1))

    f = frame.T[f_valid.T]  # column-major, as MATLAB's logical indexing
    ss_f = np.sqrt(np.sum(f**2))
    n = 2 * d + 1
    c = np.full((n, n), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i, j, t in _shifted_template_values(template, t_valid, d):
            c[i, j] = np.sum(f * t) / np.sqrt(np.sum(t**2))
        best = _first_nanmax(c.T.ravel())  # column-major, as max(C(:))
        rr, cc = best % n, best // n
        r = c[rr, cc] / ss_f
    motion = sc + np.array([rr - d, cc - d], dtype=np.float64)
    if 0 < rr < n - 1 and 0 < cc < n - 1:
        d_r, d_c = _parabola_offset(c, rr, cc)
        motion = motion - np.array([d_r, d_c])
    if np.isnan(motion).any():
        raise ValueError("xcorr2_nans: the computed motion is NaN")
    return motion, float(r)


def dftregistration_clipped(
    buf1ft: np.ndarray, buf2ft: np.ndarray, clip
) -> tuple[float, float, int, int]:
    """Whole-pixel registration by FFT cross-correlation, with a shift limit.

    Port of the ``usfac = 1`` branch of ``dftregistration_clipped.m`` (after
    Guizar-Sicairos et al., Opt. Lett. 33, 156-158, 2008), the only one SiLo
    uses. The cross-correlation is set to zero outside the window MATLAB
    keeps for ``clip`` (``fix(clip/2)`` pixels on either side of zero shift)
    before its maximum is located.

    Parameters
    ----------
    buf1ft, buf2ft : ndarray of complex, shape (rows, cols)
        Fourier transforms of the reference and of the image to register
        (zero frequency first).
    clip : float or pair of float
        MATLAB's ``clip`` for rows and columns.

    Returns
    -------
    error : float
        Translation-invariant normalized RMS error.
    diffphase : float
        Global phase difference.
    row_shift, col_shift : int
        Shift that registers the image to the reference.
    """
    clip = np.broadcast_to(np.asarray(clip, dtype=np.float64).ravel(), (2,))
    m, n = buf1ft.shape
    cc = np.fft.ifft2(buf1ft * np.conj(buf2ft))
    kr, kc = int(np.fix(clip[0] / 2)), int(np.fix(clip[1] / 2))
    keep = np.ones((m, n), bool)
    keep[kr + 1 : m - kr, :] = False
    keep[:, kc + 1 : n - kc] = False
    cc[~keep] = 0
    best = _first_nanmax(cc.real.T.ravel())  # the first maximum by column
    rloc, cloc = best % m, best // m
    cc_max = cc[rloc, cloc]
    rf_zero = np.sum(np.abs(buf1ft) ** 2) / (m * n)
    rg_zero = np.sum(np.abs(buf2ft) ** 2) / (m * n)
    with np.errstate(divide="ignore", invalid="ignore"):
        error = 1.0 - cc_max * np.conj(cc_max) / (rg_zero * rf_zero)
    error = float(np.sqrt(np.abs(error)))
    diffphase = float(np.arctan2(cc_max.imag, cc_max.real))
    # MATLAB's test, on its one-based index: "if rloc > fix(m/2)" the shift
    # is negative. On an odd axis this turns the largest positive shift,
    # (m - 1) / 2, into its alias -(m + 1) / 2 (the same circular shift);
    # kept as MATLAB returns it. With SiLo's clip of 80 the peak can only be
    # there on a canvas of 81 pixels or fewer.
    row_shift = rloc - m if rloc + 1 > m // 2 else rloc
    col_shift = cloc - n if cloc + 1 > n // 2 else cloc
    return error, diffphase, int(row_shift), int(col_shift)


def make_consensus_template(stack: np.ndarray, max_shift: int) -> np.ndarray:
    """A template that most of the given images agree with.

    Port of ``makeConsensusTemplate.m``. Every pair of images is registered
    (:func:`xcorr2_nans`, twice: around zero, then around the rounded
    result). The image with the highest median correlation to the others is
    the reference; the images that correlate with it at least as well as
    that median are moved onto it and averaged, NaN ignored.

    Parameters
    ----------
    stack : ndarray of shape (n, rows, cols)
        The images (NaN where not imaged).
    max_shift : int
        Largest shift searched between two images, and the canvas margin.

    Returns
    -------
    ndarray of shape (rows + 2 * max_shift, cols + 2 * max_shift)
        The template; image pixel ``(r, c)`` of the reference is at
        ``(r + max_shift, c + max_shift)``.
    """
    y = np.asarray(stack, dtype=np.float64)
    n, h, w = y.shape
    s = int(max_shift)
    corr = np.ones((n, n))
    motion = np.zeros((2, n, n))
    for f1 in range(n):
        for f2 in range(f1 + 1, n):
            m1, _ = xcorr2_nans(y[f2], y[f1], (0, 0), s)
            mot, corr[f1, f2] = xcorr2_nans(y[f2], y[f1], matlab_round(m1), s)
            motion[:, f1, f2] = mot
            motion[:, f2, f1] = -mot
            corr[f2, f1] = corr[f1, f2]
    template = np.full((h + 2 * s, w + 2 * s), np.nan)
    med = np.median(corr, axis=0)  # NaN where a pair had no overlap
    if np.isnan(med).all():
        return template
    ref = int(np.nanargmax(med))
    with np.errstate(invalid="ignore"):
        members = np.flatnonzero(corr[:, ref] >= med[ref])
    view_r = np.arange(h + 2 * s, dtype=np.float64)[:, None] - s
    view_c = np.arange(w + 2 * s, dtype=np.float64)[None, :] - s
    total = np.zeros(template.shape)
    count = np.zeros(template.shape)
    for f in members:
        plane = interp2_linear(
            y[f], view_r - motion[0, f, ref], view_c - motion[1, f, ref]
        )
        ok = ~np.isnan(plane)
        total[ok] += plane[ok]
        count += ok
    seen = count > 0
    template[seen] = total[seen] / count[seen]
    return template


@dataclass
class TrialWarp:
    """Every trial's map from the consensus frame to its registered image.

    MATLAB's ``warp`` structure (``SILo.m``, ``trialWarpCoords.m``), with
    zero-based coordinates.

    Attributes
    ----------
    affine : ndarray of shape (trials, 2, 3)
        ``[row_t, col_t] = affine[t] @ [row, col, 1]``; NaN for a trial that
        was not aligned. A rigid shift is ``[[1, 0, dr], [0, 1, dc]]``. For
        a patchwise field, the field's plane.
    nodes : ndarray of shape (n_nodes, 2)
        Patch centres ``(row, col)`` of the patchwise field; empty for rigid
        and affine alignment.
    sigma : float
        SD of the patchwise field's Gaussian blend (NaN without one).
    disp : ndarray of shape (trials, n_nodes, 2)
        Each trial's displacement ``(rows, cols)`` at the patch centres
        (MATLAB's ``warp.disp``, ``2 x nodes x trials``).
    patch : ndarray of int, shape (rows, cols), or None
        The patch of each consensus pixel the field was fitted on (1 to
        ``n_nodes``, 0 outside; MATLAB's ``warp.patch``). Not needed to
        apply the field.
    """

    affine: np.ndarray
    nodes: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    sigma: float = float("nan")
    disp: np.ndarray | None = None
    patch: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Normalize the arrays' types and shapes."""
        self.affine = np.asarray(self.affine, dtype=np.float64).reshape(
            -1, 2, 3
        )
        self.nodes = np.asarray(self.nodes, dtype=np.float64).reshape(-1, 2)
        if self.disp is None:
            self.disp = np.zeros(
                (self.affine.shape[0], self.nodes.shape[0], 2)
            )
        self.disp = np.asarray(self.disp, dtype=np.float64).reshape(
            self.affine.shape[0], self.nodes.shape[0], 2
        )
        self.sigma = float(self.sigma)
        if self.patch is not None:
            self.patch = np.asarray(self.patch, dtype=np.int32)

    @classmethod
    def from_shifts(cls, shifts: np.ndarray) -> "TrialWarp":
        """The rigid warp of per-trial shifts ``(trials, 2)`` (NaN rows for
        trials that were not aligned)."""
        shifts = np.asarray(shifts, dtype=np.float64).reshape(-1, 2)
        affine = np.full((shifts.shape[0], 2, 3), np.nan)
        ok = ~np.isnan(shifts[:, 0])
        affine[ok, :, :2] = np.eye(2)
        affine[ok, :, 2] = shifts[ok]
        return cls(affine)

    @classmethod
    def from_matlab(
        cls,
        affine: np.ndarray,
        nodes: np.ndarray | None = None,
        sigma: float = float("nan"),
        disp: np.ndarray | None = None,
        patch: np.ndarray | None = None,
    ) -> "TrialWarp":
        """From MATLAB's ``warp`` (one-based coordinates).

        Parameters
        ----------
        affine : ndarray of shape (trials, 2, 3)
            ``warp.affine`` with the trials first
            (``permute(warp.affine, [3 1 2])``).
        nodes : ndarray of shape (n_nodes, 2), optional
            ``warp.nodes``.
        sigma : float
            ``warp.sigma``.
        disp : ndarray of shape (trials, n_nodes, 2), optional
            ``warp.disp`` as ``permute(warp.disp, [3 2 1])``.
        patch : ndarray of shape (rows, cols), optional
            ``warp.patch``.
        """
        a = np.array(affine, dtype=np.float64).reshape(-1, 2, 3)
        a[:, :, 2] += a[:, :, 0] + a[:, :, 1] - 1
        n = np.zeros((0, 2)) if nodes is None else np.asarray(nodes) - 1.0
        return cls(a, n, sigma, disp, patch)

    def to_matlab(self) -> np.ndarray:
        """MATLAB's ``warp.affine`` as ``(trials, 2, 3)`` (one-based)."""
        a = self.affine.copy()
        a[:, :, 2] -= a[:, :, 0] + a[:, :, 1] - 1
        return a

    @property
    def n_trials(self) -> int:
        """Number of trials."""
        return self.affine.shape[0]

    @property
    def is_patchwise(self) -> bool:
        """Whether the maps are patchwise fields."""
        return self.nodes.shape[0] > 0

    @property
    def aligned(self) -> np.ndarray:
        """Which trials have a map (bool per trial)."""
        return ~np.isnan(self.affine[:, 0, 2])

    def is_shift(self, trial: int) -> bool:
        """Whether the trial's map is a shift (no affine or patchwise
        part), which extraction can add to the trial's motion."""
        return self.nodes.shape[0] == 0 and bool(
            np.array_equal(self.affine[trial, :, :2], np.eye(2))
        )

    def shift(self, trial: int) -> np.ndarray:
        """The trial's shift ``(rows, cols)``; only for :meth:`is_shift`."""
        if not self.is_shift(trial):
            raise ValueError(f"the map of trial {trial} is not a shift")
        return self.affine[trial, :, 2].copy()

    def coords(self, trial: int, rows, cols) -> tuple[np.ndarray, np.ndarray]:
        """Positions in the trial's registered image of consensus points.

        Port of ``trialWarpCoords.m`` (zero-based): through the patchwise
        field ``x + u_t(x)`` (evaluated in blocks of 50000 points), or the
        affine map.

        Parameters
        ----------
        trial : int
            Zero-based trial.
        rows, cols : array_like
            Consensus-frame coordinates (equal shapes).

        Returns
        -------
        rows_t, cols_t : ndarray
            The points' coordinates in the trial's registered image.
        """
        r = np.asarray(rows, dtype=np.float64)
        c = np.asarray(cols, dtype=np.float64)
        if self.nodes.shape[0]:
            shape = np.broadcast_shapes(r.shape, c.shape)
            r = np.broadcast_to(r, shape).ravel()
            c = np.broadcast_to(c, shape).ravel()
            d = self.disp[trial]
            r_t = r.copy()
            c_t = c.copy()
            chunk = 50000  # points per block, to bound memory for images
            for i0 in range(0, r.size, chunk):
                ix = slice(i0, i0 + chunk)
                u = patch_field_apply(self.nodes, self.sigma, r[ix], c[ix], d)
                r_t[ix] = r[ix] + u[:, 0]
                c_t[ix] = c[ix] + u[:, 1]
            return r_t.reshape(shape), c_t.reshape(shape)
        a = self.affine[trial]
        if self.is_shift(trial):  # exact, as MATLAB's 1 * r + 0 * c + shift
            return r + a[0, 2], c + a[1, 2]
        return (
            a[0, 0] * r + a[0, 1] * c + a[0, 2],
            a[1, 0] * r + a[1, 1] * c + a[1, 2],
        )

    def inverse(
        self, trial: int, rows_t, cols_t
    ) -> tuple[np.ndarray, np.ndarray]:
        """Consensus points that a trial's image points are read from.

        Port of ``trialWarpInverse.m`` (zero-based): the inverse of the
        affine part, then, for a patchwise field, ``x <- x + (y - map(x))``
        until the residual is below 1e-6 px (at most 20 times; it converges
        because the field bends by far less than a pixel per pixel).

        Parameters
        ----------
        trial : int
            Zero-based trial.
        rows_t, cols_t : array_like
            Points in the trial's registered image (equal shapes).

        Returns
        -------
        rows, cols : ndarray
            Their consensus-frame coordinates.
        """
        rt = np.asarray(rows_t, dtype=np.float64)
        ct = np.asarray(cols_t, dtype=np.float64)
        shape = np.broadcast_shapes(rt.shape, ct.shape)
        y = np.vstack(
            [
                np.broadcast_to(rt, shape).ravel(),
                np.broadcast_to(ct, shape).ravel(),
            ]
        )
        a = self.affine[trial]
        x = np.linalg.solve(a[:, :2], y - a[:, 2:3])
        if self.nodes.shape[0]:
            for _ in range(20):
                r_m, c_m = self.coords(trial, x[0], x[1])
                e = y - np.vstack([r_m, c_m])
                x = x + e
                if np.max(np.abs(e), initial=0.0) < 1e-6:
                    break
        return x[0].reshape(shape), x[1].reshape(shape)

    def grid_coords(
        self, trial: int, shape: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """:meth:`coords` of every pixel of a ``(rows, cols)`` grid."""
        rr = np.arange(shape[0], dtype=np.float64)[:, None]
        cc = np.arange(shape[1], dtype=np.float64)[None, :]
        return self.coords(trial, *np.broadcast_arrays(rr, cc))


def warp_trial_image(
    im: np.ndarray,
    warp: TrialWarp,
    trial: int,
    method: str = "linear",
    coords: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Resample a trial image into the consensus frame.

    Port of ``warpTrialImage.m`` / ``warpImageAffine.m``:
    ``out[r, c] = im[warp.coords(trial, r, c)]`` on the grid of ``im``, NaN
    outside the image or next to a NaN.

    Parameters
    ----------
    im : ndarray of shape (rows, cols) or (rows, cols, channels)
        The trial's image.
    warp : TrialWarp
        The across-trial alignment.
    trial : int
        Zero-based trial.
    method : str
        ``"linear"``, or ``"cubic"`` (falling back to linear where cubic
        would reach a NaN, so it removes no more pixels than linear).
    coords : (ndarray, ndarray), optional
        ``warp.grid_coords(trial, im.shape[:2])``, when already computed
        (a patchwise field costs a blend over the patch centres per pixel).

    Returns
    -------
    ndarray
        The resampled image, in the shape of ``im``.
    """
    if method not in ("linear", "cubic"):
        raise ValueError(
            f"interpolation {method!r} is not 'linear' or 'cubic'"
        )
    im = np.asarray(im, dtype=np.float64)
    r_t, c_t = (
        warp.grid_coords(trial, im.shape[:2]) if coords is None else coords
    )

    def one(plane):
        """One channel."""
        lin = interp2_linear(plane, r_t, c_t)
        if method == "linear":
            return lin
        cub = interp2_cubic(plane, r_t, c_t)
        return np.where(np.isnan(cub), lin, cub)

    if im.ndim == 2:
        return one(im)
    return np.stack([one(im[:, :, k]) for k in range(im.shape[2])], axis=2)


def warp_image_affine(
    im: np.ndarray, a: np.ndarray, method: str = "linear"
) -> np.ndarray:
    """Resample an image through one affine map (``warpImageAffine.m``):
    ``out[x] = im[a @ [x, 1]]``, zero-based, NaN outside or next to NaN."""
    return warp_trial_image(im, TrialWarp(np.asarray(a)[None]), 0, method)


def mask_to_consensus(
    mask: np.ndarray,
    warp: TrialWarp,
    trial: int | None,
    shape: tuple[int, int],
) -> np.ndarray:
    """Move a mask drawn on a trial's image into the consensus frame.

    Port of what ``roisToConsensusFrame.m`` and ``shiftRoiData.m`` do to an
    ROI's mask. For a shift the mask moves by minus the trial's shift,
    rounded; for another map each consensus pixel takes the mask's value at
    the nearest pixel of its position in the trial. Pixels that leave the
    frame are dropped. A mask of a trial that is unknown (None) or was not
    aligned is only cropped or padded to the frame.

    Parameters
    ----------
    mask : ndarray of bool, shape (rows, cols)
        The mask in the trial's registered image.
    warp : TrialWarp
        The across-trial alignment.
    trial : int or None
        Zero-based trial the mask was drawn on.
    shape : tuple of int
        ``(rows, cols)`` of the consensus frame.

    Returns
    -------
    ndarray of bool, shape ``shape``
    """
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros(shape, bool)
    if trial is None or not warp.aligned[trial]:
        h, w = min(shape[0], mask.shape[0]), min(shape[1], mask.shape[1])
        out[:h, :w] = mask[:h, :w]
        return out
    if warp.is_shift(trial):
        move = matlab_round(-warp.shift(trial)).astype(np.int64)
        rr, cc = np.nonzero(mask)
        rr, cc = rr + move[0], cc + move[1]
        ok = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
        out[rr[ok], cc[ok]] = True
        return out
    rr = np.arange(shape[0], dtype=np.float64)[:, None]
    cc = np.arange(shape[1], dtype=np.float64)[None, :]
    r_t, c_t = warp.coords(trial, *np.broadcast_arrays(rr, cc))
    # MATLAB rounds its one-based (positive) coordinates: halves go up
    r_t = np.floor(r_t + 0.5).astype(np.int64)
    c_t = np.floor(c_t + 0.5).astype(np.int64)
    ok = (
        (r_t >= 0) & (r_t < mask.shape[0]) & (c_t >= 0) & (c_t < mask.shape[1])
    )
    out[ok] = mask[r_t[ok], c_t[ok]]
    return out


@dataclass
class RigidAlignment:
    """Result of :func:`align_trials`.

    Attributes
    ----------
    template : ndarray of shape (rows + 2 * max_shift, cols + 2 * max_shift)
        The consensus template.
    samples : ndarray of int
        Zero-based trials the template was made from.
    shifts : ndarray of shape (trials, 2)
        Each trial's shift ``(rows, cols)`` (MATLAB's ``motOutput``); NaN
        for trials that were not aligned.
    corr : ndarray of shape (trials,)
        Correlation of each trial's mean image with the template.
    """

    template: np.ndarray
    samples: np.ndarray
    shifts: np.ndarray
    corr: np.ndarray


def template_samples(keep: np.ndarray, n_samples: int = 20) -> np.ndarray:
    """The kept trials the template is made from: up to ``n_samples``,
    evenly spaced (``round(linspace(1, n, n_samples))``, unique)."""
    kept = np.flatnonzero(keep)
    pick = matlab_round(np.linspace(1, kept.size, n_samples)).astype(int)
    return kept[np.unique(pick) - 1]


def align_trials(
    mean_sum: np.ndarray,
    act: np.ndarray,
    keep: np.ndarray,
    max_shift: int = 5,
    clip: float = 80.0,
    n_samples: int = 20,
) -> RigidAlignment:
    """Rigid alignment of the trials of one imaging path (as ``SILo.m``).

    Parameters
    ----------
    mean_sum : ndarray of shape (trials, rows, cols)
        The trials' mean images summed over channels, on one canvas (NaN
        where not imaged).
    act : ndarray of shape (trials, rows, cols)
        The trials' activity images; a trial whose activity image is all
        NaN is not aligned.
    keep : ndarray of bool, shape (trials,)
        Trials to use (those that passed file verification).
    max_shift : int
        Search range of the refinement, and the template's margin.
    clip : float
        ``clip`` of the whole-pixel estimate
        (:func:`dftregistration_clipped`).
    n_samples : int
        Largest number of trials the template is made from. Kept trials
        whose mean image is all NaN are not among them (unlike MATLAB, where
        one such trial leaves the whole path without a template).

    Returns
    -------
    RigidAlignment

    Raises
    ------
    ValueError
        If no trial is kept, or no kept trial has a mean image.
    """
    m = np.asarray(mean_sum, dtype=np.float64)
    keep = np.asarray(keep, dtype=bool).ravel()
    n_trials, h, w = m.shape
    if not keep.any():
        raise ValueError("no trials to align")
    # a kept trial without a mean image (all NaN) has no correlation with
    # any other, which would leave the path without a template (MATLAB
    # samples such a trial and ends with no sources); it is not sampled
    has_image = keep & ~np.isnan(m).all(axis=(1, 2))
    if not has_image.any():
        raise ValueError("no kept trial has a mean image to align")
    s = int(max_shift)
    samples = template_samples(has_image, n_samples)
    template = make_consensus_template(m[samples], s)
    shifts = np.full((n_trials, 2), np.nan)
    corr = np.full(n_trials, np.nan)
    with np.errstate(invalid="ignore"):
        fill = (
            np.nanmin(template) - 1 if not np.isnan(template).all() else np.nan
        )
    t_fft = np.fft.fft2(np.fmax(template, fill))
    padded = np.full(template.shape, np.nan)
    for t in range(n_trials):
        if not keep[t] or np.isnan(act[t]).all():
            continue
        padded[:] = np.nan
        padded[s : s + h, s : s + w] = m[t]
        out = dftregistration_clipped(
            t_fft, np.fft.fft2(np.fmax(padded, fill)), clip
        )
        shifts[t], corr[t] = xcorr2_nans(
            padded, template, (-out[2], -out[3]), s
        )
    return RigidAlignment(template, samples, shifts, corr)


# ------------------------------------------------------- affine refinement


@dataclass
class AlignmentInfo:
    """Diagnostics of :func:`align_trials_affine` and
    :func:`align_trials_patchwise`.

    Attributes
    ----------
    corr : ndarray of shape (trials,)
        Each trial's correlation with the final template (NaN for trials
        not fitted).
    rms_disp : ndarray of shape (trials,)
        RMS change of each trial's map from the starting maps over the
        template's pixels (the patches' pixels for the patchwise field).
    n_px, noise_sd, bend_sd, weight : ndarray or None
        Patchwise only, per patch: its pixels, the SD of its centre's
        departure from the plane across the halves (noise) and across the
        trials (bend), both axes pooled, and the reliability weight.
    """

    corr: np.ndarray
    rms_disp: np.ndarray
    n_px: np.ndarray | None = None
    noise_sd: np.ndarray | None = None
    bend_sd: np.ndarray | None = None
    weight: np.ndarray | None = None


def _fitted_trials(tforms: np.ndarray) -> np.ndarray:
    """Trials whose starting map has no NaN."""
    return np.flatnonzero(~np.isnan(tforms).any(axis=(1, 2)))


def _nanmean0(stack: np.ndarray) -> np.ndarray:
    """Mean over axis 0 with NaN ignored (NaN where nothing is left)."""
    ok = ~np.isnan(stack)
    count = ok.sum(axis=0)
    total = np.where(ok, stack, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def _affine_template(s, tforms, trials):
    """Mean of the trials warped into the common frame (linear); NaN where
    fewer than half of the trials are seen."""
    stack = np.stack(
        [warp_image_affine(s[t], tforms[t], "linear") for t in trials]
    )
    template = _nanmean0(stack)
    template[(~np.isnan(stack)).sum(axis=0) < len(trials) / 2] = np.nan
    return template


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation (NaN for fewer than two points)."""
    if a.size < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _refine_affine(s, template, a):
    """Forward-additive Gauss-Newton: minimize, over the template's pixels,
    ``sum (S(A x) - g T(x) - b)^2`` over the map ``A``, gain ``g`` and offset
    ``b`` (``refineAffine`` in ``alignTrialsAffine.m``)."""
    max_iter, tol = 50, 1e-3  # tol: pixels, largest displacement change
    r, c = _colmajor_nonzero(~np.isnan(template))
    if r.size < 500:
        return a
    t_val = template[r, c]
    r = r.astype(np.float64)
    c = c.astype(np.float64)
    r0, c0, sc = r.mean(), c.mean(), 100.0  # centred, scaled coordinates
    u, v = (r - r0) / sc, (c - c0) / sc
    corners = np.array(
        [
            [r.min(), r.max(), r.min(), r.max()],
            [c.min(), c.min(), c.max(), c.max()],
        ]
    )
    g_r, g_c = matlab_gradient(s)
    cubic = CubicInterpolant(s)
    a = a.copy()
    for _ in range(max_iter):
        r_t = a[0, 0] * r + a[0, 1] * c + a[0, 2]
        c_t = a[1, 0] * r + a[1, 1] * c + a[1, 2]
        wt = cubic(r_t, c_t)
        gr = interp2_linear(g_r, r_t, c_t)
        gc = interp2_linear(g_c, r_t, c_t)
        ok = ~np.isnan(wt) & ~np.isnan(gr) & ~np.isnan(gc)
        if ok.sum() < 500:
            return a
        gr, gc, uo, vo = gr[ok], gc[ok], u[ok], v[ok]
        jac = np.column_stack([gr * uo, gr * vo, gr, gc * uo, gc * vo, gc])
        design = np.column_stack([t_val[ok], np.ones(int(ok.sum())), -jac])
        beta = np.linalg.lstsq(design, wt[ok], rcond=None)[0]
        dp = beta[2:8]
        d_lin = np.array([[dp[0], dp[1]], [dp[3], dp[4]]]) / sc
        d_t = np.array([dp[2], dp[5]]) - d_lin @ np.array([r0, c0])
        # limit the step to 1 px anywhere on the template
        step = np.max(
            np.sqrt(((d_lin @ corners + d_t[:, None]) ** 2).sum(axis=0))
        )
        if step > 1:
            d_lin, d_t = d_lin / step, d_t / step
        a[:, :2] += d_lin
        a[:, 2] += d_t
        if step < tol:
            break
    return a


def _recentre(tforms, trials, x0):
    """Right-multiply every map so that the trials' mean linear part is the
    identity and the mean mapping of ``x0`` is unchanged."""
    a_bar = tforms[trials].mean(axis=0)
    l_inv = np.linalg.inv(a_bar[:, :2])
    b = np.eye(3)
    b[:2, :2] = l_inv
    b[:2, 2] = x0 - l_inv @ x0
    out = tforms.copy()
    for t in trials:
        out[t] = (np.vstack([tforms[t], [0.0, 0.0, 1.0]]) @ b)[:2]
    return out


def align_trials_affine(
    mean_sum: np.ndarray, tforms: np.ndarray, n_rounds: int = 3
) -> tuple[np.ndarray, AlignmentInfo]:
    """Refine the trials' maps to affine maps, groupwise.

    Port of ``alignTrialsAffine.m``. The template is the mean of the trials
    warped into the common frame; each trial's affine map is refit to it by
    Gauss-Newton (with a gain and offset, on square-root images, NaN
    ignored) and the template is rebuilt, for ``n_rounds`` rounds. Each
    round the trials' mean linear part is removed, so the common frame keeps
    the trials' average geometry, with the image centre where it was.

    Parameters
    ----------
    mean_sum : ndarray of shape (trials, rows, cols)
        One mean image per trial (NaN where unscanned).
    tforms : ndarray of shape (trials, 2, 3)
        Starting maps (e.g. the rigid shifts), zero-based; trials whose map
        has a NaN are left as they are.
    n_rounds : int
        Rounds of template and refit.

    Returns
    -------
    tforms : ndarray of shape (trials, 2, 3)
        The refined maps.
    info : AlignmentInfo
    """
    m = np.asarray(mean_sum, dtype=np.float64)
    n_trials, h, w = m.shape
    tforms = np.array(tforms, dtype=np.float64).reshape(n_trials, 2, 3)
    tforms0 = tforms.copy()
    trials = _fitted_trials(tforms)
    s = _sqrt_nan(m)
    x0 = np.array([(h - 1) / 2, (w - 1) / 2])  # MATLAB's (H+1)/2, one-based
    info = AlignmentInfo(np.full(n_trials, np.nan), np.full(n_trials, np.nan))
    if trials.size < 2:
        return tforms, info
    for _ in range(n_rounds):
        template = _affine_template(s, tforms, trials)
        for t in trials:
            tforms[t] = _refine_affine(s[t], template, tforms[t])
        tforms = _recentre(tforms, trials, x0)
    template = _affine_template(s, tforms, trials)
    sel = ~np.isnan(template)
    r, c = _colmajor_nonzero(sel)
    for t in trials:
        wim = warp_image_affine(s[t], tforms[t], "linear")
        v = sel & ~np.isnan(wim)
        info.corr[t] = _corr(wim[v], template[v])
        d = tforms[t] - tforms0[t]
        dr = d[0, 0] * r + d[0, 1] * c + d[0, 2]
        dc = d[1, 0] * r + d[1, 1] * c + d[1, 2]
        info.rms_disp[t] = np.sqrt(np.mean(dr**2 + dc**2))
    return tforms, info


# --------------------------------------------------------- patchwise field


def _plane_basis(nodes: np.ndarray, rows: np.ndarray, cols: np.ndarray):
    """The plane's design at the centres and at the points (centred, scaled
    coordinates; the mean with fewer than 3 centres)."""
    k = nodes.shape[0]
    if k >= 3:
        mu = nodes.mean(axis=0)
        sc = np.max(np.std(nodes, axis=0, ddof=1)) + _EPS
        p = np.column_stack([(nodes - mu) / sc, np.ones(k)])
        pq = np.column_stack(
            [(rows - mu[0]) / sc, (cols - mu[1]) / sc, np.ones(rows.size)]
        )
    else:
        p = np.ones((k, 1))
        pq = np.ones((rows.size, 1))
    return p, pq


def _pinv(a: np.ndarray) -> np.ndarray:
    """MATLAB's ``pinv`` (singular values below ``max(size) * eps(norm)``
    dropped)."""
    return np.linalg.pinv(a, rcond=max(a.shape) * _EPS)


def _blend(nodes, sigma, rows, cols) -> np.ndarray:
    """Gaussian weights of the centres at the points, relative to the
    nearest centre (so far points do not underflow), summing to 1."""
    d2 = (rows[:, None] - nodes[None, :, 0]) ** 2 + (
        cols[:, None] - nodes[None, :, 1]
    ) ** 2
    phi = np.exp(-(d2 - d2.min(axis=1, keepdims=True)) / (2 * sigma**2))
    return phi / phi.sum(axis=1, keepdims=True)


def patch_field_matrix(
    nodes: np.ndarray, sigma: float, rows, cols
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Displacements at points from displacements at patch centres.

    Port of ``patchFieldMatrix.m``: ``u(rows, cols) = L @ d`` for the values
    ``d`` (``(n_nodes,)`` or ``(n_nodes, 2)``) at the centres: the plane
    fitted by least squares through the centre values, plus a blend of each
    centre's departure from it with Gaussian weights of SD ``sigma``,
    normalized to sum to 1. With fewer than 3 centres the plane is their
    mean.

    Returns
    -------
    L : ndarray of shape (points, n_nodes)
    E : ndarray of shape (n_nodes, n_nodes)
        The departures from the plane, ``E @ d``.
    L_plane : ndarray of shape (points, n_nodes)
        The plane alone.
    """
    nodes = np.asarray(nodes, dtype=np.float64).reshape(-1, 2)
    r = np.asarray(rows, dtype=np.float64).ravel()
    c = np.asarray(cols, dtype=np.float64).ravel()
    p, pq = _plane_basis(nodes, r, c)
    hp = _pinv(p)
    e = np.eye(nodes.shape[0]) - p @ hp
    l_plane = pq @ hp
    return l_plane + _blend(nodes, sigma, r, c) @ e, e, l_plane


def patch_field_apply(
    nodes: np.ndarray, sigma: float, rows, cols, d: np.ndarray
) -> np.ndarray:
    """``patch_field_matrix(nodes, sigma, rows, cols)[0] @ d`` without the
    matrix (``patchFieldApply.m``): the plane's coefficients and the
    centres' departures are computed from ``d`` once."""
    nodes = np.asarray(nodes, dtype=np.float64).reshape(-1, 2)
    r = np.asarray(rows, dtype=np.float64).ravel()
    c = np.asarray(cols, dtype=np.float64).ravel()
    d = np.asarray(d, dtype=np.float64)
    p, pq = _plane_basis(nodes, r, c)
    coef = _pinv(p) @ d
    dep = d - p @ coef
    return pq @ coef + _blend(nodes, sigma, r, c) @ dep


def kmeans(
    x: np.ndarray,
    k: int,
    rng: np.random.Generator | None = None,
    replicates: int = 1,
    start: np.ndarray | None = None,
    max_iter: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """k-means with MATLAB ``kmeans``'s defaults (squared Euclidean,
    k-means++ starts, batch updates until no point moves, an empty cluster
    restarted at the point farthest from its centroid).

    Parameters
    ----------
    x : ndarray of shape (points, dims)
    k : int
        Clusters.
    rng : numpy Generator, optional
        For the k-means++ starts (needed without ``start``).
    replicates : int
        Runs from new starts; the one with the smallest total distance is
        kept.
    start : ndarray of shape (k, dims), optional
        Starting centroids (one run).
    max_iter : int
        Largest number of batch iterations.

    Returns
    -------
    labels : ndarray of int, shape (points,)
        Zero-based cluster of each point.
    centroids : ndarray of shape (k, dims)
    """
    x = np.asarray(x, dtype=np.float64)
    best = None
    runs = 1 if start is not None else replicates
    for _ in range(runs):
        if start is not None:
            cen = np.array(start, dtype=np.float64).reshape(k, -1)
        else:
            cen = _kmeans_plus_plus(x, k, rng)
        labels = np.full(x.shape[0], -1)
        for _ in range(max_iter):
            dist, new = cKDTree(cen).query(x)
            counts = np.bincount(new, minlength=k)
            for j in np.flatnonzero(counts == 0):
                # MATLAB's 'singleton': the point farthest from its
                # centroid, taken from a cluster that keeps a member
                movable = counts[new] > 1
                far = int(np.argmax(np.where(movable, dist, -1.0)))
                counts[new[far]] -= 1
                counts[j] = 1
                new[far] = j
                dist[far] = 0.0
                cen[j] = x[far]
            if np.array_equal(new, labels):
                break
            labels = new
            counts = np.bincount(labels, minlength=k).astype(np.float64)
            for dim in range(x.shape[1]):
                cen[:, dim] = (
                    np.bincount(labels, weights=x[:, dim], minlength=k)
                    / counts
                )
        total = float(((x - cen[labels]) ** 2).sum())
        if best is None or total < best[0]:
            best = (total, labels.copy(), cen.copy())
    return best[1], best[2]


def _kmeans_plus_plus(x, k, rng) -> np.ndarray:
    """k-means++ starting centroids."""
    n = x.shape[0]
    cen = np.empty((k, x.shape[1]))
    cen[0] = x[rng.integers(n)]
    d2 = ((x - cen[0]) ** 2).sum(axis=1)
    for j in range(1, k):
        total = d2.sum()
        p = d2 / total if total > 0 else np.full(n, 1.0 / n)
        cen[j] = x[rng.choice(n, p=p)]
        d2 = np.minimum(d2, ((x - cen[j]) ** 2).sum(axis=1))
    return cen


def make_patches(
    valid: np.ndarray, patch_px: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Patches of about ``patch_px`` valid pixels (``makePatches`` in
    ``alignTrialsPatchwise.m``).

    k-means on the valid pixels' coordinates (3 replicates); while a cluster
    has fewer than ``patch_px / 2`` pixels, the smallest is dropped and
    k-means rerun from the remaining centroids.

    Returns
    -------
    patch : ndarray of int32, shape valid.shape
        1 to K on the valid pixels, 0 elsewhere.
    nodes : ndarray of shape (K, 2)
        The patches' centroids ``(row, col)``, zero-based.
    """
    rows, cols = _colmajor_nonzero(valid)
    x = np.column_stack([rows, cols]).astype(np.float64)
    if not patch_px >= 1:
        raise ValueError(f"patch_px must be at least 1, not {patch_px!r}")
    # at most one patch per pixel
    k = min(x.shape[0], max(1, int(matlab_round(x.shape[0] / patch_px))))
    rng = np.random.default_rng(seed)
    labels, nodes = kmeans(x, k, rng, replicates=3)
    n = np.bincount(labels, minlength=k)
    while k > 1 and n.min() < patch_px / 2:
        nodes = np.delete(nodes, int(np.argmin(n)), axis=0)
        k -= 1
        labels, nodes = kmeans(x, k, start=nodes)
        n = np.bincount(labels, minlength=k)
    patch = np.zeros(np.shape(valid), np.int32)
    patch[rows, cols] = labels + 1
    return patch, nodes


@dataclass
class PatchwiseFit:
    """Result of :func:`align_trials_patchwise` (MATLAB's ``warp``).

    Attributes
    ----------
    nodes : ndarray of shape (K, 2)
        Patch centres, zero-based ``(row, col)``.
    sigma : float
        SD of the blend (half the median distance to the nearest centre).
    disp : ndarray of shape (trials, K, 2)
        The displacements at the centres (NaN for trials not fitted).
    patch : ndarray of int32, shape (rows, cols)
        Patch of each pixel (0 outside).
    affine : ndarray of shape (trials, 2, 3)
        The field's plane per trial, as a map like the starting maps.
    """

    nodes: np.ndarray
    sigma: float
    disp: np.ndarray
    patch: np.ndarray
    affine: np.ndarray


def _patch_template(s, d, lmat, r, c, trials):
    """The trials warped into the common frame at the fit pixels, averaged;
    NaN where fewer than half are seen."""
    v = np.empty((r.size, len(trials)))
    for k, t in enumerate(trials):
        u = lmat @ d[t].T
        v[:, k] = interp2_linear(s[t], r + u[:, 0], c + u[:, 1])
    ok = ~np.isnan(v)
    count = ok.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        template = np.where(ok, v, 0.0).sum(axis=1) / count
    template[count < len(trials) / 2] = np.nan
    return template


def _group_median(values, groups, k) -> np.ndarray:
    """Median of ``values`` per group 0..k-1 (0 for an empty group, as
    MATLAB's ``accumarray`` with ``@median``)."""
    out = np.zeros(k)
    if values.size == 0:
        return out
    order = np.lexsort((values, groups))
    v, g = values[order], groups[order]
    counts = np.bincount(g, minlength=k)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    has = counts > 0
    lo = starts + (counts - 1) // 2
    hi = starts + counts // 2
    out[has] = (v[lo[has]] + v[hi[has]]) / 2
    return out


def _interp2_linear_pair(ims, rows, cols):
    """:func:`interp2_linear` of two images at the same points (the weights
    and indices computed once)."""
    h, w = ims[0].shape
    rows = np.asarray(rows, dtype=np.float64)
    cols = np.asarray(cols, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        inside = (rows >= 0) & (rows <= h - 1) & (cols >= 0) & (cols <= w - 1)
    r = np.where(inside, rows, 0.0)
    c = np.where(inside, cols, 0.0)
    r0 = np.floor(r).astype(np.intp)
    c0 = np.floor(c).astype(np.intp)
    fr, fc = r - r0, c - c0
    r1 = np.minimum(r0 + 1, h - 1)
    c1 = np.minimum(c0 + 1, w - 1)
    outs = [np.zeros(rows.shape) for _ in ims]
    for wgt, rr, cc in (
        ((1 - fr) * (1 - fc), r0, c0),
        (fr * (1 - fc), r1, c0),
        ((1 - fr) * fc, r0, c1),
        (fr * fc, r1, c1),
    ):
        used = wgt != 0
        wu, lin = wgt[used], rr[used] * w + cc[used]
        for out, im in zip(outs, ims):
            out[used] += wu * im.ravel()[lin]
    for out in outs:
        out[~inside] = np.nan
    return outs


def _refine_field(s, template, d, lmat, r, c, pix, e_mat, lambda_rel):
    """Robust Gauss-Newton for all centre displacements of one trial
    (``refineField`` in ``alignTrialsPatchwise.m``): minimize the bisquare
    loss of ``S(x + u(x)) - g_k T(x) - b_k`` over the fit pixels (``k`` their
    patch), ``u = L @ d``, plus ``lambda |E d|^2``.

    ``d`` is ``(2, K)`` (MATLAB's layout); returns the refined ``d``. The
    normal equations are MATLAB's, of the design ``[Xs, Xd]`` (a gain and
    offset per patch, then the centres' row and column updates), with
    ``sqrt(W) Xd = -sqrt(W) [GR .* L, GC .* L]`` formed once per iteration
    (its Gram matrix as a symmetric product) and ``Xs`` sparse.
    """
    max_iter, tol = 30, 1e-3  # tol: pixels, largest centre update
    k = lmat.shape[1]
    n = r.size
    g_r, g_c = matlab_gradient(s)
    cubic = CubicInterpolant(s)
    cat_rows = np.concatenate([pix, k + pix])
    cat_cols = np.concatenate([np.arange(n), np.arange(n)])
    ee = e_mat.T @ e_mat
    zero = np.zeros((k, k))
    penalty = np.block([[ee, zero], [zero, ee]])
    w = np.ones(n)
    grad = np.empty((n, 2 * k))
    d = d.copy()
    for _ in range(max_iter):
        u = lmat @ d.T
        rq, cq = r + u[:, 0], c + u[:, 1]
        wt = cubic(rq, cq)
        gr, gc = _interp2_linear_pair((g_r, g_c), rq, cq)
        ok = ~np.isnan(template) & ~np.isnan(wt) & ~np.isnan(gr)
        ok &= ~np.isnan(gc)
        if ok.sum() < 200:
            return d
        tz = np.where(ok, template, 0.0)
        y = np.where(ok, wt, 0.0)
        gr = np.where(ok, gr, 0.0)
        gc = np.where(ok, gc, 0.0)
        wk = w * ok
        # gains and offsets per patch: diagonal blocks
        a_tt = np.diag(np.bincount(pix, wk * tz * tz, k))
        a_t1 = np.diag(np.bincount(pix, wk * tz, k))
        a_11 = np.diag(np.bincount(pix, wk, k))
        a11 = np.block([[a_tt, a_t1], [a_t1, a_11]])
        # grad = -Xd; the signs cancel in Xd' W Xd (formed as a symmetric
        # product of sqrt(W) grad)
        sw = np.sqrt(wk)
        np.multiply((sw * gr)[:, None], lmat, out=grad[:, :k])
        np.multiply((sw * gc)[:, None], lmat, out=grad[:, k:])
        a22 = grad.T @ grad
        # Xs' W Xd: the gain and offset columns have one pixel each per row
        xs = sparse.csr_matrix(
            (np.concatenate([sw * tz, sw]), (cat_rows, cat_cols)),
            shape=(2 * k, n),
        )
        a12 = -(xs @ grad)
        b1 = np.concatenate(
            [np.bincount(pix, wk * tz * y, k), np.bincount(pix, wk * y, k)]
        )
        b2 = -(grad.T @ (sw * y))
        lam = lambda_rel * np.median(np.diag(a22))
        a22 = a22 + lam * penalty
        b2 = b2 - lam * np.concatenate([ee @ d[0], ee @ d[1]])
        a = np.block([[a11, a12], [a12.T, a22]])
        # patches with no usable pixels in this trial
        a = a + (1e-10 * np.mean(np.diag(a)) + _EPS) * np.eye(4 * k)
        beta = np.linalg.solve(a, np.concatenate([b1, b2]))
        lb = lmat @ np.column_stack([beta[2 * k : 3 * k], beta[3 * k :]])
        err = (
            y
            - (tz * beta[pix] + beta[k + pix])
            + gr * lb[:, 0]
            + gc * lb[:, 1]
        )
        sc = 1.4826 * _group_median(np.abs(err[ok]), pix[ok], k) + _EPS
        uu = err[ok] / (4.685 * sc[pix[ok]])
        w[ok] = (1 - uu**2) ** 2 * (np.abs(uu) < 1)
        dd = np.vstack([beta[2 * k : 3 * k], beta[3 * k :]])
        step = np.max(np.sqrt((dd**2).sum(axis=0)))
        if step > 1:
            dd = dd / step
        d = d + dd
        if step < tol:
            break
    return d


def align_trials_patchwise(
    mean_sum: np.ndarray,
    tforms: np.ndarray,
    halves: np.ndarray | None = None,
    patch_px: float = 1500,
    n_rounds: int = 3,
    patches: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[PatchwiseFit, AlignmentInfo]:
    """Align the trials with patchwise displacement fields, groupwise.

    Port of ``alignTrialsPatchwise.m``. A common-frame pixel ``x`` is read
    from trial ``t`` at ``x + u_t(x)``, with ``u_t`` set by its values at the
    centres of patches of imaged pixels (:func:`patch_field_matrix`). The
    patches are k-means clusters of the pixels imaged in at least half of
    the trials (:func:`make_patches`). The fit is groupwise, as
    :func:`align_trials_affine`, with a gain and offset per patch, robust
    bisquare weights and a weak penalty on departures from the plane; the
    mean field across trials is held at that of the starting maps. With
    ``halves``, each trial is refit on both halves, and each centre's
    departure from the plane is scaled by ``max(0, 1 - noise variance /
    variance across trials)``, so a bend that is not above its noise
    follows the plane.

    Parameters
    ----------
    mean_sum : ndarray of shape (trials, rows, cols)
        One mean image per trial (NaN where unscanned).
    tforms : ndarray of shape (trials, 2, 3)
        Starting maps (e.g. the rigid shifts), zero-based; trials whose map
        has a NaN are not fitted.
    halves : ndarray of shape (trials, 2, rows, cols), optional
        Mean images of two halves of equal size of each trial's frames.
        Without, the departures are kept as fitted.
    patch_px : float
        Pixels per patch (``alignPatchPx``).
    n_rounds : int
        Rounds of template and refit.
    patches : (ndarray, ndarray), optional
        ``(patch, nodes)`` to use instead of :func:`make_patches` (MATLAB's
        ``warp.patch`` and zero-based ``warp.nodes``, for parity tests).

    Returns
    -------
    fit : PatchwiseFit
        Without patches (too few imaged pixels, or fewer than 2 trials):
        no centres, and ``affine`` the starting maps.
    info : AlignmentInfo
    """
    m = np.asarray(mean_sum, dtype=np.float64)
    n_trials, h, w = m.shape
    tforms = np.array(tforms, dtype=np.float64).reshape(n_trials, 2, 3)
    trials = _fitted_trials(tforms)
    s = _sqrt_nan(m)
    fit = PatchwiseFit(
        np.zeros((0, 2)),
        float("nan"),
        np.zeros((n_trials, 0, 2)),
        np.zeros((h, w), np.int32),
        tforms.copy(),
    )
    info = AlignmentInfo(np.full(n_trials, np.nan), np.full(n_trials, np.nan))
    if trials.size < 2:
        return fit, info
    if patches is None:
        patches = _imaged_patches(s, tforms, trials, patch_px)
        if patches is None:
            return fit, info
    lay = _FieldLayout(*patches)

    # centre displacements of the starting maps, MATLAB's (2, K) per trial
    disp = np.full((n_trials, 2, lay.k), np.nan)
    for t in trials:
        a = tforms[t]
        disp[t] = a[:, :2] @ lay.nodes.T + a[:, 2:3] - lay.nodes.T
    disp0 = disp.copy()
    mean0 = disp[trials].mean(axis=0)
    for _ in range(n_rounds):
        template = lay.template(s, disp, trials)
        for t in trials:
            disp[t] = lay.refine(s[t], template, disp[t])
        disp[trials] = disp[trials] - disp[trials].mean(axis=0) + mean0

    # reliability: refit each half against the final template
    bend = np.full((n_trials, 2, lay.k), np.nan)
    for t in trials:
        bend[t] = (lay.e_mat @ disp[t].T).T
    bend_var = (bend[trials] ** 2).mean(axis=(0, 1))
    if halves is None:
        noise_var = np.full(lay.k, np.nan)
        weight = np.ones(lay.k)
    else:
        noise_var = _halves_noise(lay, s, halves, disp, trials, mean0)
        with np.errstate(invalid="ignore", divide="ignore"):
            weight = np.maximum(0.0, 1 - noise_var / bend_var)
        weight[~(bend_var > 0)] = 0.0
    for t in trials:
        disp[t] = disp[t] - (1 - weight) * bend[t]

    fit.nodes, fit.sigma, fit.patch = lay.nodes, lay.sigma, lay.patch
    fit.disp = disp.transpose(0, 2, 1).copy()
    fit.affine[trials] = _field_planes(lay, disp[trials])
    info.n_px = np.bincount(lay.pix, minlength=lay.k)
    info.noise_sd = np.sqrt(noise_var)
    info.bend_sd = np.sqrt(bend_var)
    info.weight = weight
    _fit_quality(info, lay, s, disp, disp0, trials)
    return fit, info


def _field_planes(lay, disp: np.ndarray) -> np.ndarray:
    """Each field's plane as a map ``(trials, 2, 3)``: the plane at the
    origin and one pixel along each axis."""
    _, _, l_p = patch_field_matrix(
        lay.nodes,
        lay.sigma,
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    out = np.empty((disp.shape[0], 2, 3))
    for i, d in enumerate(disp):
        p = l_p @ d.T
        out[i, :, :2] = np.eye(2) + np.vstack([p[1] - p[0], p[2] - p[0]]).T
        out[i, :, 2] = p[0]
    return out


def _fit_quality(info, lay, s, disp, disp0, trials) -> None:
    """Each trial's correlation with the final template over the patches
    and the RMS change of its field from the start (into ``info``)."""
    template = lay.template(s, disp, trials)
    for t in trials:
        u = lay.lmat @ disp[t].T
        v = interp2_linear(s[t], lay.r + u[:, 0], lay.c + u[:, 1])
        ok = ~np.isnan(v) & ~np.isnan(template)
        info.corr[t] = _corr(v[ok], template[ok])
        du = u - lay.lmat @ disp0[t].T
        info.rms_disp[t] = np.sqrt(np.mean((du**2).sum(axis=1)))


def _imaged_patches(s, tforms, trials, patch_px):
    """Patches on the pixels imaged in at least half of the trials under
    the starting maps; None if fewer than 500 are."""
    seen = np.zeros(s.shape[1:])
    for t in trials:
        seen += ~np.isnan(warp_image_affine(s[t], tforms[t], "linear"))
    valid = seen >= trials.size / 2
    if valid.sum() < 500:
        return None
    return make_patches(valid, patch_px)


class _FieldLayout:
    """The fit pixels of a patchwise field and its matrices.

    Parameters
    ----------
    patch : ndarray of int, shape (rows, cols)
        Patch of each pixel (1 to K, 0 outside).
    nodes : ndarray of shape (K, 2)
        The patches' centres, zero-based.
    """

    lambda_rel = 1e-2  # departures' penalty, relative to a centre's data

    def __init__(self, patch, nodes) -> None:
        """The pixels in MATLAB's order, sigma and the field matrix."""
        self.patch = np.asarray(patch, dtype=np.int32)
        self.nodes = np.asarray(nodes, dtype=np.float64).reshape(-1, 2)
        self.k = self.nodes.shape[0]
        if self.k > 1:
            dd = np.sqrt(
                ((self.nodes[:, None] - self.nodes[None]) ** 2).sum(axis=2)
            )
            np.fill_diagonal(dd, np.inf)
            self.sigma = float(np.median(dd.min(axis=1)) / 2)
        else:
            self.sigma = float("inf")
        rows, cols = _colmajor_nonzero(self.patch > 0)
        self.pix = self.patch[rows, cols].astype(np.intp) - 1
        self.r = rows.astype(np.float64)
        self.c = cols.astype(np.float64)
        self.lmat, self.e_mat, _ = patch_field_matrix(
            self.nodes, self.sigma, self.r, self.c
        )

    def template(self, s, disp, trials):
        """The trials warped into the common frame at the fit pixels."""
        return _patch_template(s, disp, self.lmat, self.r, self.c, trials)

    def refine(self, s, template, d):
        """:func:`_refine_field` of one trial's image."""
        return _refine_field(
            s,
            template,
            d,
            self.lmat,
            self.r,
            self.c,
            self.pix,
            self.e_mat,
            self.lambda_rel,
        )


def _halves_noise(lay, s, halves, disp, trials, mean0) -> np.ndarray:
    """Each centre's noise variance of its departure from the plane: the
    trials refit on each half against the final template; a quarter of the
    variance of the halves' difference (halves of equal size each have
    twice the noise variance of the full data)."""
    hv = np.asarray(halves, dtype=np.float64)
    template = lay.template(s, disp, trials)
    n_trials = disp.shape[0]
    disp_h = np.full((2, n_trials, 2, lay.k), np.nan)
    for half in range(2):
        sh = _sqrt_nan(hv[:, half])
        for t in trials:
            disp_h[half, t] = lay.refine(sh[t], template, disp[t])
        disp_h[half, trials] += mean0 - disp_h[half, trials].mean(axis=0)
    d_bend = np.full((n_trials, 2, lay.k), np.nan)
    for t in trials:
        d_bend[t] = (lay.e_mat @ (disp_h[0, t] - disp_h[1, t]).T).T
    return (d_bend[trials] ** 2).mean(axis=(0, 1)) / 4
