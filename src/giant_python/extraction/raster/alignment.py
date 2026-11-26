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

Only the rigid mode (one shift per trial) is ported. :class:`TrialWarp`
already holds a full affine map per trial and has room for the patchwise
field, so that callers that go through :meth:`TrialWarp.coords` and
:func:`warp_trial_image` need no change when those modes are added.

Images are ``(rows, cols)`` in MATLAB's orientation and coordinates are
zero-based: a shift of ``(dr, dc)`` means that consensus pixel ``(r, c)`` is
at ``(r + dr, c + dc)`` in the trial's registered image.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.ndimage as ndimage


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
        was not aligned. A rigid shift is ``[[1, 0, dr], [0, 1, dc]]``.
    nodes : ndarray of shape (n_nodes, 2)
        Patch centres of the patchwise field; empty for rigid and affine
        alignment.
    sigma : float
        Width of the patchwise field's interpolation (NaN without one).
    disp : ndarray of shape (trials, n_nodes, 2)
        Displacement of each trial at the patch centres.
    """

    affine: np.ndarray
    nodes: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    sigma: float = float("nan")
    disp: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Normalize the arrays' types and shapes."""
        self.affine = np.asarray(self.affine, dtype=np.float64).reshape(
            -1, 2, 3
        )
        self.nodes = np.asarray(self.nodes, dtype=np.float64).reshape(-1, 2)
        if self.disp is None:
            self.disp = np.zeros((self.affine.shape[0], 0, 2))
        self.disp = np.asarray(self.disp, dtype=np.float64)

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
    def from_matlab(cls, affine: np.ndarray) -> "TrialWarp":
        """From MATLAB's ``warp.affine`` as ``(trials, 2, 3)`` (one-based
        coordinates)."""
        a = np.array(affine, dtype=np.float64).reshape(-1, 2, 3)
        a[:, :, 2] += a[:, :, 0] + a[:, :, 1] - 1
        return cls(a)

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

        Port of ``trialWarpCoords.m`` (zero-based).

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
        if self.nodes.shape[0]:
            raise NotImplementedError(
                "patchwise across-trial alignment is not ported yet"
            )
        r = np.asarray(rows, dtype=np.float64)
        c = np.asarray(cols, dtype=np.float64)
        a = self.affine[trial]
        if self.is_shift(trial):  # exact, as MATLAB's 1 * r + 0 * c + shift
            return r + a[0, 2], c + a[1, 2]
        return (
            a[0, 0] * r + a[0, 1] * c + a[0, 2],
            a[1, 0] * r + a[1, 1] * c + a[1, 2],
        )


def warp_trial_image(
    im: np.ndarray, warp: TrialWarp, trial: int, method: str = "linear"
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
        ``"linear"`` (MATLAB's ``'cubic'`` is not ported).

    Returns
    -------
    ndarray
        The resampled image, in the shape of ``im``.
    """
    if method != "linear":
        raise NotImplementedError(f"interpolation {method!r} is not ported")
    im = np.asarray(im, dtype=np.float64)
    h, w = im.shape[:2]
    rr = np.arange(h, dtype=np.float64)[:, None]
    cc = np.arange(w, dtype=np.float64)[None, :]
    r_t, c_t = warp.coords(trial, *np.broadcast_arrays(rr, cc))
    if im.ndim == 2:
        return interp2_linear(im, r_t, c_t)
    return np.stack(
        [interp2_linear(im[:, :, k], r_t, c_t) for k in range(im.shape[2])],
        axis=2,
    )


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
