"""SiLo's path-level plan: alignment, sources, pixels and subproblems.

Port of what GIAnT-MATLAB's ``SILo.m`` (main at 5a0673b) does for one imaging
path between the per-trial localizations and the extraction of the trials,
and of the grouping at the top of ``extractTrial.m``:

1. the trials' mean and activity images are put on one canvas and aligned
   (:mod:`~giant_python.extraction.raster.alignment`): a shift per trial,
   refined for the valid trials to an affine map or a patchwise field
   (``inter_trial_alignment``);
2. valid trials are those whose mean image matches the template and whose
   activity image covers enough pixels; their aligned activity images are
   averaged into the combined activity image, minus its local median;
3. sources are seeded in it and each gets a disk of selected pixels
   (:mod:`~giant_python.extraction.raster.seeding`);
4. sources whose zones touch are grouped into subproblems
   (:func:`group_sources`), which are the same in every trial.

The result is an :class:`ExtractionPlan`, stored with :func:`write_plan` and
:func:`read_plan` (``extraction_plan.h5``).

Conventions shared with the trial loader and the subproblem solver. The
consensus frame is the across-trial aligned pixel grid ``(rows, cols)`` of
the path, in MATLAB's orientation and zero-based. The selected pixels are
listed in MATLAB's column-major order of the selected-pixel map
(``find(selPix)``), so that index ``k`` of the list is row ``k`` of MATLAB's
data and footprints. A subproblem (:class:`SubproblemGroup`) holds indices
into that list and into the path's source list, and a crop of the map.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import h5py
import numpy as np
import scipy.ndimage as ndimage

from .alignment import (
    TrialWarp,
    align_trials,
    align_trials_affine,
    align_trials_patchwise,
    mask_to_consensus,
    warp_trial_image,
)
from .localization import nanmedfilt2
from .peaks import SEED_FITS
from .solver_setup import Subproblem
from .trial_data import RigidShift
from .seeding import (
    SEEDING_METHODS,
    disk_strel,
    seed_sources,
    select_source_pixels,
    sources_from_seeds,
)

ALIGNMENT_MODES = ("rigid", "affine", "patchwise")
PLAN_FORMAT = "giant-python extraction plan"
PLAN_VERSION = 1


@dataclass(frozen=True)
class PlanParams:
    """Parameters of the plan stage (GIAnT-MATLAB names in brackets).

    Attributes
    ----------
    nan_thresh : float
        Largest fraction of valid trials in which a pixel may be missing
        (``nanThresh``).
    max_shift : int
        Search range of the across-trial alignment and margin of its
        template (``maxshift`` in ``SILo.m``).
    align_clip : float
        ``clip`` of the whole-pixel estimate (80 in ``SILo.m``).
    template_trials : int
        Largest number of trials the template is made from (20).
    max_corr_thresh : float
        Upper limit of the correlation a valid trial must exceed (0.90).
    inter_trial_alignment : str
        ``"rigid"`` (a shift per trial), ``"affine"`` (an affine map per
        valid trial) or ``"patchwise"`` (a smooth displacement field per
        valid trial on top of its affine plane) (``interTrialAlignment``).
    align_patch_px : float
        Imaged pixels per patch of the patchwise field (``alignPatchPx``).
    peak_method : str
        ``"getactimpeaks"``; ``"wmf"`` is not ported yet (``peakMethod``).
    seed_fit : str
        ``"local"`` (the default: local Gaussian fits run to convergence,
        seconds per path) or ``"matlab"`` (MATLAB's capped joint fits). See
        :func:`~giant_python.extraction.raster.peaks.get_act_im_peaks`.
    seed_merge_px : float
        Reject a residual peak that ends closer than this to another peak
        (default 1.5; 0: MATLAB's rule only).
    peak_th : float
        Seed threshold in robust SDs of the combined activity image
        (``peakth``: 3 for SLAP2, 8 otherwise).
    min_peak_distance : int
        ``minPeakDistance``.
    d_xy : float
        How large sources can be, in pixels (``dXY``).
    sigma_px : float
        Gaussian sigma of the PSF in pixels (``sigma_px``).
    valid_kernel : int
        Side of the footprints' support (``validKernel``; 5 by default,
        MATLAB's is 3), which widens the zones that group sources.
    min_source_pixels : int
        A source needs more selected pixels than this (5).
    """

    nan_thresh: float = 0.33
    max_shift: int = 5
    align_clip: float = 80.0
    template_trials: int = 20
    max_corr_thresh: float = 0.90
    inter_trial_alignment: str = "rigid"
    align_patch_px: float = 1500.0
    peak_method: str = "getactimpeaks"
    seed_fit: str = "local"
    seed_merge_px: float = 1.5
    peak_th: float = 3.0
    min_peak_distance: int = 1
    d_xy: float = 3.0
    sigma_px: float = 1.33
    valid_kernel: int = 5
    min_source_pixels: int = 5

    def __post_init__(self) -> None:
        """Validate the option values."""
        if self.inter_trial_alignment not in ALIGNMENT_MODES:
            raise ValueError(
                f"inter_trial_alignment must be one of {ALIGNMENT_MODES}, "
                f"not {self.inter_trial_alignment!r}"
            )
        if self.peak_method not in SEEDING_METHODS:
            raise ValueError(
                f"peak_method must be one of {SEEDING_METHODS}, "
                f"not {self.peak_method!r}"
            )
        if self.seed_fit not in SEED_FITS:
            raise ValueError(
                f"seed_fit must be one of {SEED_FITS}, not {self.seed_fit!r}"
            )
        if not self.seed_merge_px >= 0:
            raise ValueError("seed_merge_px must be nonnegative")
        for name in (
            "d_xy",
            "sigma_px",
            "template_trials",
            "valid_kernel",
            "align_patch_px",
        ):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive")
        if self.max_shift < 0:
            raise ValueError("max_shift must not be negative")
        if not self.align_patch_px >= 1:
            raise ValueError("align_patch_px must be at least 1 pixel")

    @property
    def sel_radius(self) -> int:
        """Radius of a source's selected pixels (``selRadius``)."""
        return math.ceil(2 * self.d_xy)

    @property
    def zone_radius(self) -> int:
        """Radius of the zones whose overlap groups sources."""
        return math.ceil(1.5 * self.sigma_px + (self.valid_kernel - 1) / 2)


@dataclass
class TrialImages:
    """What the plan stage uses of one localized trial.

    Attributes
    ----------
    mean_im : ndarray of shape (rows, cols, channels)
        The trial's mean image (``RegisteredTrial.mean_im``).
    act_im : ndarray of shape (rows, cols)
        The trial's activity image (``RasterLocalization.act_im``).
    peaks : ndarray of shape (n, 4), optional
        The trial's own seeds, carried into the plan.
    discard : ndarray of bool, optional
        The trial's discarded frames, carried into the plan.
    half_ims : ndarray of shape (2, rows, cols, channels), optional
        Mean images of two halves of the trial's frames, which weight the
        patchwise alignment's patches (only that mode uses them).
    """

    mean_im: np.ndarray
    act_im: np.ndarray
    peaks: np.ndarray | None = None
    discard: np.ndarray | None = None
    half_ims: np.ndarray | None = None

    @classmethod
    def from_localization(cls, localization, trial) -> "TrialImages":
        """From the two results of
        :func:`~giant_python.extraction.raster.trial.localize_trial`."""
        return cls(
            trial.mean_im,
            localization.act_im,
            localization.peaks,
            trial.discard,
            trial.half_ims,
        )


@dataclass
class SubproblemGroup:
    """One connected group of sources: a subproblem of every trial.

    Attributes
    ----------
    sources : ndarray of int
        The group's sources, as indices into the path's source list.
    pixels : ndarray of int
        The group's pixels, as sorted indices into the selected-pixel list.
    row0, col0 : int
        Origin of the crop in the consensus frame.
    sel : ndarray of bool, shape (crop rows, crop cols)
        The group's pixels within the crop (MATLAB's ``selPix_p``); its
        column-major order is the order of ``pixels``.
    sources_rc : ndarray of int, shape (n, 2)
        The sources' positions within the crop.
    """

    sources: np.ndarray
    pixels: np.ndarray
    row0: int
    col0: int
    sel: np.ndarray
    sources_rc: np.ndarray

    def subproblem(
        self,
        y: np.ndarray,
        finv: np.ndarray,
        baseline: Optional[np.ndarray] = None,
    ) -> Subproblem:
        """The group's subproblem in one trial.

        Parameters
        ----------
        y : ndarray of shape (channels, pixels, frames)
            The trial's data at all of the path's selected pixels
            (``TrialData.y``).
        finv : ndarray of shape (pixels, frames)
            Its variance multiplier (``TrialData.finv``).
        baseline : ndarray of shape (pixels, frames), optional
            The group's external baseline (``Subproblem.baseline``), at the
            group's pixels only.

        Returns
        -------
        Subproblem
            The solver's input, with the group's pixels, sources and crop.
        """
        return Subproblem(
            y=y[:, self.pixels],
            finv=finv[self.pixels],
            sel=self.sel,
            sources_rc=self.sources_rc,
            pixels=np.asarray(self.pixels, dtype=np.int64),
            sources=np.asarray(self.sources, dtype=np.int64),
            row0=int(self.row0),
            col0=int(self.col0),
            baseline=baseline,
        )


def loader_warp(warp: TrialWarp, trial: int):
    """A trial's across-trial map in the form the trial loader takes.

    Parameters
    ----------
    warp : TrialWarp
        The plan's alignment (``ExtractionPlan.warp``).
    trial : int
        Zero-based trial; it must have been aligned.

    Returns
    -------
    RigidShift or callable
        The trial's shift (added to its motion, MATLAB's ``isShift`` path),
        or, for an affine map or a patchwise field, a function from
        zero-based consensus ``(rows, cols)`` to zero-based positions in
        the trial's registered frame (MATLAB's ``trialWarpCoords`` path).

    Raises
    ------
    ValueError
        If the trial was not aligned across trials.
    """
    if not warp.aligned[trial]:
        raise ValueError(f"trial {trial} was not aligned across trials")
    if warp.is_shift(trial):
        row, col = warp.shift(trial)
        return RigidShift(float(row), float(col))

    def coords(rows, cols):
        """Consensus to trial-frame positions (zero-based)."""
        return warp.coords(trial, rows, cols)

    return coords


def _matlab_order_labels(zones: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connected components numbered as MATLAB's ``bwconncomp(z, 4)``: by
    their first pixel in column-major order (labels from 1)."""
    labels, n = ndimage.label(zones.T)
    flat = labels.ravel()  # column-major order of zones
    _, first = np.unique(flat, return_index=True)
    first = first[1:] if flat.size and (flat == 0).any() else first
    rank = np.zeros(n + 1, dtype=np.int64)
    rank[1 + np.argsort(first)] = np.arange(1, n + 1)
    return rank[labels].T, n


def _group_pixels(
    rows: np.ndarray, cols: np.ndarray, sel: np.ndarray, nhood: np.ndarray
) -> tuple[np.ndarray, int, int]:
    """A group's selected pixels (cropped) and the crop's origin.

    The disks of ``nhood`` around the group's sources, within ``sel``.
    """
    h, w = sel.shape
    half = (nhood.shape[0] - 1) // 2
    r_lo, r_hi = max(0, rows.min() - half), min(h - 1, rows.max() + half)
    c_lo, c_hi = max(0, cols.min() - half), min(w - 1, cols.max() + half)
    seed = np.zeros((r_hi - r_lo + 1, c_hi - c_lo + 1), bool)
    seed[rows - r_lo, cols - c_lo] = True
    mask = ndimage.binary_dilation(seed, structure=nhood)
    mask &= sel[r_lo : r_hi + 1, c_lo : c_hi + 1]
    if not mask.any():
        raise ValueError("a group of sources has no selected pixels")
    in_rows = np.flatnonzero(mask.any(axis=1))
    in_cols = np.flatnonzero(mask.any(axis=0))
    crop = mask[in_rows[0] : in_rows[-1] + 1, in_cols[0] : in_cols[-1] + 1]
    return crop, r_lo + int(in_rows[0]), c_lo + int(in_cols[0])


def group_sources(
    rows: np.ndarray,
    cols: np.ndarray,
    sel: np.ndarray,
    zone_radius: int,
    sel_radius: int,
) -> list[SubproblemGroup]:
    """Group sources into separable subproblems (as ``extractTrial.m``).

    A disk of ``zone_radius`` is put on every source; sources whose disks
    are 4-connected form a group. A group's pixels are the selected pixels
    within ``sel_radius`` disks of its sources, cropped to their bounding
    box. Groups are in MATLAB's order (``bwconncomp``: by the first pixel of
    their zone, column by column), and so are each group's sources and
    pixels.

    Parameters
    ----------
    rows, cols : ndarray of int
        The sources' pixels in the consensus frame (zero-based).
    sel : ndarray of bool, shape (rows, cols)
        The path's selected pixels.
    zone_radius : int
        Radius of the grouping zones (:attr:`PlanParams.zone_radius`).
    sel_radius : int
        Radius of a source's selected pixels
        (:attr:`PlanParams.sel_radius`).

    Returns
    -------
    list of SubproblemGroup
    """
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    sel = np.asarray(sel, dtype=bool)
    if rows.size == 0:
        return []
    zones = np.zeros(sel.shape, bool)
    zones[rows, cols] = True
    zones = ndimage.binary_dilation(zones, structure=disk_strel(zone_radius))
    labels, n_groups = _matlab_order_labels(zones)
    source_label = labels[rows, cols]
    sel_index = np.full(sel.shape, -1, dtype=np.int64)
    sel_index.T[sel.T] = np.arange(sel.sum())  # column-major list index
    nhood = disk_strel(sel_radius)
    groups = []
    for g in range(1, n_groups + 1):
        src = np.flatnonzero(source_label == g)
        crop, row0, col0 = _group_pixels(rows[src], cols[src], sel, nhood)
        index = sel_index[
            row0 : row0 + crop.shape[0], col0 : col0 + crop.shape[1]
        ]
        groups.append(
            SubproblemGroup(
                sources=src,
                pixels=index.T[crop.T],
                row0=row0,
                col0=col0,
                sel=crop,
                sources_rc=np.column_stack(
                    [rows[src] - row0, cols[src] - col0]
                ),
            )
        )
    return groups


@dataclass
class ExtractionPlan:
    """The plan of one imaging path: what every trial's extraction shares.

    Images are in MATLAB's orientation; trials, sources and pixels are
    zero-based. ``T`` trials, ``(H, W)`` consensus frame, ``C`` channels,
    ``N`` sources, ``P`` selected pixels.

    Attributes
    ----------
    params : PlanParams
        The parameters the plan was made with.
    keep_trials : ndarray of bool, shape (T,)
        Trials that were given (those that passed file verification).
    template : ndarray of shape (H + 2 * max_shift, W + 2 * max_shift)
        The consensus template of the summed-channel mean images.
    template_trials : ndarray of int
        Trials the template was made from.
    shifts : ndarray of shape (T, 2)
        Each trial's shift ``(rows, cols)``: consensus pixel ``(r, c)`` is
        at ``(r, c) + shift`` in the trial's registered image; NaN for
        trials that were not aligned (MATLAB's ``motOutput``). For an
        affine or patchwise alignment, the displacement at the canvas
        centre (the offsets of the motion records); ``warp`` holds the
        maps.
    corr : ndarray of shape (T,)
        Correlation of each trial's mean image with the template.
    corr_thresh : float
        The correlation a valid trial exceeds.
    act_valid_pix : ndarray of shape (T,)
        Fraction of the frame each aligned activity image covers.
    valid_trials : ndarray of int
        Trials that are extracted and that the images below average.
    warp : TrialWarp
        Each trial's map from the consensus frame to its registered image.
    mean_ims : ndarray of shape (T, H, W, C)
        The trials' mean images on the common canvas, before alignment.
    act_ims : ndarray of shape (T, H, W)
        The trials' activity images, before alignment.
    mean_aligned : ndarray of shape (T, H, W, C)
        The mean images in the consensus frame (NaN for trials not aligned).
    act_aligned : ndarray of shape (T, H, W)
        The activity images in the consensus frame.
    mean_im : ndarray of shape (H, W, C)
        Mean of the valid trials' aligned mean images, NaN ignored.
    act_im : ndarray of shape (H, W)
        The combined activity image the sources were seeded in: the valid
        trials' mean, scaled, minus its 5 x 5 NaN-median.
    soma_mask : ndarray of bool, shape (H, W)
        Pixels in which no source was seeded.
    px_always_valid : ndarray of bool, shape (H, W)
        Pixels imaged in enough of the valid trials.
    seeds : ndarray of shape (n_seeds, 4)
        ``[amplitude, row, col, sigma]`` of every seed (MATLAB's ``thetaf``).
    source_rows, source_cols : ndarray of int, shape (N,)
        The sources' pixels.
    source_values : ndarray of shape (N,)
        The sources' seed amplitudes (``sources.V``).
    sel : ndarray of bool, shape (H, W)
        The selected pixels (the union over the sources).
    source_pixels : list of ndarray of int
        Each source's selected pixels, as sorted indices into the
        selected-pixel list.
    groups : list of SubproblemGroup
        The subproblems.
    trial_peaks : list of (ndarray or None)
        Each trial's own seeds, when given.
    trial_discard : list of (ndarray or None)
        Each trial's discarded frames, when given.
    """

    params: PlanParams
    keep_trials: np.ndarray
    template: np.ndarray
    template_trials: np.ndarray
    shifts: np.ndarray
    corr: np.ndarray
    corr_thresh: float
    act_valid_pix: np.ndarray
    valid_trials: np.ndarray
    warp: TrialWarp
    mean_ims: np.ndarray
    act_ims: np.ndarray
    mean_aligned: np.ndarray
    act_aligned: np.ndarray
    mean_im: np.ndarray
    act_im: np.ndarray
    soma_mask: np.ndarray
    px_always_valid: np.ndarray
    seeds: np.ndarray
    source_rows: np.ndarray
    source_cols: np.ndarray
    source_values: np.ndarray
    sel: np.ndarray
    source_pixels: list
    groups: list
    trial_peaks: list = field(default_factory=list)
    trial_discard: list = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        """``(rows, cols)`` of the consensus frame."""
        return self.sel.shape

    @property
    def n_trials(self) -> int:
        """Number of trials (kept or not)."""
        return self.shifts.shape[0]

    @property
    def n_sources(self) -> int:
        """Number of sources."""
        return self.source_rows.size

    @property
    def n_pixels(self) -> int:
        """Number of selected pixels."""
        return int(self.sel.sum())

    @property
    def aligned_trials(self) -> np.ndarray:
        """Trials that have a map to the consensus frame."""
        return np.flatnonzero(self.warp.aligned)

    def selected_pixels(self) -> tuple[np.ndarray, np.ndarray]:
        """Rows and columns of the selected-pixel list (the order of
        MATLAB's ``find(selPix)``: column by column)."""
        cols, rows = np.nonzero(self.sel.T)
        return rows, cols

    def source_mask(self, source: int) -> np.ndarray:
        """One source's selected pixels as a map (a slice of MATLAB's
        ``selPix``)."""
        rows, cols = self.selected_pixels()
        mask = np.zeros(self.shape, bool)
        index = self.source_pixels[source]
        mask[rows[index], cols[index]] = True
        return mask


def _canvas(trials: Sequence, keep: np.ndarray):
    """The trials' mean and activity images on one canvas (top left),
    NaN elsewhere and for trials that are not kept."""
    kept = [t for t, k in zip(trials, keep) if k]
    h = max(max(t.mean_im.shape[0], t.act_im.shape[0]) for t in kept)
    w = max(max(t.mean_im.shape[1], t.act_im.shape[1]) for t in kept)
    n_channels = np.atleast_3d(kept[0].mean_im).shape[2]
    for t in kept:
        if np.atleast_3d(t.mean_im).shape[2] != n_channels:
            raise ValueError("the trials' mean images differ in channels")
    mean_ims = np.full((len(trials), h, w, n_channels), np.nan)
    act_ims = np.full((len(trials), h, w), np.nan)
    for k, trial in enumerate(trials):
        if not keep[k]:
            continue
        m = np.atleast_3d(trial.mean_im)
        mean_ims[k, : m.shape[0], : m.shape[1]] = m
        a = np.asarray(trial.act_im)
        act_ims[k, : a.shape[0], : a.shape[1]] = a
    return mean_ims, act_ims


def _valid_trials(corr, act_aligned, max_corr_thresh):
    """SiLo's valid trials: outliers of alignment quality and coverage out.

    Returns the valid trials, the correlation threshold and each trial's
    covered fraction.
    """
    known = corr[~np.isnan(corr)]
    thresh = max_corr_thresh
    if known.size:
        spread = np.std(known, ddof=1) if known.size > 1 else 0.0
        thresh = min(max_corr_thresh, np.median(known) - 2 * spread)
    covered = np.mean(~np.isnan(act_aligned), axis=(1, 2))
    with np.errstate(invalid="ignore"):
        valid = (corr > thresh) & (covered > covered.mean() / 2)
    return np.flatnonzero(valid), float(thresh), covered


def combined_activity_image(
    act_aligned: np.ndarray, nan_thresh: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The image sources are seeded in, from aligned activity images.

    The mean over the trials (NaN where any is NaN), scaled so that its
    maximum is between 10 and 100, minus its 5 x 5 NaN-median.

    Parameters
    ----------
    act_aligned : ndarray of shape (trials, rows, cols)
        The valid trials' aligned activity images.
    nan_thresh : float
        Pixels missing in more than this fraction of trials are NaN.

    Returns
    -------
    act_im, med_im, nan_frac : ndarray of shape (rows, cols)
        The combined image, the local median that was subtracted, and the
        fraction of trials in which each pixel is NaN.
    """
    shape = act_aligned.shape[1:]
    if act_aligned.shape[0] == 0:
        nan = np.full(shape, np.nan)
        return nan, nan.copy(), nan.copy()
    act_im = act_aligned.mean(axis=0)
    top = np.nanmax(act_im) if not np.isnan(act_im).all() else np.nan
    if top > 0:
        act_im = act_im / 10.0 ** (math.floor(math.log10(top)) - 1)
    nan_frac = np.isnan(act_aligned).mean(axis=0)
    act_im[nan_frac > nan_thresh] = np.nan
    med_im = nanmedfilt2(act_im, 5)
    return act_im - med_im, med_im, nan_frac


# fields added to PlanParams after plans were first written, or whose
# default changed, with the value that describes plans written before them
# (MATLAB's seed fit, no merging, its patch size, its 3 x 3 support)
_PLAN_FIELDS_ADDED = {
    "seed_fit": "matlab",
    "seed_merge_px": 0.0,
    "align_patch_px": 1500.0,
    "valid_kernel": 3,
}


def _saved_params(text: str) -> dict:
    """A plan file's saved PlanParams; fields it predates get the values
    it was made with, not today's defaults."""
    saved = json.loads(text)
    for name, before in _PLAN_FIELDS_ADDED.items():
        saved.setdefault(name, before)
    return saved


def _soma_canvas(soma_mask, shape, warp=None, trial=None) -> np.ndarray:
    """The soma mask on the consensus canvas (top left).

    With ``trial``, the mask is in that trial's registered image and is
    moved into the consensus frame through the trial's map.
    """
    out = np.zeros(shape, bool)
    if soma_mask is None:
        return out
    mask = np.asarray(soma_mask, dtype=bool)
    if mask.ndim != 2 or mask.shape[0] > shape[0] or mask.shape[1] > shape[1]:
        raise ValueError(
            f"soma_mask is {mask.shape}, the consensus frame {shape}"
        )
    out[: mask.shape[0], : mask.shape[1]] = mask
    if trial is None:
        return out
    if not warp.aligned[trial]:
        warnings.warn(
            f"the soma mask's trial ({trial}) was not aligned across "
            "trials; using the mask without across-trial alignment"
        )
    return mask_to_consensus(out, warp, trial, shape)


def _canvas_halves(trials: Sequence, keep: np.ndarray, shape) -> np.ndarray:
    """The kept trials' half images summed over channels, on the canvas
    (MATLAB's ``Mhalves``: ``(trials, 2, rows, cols)``); NaN for trials
    without them."""
    out = np.full((len(trials), 2) + tuple(shape), np.nan)
    for k, trial in enumerate(trials):
        if not keep[k] or trial is None or trial.half_ims is None:
            continue
        h = np.asarray(trial.half_ims, dtype=np.float64)
        if h.ndim == 3:  # one channel without its axis
            h = h[..., None]
        s = h.sum(axis=3)  # NaN where a channel is, as the mean images
        out[k, :, : s.shape[1], : s.shape[2]] = s
    return out


def refine_alignment(
    warp: TrialWarp,
    mean_sum: np.ndarray,
    valid: np.ndarray,
    params: PlanParams,
    halves: np.ndarray | None = None,
):
    """Refine the rigid alignment of the valid trials (``SILo.m`` after the
    rigid fit): affine maps or patchwise fields, groupwise. Trials that are
    aligned but not valid keep their shift (as a field: the shift at every
    centre).

    Parameters
    ----------
    warp : TrialWarp
        The rigid alignment.
    mean_sum : ndarray of shape (trials, rows, cols)
        The trials' mean images summed over channels, on the canvas.
    valid : ndarray of int
        The valid trials.
    params : PlanParams
        ``inter_trial_alignment`` ('affine' or 'patchwise') and
        ``align_patch_px``.
    halves : ndarray of shape (trials, 2, rows, cols), optional
        The trials' half images summed over channels (patchwise only).

    Returns
    -------
    warp : TrialWarp
        The refined alignment.
    shifts : ndarray of shape (trials, 2)
        Each aligned trial's displacement at the canvas centre (MATLAB's
        updated ``motOutput``: the rigid-equivalent offsets of the motion
        records).
    info : AlignmentInfo
    """
    n_trials = warp.n_trials
    start = np.full(warp.affine.shape, np.nan)
    start[valid] = warp.affine[valid]
    affine = warp.affine.copy()
    if params.inter_trial_alignment == "patchwise":
        fit, info = align_trials_patchwise(
            mean_sum, start, halves, params.align_patch_px
        )
        nodes = fit.nodes
        disp = np.full((n_trials, nodes.shape[0], 2), np.nan)
        for t in np.flatnonzero(warp.aligned):
            a = warp.affine[t]
            disp[t] = (a[:, :2] @ nodes.T + a[:, 2:3] - nodes.T).T
        disp[valid] = fit.disp[valid]
        affine[valid] = fit.affine[valid]
        out = TrialWarp(affine, nodes, fit.sigma, disp, fit.patch)
    else:
        refined, info = align_trials_affine(mean_sum, start)
        affine[valid] = refined[valid]
        out = TrialWarp(affine)
    h, w = mean_sum.shape[1:]
    x0 = np.array([(h - 1) / 2, (w - 1) / 2])  # MATLAB's (H+1)/2, one-based
    shifts = np.full((n_trials, 2), np.nan)
    for t in np.flatnonzero(out.aligned):
        r_c, c_c = out.coords(t, x0[:1], x0[1:])
        shifts[t] = [r_c[0] - x0[0], c_c[0] - x0[1]]
    return out, shifts, info


def _nanmean0(stack: np.ndarray) -> np.ndarray:
    """Mean over axis 0 with NaN ignored (NaN where nothing is left)."""
    ok = ~np.isnan(stack)
    count = ok.sum(axis=0)
    total = np.where(ok, stack, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def plan_path(  # noqa: C901
    trials: Sequence[TrialImages | None],
    soma_mask: np.ndarray | None = None,
    params: PlanParams | None = None,
    keep_trials: np.ndarray | None = None,
    details: dict | None = None,
    seeds: np.ndarray | None = None,
    soma_mask_trial: int | None = None,
) -> ExtractionPlan:
    """Plan the extraction of one imaging path (the path stage of SiLo).

    Parameters
    ----------
    trials : sequence of (TrialImages or None)
        Every trial of the path in order; None for a trial without a
        localization (one that failed file verification).
    soma_mask : ndarray of bool, optional
        Pixels in which no source is seeded (the user's soma ROIs): in the
        consensus frame, or with ``soma_mask_trial`` in that trial's
        registered image.
    params : PlanParams, optional
        Parameters (defaults: GIAnT-MATLAB's SLAP2 SiLo).
    keep_trials : ndarray of bool, optional
        Trials to use among those given (default: all).
    details : dict, optional
        If given, it receives the intermediates that the plan does not
        keep: ``med_im``, ``nan_frac``, the sources before pruning
        (``rows``, ``cols``, ``values``, ``pixels``, ``keep_sources``) and,
        for a refined alignment, ``alignment_info``
        (:class:`~giant_python.extraction.raster.alignment.AlignmentInfo`).
    seeds : ndarray of shape (n, 4), optional
        ``[amplitude, row, col, sigma]`` (zero-based) of seeds to use
        instead of seeding the combined activity image.
    soma_mask_trial : int, optional
        Zero-based trial on whose registered image ``soma_mask`` was drawn;
        the mask is moved into the consensus frame through that trial's
        map (MATLAB's ``roisToConsensusFrame``). A trial that was not aligned
        leaves the mask where it is, with a warning.

    Returns
    -------
    ExtractionPlan

    Raises
    ------
    ValueError
        If no trial is kept.
    NotImplementedError
        For WMF seeding (not ported).
    """
    p = params or PlanParams()
    if seeds is None and p.peak_method != "getactimpeaks":
        raise NotImplementedError(
            "WMF seeding (findSourcesWMF) is not ported yet"
        )
    keep = np.array([t is not None for t in trials], dtype=bool)
    if keep_trials is not None:
        keep &= np.asarray(keep_trials, dtype=bool).ravel()
    if not keep.any():
        raise ValueError("no trials to plan")
    mean_ims, act_ims = _canvas(trials, keep)
    shape = act_ims.shape[1:]

    mean_sum = mean_ims.sum(axis=3)
    ali = align_trials(
        mean_sum,
        act_ims,
        keep,
        p.max_shift,
        p.align_clip,
        p.template_trials,
    )
    warp = TrialWarp.from_shifts(ali.shifts)
    shifts = ali.shifts.copy()
    aligned = np.flatnonzero(warp.aligned)
    act_aligned = np.full(act_ims.shape, np.nan)
    mean_aligned = np.full(mean_ims.shape, np.nan)
    for t in aligned:
        act_aligned[t] = warp_trial_image(act_ims[t], warp, t)
    valid, corr_thresh, covered = _valid_trials(
        ali.corr, act_aligned, p.max_corr_thresh
    )
    # the valid trials stay those of the rigid alignment, as in MATLAB
    refined = False
    if p.inter_trial_alignment != "rigid" and valid.size > 1:
        halves = (
            _canvas_halves(trials, keep, shape)
            if p.inter_trial_alignment == "patchwise"
            else None
        )
        warp, shifts, refine_info = refine_alignment(
            warp, mean_sum, valid, p, halves
        )
        if details is not None:
            details["alignment_info"] = refine_info
        refined = True
    for t in aligned:
        # a refined map's positions of the canvas, shared by the trial's two
        # images and released before the next trial (rigid: per image)
        grid = warp.grid_coords(t, shape) if refined else None
        if refined:
            act_aligned[t] = warp_trial_image(act_ims[t], warp, t, coords=grid)
        mean_aligned[t] = warp_trial_image(mean_ims[t], warp, t, coords=grid)
        del grid

    act_im, med_im, nan_frac = combined_activity_image(
        act_aligned[valid], p.nan_thresh
    )
    soma = _soma_canvas(soma_mask, shape, warp, soma_mask_trial)
    if seeds is None:
        seeds = seed_sources(
            act_im,
            p.peak_th,
            soma,
            p.min_peak_distance,
            p.peak_method,
            p.seed_fit,
            p.seed_merge_px,
        )
    seeds = np.asarray(seeds, dtype=np.float64).reshape(-1, 4)
    rows, cols, values = sources_from_seeds(seeds, shape)
    with np.errstate(invalid="ignore"):
        px_always_valid = (
            np.isnan(mean_aligned[valid, :, :, 0]).mean(axis=0) < p.nan_thresh
            if valid.size
            else np.zeros(shape, bool)
        )
    selection = select_source_pixels(
        rows, cols, px_always_valid, p.sel_radius, p.min_source_pixels
    )
    if details is not None:
        details.update(
            med_im=med_im,
            nan_frac=nan_frac,
            rows=rows,
            cols=cols,
            values=values,
            pixels=selection.pixels,
            keep_sources=selection.keep,
        )
    kept = np.flatnonzero(selection.keep)
    rows, cols, values = rows[kept], cols[kept], values[kept]
    lists = [selection.pixels[k] for k in kept]
    union = (
        np.unique(np.concatenate(lists)) if lists else np.zeros(0, np.int64)
    )
    sel = np.zeros(shape, bool)
    sel[union % shape[0], union // shape[0]] = True
    return ExtractionPlan(
        params=p,
        keep_trials=keep,
        template=ali.template,
        template_trials=ali.samples,
        shifts=shifts,
        corr=ali.corr,
        corr_thresh=corr_thresh,
        act_valid_pix=covered,
        valid_trials=valid,
        warp=warp,
        mean_ims=mean_ims,
        act_ims=act_ims,
        mean_aligned=mean_aligned,
        act_aligned=act_aligned,
        mean_im=_nanmean0(mean_aligned[valid]),
        act_im=act_im,
        soma_mask=soma,
        px_always_valid=px_always_valid,
        seeds=seeds,
        source_rows=rows,
        source_cols=cols,
        source_values=values,
        sel=sel,
        source_pixels=[np.searchsorted(union, x) for x in lists],
        groups=group_sources(rows, cols, sel, p.zone_radius, p.sel_radius),
        trial_peaks=[
            None if t is None or t.peaks is None else np.asarray(t.peaks)
            for t in trials
        ],
        trial_discard=[
            None if t is None or t.discard is None else np.asarray(t.discard)
            for t in trials
        ],
    )


def _ragged(lists: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Index lists as one array and the offsets of each list in it."""
    ptr = np.zeros(len(lists) + 1, dtype=np.int64)
    ptr[1:] = np.cumsum([len(x) for x in lists])
    flat = (
        np.concatenate(lists).astype(np.int32)
        if len(lists)
        else np.zeros(0, np.int32)
    )
    return flat, ptr


def _unragged(flat: np.ndarray, ptr: np.ndarray) -> list:
    """The lists of :func:`_ragged`."""
    flat = np.asarray(flat, dtype=np.int64)
    return [flat[a:b] for a, b in zip(ptr[:-1], ptr[1:])]


def _write_images(g: h5py.Group, name: str, data: np.ndarray) -> None:
    """Write an image array compressed (most of a multi-ROI frame is NaN)."""
    data = np.asarray(data)
    if data.dtype == bool:
        data = data.astype(np.uint8)
    kwargs = {}
    if data.size:
        kwargs = dict(compression="gzip", compression_opts=4, shuffle=True)
    g.create_dataset(name, data=data, **kwargs)


def write_plan(
    path: str | Path, plan: ExtractionPlan, group: str = "/"
) -> None:
    """Write a plan to an HDF5 file (``extraction_plan.h5``).

    Arrays are stored as numpy holds them (MATLAB's row and column
    orientation, zero-based indices). Layout, under ``group``:

    * attributes ``format``, ``version``, ``params`` (JSON), ``corr_thresh``;
    * ``alignment/``: ``keep_trials``, ``template``, ``template_trials``,
      ``shifts``, ``corr``, ``act_valid_pix``, ``valid_trials`` and
      ``warp/`` (``affine``, ``nodes``, ``disp``, attribute ``sigma``, and
      for a patchwise field ``patch``);
    * ``images/``: ``mean_ims``, ``act_ims``, ``mean_aligned``,
      ``act_aligned``, ``mean_im``, ``act_im``, ``soma_mask``,
      ``px_always_valid``;
    * ``sources/``: ``seeds``, ``rows``, ``cols``, ``values``, ``sel`` and
      each source's pixels as ``pixel_index`` with offsets ``pixel_ptr``;
    * ``groups/``: ``origin`` (row0, col0) and ``shape`` of each crop, and
      the groups' sources and pixels as ``source_index`` / ``source_ptr``
      and ``pixel_index`` / ``pixel_ptr``;
    * ``trials/<t>/``: ``peaks`` and ``discard`` of the trials that have
      them.

    Parameters
    ----------
    path : str or Path
        The file; an existing file is replaced unless ``group`` names a
        group inside it.
    plan : ExtractionPlan
        The plan.
    group : str
        Group to write into (for several imaging paths in one file).
    """
    mode = "w" if group in ("", "/") else "a"
    with h5py.File(path, mode) as f:
        if mode == "a" and group in f:
            del f[group]
        g = f if mode == "w" else f.create_group(group)
        g.attrs["format"] = PLAN_FORMAT
        g.attrs["version"] = PLAN_VERSION
        g.attrs["params"] = json.dumps(asdict(plan.params))
        g.attrs["corr_thresh"] = plan.corr_thresh
        a = g.create_group("alignment")
        a["keep_trials"] = plan.keep_trials.astype(np.uint8)
        _write_images(a, "template", plan.template)
        a["template_trials"] = plan.template_trials.astype(np.int32)
        a["shifts"] = plan.shifts
        a["corr"] = plan.corr
        a["act_valid_pix"] = plan.act_valid_pix
        a["valid_trials"] = plan.valid_trials.astype(np.int32)
        w = a.create_group("warp")
        w["affine"] = plan.warp.affine
        w["nodes"] = plan.warp.nodes
        w["disp"] = plan.warp.disp
        w.attrs["sigma"] = plan.warp.sigma
        if plan.warp.patch is not None:
            _write_images(w, "patch", plan.warp.patch)
        im = g.create_group("images")
        for name in (
            "mean_ims",
            "act_ims",
            "mean_aligned",
            "act_aligned",
            "mean_im",
            "act_im",
            "soma_mask",
            "px_always_valid",
        ):
            _write_images(im, name, getattr(plan, name))
        s = g.create_group("sources")
        s["seeds"] = plan.seeds
        s["rows"] = plan.source_rows.astype(np.int32)
        s["cols"] = plan.source_cols.astype(np.int32)
        s["values"] = plan.source_values
        _write_images(s, "sel", plan.sel)
        s["pixel_index"], s["pixel_ptr"] = _ragged(plan.source_pixels)
        gr = g.create_group("groups")
        gr["origin"] = np.array(
            [[x.row0, x.col0] for x in plan.groups], dtype=np.int32
        ).reshape(-1, 2)
        gr["shape"] = np.array(
            [x.sel.shape for x in plan.groups], dtype=np.int32
        ).reshape(-1, 2)
        gr["source_index"], gr["source_ptr"] = _ragged(
            [x.sources for x in plan.groups]
        )
        gr["pixel_index"], gr["pixel_ptr"] = _ragged(
            [x.pixels for x in plan.groups]
        )
        tr = g.create_group("trials")
        for t in range(plan.n_trials):
            for name, values in (
                ("peaks", plan.trial_peaks),
                ("discard", plan.trial_discard),
            ):
                if t < len(values) and values[t] is not None:
                    data = np.asarray(values[t])
                    if data.dtype == bool:
                        data = data.astype(np.uint8)
                    tr.create_dataset(f"{t}/{name}", data=data)


def _read_groups(gr, rows, cols, src_rows, src_cols) -> list:
    """The subproblem groups of a plan file."""
    sources = _unragged(gr["source_index"][()], gr["source_ptr"][()])
    pixels = _unragged(gr["pixel_index"][()], gr["pixel_ptr"][()])
    groups = []
    for src, pix, origin, shape in zip(
        sources, pixels, gr["origin"][()], gr["shape"][()]
    ):
        row0, col0 = int(origin[0]), int(origin[1])
        crop = np.zeros(tuple(int(x) for x in shape), bool)
        crop[rows[pix] - row0, cols[pix] - col0] = True
        groups.append(
            SubproblemGroup(
                sources=src,
                pixels=pix,
                row0=row0,
                col0=col0,
                sel=crop,
                sources_rc=np.column_stack(
                    [src_rows[src] - row0, src_cols[src] - col0]
                ),
            )
        )
    return groups


def read_plan(path: str | Path, group: str = "/") -> ExtractionPlan:
    """Read a plan written by :func:`write_plan`.

    Parameters
    ----------
    path : str or Path
        The file.
    group : str
        Group the plan was written into.

    Returns
    -------
    ExtractionPlan

    Raises
    ------
    ValueError
        If the group does not hold a plan of a known version.
    """
    with h5py.File(path, "r") as f:
        g = f[group] if group not in ("", "/") else f
        if g.attrs.get("format") != PLAN_FORMAT:
            raise ValueError(f"{path} does not hold an extraction plan")
        if int(g.attrs["version"]) > PLAN_VERSION:
            raise ValueError(
                f"{path} is plan version {int(g.attrs['version'])}; this "
                f"code reads up to {PLAN_VERSION}"
            )
        a, im, s = g["alignment"], g["images"], g["sources"]
        w = a["warp"]
        warp = TrialWarp(
            w["affine"][()],
            w["nodes"][()],
            float(w.attrs["sigma"]),
            w["disp"][()],
            w["patch"][()] if "patch" in w else None,
        )
        sel = s["sel"][()].astype(bool)
        sel_cols, sel_rows = np.nonzero(sel.T)
        src_rows = s["rows"][()].astype(np.int64)
        src_cols = s["cols"][()].astype(np.int64)
        n_trials = a["shifts"].shape[0]
        tr = g["trials"]

        def trial_data(name, as_bool=False):
            """One optional array per trial."""
            out = []
            for t in range(n_trials):
                key = f"{t}/{name}"
                data = tr[key][()] if key in tr else None
                if data is not None and as_bool:
                    data = data.astype(bool)
                out.append(data)
            return out

        return ExtractionPlan(
            params=PlanParams(**_saved_params(g.attrs["params"])),
            keep_trials=a["keep_trials"][()].astype(bool),
            template=a["template"][()],
            template_trials=a["template_trials"][()].astype(np.int64),
            shifts=a["shifts"][()],
            corr=a["corr"][()],
            corr_thresh=float(g.attrs["corr_thresh"]),
            act_valid_pix=a["act_valid_pix"][()],
            valid_trials=a["valid_trials"][()].astype(np.int64),
            warp=warp,
            mean_ims=im["mean_ims"][()],
            act_ims=im["act_ims"][()],
            mean_aligned=im["mean_aligned"][()],
            act_aligned=im["act_aligned"][()],
            mean_im=im["mean_im"][()],
            act_im=im["act_im"][()],
            soma_mask=im["soma_mask"][()].astype(bool),
            px_always_valid=im["px_always_valid"][()].astype(bool),
            seeds=s["seeds"][()],
            source_rows=src_rows,
            source_cols=src_cols,
            source_values=s["values"][()],
            sel=sel,
            source_pixels=_unragged(s["pixel_index"][()], s["pixel_ptr"][()]),
            groups=_read_groups(
                g["groups"], sel_rows, sel_cols, src_rows, src_cols
            ),
            trial_peaks=trial_data("peaks"),
            trial_discard=trial_data("discard", as_bool=True),
        )
