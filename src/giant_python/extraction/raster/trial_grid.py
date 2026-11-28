"""Solving a trial's subproblems on the trial's own pixel grid.

By default (``solve_grid='consensus'``) a trial is read at the plan's
selected pixels of the consensus frame: each pixel at its position in the
trial's registered frame through the across-trial map, plus the within-trial
motion, in one bilinear interpolation per frame. With a rigid map all pixels
share a frame's interpolation weights (the loader's fast path); with an
affine or patchwise map every pixel has its own fractional position, so
every frame goes through the general interpolation and the interpolation's
blur and noise differ from pixel to pixel.

With ``solve_grid='trial'`` the map is used only to place the plan's
subproblems in the trial. Locally the map is a shift: each subproblem (a
group of sources with its crop of selected pixels) is moved into the trial
by one whole-pixel shift, the rounded mean displacement of the map at its
sources (a trial whose map would leave a source more than a pixel from its
pixel in the moved subproblem is refused). The trial is read at those whole
pixels of its registered frame (the fast path: only the within-trial motion
is interpolated), the subproblems
are solved there with the plan's sources and crops, and the fitted
footprints are mapped back to the consensus frame once per trial
(:func:`footprints_to_consensus`) for the summaries and the viewer. The
sub-pixel rest of the map (about half a pixel per axis) is absorbed by the
footprints, which the solver fits freely within their support, and is put
back when they are mapped.

:func:`trial_grid` gives the layout of one trial; :func:`masks_to_trial`
moves the user ROIs into the trial's frame, where the loader reads the
global and ROI traces in this mode (with the trial's own mean image).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...registration.outputs import matlab_round
from .alignment import TrialWarp
from .plan import ExtractionPlan, SubproblemGroup

SOLVE_GRIDS = ("consensus", "trial")
"""Grids the subproblems can be solved on (``SiloParams.solve_grid``)."""

MAX_SOURCE_OFFSET_PX = 1.0
"""Farthest a source may lie from its pixel in the moved subproblem (px).

A footprint is fitted within ``(valid_kernel - 1) / 2`` pixels of its
source's pixel; a source off its pixel by more than a pixel would lose
part of its footprint to the support's edge. One shift per subproblem
leaves a single source within half a pixel per axis (0.71 px); more only
where the map stretches or bends across a subproblem."""


@dataclass
class TrialGrid:
    """A trial's subproblems on the trial's own pixel grid.

    Attributes
    ----------
    trial : int
        Zero-based trial.
    shifts : ndarray of int, shape (groups, 2)
        Each subproblem's whole-pixel shift ``(rows, cols)`` from the
        consensus frame into the trial's registered frame.
    points : ndarray of int, shape (pixels, 2)
        The pixels of the trial's registered frame that are read (zero
        based, column-major order); the union of the shifted subproblems'
        pixels.
    groups : list of SubproblemGroup
        The plan's groups on this grid: the same sources, crop and source
        positions within the crop; ``pixels`` index ``points``; ``row0``
        and ``col0`` are in the trial's frame.
    sources_rc : ndarray of int, shape (sources, 2)
        Each source's pixel in the trial's frame (its group's shift added).
    mapped_rc : ndarray, shape (selected pixels, 2)
        Where the map puts each of the plan's selected pixels in the trial
        (fractional), for mapping footprints back.
    residual : ndarray, shape (groups,)
        Largest distance (px) over each group's sources between where the
        map puts the source and its pixel in the moved subproblem.
    """

    trial: int
    shifts: np.ndarray
    points: np.ndarray
    groups: list
    sources_rc: np.ndarray
    mapped_rc: np.ndarray
    residual: np.ndarray

    @property
    def n_pixels(self) -> int:
        """Number of pixels read."""
        return int(self.points.shape[0])


def check_solve_grid(value: str) -> str:
    """``value`` if it is one of :data:`SOLVE_GRIDS`.

    Raises
    ------
    ValueError
        Otherwise.
    """
    if value not in SOLVE_GRIDS:
        raise ValueError(
            f"solve_grid must be one of {SOLVE_GRIDS}, not {value!r}"
        )
    return value


def trial_grid(plan: ExtractionPlan, trial: int) -> TrialGrid:
    """The plan's subproblems placed on a trial's own pixel grid.

    Parameters
    ----------
    plan : ExtractionPlan
        The path's plan.
    trial : int
        Zero-based trial; it must have been aligned.

    Each subproblem moves by the map's mean displacement over its sources,
    rounded to whole pixels.

    Returns
    -------
    TrialGrid

    Raises
    ------
    ValueError
        If the trial was not aligned across trials, or if the map puts a
        source more than :data:`MAX_SOURCE_OFFSET_PX` from its pixel in the
        moved subproblem (the map stretches or bends too much across a
        subproblem for one shift; solve on the consensus grid instead).
    """
    warp: TrialWarp = plan.warp
    if not warp.aligned[trial]:
        raise ValueError(f"trial {trial} was not aligned across trials")
    rows, cols = plan.selected_pixels()
    r_t, c_t = warp.coords(
        trial, rows.astype(np.float64), cols.astype(np.float64)
    )
    r_t = np.asarray(r_t, dtype=np.float64).ravel()
    c_t = np.asarray(c_t, dtype=np.float64).ravel()
    src_r = np.asarray(plan.source_rows, dtype=np.float64)
    src_c = np.asarray(plan.source_cols, dtype=np.float64)
    s_r, s_c = warp.coords(trial, src_r, src_c)
    s_r = np.asarray(s_r, dtype=np.float64).ravel() - src_r
    s_c = np.asarray(s_c, dtype=np.float64).ravel() - src_c
    n_groups = len(plan.groups)
    shifts = np.zeros((n_groups, 2), dtype=np.int64)
    residual = np.zeros(n_groups)
    moved_r, moved_c = [], []
    for g, grp in enumerate(plan.groups):
        pix = np.asarray(grp.pixels, dtype=np.int64)
        src = np.asarray(grp.sources, dtype=np.int64)
        mean = np.array([s_r[src].mean(), s_c[src].mean()])
        s = matlab_round(mean).astype(np.int64)
        shifts[g] = s
        off = np.sqrt((s_r[src] - s[0]) ** 2 + (s_c[src] - s[1]) ** 2)
        residual[g] = float(off.max())
        if residual[g] > MAX_SOURCE_OFFSET_PX:
            k = int(src[np.argmax(off)])
            raise ValueError(
                f"trial {trial}: the across-trial map puts source {k} "
                f"{residual[g]:.2f} px from its pixel in its subproblem "
                f"(group {g}) moved by one whole-pixel shift; more than "
                f"{MAX_SOURCE_OFFSET_PX:g} px: the map stretches or bends "
                "too much across the subproblem for the trial grid; solve "
                "on the consensus grid (solve_grid='consensus')"
            )
        moved_r.append(rows[pix] + s[0])
        moved_c.append(cols[pix] + s[1])
    points, index = _union(moved_r, moved_c)
    groups = [
        SubproblemGroup(
            sources=grp.sources,
            pixels=index[g],
            row0=int(grp.row0 + shifts[g, 0]),
            col0=int(grp.col0 + shifts[g, 1]),
            sel=grp.sel,
            sources_rc=grp.sources_rc,
        )
        for g, grp in enumerate(plan.groups)
    ]
    sources_rc = np.column_stack([plan.source_rows, plan.source_cols]).astype(
        np.int64
    )
    for g, grp in enumerate(plan.groups):
        sources_rc[np.asarray(grp.sources, dtype=np.int64)] += shifts[g]
    return TrialGrid(
        trial=int(trial),
        shifts=shifts,
        points=points,
        groups=groups,
        sources_rc=sources_rc,
        mapped_rc=np.column_stack([r_t, c_t]),
        residual=residual,
    )


def _union(rows: list, cols: list) -> tuple:
    """The distinct pixels of several lists in column-major order, and each
    list's pixels as indices into them."""
    if not rows:
        return np.zeros((0, 2), dtype=np.int64), []
    all_r, all_c = np.concatenate(rows), np.concatenate(cols)
    if all_r.size == 0:
        return np.zeros((0, 2), dtype=np.int64), [
            np.zeros(0, np.int64) for _ in rows
        ]
    r0, c0 = int(all_r.min()), int(all_c.min())
    n_r = int(all_r.max()) - r0 + 1
    key = (all_c - c0) * n_r + (all_r - r0)
    keys, inverse = np.unique(key, return_inverse=True)
    points = np.column_stack([keys % n_r + r0, keys // n_r + c0])
    ends = np.cumsum([0] + [len(r) for r in rows])
    index = [
        inverse[a:b].astype(np.int64) for a, b in zip(ends[:-1], ends[1:])
    ]
    return points.astype(np.int64), index


def footprints_to_consensus(
    footprints: np.ndarray, grid: TrialGrid, plan: ExtractionPlan
) -> np.ndarray:
    """A trial's footprints, fitted on its own grid, in the consensus frame.

    Each source's footprint is an image on its subproblem's crop in the
    trial; every selected pixel of the subproblem in the consensus frame
    takes the bilinear interpolation of that image at the map's position of
    the pixel in the trial (zero outside the crop's pixels). Each footprint
    is then scaled to keep its sum (the solver's footprints sum to 1), so
    the traces keep their units.

    Parameters
    ----------
    footprints : ndarray of shape (grid.n_pixels, sources)
        The trial's assembled footprints (NaN outside each source's
        subproblem).
    grid : TrialGrid
        The layout the trial was solved on.
    plan : ExtractionPlan
        The path's plan.

    Returns
    -------
    ndarray of float32, shape (plan.n_pixels, sources)
        NaN outside each source's subproblem, as the consensus grid's
        footprints.
    """
    fp = np.asarray(footprints)
    if fp.shape[0] != grid.n_pixels:
        raise ValueError(
            f"the footprints have {fp.shape[0]} pixels; the trial's grid "
            f"has {grid.n_pixels}"
        )
    out = np.full((plan.n_pixels, fp.shape[1]), np.nan, dtype=np.float32)
    for cons, moved in zip(plan.groups, grid.groups):
        src = np.asarray(moved.sources, dtype=np.int64)
        if src.size == 0:
            continue
        h = fp[np.ix_(moved.pixels, src)].astype(np.float64)
        known = ~np.isnan(h).all(0)
        h = np.nan_to_num(h)
        n_r, n_c = moved.sel.shape
        # the crop with a border of zeros, one image per source
        img = np.zeros((src.size, n_r + 2, n_c + 2))
        pr = grid.points[moved.pixels, 0] - moved.row0 + 1
        pc = grid.points[moved.pixels, 1] - moved.col0 + 1
        img[:, pr, pc] = h.T
        at = grid.mapped_rc[np.asarray(cons.pixels, dtype=np.int64)]
        u = np.clip(at[:, 0] - moved.row0 + 1, 0.0, n_r + 1.0)
        v = np.clip(at[:, 1] - moved.col0 + 1, 0.0, n_c + 1.0)
        vals = _bilinear(img, u, v)
        total, now = h.sum(0), vals.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.where(now > 0, total / now, 1.0)
        vals = (vals * scale[:, None]).T
        vals[:, ~known] = np.nan
        out[np.ix_(np.asarray(cons.pixels, dtype=np.int64), src)] = vals
    return out


def _bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Images ``(n, rows, cols)`` at points ``(u, v)`` inside them."""
    n_r, n_c = img.shape[1:]
    r0 = np.minimum(np.floor(u).astype(np.int64), n_r - 2)
    c0 = np.minimum(np.floor(v).astype(np.int64), n_c - 2)
    fr, fc = u - r0, v - c0
    return (
        img[:, r0, c0] * ((1 - fr) * (1 - fc))
        + img[:, r0, c0 + 1] * ((1 - fr) * fc)
        + img[:, r0 + 1, c0] * (fr * (1 - fc))
        + img[:, r0 + 1, c0 + 1] * (fr * fc)
    )


def masks_to_trial(masks, warp: TrialWarp, trial: int, shape: tuple) -> list:
    """Masks of the consensus frame moved into a trial's registered frame.

    The inverse of
    :func:`~giant_python.extraction.raster.alignment.mask_to_consensus`:
    each pixel of the trial's frame takes the mask's value at the nearest
    pixel of its consensus position (through the inverse of the map; a
    shift moves the mask by the rounded shift).

    Parameters
    ----------
    masks : sequence of ndarray of bool, shape (rows, cols)
        Masks in the consensus frame.
    warp : TrialWarp
        The across-trial alignment.
    trial : int
        Zero-based trial (aligned).
    shape : (int, int)
        Shape of the output masks (the trial's frame, top left in place).

    Returns
    -------
    list of ndarray of bool, shape ``shape``
    """
    out = []
    for mask in masks:
        mask = np.asarray(mask, dtype=bool)
        moved = np.zeros(shape, dtype=bool)
        rr, cc = np.nonzero(mask)
        if rr.size == 0:
            out.append(moved)
            continue
        if warp.is_shift(trial):
            s = matlab_round(warp.shift(trial)).astype(np.int64)
            r, c = rr + s[0], cc + s[1]
            ok = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
            moved[r[ok], c[ok]] = True
            out.append(moved)
            continue
        # the mask's pixels' positions in the trial bound the window
        r_t, c_t = warp.coords(
            trial, rr.astype(np.float64), cc.astype(np.float64)
        )
        lo_r = max(0, int(np.floor(np.min(r_t))) - 1)
        lo_c = max(0, int(np.floor(np.min(c_t))) - 1)
        hi_r = min(shape[0], int(np.ceil(np.max(r_t))) + 2)
        hi_c = min(shape[1], int(np.ceil(np.max(c_t))) + 2)
        if hi_r <= lo_r or hi_c <= lo_c:
            out.append(moved)
            continue
        wr = np.arange(lo_r, hi_r, dtype=np.float64)[:, None]
        wc = np.arange(lo_c, hi_c, dtype=np.float64)[None, :]
        x_r, x_c = warp.inverse(trial, *np.broadcast_arrays(wr, wc))
        # MATLAB rounds its one-based (positive) coordinates: halves go up
        x_r = np.floor(x_r + 0.5).astype(np.int64)
        x_c = np.floor(x_c + 0.5).astype(np.int64)
        ok = (
            (x_r >= 0)
            & (x_r < mask.shape[0])
            & (x_c >= 0)
            & (x_c < mask.shape[1])
        )
        win = np.zeros(ok.shape, dtype=bool)
        win[ok] = mask[x_r[ok], x_c[ok]]
        moved[lo_r:hi_r, lo_c:hi_c] = win
        out.append(moved)
    return out
