"""A SLAP2 trial's raw data at the selected pixels, at the extraction rate.

Port of the SLAP2 branch of ``loadTrial`` in GIAnT-MATLAB's
``processAllTrials_Async.m`` and of its helpers ``slap2FramesAt``,
``rawFramesAt``, ``compensationPad``, ``onlineCompensation`` and
``blockCompensation``, with the ``'Pixels'`` mode of ``slap2FrameSums``:

* frames are formed from the raw file at ``analyze_hz`` (200 Hz), with
  getImages' exponential kernel but without invalid samples, only at the
  raster pixels that the wanted points' bilinear neighbours fall on;
* each wanted point of the consensus frame is read at its position in the
  trial's registered frame (the across-trial warp) plus the within-trial
  motion, which is the registration's motion interpolated to the frames;
* with registration's ``compensateOnline``, frames are formed with the
  online motion correction undone, block by block;
* besides the data and its variance multiplier at the selected pixels, the
  loader returns the global trace (labeled pixels) and the user-ROI traces
  (plain and denoised by a rank-3 SVD, ``nansvd.m``).

Conventions: images are ``(rows, cols)`` in MATLAB's orientation; lines are
one-based, as in MATLAB and :mod:`giant_python.registration`; pixel and
frame indices are zero-based. Selected pixels are listed in MATLAB's
column-major order of the map (``find(selPix)``). The warp takes and returns
zero-based ``(row, col)`` coordinates.
"""

from __future__ import annotations

import math
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional, Sequence

import h5py
import numpy as np
import torch
from scipy.interpolate import PchipInterpolator

from ...io.slap2_raster import RasterLayout, raster_layout, read_raster_samples
from ...registration.compensation import LineOffsets, line_offsets
from ...registration.frames import PixelFormer
from ...registration.multiroi import matlab_colon
from ...registration.outputs import matlab_round
from .localization import matlab_prctile

SVD_COMPONENTS = 3
"""Rank of the user-ROI denoising (``nansvd(D, 3, 10, nanThresh)``)."""

SVD_ITERATIONS = 10
"""Most gap-filling iterations of the user-ROI denoising."""


@dataclass(frozen=True)
class TrialLoadParams:
    """The parameters of SILo that the trial loader uses.

    Attributes
    ----------
    analyze_hz : float
        Frame rate of the extraction (``analyzeHz``).
    activity_channel : int
        One-based channel of the activity signal (``activityChannel``); it
        becomes channel 0 of the data.
    photon_scale : float
        Raw counts per unit of the data (``photonScale``; 25 for SLAP2).
    baseline_window_s : float
        Window of the moving mean that fills missing samples
        (``baselineWindow_Glu_s``; ``baselineWindow_samps`` frames).
    nan_thresh : float
        Largest share of missing samples of a pixel or frame used in the
        user-ROI denoising (``nanThresh``).
    block_frames : int
        Frames formed around one online offset (MATLAB's ``blocksize``).
    """

    analyze_hz: float = 200.0
    activity_channel: int = 1
    photon_scale: float = 25.0
    baseline_window_s: float = 4.0
    nan_thresh: float = 0.33
    block_frames: int = 600

    @property
    def baseline_window_samps(self) -> float:
        """``baselineWindow_samps``: the fill window in frames."""
        return self.baseline_window_s * self.analyze_hz

    def ordered_channels(self, num_channels: int) -> list:
        """Zero-based file channels in analysis order (activity first)."""
        a = int(self.activity_channel)
        if not 1 <= a <= num_channels:
            raise ValueError(
                f"activity_channel {a} is not one of the {num_channels} "
                "channels (1-based)"
            )
        return list(range(a - 1, num_channels)) + list(range(a - 1))


@dataclass(frozen=True)
class RigidShift:
    """An across-trial shift: the default warp of a trial.

    Maps the consensus position ``(row, col)`` to ``(row + row_shift,
    col + col_shift)`` of the trial's registered frame (SILo's
    ``motOutput``). The loader adds a shift to the frames' motion, as MATLAB
    does; any other callable is evaluated at every wanted pixel.
    """

    row: float = 0.0
    col: float = 0.0

    def __call__(self, rows, cols):
        """Trial-frame position of consensus points."""
        return np.asarray(rows) + self.row, np.asarray(cols) + self.col


@dataclass(frozen=True)
class OnlineCompensation:
    """How frames are formed with the online motion correction undone.

    Attributes
    ----------
    offsets : LineOffsets
        Online offsets of the recording's lines.
    ref_offset : ndarray
        Header offset ``[x, y]`` whose lines stay in place.
    mapping : ndarray
        ``2 x 2`` matrix taking an offset difference ``[dx; dy]`` to the
        ``[row; col]`` displacement of the samples.
    """

    offsets: LineOffsets
    ref_offset: np.ndarray
    mapping: np.ndarray

    def displacement(self, lines):
        """Integer ``(rows, cols)`` displacement of the samples of lines.

        Parameters
        ----------
        lines : ndarray of int
            One-based lines (any shape; clipped to the recording).

        Raises
        ------
        ValueError
            If the mapping gives a displacement that is not whole pixels.
        """
        idx = np.clip(np.asarray(lines), 1, self.offsets.x.size) - 1
        dx = self.offsets.x[idx] - self.ref_offset[0]
        dy = self.offsets.y_eff[idx] - self.ref_offset[1]
        m = self.mapping
        dr = m[0, 0] * dx + m[0, 1] * dy
        dc = m[1, 0] * dx + m[1, 1] * dy
        if np.any(dr != np.round(dr)) or np.any(dc != np.round(dc)):
            raise ValueError("the mapping must give integer shifts")
        return dr.astype(np.int64), dc.astype(np.int64)


@dataclass
class TrialAlignment:
    """What the loader reads from a trial's ``*_ALIGNMENTDATA.h5``.

    Attributes
    ----------
    ds_frames : ndarray
        One-based line of each registered frame (``DSframes``).
    motion_r, motion_c : ndarray
        Row and column motion of each registered frame.
    align_hz : float
        Frame rate of the registration.
    view_r, view_c : ndarray
        Row and column of the trimmed raster image (one-based, before
        motion) that each row and column of the registered frame reads
        (``slap2.viewR(:, 1)``, ``slap2.viewC(1, :)``).
    trim_rows, trim_cols : ndarray of int
        One-based DMD rows and columns of the trimmed raster image.
    compensation : dict or None
        The ``slap2/onlineCompensation`` record when registration undid the
        online motion correction (``refOffset``, ``frameRefX``,
        ``frameRefY``, ``mapping``, ``yLead``).
    """

    ds_frames: np.ndarray
    motion_r: np.ndarray
    motion_c: np.ndarray
    align_hz: float
    view_r: np.ndarray
    view_c: np.ndarray
    trim_rows: np.ndarray
    trim_cols: np.ndarray
    compensation: Optional[dict] = None

    @property
    def view_shape(self) -> tuple:
        """``(rows, cols)`` of the trial's registered frame."""
        return self.view_r.size, self.view_c.size


def _matlab_2d(ds: h5py.Dataset, row_major: bool) -> np.ndarray:
    """A 2-D dataset in MATLAB's orientation."""
    a = np.asarray(ds[()], dtype=np.float64)
    return a if row_major else a.T


def _compensation_record(grp, row_major: bool) -> Optional[dict]:
    """The enabled ``onlineCompensation`` record of an alignment file."""
    if "onlineCompensation" not in grp:
        return None
    oc = grp["onlineCompensation"]
    if "enabled" not in oc or not np.asarray(oc["enabled"]).any():
        return None
    return {
        "refOffset": np.asarray(oc["refOffset"], np.float64).ravel(),
        "frameRefX": np.asarray(oc["frameRefX"], np.float64).ravel(),
        "frameRefY": np.asarray(oc["frameRefY"], np.float64).ravel(),
        "mapping": _matlab_2d(oc["mapping"], row_major),
        "yLead": np.asarray(oc["yLead"], np.float64).ravel(),
    }


def read_alignment(path) -> TrialAlignment:
    """Read the loader's part of a trial's alignment file.

    Parameters
    ----------
    path : str or Path
        ``<trial>_ALIGNMENTDATA.h5`` of MultiRoiRegistration (MATLAB or
        :mod:`giant_python.registration`). The per-frame variance factor is
        not read.

    Returns
    -------
    TrialAlignment
    """
    with h5py.File(path, "r") as f:
        row_major = "row_major" in f and bool(np.asarray(f["row_major"]).any())
        s2 = f["slap2"]
        return TrialAlignment(
            ds_frames=np.asarray(f["DSframes"], np.float64).ravel(),
            motion_r=np.asarray(f["motionDSr"], np.float64).ravel(),
            motion_c=np.asarray(f["motionDSc"], np.float64).ravel(),
            align_hz=float(np.asarray(f["alignHz"]).ravel()[0]),
            view_r=_matlab_2d(s2["viewR"], row_major)[:, 0].copy(),
            view_c=_matlab_2d(s2["viewC"], row_major)[0, :].copy(),
            trim_rows=np.asarray(s2["trimRows"]).ravel().astype(np.int64),
            trim_cols=np.asarray(s2["trimCols"]).ravel().astype(np.int64),
            compensation=_compensation_record(s2, row_major),
        )


def online_compensation(
    alignment: TrialAlignment,
    layout: RasterLayout,
    line_range,
    kernel_lines: int,
):
    """The compensation registration used and the motion to interpolate.

    Port of ``onlineCompensation``. If registration undid the online motion
    correction, each registered frame's motion is relative to the DMD
    coordinates of its own offset; adding that offset (mapped to rows and
    columns, relative to the reference) gives the continuous tissue motion,
    which is what can be interpolated in time.

    Parameters
    ----------
    alignment : TrialAlignment
        The trial's alignment.
    layout : RasterLayout
        The recording's layout.
    line_range : (float, float)
        First and last line of the trial and its registered frames.
    kernel_lines : int
        The wider of the registration and extraction kernels, in lines.

    Returns
    -------
    comp : OnlineCompensation or None
        None when registration did not compensate.
    motion_r, motion_c : ndarray
        Motion of the registered frames: as saved, or with compensation
        relative to the frame of the reference offset.
    """
    motion_r = alignment.motion_r.astype(np.float64).copy()
    motion_c = alignment.motion_c.astype(np.float64).copy()
    oc = alignment.compensation
    if oc is None:
        return None, motion_r, motion_c
    margin = 2 * max(3 * kernel_lines, layout.lines_per_cycle) + 8
    lead = tuple(int(v) for v in oc["yLead"])
    offsets = line_offsets(
        layout, (line_range[0] - margin, line_range[1] + margin), lead
    )
    mapping = np.asarray(oc["mapping"], np.float64)
    ref = np.asarray(oc["refOffset"], np.float64)
    d = mapping @ (np.stack([oc["frameRefX"], oc["frameRefY"]]) - ref[:, None])
    comp = OnlineCompensation(offsets, ref, mapping)
    return comp, motion_r + d[0], motion_c + d[1]


def block_compensation(comp: Optional[OnlineCompensation], lines):
    """Compensation of a block of frames, around the block's median offset.

    Port of ``blockCompensation``: forming a block around its own typical
    offset keeps the motion to apply, ``motion - shift``, as small as
    without compensation.

    Parameters
    ----------
    comp : OnlineCompensation or None
        The trial's compensation.
    lines : ndarray of int
        One-based lines of the block's frames.

    Returns
    -------
    comp_block : OnlineCompensation or None
        ``comp`` with the block's reference offset.
    shift : (float, float)
        ``(rows, cols)`` of the block's frame relative to the trial's
        reference frame; ``(0, 0)`` without compensation.
    """
    if comp is None:
        return None, (0.0, 0.0)
    idx = np.asarray(lines, dtype=np.int64) - 1
    ref = matlab_round(
        [np.median(comp.offsets.x[idx]), np.median(comp.offsets.y_eff[idx])]
    )
    shift = comp.mapping @ (ref - comp.ref_offset)
    return replace(comp, ref_offset=ref), (float(shift[0]), float(shift[1]))


def compensation_pad(
    comp: Optional[OnlineCompensation], layout: RasterLayout, times, dt: int
) -> int:
    """Largest shift that compensation applies to the lines of some frames.

    Port of ``compensationPad``: samples of lines on another offset than
    the reference move by up to this many pixels, possibly past the trimmed
    raster. 0 without compensation.
    """
    if comp is None:
        return 0
    times = np.asarray(times, dtype=np.int64)
    reach = max(3 * dt, layout.lines_per_cycle)
    lines = np.arange(
        max(1, int(times.min()) - reach),
        min(comp.offsets.x.size, int(times.max()) + reach) + 1,
    )
    dr, dc = comp.displacement(lines)
    return int(max(np.abs(dr).max(), np.abs(dc).max()))


class RasterPixelFrames:
    """getImages' frames of a block of times at chosen raster pixels.

    Port of ``rawFramesAt`` (``slap2FrameSums`` with ``Boundary =
    'exclusive'``, ``MaskInvalid = true`` and ``Pixels``): only the raster
    slots that can land on the wanted pixels are read and summed. Unlike
    getImages, invalid samples are left out, so a frame's value is the mean
    of each channel's valid samples.

    Parameters
    ----------
    layout : RasterLayout
        The recording's layout.
    ids : ndarray of int
        One-based raster pixel IDs to form (sorted, unique). With
        compensation they are positions in the frame of the reference
        offset.
    times : ndarray of int
        One-based line times that will be requested (sets what is read).
    dt : int
        Kernel time constant in lines.
    channels : sequence of int
        Zero-based file channels, in the order of the output.
    comp : OnlineCompensation, optional
        Undo the online motion correction around ``comp.ref_offset``.
    device : str or torch.device
        Where sums are formed.
    band : int
        Cycles per matrix product (see :class:`FrameFormer`).
    """

    def __init__(
        self,
        layout: RasterLayout,
        ids: np.ndarray,
        times,
        dt: int,
        channels: Sequence[int],
        comp: Optional[OnlineCompensation] = None,
        device="cpu",
        band: int = 16,
    ):
        """Read the samples of the slots that land on the wanted pixels."""
        times = np.asarray(times, dtype=np.int64)
        self.dt = int(dt)
        self.channels = list(channels)
        self.comp = comp
        self.n_ids = int(ids.size)
        self._shape = (layout.n_rows, layout.n_cols)
        self._lut = np.full(layout.n_rows * layout.n_cols + 1, -1, np.int64)
        self._lut[ids] = np.arange(ids.size)
        p = layout.lines_per_cycle
        reach = max(3 * self.dt, p)
        lo = max(1, int(times.min()) - reach)
        hi = min(layout.total_lines, int(times.max()) + reach)
        shifts = np.zeros((1, 2), dtype=np.int64)
        if comp is not None:
            dr, dc = comp.displacement(np.arange(lo, hi + 1))
            shifts = np.unique(np.stack([dr, dc], 1), axis=0)
        slot_r = (layout.slot_pixel - 1) // layout.n_cols
        slot_c = (layout.slot_pixel - 1) % layout.n_cols
        keep = np.zeros(slot_r.size, dtype=bool)
        for dr, dc in shifts:
            keep |= self._landing(slot_r, slot_c, int(dr), int(dc)) >= 0
        sub = replace(
            layout,
            slot_line=layout.slot_line[keep],
            slot_pos=layout.slot_pos[keep],
            slot_pixel=layout.slot_pixel[keep],
        )
        c0, c1 = (lo - 1) // p, (hi - 1) // p + 1
        samples = read_raster_samples(sub, (c0, c1))
        self.former = PixelFormer(
            sub, samples, c0, 1, device, exclusive=True, band=band
        )

    def _landing(self, rows, cols, dr: int, dc: int) -> np.ndarray:
        """Index into ``ids`` of pixels moved by ``(dr, dc)``; -1 if none."""
        n_rows, n_cols = self._shape
        r, c = rows + dr, cols + dc
        inside = (r >= 0) & (r < n_rows) & (c >= 0) & (c < n_cols)
        return self._lut[np.where(inside, r * n_cols + c + 1, 0)]

    def frames(self, times):
        """Frame values and weights at the wanted pixels.

        Parameters
        ----------
        times : array_like of int
            One-based line times, among those given at construction.

        Returns
        -------
        values : torch.Tensor of float64
            ``(channels, n_ids + 1, frames)``: the mean of each channel's
            valid samples, NaN where a frame has none.
        weight : torch.Tensor of float64
            ``(n_ids + 1, frames)``: the summed kernel weight of the first
            channel's valid samples (0 where there are none).

        The last row of both is NaN: it stands for a pixel outside the
        image when frames are interpolated.
        """
        times = np.asarray(times, dtype=np.int64).ravel()
        f = self.former
        dev = f.device
        size = (len(self.channels), self.n_ids + 1, times.size)
        if self.comp is None:
            # one group: each sampled pixel's sums go to its own row
            ((_, _, s_num, s_den),) = f.pixel_sums(
                times, self.dt, self.channels
            )
            rows = torch.from_numpy(self._landing(f.rows, f.cols, 0, 0))
            rows = rows.to(dev)
            values = torch.full(
                size, float("nan"), dtype=torch.float64, device=dev
            )
            weight = torch.zeros(size[1:], dtype=torch.float64, device=dev)
            values.index_copy_(1, rows, s_num.div_(s_den))
            weight.index_copy_(0, rows, s_den[0])
            weight[self.n_ids] = float("nan")
            return values, weight
        num = torch.zeros(size, dtype=torch.float64, device=dev)
        den = torch.zeros_like(num)
        for (dr, dc), sel, s_num, s_den in f.pixel_sums(
            times, self.dt, self.channels, self.comp.displacement
        ):
            dest = self._landing(f.rows, f.cols, dr, dc)
            keep = torch.from_numpy(np.flatnonzero(dest >= 0)).to(dev)
            rows = torch.from_numpy(dest[dest >= 0]).to(dev)
            cols = torch.from_numpy(sel).to(dev)
            for total, part in ((num, s_num), (den, s_den)):
                if keep.numel() < part.shape[1]:
                    part = part.index_select(1, keep)
                if sel.size < times.size:  # frames without this shift: 0
                    wide = torch.zeros(
                        part.shape[:2] + (times.size,),
                        dtype=torch.float64,
                        device=dev,
                    )
                    part = wide.index_copy_(2, cols, part)
                total.index_add_(1, rows, part)
        values = num.div_(den)
        weight = den[0]
        values[:, self.n_ids] = float("nan")
        weight[self.n_ids] = float("nan")
        return values, weight


def _weighted_sum(values, weight, neighbours):
    """Weighted bilinear sums of frames at points.

    ``values`` is ``(channels, pixels, frames)`` and ``weight`` ``(pixels,
    frames)``. ``neighbours`` yields, for each of the four neighbours, the
    row of ``values`` and ``weight`` that holds it for every point and
    frame, ``(points, frames)``, and its bilinear coefficient (broadcast to
    that shape). The coefficient times the neighbour's weight is its share;
    the result is NaN where a neighbour is.

    Returns
    -------
    y : torch.Tensor
        ``(channels, points, frames)`` values.
    fq : torch.Tensor
        ``(points, frames)`` variance multiplier: the reciprocal of the
        summed shares (NaN where that is 0 or undefined).
    """
    n_f = values.shape[2]
    cols = torch.arange(n_f, device=values.device)[None, :]
    num = den = None
    for k, (row, coef) in enumerate(neighbours):
        # torch.take walks the flattened array in order, which is several
        # times faster than gather along the first axis
        flat = row * n_f + cols
        wf = torch.take(weight, flat).mul_(coef)
        if num is None:
            num = torch.empty(
                (values.shape[0],) + tuple(flat.shape),
                dtype=torch.float64,
                device=values.device,
            )
        for ch in range(values.shape[0]):
            term = torch.take(values[ch], flat).mul_(wf)
            if k == 0:
                num[ch] = term
            else:
                num[ch].add_(term)
        den = wf if den is None else den.add_(wf)
    num.div_(den)
    den.reciprocal_()
    return num, den.masked_fill_(torch.isinf(den), float("nan"))


def _interpolate(values, weight, tpos, shape, r_d, c_d, m_r, m_c):
    """Bilinear interpolation of frames at moving points (``interpFrames``).

    Point ``q`` is read in frame ``f`` at row ``r_d[q] + m_r[f]`` and column
    ``c_d[q] + m_c[f]`` (one-based) of the trimmed image of ``shape``. Each
    of the four neighbours is weighted by its bilinear coefficient times its
    weight; a neighbour outside the image, or without valid samples, makes
    the result NaN. ``tpos`` maps the one-based row-major pixel index (0:
    outside) to the row of ``values`` (channels, pixels, frames) and
    ``weight`` (pixels, frames).

    Returns
    -------
    y : torch.Tensor
        ``(channels, points, frames)`` values.
    fq : torch.Tensor
        ``(points, frames)`` variance multiplier, the reciprocal of the
        summed weights (NaN where that is 0 or undefined).
    """
    n1, n2 = shape
    r = r_d[:, None] + m_r[None, :]
    c = c_d[:, None] + m_c[None, :]
    rf, cf = torch.floor(r), torch.floor(c)
    fr, fc = r - rf, c - cf
    rc, cc = torch.ceil(r), torch.ceil(c)

    def neighbours():
        """Rows and coefficients of the four neighbours."""
        for rk, ck, w in (
            (rf, cf, (1 - fc) * (1 - fr)),
            (rf, cc, fc * (1 - fr)),
            (rc, cf, (1 - fc) * fr),
            (rc, cc, fc * fr),
        ):
            ok = (rk >= 1) & (rk <= n1) & (ck >= 1) & (ck <= n2)
            yield tpos[(((rk - 1) * n2 + ck) * ok).long()], w

    return _weighted_sum(values, weight, neighbours())


class _WholePixels:
    """Neighbour lookup for points at whole pixels of the trimmed image.

    With points at whole pixels (a rigid across-trial shift, which is added
    to the motion), all points of a frame share their neighbours' offsets
    and bilinear coefficients. The pixel-to-row table is padded so that any
    neighbour of any point within the block's motion has a place in it
    (those outside the image map to the NaN row): a neighbour's row is then
    one table lookup at the point's place plus the frame's offset.

    Parameters
    ----------
    tpos : ndarray of int
        Row of each pixel of the trimmed image (row-major, after a leading
        entry for "outside").
    shape : (int, int)
        The trimmed image.
    r_d, c_d : ndarray
        The points' one-based rows and columns (whole numbers).
    m_r, m_c : ndarray
        The block's motion.
    device : torch.device
        Where frames are interpolated.
    """

    def __init__(self, tpos, shape, r_d, c_d, m_r, m_c, device):
        """Pad the table by the points' and the motion's reach."""
        n1, n2 = shape
        r_i, c_i = r_d.astype(np.int64), c_d.astype(np.int64)
        reach_r = (math.floor(m_r.min()), math.ceil(m_r.max()))
        reach_c = (math.floor(m_c.min()), math.ceil(m_c.max()))
        pad = max(
            0,
            1 - (int(r_i.min()) + reach_r[0]),
            int(r_i.max()) + reach_r[1] - n1,
            1 - (int(c_i.min()) + reach_c[0]),
            int(c_i.max()) + reach_c[1] - n2,
        )
        self.pad = pad
        self.width = n2 + 2 * pad
        table = np.full((n1 + 2 * pad, self.width), tpos[0], dtype=np.int64)
        table[pad : pad + n1, pad : pad + n2] = tpos[1:].reshape(n1, n2)
        self.table = torch.from_numpy(table.ravel()).to(device)
        base = (r_i - 1 + pad) * self.width + (c_i - 1 + pad)
        self.base = torch.from_numpy(base).to(device)[:, None]

    def interpolate(self, values, weight, m_r, m_c):
        """:func:`_interpolate` for these points (equal to rounding).

        ``m_r`` and ``m_c`` are NumPy arrays of the frames' motion.
        """
        dev = values.device
        lo_r, lo_c = np.floor(m_r), np.floor(m_c)
        hi_r, hi_c = np.ceil(m_r), np.ceil(m_c)
        fr, fc = m_r - lo_r, m_c - lo_c
        coef = np.stack(
            [(1 - fc) * (1 - fr), fc * (1 - fr), (1 - fc) * fr, fc * fr]
        )
        off = np.stack(
            [
                lo_r * self.width + lo_c,
                lo_r * self.width + hi_c,
                hi_r * self.width + lo_c,
                hi_r * self.width + hi_c,
            ]
        ).astype(np.int64)
        coef = torch.from_numpy(coef).to(dev)
        off = torch.from_numpy(off).to(dev)
        neighbours = (
            (self.table[self.base + off[k][None, :]], coef[k])
            for k in range(4)
        )
        return _weighted_sum(values, weight, neighbours)


def _whole_pixel_frames(m_r: np.ndarray, m_c: np.ndarray) -> bool:
    """Whether whole-pixel points share their neighbours in these frames.

    Not when a motion is within rounding of a whole pixel without being
    one: MATLAB's ``floor(point + motion)`` then depends on the point.
    """
    for m in (m_r, m_c):
        f = m - np.floor(m)
        if np.any((f != 0) & ((f < 1e-9) | (f > 1 - 1e-9))):
            return False
    return True


def _mark_pixels(shape, r_d, c_d, m_r, m_c) -> np.ndarray:
    """Pixels of the trimmed image within a block's motion of each point."""
    n1, n2 = shape
    mark = np.zeros(shape, dtype=bool)
    r0 = np.floor(r_d + m_r.min()).astype(np.int64)
    c0 = np.floor(c_d + m_c.min()).astype(np.int64)
    for dr in range(int(math.ceil(m_r.max() - m_r.min())) + 2):
        for dc in range(int(math.ceil(m_c.max() - m_c.min())) + 2):
            r, c = r0 + dr, c0 + dc
            ok = (r >= 1) & (r <= n1) & (c >= 1) & (c <= n2)
            mark[r[ok] - 1, c[ok] - 1] = True
    return mark


def frames_at(
    layout: RasterLayout,
    channels: Sequence[int],
    times,
    dt: int,
    trim_rows: np.ndarray,
    trim_cols: np.ndarray,
    r_d: np.ndarray,
    c_d: np.ndarray,
    m_r: np.ndarray,
    m_c: np.ndarray,
    comp: Optional[OnlineCompensation] = None,
    device="cpu",
    chunk: int = 128,
    band: int = 16,
    whole_pixels: bool = True,
):
    """Motion-corrected frames at points of the trimmed raster image.

    Port of ``slap2FramesAt``: the frames getImages returns (without its
    invalid samples), trimmed to ``trim_rows x trim_cols`` and interpolated
    as ``interpFrames`` does, at rows ``r_d + m_r[f]`` and columns ``c_d +
    m_c[f]`` of frame ``f``. Frames are formed only at the raster pixels
    that these points' neighbours can fall on.

    Parameters
    ----------
    layout : RasterLayout
        The recording's layout.
    channels : sequence of int
        Zero-based file channels, in the order of the output.
    times : array_like of int
        One-based line times of the frames.
    dt : int
        Kernel time constant in lines.
    trim_rows, trim_cols : ndarray of int
        One-based DMD rows and columns of the trimmed image (contiguous).
    r_d, c_d : ndarray
        One-based row and column of each point in the trimmed image.
    m_r, m_c : ndarray
        Motion of each frame. With ``comp``, motion estimated on frames
        formed around ``comp.ref_offset``.
    comp : OnlineCompensation, optional
        Form the frames with the online motion correction undone.
    device : str or torch.device
        Where frames are formed and interpolated.
    chunk : int
        Frames interpolated at a time.
    band : int
        Cycles per matrix product (see :class:`FrameFormer`).
    whole_pixels : bool
        Use the faster interpolation for points at whole pixels (the same
        to rounding).

    Yields
    ------
    frames : slice
        The frames of this part.
    y : torch.Tensor of float64
        ``(channels, points, frames)`` values.
    fq : torch.Tensor of float64
        ``(points, frames)`` variance multiplier.
    """
    times = np.asarray(times, dtype=np.int64).ravel()
    m_r = np.asarray(m_r, dtype=np.float64)
    m_c = np.asarray(m_c, dtype=np.float64)
    r_d, c_d = np.asarray(r_d, np.float64), np.asarray(c_d, np.float64)
    pad = compensation_pad(comp, layout, times, dt)
    if pad > 0:
        # samples moved past the trimmed raster are read too
        r1 = max(1, int(trim_rows[0]) - pad)
        c1 = max(1, int(trim_cols[0]) - pad)
        r_d = r_d + (int(trim_rows[0]) - r1)
        c_d = c_d + (int(trim_cols[0]) - c1)
        trim_rows = np.arange(r1, min(layout.n_rows, trim_rows[-1] + pad) + 1)
        trim_cols = np.arange(c1, min(layout.n_cols, trim_cols[-1] + pad) + 1)
    shape = (trim_rows.size, trim_cols.size)
    mark = _mark_pixels(shape, r_d, c_d, m_r, m_c)
    rt, ct = np.nonzero(mark)
    ids = trim_cols[ct] + layout.n_cols * (trim_rows[rt] - 1)
    raw = RasterPixelFrames(
        layout, ids, times, dt, channels, comp, device, band
    )
    dev = raw.former.device
    tpos = np.full(shape[0] * shape[1] + 1, ids.size, dtype=np.int64)
    tpos[np.flatnonzero(mark) + 1] = np.arange(ids.size)
    whole = None
    if (
        whole_pixels
        and np.all(r_d == np.round(r_d))
        and np.all(c_d == np.round(c_d))
    ):
        whole = _WholePixels(tpos, shape, r_d, c_d, m_r, m_c, dev)
    tpos = torch.from_numpy(tpos).to(dev)
    r_t, c_t = torch.from_numpy(r_d).to(dev), torch.from_numpy(c_d).to(dev)
    for a in range(0, times.size, chunk):
        part = slice(a, min(times.size, a + chunk))
        values, weight = raw.frames(times[part])
        if whole is not None and _whole_pixel_frames(m_r[part], m_c[part]):
            y, fq = whole.interpolate(values, weight, m_r[part], m_c[part])
        else:
            y, fq = _interpolate(
                values,
                weight,
                tpos,
                shape,
                r_t,
                c_t,
                torch.from_numpy(m_r[part]).to(dev),
                torch.from_numpy(m_c[part]).to(dev),
            )
        yield part, y, fq


def _linspace(a: float, b: float, n: int) -> np.ndarray:
    """MATLAB's ``linspace``: ``a + (0:n-1) * (b - a) / (n - 1)``."""
    if n == 1:
        return np.array([float(b)])
    y = a + (np.arange(n) * (b - a)) / (n - 1)
    y[0], y[-1] = a, b
    return y


def frame_lines(first_line: float, last_line: float, dt: float) -> np.ndarray:
    """One-based line of each extraction frame: ``ceil(first:dt:last)``."""
    return np.ceil(matlab_colon(first_line, dt, last_line)).astype(np.int64)


def block_edges(n_frames: int, block: int) -> np.ndarray:
    """First frame of each block, and ``n_frames`` (zero-based).

    MATLAB's ``round(linspace(1, nFrames + 1, nBlocks + 1))``.
    """
    n_blocks = int(math.ceil(n_frames / block))
    edges = matlab_round(_linspace(1, n_frames + 1, n_blocks + 1))
    return edges.astype(np.int64) - 1


def upsample_motion(ds_frames, motion, lines) -> np.ndarray:
    """Motion of the registered frames interpolated to other lines.

    MATLAB's ``interp1(DSframes, motion, lines, 'pchip', 'extrap')``; the
    motion of a single registered frame is used throughout. Registered
    frames without motion (NaN: frames without samples in
    :mod:`giant_python.registration`) are left out; MATLAB's registration
    gives every frame a motion.

    Raises
    ------
    ValueError
        If no registered frame has a motion.
    """
    ok = np.isfinite(motion)
    lines = np.asarray(lines, dtype=np.float64)
    if ok.sum() > 1:
        return PchipInterpolator(ds_frames[ok], motion[ok], extrapolate=True)(
            lines
        )
    if not ok.any():
        raise ValueError("the trial's registration has no motion")
    return np.full(lines.shape, float(motion[ok][0]))


def upsample_discard(discard, n_frames: int) -> np.ndarray:
    """Discarded registered frames at the extraction's frames.

    MATLAB's ``interp1(1:n, double(discard), linspace(1, n, nFrames)) > 0``:
    a frame is discarded if either registered frame around it is.
    """
    d = np.asarray(discard, dtype=bool).ravel()
    if d.size <= 1:
        return np.full(n_frames, bool(d.any()))
    x = _linspace(1.0, float(d.size), n_frames)
    lo = np.clip(np.floor(x).astype(np.int64), 1, d.size)
    hi = np.clip(np.ceil(x).astype(np.int64), 1, d.size)
    return d[lo - 1] | d[hi - 1]


def medfilt3(a: np.ndarray) -> np.ndarray:
    """MATLAB's ``medfilt2(a, [3 3])``: zeros outside the image, and NaN
    wherever the window holds a NaN."""
    p = np.pad(np.asarray(a, dtype=np.float64), 1)
    win = np.lib.stride_tricks.sliding_window_view(p, (3, 3))
    return np.median(win.reshape(a.shape + (9,)), axis=-1)


def labeled_pixels(mean_activity: np.ndarray) -> np.ndarray:
    """Pixels of the global trace: brighter than three times the 25th
    percentile of the median-filtered mean image of the activity channel."""
    lab = medfilt3(mean_activity)
    with np.errstate(invalid="ignore"):
        bright = lab > 3 * matlab_prctile(lab, 25)
    return ~np.isnan(mean_activity) & bright


def _fit(a: np.ndarray, shape, fill) -> np.ndarray:
    """Crop or pad the leading two axes to ``shape``, corner in place."""
    out = np.full(tuple(shape) + a.shape[2:], fill, dtype=a.dtype)
    r, c = min(shape[0], a.shape[0]), min(shape[1], a.shape[1])
    out[:r, :c] = a[:r, :c]
    return out


def _window_reach(window: float) -> tuple:
    """Frames a moving window reaches back and ahead: ``i - w/2 <= j < i +
    w/2``, as MATLAB places windows of even, odd or fractional length."""
    return int(math.floor(window / 2)), int(math.ceil(window / 2)) - 1


def movmean(a: np.ndarray, window: float) -> np.ndarray:
    """MATLAB ``smoothdata(a, 2, 'movmean', window, 'omitmissing')``.

    The mean of the values in each window along the last axis; the window
    shrinks at the ends; NaN where it holds no value.
    """
    back, ahead = _window_reach(window)
    ok = ~np.isnan(a)
    n = a.shape[-1]
    zero = np.zeros(a.shape[:-1] + (1,))
    hi = np.minimum(np.arange(n) + ahead + 1, n)
    lo = np.maximum(np.arange(n) - back, 0)
    if ok.all():  # the same sums and counts, without the bookkeeping
        s = np.concatenate([zero, np.cumsum(a, -1)], -1)
        return (s[..., hi] - s[..., lo]) / (hi - lo).astype(np.float64)
    s = np.concatenate([zero, np.cumsum(np.where(ok, a, 0.0), -1)], -1)
    cnt = np.concatenate([zero, np.cumsum(ok, -1, dtype=np.float64)], -1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (s[..., hi] - s[..., lo]) / (cnt[..., hi] - cnt[..., lo])


def _fill_passes(ok: np.ndarray, window: float) -> int:
    """Moving-mean passes until rows with these valid samples are full."""
    back, ahead = _window_reach(window)
    n = ok.shape[-1]
    hi = np.minimum(np.arange(n) + ahead + 1, n)
    lo = np.maximum(np.arange(n) - back, 0)
    passes = 0
    ok = ok[~ok.all(1)]
    while ok.size:
        cnt = np.concatenate(
            [np.zeros((ok.shape[0], 1), np.int32), np.cumsum(ok, 1, np.int32)],
            1,
        )
        ok = cnt[:, hi] > cnt[:, lo]
        passes += 1
        ok = ok[~ok.all(1)]
    return passes


def _start_at_zero(ok: np.ndarray) -> np.ndarray:
    """Valid samples of rows, a row without any starting from frame 0
    (MATLAB sets the first frame of such a pixel to 0 before filling)."""
    ok = ok.copy()
    ok[~ok.any(1), 0] = True
    return ok


def _nan_std(a: np.ndarray) -> np.ndarray:
    """MATLAB ``std(a, 0, 2, 'omitmissing')`` (N - 1; 0 for one value)."""
    ok = ~np.isnan(a)
    n = ok.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(ok, a, 0.0).sum(1) / n
        ss = (np.where(ok, a - m[:, None], 0.0) ** 2).sum(1)
        s = np.sqrt(ss / np.maximum(n - 1, 1))
    s[n == 0] = np.nan
    return s


def _threads() -> int:
    """Threads for row-wise NumPy work (which releases the GIL)."""
    return max(1, min(8, (os.cpu_count() or 2) // 2))


def _drop_outliers(part: np.ndarray, common: np.ndarray, window: float):
    """Set a block of rows' aberrantly negative samples to NaN, in place.

    Returns the moving-mean passes that fill these rows, and whether any
    row misses only the frames that ``common`` misses.
    """
    v = part.astype(np.float64)
    with np.errstate(invalid="ignore"):
        low = v < -2 * _nan_std(v)[:, None]
    part[low] = np.nan
    ok = _start_at_zero(~(np.isnan(v) | low))
    other = (ok != common).any(1)
    return _fill_passes(ok[other], window), not other.all()


def _fill_rows(part, finv_part, passes: int, window: float, fill_finv):
    """Fill a block of rows' missing samples, in place.

    ``finv_part`` (the rows' variance multiplier, or None for channels
    other than the activity channel) gets ``fill_finv`` where missing.
    """
    nans = np.isnan(part)
    if finv_part is not None:
        finv_part[nans] = fill_finv
    if not nans.any():
        return
    v = part.astype(np.float64)
    v[nans.all(1), 0] = 0.0
    for _ in range(passes):
        v = movmean(v, window)
    part[nans] = v[nans]


def finish_pixels(
    y: np.ndarray,
    finv: np.ndarray,
    discard: np.ndarray,
    window: float,
    rows: int = 512,
) -> None:
    """The end of ``loadTrial``, in place: discards, outliers, gap filling.

    Discarded frames and samples below minus twice the pixel's standard
    deviation become missing. Missing samples are filled by repeated moving
    means over ``window`` frames (as many passes over the whole array as its
    longest gap needs; a pixel without any sample starts from 0 at its
    first frame), and get 1000 times the mean variance multiplier where the
    activity channel is missing.

    Parameters
    ----------
    y : ndarray
        ``(channels, pixels, frames)`` data, NaN where missing.
    finv : ndarray
        ``(pixels, frames)`` variance multiplier.
    discard : ndarray of bool
        Discarded frames.
    window : float
        Length of the moving mean in frames.
    rows : int
        Pixels processed at a time.
    """
    n_ch, n_px, _ = y.shape
    y[:, :, discard] = np.nan
    blocks = [
        (ch, slice(a, a + rows))
        for ch in range(n_ch)
        for a in range(0, n_px, rows)
    ]
    # most pixels miss only the discarded frames: count their passes once
    common = _start_at_zero(~np.asarray(discard, dtype=bool)[None, :])
    common_passes = _fill_passes(common, window)
    total, count = 0.0, 0
    for a in range(0, n_px, rows):
        part = finv[a : a + rows]
        total += float(np.nansum(part, dtype=np.float64))
        count += int((~np.isnan(part)).sum())
    fill_finv = 1000.0 * total / count if count else np.nan
    with ThreadPoolExecutor(_threads()) as pool:
        found = list(
            pool.map(
                lambda b: _drop_outliers(y[b[0], b[1]], common, window),
                blocks,
            )
        )
        passes = max([n for n, _ in found], default=0)
        if any(shared for _, shared in found):
            passes = max(passes, common_passes)
        list(
            pool.map(
                lambda b: _fill_rows(
                    y[b[0], b[1]],
                    finv[b[1]] if b[0] == 0 else None,
                    passes,
                    window,
                    fill_finv,
                ),
                blocks,
            )
        )


def nansvd(
    d: np.ndarray, num_comps: int = 3, max_iter: int = 20, nan_frac=0.3
):
    """Truncated SVD of data with gaps (``nansvd.m``).

    Rows and columns with at least ``nan_frac`` of their values missing are
    left out; a moving mean over a fifth of the columns is subtracted; the
    remaining gaps are filled iteratively with the rank-``num_comps``
    estimate.

    Parameters
    ----------
    d : ndarray
        ``(pixels, timepoints)`` data, NaN where missing.
    num_comps : int
        Number of components.
    max_iter : int
        Most gap-filling iterations.
    nan_frac : float
        Share of missing values at which a row or column is left out.

    Returns
    -------
    u : ndarray
        ``(pixels, num_comps)`` left singular vectors, NaN in the rows left
        out and in components that the data are too small for.
    s : ndarray
        ``(num_comps,)`` singular values (0 for missing components).
    v : ndarray
        ``(timepoints, num_comps)`` right singular vectors.
    bg : ndarray
        ``(pixels, timepoints)`` subtracted moving mean (NaN where left
        out).
    """
    d = np.asarray(d, dtype=np.float64)
    missing = np.isnan(d)
    rows = np.ones(d.shape[0], dtype=bool)
    cols = np.ones(d.shape[1], dtype=bool)
    n_r = n_c = -1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # empty means
        while not (cols.sum() == n_c and rows.sum() == n_r):
            n_c, n_r = cols.sum(), rows.sum()
            rows = missing[:, cols].mean(1) < nan_frac
            cols = missing[rows, :].mean(0) < nan_frac
    u = np.full((d.shape[0], num_comps), np.nan)
    v = np.full((d.shape[1], num_comps), np.nan)
    s = np.zeros(num_comps)
    bg = np.full(d.shape, np.nan)
    x = d[np.ix_(rows, cols)]
    if x.size == 0:
        return u, s, v, bg
    bg_x = movmean(x, d.shape[1] / 5)
    x = x - bg_x
    gaps = np.isnan(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows
        estimate = np.repeat(np.nanmean(x, 1)[:, None], x.shape[1], 1)
    k = min(num_comps, min(x.shape))
    for _ in range(max_iter):
        x[gaps] = estimate[gaps]
        uu, ss, vt = np.linalg.svd(x, full_matrices=False)
        uu, ss, vt = uu[:, :k], ss[:k], vt[:k]
        if not gaps.any():
            break  # nothing to fill: further iterations change nothing
        estimate = (uu * ss) @ vt
        with np.errstate(invalid="ignore", divide="ignore"):
            crit = np.max(((estimate[gaps] - x[gaps]) / x[gaps]) ** 2)
        if crit < 1e-6:
            break
    u[rows, :k] = uu
    v[cols, :k] = vt.T
    s[:k] = ss
    bg[np.ix_(rows, cols)] = bg_x
    return u, s, v, bg


def roi_svd_trace(d: np.ndarray, nan_thresh: float) -> np.ndarray:
    """A user ROI's trace denoised with the SVD, as ``loadTrial`` does.

    Of the three leading components of the ROI's pixels, the one that
    weights the pixels most evenly (times its singular value) is kept; the
    trace is the mean over pixels of that component plus the slow part.

    Parameters
    ----------
    d : ndarray
        ``(pixels, frames)`` data of the ROI's pixels, NaN where missing.
    nan_thresh : float
        ``nanThresh``.

    Returns
    -------
    ndarray
        ``(frames,)`` trace; NaN where the ROI has no usable data.
    """
    out = np.full(d.shape[1], np.nan)
    if d.size == 0 or np.isnan(d).all():
        return out
    u, s, v, bg = nansvd(d, SVD_COMPONENTS, SVD_ITERATIONS, nan_thresh)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # padded components
        likeness = np.abs(np.nanmean(u, 0)) / np.sqrt(np.nanmean(u**2, 0)) * s
        if np.isnan(likeness).all():
            return out
        pc = int(np.nanargmax(likeness))
        return np.nanmean(bg + np.outer(u[:, pc] * s[pc], v[:, pc]), 0)


@dataclass
class TrialData:
    """One trial's data for source extraction (MATLAB's ``CD``).

    Attributes
    ----------
    y : ndarray
        ``(channels, pixels, frames)`` data in counts / ``photon_scale``,
        channel 0 the activity channel; missing samples are filled in.
    finv : ndarray
        ``(pixels, frames)`` variance multiplier (large where filled in).
    frame_lines : ndarray of int
        One-based line of each frame.
    motion_r, motion_c : ndarray
        Row and column motion applied at each frame, including the trial's
        shift (for another warp: the displacement at the image centre).
    discard : ndarray of bool
        Frames discarded for motion.
    global_f : ndarray
        ``(channels, frames)`` global trace of the labeled pixels, in counts
        and in the file's channel order (``CD.global.F``).
    roi_f, roi_fsvd : ndarray
        ``(rois, channels, frames)`` user-ROI traces, plain and denoised, in
        counts and in analysis channel order (``CD.ROIs.F``, ``Fsvd``).
    pixel_rc : ndarray of int
        ``(pixels, 2)`` zero-based consensus row and column of each pixel,
        in column-major order of the map.
    trial_rc : ndarray
        ``(pixels, 2)`` zero-based position of each pixel in the trial's
        registered frame.
    photon_scale : float
        Counts per unit of ``y``.
    timings : dict
        Wall time in seconds of each phase.
    """

    y: np.ndarray
    finv: np.ndarray
    frame_lines: np.ndarray
    motion_r: np.ndarray
    motion_c: np.ndarray
    discard: np.ndarray
    global_f: np.ndarray
    roi_f: np.ndarray
    roi_fsvd: np.ndarray
    pixel_rc: np.ndarray
    trial_rc: np.ndarray
    photon_scale: float
    timings: dict = field(default_factory=dict)


class _Traces:
    """The global and user-ROI traces, accumulated block by block."""

    def __init__(self, mean_px, labeled, masks, order, n_frames, qpos, dev):
        """Index the labeled and ROI pixels among the sampled points.

        ``mean_px`` is the view's mean image ``(rows, cols, channels)`` in
        analysis channel order; ``qpos`` maps a view pixel to its point.
        """
        n_ch = mean_px.shape[2]
        self.order = order
        with np.errstate(invalid="ignore"):
            self.sum_f = np.nansum(mean_px.reshape(-1, n_ch), 0)

        def tensor(a):
            """A NumPy array on the device."""
            return torch.from_numpy(np.ascontiguousarray(a)).to(dev)

        self.q_lab = tensor(qpos[labeled])
        self.m_lab = tensor(mean_px[labeled])
        self.q_roi = [tensor(qpos[m]) for m in masks]
        self.m_roi = [tensor(mean_px[m]) for m in masks]
        self.sum_roi = [np.nansum(mean_px[m], 0) for m in masks]
        self.global_f = np.full((n_ch, n_frames), np.nan)
        self.roi_f = np.full((len(masks), n_ch, n_frames), np.nan)
        self.roi_px = [
            np.full((n_ch, q.numel(), n_frames), np.nan) for q in self.q_roi
        ]

    def add(self, frames: slice, y: torch.Tensor) -> None:
        """Add the frames of one part: ``y`` is (channels, points, frames)."""
        for k in range(y.shape[0]):
            g = _ratio(
                y[k].index_select(0, self.q_lab), self.m_lab[:, k], False
            )
            self.global_f[self.order[k], frames] = (
                g.cpu().numpy() * self.sum_f[k]
            )
            for r, q in enumerate(self.q_roi):
                t = y[k].index_select(0, q)
                f = _ratio(t, self.m_roi[r][:, k], True).cpu().numpy()
                self.roi_px[r][k][:, frames] = t.cpu().numpy()
                with np.errstate(invalid="ignore"):
                    self.roi_f[r, k, frames] = f * self.sum_roi[r][k]

    def roi_fsvd(self, nan_thresh: float) -> np.ndarray:
        """The ROI traces denoised by the SVD of their pixels."""
        out = np.full(self.roi_f.shape, np.nan)
        for r, px in enumerate(self.roi_px):
            for k in range(px.shape[0]):
                out[r, k] = roi_svd_trace(px[k], nan_thresh)
        return out


def _ratio(t: torch.Tensor, m: torch.Tensor, mean_gaps: bool) -> torch.Tensor:
    """Summed data over the summed mean image of the same pixels.

    ``t`` is ``(pixels, frames)``, ``m`` the pixels' mean image. Pixels
    whose sample is missing are left out of both sums; a pixel whose mean
    is missing is left out of both for the ROI traces (``mean_gaps``) and
    only out of the mean's sum for the global trace, as in MATLAB.
    """
    gap = torch.isnan(t)
    either = gap | torch.isnan(m)[:, None]
    num = t.masked_fill(either if mean_gaps else gap, 0.0).sum(0)
    den = m[:, None].expand_as(t).masked_fill(either, 0.0).sum(0)
    return num / den


def _points(need, alignment: TrialAlignment, warp):
    """Where the wanted pixels are read in the trimmed raster image.

    Returns
    -------
    r_d, c_d : ndarray
        One-based row and column of each point in the trimmed image.
    shift : (float, float)
        Motion to add to every frame (a rigid warp's shift).
    centre : (float, float)
        Displacement that the warp adds at the image centre.
    qpos : ndarray of int
        Point of each pixel of the view (-1: not read).
    """
    rq, cq = np.nonzero(need)
    qpos = np.full(need.shape, -1, dtype=np.int64)
    qpos[rq, cq] = np.arange(rq.size)
    if isinstance(warp, RigidShift):
        r_d, c_d = alignment.view_r[rq], alignment.view_c[cq]
        return r_d, c_d, (warp.row, warp.col), (0.0, 0.0), qpos
    for v in (alignment.view_r, alignment.view_c):
        if np.any(np.diff(v) != 1):
            raise ValueError("expected unit-step viewR and viewC")
    r_t, c_t = warp(rq.astype(np.float64), cq.astype(np.float64))
    r_d = alignment.view_r[0] + np.asarray(r_t, dtype=np.float64)
    c_d = alignment.view_c[0] + np.asarray(c_t, dtype=np.float64)
    x0 = ((need.shape[0] - 1) / 2, (need.shape[1] - 1) / 2)
    r_c, c_c = warp(np.array([x0[0]]), np.array([x0[1]]))
    centre = (float(r_c[0]) - x0[0], float(c_c[0]) - x0[1])
    return r_d, c_d, (0.0, 0.0), centre, qpos


def _check_layout(layout: RasterLayout) -> None:
    """Reject recordings that the loader does not support."""
    if layout.fast_zs.size != 1:
        raise ValueError("only single-plane acquisitions are supported")


def load_trial(  # noqa: C901
    dat_path,
    first_line: float,
    last_line: float,
    alignment,
    sel_pix: np.ndarray,
    warp: Callable,
    discard_frames,
    mean_image: np.ndarray,
    roi_masks: Sequence[np.ndarray] = (),
    params: Optional[TrialLoadParams] = None,
    device="cpu",
    dtype=np.float32,
) -> TrialData:
    """Read one SLAP2 trial at the selected pixels (``loadTrial``).

    Parameters
    ----------
    dat_path : str or Path
        The trial's raw ``.dat`` file.
    first_line, last_line : float
        One-based line range of the trial (trial table ``first_line`` and
        ``last_line``; may be fractional).
    alignment : TrialAlignment, str or Path
        The trial's alignment, or its ``*_ALIGNMENTDATA.h5``.
    sel_pix : ndarray of bool
        ``(rows, cols)`` selected pixels of the path, in the consensus
        frame. The trial's registered frame shares its top left corner.
    warp : RigidShift or callable
        The trial's warp: ``warp(rows, cols)`` gives the zero-based position
        in the trial's registered frame of zero-based consensus points.
    discard_frames : array_like of bool
        Registered frames discarded for motion (from localization).
    mean_image : ndarray
        ``(rows, cols, channels)`` mean image of the trial aligned to the
        consensus frame, in the file's channel order (labeled pixels and
        the traces' normalization).
    roi_masks : sequence of ndarray of bool
        User-ROI masks in the consensus frame.
    params : TrialLoadParams, optional
        Settings (default: SILo's for SLAP2).
    device : str or torch.device
        Where frames are formed and interpolated.
    dtype : numpy dtype
        Precision of ``y`` and ``finv`` (float64 reproduces MATLAB to
        rounding error; float32 halves the memory).

    Returns
    -------
    TrialData

    Raises
    ------
    ValueError
        For recordings with several fast-z planes, an activity channel
        outside the recording's, or a mean image without all channels.
    """
    p = params or TrialLoadParams()
    dev = torch.device(device)
    t0 = time.perf_counter()
    timings = {}
    layout = raster_layout(dat_path)
    _check_layout(layout)
    if not isinstance(alignment, TrialAlignment):
        alignment = read_alignment(Path(alignment))
    n_ch = layout.num_channels
    order = p.ordered_channels(n_ch)
    mean_image = np.asarray(mean_image, dtype=np.float64)
    if mean_image.ndim != 3 or mean_image.shape[2] != n_ch:
        raise ValueError(f"the mean image must have {n_ch} channels")
    line_rate = 1.0 / layout.line_period_s
    dt = line_rate / p.analyze_hz
    dtk = int(math.ceil(dt))
    lines = frame_lines(float(first_line), float(last_line), dt)
    n_f = lines.size

    ds = alignment.ds_frames
    kernel = max(int(math.ceil(line_rate / alignment.align_hz)), dtk)
    comp, mot_r, mot_c = online_compensation(
        alignment,
        layout,
        (min(first_line, ds[0]), max(last_line, ds[-1])),
        kernel,
    )

    shape = alignment.view_shape
    sel = np.asarray(sel_pix, dtype=bool)
    sel_view = _fit(sel, shape, False)
    mean_view = _fit(mean_image, shape, np.nan)[:, :, order]
    masks = [_fit(np.asarray(m, dtype=bool), shape, False) for m in roi_masks]
    labeled = labeled_pixels(mean_view[:, :, 0])
    need = sel_view | labeled
    for m in masks:
        need |= m
    r_d, c_d, shift, centre, qpos = _points(need, alignment, warp)
    motion_r = upsample_motion(ds, mot_r, lines) + shift[0]
    motion_c = upsample_motion(ds, mot_c, lines) + shift[1]

    cols_s, rows_s = np.nonzero(sel.T)  # column-major order of the map
    inside = (rows_s < shape[0]) & (cols_s < shape[1])
    q_sel = qpos[rows_s[inside], cols_s[inside]]
    dest = slice(None) if inside.all() else np.flatnonzero(inside)
    y = np.full((n_ch, rows_s.size, n_f), np.nan, dtype=dtype)
    finv = np.full((rows_s.size, n_f), np.nan, dtype=dtype)
    traces = _Traces(mean_view, labeled, masks, order, n_f, qpos, dev)
    q_sel = torch.from_numpy(q_sel).to(dev)
    as_stored = torch.from_numpy(np.empty(0, dtype)).dtype
    timings["setup"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    edges = block_edges(n_f, p.block_frames)
    for b in range(edges.size - 1):
        fb = np.arange(edges[b], edges[b + 1])
        comp_b, d_b = block_compensation(comp, lines[fb])
        for part, yq, fq in frames_at(
            layout,
            order,
            lines[fb],
            dtk,
            alignment.trim_rows,
            alignment.trim_cols,
            r_d,
            c_d,
            motion_r[fb] - d_b[0],
            motion_c[fb] - d_b[1],
            comp_b,
            device,
        ):
            fr = slice(fb[part][0], fb[part][-1] + 1)
            y[:, dest, fr] = (
                yq.index_select(1, q_sel).to(as_stored).cpu().numpy()
            )
            finv[dest, fr] = (
                fq.index_select(0, q_sel).to(as_stored).cpu().numpy()
            )
            traces.add(fr, yq)
    timings["frames"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    roi_fsvd = traces.roi_fsvd(p.nan_thresh)
    timings["roi_svd"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    discard = upsample_discard(discard_frames, n_f)
    finish_pixels(y, finv, discard, p.baseline_window_samps)
    y /= p.photon_scale
    for a in (traces.global_f, traces.roi_f, roi_fsvd):
        a[..., discard] = np.nan
    timings["finish"] = time.perf_counter() - t0

    pixel_rc = np.stack([rows_s, cols_s], 1)
    r_t, c_t = warp(rows_s.astype(np.float64), cols_s.astype(np.float64))
    return TrialData(
        y=y,
        finv=finv,
        frame_lines=lines,
        motion_r=motion_r + centre[0],
        motion_c=motion_c + centre[1],
        discard=discard,
        global_f=traces.global_f,
        roi_f=traces.roi_f,
        roi_fsvd=roi_fsvd,
        pixel_rc=pixel_rc,
        trial_rc=np.stack([np.asarray(r_t), np.asarray(c_t)], 1),
        photon_scale=float(p.photon_scale),
        timings=timings,
    )
