"""Frames in the registration window, with or without online compensation.

A frame source returns, for given frame times, each window pixel's value per
channel (NaN where it has no valid sample) and its freshness (0 there), as
MultiRoiRegistration's ``getImageWrapper`` does after trimming:

* :class:`PlainFrames`: ``slap2Frames`` frames, trimmed to the bounding box
  of the pixels valid in the first initial frame (the first one that has
  samples; MATLAB's window would be empty).
* :class:`CompensatedFrames`: ``slap2Frames(..., CompensateOnline=true,
  RefOffset=<frame offset>)`` frames, trimmed to the raster's extent. A
  sample moved by its line's online offset can land on a pixel the raster
  does not sample; the window's pixel list is the union of every position
  the trial's samples reach.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .compensation import (
    LineOffsets,
    frame_offsets,
    line_offsets,
    line_shifts,
)
from .frames import FrameFormer
from .template import window_image


def has_samples(values: torch.Tensor) -> np.ndarray:
    """Which frames have at least one valid sample.

    Parameters
    ----------
    values : torch.Tensor
        ``(channels, frames, pixels)`` frame values, NaN where a pixel has
        no valid sample.

    Returns
    -------
    ndarray of bool
        ``(frames,)``.
    """
    return (~torch.isnan(values.sum(0))).any(-1).cpu().numpy()


@dataclass
class Window:
    """The trimmed raster window that frames are registered in.

    Attributes
    ----------
    trim_rows, trim_cols : ndarray of int
        One-based DMD rows and columns of the window (MATLAB trimRows /
        trimCols).
    rows, cols : torch.Tensor of int
        Zero-based window positions of the pixels that frames report.
    """

    trim_rows: np.ndarray
    trim_cols: np.ndarray
    rows: torch.Tensor
    cols: torch.Tensor

    @property
    def shape(self) -> tuple:
        """Window ``(rows, cols)``."""
        return self.trim_rows.size, self.trim_cols.size

    def image(self, values: torch.Tensor, fill: float) -> torch.Tensor:
        """Place per-pixel values ``(..., pixels)`` into window images."""
        flat = self.rows * self.shape[1] + self.cols
        return window_image(values, flat, self.shape, fill)


class PlainFrames:
    """Uncompensated frames at the former's pixels inside the window.

    Parameters
    ----------
    former : FrameFormer
        Forms the frames.
    valid : ndarray of bool
        Which of the former's pixels are valid in the first initial frame;
        their bounding box is the window.
    dt : int
        Kernel time constant in lines.
    """

    compensation: Optional[LineOffsets] = None

    def __init__(self, former: FrameFormer, valid: np.ndarray, dt: int):
        """Trim the window to the valid pixels' bounding box."""
        if not valid.any():
            raise ValueError("the first initial frame has no samples")
        dev = former.device
        r0, r1 = former.rows[valid].min(), former.rows[valid].max()
        c0, c1 = former.cols[valid].min(), former.cols[valid].max()
        inside = np.flatnonzero(
            (former.rows >= r0)
            & (former.rows <= r1)
            & (former.cols >= c0)
            & (former.cols <= c1)
        )
        self.former = former
        self.dt = int(dt)
        self.pixels = torch.from_numpy(inside).to(dev)
        self.window = Window(
            trim_rows=np.arange(r0, r1 + 1) + 1,
            trim_cols=np.arange(c0, c1 + 1) + 1,
            rows=torch.from_numpy(former.rows[inside] - r0).to(dev),
            cols=torch.from_numpy(former.cols[inside] - c0).to(dev),
        )

    def frames(self, times):
        """``(channels, frames, pixels)`` values and ``(frames, pixels)``
        freshness at the window's pixels."""
        vals, fresh = self.former.frames(times, self.dt)
        return vals[:, :, self.pixels], fresh[:, self.pixels]

    def references(self, times) -> Optional[np.ndarray]:
        """Frame reference offsets (none without compensation)."""
        return None


class CompensatedFrames:
    """Frames with the online motion correction undone line by line.

    Parameters
    ----------
    former : FrameFormer
        Forms the per-displacement sums.
    offsets : LineOffsets
        Line offsets of the recording.
    dt : int
        Kernel time constant in lines.
    times : array_like of int
        Every frame time the trial will request (sets the pixel list).
    """

    def __init__(
        self, former: FrameFormer, offsets: LineOffsets, dt: int, times
    ):
        """Trim to the raster's extent and list every reachable pixel."""
        self.former = former
        self.compensation = offsets
        self.dt = int(dt)
        dev = former.device
        self._r0, r1 = int(former.rows.min()), int(former.rows.max())
        self._c0, c1 = int(former.cols.min()), int(former.cols.max())
        trim_rows = np.arange(self._r0, r1 + 1) + 1
        trim_cols = np.arange(self._c0, c1 + 1) + 1
        self._shape = (trim_rows.size, trim_cols.size)
        reach = [
            self._landing(dr, dc) for dr, dc in self._displacements(times)
        ]
        cand = np.unique(np.concatenate([f[f >= 0] for f in reach]))
        self._lookup = np.full(self._shape[0] * self._shape[1], -1)
        self._lookup[cand] = np.arange(cand.size)
        self._dest = {}
        self.window = Window(
            trim_rows=trim_rows,
            trim_cols=trim_cols,
            rows=torch.from_numpy(cand // self._shape[1]).to(dev),
            cols=torch.from_numpy(cand % self._shape[1]).to(dev),
        )

    def references(self, times) -> np.ndarray:
        """``(frames, 2)`` [x, y] offset each frame mostly saw."""
        return frame_offsets(
            self.compensation,
            times,
            self.dt,
            self.former.layout.lines_per_cycle,
        )

    def _displacements(self, times) -> np.ndarray:
        """Every (dr, dc) a sample of these frames can be moved by."""
        times = np.asarray(times, dtype=np.int64).ravel()
        refs = self.references(times)
        p = self.former.layout.lines_per_cycle
        dt_read = max(3 * self.dt, p)
        span = np.arange(-dt_read, dt_read + 1)
        pairs = []
        for a in range(0, times.size, 256):
            lines = times[a : a + 256, None] + span[None, :]
            ok = (lines >= 1) & (lines <= self.compensation.x.size)
            dr, dc = line_shifts(self.compensation, lines, refs[a : a + 256])
            pairs.append(np.stack([dr[ok], dc[ok]], 1))
        return np.unique(np.concatenate(pairs), axis=0)

    def _landing(self, dr: int, dc: int) -> np.ndarray:
        """Window flat index of each former pixel moved by (dr, dc)."""
        h, w = self._shape
        r = self.former.rows - self._r0 + dr
        c = self.former.cols - self._c0 + dc
        inside = (r >= 0) & (r < h) & (c >= 0) & (c < w)
        return np.where(inside, r * w + c, -1)

    def _destination(self, dr: int, dc: int):
        """Pixel-list index of each former pixel moved by (dr, dc)."""
        key = (dr, dc)
        if key not in self._dest:
            flat = self._landing(dr, dc)
            idx = np.where(flat >= 0, self._lookup[np.maximum(flat, 0)], -1)
            if np.any((flat >= 0) & (idx < 0)):
                raise ValueError(f"displacement {key} was not listed")
            keep = np.flatnonzero(idx >= 0)
            dev = self.former.device
            self._dest[key] = (
                torch.from_numpy(keep).to(dev),
                torch.from_numpy(idx[keep]).to(dev),
            )
        return self._dest[key]

    def frames(self, times):
        """``(channels, frames, pixels)`` values and ``(frames, pixels)``
        freshness at the window's pixels, samples moved per line."""
        times = np.asarray(times, dtype=np.int64).ravel()
        refs = self.references(times)
        n_ch = self.former.layout.num_channels
        n = self.window.rows.numel()
        dev = self.former.device
        num = torch.zeros(
            n_ch, times.size * n, dtype=torch.float64, device=dev
        )
        den = torch.zeros_like(num)

        def shifts(lines):
            """Displacement of each sample's line for its frame."""
            return line_shifts(self.compensation, lines, refs)

        for (dr, dc), sel, num_g, den_g in self.former.shifted_sums(
            times, self.dt, shifts
        ):
            keep, dest = self._destination(dr, dc)
            frames = torch.from_numpy(sel).to(dev)
            flat = (frames[:, None] * n + dest[None, :]).reshape(-1)
            num.index_add_(1, flat, num_g[:, :, keep].reshape(n_ch, -1))
            den.index_add_(1, flat, den_g[:, :, keep].reshape(n_ch, -1))
        num = num.reshape(n_ch, times.size, n)
        den = den.reshape(n_ch, times.size, n)
        return (num / den).to(torch.float32), den[0].to(torch.float32)


def _window_pixels(former: FrameFormer, init, dt: int) -> np.ndarray:
    """The former's pixels valid in the first initial frame with samples.

    MATLAB trims to the first initial frame. When it has no samples (e.g.
    a trial whose recording starts late), the next initial frame with
    samples is used; when none has any, the whole raster.
    """
    for k in range(len(init)):
        vals, _ = former.frames(init[k : k + 1], dt)
        valid = ~torch.isnan(vals.sum(0)[0]).cpu().numpy()
        if valid.any():
            return valid
    return np.ones(former.rows.size, dtype=bool)


def make_source(
    layout, former, first_line, last_line, init, ds, dt, compensate: bool
):
    """The trial's frame source.

    Parameters
    ----------
    layout : RasterLayout
        The recording's layout.
    former : FrameFormer
        Frame former over the trial's cycles.
    first_line, last_line : float
        The trial's line range (may be fractional).
    init, ds : ndarray of int
        Initial and registered frame times.
    dt : int
        Kernel time constant in lines.
    compensate : bool
        Undo the online motion correction (compensateOnline).

    Returns
    -------
    PlainFrames or CompensatedFrames
        The source.
    """
    if not compensate:
        return PlainFrames(former, _window_pixels(former, init, dt), dt)
    margin = 2 * max(3 * dt, layout.lines_per_cycle) + 8
    offsets = line_offsets(layout, (first_line - margin, last_line + margin))
    return CompensatedFrames(former, offsets, dt, np.concatenate([init, ds]))
