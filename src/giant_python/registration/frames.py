"""Raster frames from raw SLAP2 samples with getImage's exponential kernel.

Equivalent to GIAnT-MATLAB ``slap2Frames`` (``Kernel='exp'``, inclusive
boundary, invalid samples masked per channel, no online compensation): the
frame at line time ``t`` weights each sample on line ``l`` by
``exp(-|t - l| / dt)`` for ``|t - l| <= max(3 dt, P)``, restricted to lines
of the requested fast-z plane, and each pixel's value is the weighted mean of
its samples. Frames are formed at the sampled pixels only.
"""

from typing import Tuple

import numpy as np
import torch

from ..io.slap2_raster import INVALID_SAMPLE, RasterLayout


def _ceil_div(a: np.ndarray, b: int) -> np.ndarray:
    """Integer ceil(a / b) for positive b, exact for negative a."""
    return -((-a) // b)


class FrameFormer:
    """Form exp-kernel frames at the sampled pixels of one fast-z plane.

    Parameters
    ----------
    layout : RasterLayout
        The recording's layout.
    samples : ndarray of int16
        Raw samples ``(channels, slots, cycles)`` of all raster slots, for
        the zero-based cycles ``cycle0 ...``.
    cycle0 : int
        Zero-based index of the first cycle in ``samples``.
    z_idx : int
        One-based fast-z plane.
    device : str or torch.device
        Where sums are formed.

    Attributes
    ----------
    pixels : ndarray of int
        Sorted one-based pixel IDs that have samples on the plane.
    rows, cols : ndarray of int
        Zero-based DMD image row and column of each pixel.
    """

    def __init__(
        self,
        layout: RasterLayout,
        samples: np.ndarray,
        cycle0: int,
        z_idx: int = 1,
        device="cpu",
    ):
        """Build the pixel-major slot table and move samples to device."""
        self.layout = layout
        self.device = torch.device(device)
        self.cycle0 = int(cycle0)
        self.n_cycles = samples.shape[2]
        keep = layout.line_fast_z[layout.slot_line - 1] == z_idx
        slots = np.flatnonzero(keep)
        pix = layout.slot_pixel[slots]
        self.pixels, inv = np.unique(pix, return_inverse=True)
        counts = np.bincount(inv, minlength=self.pixels.size)
        order = np.argsort(inv, kind="stable")
        rank = np.arange(slots.size) - np.repeat(
            np.cumsum(counts) - counts, counts
        )
        width = int(counts.max()) if counts.size else 0
        table = np.full((self.pixels.size, width), -1, dtype=np.int64)
        table[inv[order], rank] = slots[order]
        self.rows = (self.pixels - 1) // layout.n_cols
        self.cols = (self.pixels - 1) % layout.n_cols
        self._table = torch.from_numpy(table).to(self.device)
        line = np.where(table >= 0, layout.slot_line[np.maximum(table, 0)], 1)
        self._line = torch.from_numpy(line - 1).to(self.device)
        self._samples = torch.from_numpy(np.ascontiguousarray(samples)).to(
            self.device
        )

    @property
    def n_pixels(self) -> int:
        """Number of sampled pixels."""
        return int(self.pixels.size)

    def _tables(self, times: np.ndarray, dt: int):
        """Local cycle and weight of each (frame, line index, k) sample."""
        lay = self.layout
        p = lay.lines_per_cycle
        dt_read = max(3 * dt, p)
        k_max = 2 * dt_read // p + 1
        lo = np.maximum(1, times - dt_read)[:, None]
        hi = np.minimum(lay.total_lines, times + dt_read)[:, None]
        line_idx = np.arange(1, p + 1)[None, :]
        c_min = _ceil_div(lo - line_idx, p)
        c_max = (hi - line_idx) // p
        c = c_min[:, :, None] + np.arange(k_max)[None, None, :]
        ok = c <= c_max[:, :, None]
        lines = c * p + line_idx[:, :, None]
        w = np.where(
            ok, np.exp(-np.abs((times[:, None, None] - lines) / dt)), 0.0
        )
        local = c - self.cycle0
        if np.any(ok & ((local < 0) | (local >= self.n_cycles))):
            raise ValueError("frame kernel reaches cycles that were not read")
        local = np.clip(local, 0, self.n_cycles - 1)
        return local, w

    def sums(
        self, times, dt: int, channels=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Weighted sample sums of frames.

        Parameters
        ----------
        times : array_like of int
            One-based line times of the frames.
        dt : int
            Kernel time constant in lines.
        channels : sequence of int, optional
            Zero-based channels (default: all).

        Returns
        -------
        num, den : torch.Tensor of float64
            ``(channels, frames, pixels)`` sums of weighted valid samples and
            of their weights (``slap2FrameSums`` Num and Den).
        """
        times = np.asarray(times, dtype=np.int64).ravel()
        if channels is None:
            channels = range(self.layout.num_channels)
        channels = list(channels)
        local, w = self._tables(times, int(dt))
        dev = self.device
        local = torch.from_numpy(local).to(dev)
        w = torch.from_numpy(w).to(dev)
        n_f, n_pix = times.size, self.n_pixels
        num = torch.zeros(
            len(channels), n_f, n_pix, dtype=torch.float64, device=dev
        )
        den = torch.zeros_like(num)
        for j in range(self._table.shape[1]):
            slot = self._table[:, j]
            has = slot >= 0
            li = self._line[:, j]
            cyc = local[:, li, :]
            wj = w[:, li, :] * has[None, :, None]
            flat = slot.clamp(min=0)[None, :, None] * self.n_cycles + cyc
            for k, ch in enumerate(channels):
                x = self._samples[ch].reshape(-1)[flat]
                valid = x != INVALID_SAMPLE
                xv = torch.where(valid, x, 0).to(torch.float64)
                num[k] += (wj * xv).sum(-1)
                den[k] += (wj * valid).sum(-1)
        return num, den

    def frames(
        self, times, dt: int, channels=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frames and freshness, as ``slap2Frames`` returns them.

        Returns
        -------
        values : torch.Tensor of float32
            ``(channels, frames, pixels)`` weighted means, NaN where a pixel
            has no valid sample.
        freshness : torch.Tensor of float32
            ``(frames, pixels)`` summed weights of the first channel.
        """
        num, den = self.sums(times, dt, channels)
        values = (num / den).to(torch.float32)
        return values, den[0].to(torch.float32)
