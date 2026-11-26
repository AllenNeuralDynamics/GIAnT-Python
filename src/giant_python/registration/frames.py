"""Raster frames from raw SLAP2 samples with getImage's exponential kernel.

Equivalent to GIAnT-MATLAB ``slap2Frames`` (``Kernel='exp'``, inclusive
boundary, invalid samples masked per channel, no online compensation): the
frame at line time ``t`` weights each sample on line ``l`` by
``exp(-|t - l| / dt)`` for ``|t - l| <= max(3 dt, P)``, restricted to lines
of the requested fast-z plane, and each pixel's value is the weighted mean of
its samples. Frames are formed at the sampled pixels only.

Source extraction forms its frames with the same sums (``slap2FrameSums``
with the exclusive boundary of getImages, see ``FrameFormer(exclusive=True)``
and :class:`PixelFormer`).
"""

from typing import Tuple

import numpy as np
import torch

from ..io.slap2_raster import INVALID_SAMPLE, RasterLayout


def _bands(lo: np.ndarray, hi: np.ndarray, band: int):
    """Split frames into groups whose cycle ranges span at most ``band``.

    Frames are taken in order of their first cycle; a frame without samples
    (``hi < 0``) joins the current group.
    """
    order = np.argsort(lo, kind="stable")
    groups, cur, start, end = [], [], None, None
    for f in order:
        if hi[f] < 0:
            cur.append(f)
            continue
        if cur and start is not None and max(end, hi[f]) - start >= band:
            groups.append(np.array(cur))
            cur, start, end = [], None, None
        cur.append(f)
        start = lo[f] if start is None else start
        end = hi[f] if end is None else max(end, hi[f])
    if cur:
        groups.append(np.array(cur))
    return groups


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
    exclusive : bool
        Leave out the lines exactly ``max(3 dt, P)`` from a frame's time, as
        getImages does (``slap2FrameSums`` with ``Boundary='exclusive'``;
        source extraction's frames). Default: include them, as getImage.
    band : int
        Most cycles spanned by the frames summed in one matrix product.
        Sums do not depend on it beyond rounding; a small band is faster
        when each frame's kernel covers few cycles.

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
        exclusive: bool = False,
        band: int = 256,
    ):
        """Build the pixel-major slot table and move samples to device."""
        self.layout = layout
        self.device = torch.device(device)
        self.exclusive = bool(exclusive)
        self.band = int(band)
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
        # slots by line index, padded: (P, widest line); a pixel's slots as
        # positions in that layout, for turning slot sums into pixel sums
        p = layout.lines_per_cycle
        per_line = [
            slots[layout.slot_line[slots] == ln] for ln in range(1, p + 1)
        ]
        self._line_width = max(1, max(len(x) for x in per_line))
        line_slots = np.full((p, self._line_width), -1, dtype=np.int64)
        flat_of = np.full(layout.slot_line.size, -1, dtype=np.int64)
        for i, x in enumerate(per_line):
            line_slots[i, : x.size] = x
            flat_of[x] = i * self._line_width + np.arange(x.size)
        self._line_slots = torch.from_numpy(line_slots).to(self.device)
        pix_flat = np.where(table >= 0, flat_of[np.maximum(table, 0)], -1)
        self._pix_flat = torch.from_numpy(pix_flat).to(self.device)
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
        reach = dt_read - 1 if self.exclusive else dt_read
        lo = np.maximum(1, times - reach)[:, None]
        hi = np.minimum(lay.total_lines, times + reach)[:, None]
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
        return local, w, lines, ok

    def _accumulate(self, local, w, channels, band=None):
        """Num and Den at the pixels for given sample cycles and weights.

        ``local`` and ``w`` are ``(frames, P, k)`` NumPy arrays: the local
        cycle and weight of each line index's k-th sample of each frame.
        Frames are taken in groups spanning at most ``band`` cycles (default
        ``self.band``); for each group, every line's samples are weighted by
        one matrix product (frames x cycles) @ (cycles x slots), in double
        precision.
        """
        band = self.band if band is None else band
        dev = self.device
        n_f, n_pix = local.shape[0], self.n_pixels
        num = torch.zeros(
            len(channels), n_f, n_pix, dtype=torch.float64, device=dev
        )
        den = torch.zeros_like(num)
        used = w > 0
        lo = np.where(used, local, np.iinfo(np.int64).max).min((1, 2))
        hi = np.where(used, local, -1).max((1, 2))
        for group in _bands(lo, hi, band):
            self._accumulate_band(
                local[group],
                w[group],
                used[group],
                channels,
                group,
                num,
                den,
            )
        return num, den

    def _accumulate_band(self, local, w, used, channels, group, num, den):
        """Add one band of frames' sums to ``num`` and ``den`` (in place)."""
        dev = self.device
        if not used.any():
            return
        c0 = int(local[used].min())
        c1 = int(local[used].max()) + 1
        nf, p, _ = local.shape
        weights = np.zeros((nf, p, c1 - c0))
        fi, li, ki = np.nonzero(used)
        weights[fi, li, local[fi, li, ki] - c0] = w[fi, li, ki]
        wt = torch.from_numpy(weights).to(dev).permute(1, 0, 2)  # P, F, C
        pad = (self._line_slots >= 0)[:, :, None]
        rows = torch.from_numpy(group).to(dev)
        for k, ch in enumerate(channels):
            x = self._samples[ch][:, c0:c1][self._line_slots.clamp(min=0)]
            valid = (x != INVALID_SAMPLE) & pad
            xv = torch.where(valid, x, 0).to(torch.float64)
            s_num = torch.bmm(wt, xv.transpose(1, 2))  # P, F, slots
            s_den = torch.bmm(wt, valid.to(torch.float64).transpose(1, 2))
            s_num = s_num.permute(1, 0, 2).reshape(nf, -1)
            s_den = s_den.permute(1, 0, 2).reshape(nf, -1)
            add_n = torch.zeros(
                nf, self.n_pixels, dtype=torch.float64, device=dev
            )
            add_d = torch.zeros_like(add_n)
            for j in range(self._pix_flat.shape[1]):
                pos = self._pix_flat[:, j]
                has = (pos >= 0).to(torch.float64)
                add_n += s_num[:, pos.clamp(min=0)] * has
                add_d += s_den[:, pos.clamp(min=0)] * has
            num[k].index_add_(0, rows, add_n)
            den[k].index_add_(0, rows, add_d)

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
        channels = self._channels(channels)
        local, w, _, _ = self._tables(times, int(dt))
        return self._accumulate(local, w, channels)

    def _channels(self, channels) -> list:
        """Zero-based channels, all by default."""
        if channels is None:
            return list(range(self.layout.num_channels))
        return list(channels)

    def shifted_sums(self, times, dt: int, shift_fn, channels=None):
        """Sums of frames whose samples move by a per-line displacement.

        Each frame's samples are grouped by the integer displacement
        ``shift_fn`` assigns to their line (as ``slap2FrameSums`` does with
        ``CompensateOnline``); each group is summed at the pixels' own
        positions, to be placed at the displaced positions by the caller.

        Parameters
        ----------
        times : array_like of int
            One-based line times of the frames.
        dt : int
            Kernel time constant in lines.
        shift_fn : callable
            ``shift_fn(lines) -> (dr, dc)`` for an ``(frames, P, k)`` array
            of one-based lines.
        channels : sequence of int, optional
            Zero-based channels (default: all).

        Yields
        ------
        (dr, dc), frames, num, den
            The displacement, the indices of the frames with samples in the
            group, and the group's Num and Den ``(channels, frames, pixels)``.
        """
        times = np.asarray(times, dtype=np.int64).ravel()
        channels = self._channels(channels)
        local, w, lines, ok = self._tables(times, int(dt))
        dr, dc = shift_fn(lines)
        pairs = np.unique(np.stack([dr[ok], dc[ok]], 1), axis=0)
        for r, c in pairs:
            group = ok & (dr == r) & (dc == c)
            sel = np.flatnonzero(group.any(axis=(1, 2)))
            wg = np.where(group[sel], w[sel], 0.0)
            num, den = self._accumulate(local[sel], wg, channels)
            yield (int(r), int(c)), sel, num, den

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


class PixelFormer(FrameFormer):
    """Weighted sums for many short frames, arranged pixel by pixel.

    Source extraction forms thousands of frames whose kernels each cover a
    few cycles. :meth:`pixel_sums` gives the sums of :meth:`sums` and
    :meth:`shifted_sums` (equal to rounding) as ``(channels, pixels,
    frames)``: a pixel's slots are added with one indexed sum per band, and
    the summed weights are formed without the samples where a band has no
    invalid sample. On a CPU this is several times faster than the parent's
    sums, whose fixed order of additions registration keeps.
    """

    def __init__(self, *args, **kwargs):
        """Map every place of the padded line layout to its pixel."""
        super().__init__(*args, **kwargs)
        has = self._pix_flat >= 0
        pix = has.nonzero(as_tuple=True)[0]
        flat = self._pix_flat[has]
        # padded places go to an extra pixel that is dropped
        self._flat_pix = torch.full(
            (self._line_slots.numel(),),
            self.n_pixels,
            dtype=torch.long,
            device=self.device,
        )
        self._flat_pix[flat] = pix
        self._slot_pix = pix
        self._slot_line = torch.div(
            flat, self._line_width, rounding_mode="floor"
        )

    def pixel_sums(self, times, dt: int, channels=None, shift_fn=None):
        """Sums of frames, grouped by the displacement of their samples.

        Parameters
        ----------
        times : array_like of int
            One-based line times of the frames.
        dt : int
            Kernel time constant in lines.
        channels : sequence of int, optional
            Zero-based channels (default: all).
        shift_fn : callable, optional
            ``shift_fn(lines) -> (dr, dc)`` as for :meth:`shifted_sums`.
            Default: no displacement, one group of all frames.

        Yields
        ------
        (dr, dc), frames, num, den
            The displacement, the indices of the frames with samples in the
            group, and the group's Num and Den ``(channels, pixels,
            frames)`` at the pixels' own positions.
        """
        times = np.asarray(times, dtype=np.int64).ravel()
        channels = self._channels(channels)
        local, w, lines, ok = self._tables(times, int(dt))
        if shift_fn is None:
            num, den = self._pixel_accumulate(local, w, channels)
            yield (0, 0), np.arange(times.size), num, den
            return
        dr, dc = shift_fn(lines)
        pairs = np.unique(np.stack([dr[ok], dc[ok]], 1), axis=0)
        for r, c in pairs:
            group = ok & (dr == r) & (dc == c)
            sel = np.flatnonzero(group.any(axis=(1, 2)))
            wg = np.where(group[sel], w[sel], 0.0)
            num, den = self._pixel_accumulate(local[sel], wg, channels)
            yield (int(r), int(c)), sel, num, den

    def _pixel_accumulate(self, local, w, channels):
        """Num and Den ``(channels, pixels, frames)``, band by band."""
        size = (len(channels), self.n_pixels, local.shape[0])
        num = torch.zeros(size, dtype=torch.float64, device=self.device)
        den = torch.zeros_like(num)
        used = w > 0
        lo = np.where(used, local, np.iinfo(np.int64).max).min((1, 2))
        hi = np.where(used, local, -1).max((1, 2))
        for group in _bands(lo, hi, self.band):
            self._pixel_band(
                local[group], w[group], used[group], channels, group, num, den
            )
        return num, den

    def _to_pixels(self, sums: torch.Tensor, index: torch.Tensor):
        """Add rows of ``sums`` (places, frames) into their pixels."""
        out = torch.zeros(
            self.n_pixels + 1,
            sums.shape[1],
            dtype=torch.float64,
            device=self.device,
        )
        return out.index_add_(0, index, sums)[: self.n_pixels]

    def _add_frames(self, total, group, part) -> None:
        """Add a band's ``(pixels, frames)`` sums into its frames."""
        if group.size and np.all(np.diff(group) == 1):
            total[:, int(group[0]) : int(group[-1]) + 1] += part
        else:
            total.index_add_(1, torch.from_numpy(group).to(self.device), part)

    def _pixel_band(self, local, w, used, channels, group, num, den):
        """Add one band of frames' sums to ``num`` and ``den`` (in place)."""
        if not used.any() or self._slot_pix.numel() == 0:
            return
        c0 = int(local[used].min())
        c1 = int(local[used].max()) + 1
        nf, p, _ = local.shape
        weights = np.zeros((p, c1 - c0, nf))
        fi, li, ki = np.nonzero(used)
        weights[li, local[fi, li, ki] - c0, fi] = w[fi, li, ki]
        wt = torch.from_numpy(weights).to(self.device)  # P, C, F
        slots = self._line_slots.clamp(min=0)
        den_valid = None  # the summed weights where every sample is valid
        for k, ch in enumerate(channels):
            x = self._samples[ch][:, c0:c1][slots]  # P, width, C
            valid = x != INVALID_SAMPLE
            xv = torch.where(valid, x, 0).to(torch.float64)
            s_num = torch.bmm(xv, wt).reshape(-1, nf)  # places, F
            self._add_frames(
                num[k], group, self._to_pixels(s_num, self._flat_pix)
            )
            if bool(valid.all()):
                if den_valid is None:
                    den_valid = self._to_pixels(
                        wt.sum(1)[self._slot_line], self._slot_pix
                    )
                d = den_valid
            else:
                s_den = torch.bmm(valid.to(torch.float64), wt)
                d = self._to_pixels(s_den.reshape(-1, nf), self._flat_pix)
            self._add_frames(den[k], group, d)
