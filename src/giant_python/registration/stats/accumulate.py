"""Streaming accumulation of a trial's statistics, frame by frame.

:class:`TrialStatsAccumulator` takes the registered frames in order (as
registration writes them, or as ``giant reg-stats`` reads them back). Every
:data:`GEO_CHUNK` frames it forms their photon-noise sums
(:class:`.geometry.FrameGeometry`), reduced per frame and channel to

* ``G0 = A2(t, t)`` and ``R0 = A3(t, t, t)`` (shared by the channels),
* ``F(t) = sum_l A2(t, t + l)`` (pairs with later frames),
* ``B(t) = sum_l A2(t - l, t)`` (pairs with earlier frames),
* ``C(t)``: the sum over pairs ``(u, u')`` with ``u <= t < u'`` (pairs that
  cross the end of frame ``t``),
* ``K(t) = sum_u A3(t, t, u)`` over ``|u - t| <= n_lags``,

each over pairs of frames that are both valid at the pixel in that channel.
Sums of ``A2`` over any run of frames follow from these exactly: the pairs
within frames ``a .. b`` are ``sum_{a..b} F - C(b)``. Every
:data:`BLOCK` frames (once ``FAST_HALF`` frames past the block have their
sums) the block is added to the per-pixel sums. Frames are kept in ring
buffers of :data:`RING` frames.

Only pixels that have been valid are kept: the pixel index grows (by the new
pixels, dilated by :data:`GROW_PX`) when a frame has valid pixels outside
it; a pixel that joins the index was missing in every earlier frame.

Moments are summed in float64 from the float32 frames, photon-noise sums in
float32; the mean is also accumulated in float32 exactly as registration
forms ``meanIM``, so the ``mean`` map equals the alignment file's ``meanIM``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .maps import finish_maps

BLOCK = 64
"""Frames per block of the slow part (and per processing step)."""
BLOCK_MIN = 48
"""Valid frames a block needs at a pixel (75%)."""
FAST_HALF = 32
"""Half width of the running mean (65 frames)."""
FAST_MIN = 49
"""Valid frames a running-mean window needs at a pixel (75% of 65)."""
BIN_T = 8
"""Frames per time bin of the SVD."""
BIN_T_MIN = 6
"""Valid frames a time bin needs at a pixel."""
GROW_PX = 1
"""Dilation (pixels) of new pixels added to the index."""
GEO_CHUNK = 32
"""Frames whose photon-noise sums are formed at a time."""
PER_CHANNEL = ("F", "B", "C", "K")
"""Per-frame photon-noise sums kept per channel (see the module text)."""


class _Accumulators:
    """Per-pixel float64 sums ``(channels, n_idx)``, extended on growth."""

    def __init__(self, n_channels: int, device):
        """No sums yet."""
        self.c = n_channels
        self.device = device
        self.n = 0
        self.data = {}

    def __getitem__(self, name: str) -> torch.Tensor:
        """The sums ``name`` (zeros on first use)."""
        if name not in self.data:
            self.data[name] = torch.zeros(
                (self.c, self.n), dtype=torch.float64, device=self.device
            )
        return self.data[name]

    def __setitem__(self, name: str, value: torch.Tensor) -> None:
        """Replace the sums ``name``."""
        self.data[name] = value

    def extend(self, n_new: int) -> None:
        """Append zeros for ``n_new`` pixels."""
        self.n += n_new
        for k, v in self.data.items():
            pad = v.new_zeros(v.shape[:-1] + (n_new,))
            self.data[k] = torch.cat([v, pad], -1)


class _Ring:
    """Per-frame arrays ``(..., frames, n_idx)``; frame ``t`` is row
    ``t % cap``. Extended with each field's fill value on growth."""

    def __init__(self, cap: int):
        """Empty fields."""
        self.cap = cap
        self.fields = {}

    def put(self, name: str, frames, values: torch.Tensor, fill) -> None:
        """Store ``values`` ``(..., len(frames), n)`` as frames ``frames``."""
        if name not in self.fields:
            shape = values.shape[:-2] + (self.cap, values.shape[-1])
            self.fields[name] = (
                torch.full(
                    shape, fill, dtype=values.dtype, device=values.device
                ),
                fill,
            )
        buf = self.fields[name][0]
        slots = torch.tensor(
            [f % self.cap for f in frames], device=values.device
        )
        buf.index_copy_(buf.ndim - 2, slots, values)

    def rows(self, name: str, frames) -> torch.Tensor:
        """Frames ``frames`` of field ``name``: ``(..., len, n)``."""
        buf = self.fields[name][0]
        slots = torch.tensor([f % self.cap for f in frames], device=buf.device)
        return buf.index_select(buf.ndim - 2, slots)

    def extend(self, n_new: int) -> None:
        """Append ``n_new`` pixels (fill values) to every field."""
        for name, (buf, fill) in self.fields.items():
            pad = buf.new_full(buf.shape[:-1] + (n_new,), fill)
            self.fields[name] = (torch.cat([buf, pad], -1), fill)


def _window(cs: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor):
    """Sums of rows ``lo .. hi`` (inclusive) from a cumulative sum with a
    leading zero row; ``lo`` and ``hi`` are row indices per centre."""
    return cs[hi + 1] - cs[lo]


def _cumsum0(x: torch.Tensor) -> torch.Tensor:
    """Cumulative sum over rows with a leading zero row."""
    return torch.cat([torch.zeros_like(x[:1]), torch.cumsum(x, 0)], 0)


class TrialStatsAccumulator:
    """Statistics of one trial's registered movie.

    Parameters
    ----------
    geometry : FrameGeometry
        The trial's frame geometry (its ``ds`` are the frames to come).
    gains : list of GainCurve
        Photon-noise gains per channel.
    n_channels : int
        Channels per frame.
    """

    def __init__(self, geometry, gains, n_channels: int):
        """Empty sums for ``geometry.ds.size`` frames."""
        self.geo = geometry
        self.gains = list(gains)
        self.device = geometry.device
        self.c = int(n_channels)
        self.n_frames = int(geometry.ds.size)
        self.n_lags = geometry.n_lags
        h, w = geometry.out_shape
        self.shape = (h, w)
        dev = self.device
        self.sum32 = torch.zeros((self.c, h, w), device=dev)
        self.count32 = torch.zeros((self.c, h, w), device=dev)
        self.in_index = torch.zeros(h * w, dtype=torch.bool, device=dev)
        self.index = torch.zeros(0, dtype=torch.long, device=dev)
        self.acc = _Accumulators(self.c, dev)
        self.ring = _Ring(
            BLOCK + 2 * FAST_HALF + 2 * self.n_lags + 2 * GEO_CHUNK
        )
        self.hist = None
        self.lagbuf = None
        self.motion = np.full((self.n_frames, 2), np.nan)
        self.present = np.zeros(self.n_frames, dtype=bool)
        self.bins = []
        self.kappa = [0.0, 0]
        self.n_added = 0
        self.geo_done = 0
        self.block_done = 0
        lay = self.geo.former.layout
        k_max = 2 * self.geo.reach // lay.lines_per_cycle + 1
        self.n_hist = 4 * int(self.geo.pixel_lines.shape[1]) * k_max + 1

    # ---------------------------------------------------------------- intake

    def add(self, k: int, regs, var_factor: torch.Tensor, motion) -> None:
        """Take registered frame ``k``.

        Parameters
        ----------
        k : int
            Frame index (frames must come in order).
        regs : sequence of torch.Tensor
            ``(rows, cols)`` float32 frame of each channel.
        var_factor : torch.Tensor
            ``(rows, cols)`` variance factor of the frame.
        motion : (float, float)
            The frame's row and column motion (NaN for a missing frame).
        """
        if k != self.n_added:
            raise ValueError(f"frame {k} came after {self.n_added - 1}")
        x = torch.stack([r.to(self.device, torch.float32) for r in regs])
        good = ~torch.isnan(x)
        self.sum32 += torch.where(good, x, 0.0)
        self.count32 += good
        valid = good.any(0).reshape(-1)
        new = valid & ~self.in_index
        if bool(new.any()):
            self._grow(new)
        xs = x.reshape(self.c, -1)[:, self.index]
        vf = var_factor.to(self.device, torch.float32).reshape(-1)
        self.ring.put("x", [k], xs[:, None], float("nan"))
        self.ring.put("vf", [k], vf[self.index][None], float("nan"))
        self.ring.put("v", [k], ~torch.isnan(xs)[:, None], False)
        self.motion[k] = motion
        self.present[k] = bool(np.all(np.isfinite(motion))) and bool(
            valid.any()
        )
        self.n_added += 1
        self._advance(final=False)

    def add_missing(self, k: int) -> None:
        """Take frame ``k`` as missing (no samples)."""
        blank = torch.full(self.shape, float("nan"), device=self.device)
        self.add(k, [blank] * self.c, blank, (np.nan, np.nan))

    def _grow(self, new: torch.Tensor) -> None:
        """Add the new valid pixels, dilated by :data:`GROW_PX`."""
        g = GROW_PX
        img = new.reshape(1, 1, *self.shape).float()
        img = F.max_pool2d(img, 2 * g + 1, stride=1, padding=g)
        add = (img.reshape(-1) > 0) & ~self.in_index
        idx = torch.nonzero(add).ravel()
        self.in_index[idx] = True
        self.index = torch.cat([self.index, idx])
        n_new = int(idx.numel())
        self.acc.extend(n_new)
        self.ring.extend(n_new)
        if self.hist is not None:
            pad = self.hist.new_zeros((self.c, n_new, self.n_hist))
            self.hist = torch.cat([self.hist, pad], 1)
        if self.lagbuf is not None:
            self.lagbuf = {
                k: torch.cat([v, v.new_zeros(v.shape[:-1] + (n_new,))], -1)
                for k, v in self.lagbuf.items()
            }

    # ------------------------------------------------------------ schedule

    def _advance(self, final: bool) -> None:
        """Form the frame sums and blocks whose inputs have all arrived."""
        end = self.n_added if final else self.n_added - self.n_lags
        while self.geo_done < end and (
            final or end - self.geo_done >= GEO_CHUNK
        ):
            g1 = min(end, self.geo_done + GEO_CHUNK)
            self._geometry(self.geo_done, g1)
            self.geo_done = g1
        n_blocks = -(-self.n_frames // BLOCK)
        while self.block_done < n_blocks:
            b1 = min(self.n_frames, (self.block_done + 1) * BLOCK)
            if not (
                final or self.geo_done >= min(self.n_frames, b1 + FAST_HALF)
            ):
                break
            self._block(self.block_done)
            self.block_done += 1

    # ------------------------------------------------------------ geometry

    def _geometry(self, g0: int, g1: int) -> None:
        """Photon-noise sums of frames ``[g0, g1)``, reduced per frame."""
        lag = self.n_lags
        f_ext = min(self.n_added, g1 + lag)
        out = self.geo.registered(
            g0,
            g1,
            f_ext,
            self.index,
            self.motion[g0:f_ext],
            self.present[g0:f_ext],
            self.ring.rows("vf", range(g0, f_ext)),
        )
        self._spatial_factor(g0, g1)
        frames = range(g0, g1)
        self.ring.put("G0", frames, out["G"][0][None], 0.0)
        self.ring.put("R0", frames, out["R0"][None], 0.0)
        self.ring.put("N0", frames, out["N0"][None], 0.0)
        per = self._per_channel(g0, g1, f_ext, out)
        for name in PER_CHANNEL:
            self.ring.put(name, frames, per[name], 0.0)

    def _lag_history(self, out: dict, g0: int):
        """Per-lag ``A2`` and ``A3(t, t + l, t + l)`` of the ``n_lags``
        frames before ``g0`` and of the chunk: ``(n_lags, lag + n, n_idx)``
        each (zero rows before frame 0); keeps the last rows for the next
        chunk."""
        lag = self.n_lags
        g_new, rp_new = out["G"][1:], out["Rp"]
        if self.lagbuf is None or g0 == 0:
            zeros = g_new.new_zeros((lag, lag, g_new.shape[-1]))
            self.lagbuf = {"G": zeros, "Rp": zeros.clone()}
        g_all = torch.cat([self.lagbuf["G"], g_new], 1)
        rp_all = torch.cat([self.lagbuf["Rp"], rp_new], 1)
        self.lagbuf = {"G": g_all[:, -lag:], "Rp": rp_all[:, -lag:]}
        return g_all, rp_all

    def _per_channel(self, g0: int, g1: int, f_ext: int, out: dict) -> dict:
        """``F``, ``B``, ``C`` and ``K`` of frames ``[g0, g1)`` per channel
        ``(channels, g1 - g0, n_idx)`` (see the module text)."""
        lag, n = self.n_lags, g1 - g0
        n_idx = self.index.numel()
        v = self.ring.rows("v", range(max(0, g0 - lag), f_ext)).float()
        lead = lag - min(lag, g0)
        tail = g1 + lag - f_ext
        v = torch.cat(
            [
                v.new_zeros((self.c, lead, n_idx)),
                v,
                v.new_zeros((self.c, tail, n_idx)),
            ],
            1,
        )  # row of frame f: f - g0 + lag
        if lag == 0:
            zero = v.new_zeros((self.c, n, n_idx))
            k = out["R0"][None] * v[:, :n]
            return {"F": zero, "B": zero, "C": zero, "K": k}
        g_all, rp_all = self._lag_history(out, g0)
        res = {name: v.new_zeros((self.c, n, n_idx)) for name in PER_CHANNEL}
        res["K"] = res["K"] + out["R0"][None]
        rows = torch.arange(n, device=self.device) + lag  # frames t
        for ell in range(1, lag + 1):
            r = slice(0, lag + n)  # frames g0 - lag .. g1 - 1
            gm = g_all[ell - 1][None] * v[:, r] * v[:, ell : ell + lag + n]
            res["F"] += gm[:, rows]
            res["B"] += gm[:, rows - ell]
            for i in range(ell):
                res["C"] += gm[:, rows - i]
            res["K"] += v[:, rows + ell] * out["R"][ell - 1][None]
            res["K"] += v[:, rows - ell] * rp_all[ell - 1][rows - ell][None]
        res["K"] = res["K"] * v[:, rows]
        return res

    def _spatial_factor(self, g0: int, g1: int) -> None:
        """Add frames' bilinear factors (2 x 2 sums) to the trial mean."""
        from .geometry import spatial_factor

        _, _, frac = self.geo.corners(self.motion[g0:g1], self.present[g0:g1])
        for i in np.flatnonzero(self.present[g0:g1]):
            self.kappa[0] += spatial_factor(frac[i])
            self.kappa[1] += 1

    # -------------------------------------------------------------- blocks

    def _block(self, j: int) -> None:
        """Add block ``j`` to every per-pixel sum."""
        b0 = j * BLOCK
        b1 = min(self.n_frames, b0 + BLOCK)
        s0 = max(0, b0 - FAST_HALF)
        s1 = min(self.n_frames, b1 + FAST_HALF)
        span = range(s0, s1)
        rows = {
            name: self.ring.rows(name, span)
            for name in ("x", "v", "G0", "R0", "N0") + PER_CHANNEL
        }
        bins = []
        for ch in range(self.c):
            blk = _BlockChannel(self, ch, rows, (s0, b0, b1))
            blk.add_all()
            bins.append(blk.bin_means)
        self.bins.append(torch.stack(bins).cpu())

    # -------------------------------------------------------------- finish

    def finish(self) -> dict:
        """Process the remaining frames and return the trial's statistics
        (see :func:`.maps.finish_maps`)."""
        if self.n_added != self.n_frames:
            raise ValueError(
                f"{self.n_added} of {self.n_frames} frames were added"
            )
        self._advance(final=True)
        return finish_maps(self)


class _BlockChannel:
    """One channel's sums over one block (frames ``[b0, b1)``)."""

    def __init__(self, owner, ch, rows: dict, span):
        """Rows of the span ``[s0, ...)`` for channel ``ch``."""
        self.o = owner
        self.ch = ch
        self.acc = owner.acc
        self.s0, self.b0, self.b1 = span
        self.valid = rows["v"][ch]
        self.v = self.valid.double()
        self.x = rows["x"][ch]
        self.x0 = torch.where(self.valid, self.x, 0.0).double()
        vf = self.valid.float()
        self.g0 = rows["G0"][0] * vf
        self.r0 = rows["R0"][0] * vf
        self.n0 = rows["N0"][0]
        self.f, self.b, self.cr, self.k = (rows[n][ch] for n in PER_CHANNEL)
        self.rows = slice(self.b0 - self.s0, self.b1 - self.s0)
        self.bin_means = None

    def add_all(self) -> None:
        """Every sum of this block."""
        self._single()
        if self.b1 - self.b0 == BLOCK:
            self._slow()
        self.bin_means = self._bins()
        self._fast()
        self._zero_hist()

    def _add(self, name: str, value: torch.Tensor) -> None:
        """Add to the per-pixel sums ``name`` of this channel."""
        self.acc[name][self.ch] += value

    def _pairs_within(self, a: int, b: int) -> torch.Tensor:
        """``sum_{t, u in [a, b)} A2(t, u)`` over valid pairs (span rows)."""
        own = self.g0[a:b].sum(0) + 2 * self.f[a:b].sum(0)
        return own - 2 * self.cr[b - 1]

    def _single(self) -> None:
        """Single-frame moments and their photon-noise sums."""
        rows = self.rows
        v, x0 = self.v[rows], self.x0[rows]
        self._add("n", v.sum(0))
        self._add("sx", x0.sum(0))
        self._add("sx2", (x0 * x0).sum(0))
        zero = self.valid[rows] & (self.x[rows] == 0)
        self._add("zeros", zero.double().sum(0))
        self._add("s0", self.g0[rows].sum(0).double())
        self._add("s1", self.f[rows].sum(0).double())

    def _slow(self) -> None:
        """The mean of the block, its photon-noise variance and its
        covariance with the mean of the previous block."""
        a, b = self.rows.start, self.rows.stop
        nj = self.v[a:b].sum(0)
        okj = nj >= BLOCK_MIN
        mean = self.x0[a:b].sum(0) / nj.clamp(min=1)
        w = self._pairs_within(a, b).double()
        cross = self.cr[a - 1].double() if a > 0 else torch.zeros_like(nj)
        zero = torch.zeros_like(nj)
        prev_n = self.acc["slow_prev_n"][self.ch]
        both = okj & (prev_n >= BLOCK_MIN)
        self._add("nb", okj.double())
        self._add("sb", torch.where(okj, mean, zero))
        self._add("sb2", torch.where(okj, mean * mean, zero))
        self._add("vb", torch.where(okj, w / nj.clamp(min=1) ** 2, zero))
        den = (nj * prev_n).clamp(min=1)
        self._add("cb", torch.where(both, cross / den, zero))
        self.acc["slow_prev_n"][self.ch] = nj

    def _bins(self) -> torch.Tensor:
        """Means of the full 8-frame bins of the block (NaN where fewer
        than :data:`BIN_T_MIN` frames are valid); adds their photon-noise
        variances."""
        a = self.rows.start
        nb = (self.b1 - self.b0) // BIN_T
        n = self.v.shape[1]
        if nb == 0:
            return self.x.new_zeros((0, n))
        rows = slice(a, a + nb * BIN_T)
        cnt = self.v[rows].reshape(nb, BIN_T, n).sum(1)
        ok = cnt >= BIN_T_MIN
        mean = self.x0[rows].reshape(nb, BIN_T, n).sum(1) / cnt.clamp(min=1)
        g = self.g0[rows].reshape(nb, BIN_T, n).sum(1)
        f = self.f[rows].reshape(nb, BIN_T, n).sum(1)
        ends = self.cr[a + BIN_T - 1 : a + nb * BIN_T : BIN_T]
        w = (g + 2 * f - 2 * ends).double()
        var = torch.where(ok, w / cnt.clamp(min=1) ** 2, 0.0)
        self._add("v8", var.sum(0))
        self._add("n8", ok.double().sum(0))
        return torch.where(ok, mean, float("nan")).float()

    def _fast_rows(self):
        """Centres of the block (span rows) and their window bounds."""
        nf = self.o.n_frames
        c0 = max(self.b0, FAST_HALF)
        c1 = min(self.b1, nf - FAST_HALF)
        if c1 <= c0:
            return None
        rc = torch.arange(c0, c1, device=self.v.device) - self.s0
        return rc, rc - FAST_HALF, rc + FAST_HALF

    def _fast(self) -> None:
        """Fast part: frame minus its 65-frame running mean."""
        rows = self._fast_rows()
        if rows is None:
            return
        rc, lo, hi = rows
        n = _window(_cumsum0(self.v), lo, hi)
        s = _window(_cumsum0(self.x0), lo, hi)
        ok = self.valid[rc] & (n >= FAST_MIN)
        nn = n.clamp(min=1)
        z = self.x[rc].double() - s / nn
        zero = torch.zeros_like(n)
        z = torch.where(ok, z, zero)
        self._add("nf", ok.double().sum(0))
        self._add("sz", z.sum(0))
        self._add("sz2", (z * z).sum(0))
        self._add("sz3", (z * z * z).sum(0))
        q2, q3 = self._fast_photon(rc, lo, hi, nn.float())
        self._add("q2", torch.where(ok, q2.double(), zero).sum(0))
        self._add("q3", torch.where(ok, q3.double(), zero).sum(0))

    def _fast_photon(self, rc, lo, hi, n):
        """Photon-noise variance and third cumulant of the fast part of
        each centre, per unit ``g2 mu`` and ``g3 mu`` (see the docs).

        ``Var(Z_t) = A2(t, t) - 2 L_t / n + P_t / n^2`` with ``L_t`` the
        sum of ``A2(t, u)`` over the window and ``P_t`` the sum over pairs
        of window frames; ``kappa3(Z_t) = R0 - 3 K_t / n + 3 L_t^2 / n^2 -
        sum_u L_u^2 / n^3`` (the last two terms approximate the products of
        three coefficients of which two are of other frames).
        """
        lsum = self.g0 + self.f + self.b
        cs_pairs = _cumsum0(self.g0 + 2 * self.f)
        pairs = _window(cs_pairs, lo, hi) - 2 * self.cr[hi]
        lc = lsum[rc]
        q2 = self.g0[rc] - 2 * lc / n + pairs / n**2
        l2 = _window(_cumsum0(lsum * lsum), lo, hi)
        q3 = self.r0[rc] - 3 * self.k[rc] / n + 3 * lc * lc / n**2
        return q2, q3 - l2 / n**3

    def _zero_hist(self) -> None:
        """Count valid frames by their number of contributing samples."""
        o = self.o
        rows = self.rows
        n = self.v.shape[1]
        if o.hist is None:
            o.hist = torch.zeros(
                (o.c, n, o.n_hist), dtype=torch.float64, device=o.device
            )
        idx = self.n0[rows].round().long().clamp(0, o.n_hist - 1)
        o.hist[self.ch].scatter_add_(1, idx.T, self.v[rows].T)
