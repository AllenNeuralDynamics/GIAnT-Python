"""Which raw samples enter each registered pixel, and with what weight.

A registered frame is linear in the raw samples. Frame formation gives raw
pixel ``k`` at frame ``t`` the weighted mean ``sum_s w_s(t) x_s / F1_k(t)``
of its samples, with exp-kernel weights ``w_s(t) = exp(-|t - l_s| / dt)``
for samples within ``max(3 dt, P)`` lines (:mod:`..frames`). Registration
resamples it by a rigid shift with bilinear coefficients ``c_t(a, b)`` on
the four raw pixels around each output pixel, weighted by their summed
weights (:mod:`..interpolation`):

    X_t(p) = sum_ab c_t(ab) sum_{s in k_ab} w_s(t) x_s / D_t(p),
    D_t(p) = sum_ab c_t(ab) F1_{k_ab}(t) = 1 / varFacDS.

So ``X_t(p) = sum_s a_ts x_s`` with ``a_ts = c_t(ab) w_s(t) / D_t(p)``.
Under photon noise (:mod:`.gain`) and a pixel brightness ``mu`` that is the
same for those samples, ``Cov(X_t, X_u) = g2 mu A2(t, u)`` and the third
cross-cumulant is ``g3 mu A3(t, t, u)``, where

    A2(t, u)    = sum_s a_ts a_us,
    A3(t, t, u) = sum_s a_ts^2 a_us,   A3(t, u, u) = sum_s a_ts a_us^2.

Frames ``t`` and ``t + l`` share samples only for ``l <= n_lags`` (both
kernels must reach the sample: ``l * spacing <= 2 max(3 dt, P)``; 6 at
80 Hz on SLAP2). This module computes these sums exactly from the raster's
layout and the frame times, assuming every sample in reach is valid, and
rescales them with the actual ``D_t(p)`` from ``varFacDS``: where samples
were invalid, the sums of the remaining ones are taken as proportional
(``F_j ~ (D_actual / D_all_valid) F_j``), which is exact when none is.

With online compensation (:class:`..sources.CompensatedFrames`) each
sample is placed by its line's displacement for the frame; a sample shared
by two frames is placed by the earlier frame's displacement (the two differ
only for frames whose reference offset differs, around an x/y switch).

Per frame the outputs are, at a set of output pixels:

* ``G[0] = A2(t, t)``, ``G[l] = A2(t, t + l)`` (``l = 1 .. n_lags``);
* ``R0 = A3(t, t, t)``, ``R[l] = A3(t, t, t + l)``,
  ``Rp[l] = A3(t, t + l, t + l)``;
* ``N0``: the number of samples with a non-zero coefficient (for the
  photon-noise share of zeros).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ..compensation import line_shifts

MAX_LAGS = 7
"""Largest lag supported: below the SVD's 8-frame bins, so that a pair of
frames that share samples never spans a whole bin (SLAP2 rasters reach it
at about 530 lines per cycle; 80 Hz frames share samples over 6 frames
up to 402 lines)."""

PAIR_ELEMENTS = 20_000_000
"""Largest pair-sum array formed at once (lag quantities x frames x
pixels; bounds memory)."""

CORNERS = ((0, 0), (0, 1), (1, 0), (1, 1))
"""Bilinear neighbours ``(a, b)`` in ``interp_frame``'s order."""


def bilinear_corners(motion_r: float, motion_c: float, margin: int):
    """Offsets and coefficients of ``interp_frame``'s four neighbours.

    Output pixel ``(i, j)`` reads raw window pixels ``(i + off_r + a,
    j + off_c + b)`` for ``a, b`` in ``{0, 1}`` with coefficients in the
    order ``(0, 0), (0, 1), (1, 0), (1, 1)``, rounded to float32 as
    ``interp_frame`` rounds them.

    Returns
    -------
    off_r, off_c : int
    coef : ndarray of float64, shape (4,)
    frac : (float, float)
        The fractional row and column shifts.
    """
    first_r = 1 - margin + float(motion_r)
    first_c = 1 - margin + float(motion_c)
    fr = first_r - math.floor(first_r)
    fc = first_c - math.floor(first_c)
    coef = np.array(
        [
            float(np.float32(v))
            for v in (
                (1 - fc) * (1 - fr),
                fc * (1 - fr),
                (1 - fc) * fr,
                fc * fr,
            )
        ]
    )
    return math.floor(first_r) - 1, math.floor(first_c) - 1, coef, (fr, fc)


def spatial_factor(frac) -> float:
    """Variance of a 2 x 2 sum of registered pixels relative to four times
    one pixel's, from the shared raw pixels of bilinear interpolation
    (equal raw pixels assumed): ``(1 + rho_r) (1 + rho_c)``."""
    out = 1.0
    for f in frac:
        out *= 1.0 + f * (1 - f) / ((1 - f) ** 2 + f**2)
    return out


def _pair_products(w_first, w_partner, cmin, lag, n_first):
    """Kernel products of frames ``f`` and ``f + lag`` per line.

    Parameters
    ----------
    w_first : ndarray ``(n, P, k)``
        Weights of each frame's k-th sample of each line (0 if out of reach
        or not in the displacement group), used for frame ``f``.
    w_partner : ndarray ``(n, P, k)``
        The same weights for frame ``f + lag`` (all samples in reach).
    cmin : ndarray ``(n, P)``
        Cycle of each frame's first (k = 0) sample of each line.
    lag : int
        Frame lag.
    n_first : int
        Frames ``f`` (``f + lag`` must be below ``n``).

    Returns
    -------
    ndarray ``(3, n_first, P, k)``
        ``w_f w_g``, ``w_f^2 w_g`` and ``w_f w_g^2`` at frame ``f``'s
        samples (``g = f + lag``).
    """
    n, _, k = w_first.shape
    out = np.zeros((3, n_first) + w_first.shape[1:])
    m = max(0, min(n_first, n - lag))
    if m == 0:
        return out
    wa = w_first[:m]
    shift = cmin[lag : lag + m] - cmin[:m]
    kk = np.arange(k)[None, None, :] - shift[:, :, None]
    inside = (kk >= 0) & (kk < k)
    wb = np.take_along_axis(w_partner[lag : lag + m], np.clip(kk, 0, k - 1), 2)
    wb = np.where(inside, wb, 0.0)
    out[0, :m] = wa * wb
    out[1, :m] = wa * wa * wb
    out[2, :m] = wa * wb * wb
    return out


class FrameGeometry:
    """Photon-noise sums of a trial's registered frames.

    Parameters
    ----------
    source : PlainFrames or CompensatedFrames
        The trial's frame source (its former holds the layout and reads
        the trial's cycles).
    ds : array_like of int
        One-based line times of the registered frames.
    margin : int
        Padding of the registered grid (``maxshift``).

    Attributes
    ----------
    n_lags : int
        Largest lag with shared samples.
    out_shape : (int, int)
        Registered frame shape.
    """

    def __init__(self, source, ds, margin: int):
        """Pixel tables of the window and the frames' kernel range."""
        self.source = source
        self.former = source.former
        self.device = self.former.device
        self.dt = int(source.dt)
        self.ds = np.asarray(ds, dtype=np.int64).ravel()
        self.margin = int(margin)
        lay = self.former.layout
        self.reach = max(3 * self.dt, lay.lines_per_cycle)
        spacing = int(np.diff(self.ds).min()) if self.ds.size > 1 else 1
        self.n_lags = int((2 * self.reach) // max(1, spacing))
        if self.n_lags > MAX_LAGS:
            raise ValueError(
                f"frames share samples over {self.n_lags} frames; the "
                f"statistics support at most {MAX_LAGS} (a raster cycle "
                f"of {lay.lines_per_cycle} lines is too long)"
            )
        self.window = source.window
        h, w = self.window.shape
        self.out_shape = (h + 2 * self.margin, w + 2 * self.margin)
        self.n_win = int(self.window.rows.numel())
        self.lookup = torch.full(
            (h * w,), self.n_win, dtype=torch.long, device=self.device
        )
        flat = self.window.rows * w + self.window.cols
        self.lookup[flat] = torch.arange(self.n_win, device=self.device)
        pix = self.former._pix_flat
        self.pixel_lines = torch.where(
            pix >= 0,
            torch.div(pix, self.former._line_width, rounding_mode="floor"),
            -1,
        )
        self.refs = None
        if source.compensation is not None:
            self.refs = source.references(self.ds)
        self._incidence = {}

    # ------------------------------------------------------------ raw sums

    def _destinations(self, dr: int, dc: int):
        """Former pixels and their window-list positions at a displacement."""
        if self.source.compensation is None:
            keep = self.source.pixels
            return keep, torch.arange(keep.numel(), device=self.device)
        return self.source._destination(dr, dc)

    def incidence(self, dr: int, dc: int) -> torch.Tensor:
        """``(P, n_win + 1)`` float32: line ``l`` has a slot of the window
        pixel (the last column, for raw pixels without samples, is 0)."""
        key = (int(dr), int(dc))
        if key not in self._incidence:
            p = self.former.layout.lines_per_cycle
            m = torch.zeros(
                (p, self.n_win + 1), dtype=torch.float32, device=self.device
            )
            keep, dest = self._destinations(*key)
            lines = self.pixel_lines[keep]
            for j in range(lines.shape[1]):
                ok = lines[:, j] >= 0
                m.index_put_(
                    (lines[ok, j], dest[ok]),
                    torch.ones(int(ok.sum()), device=self.device),
                    accumulate=True,
                )
            self._incidence[key] = m
        return self._incidence[key]

    def _groups(self, f0: int, f_ext: int, lines, ok):
        """``[((dr, dc), mask)]``: samples of each displacement."""
        if self.refs is None:
            return [((0, 0), ok)]
        dr, dc = line_shifts(
            self.source.compensation, lines, self.refs[f0:f_ext]
        )
        pairs = np.unique(np.stack([dr[ok], dc[ok]], 1), axis=0)
        return [
            ((int(r), int(c)), ok & (dr == r) & (dc == c)) for r, c in pairs
        ]

    def line_tables(self, f0: int, f1: int, f_ext: int) -> list:
        """Kernel sums per line, by displacement group.

        Parameters
        ----------
        f0, f1 : int
            Frames whose pair sums are wanted.
        f_ext : int
            End of the frames read (partners ``f + l`` must be below it).

        Returns
        -------
        list of ((dr, dc), single, pair)
            ``single``: ``(4, f_ext - f0, P)`` sample count, ``sum w``,
            ``sum w^2``, ``sum w^3``; ``pair``: ``(3 n_lags, f1 - f0, P)``
            ``H_l``, ``T_l``, ``U_l`` for ``l = 1 .. n_lags`` (lag-major).
        """
        local, w, lines, ok = self.former._tables(self.ds[f0:f_ext], self.dt)
        cmin = local[:, :, 0]
        n_first = f1 - f0
        out = []
        for key, mask in self._groups(f0, f_ext, lines, ok):
            wg = np.where(mask, w, 0.0)
            single = np.stack(
                [mask.sum(2), wg.sum(2), (wg**2).sum(2), (wg**3).sum(2)]
            ).astype(np.float64)
            pair = np.zeros((3 * self.n_lags, n_first, w.shape[1]))
            for lag in range(1, self.n_lags + 1):
                # a shared sample is placed by the earlier frame's group
                prod = _pair_products(wg, w, cmin, lag, n_first)
                pair[3 * (lag - 1) : 3 * lag] = prod.sum(3)
            out.append((key, single, pair))
        return out

    def _raw(self, tables: list, which: int, frames: slice) -> torch.Tensor:
        """Line tables summed over each window pixel's lines (and the
        displacement groups): ``(quantities, frames, n_win + 1)`` float32,
        0 in the last column."""
        total = None
        for key, *arrays in tables:
            t = arrays[which - 1][:, frames]
            t = torch.from_numpy(t.astype(np.float32)).to(self.device)
            m = self.incidence(*key)
            img = (t.reshape(-1, t.shape[-1]) @ m).reshape(
                t.shape[:2] + (m.shape[1],)
            )
            total = img if total is None else total + img
        return total

    # ----------------------------------------------------- registered sums

    def corners(self, motion: np.ndarray, present: np.ndarray):
        """Each frame's bilinear offsets, coefficients and fractions.

        Parameters
        ----------
        motion : ndarray ``(n, 2)``
            Row and column motion of the frames.
        present : ndarray of bool ``(n,)``
            Frames with samples (others get zero coefficients).

        Returns
        -------
        off : ndarray of int64 ``(n, 2)``
        coef : ndarray of float64 ``(n, 4)``
        frac : ndarray of float64 ``(n, 2)``
        """
        n = motion.shape[0]
        off = np.zeros((n, 2), dtype=np.int64)
        coef = np.zeros((n, 4))
        frac = np.zeros((n, 2))
        for i in np.flatnonzero(present):
            r, c, cf, fr = bilinear_corners(*motion[i], self.margin)
            off[i], coef[i], frac[i] = (r, c), cf, fr
        return off, coef, frac

    def corner_index(self, out_idx: torch.Tensor, off: np.ndarray):
        """``(n, 4, n_idx)`` window-list index of each output pixel's raw
        neighbours (``n_win`` outside the window or where unsampled)."""
        h, w = self.window.shape
        big_w = self.out_shape[1]
        rr = torch.div(out_idx, big_w, rounding_mode="floor")
        cc = out_idx - rr * big_w
        offs = torch.from_numpy(off).to(self.device)
        out = []
        for a, b in CORNERS:
            r = rr[None, :] + offs[:, 0:1] + a
            c = cc[None, :] + offs[:, 1:2] + b
            inside = (r >= 0) & (r < h) & (c >= 0) & (c < w)
            flat = torch.where(inside, r * w + c, 0)
            out.append(torch.where(inside, self.lookup[flat], self.n_win))
        return torch.stack(out, 1)

    def pair_coefficients(self, off, coef, n_first) -> np.ndarray:
        """Products of the corner coefficients of frames ``t`` and ``t + l``
        at the raw pixels the two share.

        Returns
        -------
        ndarray ``(3 n_lags, n_first, 4)``
            Lag-major ``c_t c_u``, ``c_t^2 c_u``, ``c_t c_u^2`` per corner
            of ``t`` (``u = t + l``; 0 past the frames given).
        """
        n = off.shape[0]
        out = np.zeros((3 * self.n_lags, n_first, 4))
        ab = np.array(CORNERS)
        for lag in range(1, self.n_lags + 1):
            m = max(0, min(n_first, n - lag))
            t = np.arange(m)
            d = off[t] - off[t + lag]
            a2 = ab[None, :, 0] + d[:, 0:1]
            b2 = ab[None, :, 1] + d[:, 1:2]
            ok = (a2 >= 0) & (a2 <= 1) & (b2 >= 0) & (b2 <= 1)
            k2 = np.clip(2 * a2 + b2, 0, 3)
            cu = np.where(ok, np.take_along_axis(coef[t + lag], k2, 1), 0.0)
            ct = coef[t]
            q = 3 * (lag - 1)
            out[q, :m] = ct * cu
            out[q + 1, :m] = ct * ct * cu
            out[q + 2, :m] = ct * cu * cu
        return out

    def _norms(self, single, cidx, coef, varfac) -> dict:
        """Per-frame normalisers and single-frame sums at output pixels.

        Returns
        -------
        dict of torch.Tensor ``(n, n_idx)`` float32
            ``n0``, ``d_g`` (all-valid ``D``), ``q2`` and ``q3`` (the summed
            powers), ``n2 = sqrt(d_g d_a)``, ``n3 = (d_g d_a^2)^(1/3)``
            (``d_a = 1 / varFacDS``) and ``ok``.
        """
        c = torch.from_numpy(coef.astype(np.float32)).to(self.device)
        acc = [0.0, 0.0, 0.0, 0.0]
        for k in range(4):
            g = torch.gather(single, 2, cidx[:, k][None].expand(4, -1, -1))
            ck = c[:, k : k + 1]
            pw = ((ck > 0).float(), ck, ck * ck, ck * ck * ck)
            for j in range(4):
                acc[j] = acc[j] + pw[j] * g[j]
        n0, d_g, q2, q3 = acc
        d_a = 1.0 / varfac.float()
        ok = torch.isfinite(d_a) & (d_a > 0) & (d_g > 0)
        zero = torch.zeros_like(d_g)
        n2 = torch.where(ok, torch.sqrt(d_g * d_a), zero)
        n3 = torch.where(ok, (d_g * d_a * d_a) ** (1.0 / 3), zero)
        return dict(n0=n0, d_g=d_g, q2=q2, q3=q3, n2=n2, n3=n3, ok=ok)

    def registered(self, f0, f1, f_ext, out_idx, motion, present, varfac):
        """Photon-noise sums of frames ``[f0, f1)`` at output pixels.

        Parameters
        ----------
        f0, f1, f_ext : int
            Frames wanted, and the end of the frames given (partners).
        out_idx : torch.Tensor of int64 ``(n_idx,)``
            Flat indices into the registered grid.
        motion : ndarray ``(f_ext - f0, 2)``
        present : ndarray of bool ``(f_ext - f0,)``
        varfac : torch.Tensor ``(f_ext - f0, n_idx)``
            ``varFacDS`` at the output pixels (NaN where undefined).

        Returns
        -------
        dict of torch.Tensor (float32)
            ``G`` ``(1 + n_lags, n, n_idx)``, ``R0`` ``(n, n_idx)``, ``R``
            and ``Rp`` ``(n_lags, n, n_idx)``, ``N0`` ``(n, n_idx)``, with
            ``n = f1 - f0``; 0 where a frame has no defined value.
        """
        n = f1 - f0
        off, coef, _ = self.corners(motion, present)
        cidx = self.corner_index(out_idx, off)
        tables = self.line_tables(f0, f1, f_ext)
        single = self._raw(tables, 1, slice(None))
        nrm = self._norms(single, cidx, coef, varfac)
        ok = nrm["ok"][:n]
        zero = torch.zeros_like(nrm["d_g"][:n])
        g0 = torch.where(ok, nrm["q2"][:n] / nrm["n2"][:n] ** 2, zero)
        out = {
            "R0": torch.where(ok, nrm["q3"][:n] / nrm["n3"][:n] ** 3, zero),
            "N0": torch.where(ok, nrm["n0"][:n], zero),
        }
        lag3 = 3 * self.n_lags
        pairs = zero.new_zeros((lag3, n, zero.shape[1]))
        if self.n_lags:
            cc = self.pair_coefficients(off, coef, n)
            cc = torch.from_numpy(cc.astype(np.float32)).to(self.device)
            step = max(1, int(PAIR_ELEMENTS // max(1, lag3 * zero.shape[1])))
            for a in range(0, n, step):
                b = min(n, a + step)
                pairs[:, a:b] = self._pair_part(tables, cidx, cc, nrm, a, b)
        out["G"] = torch.cat([g0[None], pairs[0::3]])
        out["R"] = pairs[1::3]
        out["Rp"] = pairs[2::3]
        return out

    def _pair_part(self, tables, cidx, cc, nrm, a: int, b: int):
        """``A2``, ``A3(t, t, u)``, ``A3(t, u, u)`` of frames ``[a, b)``
        (relative to the chunk) with ``u = t + l``: ``(3 n_lags, b - a,
        n_idx)`` lag-major."""
        raw = self._raw(tables, 2, slice(a, b))
        lag3 = raw.shape[0]
        acc = None
        for k in range(4):
            g = torch.gather(raw, 2, cidx[a:b, k][None].expand(lag3, -1, -1))
            term = cc[:, a:b, k : k + 1] * g
            acc = term if acc is None else acc + term
        n2, n3, ok = nrm["n2"], nrm["n3"], nrm["ok"]
        m = n2.shape[0]
        dens, both = [], []
        for lag in range(1, self.n_lags + 1):
            u = torch.arange(a, b, device=self.device) + lag
            has = (u < m)[:, None]
            uc = u.clamp(max=m - 1)
            both.append(ok[a:b] & ok[uc] & has)
            dens += [
                n2[a:b] * n2[uc],
                n3[a:b] * n3[a:b] * n3[uc],
                n3[a:b] * n3[uc] * n3[uc],
            ]
        mask = torch.stack(both).repeat_interleave(3, 0)
        den = torch.stack(dens)
        return torch.where(mask, acc / den, torch.zeros_like(acc))
