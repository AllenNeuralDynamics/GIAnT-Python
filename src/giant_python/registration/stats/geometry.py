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

Consecutive samples on a line share photons (:mod:`.gain`), so a registered
pixel's photon noise also has terms from pairs ``(s, n)`` of consecutive
samples that both enter it (in two of its raw pixels, usually neighbours
along the line): ``2 c2 mu sum a_s a_n`` in the variance and ``3 mu sum
(c3a a_s^2 a_n + c3b a_s a_n^2)`` in the third cumulant. Both samples are
on the same line, so they have the same kernel weight. Per frame:

* ``QA2 = sum a_s a_n``, ``QA3a = sum a_s^2 a_n``, ``QA3b = sum a_s a_n^2``
  over such pairs, and ``M0``, the number of pairs both of whose samples
  have a non-zero coefficient.

These are exact for a frame with itself; for two frames that share samples
the accumulator scales them by the pixel's own ratio (see ``maps``).
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

NEIGHBOURS = tuple(
    (dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)
)
"""Directions from a sample's pixel to the next sample's pixel that two
corners of one output pixel can have."""


def corner_pairs(delta):
    """``[(k, k2)]``: corners ``k`` and ``k2`` with ``k2 = k + delta``."""
    out = []
    for k, (a, b) in enumerate(CORNERS):
        a2, b2 = a + delta[0], b + delta[1]
        if 0 <= a2 <= 1 and 0 <= b2 <= 1:
            out.append((k, 2 * a2 + b2))
    return out


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


def pair_bin_factor(frac, shares: dict) -> float:
    """Consecutive-sample pairs inside a 2 x 2 sum of registered pixels,
    relative to four times one pixel's own single-sample variance (equal
    raw pixels assumed, as :func:`spatial_factor`).

    Per axis with fractional shift ``f`` the sum's raw-pixel weights are
    ``(1 - f, 1, f)``: ``A = (1 - f)^2 + 1 + f^2`` along an axis the pair
    does not step on, ``1`` along one it steps on, against ``Q = (1 - f)^2
    + f^2`` for one pixel. With ``pi_d`` the share of samples whose next
    sample is in direction ``d``: ``sum_d pi_d S_d / (4 Q_r Q_c)``.

    Parameters
    ----------
    frac : (float, float)
        The frame's fractional row and column shifts.
    shares : dict
        ``{(dr, dc): pi_d}`` (:attr:`FrameGeometry.pair_share`).
    """
    q = [(1 - f) ** 2 + f**2 for f in frac]
    a = [(1 - f) ** 2 + 1 + f**2 for f in frac]
    total = 0.0
    for (dr, dc), share in shares.items():
        total += (
            share * (a[0] if dr == 0 else 1.0) * (a[1] if dc == 0 else 1.0)
        )
    return total / (4 * q[0] * q[1])


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
        # no partners beyond the trial (a one-frame trial has none)
        self.n_lags = min(
            int((2 * self.reach) // max(1, spacing)), max(0, self.ds.size - 1)
        )
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
        self._adjacent = {}
        self.slot_dirs = self._slot_directions()
        self.pair_share = self._pair_shares()
        self.directions = sorted(
            {tuple(int(v) for v in d) for d in self.slot_dirs[1].tolist()}
        )
        self.tables = self._precompute()

    # ------------------------------------------------------------ raw sums

    def _destinations(self, dr: int, dc: int):
        """Former pixels and their window-list positions at a displacement."""
        if self.source.compensation is None:
            keep = self.source.pixels
            return keep, torch.arange(keep.numel(), device=self.device)
        return self.source._destination(dr, dc)

    def _slot_directions(self):
        """Slots of the plane whose next sample is in a neighbouring pixel.

        Returns
        -------
        slots : ndarray of int
            Layout slot indices.
        dirs : ndarray of int ``(n, 2)``
            Row and column from the slot's pixel to the next sample's.
        """
        from .gain import next_slots

        lay = self.former.layout
        nxt = next_slots(lay.slot_line, lay.slot_pos)
        on_plane = lay.line_fast_z[lay.slot_line - 1] == 1
        s = np.flatnonzero((nxt >= 0) & on_plane)
        n = nxt[s]
        pr, pc = (lay.slot_pixel - 1) // lay.n_cols, (
            lay.slot_pixel - 1
        ) % lay.n_cols
        d = np.stack([pr[n] - pr[s], pc[n] - pc[s]], 1)
        near = (np.abs(d) <= 1).all(1) & (d != 0).any(1)
        return s[near], d[near]

    def _pair_shares(self) -> dict:
        """``{(dr, dc): share}``: the plane's samples whose next sample is
        in the neighbouring pixel ``(dr, dc)``, as a share of all of them."""
        lay = self.former.layout
        n = int((lay.line_fast_z[lay.slot_line - 1] == 1).sum())
        dirs = self.slot_dirs[1]
        if n == 0 or dirs.size == 0:
            return {}
        keys, counts = np.unique(dirs, axis=0, return_counts=True)
        return {
            (int(r), int(c)): float(k) / n for (r, c), k in zip(keys, counts)
        }

    def adjacency(self, dr: int, dc: int, delta) -> torch.Tensor:
        """``(P, n_win + 1)`` float32: slots on line ``l`` of the window
        pixel whose next sample is at ``pixel + delta`` (displacement
        ``(dr, dc)``)."""
        key = (int(dr), int(dc), tuple(delta))
        if key not in self._adjacent:
            lay = self.former.layout
            p = lay.lines_per_cycle
            m = torch.zeros(
                (p, self.n_win + 1), dtype=torch.float32, device=self.device
            )
            slots, dirs = self.slot_dirs
            sel = slots[(dirs == np.asarray(delta)).all(1)]
            keep, dest = self._destinations(dr, dc)
            where = np.full(self.former.pixels.size, -1, dtype=np.int64)
            where[keep.cpu().numpy()] = dest.cpu().numpy()
            pix = np.searchsorted(self.former.pixels, lay.slot_pixel[sel])
            to = where[pix]
            ok = to >= 0
            lines = torch.from_numpy(lay.slot_line[sel][ok] - 1)
            m.index_put_(
                (
                    lines.to(self.device),
                    torch.from_numpy(to[ok]).to(self.device),
                ),
                torch.ones(int(ok.sum()), device=self.device),
                accumulate=True,
            )
            self._adjacent[key] = m
        return self._adjacent[key]

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

    def _precompute(self) -> dict:
        """Line tables of every frame, by displacement group, on the device.

        Returns
        -------
        dict
            ``(dr, dc) -> (rows, single, pair)``: ``rows`` ``(frames,)``
            int64, each frame's row in the group's tables (-1 if the group
            has no samples of it); ``single`` ``(4, n_g, P)`` and ``pair``
            ``(3 n_lags, n_g, P)`` float32 (see :meth:`line_tables`).
        """
        n = self.ds.size
        parts = {}
        for a in range(0, n, 512):
            b = min(n, a + 512)
            e = min(n, b + self.n_lags)
            for key, single, pair in self.line_tables(a, b, e):
                has = single[0, : b - a].sum(1) > 0
                frames = a + np.flatnonzero(has)
                parts.setdefault(key, []).append(
                    (frames, single[:, : b - a][:, has], pair[:, has])
                )
        out = {}
        for key, chunks in parts.items():
            frames = np.concatenate([c[0] for c in chunks])
            rows = np.full(n, -1, dtype=np.int64)
            rows[frames] = np.arange(frames.size)
            tabs = [
                torch.from_numpy(
                    np.concatenate([c[i] for c in chunks], 1).astype(
                        np.float32
                    )
                ).to(self.device)
                for i in (1, 2)
            ]
            out[key] = (rows, tabs[0], tabs[1])
        return out

    def _raw(self, which: int, a: int, b: int) -> torch.Tensor:
        """Line tables of frames ``[a, b)`` summed over each window pixel's
        lines and the displacement groups: ``(quantities, b - a, n_win +
        1)`` float32 (``which`` 1: single-frame sums, 2: pair sums), 0 in
        the last column."""
        total = None
        for key, (rows, single, pair) in self.tables.items():
            r = rows[a:b]
            sel = np.flatnonzero(r >= 0)
            if sel.size == 0:
                continue
            tab = single if which == 1 else pair
            if sel.size == b - a and r[-1] - r[0] == b - a - 1:
                t = tab[:, r[0] : r[-1] + 1]
            else:
                t = tab[:, torch.from_numpy(r[sel]).to(self.device)]
            img = self._images(key, t, which)
            if sel.size != b - a:
                full = img.new_zeros((img.shape[0], b - a, img.shape[2]))
                full[:, torch.from_numpy(sel).to(self.device)] = img
                img = full
            total = img if total is None else total + img
        if total is None:
            q = 4 + 3 * len(self.directions) if which == 1 else 3 * self.n_lags
            total = torch.zeros((q, b - a, self.n_win + 1), device=self.device)
        return total

    def _images(self, key, t: torch.Tensor, which: int) -> torch.Tensor:
        """Line tables ``t`` ``(q, frames, P)`` on the window pixels; for the
        single sums also, per direction of :attr:`directions`, ``sum w^2``,
        ``sum w^3`` and the sample count over the slots whose next sample is
        in that direction (appended in that order)."""
        flat = t.reshape(-1, t.shape[-1])
        m = self.incidence(*key)
        imgs = [(flat @ m).reshape(t.shape[:2] + (m.shape[1],))]
        if which == 1:
            sub = t[[2, 3, 0]].reshape(-1, t.shape[-1])
            for delta in self.directions:
                ma = self.adjacency(key[0], key[1], delta)
                imgs.append(
                    (sub @ ma).reshape((3,) + t.shape[1:2] + (m.shape[1],))
                )
        return torch.cat(imgs) if len(imgs) > 1 else imgs[0]

    # ----------------------------------------------------- registered sums

    def corners(self, motion: np.ndarray, present: np.ndarray):
        """Each frame's bilinear offsets, coefficients and fractions
        (:func:`bilinear_corners`, for many frames).

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
        present = np.asarray(present, dtype=bool)
        first = (1 - self.margin) + np.where(
            present[:, None], np.asarray(motion, np.float64), 0.0
        )
        whole = np.floor(first)
        frac = first - whole
        fr, fc = frac[:, 0], frac[:, 1]
        coef = np.stack(
            [(1 - fc) * (1 - fr), fc * (1 - fr), (1 - fc) * fr, fc * fr], 1
        )
        coef = coef.astype(np.float32).astype(np.float64)
        off = (whole - 1).astype(np.int64)
        coef[~present] = 0.0
        off[~present] = 0
        frac[~present] = 0.0
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
        adj = [0.0, 0.0, 0.0, 0.0]
        for k in range(4):
            pairs, rows = self._corner_rows(k)
            sub = single.index_select(0, rows)
            g = torch.gather(
                sub, 2, cidx[:, k][None].expand(len(rows), -1, -1)
            )
            ck = c[:, k : k + 1]
            pw = ((ck > 0).float(), ck, ck * ck, ck * ck * ck)
            for j in range(4):
                acc[j] = acc[j] + pw[j] * g[j]
            self._adjacent_terms(adj, g, c, k, pairs)
        n0, d_g, q2, q3 = acc
        d_a = 1.0 / varfac.float()
        ok = torch.isfinite(d_a) & (d_a > 0) & (d_g > 0)
        zero = torch.zeros_like(d_g)
        n2 = torch.where(ok, torch.sqrt(d_g * d_a), zero)
        n3 = torch.where(ok, (d_g * d_a * d_a) ** (1.0 / 3), zero)
        qa2, qa3a, qa3b, m0 = (a + zero for a in adj)
        return dict(
            n0=n0, d_g=d_g, q2=q2, q3=q3, n2=n2, n3=n3, ok=ok,
            qa2=qa2, qa3a=qa3a, qa3b=qa3b, m0=m0,
        )  # fmt: skip

    def _corner_rows(self, k: int):
        """Corner ``k``'s partners and the rows of the single sums it needs.

        Returns
        -------
        pairs : list of int
            Partner corner ``k2 = k + delta`` for each direction (of
            :attr:`directions`, in order) in which ``k`` is the first
            sample's corner.
        rows : torch.Tensor of int64
            The four own sums, then the three pair sums of each of those
            directions.
        """
        if not hasattr(self, "_rows"):
            self._rows = {}
        if k not in self._rows:
            pairs, rows = [], [0, 1, 2, 3]
            for i, delta in enumerate(self.directions):
                for k1, k2 in corner_pairs(delta):
                    if k1 == k:
                        pairs.append(k2)
                        rows += [4 + 3 * i, 5 + 3 * i, 6 + 3 * i]
            rows = torch.tensor(rows, dtype=torch.long, device=self.device)
            self._rows[k] = (pairs, rows)
        return self._rows[k]

    def _adjacent_terms(self, adj: list, g, c, k: int, pairs) -> None:
        """Add corner ``k``'s pairs of consecutive samples to ``adj``
        (``sum c c' w^2``, ``sum c^2 c' w^3``, ``sum c c'^2 w^3``, pair
        counts), from its gathered sums ``g`` (rows of
        :meth:`_corner_rows`)."""
        c1 = c[:, k : k + 1]
        for j, k2 in enumerate(pairs):
            c2 = c[:, k2 : k2 + 1]
            w2, w3, cnt = g[4 + 3 * j], g[5 + 3 * j], g[6 + 3 * j]
            adj[0] = adj[0] + c1 * c2 * w2
            adj[1] = adj[1] + c1 * c1 * c2 * w3
            adj[2] = adj[2] + c1 * c2 * c2 * w3
            both = ((c1 > 0) & (c2 > 0)).float()
            adj[3] = adj[3] + both * cnt

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
            and ``Rp`` ``(n_lags, n, n_idx)``, ``N0``, ``QA2``, ``QA3a``,
            ``QA3b`` and ``M0`` ``(n, n_idx)``, with ``n = f1 - f0``; 0
            where a frame has no defined value.
        """
        n = f1 - f0
        off, coef, _ = self.corners(motion, present)
        cidx = self.corner_index(out_idx, off)
        single = self._raw(1, f0, f_ext)
        nrm = self._norms(single, cidx, coef, varfac)
        ok = nrm["ok"][:n]
        zero = torch.zeros_like(nrm["d_g"][:n])
        g0 = torch.where(ok, nrm["q2"][:n] / nrm["n2"][:n] ** 2, zero)
        n2s, n3c = nrm["n2"][:n] ** 2, nrm["n3"][:n] ** 3
        out = {
            "R0": torch.where(ok, nrm["q3"][:n] / n3c, zero),
            "N0": torch.where(ok, nrm["n0"][:n], zero),
            "QA2": torch.where(ok, nrm["qa2"][:n] / n2s, zero),
            "QA3a": torch.where(ok, nrm["qa3a"][:n] / n3c, zero),
            "QA3b": torch.where(ok, nrm["qa3b"][:n] / n3c, zero),
            "M0": torch.where(ok, nrm["m0"][:n], zero),
        }
        lag3 = 3 * self.n_lags
        pairs = zero.new_zeros((lag3, n, zero.shape[1]))
        if self.n_lags:
            cc = self.pair_coefficients(off, coef, n)
            cc = torch.from_numpy(cc.astype(np.float32)).to(self.device)
            step = max(1, int(PAIR_ELEMENTS // max(1, lag3 * zero.shape[1])))
            for a in range(0, n, step):
                b = min(n, a + step)
                pairs[:, a:b] = self._pair_part(f0, cidx, cc, nrm, a, b)
        out["G"] = torch.cat([g0[None], pairs[0::3]])
        out["R"] = pairs[1::3]
        out["Rp"] = pairs[2::3]
        return out

    def _pair_part(self, f0: int, cidx, cc, nrm, a: int, b: int):
        """``A2``, ``A3(t, t, u)``, ``A3(t, u, u)`` of frames ``[a, b)``
        (relative to the chunk starting at frame ``f0``) with ``u = t + l``:
        ``(3 n_lags, b - a, n_idx)`` lag-major."""
        raw = self._raw(2, f0 + a, f0 + b)
        lag3 = raw.shape[0]
        acc = None
        for k in range(4):
            g = torch.gather(raw, 2, cidx[a:b, k][None].expand(lag3, -1, -1))
            term = cc[:, a:b, k : k + 1] * g
            acc = term if acc is None else acc + term
        n2, n3, ok = nrm["n2"], nrm["n3"], nrm["ok"]
        m = n2.shape[0]
        lags = torch.arange(1, self.n_lags + 1, device=self.device)
        u = torch.arange(a, b, device=self.device)[None, :] + lags[:, None]
        uc = u.clamp(max=m - 1)
        both = ok[a:b][None] & ok[uc] & (u < m)[:, :, None]
        n2a, n3a = n2[a:b][None], n3[a:b][None]
        n2u, n3u = n2[uc], n3[uc]
        den = torch.stack(
            [n2a * n2u, n3a * n3a * n3u, n3a * n3u * n3u], 1
        ).reshape(acc.shape)
        mask = both.repeat_interleave(3, 0)
        return torch.where(mask, acc / den, torch.zeros_like(acc))
