"""Photon-noise model of raw SLAP2 samples: gain curves over brightness.

A raw sample is the summed height of the photon pulses that arrived during
it. Pulse heights vary (a single photon gives about 1 to 250 raw units), so a
sample with mean ``mu`` has, under photon noise alone, variance
``g2 * mu`` and third cumulant ``g3 * mu`` (compound Poisson), where
``g2 = E[A^2] / E[A]`` and ``g3 = E[A^3] / E[A]`` for pulse height ``A``. On
SLAP2 both depend on brightness (``g2`` is about 110 in the dimmest slots and
130-140 in bright ones), so each is measured per channel as a curve over the
slots' mean.

The curve comes from the trial's own raw samples: for every slot and every
three successive cycles ``x0, x1, x2`` with valid samples, the second
difference ``d = 2 x1 - x0 - x2`` has ``E[d^2] = 6 kappa2`` and
``E[d^3] = 6 kappa3`` and is insensitive to linear drift. Slots are sorted by
their mean and split into :data:`N_BINS` groups of equal size; per group
``g2 = sum(d^2) / (6 sum(x1))`` and ``g3 = sum(d^3) / (6 sum(x1))``. All
sums are exact integers, so the curve does not depend on the device.

The share of exact zeros gives the rate of non-zero samples per sample,
``nu = -log(zero share)``, stored per unit mean (``nu / mu``); it gives the
photon-noise share of zeros of a frame (see ``accumulate``).

Consecutive samples on a line share photons: a pulse that falls on the
boundary between two samples is split between them, so the sample and the
next one (usually the neighbouring pixel) are correlated (about 0.13-0.25 on
SLAP2). For every pair of consecutive samples ``(s, n)`` the same second
differences give ``E[d_s d_n] = 6 cov``, ``E[d_s^2 d_n] = 6 kappa(s, s, n)``
and ``E[d_s d_n^2] = 6 kappa(s, n, n)``; pairs are grouped by their mean like
slots, and each is stored per unit mean (``c2``, ``c3a``, ``c3b``). The
excess of exact zeros in both samples over independence gives the rate of
shared non-zero events, ``nu_s = log(P(both 0) / (P(s 0) P(n 0)))``, stored
relative to the samples' own rate (``zero_shared = nu_s / nu``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ...io.slap2_raster import INVALID_SAMPLE

N_BINS = 16
"""Brightness groups of the gain curve (equal numbers of slots)."""

SLOTS_PER_PASS = {"cuda": 8192, "cpu": 128}
"""Slots summed at a time, per device type (bounds memory; on the CPU
small passes stay in cache: 3x faster). The sums are exact integers, so
the pass size does not change them."""


def _pass_size(device: torch.device) -> int:
    """:data:`SLOTS_PER_PASS` of a device."""
    return SLOTS_PER_PASS.get(device.type, SLOTS_PER_PASS["cpu"])


@dataclass
class GainCurve:
    """One channel's photon-noise gains over brightness.

    Attributes
    ----------
    mu : ndarray of float64
        ``(K,)`` mean raw value per sample of each brightness group
        (increasing).
    g2, g3 : ndarray of float64
        ``(K,)`` variance and third cumulant per unit mean.
    nu_per_mu : ndarray of float64
        ``(K,)`` rate of non-zero samples per unit mean (NaN for a group
        without zeros).
    n_slots : ndarray of int64
        ``(K,)`` slots in each group.
    """

    mu: np.ndarray
    g2: np.ndarray
    g3: np.ndarray
    nu_per_mu: np.ndarray
    n_slots: np.ndarray
    mu_pair: np.ndarray = field(default_factory=lambda: np.zeros(0))
    c2: np.ndarray = field(default_factory=lambda: np.zeros(0))
    c3a: np.ndarray = field(default_factory=lambda: np.zeros(0))
    c3b: np.ndarray = field(default_factory=lambda: np.zeros(0))
    zero_shared: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_pairs: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))

    def at(self, mean) -> dict:
        """Gains at per-sample means ``mean`` (any shape).

        Linear in ``log(mean)`` between the groups' means and constant
        beyond the first and last group (0 for the pair terms of a curve
        without pairs). The non-zero rate uses the groups that have zeros
        (NaN if none has).

        Returns
        -------
        dict of ndarray (float64, the shape of ``mean``)
            ``g2``, ``g3``, ``nu`` (non-zero rate per sample), and the pair
            terms ``c2``, ``c3a``, ``c3b`` (per unit mean).
        """
        mean = np.asarray(mean, dtype=np.float64)
        x = np.log(np.maximum(mean, 1e-12))
        out = {}
        for name, mu, ys in (
            ("single", self.mu, ("g2", "g3", "nu_per_mu")),
            ("pair", self.mu_pair, ("c2", "c3a", "c3b")),
        ):
            for y in ys:
                v = np.asarray(getattr(self, y), dtype=np.float64)
                ok = np.isfinite(v)
                if not ok.any():
                    fill = np.nan if name == "single" else 0.0
                    out[y] = np.full(mean.shape, fill)
                else:
                    out[y] = np.interp(x, np.log(mu[ok]), v[ok])
        out["nu"] = mean * out.pop("nu_per_mu")
        bad = ~np.isfinite(mean)
        return {k: np.where(bad, np.nan, v) for k, v in out.items()}

    def shared_zero_fraction(self) -> float:
        """The pairs' rate of shared non-zero events relative to their own
        rate, over the dimmer half of the pair groups (where zeros are
        frequent; 0 without pairs)."""
        z = self.zero_shared[np.isfinite(self.zero_shared)]
        if z.size == 0:
            return 0.0
        return float(np.median(z[: max(1, (z.size + 1) // 2)]))

    def describe(self) -> dict:
        """The curve as arrays (for the statistics file)."""
        return {
            k: getattr(self, k)
            for k in (
                "mu", "g2", "g3", "nu_per_mu", "n_slots",
                "mu_pair", "c2", "c3a", "c3b", "zero_shared", "n_pairs",
            )
        }  # fmt: skip


def _slot_sums(x: torch.Tensor) -> np.ndarray:
    """Exact per-slot sums of one channel's samples ``(slots, cycles)``.

    Returns
    -------
    ndarray of int64
        ``(slots, 7)``: valid samples, their sum, zeros, triples, and the
        sums of ``x1``, ``d^2`` and ``d^3`` over triples.
    """
    x = x.to(torch.int64)
    valid = x != INVALID_SAMPLE
    x0 = torch.where(valid, x, 0)
    out = [valid.sum(1), x0.sum(1), (valid & (x == 0)).sum(1)]
    ok = valid[:, :-2] & valid[:, 1:-1] & valid[:, 2:]
    d = 2 * x0[:, 1:-1] - x0[:, :-2] - x0[:, 2:]
    d = torch.where(ok, d, 0)
    mid = torch.where(ok, x0[:, 1:-1], 0)
    out += [ok.sum(1), mid.sum(1), (d * d).sum(1), (d * d * d).sum(1)]
    return torch.stack(out, 1).cpu().numpy()


def slot_sums(samples: torch.Tensor, channel: int) -> np.ndarray:
    """:func:`_slot_sums` of every slot, in passes of
    :data:`SLOTS_PER_PASS`.

    Parameters
    ----------
    samples : torch.Tensor of int16
        ``(channels, slots, cycles)`` raw samples (as a frame former holds
        them).
    channel : int
        Zero-based channel.
    """
    n = samples.shape[1]
    if samples.shape[2] < 3:
        return np.zeros((n, 7), dtype=np.int64)
    step = _pass_size(samples.device)
    parts = [
        _slot_sums(samples[channel, a : a + step]) for a in range(0, n, step)
    ]
    return np.concatenate(parts) if parts else np.zeros((0, 7), np.int64)


def validity_mismatch(samples: torch.Tensor) -> list:
    """Per channel, the samples whose validity differs from the first
    channel's.

    Frames are formed from each channel's own valid samples, but
    registration's freshness and ``varFacDS``, and so the photon-noise
    model, follow the first channel's; where another channel's invalid
    samples differ, its expectations assume the first channel's samples.

    Parameters
    ----------
    samples : torch.Tensor of int16
        ``(channels, slots, cycles)`` raw samples.

    Returns
    -------
    list of int
        One count per channel (0 for the first).
    """
    out = [0] * samples.shape[0]
    step = _pass_size(samples.device)
    for a in range(0, samples.shape[1], step):
        first = samples[0, a : a + step] != INVALID_SAMPLE
        for ch in range(1, samples.shape[0]):
            other = samples[ch, a : a + step] != INVALID_SAMPLE
            out[ch] += int((first != other).sum())
    return out


def next_slots(slot_line: np.ndarray, slot_pos: np.ndarray) -> np.ndarray:
    """The slot sampled right after each slot on its line (-1 for none)."""
    order = np.lexsort((slot_pos, slot_line))
    nxt = np.full(slot_line.size, -1, dtype=np.int64)
    a, b = order[:-1], order[1:]
    follows = (slot_line[a] == slot_line[b]) & (slot_pos[b] == slot_pos[a] + 1)
    nxt[a[follows]] = b[follows]
    return nxt


def _pair_sums(xa: torch.Tensor, xb: torch.Tensor) -> np.ndarray:
    """Exact sums over cycles of consecutive-sample pairs ``(slots,
    cycles)`` each.

    Returns
    -------
    ndarray of int64
        ``(pairs, 9)``: triples valid in both, ``sum(x1_a + x1_b)``,
        ``sum(d_a d_b)``, ``sum(d_a^2 d_b)``, ``sum(d_a d_b^2)``; cycles
        valid in both, and of them the zeros of ``a``, of ``b`` and of
        both.
    """
    xa, xb = xa.to(torch.int64), xb.to(torch.int64)
    va, vb = xa != INVALID_SAMPLE, xb != INVALID_SAMPLE
    a0, b0 = torch.where(va, xa, 0), torch.where(vb, xb, 0)

    def second(x0, v):
        """Second differences and where they are defined."""
        ok = v[:, :-2] & v[:, 1:-1] & v[:, 2:]
        return 2 * x0[:, 1:-1] - x0[:, :-2] - x0[:, 2:], ok

    da, oka = second(a0, va)
    db, okb = second(b0, vb)
    ok = oka & okb
    da, db = torch.where(ok, da, 0), torch.where(ok, db, 0)
    mid = torch.where(ok, a0[:, 1:-1] + b0[:, 1:-1], 0)
    both = va & vb
    za, zb = both & (xa == 0), both & (xb == 0)
    out = [ok.sum(1), mid.sum(1), (da * db).sum(1), (da * da * db).sum(1)]
    out += [(da * db * db).sum(1), both.sum(1), za.sum(1), zb.sum(1)]
    out.append((za & zb).sum(1))
    return torch.stack(out, 1).cpu().numpy()


def pair_sums(samples: torch.Tensor, channel: int, nxt: np.ndarray):
    """:func:`_pair_sums` of every slot with a next slot, in passes."""
    first = np.flatnonzero(nxt >= 0)
    if samples.shape[2] < 3 or first.size == 0:
        return np.zeros((0, 9), dtype=np.int64)
    dev = samples.device
    step = _pass_size(dev)
    parts = []
    for a in range(0, first.size, step):
        i = torch.from_numpy(first[a : a + step]).to(dev)
        j = torch.from_numpy(nxt[first[a : a + step]]).to(dev)
        parts.append(_pair_sums(samples[channel, i], samples[channel, j]))
    return np.concatenate(parts)


def pair_curve(sums: np.ndarray, n_bins: int = N_BINS) -> dict:
    """Pair terms of :class:`GainCurve` from :func:`pair_sums`.

    Pairs with at least 3 triples and a positive mean are sorted by their
    mean and split into ``n_bins`` groups of equal size.
    """
    n_trip, s_mid, s2, s3a, s3b, n_both, za, zb, zab = sums.T
    use = np.flatnonzero((n_trip >= 3) & (s_mid > 0))
    order = use[np.argsort(s_mid[use] / n_trip[use], kind="stable")]
    rows = []
    for grp in np.array_split(order, min(n_bins, max(order.size, 1))):
        if grp.size == 0:
            continue
        mid = 0.5 * float(s_mid[grp].sum())  # sum of the pairs' means
        den = 6.0 * mid
        nb = float(n_both[grp].sum())
        pa, pb, pab = (
            za[grp].sum() / nb,
            zb[grp].sum() / nb,
            zab[grp].sum() / nb,
        )
        nu = (
            -0.5 * (np.log(pa) + np.log(pb))
            if 0 < pa < 1 and 0 < pb < 1
            else np.nan
        )
        shared = np.log(pab / (pa * pb)) / nu if pab > 0 and nu > 0 else np.nan
        rows.append(
            (mid / n_trip[grp].sum(), s2[grp].sum() / den,
             s3a[grp].sum() / den, s3b[grp].sum() / den, shared, grp.size)
        )  # fmt: skip
    arr = np.array(rows, dtype=np.float64).reshape(-1, 6)
    arr = arr[np.isfinite(arr[:, :4]).all(1)]
    return {
        "mu_pair": arr[:, 0],
        "c2": arr[:, 1],
        "c3a": arr[:, 2],
        "c3b": arr[:, 3],
        "zero_shared": arr[:, 4],
        "n_pairs": arr[:, 5].astype(np.int64),
    }


def curve_from_sums(sums: np.ndarray, n_bins: int = N_BINS) -> GainCurve:
    """The gain curve of per-slot sums (:func:`slot_sums`).

    Slots with at least 3 triples and a positive mean are sorted by mean
    (ties by slot) and split into ``n_bins`` groups of equal size (sizes
    differ by at most one); groups without photons are dropped. A group
    without zeros keeps its gains; its non-zero rate is NaN.
    """
    n_valid, total, zeros, n_trip, s1, s2, s3 = sums.T
    use = np.flatnonzero((n_trip >= 3) & (total > 0))
    mean = total[use] / n_valid[use]
    order = use[np.argsort(mean, kind="stable")]
    rows = []
    for grp in np.array_split(order, min(n_bins, max(order.size, 1))):
        if grp.size == 0 or s1[grp].sum() <= 0:
            continue
        mu = total[grp].sum() / n_valid[grp].sum()
        zero = zeros[grp].sum() / n_valid[grp].sum()
        nu = -np.log(zero) if 0 < zero < 1 else np.nan
        den = 6.0 * float(s1[grp].sum())
        rows.append(
            (mu, s2[grp].sum() / den, s3[grp].sum() / den, nu / mu, grp.size)
        )
    arr = np.array(rows, dtype=np.float64).reshape(-1, 5)
    keep = np.isfinite(arr[:, :3]).all(1)
    arr = arr[keep]
    return GainCurve(
        mu=arr[:, 0],
        g2=arr[:, 1],
        g3=arr[:, 2],
        nu_per_mu=arr[:, 3],
        n_slots=arr[:, 4].astype(np.int64),
    )


def measure_gain_curves(
    samples: torch.Tensor, nxt=None, n_bins: int = N_BINS
) -> list:
    """Every channel's :class:`GainCurve` from a trial's raw samples.

    Parameters
    ----------
    samples : torch.Tensor of int16
        ``(channels, slots, cycles)``, e.g. ``FrameFormer._samples``.
    nxt : ndarray of int, optional
        :func:`next_slots` of the layout (without, no pair terms).
    n_bins : int
        Brightness groups.

    Returns
    -------
    list of GainCurve
        One per channel.
    """
    out = []
    for ch in range(samples.shape[0]):
        curve = curve_from_sums(slot_sums(samples, ch), n_bins)
        if nxt is not None:
            for k, v in pair_curve(
                pair_sums(samples, ch, nxt), n_bins
            ).items():
                setattr(curve, k, v)
        out.append(curve)
    return out
