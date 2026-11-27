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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ...io.slap2_raster import INVALID_SAMPLE

N_BINS = 16
"""Brightness groups of the gain curve (equal numbers of slots)."""

SLOTS_PER_PASS = 8192
"""Slots summed at a time (bounds memory)."""


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
        ``(K,)`` rate of non-zero samples per unit mean.
    n_slots : ndarray of int64
        ``(K,)`` slots in each group.
    """

    mu: np.ndarray
    g2: np.ndarray
    g3: np.ndarray
    nu_per_mu: np.ndarray
    n_slots: np.ndarray

    def at(self, mean) -> tuple:
        """Gains at per-sample means ``mean`` (any shape).

        Linear in ``log(mean)`` between the groups' means and constant
        beyond the first and last group.

        Returns
        -------
        g2, g3, nu : ndarray of float64
            Same shape as ``mean``; ``nu`` is the non-zero rate per sample.
        """
        mean = np.asarray(mean, dtype=np.float64)
        if self.mu.size == 0:
            nan = np.full(mean.shape, np.nan)
            return nan, nan.copy(), nan.copy()
        x = np.log(np.maximum(mean, 1e-12))
        xp = np.log(self.mu)
        g2 = np.interp(x, xp, self.g2)
        g3 = np.interp(x, xp, self.g3)
        nu = mean * np.interp(x, xp, self.nu_per_mu)
        bad = ~np.isfinite(mean)
        for a in (g2, g3, nu):
            a[bad] = np.nan
        return g2, g3, nu

    def describe(self) -> dict:
        """The curve as arrays (for the statistics file)."""
        return {
            "mu": self.mu,
            "g2": self.g2,
            "g3": self.g3,
            "nu_per_mu": self.nu_per_mu,
            "n_slots": self.n_slots,
        }


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
    parts = [
        _slot_sums(samples[channel, a : a + SLOTS_PER_PASS])
        for a in range(0, n, SLOTS_PER_PASS)
    ]
    return np.concatenate(parts) if parts else np.zeros((0, 7), np.int64)


def curve_from_sums(sums: np.ndarray, n_bins: int = N_BINS) -> GainCurve:
    """The gain curve of per-slot sums (:func:`slot_sums`).

    Slots with at least 3 triples and a positive mean are sorted by mean
    (ties by slot) and split into ``n_bins`` groups of equal size (sizes
    differ by at most one); groups without photons are dropped.
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
    keep = np.isfinite(arr).all(1)
    arr = arr[keep]
    return GainCurve(
        mu=arr[:, 0],
        g2=arr[:, 1],
        g3=arr[:, 2],
        nu_per_mu=arr[:, 3],
        n_slots=arr[:, 4].astype(np.int64),
    )


def measure_gain_curves(samples: torch.Tensor, n_bins: int = N_BINS) -> list:
    """Every channel's :class:`GainCurve` from a trial's raw samples.

    Parameters
    ----------
    samples : torch.Tensor of int16
        ``(channels, slots, cycles)``, e.g. ``FrameFormer._samples``.
    n_bins : int
        Brightness groups.

    Returns
    -------
    list of GainCurve
        One per channel.
    """
    return [
        curve_from_sums(slot_sums(samples, ch), n_bins)
        for ch in range(samples.shape[0])
    ]
