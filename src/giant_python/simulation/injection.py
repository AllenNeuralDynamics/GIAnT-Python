"""Synthetic events injected into the raw samples of a SLAP2 recording.

A copy of a session's raw ``.dat`` files is made in which the raster samples
of one channel hold extra photons from synthetic sources whose positions,
footprints, event times, sizes and time course are known; a truth file
records them. Everything else in the copy (file and line headers, line
timing, the other channel, integration superpixels, invalid samples) is the
original's, byte for byte.

**Where the sources are.** Sources are fixed in the tissue as the session's
own registration sees it. Each path has a *path frame*: the registered frame
of its reference trial (by default the first injected trial), with zero-based
``(row, col)`` coordinates of the registered movie (as
:func:`~giant_python.extraction.raster.trial.load_registered_trial` returns
it, MATLAB orientation; pixel centres at integers). Trial ``t``'s registered
frame is the path frame shifted by ``trial_shift[t]`` (a position ``p`` in
the path frame is ``p + trial_shift[t]`` in the trial's frame), estimated by
aligning the trials' mean images (:func:`align_images`) unless given, for
example as SILo's across-trial alignment.

A sample of DMD pixel ``(r, c)`` (zero-based) on line ``l`` lies at
``(r, c) + shift(l)`` in the trial's registered frame (:func:`trial_geometry`)
with, as SILo's ``loadTrial`` combines them::

    shift(l) = o(l) - (trim - 1) + maxshift - M(l)

where ``trim`` is the first DMD row and column of the registration window
(one-based ``trimRows(1)``, ``trimCols(1)``), ``M(l)`` the registration's
motion interpolated to the line (``pchip``, extrapolated), and, when the
registration undid the online motion correction (``compensateOnline``),
``o(l) = mapping @ ([x(l), y_eff(l)] - refOffset)`` with the line's online
offsets and ``M`` the continuous motion ``motionDS + mapping @ (frameRef -
refOffset)``; without compensation ``o = 0`` and ``M = motionDS`` (the
registration's motion then includes the online offsets). A source therefore
stays still in the registered movie of the original data; in the tissue it
moves with the registration's error.

**What is added.** The expected extra raw counts of a sample of the injected
channel are ``photon_scale * sum_j f_j(p - p_j) * a_j(t)`` over sources
``j``: ``f_j`` is the footprint (peak 1), ``p`` the sample's registered
position, ``p_j`` the source's, and ``a_j`` the source's events convolved
with the indicator's time course at the line's time ``(l - 1) *
line_period``. Event sizes are in the units of GIAnT's ground-truth refit
test: frame counts / ``photon_scale`` at the source's peak pixel at the
event's peak, in SILo's 200 Hz frames. Frames formed from raw samples keep
only part of an instantaneous peak (their kernel averages over several
milliseconds and the pixel's few samples per cycle, resampling blurs it), so
each source's tissue amplitude is its event size divided by the share its
frames keep, measured on noiseless frames of unit events
(:func:`calibrate_frame_peaks`). The
expectation is divided by the raw units per photon ``g``; a Poisson draw
``N`` of it adds ``floor(N * g + U)`` raw counts (``U`` uniform on [0, 1), so
the mean is exactly ``N * g``), saturating at 32767. Invalid samples
(-32768) are left as they are.

``g`` is measured on the original recording (:func:`measure_raw_per_photon`)
as the variance-to-mean ratio of its raw samples, so injected photons carry
the shot noise of real ones; it is larger than the mean single-photon pulse
height, which is also recorded.

Run as ``python -m giant_python.simulation.injection inject TRIAL_TABLE
OUT_DIR`` (or call :func:`inject_session`); after registering the copy
(``giant register OUT_DIR/trial_table.h5``), ``... compare OUT_DIR`` (or
:func:`record_copy_registration`) records where the copy's registered frames
put the sources.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import shutil
import sys
import time
import warnings
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
from scipy import ndimage
from scipy.interpolate import PchipInterpolator
from scipy.spatial import cKDTree

from ..io.hdf5 import load_struct_h5, save_struct_h5
from ..io.slap2_raster import (
    INVALID_SAMPLE,
    RasterLayout,
    raster_layout,
    read_raster_samples,
)
from ..registration.compensation import line_offsets

log = logging.getLogger(__name__)

PHOTON_SCALE = 25.0
"""Raw counts of one size unit (GIAnT's ``photonScale`` for SLAP2)."""

SIZE_CLASSES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
"""Event sizes of GIAnT's ground-truth refit test."""

SAMPLE_MAX = 32767
"""Largest raw sample (int16)."""

TRUTH_FORMAT = "giant_python.simulation.injection/1"
"""``format`` attribute of truth files."""

TRUTH_NAME = "injection_truth.h5"
"""Name of the truth file in the output folder."""

_STREAM_PLACE, _STREAM_EVENTS, _STREAM_PHOTONS, _STREAM_CALIBRATE = 1, 2, 3, 4


def _rng(seed: int, *key: int) -> np.random.Generator:
    """An independent generator for one purpose (``key``) of a seed."""
    ss = np.random.SeedSequence(
        int(seed), spawn_key=tuple(int(k) for k in key)
    )
    return np.random.default_rng(ss)


# ---------------------------------------------------------------- settings


@dataclass(frozen=True)
class InjectionSettings:
    """What is injected into every source and trial of a session.

    Attributes
    ----------
    size_classes : tuple of float
        Event sizes in units of ``photon_scale`` raw counts (see
        ``size_reference``).
    class_weights : tuple of float, optional
        Relative frequency of each size class (default: equal).
    rate_hz : float
        Rate of each source's events outside their dead time: the interval
        between events is ``min_interval_s`` plus an exponential with mean
        ``1 / rate_hz`` (so the mean rate is ``1 / (min_interval_s + 1 /
        rate_hz)``). 0 injects nothing.
    min_interval_s : float
        Dead time after each event of a source.
    edge_s : float
        Event onsets keep at least this distance from the trial's first and
        last line.
    tau_s : float
        Decay time constant of the indicator (SILo's ``tau_s``).
    rise_s : float
        Rise time constant; 0 for an instant rise. Otherwise the time course
        is ``exp(-t / tau_s) - exp(-t / rise_s)`` scaled to a peak of 1.
    sigma_px : float
        Width of the default Gaussian footprint (SILo's ``sigma_px``), in
        registered pixels.
    radius_sigmas : float
        The Gaussian footprint is zero beyond this many sigmas.
    photon_scale : float
        Raw counts of one size unit at the peak.
    channel : int
        Zero-based channel that photons are added to.
    raw_per_photon : float, optional
        Raw units per photon; by default measured on each path's reference
        trial (:func:`measure_raw_per_photon`).
    analyze_hz : float
        Rate of the frames (SILo's ``analyzeHz``) at which sizes are
        calibrated and the truth gives each source's traces.
    seed : int
        Seeds placement, events, calibration and photon draws.
    size_reference : str
        ``'frames'`` (default): a size is the event's expected peak in
        SILo's ``analyze_hz`` frames at the source's peak pixel, in frame
        counts / ``photon_scale`` (the units of GIAnT's ground-truth refit
        test); each source's injected amplitude is scaled by what its frames
        keep of an event's peak (:func:`calibrate_frame_peaks`).
        ``'tissue'``: a size is the instantaneous peak at the footprint's
        centre, before frames are formed.
    calibration_s : float
        Length of the trial stretch used to calibrate frame peaks.
    frame_traces : bool
        Also give each source's noiseless trace in ``analyze_hz`` frames at
        its peak pixel (formed as SILo forms frames; a render of every
        trial).
    device : str
        Where frames are formed for calibration and frame traces
        (``'auto'``: CUDA when available).
    """

    size_classes: Tuple[float, ...] = SIZE_CLASSES
    class_weights: Optional[Tuple[float, ...]] = None
    rate_hz: float = 1.0
    min_interval_s: float = 0.1
    edge_s: float = 0.25
    tau_s: float = 0.024
    rise_s: float = 0.0
    sigma_px: float = 1.33
    radius_sigmas: float = 4.0
    photon_scale: float = PHOTON_SCALE
    channel: int = 0
    raw_per_photon: Optional[float] = None
    analyze_hz: float = 200.0
    seed: int = 0
    size_reference: str = "frames"
    calibration_s: float = 4.0
    frame_traces: bool = True
    device: str = "auto"

    def __post_init__(self):
        """Reject settings that cannot describe events."""
        w = self.class_weights
        rules = [
            (
                not self.size_classes or min(self.size_classes) < 0,
                "size_classes must be non-negative sizes",
            ),
            (
                w is not None
                and (
                    len(w) != len(self.size_classes)
                    or min(w) < 0
                    or sum(w) <= 0
                ),
                "class_weights must match size_classes",
            ),
            (
                min(self.rate_hz, self.min_interval_s, self.edge_s) < 0,
                "rate_hz, min_interval_s, edge_s must be >= 0",
            ),
            (
                self.tau_s <= 0 or not 0 <= self.rise_s < self.tau_s,
                "need tau_s > 0 and 0 <= rise_s < tau_s",
            ),
            (
                min(
                    self.sigma_px,
                    self.radius_sigmas,
                    self.photon_scale,
                    self.analyze_hz,
                    self.calibration_s,
                )
                <= 0,
                "sigma_px, radius_sigmas, photon_scale, analyze_hz and "
                "calibration_s must be positive",
            ),
            (
                self.raw_per_photon is not None
                and not (
                    math.isfinite(self.raw_per_photon)
                    and self.raw_per_photon > 0
                ),
                "raw_per_photon must be positive and finite",
            ),
            (self.channel < 0, "channel must be a zero-based channel index"),
            (
                self.size_reference not in ("frames", "tissue"),
                "size_reference must be 'frames' or 'tissue'",
            ),
        ]
        for bad, message in rules:
            if bad:
                raise ValueError(message)

    @property
    def weights(self) -> np.ndarray:
        """Probabilities of the size classes."""
        w = np.ones(len(self.size_classes))
        if self.class_weights is not None:
            w = np.asarray(self.class_weights, dtype=np.float64)
        return w / w.sum()


@dataclass(frozen=True)
class PlacementSettings:
    """Where :func:`place_sources` may put sources.

    Attributes
    ----------
    min_separation_px : float
        Least distance between injected sources.
    edge_px : float
        Least distance from pixels without data in any trial (the field's
        edges and holes).
    exclude_px : float
        Least distance from existing sources (for example SILo's sources of
        a sensor session).
    label_factor : float
        Labeled pixels: the 3 x 3 median of the mean image above this
        factor times its 25th percentile (SILo's ``loadTrial`` uses 3).
    """

    min_separation_px: float = 6.0
    edge_px: float = 5.0
    exclude_px: float = 6.0
    label_factor: float = 3.0


# ----------------------------------------------------- time course, events


def peak_delay_s(tau_s: float, rise_s: float = 0.0) -> float:
    """Time from onset to the peak of the time course."""
    if rise_s <= 0:
        return 0.0
    return tau_s * rise_s / (tau_s - rise_s) * math.log(tau_s / rise_s)


def time_course(t_s, tau_s: float, rise_s: float = 0.0) -> np.ndarray:
    """The indicator's response to an event at time 0, with a peak of 1.

    Parameters
    ----------
    t_s : array_like
        Times after onset (negative: before the event).
    tau_s, rise_s : float
        Decay and rise time constants (``rise_s`` 0: instant rise).

    Returns
    -------
    ndarray
        0 before onset; ``exp(-t / tau_s)``, or the difference of
        exponentials scaled to a peak of 1.
    """
    t = np.asarray(t_s, dtype=np.float64)
    out = np.zeros_like(t)
    on = t >= 0
    if rise_s <= 0:
        out[on] = np.exp(-t[on] / tau_s)
        return out
    tp = peak_delay_s(tau_s, rise_s)
    norm = math.exp(-tp / tau_s) - math.exp(-tp / rise_s)
    out[on] = (np.exp(-t[on] / tau_s) - np.exp(-t[on] / rise_s)) / norm
    return out


def kernel_extent_s(settings: InjectionSettings) -> float:
    """Time after onset beyond which the time course is below 1e-6."""
    return peak_delay_s(settings.tau_s, settings.rise_s) + settings.tau_s * (
        6 * math.log(10)
    )


@dataclass
class Events:
    """Events of a set of sources.

    Attributes
    ----------
    source : ndarray of int
        Zero-based source index.
    trial : ndarray of int
        One-based trial (column of the original trial table).
    onset_line : ndarray of float
        Onset as a one-based line coordinate (line ``l`` is acquired at
        ``(l - 1) * line_period_s`` after the recording's first line).
    size_class : ndarray of int
        Index into ``size_classes``.
    size : ndarray of float
        Size (see :class:`InjectionSettings`).
    """

    source: np.ndarray
    trial: np.ndarray
    onset_line: np.ndarray
    size_class: np.ndarray
    size: np.ndarray

    @classmethod
    def empty(cls) -> "Events":
        """No events."""
        z = np.zeros(0)
        return cls(
            z.astype(np.int64), z.astype(np.int64), z, z.astype(np.int64), z
        )

    @classmethod
    def concat(cls, parts: Sequence["Events"]) -> "Events":
        """Events of several parts, in order."""
        if not parts:
            return cls.empty()
        return cls(
            *[
                np.concatenate([getattr(p, k) for p in parts])
                for k in (
                    "source",
                    "trial",
                    "onset_line",
                    "size_class",
                    "size",
                )
            ]
        )

    def __len__(self) -> int:
        """Number of events."""
        return int(self.source.size)

    def select(self, mask) -> "Events":
        """The events where ``mask`` is true."""
        return Events(
            self.source[mask],
            self.trial[mask],
            self.onset_line[mask],
            self.size_class[mask],
            self.size[mask],
        )


def draw_events(
    n_sources: int,
    first_line: float,
    last_line: float,
    line_period_s: float,
    settings: InjectionSettings,
    rng: np.random.Generator,
    trial: int = 1,
) -> Events:
    """Random events of each source in one trial.

    Each source's onsets form a renewal process: the first falls an
    exponential interval (mean ``1 / rate_hz``) after the trial's first line
    plus ``edge_s``, each next one ``min_interval_s`` plus an exponential
    interval after the previous; onsets later than the last line minus
    ``edge_s`` are dropped. Sizes are drawn independently from the size
    classes.

    Parameters
    ----------
    n_sources : int
        Number of sources.
    first_line, last_line : float
        The trial's line range (one-based, may be fractional).
    line_period_s : float
        Line period.
    settings : InjectionSettings
        Rates, dead time, edges and sizes.
    rng : numpy.random.Generator
        Source of randomness.
    trial : int
        Trial number stored with the events.

    Returns
    -------
    Events
        Sorted by source, then onset.
    """
    edge = settings.edge_s / line_period_s
    lo, hi = float(first_line) + edge, float(last_line) - edge
    if settings.rate_hz <= 0 or n_sources == 0 or hi <= lo:
        return Events.empty()
    mean_gap = 1.0 / settings.rate_hz / line_period_s
    dead = settings.min_interval_s / line_period_s
    span = hi - lo
    n_draw = int(span / (mean_gap + dead) * 1.5) + 20
    parts = []
    for j in range(n_sources):
        while True:
            gaps = rng.exponential(mean_gap, n_draw) + dead
            gaps[0] -= dead
            t = lo + np.cumsum(gaps)
            if t[-1] > hi:
                break
            n_draw *= 2
        t = t[t <= hi]
        cls = rng.choice(
            len(settings.size_classes), t.size, p=settings.weights
        )
        parts.append(
            Events(
                np.full(t.size, j, dtype=np.int64),
                np.full(t.size, int(trial), dtype=np.int64),
                t,
                cls.astype(np.int64),
                np.asarray(settings.size_classes, dtype=np.float64)[cls],
            )
        )
    return Events.concat(parts)


def event_traces(
    events: Events,
    n_sources: int,
    lines: Tuple[int, int],
    line_period_s: float,
    settings: InjectionSettings,
) -> np.ndarray:
    """Each source's noiseless amplitude at every line of a range.

    Parameters
    ----------
    events : Events
        Events of one trial.
    n_sources : int
        Number of sources.
    lines : (int, int)
        First and last one-based line.
    line_period_s : float
        Line period.
    settings : InjectionSettings
        Time constants.

    Returns
    -------
    ndarray of float64
        ``(sources, lines)``: sum of size times time course of each
        source's events, in size units.
    """
    lo, hi = int(lines[0]), int(lines[1])
    out = np.zeros((n_sources, hi - lo + 1))
    extent = kernel_extent_s(settings) / line_period_s
    for j, t0, a in zip(events.source, events.onset_line, events.size):
        l0 = max(lo, int(math.ceil(t0)))
        l1 = min(hi, int(math.floor(t0 + extent)))
        if l1 < l0:
            continue
        u = (np.arange(l0, l1 + 1) - t0) * line_period_s
        out[j, l0 - lo : l1 - lo + 1] += a * time_course(
            u, settings.tau_s, settings.rise_s
        )
    return out


# ---------------------------------------------------------------- footprints


class GaussianFootprint:
    """A Gaussian footprint with a peak of 1 at the source position.

    Footprints are callables of row and column offsets from the source with
    a ``radius`` (zero beyond it) and a ``peak_offset`` (where their maximum
    is, relative to the source).

    Parameters
    ----------
    sigma_px : float
        Standard deviation along rows (and columns, by default).
    radius_sigmas : float
        Zero beyond this many (largest) sigmas.
    sigma_c_px : float, optional
        Standard deviation along columns.
    """

    kind = "gaussian"

    def __init__(
        self,
        sigma_px: float = 1.33,
        radius_sigmas: float = 4.0,
        sigma_c_px: Optional[float] = None,
    ):
        """Store the widths and the support radius."""
        sc = sigma_px if sigma_c_px is None else sigma_c_px
        self.sigma = (float(sigma_px), float(sc))
        self.radius = float(radius_sigmas) * max(self.sigma)
        self.peak_offset = (0.0, 0.0)

    def __call__(self, dr, dc) -> np.ndarray:
        """Value at row and column offsets from the source."""
        dr = np.asarray(dr, dtype=np.float64)
        dc = np.asarray(dc, dtype=np.float64)
        q = (dr / self.sigma[0]) ** 2 + (dc / self.sigma[1]) ** 2
        return np.where(dr**2 + dc**2 > self.radius**2, 0.0, np.exp(-0.5 * q))

    def describe(self) -> dict:
        """Parameters, as stored in the truth file."""
        return {
            "kind": self.kind,
            "sigma_rc": list(self.sigma),
            "radius_px": self.radius,
        }


class ImageFootprint:
    """A footprint given as an image, interpolated bilinearly.

    Parameters
    ----------
    image : array_like
        Non-negative weights on a pixel grid; scaled to a maximum of 1.
    center : (float, float), optional
        Zero-based ``(row, col)`` of the image that sits on the source
        position (default: the image's centre).
    """

    kind = "image"

    def __init__(self, image, center: Optional[Sequence[float]] = None):
        """Normalize the image and find its support radius."""
        img = np.asarray(image, dtype=np.float64)
        if img.ndim != 2 or np.nanmin(img) < 0 or np.nanmax(img) <= 0:
            raise ValueError("a footprint image must be 2-D, >= 0, not all 0")
        self.image = np.nan_to_num(img) / np.nanmax(img)
        h, w = img.shape
        if center is None:
            center = ((h - 1) / 2.0, (w - 1) / 2.0)
        self.center = (float(center[0]), float(center[1]))
        corners = np.array([[0, 0], [0, w - 1], [h - 1, 0], [h - 1, w - 1]])
        self.radius = float(
            np.max(np.hypot(*(corners - np.array(self.center)).T))
        )
        top = np.unravel_index(np.argmax(self.image), self.image.shape)
        self.peak_offset = (
            float(top[0] - self.center[0]),
            float(top[1] - self.center[1]),
        )

    def __call__(self, dr, dc) -> np.ndarray:
        """Value at row and column offsets from the source (0 outside)."""
        dr = np.asarray(dr, dtype=np.float64)
        dc = np.asarray(dc, dtype=np.float64)
        coords = [self.center[0] + dr.ravel(), self.center[1] + dc.ravel()]
        out = ndimage.map_coordinates(
            self.image, coords, order=1, mode="constant", cval=0.0
        )
        return out.reshape(dr.shape)

    def describe(self) -> dict:
        """Parameters, as stored in the truth file."""
        return {
            "kind": self.kind,
            "center_rc": list(self.center),
            "radius_px": self.radius,
        }


def peak_pixels(positions: np.ndarray, footprints: Sequence) -> np.ndarray:
    """Each source's peak pixel: nearest to its footprint's maximum.

    Returns
    -------
    ndarray of int
        ``(sources, 2)`` zero-based ``(row, col)``.
    """
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    off = np.array(
        [getattr(fp, "peak_offset", (0.0, 0.0)) for fp in footprints],
        dtype=np.float64,
    ).reshape(-1, 2)
    return np.rint(pos + off).astype(np.int64)


def footprint_patch(footprint, position, half: int) -> Tuple[np.ndarray, list]:
    """A footprint on the integer pixels around a position.

    Returns
    -------
    patch : ndarray
        ``(2 half + 1, 2 half + 1)`` values at pixels ``origin + (i, j)``.
    origin : list of int
        Zero-based ``(row, col)`` of the patch's first pixel.
    """
    r0 = int(round(position[0])) - half
    c0 = int(round(position[1])) - half
    rr, cc = np.mgrid[r0 : r0 + 2 * half + 1, c0 : c0 + 2 * half + 1]
    return footprint(rr - position[0], cc - position[1]), [r0, c0]


# ---------------------------------------------------- raw units per photon


@dataclass
class PhotonGain:
    """Raw units per photon of a recording's channel.

    Attributes
    ----------
    raw_per_photon : float
        Variance-to-mean ratio of the raw samples (successive-cycle
        differences of each slot, pooled over slots between the 25th and
        75th percentile of brightness): the raw units per photon that give
        Poisson photons the recording's shot noise. Used for injection.
    mean_pulse : float
        Mean single-photon pulse height: the summed samples of the dimmest
        quarter of slots over their photons, counted from each slot's share
        of empty samples (Poisson).
    n_slots : int
        Slots pooled for ``raw_per_photon``.
    cycles : (int, int)
        Zero-based cycle range used.
    channel : int
        Zero-based channel.
    """

    raw_per_photon: float
    mean_pulse: float
    n_slots: int
    cycles: Tuple[int, int]
    channel: int

    def describe(self) -> dict:
        """Fields, as stored in the truth file."""
        out = asdict(self)
        out["cycles"] = list(self.cycles)
        out["method"] = (
            "variance/mean of successive-cycle differences per slot, pooled "
            "over slots between the 25th and 75th brightness percentile"
        )
        return out


def measure_raw_per_photon(
    layout: RasterLayout,
    channel: int = 0,
    cycles: Optional[Tuple[int, int]] = None,
    max_cycles: int = 1000,
    slots_per_read: int = 8192,
) -> PhotonGain:
    """Measure raw units per photon from a recording's raw samples.

    Parameters
    ----------
    layout : RasterLayout
        The recording.
    channel : int
        Zero-based channel.
    cycles : (int, int), optional
        Zero-based cycle range (default: the first ``max_cycles``).
    max_cycles : int
        Cycles used by default.
    slots_per_read : int
        Slots read at a time (bounds memory).

    Returns
    -------
    PhotonGain
    """
    if cycles is None:
        cycles = (0, min(layout.num_cycles, max_cycles))
    c0, c1 = int(cycles[0]), int(cycles[1])
    if c1 - c0 < 3:
        raise ValueError("need at least 3 cycles to measure the photon gain")
    n_slots = layout.slot_line.size
    s_sum = np.zeros(n_slots)
    d2_sum = np.zeros(n_slots)
    n_pair = np.zeros(n_slots)
    for a in range(0, n_slots, slots_per_read):
        sl = np.arange(a, min(n_slots, a + slots_per_read))
        x = read_raster_samples(layout, (c0, c1), sl)[channel]
        x = x.astype(np.float64)
        ok = (x[:, 1:] != INVALID_SAMPLE) & (x[:, :-1] != INVALID_SAMPLE)
        d = np.where(ok, x[:, 1:] - x[:, :-1], 0.0)
        m = np.where(ok, x[:, 1:] + x[:, :-1], 0.0)
        d2_sum[sl] = 0.5 * (d**2).sum(1)
        s_sum[sl] = 0.5 * m.sum(1)
        n_pair[sl] = ok.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s_sum / n_pair
    use = (n_pair >= 10) & (mean > 0)
    if not use.any():
        raise ValueError("no slot has photons; cannot measure the gain")
    lo, hi = np.percentile(mean[use], [25, 75])
    mid = use & (mean >= lo) & (mean <= hi)
    gain = float(d2_sum[mid].sum() / s_sum[mid].sum())
    dim = np.flatnonzero(use & (mean <= lo))
    total = photons = 0.0
    for a in range(0, dim.size, slots_per_read):
        x = read_raster_samples(layout, (c0, c1), dim[a : a + slots_per_read])
        x = x[channel]
        ok = x != INVALID_SAMPLE
        n = ok.sum(1)
        zero = ((x == 0) & ok).sum(1)
        has = (zero > 0) & (zero < n)
        # photons per sample from the share of empty samples (Poisson)
        rate = -np.log(zero[has] / n[has])
        total += float(np.where(ok, x, 0)[has].sum())
        photons += float((rate * n[has]).sum())
    return PhotonGain(
        raw_per_photon=gain,
        mean_pulse=total / photons if photons > 0 else float("nan"),
        n_slots=int(mid.sum()),
        cycles=(c0, c1),
        channel=int(channel),
    )


# ------------------------------------------------------ placement, alignment


def labeled_pixels(mean_im: np.ndarray, factor: float = 3.0) -> np.ndarray:
    """Labeled (dendrite) pixels of a mean image, as SILo's ``loadTrial``.

    The 3 x 3 median of the image (pixels without data read as 0) above
    ``factor`` times its 25th percentile over pixels with data.
    """
    m = np.asarray(mean_im, dtype=np.float64)
    data = np.isfinite(m)
    if not data.any():
        return np.zeros(m.shape, dtype=bool)
    med = ndimage.median_filter(np.where(data, m, 0.0), 3, mode="constant")
    return data & (med > factor * np.percentile(med[data], 25))


def place_sources(
    mean_im: np.ndarray,
    n_sources: int,
    rng: np.random.Generator,
    settings: PlacementSettings = PlacementSettings(),
    valid: Optional[np.ndarray] = None,
    exclude_rc: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
    """Random source positions on labeled pixels, apart from each other.

    Parameters
    ----------
    mean_im : ndarray
        ``(rows, cols)`` mean image of the injected channel in the path
        frame (NaN without data).
    n_sources : int
        Sources wanted; fewer are returned if the candidates run out.
    rng : numpy.random.Generator
        Picks candidates in random order.
    settings : PlacementSettings
        Distances and labeling.
    valid : ndarray of bool, optional
        Pixels with data in every trial (default: finite ``mean_im``).
    exclude_rc : ndarray, optional
        ``(n, 2)`` existing source positions to keep away from.

    Returns
    -------
    positions : ndarray of float
        ``(k, 2)`` zero-based ``(row, col)``, integers, ``k <= n_sources``.
    n_candidates : int
        Pixels that met the labeling, edge and exclusion rules.
    """
    m = np.asarray(mean_im, dtype=np.float64)
    if valid is None:
        valid = np.isfinite(m)
    padded = np.pad(np.asarray(valid, dtype=bool), 1)
    edge_dist = ndimage.distance_transform_edt(padded)[1:-1, 1:-1]
    cand = labeled_pixels(m, settings.label_factor) & valid
    cand &= edge_dist >= settings.edge_px
    rc = np.argwhere(cand).astype(np.float64)
    if exclude_rc is not None and len(exclude_rc) and rc.size:
        d, _ = cKDTree(np.asarray(exclude_rc, np.float64)).query(rc)
        rc = rc[d >= settings.exclude_px]
    n_cand = int(rc.shape[0])
    chosen: List[np.ndarray] = []
    tree_pts = np.zeros((0, 2))
    for i in rng.permutation(n_cand):
        if len(chosen) >= n_sources:
            break
        p = rc[i]
        if (
            tree_pts.size
            and np.min(np.hypot(*(tree_pts - p).T))
            < settings.min_separation_px
        ):
            continue
        chosen.append(p)
        tree_pts = np.vstack([tree_pts, p])
    return np.array(chosen).reshape(-1, 2), n_cand


def _highpass(x: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """An image minus its Gaussian background (0 without data), and its
    data mask."""
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    fill = np.where(ok, x, np.nanmean(x) if ok.any() else 0.0)
    return np.where(ok, fill - ndimage.gaussian_filter(fill, sigma), 0.0), ok


def _ncc(a, va, b, vb, dr: int, dc: int) -> float:
    """Normalized correlation of ``a(p)`` and ``b(p + (dr, dc))`` where both
    have data (-inf if they share fewer than 100 pixels)."""
    h, w = a.shape
    ra = slice(max(0, -dr), min(h, h - dr))
    ca = slice(max(0, -dc), min(w, w - dc))
    rb = slice(max(0, dr), min(h, h + dr))
    cb = slice(max(0, dc), min(w, w + dc))
    ok = va[ra, ca] & vb[rb, cb]
    if ok.sum() < 100:
        return -np.inf
    x, y = a[ra, ca][ok], b[rb, cb][ok]
    x, y = x - x.mean(), y - y.mean()
    den = math.sqrt((x @ x) * (y @ y))
    return float((x @ y) / den) if den > 0 else -np.inf


def _vertex(c_m: float, c_0: float, c_p: float) -> float:
    """Offset of a parabola's vertex through three equally spaced values."""
    den = c_m - 2 * c_0 + c_p
    if not np.isfinite(den) or den >= 0:
        return 0.0
    return float(np.clip(0.5 * (c_m - c_p) / den, -0.5, 0.5))


def fit_shape(image: np.ndarray, shape) -> np.ndarray:
    """An image cropped or NaN-padded at its bottom and right to ``shape``
    (registered grids share their top-left corner, as in SILo)."""
    out = np.full(shape, np.nan)
    h, w = min(shape[0], image.shape[0]), min(shape[1], image.shape[1])
    out[:h, :w] = image[:h, :w]
    return out


def align_images(
    reference: np.ndarray,
    image: np.ndarray,
    max_shift: int = 8,
    highpass_px: float = 4.0,
) -> np.ndarray:
    """Rigid shift between two images (masked normalized cross-correlation).

    Parameters
    ----------
    reference, image : ndarray
        ``(rows, cols)``, NaN without data; ``image`` is cropped or padded to
        the reference's shape.
    max_shift : int
        Largest integer shift searched.
    highpass_px : float
        Gaussian width of the background removed before matching.

    Returns
    -------
    ndarray of float
        ``s = (dr, dc)`` such that ``image(p + s)`` matches
        ``reference(p)``, refined to subpixel by parabolas through the
        correlation peak.
    """
    a, va = _highpass(reference, highpass_px)
    b, vb = _highpass(fit_shape(image, a.shape), highpass_px)
    span = range(-max_shift, max_shift + 1)
    cc = np.array([[_ncc(a, va, b, vb, dr, dc) for dc in span] for dr in span])
    i, j = np.unravel_index(np.argmax(cc), cc.shape)
    if not np.isfinite(cc[i, j]):
        raise ValueError("the images do not overlap")
    n = cc.shape[0]
    di = _vertex(*cc[i - 1 : i + 2, j]) if 0 < i < n - 1 else 0.0
    dj = _vertex(*cc[i, j - 1 : j + 2]) if 0 < j < n - 1 else 0.0
    return np.array([i - max_shift + di, j - max_shift + dj])


def shift_image(image: np.ndarray, shift, shape=None) -> np.ndarray:
    """``out(p) = image(p + shift)`` (bilinear; NaN where undefined).

    ``shape`` sets the output grid (default: the image's); the whole image
    is sampled, whatever its size.
    """
    img = np.asarray(image, dtype=np.float64)
    ok = np.isfinite(img).astype(np.float64)
    shape = img.shape if shape is None else tuple(shape)
    rr, cc = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float64)
    coords = [rr + shift[0], cc + shift[1]]
    val = ndimage.map_coordinates(
        np.where(ok > 0, img, 0.0), coords, order=1, cval=0.0
    )
    wt = ndimage.map_coordinates(ok, coords, order=1, cval=0.0)
    out = np.full(shape, np.nan)
    good = wt > 0.999
    out[good] = val[good] / wt[good]
    return out


# ------------------------------------------------------------- geometry


@dataclass
class Alignment:
    """What injection uses from a trial's ``*_ALIGNMENTDATA.h5``.

    Attributes
    ----------
    path : str
        The file.
    ds_frames : ndarray of float
        Line times of the registered (DS) frames.
    motion_r, motion_c : ndarray of float
        ``motionDSr``, ``motionDSc`` (NaN for missing frames).
    trim_r0, trim_c0 : int
        One-based first DMD row and column of the registration window.
    margin : int
        ``maxshift``: padding of the registered grid.
    compensation : dict or None
        ``refOffset``, ``frameRefX``, ``frameRefY``, ``mapping`` and
        ``yLead`` when the registration undid the online correction.
    mean_im : ndarray of float
        ``(channels, rows, cols)`` mean registered image.
    """

    path: str
    ds_frames: np.ndarray
    motion_r: np.ndarray
    motion_c: np.ndarray
    trim_r0: int
    trim_c0: int
    margin: int
    compensation: Optional[dict]
    mean_im: np.ndarray

    @property
    def shape(self) -> Tuple[int, int]:
        """Registered frame ``(rows, cols)``."""
        return self.mean_im.shape[1], self.mean_im.shape[2]


def read_alignment(path: Union[str, Path]) -> Alignment:
    """Read the fields injection needs from an alignment file.

    Works for MATLAB- and Python-written files (both column-major).
    """

    def vec(ds):
        """A dataset as a flat float vector."""
        return np.asarray(ds[()], dtype=np.float64).ravel()

    with h5py.File(path, "r") as f:
        s2 = f["slap2"]
        comp = None
        if "onlineCompensation" in s2:
            oc = s2["onlineCompensation"]
            if "enabled" in oc and bool(vec(oc["enabled"])[0]):
                comp = {
                    "refOffset": vec(oc["refOffset"]),
                    "frameRefX": vec(oc["frameRefX"]),
                    "frameRefY": vec(oc["frameRefY"]),
                    "mapping": np.asarray(oc["mapping"][()], np.float64).T,
                    "yLead": vec(oc["yLead"]),
                }
        return Alignment(
            path=str(path),
            ds_frames=vec(f["DSframes"]),
            motion_r=vec(f["motionDSr"]),
            motion_c=vec(f["motionDSc"]),
            trim_r0=int(vec(s2["trimRows"])[0]),
            trim_c0=int(vec(s2["trimCols"])[0]),
            margin=int(round(1 - np.min(np.asarray(s2["viewR"][()])))),
            compensation=comp,
            mean_im=np.asarray(f["meanIM"][()], dtype=np.float64).T,
        )


def _continuous(alignment: Alignment) -> np.ndarray:
    """``(2, frames)`` continuous motion of every DS frame (NaN if missing).

    With compensation, the recorded motion plus ``mapping @ (frameRef -
    refOffset)``: the motion relative to the frame of ``refOffset``.
    """
    move = np.stack([alignment.motion_r, alignment.motion_c])
    comp = alignment.compensation
    if comp is not None:
        ref = np.stack([comp["frameRefX"], comp["frameRefY"]])
        move = move + comp["mapping"] @ (ref - comp["refOffset"][:, None])
    return move


def continuous_motion(alignment: Alignment):
    """DS frame times and the motion to interpolate (``onlineCompensation``).

    Returns
    -------
    times, motion_r, motion_c : ndarray
        Frames with finite motion only. With compensation, the motion is
        relative to the frame of ``refOffset`` (as SILo's ``loadTrial``).
    """
    t = alignment.ds_frames
    mr, mc = _continuous(alignment)
    ok = np.isfinite(mr) & np.isfinite(mc)
    if not ok.any():
        raise ValueError(f"{alignment.path} has no frame with motion")
    return t[ok], mr[ok], mc[ok]


def motion_at(alignment: Alignment, lines) -> np.ndarray:
    """Continuous motion at line times (``pchip``, extrapolated).

    Returns
    -------
    ndarray
        ``(2, n)`` rows and columns.
    """
    t, mr, mc = continuous_motion(alignment)
    lines = np.asarray(lines, dtype=np.float64)
    if t.size == 1:
        return np.stack(
            [np.full(lines.shape, mr[0]), np.full(lines.shape, mc[0])]
        )
    return np.stack(
        [PchipInterpolator(t, m, extrapolate=True)(lines) for m in (mr, mc)]
    )


@dataclass
class TrialGeometry:
    """Registered position of the samples of every line of a trial.

    A sample of DMD pixel ``(r, c)`` (zero-based) on line ``l`` lies at
    ``(r, c) + shift[l - first_line]`` in the trial's registered frame.

    Attributes
    ----------
    first_line : int
        One-based line of ``shift[0]``.
    shift : ndarray of float
        ``(lines, 2)`` row and column displacement.
    compensated : bool
        Whether the registration undid the online correction.
    """

    first_line: int
    shift: np.ndarray
    compensated: bool

    @property
    def last_line(self) -> int:
        """One-based last line covered."""
        return self.first_line + self.shift.shape[0] - 1


def trial_geometry(
    layout: RasterLayout, alignment: Alignment, lines: Tuple[int, int]
) -> TrialGeometry:
    """Where each line's samples lie in the trial's registered frame.

    Parameters
    ----------
    layout : RasterLayout
        The recording (its line headers give the online offsets).
    alignment : Alignment
        The trial's registration.
    lines : (int, int)
        First and last one-based line.

    Returns
    -------
    TrialGeometry
        See the module docstring for the formula.
    """
    lo, hi = int(lines[0]), int(lines[1])
    ln = np.arange(lo, hi + 1)
    move = motion_at(alignment, ln)
    off = np.zeros((2, ln.size))
    comp = alignment.compensation
    if comp is not None:
        lead = tuple(int(v) for v in comp["yLead"])
        offs = line_offsets(layout, (lo, hi), lead=lead)
        xy = np.stack([offs.x[ln - 1], offs.y_eff[ln - 1]]).astype(np.float64)
        off = comp["mapping"] @ (xy - comp["refOffset"][:, None])
    trim = np.array([alignment.trim_r0, alignment.trim_c0], np.float64)
    shift = off - (trim - 1)[:, None] + alignment.margin - move
    return TrialGeometry(lo, np.ascontiguousarray(shift.T), comp is not None)


def trial_lines(layout: RasterLayout, first, last) -> Tuple[int, int]:
    """Whole lines of a trial's (possibly fractional) line range."""
    lo = max(1, int(math.ceil(float(first))))
    hi = min(layout.total_lines, int(math.floor(float(last))))
    if hi < lo:
        raise ValueError(f"trial lines {first}..{last} hold no whole line")
    return lo, hi


# ------------------------------------------------------------- injection


class SampleModel:
    """Expected extra raw counts at the raster samples of one trial.

    Parameters
    ----------
    layout : RasterLayout
        The recording.
    geometry : TrialGeometry
        Registered position of each line's samples.
    positions : ndarray
        ``(sources, 2)`` source positions in the trial's registered frame.
    footprints : sequence
        One footprint per source (callables of row and column offsets with
        a ``radius``).
    traces : ndarray
        ``(sources, lines)`` amplitude of each source at each line of
        ``geometry`` (size units).
    photon_scale : float
        Raw counts per size unit.

    Attributes
    ----------
    slots : ndarray of int
        Raster slots any source can reach during the trial (the only ones
        :meth:`expected` evaluates).
    """

    def __init__(
        self,
        layout: RasterLayout,
        geometry: TrialGeometry,
        positions: np.ndarray,
        footprints: Sequence,
        traces: np.ndarray,
        photon_scale: float = PHOTON_SCALE,
    ):
        """Find the (slot, source) pairs within reach."""
        self.layout = layout
        self.geometry = geometry
        self.positions = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
        self.footprints = list(footprints)
        self.traces = np.asarray(traces, dtype=np.float64)
        self.photon_scale = float(photon_scale)
        n_src = self.positions.shape[0]
        if len(self.footprints) != n_src or self.traces.shape != (
            n_src,
            geometry.shift.shape[0],
        ):
            raise ValueError("positions, footprints and traces disagree")
        pix = layout.slot_pixel - 1
        r0 = (pix // layout.n_cols).astype(np.float64)
        c0 = (pix % layout.n_cols).astype(np.float64)
        smin, smax = geometry.shift.min(0), geometry.shift.max(0)
        active = np.flatnonzero(np.any(self.traces != 0, axis=1))
        ps, pj = [], []
        for j in active:
            rad = self.footprints[j].radius
            pr, pc = self.positions[j]
            sel = np.flatnonzero(
                (r0 >= pr - rad - smax[0])
                & (r0 <= pr + rad - smin[0])
                & (c0 >= pc - rad - smax[1])
                & (c0 <= pc + rad - smin[1])
            )
            ps.append(sel)
            pj.append(np.full(sel.size, j))
        pair_slot = np.concatenate(ps) if ps else np.zeros(0, np.int64)
        self.pair_src = np.concatenate(pj) if pj else np.zeros(0, np.int64)
        self.slots, self.pair_k = np.unique(pair_slot, return_inverse=True)
        self._r0, self._c0 = r0[self.slots], c0[self.slots]
        groups: Dict[int, list] = {}
        for g, j in enumerate(self.pair_src):
            groups.setdefault(id(self.footprints[j]), []).append(g)
        self._groups = [np.array(v) for v in groups.values()]

    def lines(self, cycles: np.ndarray) -> np.ndarray:
        """One-based line of each (cycle, slot) of :attr:`slots`."""
        p = self.layout.lines_per_cycle
        return cycles[:, None] * p + self.layout.slot_line[self.slots][None, :]

    def expected(self, cycles) -> np.ndarray:
        """Expected extra raw counts of the reachable slots.

        Parameters
        ----------
        cycles : array_like of int
            Zero-based cycles.

        Returns
        -------
        ndarray of float64
            ``(cycles, slots)``; 0 on lines outside the geometry.
        """
        cycles = np.asarray(cycles, dtype=np.int64).ravel()
        n_c, n_k = cycles.size, self.slots.size
        out = np.zeros(n_c * n_k)
        if n_k == 0:
            return out.reshape(n_c, n_k)
        g = self.geometry
        lines = self.lines(cycles)
        inside = (lines >= g.first_line) & (lines <= g.last_line)
        li = np.clip(lines - g.first_line, 0, g.shift.shape[0] - 1)
        sh_r, sh_c = g.shift[li, 0], g.shift[li, 1]
        rows = np.arange(n_c)[:, None] * n_k
        for grp in self._groups:
            k = self.pair_k[grp]
            j = self.pair_src[grp]
            fp = self.footprints[j[0]]
            dr = self._r0[k][None, :] + sh_r[:, k] - self.positions[j, 0]
            dc = self._c0[k][None, :] + sh_c[:, k] - self.positions[j, 1]
            v = fp(dr, dc) * self.traces[j[None, :], li[:, k]]
            v[~inside[:, k]] = 0.0
            out += np.bincount(
                (rows + k[None, :]).ravel(), v.ravel(), minlength=n_c * n_k
            )
        return out.reshape(n_c, n_k) * self.photon_scale


@dataclass
class InjectionStats:
    """What was added to one trial-path.

    Attributes
    ----------
    expected_photons : float
        Sum of the expected photons over all reachable samples.
    expected_photons_invalid : float
        Part of it at invalid samples (left untouched).
    drawn_photons : int
        Photons drawn at valid samples.
    added_raw : int
        Raw counts added.
    lost_raw : int
        Raw counts not added because samples saturated.
    saturated : int
        Samples that hit 32767.
    changed_samples : int
        Samples whose value changed.
    reachable_slots : int
        Slots any source can reach.
    seconds : float
        Wall time of the injection.
    render_seconds : float
        Wall time of the frame traces.
    """

    expected_photons: float = 0.0
    expected_photons_invalid: float = 0.0
    drawn_photons: int = 0
    added_raw: int = 0
    lost_raw: int = 0
    saturated: int = 0
    changed_samples: int = 0
    reachable_slots: int = 0
    seconds: float = 0.0
    render_seconds: float = 0.0


def _file_spans(layout: RasterLayout, c0: int, c1: int):
    """Yield (file index, first local cycle, count, first global cycle)."""
    ends = np.cumsum(layout.file_cycles)
    starts = ends - layout.file_cycles
    for f, (s, e) in enumerate(zip(starts, ends)):
        lo, hi = max(c0, s), min(c1, e)
        if lo < hi:
            yield f, int(lo - s), int(hi - lo), int(lo)


def inject_samples(
    layout: RasterLayout,
    model: SampleModel,
    raw_per_photon: float,
    channel: int,
    seed_key: Sequence[int],
    chunk_cycles: int = 64,
) -> InjectionStats:
    """Add Poisson photons of a sample model to a recording, in place.

    Parameters
    ----------
    layout : RasterLayout
        The recording to change (its ``files`` are opened read-write).
    model : SampleModel
        Expected extra raw counts.
    raw_per_photon : float
        Raw units per photon ``g``.
    channel : int
        Zero-based channel changed.
    seed_key : sequence of int
        ``(seed, path, trial)``: each cycle's draws come from their own
        generator keyed by it and the cycle, so they do not depend on
        ``chunk_cycles``.
    chunk_cycles : int
        Cycles evaluated at a time.

    Returns
    -------
    InjectionStats
    """
    if int(chunk_cycles) < 1:
        raise ValueError("chunk_cycles must be at least 1")
    t_start = time.perf_counter()
    stats = InjectionStats(reachable_slots=int(model.slots.size))
    if model.slots.size == 0:
        stats.seconds = time.perf_counter() - t_start
        return stats
    if not 0 <= channel < layout.num_channels:
        raise ValueError(f"channel {channel} is not recorded")
    g = float(raw_per_photon)
    p = layout.lines_per_cycle
    geo = model.geometry
    c_lo = (geo.first_line - 1) // p
    c_hi = (geo.last_line - 1) // p + 1
    li = layout.slot_line[model.slots] - 1
    word = (
        layout.row_base[li]
        + channel * layout.n_per_channel[li]
        + layout.slot_pos[model.slots]
    )
    seed, *key = (int(k) for k in seed_key)
    for f, local, count, first in _file_spans(layout, c_lo, c_hi):
        mm = np.memmap(
            layout.files[f],
            dtype=np.int16,
            mode="r+",
            offset=2 * int(layout.file_offset_words[f]),
            shape=(int(layout.file_cycles[f]), layout.words_per_cycle),
        )
        for a in range(0, count, chunk_cycles):
            b = min(count, a + chunk_cycles)
            cycles = np.arange(first + a, first + b)
            lam = model.expected(cycles) / g
            for i, c in enumerate(cycles):
                rng = _rng(seed, _STREAM_PHOTONS, *key, int(c))
                n = rng.poisson(lam[i])
                nz = np.flatnonzero(n)
                row = mm[local + a + i]
                old = row[word].astype(np.int64)
                bad = old == INVALID_SAMPLE
                stats.expected_photons += float(lam[i].sum())
                stats.expected_photons_invalid += float(lam[i][bad].sum())
                if nz.size == 0:
                    continue
                inc = np.floor(n[nz] * g + rng.random(nz.size)).astype(
                    np.int64
                )
                ok = ~bad[nz]
                nz, inc = nz[ok], inc[ok]
                new = np.minimum(old[nz] + inc, SAMPLE_MAX)
                stats.drawn_photons += int(n[nz].sum())
                stats.added_raw += int((new - old[nz]).sum())
                stats.lost_raw += int((old[nz] + inc - new).sum())
                stats.saturated += int(np.sum(old[nz] + inc > SAMPLE_MAX))
                changed = new != old[nz]
                stats.changed_samples += int(changed.sum())
                row[word[nz[changed]]] = new[changed].astype(np.int16)
        mm.flush()
        del mm
    stats.seconds = time.perf_counter() - t_start
    return stats


# ------------------------------------------------------------ session level


def meta_path(dat: Union[str, Path]) -> Path:
    """The ``.meta`` file of a ``.dat`` file (as GIAnT's ``loadMetadata``)."""
    dat = Path(dat)
    m = re.search(r"DMD\d", dat.stem)
    if m is None:
        raise ValueError(f"{dat.name} does not name its DMD")
    return dat.with_name(dat.stem[: m.end()] + ".meta")


def _copy_recording(layout: RasterLayout, dest: Path, done: set) -> None:
    """Copy a recording's data files (and ``.meta``) once into ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    for src in [Path(p) for p in layout.files]:
        if src.name not in done:
            shutil.copyfile(src, dest / src.name)
            done.add(src.name)
        try:
            meta = meta_path(src)
        except ValueError:
            continue
        if meta.exists() and meta.name not in done:
            shutil.copyfile(meta, dest / meta.name)
            done.add(meta.name)
        elif not meta.exists():
            log.warning("%s has no %s; not copied", src.name, meta.name)


def _recording_files(names, keep, paths, data_dir: Path) -> dict:
    """Each injected cell's data files; reject copies that would collide.

    Returns
    -------
    dict
        ``{zero-based path: {zero-based trial: tuple of resolved files}}``.

    Raises
    ------
    ValueError
        If two different originals (data or ``.meta`` files) would be
        copied to the same name in the copy's flat folder.
    """
    out, origin = {}, {}
    for dmd in paths:
        d = int(dmd) - 1
        if not 0 <= d < names.shape[0]:
            raise ValueError(f"path {dmd} is not in the trial table")
        out[d] = {}
        for t in keep:
            if not str(names[d, t] or "").strip():
                continue
            layout = raster_layout(data_dir / str(names[d, t]))
            files = tuple(str(Path(p).resolve()) for p in layout.files)
            out[d][int(t)] = files
            for f in files:
                extra = []
                try:
                    extra = [str(meta_path(f).resolve())]
                except ValueError:
                    pass
                for src in [f] + extra:
                    name = Path(src).name
                    if origin.setdefault(name, src) != src:
                        raise ValueError(
                            f"{origin[name]} and {src} would both be copied "
                            f"as {name}"
                        )
    return out


def _relocated(layout: RasterLayout, dest: Path) -> RasterLayout:
    """The same layout with its files in ``dest``."""
    files = tuple(str(dest / Path(p).name) for p in layout.files)
    return replace(layout, files=files)


def _cols(value, keep: np.ndarray, n: int):
    """Columns ``keep`` of a per-trial value (others unchanged)."""
    arr = np.asarray(value)
    if arr.ndim == 2 and arr.shape[1] == n:
        return arr[:, keep]
    return value


def write_copy_table(
    table: dict,
    keep: np.ndarray,
    data_dir: Path,
    save_dir: Path,
    paths: Optional[Sequence[int]] = None,
) -> Path:
    """The trial table of the injected copy (``save_dir/trial_table.h5``).

    The original's table with only the trials ``keep`` (zero-based columns),
    ``datadr`` and ``savedr`` pointing to the copy (absolute), no
    registration or extraction results, filenames reduced to the copies'
    names in ``data_dir``, and empty filenames for paths not in ``paths``
    (one-based DMDs; default all), whose files are not copied.
    """
    n = np.atleast_2d(np.asarray(table["filename"], dtype=object)).shape[1]
    out = {}
    for key, value in table.items():
        if key in ("motion_correction", "source_extraction", "row_major"):
            continue
        if key == "slap2_info" and isinstance(value, dict):
            out[key] = {k: _cols(v, keep, n) for k, v in value.items()}
        elif key == "filename":
            names = np.atleast_2d(np.asarray(value, dtype=object))[:, keep]
            for d in range(names.shape[0]):
                for t in range(names.shape[1]):
                    cell = str(names[d, t] or "").strip()
                    # the copies are flat in data_dir
                    names[d, t] = Path(cell).name if cell else ""
                if paths is not None and d + 1 not in paths:
                    names[d] = ""
            out[key] = names
        else:
            out[key] = _cols(value, keep, n)
    out["datadr"] = str(Path(data_dir).absolute())
    out["savedr"] = str(Path(save_dir).absolute())
    path = Path(save_dir) / "trial_table.h5"
    save_struct_h5(path, out)
    return path


def _grid(value, n_paths: int, n: int) -> np.ndarray:
    """A per-path, per-trial value as a ``(paths, trials)`` array."""
    return np.asarray(value).reshape(n_paths, n)


def _frame_lines(first, last, line_hz: float, hz: float) -> np.ndarray:
    """``ceil(first:dt:last)`` (SILo's 200 Hz frame lines)."""
    from ..registration.multiroi import matlab_colon

    return np.ceil(
        matlab_colon(float(first), line_hz / hz, float(last))
    ).astype(np.int64)


@dataclass
class PathPlan:
    """Sources of one path and the frame they are placed in.

    Attributes
    ----------
    positions : ndarray
        ``(sources, 2)`` in the path frame.
    footprints : list
        One footprint per source.
    trial_shift : ndarray
        ``(trials, 2)``: trial frame = path frame + shift.
    mean_image : ndarray
        Mean image of the injected channel in the path frame.
    valid : ndarray of bool
        Pixels with data in every trial.
    n_candidates : int
        Candidate pixels of the placement (-1 if positions were given).
    """

    positions: np.ndarray
    footprints: list
    trial_shift: np.ndarray
    mean_image: np.ndarray
    valid: np.ndarray
    n_candidates: int = -1


def plan_path(
    alignments: Sequence[Alignment],
    n_sources: int,
    settings: InjectionSettings,
    placement: PlacementSettings,
    rng: np.random.Generator,
    positions: Optional[np.ndarray] = None,
    exclude_rc: Optional[np.ndarray] = None,
    trial_shift: Optional[np.ndarray] = None,
    footprints: Optional[Sequence] = None,
    max_shift: int = 8,
) -> PathPlan:
    """Align a path's trials and place its sources.

    The path frame is the first alignment's registered frame. Trial shifts
    are estimated from the mean images unless given; the path's mean image
    is the mean of the trials' mean images moved into the path frame.
    """
    ch = settings.channel
    ref = alignments[0].mean_im[ch]
    if trial_shift is None:
        trial_shift = np.array(
            [
                (
                    np.zeros(2)
                    if k == 0
                    else align_images(ref, a.mean_im[ch], max_shift)
                )
                for k, a in enumerate(alignments)
            ]
        )
    trial_shift = np.asarray(trial_shift, dtype=np.float64).reshape(-1, 2)
    if trial_shift.shape[0] != len(alignments):
        raise ValueError("one trial shift per trial is needed")
    moved = np.stack(
        [
            shift_image(a.mean_im[ch], s, ref.shape)
            for a, s in zip(alignments, trial_shift)
        ]
    )
    valid = np.all(np.isfinite(moved), axis=0)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels
        mean_image = np.nanmean(moved, axis=0)
    n_cand = -1
    if positions is None:
        positions, n_cand = place_sources(
            np.where(valid, mean_image, np.nan),
            n_sources,
            rng,
            placement,
            valid,
            exclude_rc,
        )
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    if footprints is None:
        fp = GaussianFootprint(settings.sigma_px, settings.radius_sigmas)
        footprints = [fp] * positions.shape[0]
    if len(footprints) != positions.shape[0]:
        raise ValueError("one footprint per source is needed")
    return PathPlan(
        positions,
        list(footprints),
        trial_shift,
        mean_image,
        valid,
        n_cand,
    )


def _render_device(device: str) -> str:
    """``'auto'``: CUDA when available, else the CPU."""
    from ..registration.multiroi import resolve_device

    return str(resolve_device(device))


def expected_frames(
    layout: RasterLayout,
    alignment: Alignment,
    model: SampleModel,
    frame_lines: np.ndarray,
    pixels_rc: np.ndarray,
    shift_rc,
    hz: float,
    channel: int = 0,
    device: str = "cpu",
) -> np.ndarray:
    """Noiseless registered frames of a sample model, at given pixels.

    The model's expected counts are formed into frames at ``frame_lines``
    as SILo's extraction forms them at ``hz`` (kernel ``ceil(line rate /
    hz)`` lines, the samples exactly at its reach and the recording's
    invalid samples left out) and resampled with the
    registration's motion into the path frame (the trial frame moved by
    ``shift_rc``; see :func:`~giant_python.simulation.render.
    iter_registered_frames`).

    Parameters
    ----------
    layout, alignment, model
        The recording, its registration and what is injected.
    frame_lines : ndarray of int
        One-based frame times.
    pixels_rc : ndarray of int
        ``(n, 2)`` zero-based pixels of the path frame; a pixel is NaN where
        its position in the trial's registered frame has no data.
    shift_rc : (float, float)
        Trial frame = path frame + shift.
    hz : float
        Frame rate that sets the kernel.
    channel : int
        Channel whose invalid samples are left out. Freshness (the
        resampling weights) comes from channel 0's valid samples, as in
        registration.
    device : str
        Where frames are formed.

    Returns
    -------
    ndarray of float
        ``(n, frames)`` in raw counts (NaN where a pixel has no data).
    """
    from .render import iter_registered_frames

    frame_lines = np.asarray(frame_lines, dtype=np.int64)
    p = layout.lines_per_cycle
    dt = int(math.ceil((1.0 / layout.line_period_s) / hz))
    reach = max(3 * dt, p)
    c0 = max(0, (int(frame_lines.min()) - reach - 1) // p)
    c1 = min(layout.num_cycles, (int(frame_lines.max()) + reach - 1) // p + 1)
    # channel 0 sets the freshness; the injected channel goes last
    n_ch = 1 if channel == 0 else 2
    samples = np.zeros((n_ch, layout.slot_line.size, c1 - c0), np.float32)
    for a in range(c0, c1, 256):
        cyc = np.arange(a, min(c1, a + 256))
        samples[-1][model.slots[:, None], (cyc - c0)[None, :]] = (
            model.expected(cyc).T
        )
    orig = read_raster_samples(layout, (c0, c1))
    samples[-1][orig[channel] == INVALID_SAMPLE] = INVALID_SAMPLE
    samples[0][orig[0] == INVALID_SAMPLE] = INVALID_SAMPLE
    del orig
    # whole pixels of the shift index the trial grid; the rest is rendered
    whole = np.rint(np.asarray(shift_rc, dtype=np.float64))
    part = np.asarray(shift_rc, dtype=np.float64) - whole
    pix = np.asarray(pixels_rc, dtype=np.int64).reshape(-1, 2)
    pix = pix + whole.astype(np.int64)
    h, w = alignment.shape
    inside = (
        (pix[:, 0] >= 0) & (pix[:, 0] < h) & (pix[:, 1] >= 0) & (pix[:, 1] < w)
    )
    pr = np.clip(pix[:, 0], 0, h - 1)
    pc = np.clip(pix[:, 1], 0, w - 1)
    out = np.full((pix.shape[0], frame_lines.size), np.nan)
    for idx, frames, _ in iter_registered_frames(
        layout,
        samples,
        c0,
        alignment,
        frame_lines,
        dt,
        device,
        128,
        part,
        exclusive=True,
    ):
        out[:, idx] = frames[-1][:, pr, pc].T.cpu().numpy()
    out[~inside] = np.nan
    return out


def overlap_classes(positions: np.ndarray, footprints: Sequence) -> np.ndarray:
    """Classes of sources whose footprints do not overlap (greedy colouring).

    Sources ``i`` and ``j`` overlap when they are closer than the sum of
    their footprints' radii.

    Returns
    -------
    ndarray of int
        Class of each source, from 0.
    """
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    radius = np.array([fp.radius for fp in footprints], dtype=np.float64)
    dist = np.hypot(*(pos[:, None, :] - pos[None, :, :]).transpose(2, 0, 1))
    near = dist < radius[:, None] + radius[None, :]
    cls = np.full(pos.shape[0], -1)
    for j in range(pos.shape[0]):
        taken = set(cls[near[j] & (cls >= 0)])
        cls[j] = next(c for c in range(pos.shape[0]) if c not in taken)
    return cls


def calibrate_frame_peaks(
    layout: RasterLayout,
    alignment: Alignment,
    geometry: TrialGeometry,
    positions: np.ndarray,
    shift_rc,
    footprints: Sequence,
    settings: InjectionSettings,
    rng: np.random.Generator,
    device: str = "cpu",
) -> np.ndarray:
    """How much of an event's peak each source's frames keep.

    Isolated unit events (tissue units) of every source are placed in a
    ``calibration_s`` stretch of the trial, at random phases of the frame
    grid; their noiseless frames at ``analyze_hz`` are formed as SILo forms
    them (:func:`expected_frames`), in the path frame. Sources whose
    footprints overlap take turns (:func:`overlap_classes`), so that each
    event is alone at its source's peak pixel.

    Raises
    ------
    ValueError
        If a source has no data at its peak pixel.

    Returns
    -------
    ndarray of float
        Per source: the mean, over its events, of the event's largest frame
        value at the source's peak pixel (the path-frame pixel nearest its
        footprint's maximum, :func:`peak_pixels`), in units of
        ``photon_scale``.
    """
    period = layout.line_period_s
    n_src = positions.shape[0]
    dt = int(math.ceil((1.0 / period) / settings.analyze_hz))
    reach = max(3 * dt, layout.lines_per_cycle)
    tail = peak_delay_s(settings.tau_s, settings.rise_s) + 6 * settings.tau_s
    gap = tail / period + 2 * reach
    a = geometry.first_line + settings.edge_s / period + reach
    b = min(
        geometry.last_line - reach - gap,
        a + settings.calibration_s / period,
    )
    turn = overlap_classes(positions, footprints)
    n_turns = int(turn.max()) + 1 if n_src else 1
    slots = np.arange(int(max(0.0, b - a) / gap))
    if slots.size < n_turns:
        raise ValueError(
            "the trial is too short to calibrate frame peaks (needs "
            f"{n_turns} event slots of {gap * period:.3f} s)"
        )
    src, onset = [], []
    for j in range(n_src):
        k = slots[slots % n_turns == turn[j]]
        src.append(np.full(k.size, j))
        onset.append(a + k * gap + rng.uniform(0, gap - tail / period, k.size))
    src = np.concatenate(src) if src else np.zeros(0, np.int64)
    onset = np.concatenate(onset) if onset else np.zeros(0)
    ev = Events(
        src,
        np.ones(src.size, np.int64),
        onset,
        np.zeros(src.size, np.int64),
        np.ones(src.size),
    )
    lines = (geometry.first_line, geometry.last_line)
    traces = event_traces(ev, n_src, lines, period, settings)
    model = SampleModel(
        layout,
        geometry,
        positions + np.asarray(shift_rc),
        footprints,
        traces,
        settings.photon_scale,
    )
    step = (1.0 / period) / settings.analyze_hz
    fl = np.unique(np.ceil(np.arange(a - reach, b + gap, step))).astype(
        np.int64
    )
    vals = expected_frames(
        layout,
        alignment,
        model,
        fl,
        peak_pixels(positions, footprints),
        shift_rc,
        settings.analyze_hz,
        settings.channel,
        device,
    )
    peaks = np.full(src.size, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # pixels w/o data
        for e, (j, t0) in enumerate(zip(src, ev.onset_line)):
            sel = (fl >= t0 - reach) & (fl <= t0 + tail / period)
            if sel.any():
                peaks[e] = np.nanmax(vals[j, sel])
        gain = np.array(
            [
                np.nanmean(peaks[src == j]) if np.any(src == j) else np.nan
                for j in range(n_src)
            ]
        )
    gain = gain / settings.photon_scale
    bad = np.flatnonzero(~(gain > 0))
    if bad.size:
        raise ValueError(
            f"sources {bad.tolist()} have no data at their peak pixel, so "
            "their frame peaks cannot be calibrated; move them or use "
            "size_reference='tissue'"
        )
    return gain


def _inject_trial(ctx: dict, k: int, t: int) -> Tuple[dict, Events]:
    """Inject one trial-path (see :func:`inject_session`)."""
    s, plan, dmd = ctx["settings"], ctx["plan"], ctx["dmd"]
    layout = raster_layout(ctx["data_dir"] / str(ctx["names"][t]))
    _copy_recording(layout, ctx["raw_dir"], ctx["copied"])
    first, last = ctx["first"][t], ctx["last"][t]
    lo, hi = trial_lines(layout, first, last)
    align = ctx["aligns"][k]
    geo = trial_geometry(layout, align, (lo, hi))
    n_src = plan.positions.shape[0]
    ev = draw_events(
        n_src,
        first,
        last,
        layout.line_period_s,
        s,
        _rng(s.seed, _STREAM_EVENTS, dmd, t + 1),
        trial=t + 1,
    )
    scaled = Events(
        ev.source,
        ev.trial,
        ev.onset_line,
        ev.size_class,
        ev.size * ctx["scale"][ev.source],
    )
    period = layout.line_period_s
    traces = event_traces(scaled, n_src, (lo, hi), period, s)
    shift = plan.trial_shift[k]
    model = SampleModel(
        layout,
        geo,
        plan.positions + shift,
        plan.footprints,
        traces,
        s.photon_scale,
    )
    stats = inject_samples(
        _relocated(layout, ctx["raw_dir"]),
        model,
        ctx["gain"].raw_per_photon,
        s.channel,
        (s.seed, dmd, t + 1),
        ctx["chunk_cycles"],
    )
    fl = _frame_lines(first, last, 1.0 / period, s.analyze_hz)
    ideal = event_traces(ev, n_src, (lo, hi), period, s)
    trace = ideal[:, np.clip(fl, lo, hi) - lo]
    trace[:, (fl < lo) | (fl > hi)] = 0.0
    detail = {
        "trial": t + 1,
        "copy_trial": ctx["copy_col"][t],
        "filename": str(ctx["names"][t]),
        "alignment": align.path,
        "first_line": float(first),
        "last_line": float(last),
        "lines": (lo, hi),
        "compensated": geo.compensated,
        "line_period_s": period,
        "positions": plan.positions + shift,
        "frame_lines": fl,
        "trace": trace.astype(np.float32),
        "shift_range": np.stack([geo.shift.min(0), geo.shift.max(0)]),
        "stats": stats,
    }
    if s.frame_traces:
        t0 = time.perf_counter()
        frames = expected_frames(
            layout,
            align,
            model,
            fl,
            peak_pixels(plan.positions, plan.footprints),
            shift,
            s.analyze_hz,
            s.channel,
            ctx["device"],
        )
        reach = max(
            3 * int(math.ceil((1.0 / period) / s.analyze_hz)),
            layout.lines_per_cycle,
        )
        for a, b in ctx["neighbours"](t):
            # kernels reading another injected trial's samples
            frames[:, (fl + reach >= a) & (fl - reach <= b)] = np.nan
        detail["trace_frames"] = (frames / s.photon_scale).astype(np.float32)
        stats.render_seconds = time.perf_counter() - t0
    log.info(
        "path %d trial %d: %d events, %.0f photons expected, %d drawn, "
        "%.1f s (+ %.1f s frames)",
        dmd,
        t + 1,
        len(ev),
        stats.expected_photons,
        stats.drawn_photons,
        stats.seconds,
        stats.render_seconds,
    )
    return detail, ev


def _path_gain(settings: InjectionSettings, layout: RasterLayout):
    """Raw units per photon: given, or measured on the reference trial."""
    if settings.raw_per_photon is not None:
        return PhotonGain(
            settings.raw_per_photon, float("nan"), 0, (0, 0), settings.channel
        )
    return measure_raw_per_photon(layout, settings.channel)


def _neighbours(ctx: dict, cells: list):
    """For trial ``t``: line ranges of the other injected trials of the
    same recording (trials whose recordings share a data file, which may be
    named by different files of a CYCLE-split recording)."""
    files = {t: set(ctx["files"][t]) for t in cells}

    def ranges(t):
        """``(first, last)`` lines of the other trials of ``t``'s recording."""
        out = []
        for u in cells:
            if u != t and files[u] & files[t]:
                out.append((float(ctx["first"][u]), float(ctx["last"][u])))
        return out

    return ranges


def _inject_path(ctx: dict, cells: list, options: dict) -> dict:
    """Plan, calibrate and inject one path (see :func:`inject_session`)."""
    s, dmd = ctx["settings"], ctx["dmd"]
    ctx["neighbours"] = _neighbours(ctx, cells)
    ctx["aligns"] = [
        read_alignment(ctx["moco_dir"] / str(ctx["adata"][t])) for t in cells
    ]
    plan = plan_path(
        ctx["aligns"],
        options["n_sources"],
        s,
        ctx["placement"],
        _rng(s.seed, _STREAM_PLACE, dmd),
        options["positions"],
        options["exclude"],
        options["trial_shift"],
        options["footprints"],
    )
    ctx["plan"] = plan
    n_src = plan.positions.shape[0]
    log.info(
        "path %d: %d sources (%d candidates)", dmd, n_src, plan.n_candidates
    )
    ref = raster_layout(ctx["data_dir"] / str(ctx["names"][cells[0]]))
    ctx["gain"] = _path_gain(s, ref)
    log.info(
        "path %d: %.2f raw units per photon (mean pulse %.1f)",
        dmd,
        ctx["gain"].raw_per_photon,
        ctx["gain"].mean_pulse,
    )
    peaks = np.ones(n_src)
    if s.size_reference == "frames" and n_src and s.rate_hz > 0:
        lo, hi = trial_lines(
            ref, ctx["first"][cells[0]], ctx["last"][cells[0]]
        )
        peaks = calibrate_frame_peaks(
            ref,
            ctx["aligns"][0],
            trial_geometry(ref, ctx["aligns"][0], (lo, hi)),
            plan.positions,
            plan.trial_shift[0],
            plan.footprints,
            s,
            _rng(s.seed, _STREAM_CALIBRATE, dmd),
            ctx["device"],
        )
        log.info(
            "path %d: frames keep %.2f of an event's peak (median; "
            "%.2f-%.2f)",
            dmd,
            np.median(peaks),
            peaks.min(),
            peaks.max(),
        )
    ctx["scale"] = 1.0 / peaks
    details, events = [], []
    for k, t in enumerate(cells):
        d, ev = _inject_trial(ctx, k, t)
        details.append(d)
        events.append(ev)
    return {
        "plan": plan,
        "gain": ctx["gain"],
        "frame_peak": peaks,
        "events": Events.concat(events),
        "trials": details,
    }


def inject_session(
    trial_table: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    trials: Optional[Sequence[int]] = None,
    paths: Sequence[int] = (1, 2),
    n_sources: int = 50,
    settings: Optional[InjectionSettings] = None,
    placement: Optional[PlacementSettings] = None,
    positions: Optional[Dict[int, np.ndarray]] = None,
    exclude: Optional[Dict[int, np.ndarray]] = None,
    trial_shifts: Optional[Dict[int, np.ndarray]] = None,
    footprints: Optional[Dict[int, Sequence]] = None,
    data_dir: Optional[Union[str, Path]] = None,
    moco_dir: Optional[Union[str, Path]] = None,
    chunk_cycles: int = 64,
) -> dict:
    """Make an injected copy of a registered SLAP2 session and its truth.

    Parameters
    ----------
    trial_table : str or Path
        The original session's ``trial_table.h5``, registered (its
        ``motion_correction/fn_adata`` names the alignment files).
    out_dir : str or Path
        Output folder: the copy's data files and ``.meta`` files go to
        ``raw/``, its trial table (only the injected trials, nothing
        registered) to ``trial_table.h5`` and the truth to
        ``injection_truth.h5``.
    trials : sequence of int, optional
        One-based trials (columns of the table) to inject (default: all).
    paths : sequence of int
        DMDs to inject.
    n_sources : int
        Sources placed per path (when ``positions`` does not give them).
    settings, placement : optional
        What is injected and where sources may go.
    positions, exclude, trial_shifts, footprints : dict, optional
        Per DMD: source positions in the path frame (instead of placing
        them), existing sources to keep away from (path frame), trial shifts
        (one row per injected trial; trial frame = path frame + shift) and
        footprints (one per source).
    data_dir, moco_dir : str or Path, optional
        Where the raw data and the registration are (default: the table's
        ``datadr`` and ``savedr/motion_correction``).
    chunk_cycles : int
        Cycles evaluated at a time.

    Returns
    -------
    dict
        Summary: the output files, per path the sources, gain and
        statistics.
    """
    settings = settings or InjectionSettings()
    out_dir = Path(out_dir).absolute()
    table = load_struct_h5(trial_table)
    names = np.atleast_2d(np.asarray(table["filename"], dtype=object))
    n_paths, n_trials = names.shape
    info = table["slap2_info"]
    trials = list(range(1, n_trials + 1)) if trials is None else trials
    trials = [int(t) for t in trials]
    paths = [int(p) for p in paths]
    if not trials or min(trials) < 1 or max(trials) > n_trials:
        raise ValueError(f"trials must be in 1..{n_trials}")
    if len(set(trials)) < len(trials) or len(set(paths)) < len(paths):
        raise ValueError("trials and paths must not repeat")
    keep = np.array(trials) - 1
    data_dir = Path(data_dir or table["datadr"]).absolute()
    moco_dir = Path(
        moco_dir or Path(table["savedr"]) / "motion_correction"
    ).absolute()
    adata = np.atleast_2d(
        np.asarray(table["motion_correction"]["fn_adata"], dtype=object)
    )
    if int(chunk_cycles) < 1:
        raise ValueError("chunk_cycles must be at least 1")
    if (out_dir / "raw").resolve() == data_dir.resolve():
        raise ValueError(
            f"{out_dir / 'raw'} is the original's data folder; the copy "
            "must go elsewhere"
        )
    found = [
        name
        for name in ("raw", "trial_table.h5", TRUTH_NAME, "motion_correction")
        if (out_dir / name).exists()
    ]
    if found:
        raise ValueError(
            f"{out_dir} already holds {', '.join(found)}; inject into a new "
            "folder"
        )
    base = {
        "settings": settings,
        "placement": placement or PlacementSettings(),
        "data_dir": data_dir,
        "moco_dir": moco_dir,
        "raw_dir": out_dir / "raw",
        "copied": set(),
        "copy_col": {int(t): i + 1 for i, t in enumerate(keep)},
        "chunk_cycles": chunk_cycles,
        "device": _render_device(settings.device),
    }
    recordings = _recording_files(names, keep, paths, data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)  # inputs checked
    truth_paths, summary = {}, {"paths": {}}
    for dmd in paths:
        d = int(dmd) - 1
        if not 0 <= d < n_paths:
            raise ValueError(f"path {dmd} is not in the trial table")
        cells = [int(t) for t in keep if str(names[d, t] or "").strip()]
        if len(cells) < len(keep):
            log.warning(
                "path %d: %d trials have no recording",
                dmd,
                len(keep) - len(cells),
            )
        if not cells:
            continue
        ctx = dict(
            base,
            dmd=int(dmd),
            names=names[d],
            files=recordings[d],
            adata=adata[d],
            first=_grid(info["first_line"], n_paths, n_trials)[d],
            last=_grid(info["last_line"], n_paths, n_trials)[d],
        )
        options = {
            "n_sources": n_sources,
            "positions": (positions or {}).get(dmd),
            "exclude": (exclude or {}).get(dmd),
            "trial_shift": (trial_shifts or {}).get(dmd),
            "footprints": (footprints or {}).get(dmd),
        }
        p = _inject_path(ctx, cells, options)
        truth_paths[dmd] = p
        summary["paths"][dmd] = {
            "n_sources": int(p["plan"].positions.shape[0]),
            "n_candidates": p["plan"].n_candidates,
            "raw_per_photon": p["gain"].raw_per_photon,
            "mean_pulse": p["gain"].mean_pulse,
            "frame_peak_median": (
                float(np.median(p["frame_peak"]))
                if p["frame_peak"].size
                else None
            ),
            "n_events": len(p["events"]),
            "trials": [
                {"trial": q["trial"], **asdict(q["stats"])}
                for q in p["trials"]
            ],
        }
    copy_table = write_copy_table(
        table, keep, base["raw_dir"], out_dir, sorted(truth_paths)
    )
    origin = {
        "trial_table": str(Path(trial_table).absolute()),
        "data_dir": str(data_dir),
        "moco_dir": str(moco_dir),
        "copy_dir": str(base["raw_dir"]),
        "copy_trial_table": str(copy_table),
        "trials": trials,
    }
    write_truth(
        out_dir / TRUTH_NAME, truth_paths, settings, base["placement"], origin
    )
    summary.update(
        out_dir=str(out_dir),
        raw_dir=str(base["raw_dir"]),
        truth=str(out_dir / TRUTH_NAME),
        copy_trial_table=str(copy_table),
    )
    return summary


# --------------------------------------------------------------- truth file


def _code_version() -> str:
    """The package version."""
    from .. import __version__

    return f"giant_python {__version__}"


def write_truth(
    path: Union[str, Path],
    paths: dict,
    settings: InjectionSettings,
    placement: PlacementSettings,
    origin: dict,
) -> None:
    """Write the truth file (see :func:`read_truth` for its layout)."""
    with h5py.File(path, "w") as f:
        f.attrs["format"] = TRUTH_FORMAT
        f.attrs["created"] = datetime.now().astimezone().isoformat()
        f.attrs["code"] = _code_version()
        f.attrs["seed"] = int(settings.seed)
        f.attrs["settings"] = json.dumps(asdict(settings))
        f.attrs["placement"] = json.dumps(asdict(placement))
        f.attrs["origin"] = json.dumps(origin)
        f.attrs["frame"] = (
            "Positions are zero-based (row, col) of the registered movie of "
            "the ORIGINAL session's registration (MATLAB orientation, pixel "
            "centres at integers). pathN/positions_rc is in the path frame "
            "(the registered frame of the path's first injected trial); "
            "trial frame = path frame + trial_shift_rc; "
            "trials/TN/positions_rc is in trial N's registered frame."
        )
        f.attrs["units"] = (
            "Sizes and traces are in photon_scale raw counts (frame counts / "
            "25). size_reference 'frames': a size is the event's expected "
            "peak in analyze_hz frames at the source's peak pixel (path "
            "frame), as GIAnT's refit test counts it; the tissue amplitude "
            "injected is size / frame_peak. trace: sum of size x time course "
            "at the frame lines (peaks = sizes); trace_frames: the noiseless "
            "analyze_hz frames at the peak pixel. Lines are one-based, line l "
            "at (l - 1) * line_period_s after the recording's first line; "
            "onset_s is from the trial's first_line."
        )
        for dmd, p in paths.items():
            plan, gain, ev = p["plan"], p["gain"], p["events"]
            g = f.create_group(f"path{dmd}")
            g.attrs["dmd"] = int(dmd)
            g.attrs["raw_per_photon"] = float(gain.raw_per_photon)
            g.attrs["gain"] = json.dumps(gain.describe())
            g.attrs["n_candidates"] = int(plan.n_candidates)
            g.create_dataset("frame_peak", data=p["frame_peak"])
            g.create_dataset(
                "peak_pixel_rc",
                data=peak_pixels(plan.positions, plan.footprints),
            )
            g.create_dataset("positions_rc", data=plan.positions)
            g.create_dataset("trial_shift_rc", data=plan.trial_shift)
            g.create_dataset(
                "trials", data=np.array([t["trial"] for t in p["trials"]])
            )
            g.create_dataset(
                "mean_image",
                data=plan.mean_image.astype(np.float32),
                compression="gzip",
            )
            g.create_dataset("valid", data=plan.valid, compression="gzip")
            half = int(
                math.ceil(
                    max((fp.radius for fp in plan.footprints), default=0)
                )
            )
            patches, origins = [], []
            for pos, fp in zip(plan.positions, plan.footprints):
                patch, o = footprint_patch(fp, pos, half)
                patches.append(patch)
                origins.append(o)
            fg = g.create_group("footprint")
            fg.create_dataset(
                "patch",
                data=np.array(patches, dtype=np.float32).reshape(
                    -1, 2 * half + 1, 2 * half + 1
                ),
            )
            fg.create_dataset(
                "patch_origin_rc", data=np.array(origins).reshape(-1, 2)
            )
            fg.attrs["params"] = json.dumps(
                [fp.describe() for fp in plan.footprints]
            )
            eg = g.create_group("events")
            line_period = {t["trial"]: t["line_period_s"] for t in p["trials"]}
            first = {t["trial"]: t["first_line"] for t in p["trials"]}
            lp = np.array([line_period[t] for t in ev.trial])
            fl = np.array([first[t] for t in ev.trial])
            delay = peak_delay_s(settings.tau_s, settings.rise_s)
            eg.create_dataset("source", data=ev.source)
            eg.create_dataset("trial", data=ev.trial)
            eg.create_dataset("onset_line", data=ev.onset_line)
            eg.create_dataset("onset_s", data=(ev.onset_line - fl) * lp)
            eg.create_dataset(
                "peak_line",
                data=ev.onset_line + (delay / lp if lp.size else 0.0),
            )
            eg.create_dataset("size", data=ev.size)
            eg.create_dataset("size_class", data=ev.size_class)
            tg = g.create_group("trials_detail")
            for t in p["trials"]:
                h = tg.create_group(f"T{t['trial']}")
                for key in (
                    "trial",
                    "copy_trial",
                    "first_line",
                    "last_line",
                    "line_period_s",
                    "compensated",
                    "filename",
                    "alignment",
                ):
                    h.attrs[key] = t[key]
                h.attrs["lines"] = np.array(t["lines"])
                h.attrs["stats"] = json.dumps(asdict(t["stats"]))
                h.create_dataset("positions_rc", data=t["positions"])
                h.create_dataset("frame_lines", data=t["frame_lines"])
                for key in ("trace", "trace_frames"):
                    if key in t:
                        h.create_dataset(key, data=t[key], compression="gzip")
                h.create_dataset("shift_range", data=t["shift_range"])


def read_truth(path: Union[str, Path]) -> dict:
    """Read a truth file.

    Returns
    -------
    dict
        ``settings``, ``placement``, ``origin`` (dicts), ``seed``,
        ``frame`` and ``units`` (descriptions), and per DMD ``paths[k]``:
        ``positions_rc`` (sources x 2, path frame), ``trial_shift_rc``
        (trials x 2), ``trials`` (one-based original trial numbers),
        ``raw_per_photon``, ``gain``, ``frame_peak`` (share of an event's
        peak each source's frames keep), ``peak_pixel_rc``, ``mean_image``,
        ``valid``,
        ``footprint`` (``patch``, ``patch_origin_rc``, ``params``),
        ``events`` (``source``, ``trial``, ``onset_line``, ``onset_s``,
        ``peak_line``, ``size``, ``size_class``) and ``trials_detail[t]``
        (``positions_rc`` in the trial's registered frame, ``frame_lines``
        at ``analyze_hz``, ``trace`` and ``trace_frames`` sources x frames
        (``trace_frames`` NaN where a frame's kernel reads another injected
        trial of the same file), ``stats`` and the trial's lines; after
        :func:`record_copy_registration` also ``copy_shift_rc`` and
        ``positions_copy_rc``).
    """
    out = {"paths": {}}
    with h5py.File(path, "r") as f:
        if f.attrs.get("format") != TRUTH_FORMAT:
            raise ValueError(f"{path} is not a {TRUTH_FORMAT} file")
        for key in ("settings", "placement", "origin"):
            out[key] = json.loads(f.attrs[key])
        for key in ("seed", "frame", "units", "code", "created"):
            out[key] = f.attrs[key]
        for name, g in f.items():
            p = {k: g[k][()] for k in g if isinstance(g[k], h5py.Dataset)}
            p.update({k: g.attrs[k] for k in g.attrs})
            p["gain"] = json.loads(p["gain"])
            fg = g["footprint"]
            p["footprint"] = {k: fg[k][()] for k in fg}
            p["footprint"]["params"] = json.loads(fg.attrs["params"])
            p["events"] = {k: g["events"][k][()] for k in g["events"]}
            detail = {}
            for tname, h in g["trials_detail"].items():
                d = {k: h[k][()] for k in h}
                d.update({k: h.attrs[k] for k in h.attrs})
                d["stats"] = json.loads(d["stats"])
                detail[int(d["trial"])] = d
            p["trials_detail"] = detail
            out["paths"][int(g.attrs["dmd"])] = p
    return out


# ------------------------------------------- registration of the injected copy


def _grid_origin(alignment: Alignment) -> np.ndarray:
    """What a registration adds to a sample's DMD position besides motion.

    ``shift(l) = o(l) - (trim - 1) + maxshift - M(l)`` (see
    :func:`trial_geometry`); the constant part is ``maxshift - (trim - 1)``
    less ``mapping @ refOffset`` when compensated.
    """
    trim = np.array([alignment.trim_r0, alignment.trim_c0], np.float64)
    origin = alignment.margin - (trim - 1)
    comp = alignment.compensation
    if comp is not None:
        origin = origin - comp["mapping"] @ comp["refOffset"]
    return origin


def registration_offset(original: Alignment, copy: Alignment) -> dict:
    """Where a copy's registration puts what the original's put at ``p``.

    Two registrations of the same lines can differ by a constant: a grid is
    anchored on its initial template, and its window, padding and online
    reference offset may differ. A tissue point at ``p`` in the original's
    registered frame is at ``p + shift_rc`` in the copy's.

    Raises
    ------
    ValueError
        For registrations of different frames, or when only one of them
        undid the online correction (they then differ line by line).

    Returns
    -------
    dict
        ``shift_rc``; ``motion_offset_rc``: median of the copy's minus the
        original's continuous motion; ``rms_px`` and ``max_px``: length of
        the motion difference after removing that median; ``n_frames``.
    """
    if not np.array_equal(original.ds_frames, copy.ds_frames):
        raise ValueError("the registrations are of different frames")
    if (original.compensation is None) != (copy.compensation is None):
        raise ValueError(
            "only one registration undid the online motion correction"
        )
    d = (_continuous(copy) - _continuous(original)).T
    d = d[np.all(np.isfinite(d), axis=1)]
    if d.shape[0] == 0:
        raise ValueError("no frame has motion in both registrations")
    motion = np.median(d, axis=0)
    r = np.hypot(*(d - motion).T)
    return {
        "shift_rc": _grid_origin(copy) - _grid_origin(original) - motion,
        "motion_offset_rc": motion,
        "rms_px": float(np.sqrt(np.mean(r**2))),
        "max_px": float(r.max()),
        "n_frames": int(d.shape[0]),
    }


def record_copy_registration(out_dir: Union[str, Path]) -> dict:
    """Compare the copy's registration with the original's; record it.

    ``out_dir`` is an :func:`inject_session` output whose ``trial_table.h5``
    has been registered (``giant register``). For each injected trial-path
    the truth file gains ``copy_shift_rc`` and ``positions_copy_rc`` (the
    sources in the copy's registered frame: ``positions_rc + copy_shift_rc``)
    and the attributes ``copy_alignment``, ``copy_motion_rms_px`` and
    ``copy_motion_max_px``.

    Returns
    -------
    dict
        ``{dmd: {trial: registration_offset(...)}}``.
    """
    out_dir = Path(out_dir)
    table = load_struct_h5(out_dir / "trial_table.h5")
    if "motion_correction" not in table:
        raise ValueError(f"{out_dir / 'trial_table.h5'} is not registered")
    adata = np.atleast_2d(
        np.asarray(table["motion_correction"]["fn_adata"], dtype=object)
    )
    moco = Path(table["savedr"]) / "motion_correction"
    result = {}
    with h5py.File(out_dir / TRUTH_NAME, "r+") as f:
        for g in f.values():
            dmd = int(g.attrs["dmd"])
            result[dmd] = {}
            for h in g["trials_detail"].values():
                k = int(h.attrs["copy_trial"]) - 1
                path = moco / str(adata[dmd - 1, k])
                cmp = registration_offset(
                    read_alignment(h.attrs["alignment"]), read_alignment(path)
                )
                for key in ("copy_shift_rc", "positions_copy_rc"):
                    if key in h:
                        del h[key]
                h["copy_shift_rc"] = cmp["shift_rc"]
                h["positions_copy_rc"] = (
                    h["positions_rc"][()] + cmp["shift_rc"]
                )
                h.attrs["copy_alignment"] = str(path)
                h.attrs["copy_motion_rms_px"] = cmp["rms_px"]
                h.attrs["copy_motion_max_px"] = cmp["max_px"]
                result[dmd][int(h.attrs["trial"])] = cmp
    return result


# --------------------------------------------------------------------- CLI


def _read_per_path(path: Optional[str]) -> dict:
    """``{dmd: array}`` from an HDF5 file with datasets ``path1``, ..."""
    if not path:
        return {}
    out = {}
    with h5py.File(path, "r") as f:
        for name in f:
            m = re.fullmatch(r"path(\d+)", name)
            if m:
                out[int(m.group(1))] = np.asarray(f[name][()], np.float64)
    return out


def _add_inject_args(ap: argparse.ArgumentParser) -> None:
    """Options of ``inject``."""
    d = InjectionSettings()
    pl = PlacementSettings()
    ap.add_argument("trial_table", help="the original's trial_table.h5")
    ap.add_argument(
        "out_dir", help=f"output folder (raw/, trial_table.h5, {TRUTH_NAME})"
    )
    ap.add_argument(
        "--trials",
        type=int,
        nargs="+",
        help="one-based trials to inject (default: all)",
    )
    ap.add_argument("--paths", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--n-sources", type=int, default=50)
    ap.add_argument("--rate-hz", type=float, default=d.rate_hz)
    ap.add_argument("--min-interval-s", type=float, default=d.min_interval_s)
    ap.add_argument("--edge-s", type=float, default=d.edge_s)
    ap.add_argument(
        "--sizes", type=float, nargs="+", default=list(d.size_classes)
    )
    ap.add_argument("--class-weights", type=float, nargs="+")
    ap.add_argument("--tau-s", type=float, default=d.tau_s)
    ap.add_argument("--rise-s", type=float, default=d.rise_s)
    ap.add_argument("--sigma-px", type=float, default=d.sigma_px)
    ap.add_argument("--photon-scale", type=float, default=d.photon_scale)
    ap.add_argument(
        "--channel",
        type=int,
        default=d.channel,
        help="zero-based channel (default 0)",
    )
    ap.add_argument(
        "--raw-per-photon",
        type=float,
        help="default: measured on each path's first trial",
    )
    ap.add_argument("--analyze-hz", type=float, default=d.analyze_hz)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument(
        "--size-reference",
        choices=("frames", "tissue"),
        default=d.size_reference,
        help="sizes as peaks in analyze_hz frames (default) or in the tissue",
    )
    ap.add_argument("--calibration-s", type=float, default=d.calibration_s)
    ap.add_argument(
        "--no-frame-traces",
        action="store_true",
        help="do not render each source's noiseless frame trace",
    )
    ap.add_argument("--device", default=d.device)
    ap.add_argument(
        "--min-separation-px", type=float, default=pl.min_separation_px
    )
    ap.add_argument("--edge-px", type=float, default=pl.edge_px)
    ap.add_argument("--exclude-px", type=float, default=pl.exclude_px)
    ap.add_argument(
        "--exclude",
        help="HDF5 with datasets pathK: existing sources (n x 2, zero-based "
        "row/col, path frame)",
    )
    ap.add_argument(
        "--positions",
        help="HDF5 with datasets pathK: source positions (path frame) "
        "instead of placing them",
    )
    ap.add_argument(
        "--trial-shifts",
        help="HDF5 with datasets pathK: one (row, col) shift per injected "
        "trial",
    )
    ap.add_argument("--data-dir", help="raw data folder (default: datadr)")
    ap.add_argument(
        "--moco-dir",
        help="registration folder (default: savedr/motion_correction)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Command-line options of the ``python -m`` entry point."""
    ap = argparse.ArgumentParser(
        prog="python -m giant_python.simulation.injection",
        description="Inject synthetic events into a copy of a registered "
        "SLAP2 session's raw data and write their ground truth.",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="command", required=True)
    _add_inject_args(
        sub.add_parser("inject", help="make the injected copy and its truth")
    )
    cmp = sub.add_parser(
        "compare",
        help="after registering OUT_DIR/trial_table.h5: record the copy's "
        "registration (offset, motion difference) in the truth file",
    )
    cmp.add_argument("out_dir", help="an inject output folder")
    return ap


def _settings_from_args(args) -> Tuple[InjectionSettings, PlacementSettings]:
    """Settings of ``inject``."""
    settings = InjectionSettings(
        size_classes=tuple(args.sizes),
        class_weights=(
            tuple(args.class_weights) if args.class_weights else None
        ),
        rate_hz=args.rate_hz,
        min_interval_s=args.min_interval_s,
        edge_s=args.edge_s,
        tau_s=args.tau_s,
        rise_s=args.rise_s,
        sigma_px=args.sigma_px,
        photon_scale=args.photon_scale,
        channel=args.channel,
        raw_per_photon=args.raw_per_photon,
        analyze_hz=args.analyze_hz,
        seed=args.seed,
        size_reference=args.size_reference,
        calibration_s=args.calibration_s,
        frame_traces=not args.no_frame_traces,
        device=args.device,
    )
    placement = PlacementSettings(
        min_separation_px=args.min_separation_px,
        edge_px=args.edge_px,
        exclude_px=args.exclude_px,
    )
    return settings, placement


def _json_default(x):
    """Arrays as lists, anything else as text."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run ``inject`` or ``compare``; print a JSON summary."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(message)s",
    )
    if args.command == "compare":
        summary = record_copy_registration(args.out_dir)
    else:
        settings, placement = _settings_from_args(args)
        summary = inject_session(
            args.trial_table,
            args.out_dir,
            trials=args.trials,
            paths=args.paths,
            n_sources=args.n_sources,
            settings=settings,
            placement=placement,
            positions=_read_per_path(args.positions),
            exclude=_read_per_path(args.exclude),
            trial_shifts=_read_per_path(args.trial_shifts),
            data_dir=args.data_dir,
            moco_dir=args.moco_dir,
        )
    print(json.dumps(summary, indent=1, default=_json_default))
    return 0


if __name__ == "__main__":
    sys.exit(main())
