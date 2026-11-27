"""Per-trial summary statistics of a registered SLAP2 movie.

Computed from the registered movie alone, relative to photon noise, for the
F0 network's next version (``slap2-F0-network``, plan
``slap2-F0-network_next-version-plan.md``): single-frame moments and the
share of zeros, the variance of 64-frame block means (slow), the second
and third moments of the frame minus its 65-frame running mean (fast), and
the top three shared modes of the binned movie (SVD). Every moment comes
with its expectation under photon noise alone, derived from the raw
samples' gain and from frame formation and registration, so that the
normalised maps read 0 for photon noise at any brightness.

Registration computes them while it writes the movie
(:mod:`..multiroi`); ``giant reg-stats`` computes the same maps for movies
registered earlier, MATLAB's included. See ``docs/source/reg_stats.rst``.
"""

from __future__ import annotations

import logging

from .accumulate import TrialStatsAccumulator
from .gain import GainCurve, measure_gain_curves, validity_mismatch
from .geometry import FrameGeometry
from .io import (
    STATS_VERSION,
    alignment_digest,
    read_stats,
    stats_path,
    write_stats,
)
from .session import combine_trials

log = logging.getLogger(__name__)

__all__ = [
    "STATS_VERSION",
    "FrameGeometry",
    "GainCurve",
    "TrialStatsAccumulator",
    "alignment_digest",
    "combine_trials",
    "measure_gain_curves",
    "new_accumulator",
    "read_stats",
    "stats_path",
    "write_stats",
]


def new_accumulator(source, ds, margin: int, n_channels: int):
    """The statistics accumulator of a trial.

    Parameters
    ----------
    source : PlainFrames or CompensatedFrames
        The trial's frame source; its former's samples give the gains.
    ds : array_like of int
        Frame times of the registered frames.
    margin : int
        ``maxshift`` (padding of the registered grid).
    n_channels : int
        Channels.

    Returns
    -------
    TrialStatsAccumulator
    """
    from .gain import next_slots

    lay = source.former.layout
    samples = source.former._samples
    gains = measure_gain_curves(
        samples, next_slots(lay.slot_line, lay.slot_pos)
    )
    differ = validity_mismatch(samples)
    for ch, n in enumerate(differ):
        if n:
            log.warning(
                "channel %d: %d samples whose validity differs from channel "
                "1's; its photon-noise expectations assume channel 1's",
                ch + 1,
                n,
            )
    geometry = FrameGeometry(source, ds, margin)
    acc = TrialStatsAccumulator(geometry, gains, n_channels)
    acc.validity_differs = differ
    return acc
