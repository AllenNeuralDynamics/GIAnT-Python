"""Session-level maps: the frame-weighted mean of a session's trial maps.

Registered trials do not share a frame: registration aligns each trial to a
template made from its own frames, so the same tissue lies at different
(sub-pixel) positions in different trials. Registration therefore writes
per-trial maps only. SILo's plan stage measures each trial's map from a
consensus frame to its registered frame (``extraction.raster.alignment.
TrialWarp``, a rigid shift by default); given it, :func:`combine_trials`
resamples every trial's maps into the consensus frame (bilinear, NaN next to
NaN, as ``warp_trial_image``) and averages them, each map weighted by the
frames behind it at the pixel:

* single-frame maps and the share of zeros by ``n_valid``;
* slow maps by ``n_blocks``; fast maps by ``n_fast``;
* the gains and ``sigma8_ph`` by ``n_valid``.

Raw moments and their photon-noise expectations are averaged; the
normalised maps are then formed from the averages (``var_rel = var /
var_ph - 1`` and so on), not averaged themselves. Variances stay about each
trial's own mean (differences between trials' means do not enter). The
shared modes (SVD) are per trial and are not combined.
"""

from __future__ import annotations

import numpy as np

WEIGHTS = {
    "n_valid": (
        "mean", "var", "var_q", "var_qa", "var_ph", "zero_share", "zero_ph",
        "gain_g2", "gain_g3", "gain_c2", "gain_c3a", "gain_c3b", "sigma8_ph",
    ),
    "n_blocks": ("slow_var", "slow_q", "slow_qa", "slow_var_ph"),
    "n_fast": ("fast_var", "fast_q", "fast_qa", "fast_var_ph", "fast_m3",
               "fast_q3", "fast_q3a", "fast_q3b", "fast_m3_ph"),
}  # fmt: skip
"""Which frame count weights each averaged map."""


NORMALISED = {
    "var_rel": (("var", "var_ph"), lambda a, b: a / b - 1),
    "slow_rel": (("slow_var", "slow_var_ph"), lambda a, b: a / b - 1),
    "fast_rel": (("fast_var", "fast_var_ph"), lambda a, b: a / b - 1),
    "fast_skew_rel": (
        ("fast_m3", "fast_m3_ph", "fast_var_ph"),
        lambda m3, m3ph, v: (m3 - m3ph) / v**1.5,
    ),
    "zero_excess": (("zero_share", "zero_ph"), lambda a, b: a - b),
}
"""Each normalised map: the averaged maps it is formed from, and how."""


def _normalised(m: dict) -> dict:
    """The normalised maps from averaged moments and expectations (those
    whose inputs are there)."""
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for name, (inputs, fn) in NORMALISED.items():
            if all(k in m for k in inputs):
                out[name] = fn(*(m[k] for k in inputs))
    return out


def _resample(img: np.ndarray, warp, trial: int, shape) -> np.ndarray:
    """A trial's ``(channels, rows, cols)`` map in the consensus frame,
    placed at the top left of the ``shape`` canvas first (NaN elsewhere),
    as SILo places trials of different sizes."""
    from ...extraction.raster.alignment import warp_trial_image

    canvas = np.full((img.shape[0],) + tuple(shape), np.nan)
    canvas[:, : img.shape[1], : img.shape[2]] = img
    return np.moveaxis(
        warp_trial_image(np.moveaxis(canvas, 0, -1), warp, trial), -1, 0
    )


def combine_trials(stats: list, warp=None, shifts=None) -> dict:
    """Session maps of one imaging path.

    Parameters
    ----------
    stats : list of dict
        Each trial's statistics (:func:`.io.read_stats`), in the warp's
        trial order; None for a trial without statistics.
    warp : TrialWarp, optional
        SILo's map from the consensus frame to each trial's registered
        frame (``ExtractionPlan.warp``). Trials it did not align are left
        out.
    shifts : array_like ``(trials, 2)``, optional
        Instead of ``warp``: rigid shifts (consensus pixel ``(r, c)`` is at
        ``(r + dr, c + dc)`` in the trial), NaN for trials left out.

    Raises
    ------
    ValueError
        If the trials' statistics differ in their number of channels.

    Returns
    -------
    dict
        Maps ``(channels, rows, cols)`` in the consensus frame (trials of
        different sizes are placed at the top left of the largest, as SILo
        places them): the frame
        counts summed, the moments and expectations averaged, the
        normalised maps formed from them, and ``n_trials`` (trials with data
        at each pixel).
    """
    from ...extraction.raster.alignment import TrialWarp

    if warp is None:
        warp = TrialWarp.from_shifts(shifts)
    used = [
        t for t, st in enumerate(stats) if st is not None and warp.aligned[t]
    ]
    shape = (0, 0)
    if used:
        shape = tuple(
            max(stats[t]["maps"]["mean"].shape[d] for t in used)
            for d in (1, 2)
        )
        if len({stats[t]["maps"]["mean"].shape[0] for t in used}) > 1:
            raise ValueError("the trials' statistics differ in channels")
    sums, wsum, out = {}, {}, {}
    for t in used:
        maps = stats[t]["maps"]
        for wname, names in WEIGHTS.items():
            w = _resample(np.nan_to_num(maps[wname]), warp, t, shape)
            w = np.nan_to_num(w)
            out[wname] = out.get(wname, 0) + w
            for name in names:
                if name not in maps:
                    continue
                v = _resample(maps[name], warp, t, shape)
                ok = np.isfinite(v) & (w > 0)
                sums[name] = sums.get(name, 0) + np.where(ok, w * v, 0)
                wsum[name] = wsum.get(name, 0) + np.where(ok, w, 0)
        valid = np.isfinite(maps["mean"]).astype(float)
        has = _resample(valid, warp, t, shape)
        out["n_trials"] = out.get("n_trials", 0) + np.nan_to_num(has == 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        for name in sums:
            out[name] = np.where(
                wsum[name] > 0, sums[name] / wsum[name], np.nan
            )
    if sums:
        out.update(_normalised(out))
    return out
