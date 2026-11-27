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
        "mean", "var", "var_q", "var_ph", "zero_share", "zero_ph",
        "gain_g2", "gain_g3", "sigma8_ph",
    ),
    "n_blocks": ("slow_var", "slow_q", "slow_var_ph"),
    "n_fast": ("fast_var", "fast_q", "fast_var_ph", "fast_m3", "fast_q3",
               "fast_m3_ph"),
}  # fmt: skip
"""Which frame count weights each averaged map."""


def _normalised(m: dict) -> dict:
    """The normalised maps from averaged moments and expectations."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "var_rel": m["var"] / m["var_ph"] - 1,
            "slow_rel": m["slow_var"] / m["slow_var_ph"] - 1,
            "fast_rel": m["fast_var"] / m["fast_var_ph"] - 1,
            "fast_skew_rel": (m["fast_m3"] - m["fast_m3_ph"])
            / m["fast_var_ph"] ** 1.5,
            "zero_excess": m["zero_share"] - m["zero_ph"],
        }


def _resample(img: np.ndarray, warp, trial: int) -> np.ndarray:
    """A trial's ``(channels, rows, cols)`` map in the consensus frame."""
    from ...extraction.raster.alignment import warp_trial_image

    return np.moveaxis(
        warp_trial_image(np.moveaxis(img, 0, -1), warp, trial), -1, 0
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

    Returns
    -------
    dict
        Maps ``(channels, rows, cols)`` in the consensus frame: the frame
        counts summed, the moments and expectations averaged, the
        normalised maps formed from them, and ``n_trials`` (trials with data
        at each pixel).
    """
    from ...extraction.raster.alignment import TrialWarp

    if warp is None:
        warp = TrialWarp.from_shifts(shifts)
    sums, wsum, out = {}, {}, {}
    for t, st in enumerate(stats):
        if st is None or not warp.aligned[t]:
            continue
        maps = st["maps"]
        for wname, names in WEIGHTS.items():
            w = _resample(np.nan_to_num(maps[wname]), warp, t)
            w = np.nan_to_num(w)
            out[wname] = out.get(wname, 0) + w
            for name in names:
                v = _resample(maps[name], warp, t)
                ok = np.isfinite(v) & (w > 0)
                sums[name] = sums.get(name, 0) + np.where(ok, w * v, 0)
                wsum[name] = wsum.get(name, 0) + np.where(ok, w, 0)
        has = _resample(np.isfinite(maps["mean"]).astype(float), warp, t)
        out["n_trials"] = out.get("n_trials", 0) + np.nan_to_num(has == 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        for name in sums:
            out[name] = np.where(
                wsum[name] > 0, sums[name] / wsum[name], np.nan
            )
    if sums:
        out.update(_normalised(out))
    return out
