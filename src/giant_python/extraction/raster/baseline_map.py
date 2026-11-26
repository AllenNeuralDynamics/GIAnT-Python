"""The F0 network's baseline on the data of source extraction.

``slap2-f0 predict-registered`` writes one ``<trial>_BASELINE-80Hz.h5`` per
trial-path: the baseline of the trial's registered 80 Hz movie, in movie
counts, on the trial's registered frame. Source extraction works at 200 Hz,
at the selected pixels of the path's consensus frame, in counts /
``photon_scale``
(:class:`~giant_python.extraction.raster.trial_data.TrialData`).
:func:`map_baseline` carries the one to the other:

1. **Space.** Each selected pixel is sampled at its position in the trial's
   registered frame (``TrialData.trial_rc``: the across-trial map that the
   data were read through), bilinearly on the registered frame's pixel
   grid (pixel centres at whole numbers, zero-based).
2. **Time.** Linear interpolation in line number, from the lines of the
   registered frames (``DSframes`` of the alignment file) to the lines of
   the extraction's frames (``TrialData.frame_lines``). Extraction frames
   before the first or after the last registered frame take that frame's
   value.
3. **Units.** Division by ``photon_scale``.

Where the network gives no value:

- **Pixels.** The network's baseline is NaN outside the trial's valid
  pixels (those imaged in at least ``1 - nanThresh`` of the frames). A
  selected pixel is sampled from those of its four neighbours that have a
  value, the bilinear weights renormalized, if they carry at least
  ``min_weight`` of the weight (a neighbour outside the registered frame
  has none). Otherwise the pixel has no network baseline: it is flagged in
  :attr:`MappedBaseline.missing`, and gets the data's own starting baseline
  (``splitFreq``, as MATLAB's solver starts from) when the data are given,
  or NaN.
- **Frames.** The network gives a value at every registered frame,
  including those that SiLo discards for motion (its input is NaN there,
  so the value comes from the neighbouring frames). Those values are kept
  unless ``drop_frames`` names the frames. A registered frame at which a
  pixel has no value (NaN, or dropped) is left out of that pixel's time
  interpolation. The extraction's discarded frames hold filled-in data
  with a large variance multiplier, so their baseline barely enters the
  fits.

What the mapping cannot carry (measured by the WP6 test of the SILo port):
brightness changes shared across pixels that are faster than the 80 Hz
frames resolve, the slight blur of interpolating the registered movie once
more than the data, and the network's own errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .solver_baseline import initial_baseline
from .solver_setup import SolverParams

BASELINE_SUFFIX = "_BASELINE-80Hz.h5"
_REGISTERED = "_REGISTERED"


def baseline_file(baseline_dir, movie_or_stem: str) -> Path:
    """The network baseline file of a trial.

    Parameters
    ----------
    baseline_dir : str or Path
        Folder of ``slap2-f0 predict-registered``'s outputs.
    movie_or_stem : str
        The registered movie's file name
        (``<stem>_REGISTERED_DOWNSAMPLED-80Hz.h5``) or the trial's stem
        (``E1T1DMD1``).
    """
    name = Path(movie_or_stem).name
    stem = name[: name.index(_REGISTERED)] if _REGISTERED in name else name
    return Path(baseline_dir) / f"{stem}{BASELINE_SUFFIX}"


def read_network_baseline(path) -> tuple[np.ndarray, dict]:
    """A network baseline file as ``(frames, rows, cols)`` float32 counts.

    ``slap2-f0`` writes ``/baseline`` as (frame, MATLAB column, MATLAB
    row), so that MATLAB's ``h5read`` gives SiLo's rows x cols x frames;
    here it is transposed to (frame, row, col), the orientation of
    :func:`~giant_python.extraction.raster.trial.load_registered_trial`'s
    movie. NaN outside the trial's valid pixels.

    Returns
    -------
    values : ndarray of float32, shape (frames, rows, cols)
    attrs : dict
        The file's attributes (checkpoint, gain, ...).
    """
    with h5py.File(path, "r") as f:
        values = np.asarray(f["baseline"][()], dtype=np.float32)
        attrs = {k: f.attrs[k] for k in f.attrs}
    if values.ndim != 3:
        raise ValueError(
            f"{path}: /baseline must be (frame, col, row), not "
            f"{values.shape}"
        )
    return np.ascontiguousarray(values.transpose(0, 2, 1)), attrs


@dataclass
class MappedBaseline:
    """The network's baseline at a trial's selected pixels and frames.

    Attributes
    ----------
    values : ndarray of float64, shape (pixels, frames)
        Baseline in counts / ``photon_scale`` (the data's units), at the
        extraction's frames. Pixels without a network baseline hold the
        fallback (or NaN without one).
    missing : ndarray of bool, shape (pixels,)
        Pixels without a network baseline.
    weight : ndarray of float64, shape (pixels,)
        Share of each pixel's bilinear weight that fell on neighbours with
        a value (the smallest over frames).
    """

    values: np.ndarray
    missing: np.ndarray
    weight: np.ndarray


def sample_bilinear(
    stack: np.ndarray,
    rc: np.ndarray,
    min_weight: float = 0.5,
    block: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a ``(frames, rows, cols)`` stack at sub-pixel positions.

    Bilinear on the stack's pixel grid; per frame, the neighbours without
    a value (NaN, or outside the stack) are left out and the others'
    weights renormalized. A sample whose neighbours with a value carry
    less than ``min_weight`` of the weight is NaN.

    Parameters
    ----------
    stack : ndarray of shape (frames, rows, cols)
    rc : ndarray of shape (n, 2)
        Zero-based (row, col) positions.
    min_weight : float
        Least share of the bilinear weight that must have a value.
    block : int
        Positions sampled at a time (memory).

    Returns
    -------
    values : ndarray of float64, shape (n, frames)
    weight : ndarray of float64, shape (n,)
        Share of the weight with a value, smallest over frames.
    """
    if not 0 < min_weight <= 1:
        raise ValueError("min_weight must lie in (0, 1]")
    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError("stack must be (frames, rows, cols)")
    rc = np.asarray(rc, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(rc).all():
        raise ValueError("positions must be finite")
    n_t, n_r, n_c = stack.shape
    n = rc.shape[0]
    out = np.full((n, n_t), np.nan)
    weight = np.zeros(n)
    r0 = np.floor(rc[:, 0]).astype(np.int64)
    c0 = np.floor(rc[:, 1]).astype(np.int64)
    fr = rc[:, 0] - r0
    fc = rc[:, 1] - c0
    corners = (
        (0, 0, (1 - fr) * (1 - fc)),
        (1, 0, fr * (1 - fc)),
        (0, 1, (1 - fr) * fc),
        (1, 1, fr * fc),
    )
    flat = stack.reshape(n_t, -1)
    for a in range(0, n, block):
        b = min(a + block, n)
        num = np.zeros((b - a, n_t))
        den = np.zeros((b - a, n_t))
        for dr, dc, w in corners:
            r, c, w = r0[a:b] + dr, c0[a:b] + dc, w[a:b]
            inside = (r >= 0) & (r < n_r) & (c >= 0) & (c < n_c) & (w > 0)
            if not inside.any():
                continue
            k = np.flatnonzero(inside)
            v = flat[:, r[k] * n_c + c[k]].T.astype(np.float64)
            ok = np.isfinite(v)
            num[k] += np.where(ok, v, 0.0) * w[k, None]
            den[k] += ok * w[k, None]
        good = den >= min_weight - 1e-12
        with np.errstate(invalid="ignore", divide="ignore"):
            out[a:b] = np.where(good, num / den, np.nan)
        weight[a:b] = den.min(1) if n_t else 0.0
    return out, weight


def interp_frames(
    values: np.ndarray, src_lines: np.ndarray, dst_lines: np.ndarray
) -> np.ndarray:
    """Linear interpolation in line number, per row of ``values``.

    NaN samples are left out of their row's interpolation; destinations
    beyond the first or last source line with a value take that value. A
    row without any value stays NaN.

    Parameters
    ----------
    values : ndarray of shape (n, sources)
    src_lines : ndarray of shape (sources,)
        Increasing lines of the source frames.
    dst_lines : ndarray of shape (frames,)

    Returns
    -------
    ndarray of float64, shape (n, frames)
    """
    values = np.asarray(values, dtype=np.float64)
    src = np.asarray(src_lines, dtype=np.float64).ravel()
    dst = np.asarray(dst_lines, dtype=np.float64).ravel()
    if values.ndim != 2 or values.shape[1] != src.size:
        raise ValueError(
            f"values are {values.shape}; {src.size} source lines need "
            "(n, sources)"
        )
    if src.size == 0 or np.any(np.diff(src) <= 0):
        raise ValueError("source lines must be increasing and not empty")
    out = np.full((values.shape[0], dst.size), np.nan)
    ok = np.isfinite(values)
    full = ok.all(1)
    if full.any():
        if src.size == 1:
            out[full] = values[full, :1]
        else:
            j = np.clip(np.searchsorted(src, dst, side="right") - 1, 0, None)
            j = np.minimum(j, src.size - 2)
            f = np.clip((dst - src[j]) / (src[j + 1] - src[j]), 0.0, 1.0)
            v = values[full]
            out[full] = v[:, j] * (1 - f) + v[:, j + 1] * f
    for i in np.flatnonzero(~full & ok.any(1)):
        out[i] = np.interp(dst, src[ok[i]], values[i, ok[i]])
    return out


def map_baseline(
    baseline: np.ndarray,
    trial_rc: np.ndarray,
    ds_lines: np.ndarray,
    frame_lines: np.ndarray,
    photon_scale: float,
    *,
    min_weight: float = 0.5,
    drop_frames: np.ndarray | None = None,
    fallback_data: np.ndarray | None = None,
    solver_params=None,
) -> MappedBaseline:
    """The network's baseline at a trial's selected pixels and frames.

    Parameters
    ----------
    baseline : ndarray of shape (registered frames, rows, cols)
        The network's baseline in movie counts
        (:func:`read_network_baseline`).
    trial_rc : ndarray of shape (pixels, 2)
        Zero-based positions of the selected pixels in the trial's
        registered frame (``TrialData.trial_rc``).
    ds_lines : ndarray of shape (registered frames,)
        One-based line of each registered frame (``DSframes``,
        ``TrialAlignment.ds_frames``).
    frame_lines : ndarray of shape (frames,)
        One-based line of each extraction frame (``TrialData.frame_lines``).
    photon_scale : float
        Counts per unit of the data (``TrialData.photon_scale``).
    min_weight : float
        Least share of a pixel's bilinear weight that must fall on pixels
        with a network value (see :func:`sample_bilinear`).
    drop_frames : ndarray of bool, shape (registered frames,), optional
        Registered frames whose network values are not used (they are
        interpolated over instead).
    fallback_data : ndarray of shape (pixels, frames), optional
        The activity channel's data (``TrialData.y[0]``). Pixels without a
        network baseline get its starting baseline
        (:func:`~giant_python.extraction.raster.solver_baseline.initial_baseline`).
    solver_params : SolverParams, optional
        Parameters of that starting baseline (default: SILo's).

    Returns
    -------
    MappedBaseline
    """
    baseline = np.asarray(baseline)
    ds_lines = np.asarray(ds_lines, dtype=np.float64).ravel()
    if baseline.ndim != 3 or baseline.shape[0] != ds_lines.size:
        raise ValueError(
            f"the baseline is {baseline.shape}; the registration has "
            f"{ds_lines.size} frames"
        )
    if not (photon_scale > 0 and np.isfinite(photon_scale)):
        raise ValueError("photon_scale must be positive and finite")
    trial_rc = np.asarray(trial_rc, dtype=np.float64).reshape(-1, 2)
    frame_lines = np.asarray(frame_lines, dtype=np.float64).ravel()
    keep = np.ones(ds_lines.size, dtype=bool)
    if drop_frames is not None:
        drop = np.asarray(drop_frames, dtype=bool).ravel()
        if drop.size != ds_lines.size:
            raise ValueError(
                f"drop_frames has {drop.size} frames; the registration has "
                f"{ds_lines.size}"
            )
        keep = ~drop
    if not keep.any():
        raise ValueError("every registered frame is dropped")
    if keep.all():  # no copy of the whole movie
        sampled, weight = sample_bilinear(baseline, trial_rc, min_weight)
    else:
        sampled, weight = sample_bilinear(baseline[keep], trial_rc, min_weight)
    values = interp_frames(sampled, ds_lines[keep], frame_lines)
    values /= photon_scale
    missing = ~np.isfinite(values).any(1)
    if fallback_data is not None and missing.any():
        data = np.asarray(fallback_data, dtype=np.float64)
        if data.shape != values.shape:
            raise ValueError(
                f"fallback_data is {data.shape}; the baseline is "
                f"{values.shape}"
            )
        params = solver_params or SolverParams()
        values[missing] = initial_baseline(data[missing], params)
    return MappedBaseline(values=values, missing=missing, weight=weight)
