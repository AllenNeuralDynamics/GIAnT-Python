"""Per-trial F0 images from the F0 network's baseline.

``slap2-f0 predict-registered`` writes one ``<stem>_BASELINE-80Hz.h5`` per
trial-path: the baseline of the trial's registered 80 Hz movie, in movie
counts, ``(frame, MATLAB column, MATLAB row)``, NaN outside the trial's
valid pixels. For the spine-size readout each trial's baseline is averaged
over its kept frames (those not discarded for motion, as SiLo discards them
from the alignment file's ``recNegErr``, and not in the first or last
``trim_s`` seconds) and resampled into a run's consensus frame through that
run's across-trial map (bilinear, NaN outside the image or next to NaN, as
``warp_trial_image``). Trials of different sizes are placed at the top left
of the consensus canvas first, as SiLo places them.

The network gives a value at every registered frame, including those that
SiLo discards (it fills them from neighbouring frames); they are left out
here, as GIAnT-MATLAB's ``F0im`` leaves out ``discard_frames``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from ..extraction.raster.alignment import interp2_linear
from ..extraction.raster.baseline_map import baseline_file
from ..extraction.raster.trial import discard_frames

ALIGNMENT_SUFFIX = "_ALIGNMENTDATA.h5"


def kept_frames(
    rec_neg_err: np.ndarray,
    align_hz: float,
    trim_s: float = 1.0,
    motion_thresh: float = 2.0,
    discard_initial_s: float = 0.0,
) -> np.ndarray:
    """A trial's registered frames that enter its F0 image.

    Frames that SiLo discards for motion (:func:`~giant_python.extraction.
    raster.trial.discard_frames`) are left out, and so are the first and last
    ``round(trim_s * align_hz)`` frames. If nothing would remain, the trim is
    dropped (GIAnT-MATLAB averages a trial too short to trim whole).

    Parameters
    ----------
    rec_neg_err : array_like
        The alignment file's ``recNegErr`` (one value per registered frame).
    align_hz : float
        Frame rate of the registered movie.
    trim_s : float
        Seconds left out at each end of the trial.
    motion_thresh, discard_initial_s : float
        SiLo's motion threshold and initial discard.

    Returns
    -------
    ndarray of bool, shape (frames,)
    """
    rne = np.asarray(rec_neg_err, dtype=np.float64).ravel()
    keep = ~discard_frames(rne, align_hz, motion_thresh, discard_initial_s)
    n_trim = int(math.floor(trim_s * align_hz + 0.5))
    trimmed = keep.copy()
    if n_trim > 0:
        trimmed[:n_trim] = False
        trimmed[max(0, rne.size - n_trim) :] = False
    return trimmed if trimmed.any() else keep


def read_rec_neg_err(alignment_path) -> np.ndarray:
    """``recNegErr`` of an alignment file (one value per registered frame)."""
    with h5py.File(alignment_path, "r") as f:
        return np.asarray(f["recNegErr"][()], dtype=np.float64).ravel()


def baseline_trial_mean(
    path, keep: np.ndarray, chunk: int = 64
) -> tuple[np.ndarray, int]:
    """Mean of a network baseline file over the kept frames.

    The file is read in blocks of ``chunk`` frames, so a trial never has to
    fit in memory as float32.

    Parameters
    ----------
    path : str or Path
        A ``<stem>_BASELINE-80Hz.h5`` file.
    keep : ndarray of bool, shape (frames,)
        Frames to average.
    chunk : int
        Frames read at once.

    Returns
    -------
    image : ndarray of float64, shape (rows, cols)
        Movie counts; NaN where no kept frame has a value.
    n_frames : int
        Number of frames averaged.
    """
    keep = np.asarray(keep, dtype=bool).ravel()
    with h5py.File(path, "r") as f:
        ds = f["baseline"]
        if ds.ndim != 3:
            raise ValueError(f"{path}: /baseline must be (frame, col, row)")
        if ds.shape[0] != keep.size:
            raise ValueError(
                f"{path}: {ds.shape[0]} frames, but {keep.size} frame flags"
            )
        total = np.zeros(ds.shape[1:], np.float64)
        count = np.zeros(ds.shape[1:], np.int64)
        for i0 in range(0, ds.shape[0], chunk):
            k = keep[i0 : i0 + chunk]
            if not k.any():
                continue
            block = np.asarray(ds[i0 : i0 + k.size], dtype=np.float64)[k]
            ok = np.isfinite(block)
            total += np.where(ok, block, 0.0).sum(axis=0)
            count += ok.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return np.ascontiguousarray(mean.T), int(keep.sum())


def to_consensus(
    image: np.ndarray, trial_map, trial: int, shape: tuple[int, int]
) -> np.ndarray:
    """Resample a trial's image into the consensus frame.

    Parameters
    ----------
    image : ndarray of shape (rows, cols)
        The image on the trial's registered frame.
    trial_map : object with ``grid_coords(trial, shape)``
        The run's map from consensus to each trial's registered image
        (:class:`~giant_python.spine.summary_io.TrialMap` or
        :class:`~giant_python.extraction.raster.alignment.TrialWarp`).
    trial : int
        Zero-based trial.
    shape : tuple of int
        ``(rows, cols)`` of the consensus frame.

    Returns
    -------
    ndarray of float64, shape ``shape``
        NaN outside the image or next to NaN.
    """
    canvas = np.full(tuple(shape), np.nan)
    h = min(shape[0], image.shape[0])
    w = min(shape[1], image.shape[1])
    canvas[:h, :w] = image[:h, :w]
    rows, cols = trial_map.grid_coords(trial, tuple(shape))
    return interp2_linear(canvas, rows, cols)


def file_identity(path) -> dict:
    """Name, size and modification time of a file (its identity for
    caching)."""
    st = Path(path).stat()
    return {
        "name": Path(path).name,
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


@dataclass
class F0Stack:
    """Per-trial F0 images of one imaging path, in a run's consensus frame.

    Attributes
    ----------
    images : ndarray of float32, shape (trials, rows, cols)
        NaN for trials without a baseline or a map.
    n_frames : ndarray of int, shape (trials,)
        Frames averaged per trial (0 without an image).
    inputs : list of dict
        Per trial, the identities of the baseline and alignment files (None
        without an image).
    attrs : dict
        Settings and the first baseline file's attributes (checkpoint,
        step).
    """

    images: np.ndarray
    n_frames: np.ndarray
    inputs: list = field(default_factory=list)
    attrs: dict = field(default_factory=dict)


def network_f0_stack(
    baseline_dir,
    moco_dir,
    stems: list,
    trial_map,
    shape: tuple[int, int],
    align_hz: float = 80.0,
    trim_s: float = 1.0,
    motion_thresh: float = 2.0,
    discard_initial_s: float = 0.0,
    progress=None,
) -> F0Stack:
    """Per-trial F0 images of one path from the network's baselines.

    Parameters
    ----------
    baseline_dir : str or Path
        Folder of ``<stem>_BASELINE-80Hz.h5`` files.
    moco_dir : str or Path
        Folder of the ``<stem>_ALIGNMENTDATA.h5`` files the baselines were
        predicted from (their ``recNegErr`` gives the discarded frames).
    stems : list of str or None
        Each trial's stem (``E1T1DMD1``), None for a trial to leave out.
    trial_map : TrialMap or TrialWarp
        The run's map from consensus to each trial's registered image.
    shape : tuple of int
        ``(rows, cols)`` of the consensus frame.
    align_hz, trim_s, motion_thresh, discard_initial_s : float
        See :func:`kept_frames`.
    progress : callable, optional
        Called with ``(trial, n_trials, stem)`` before each trial.

    Returns
    -------
    F0Stack
    """
    n = len(stems)
    images = np.full((n,) + tuple(shape), np.nan, np.float32)
    n_frames = np.zeros(n, np.int64)
    inputs: list = [None] * n
    attrs: dict = {
        "align_hz": float(align_hz),
        "trim_s": float(trim_s),
        "motion_thresh": float(motion_thresh),
        "discard_initial_s": float(discard_initial_s),
    }
    for t, stem in enumerate(stems):
        if progress is not None:
            progress(t, n, stem)
        if stem is None or not bool(trial_map.aligned[t]):
            continue
        bpath = baseline_file(baseline_dir, stem)
        apath = Path(moco_dir) / f"{stem}{ALIGNMENT_SUFFIX}"
        if not bpath.is_file() or not apath.is_file():
            continue
        keep = kept_frames(
            read_rec_neg_err(apath),
            align_hz,
            trim_s,
            motion_thresh,
            discard_initial_s,
        )
        mean, used = baseline_trial_mean(bpath, keep)
        images[t] = to_consensus(mean, trial_map, t, shape)
        n_frames[t] = used
        inputs[t] = {
            "baseline": file_identity(bpath),
            "alignment": file_identity(apath),
        }
        if "checkpoint" not in attrs:
            with h5py.File(bpath, "r") as f:
                for key in ("checkpoint", "step", "precision"):
                    if key in f.attrs:
                        v = f.attrs[key]
                        attrs[key] = v.item() if hasattr(v, "item") else v
    return F0Stack(images, n_frames, inputs, attrs)


def stack_key(*parts) -> str:
    """A short hash of JSON-serializable parts and arrays (cache keys)."""
    h = hashlib.sha1()
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(str(p.shape).encode())
            h.update(np.ascontiguousarray(p).tobytes())
        else:
            h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()[:12]
