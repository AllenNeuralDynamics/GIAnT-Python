"""Load and localize one registered SLAP2 trial as SiLo does.

Port of GIAnT-MATLAB's ``loadAndProcessTrialAsync.m``: the registered 80 Hz
movie (``*_REGISTERED_DOWNSAMPLED-80Hz.tif``, channels interleaved by frame)
and its ``*_ALIGNMENTDATA.h5`` give the activity channel, NaN at the frames
discarded for motion (from ``recNegErr``) and at the first
``discardInitial_s``, the variance factor ``slap2.varFacDS`` and the trial's
mean image; :func:`localize_trial` then runs
:func:`~giant_python.extraction.raster.localization.localize_sources`, with
the F0 network's ``<trial>_BASELINE-80Hz.h5`` when a baseline folder is given
(GIAnT-MATLAB branch ``localization-network-baseline``).

Arrays are ``(frames, rows, cols)`` in MATLAB's row and column orientation
(the TIFF pages are transposed, as MATLAB's reader returns them).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import h5py
import numpy as np

from .localization import (
    RasterLocalization,
    RasterLocalizationParams,
    check_inputs,
    localize_sources,
    matlab_prctile,
)


def _medfilt1_zeropad(x: np.ndarray, n: int) -> np.ndarray:
    """MATLAB ``medfilt1(x, n)``: zero padding; for even n the median of
    ``x(k - n/2 : k + n/2 - 1)``."""
    lo = n // 2 if n % 2 == 0 else (n - 1) // 2
    hi = n - 1 - lo
    xp = np.concatenate([np.zeros(lo), x, np.zeros(hi)])
    return np.median(np.lib.stride_tricks.sliding_window_view(xp, n), axis=1)


def _dilate(b: np.ndarray, w: int) -> np.ndarray:
    """1-D binary dilation with a window of w (outside counts as false)."""
    h = w // 2
    p = np.concatenate([np.zeros(h, bool), b, np.zeros(h, bool)])
    return np.lib.stride_tricks.sliding_window_view(p, w).any(1)


def _erode(b: np.ndarray, w: int) -> np.ndarray:
    """1-D binary erosion with a window of w (outside counts as true)."""
    h = w // 2
    p = np.concatenate([np.ones(h, bool), b, np.ones(h, bool)])
    return np.lib.stride_tricks.sliding_window_view(p, w).all(1)


def discard_frames(
    rec_neg_err: np.ndarray,
    align_hz: float,
    motion_thresh: float = 2.0,
    discard_initial_s: float = 0.0,
) -> np.ndarray:
    """SiLo's motion discards from the alignment's ``recNegErr``.

    ``recNegErr`` minus its 4 s median, normalized by the larger of its 5th
    percentile's magnitude and 0.005, above ``motion_thresh`` (dilated by
    about 50 ms, kept where above half the threshold, then closed); plus the
    first ``discard_initial_s``.
    """
    rne = np.asarray(rec_neg_err, dtype=np.float64).ravel()
    tmp = rne - _medfilt1_zeropad(rne, int(math.floor(4 * align_hz + 0.5)))
    tmp = -tmp / min(-0.005, matlab_prctile(tmp, 5))
    window = 2 * math.ceil(0.025 * align_hz) + 1
    over = _dilate(tmp > motion_thresh, window) & (tmp > motion_thresh / 2)
    d = _erode(_dilate(over, window), window)
    d[: math.ceil(discard_initial_s * align_hz)] = True
    return d


@dataclass
class RegisteredTrial:
    """One trial prepared for localization.

    Attributes
    ----------
    movie : ndarray of shape (frames, rows, cols)
        The activity channel, NaN at discarded frames.
    var_factor : ndarray of the same shape, or None
        ``slap2.varFacDS`` (None for raster imaging).
    mean_im : ndarray of shape (rows, cols, channels)
        Mean over all frames, NaN where more than ``nan_thresh`` is missing.
    discard : ndarray of bool
        Discarded frames.
    align_hz : float
        Frame rate of the registered movie.
    """

    movie: np.ndarray
    var_factor: np.ndarray | None
    mean_im: np.ndarray
    discard: np.ndarray
    align_hz: float


def _check_channels(num_channels, activity_channel) -> None:
    """Reject channel arguments that would deinterleave or select wrongly."""
    for name, v in (
        ("num_channels", num_channels),
        ("activity_channel", activity_channel),
    ):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise ValueError(f"{name} must be an integer, not {v!r}")
    if num_channels < 1:
        raise ValueError(
            f"num_channels must be at least 1, not {num_channels}"
        )
    if not 1 <= activity_channel <= num_channels:
        raise ValueError(
            f"activity_channel {activity_channel} is not one of the "
            f"{num_channels} channels (1-based)"
        )


def _read_tiff_pages(path) -> np.ndarray:
    """All pages of a registered TIFF as one ``(pages, rows, cols)`` array.

    MATLAB's ``Fast_BigTiff_Write`` gives one series of all pages, but
    GIAnT-Python's registration writes one page per ``TiffWriter.write``, a
    series each; ``tifffile.imread`` would return only the first.
    """
    import tifffile

    with tifffile.TiffFile(path) as tf:
        if len(tf.series) == 1 and len(tf.series[0].shape) == 3:
            return tf.series[0].asarray()
        pages = tf.pages
        first = pages[0].asarray()
        out = np.empty((len(pages),) + first.shape, first.dtype)
        out[0] = first
        for k in range(1, len(pages)):
            out[k] = pages[k].asarray()
        return out


def load_registered_trial(
    moco_dir: str | Path,
    fn: str,
    num_channels: int,
    activity_channel: int = 1,
    motion_thresh: float = 2.0,
    discard_initial_s: float = 0.0,
    nan_thresh: float = 0.33,
) -> RegisteredTrial:
    """Read a registered trial as ``loadAndProcessTrialAsync`` does.

    Parameters
    ----------
    moco_dir : str or Path
        GIAnT's motion-correction folder.
    fn : str
        The registered movie's file name (``..._REGISTERED_DOWNSAMPLED-80Hz``
        ``.tif`` or ``.h5``).
    num_channels : int
        Channels interleaved in the movie.
    activity_channel : int
        1-based channel of the activity signal (``activityChannel``).
    motion_thresh, discard_initial_s, nan_thresh : float
        SiLo's ``motionThresh``, ``discardInitial_s`` and ``nanThresh``.

    Returns
    -------
    RegisteredTrial

    Raises
    ------
    ValueError
        If ``num_channels`` is not a positive integer or ``activity_channel``
        not one of its 1-based channels (0 would select the last one).

    Warns
    -----
    UserWarning
        If the page count is not a multiple of ``num_channels`` (the pages
        after the last full frame are left out).
    """
    _check_channels(num_channels, activity_channel)
    moco_dir = Path(moco_dir)
    path = moco_dir / fn
    if path.suffix == ".h5":
        with h5py.File(path, "r") as f:
            name = next(iter(f.keys()))
            raw = f[name][
                ()
            ]  # (pages, cols, rows): MATLAB's rows x cols x pages
    else:
        raw = _read_tiff_pages(path)  # (pages, MATLAB col, MATLAB row)
    raw = np.asarray(raw, dtype=np.float32).transpose(0, 2, 1)
    n_frames = raw.shape[0] // num_channels
    if raw.shape[0] != n_frames * num_channels:
        warnings.warn(
            f"{fn}: {raw.shape[0]} pages are not a multiple of "
            f"{num_channels} channels; the last "
            f"{raw.shape[0] - n_frames * num_channels} are left out"
        )
    stack = raw[: n_frames * num_channels].reshape(
        n_frames, num_channels, raw.shape[1], raw.shape[2]
    )
    with np.errstate(invalid="ignore"):
        mean_im = np.nanmean(stack, axis=0).transpose(1, 2, 0)
    missing = np.isnan(stack).mean(axis=(0, 1)) > nan_thresh
    mean_im[missing] = np.nan
    movie = np.ascontiguousarray(stack[:, activity_channel - 1])
    del stack, raw

    stem = fn[: fn.index("_REGISTERED")]
    with h5py.File(moco_dir / f"{stem}_ALIGNMENTDATA.h5", "r") as f:
        align_hz = float(np.asarray(f["alignHz"]).ravel()[0])
        rne = np.asarray(f["recNegErr"]).ravel()
        vf = None
        if "slap2" in f and "varFacDS" in f["slap2"]:
            # HDF5 holds MATLAB's rows x cols x frames as (frames, cols, rows)
            vf = np.asarray(f["slap2/varFacDS"][()], np.float32)
            vf = np.ascontiguousarray(vf.transpose(0, 2, 1))
    discard = discard_frames(
        rne[:n_frames], align_hz, motion_thresh, discard_initial_s
    )
    movie[discard] = np.nan
    return RegisteredTrial(movie, vf, mean_im, discard, align_hz)


def localize_trial(
    moco_dir: str | Path,
    fn: str,
    num_channels: int,
    params: RasterLocalizationParams | None = None,
    baseline_dir: str | Path | None = None,
    activity_channel: int = 1,
    motion_thresh: float = 2.0,
    discard_initial_s: float = 0.0,
) -> tuple[RasterLocalization, RegisteredTrial]:
    """Load one registered trial and localize it (SiLo's per-trial step).

    With ``baseline_dir``, ``<trial>_BASELINE-80Hz.h5`` (``slap2-f0
    predict-registered``) replaces the moving median. As in MATLAB, a failed
    localization is retried with the first and last rows and columns set to
    NaN (invalid inputs, such as a wrong baseline size, are not retried).
    """
    p = params or RasterLocalizationParams()
    trial = load_registered_trial(
        moco_dir,
        fn,
        num_channels,
        activity_channel,
        motion_thresh,
        discard_initial_s,
        p.nan_thresh,
    )
    p = replace(p, align_hz=trial.align_hz)
    baseline = None
    if baseline_dir:
        stem = fn[: fn.index("_REGISTERED")]
        with h5py.File(Path(baseline_dir) / f"{stem}_BASELINE-80Hz.h5") as f:
            # (frame, MATLAB col, MATLAB row) as written by slap2-f0
            baseline = np.ascontiguousarray(
                np.asarray(f["baseline"][()], np.float32).transpose(0, 2, 1)
            )
    check_inputs(trial.movie.shape, p, baseline, trial.var_factor)
    try:
        loc = localize_sources(trial.movie, trial.var_factor, p, baseline)
    except Exception:  # MATLAB's catch: retry without the outer pixels
        movie = trial.movie.copy()
        movie[:, [0, -1], :] = np.nan
        movie[:, :, [0, -1]] = np.nan
        loc = localize_sources(movie, trial.var_factor, p, baseline)
    return loc, trial
