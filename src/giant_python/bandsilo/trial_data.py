"""Read low-resolution superpixel activity for band-scan trials.

Ported from ``extractSLAP2IntegrationSources.py`` (``get_trial_data`` ref
715-842, ``nearest_interp``/``fast_dilation`` ref 166-205, and the per-trial
low-res caching block ref 1851-1892). The SLAP2 file open (``slap2_utils``)
and the OpenCV dilation fallback import their heavy dependencies lazily so
importing this module stays cheap. The accumulation core, downsample-frame
grid, interpolation, fast dilation, and the low-res assembly/cache IO are all
testable without ``slap2_utils``.
"""

from __future__ import annotations

import os
import re
from typing import Optional

import numpy as np

from .hdf5 import load_alignment_data_h5


def nearest_interp(
    x: np.ndarray, xp: np.ndarray, yp: np.ndarray
) -> np.ndarray:
    """Nearest-neighbor interpolation (MATLAB ``interp1(...,'nearest')``).

    Parameters
    ----------
    x : ndarray
        Query points.
    xp : ndarray
        Sample positions (assumed sorted ascending).
    yp : ndarray
        Sample values aligned with ``xp``.

    Returns
    -------
    ndarray
        ``yp`` sampled at the nearest ``xp`` for each ``x``.
    """
    if len(xp) == 1:
        return yp
    x_bds = xp[:-1] / 2.0 + xp[1:] / 2.0
    idx = np.searchsorted(x_bds, x, side="left")
    idx = np.clip(idx, 0, len(xp) - 1)
    return yp[idx]


def fast_dilation(
    mask: np.ndarray,
    kernel: Optional[np.ndarray] = None,
    iterations: int = 1,
) -> np.ndarray:
    """Binary-dilate the trailing two axes of ``mask``.

    The common 3x3 / single-iteration case is handled with a pure-numpy shift
    accumulation (fast, no OpenCV); other cases fall back to per-slice
    ``cv2.dilate`` over all leading axes.

    Parameters
    ----------
    mask : ndarray
        Boolean (or 0/1) array; the last two axes are the image plane.
    kernel : ndarray, optional
        Structuring element; defaults to a 7x7 block.
    iterations : int
        Number of dilation iterations (generic path only).

    Returns
    -------
    ndarray of bool
        The dilated mask.
    """
    if kernel is None:
        kernel = np.ones((7, 7), np.uint8)

    if iterations == 1 and kernel.shape == (3, 3) and np.all(kernel):
        m = mask.astype(bool, copy=False)
        pad = [(0, 0)] * (m.ndim - 2) + [(1, 1), (1, 1)]
        p = np.pad(m, pad, mode="constant", constant_values=False)
        out = np.zeros_like(m, dtype=bool)
        h, w = m.shape[-2], m.shape[-1]
        for dr in range(3):
            r_end = dr + h
            for dc in range(3):
                c_end = dc + w
                out |= p[..., dr:r_end, dc:c_end]
        return out

    import cv2

    out = np.empty_like(mask, dtype=bool)
    for idx in np.ndindex(mask.shape[:-2]):
        out[idx] = cv2.dilate(
            mask[idx].astype(np.uint8, copy=False),
            kernel,
            iterations=iterations,
        ).astype(bool, copy=False)
    return out


def compute_ds_frames(
    first_line: float, last_line: float, dt: float
) -> np.ndarray:
    """Return the downsampled-frame line grid for one trial.

    Parameters
    ----------
    first_line, last_line : float
        First/last acquisition line (1-based, inclusive).
    dt : float
        Downsample step, in lines (``1 / samp_freq / line_period_s``).

    Returns
    -------
    ndarray
        ``ceil`` of the line positions of each downsampled frame.
    """
    return np.ceil(np.arange(first_line, last_line + 1, dt))


def _build_time_windows(
    ds_frames: np.ndarray, dt: float, lines_per_cycle: int, num_cycles: int
):
    """Return per-frame time windows and their line/cycle index arrays."""
    dt_read = max(3 * dt, lines_per_cycle)
    max_line = num_cycles * lines_per_cycle
    time_windows = [
        np.arange(
            max(1, np.floor(f - dt_read)),
            min(np.ceil(f + dt_read), max_line) + 1,
        )
        for f in ds_frames
    ]
    line_indices = [(tw - 1) % lines_per_cycle + 1 for tw in time_windows]
    cycle_indices = [
        np.floor((tw - 1) / lines_per_cycle) + 1 for tw in time_windows
    ]
    return time_windows, line_indices, cycle_indices


def _build_line_cycle_cache(line_indices: list, cycle_indices: list):
    """Return unique (line, cycle) arrays and their ``(line, cycle)`` map."""
    pairs = set()
    for lines, cycles in zip(line_indices, cycle_indices):
        for li, ci in zip(lines, cycles):
            pairs.add((int(li), int(ci)))
    all_lines = np.array([p[0] for p in pairs])
    all_cycles = np.array([p[1] for p in pairs])
    mapping = {
        (int(li), int(ci)): i
        for i, (li, ci) in enumerate(zip(all_lines, all_cycles))
    }
    return all_lines, all_cycles, mapping


def _accumulate_line(
    acc: dict,
    line_data: np.ndarray,
    positions: np.ndarray,
    z_idx: np.ndarray,
    super_pixel_ids: np.ndarray,
    weight: float,
    ds_ix: int,
) -> None:
    """Accumulate one line's weighted superpixel data into ``acc`` in place."""
    lookup_values = positions * 100 + z_idx
    matching_mask = np.isin(super_pixel_ids, lookup_values)
    matching_indices = np.where(matching_mask)[0]
    if len(matching_indices) == 0:
        return
    value_to_pos = dict(
        zip(lookup_values.astype(np.uint32), range(len(lookup_values)))
    )
    matched = [
        value_to_pos[int(super_pixel_ids[idx])] for idx in matching_indices
    ]
    acc["data"][matching_indices, ds_ix] += line_data[matched, 0] * weight
    acc["data_count"][matching_indices, ds_ix] += weight
    if acc["data2"] is not None:
        acc["data2"][matching_indices, ds_ix] += line_data[matched, 1] * weight
        acc["data_count2"][matching_indices, ds_ix] += weight


def accumulate_superpixel_data(
    data_file,
    ds_frames: np.ndarray,
    dt: float,
    super_pixel_ids: np.ndarray,
    num_channels: int,
    all_channels: bool,
) -> dict:
    """Accumulate weighted per-superpixel activity over the DS-frame grid.

    Port of the core loop of ``get_trial_data``: for each downsampled frame,
    weight nearby acquisition lines by ``exp(-|dframe|/dt)`` and scatter their
    superpixel line data into the ``(n_superpixels, n_ds_frames)`` accumulator.

    Parameters
    ----------
    data_file : object
        SLAP2 data-file-like object exposing ``header['linesPerCycle']``,
        ``numCycles``, ``getLineData``, ``lineDataNumElements``,
        ``lineSuperPixelIDs``, and ``lineFastZIdxs``.
    ds_frames : ndarray
        Downsampled-frame line grid from :func:`compute_ds_frames`.
    dt : float
        Downsample step, in lines.
    super_pixel_ids : ndarray
        Superpixel id lookup for this DMD, shape ``(n_superpixels, 1)``.
    num_channels : int
        Number of acquisition channels.
    all_channels : bool
        Read all channels (``getLineData(..., None)``) vs channel 1 only.

    Returns
    -------
    dict
        ``{"data", "data_count", "data2", "data_count2"}``. The second-channel
        entries are ``None`` unless ``all_channels`` and ``num_channels >= 2``.
    """
    lines_per_cycle = data_file.header["linesPerCycle"]
    n_ds = len(ds_frames)
    num_sp = super_pixel_ids.shape[0]
    sp_flat = np.asarray(super_pixel_ids).reshape(-1)
    read_second = all_channels and num_channels >= 2

    time_windows, line_indices, cycle_indices = _build_time_windows(
        ds_frames, dt, lines_per_cycle, data_file.numCycles
    )
    all_lines, all_cycles, mapping = _build_line_cycle_cache(
        line_indices, cycle_indices
    )
    all_line_data = data_file.getLineData(
        all_lines, all_cycles, 1 if not all_channels else None
    )

    acc = {
        "data": np.zeros((num_sp, n_ds), dtype=np.float32),
        "data_count": np.zeros((num_sp, n_ds), dtype=np.float32),
        "data2": (
            np.zeros((num_sp, n_ds), dtype=np.float32) if read_second else None
        ),
        "data_count2": (
            np.zeros((num_sp, n_ds), dtype=np.float32) if read_second else None
        ),
    }

    for ds_ix in range(n_ds):
        weights = np.exp(-np.abs(ds_frames[ds_ix] - time_windows[ds_ix]) / dt)
        frame_lines = line_indices[ds_ix]
        frame_cycles = cycle_indices[ds_ix]
        for i, li in enumerate(frame_lines):
            line_idx = int(li)
            if data_file.lineDataNumElements[line_idx - 1] == 0:
                continue
            cache_idx = mapping[(line_idx, int(frame_cycles[i]))]
            _accumulate_line(
                acc,
                all_line_data[cache_idx],
                data_file.lineSuperPixelIDs[line_idx - 1],
                data_file.lineFastZIdxs[line_idx - 1],
                sp_flat,
                weights[i],
                ds_ix,
            )
    return acc


def _open_slap2_file(path: str):  # pragma: no cover - requires slap2_utils
    """Open a SLAP2 data file (``MultiDataFiles`` for CYCLE globs else single).

    Parameters
    ----------
    path : str
        Absolute path to the SLAP2 ``.dat`` (or CYCLE-pattern) file.

    Returns
    -------
    object
        A ``slap2_utils`` data-file object.
    """
    import importlib

    import slap2_utils

    importlib.reload(slap2_utils)
    if re.search(r"CYCLE\d+", path):
        return slap2_utils.MultiDataFiles(path)
    return slap2_utils.DataFile(path)


def read_band_trial_data(
    trial_ix: int,
    keep_trial: bool,
    dmd_ix: int,
    samp_freq: float,
    super_pixel_ids: np.ndarray,
    datadr: str,
    trial_table: dict,
    num_channels: int,
    all_channels: bool = True,
) -> Optional[dict]:
    """Read one band trial's low-res superpixel activity.

    Opens the trial's SLAP2 file, builds the downsampled-frame grid,
    accumulates weighted superpixel activity, and loads the matching alignment
    data. Counts are returned unnormalized (the caller forms
    ``data / data_count``); activity is divided by 100 to undo the acquisition
    scaling.

    Parameters
    ----------
    trial_ix : int
        Trial index into ``trial_table`` columns.
    keep_trial : bool
        When False, the trial is skipped and ``None`` is returned.
    dmd_ix : int
        0-based DMD index into ``trial_table`` rows.
    samp_freq : float
        Analysis/alignment sample rate, in Hz.
    super_pixel_ids : ndarray
        Superpixel id lookup for this DMD.
    datadr : str
        Raw-data directory holding the SLAP2 files.
    trial_table : dict
        Normalized trial table (``filename``, ``first_line``, ``last_line``,
        ``fn_adata``).
    num_channels : int
        Number of acquisition channels.
    all_channels : bool
        Read all channels (default) vs channel 1 only.

    Returns
    -------
    dict or None
        ``{"data", "data_count", "alignment", "ds_frames", "data2",
        "data_count2"}`` for kept trials, else ``None``.
    """
    if not keep_trial:
        return None

    source_fn = trial_table["filename"][dmd_ix, trial_ix]
    first_line = trial_table["first_line"][dmd_ix, trial_ix]
    last_line = trial_table["last_line"][dmd_ix, trial_ix]

    data_file = _open_slap2_file(os.path.join(datadr, source_fn))
    dt = 1.0 / samp_freq / data_file.metaData.linePeriod_s
    ds_frames = compute_ds_frames(first_line, last_line, dt)

    acc = accumulate_superpixel_data(
        data_file, ds_frames, dt, super_pixel_ids, num_channels, all_channels
    )
    alignment = load_alignment_data_h5(
        trial_table["fn_adata"][dmd_ix, trial_ix]
    )

    data2 = acc["data2"]
    return {
        "data": acc["data"] / 100.0,
        "data_count": acc["data_count"],
        "alignment": alignment,
        "ds_frames": ds_frames,
        "data2": None if data2 is None else data2 / 100.0,
        "data_count2": acc["data_count2"],
    }


def assemble_lowres_data(results: list, num_channels: int) -> dict:
    """Concatenate per-trial reader results into low-res arrays.

    Port of the concatenation block that stitches per-trial outputs into the
    per-DMD low-res dataset. ``None`` results (skipped trials) are dropped.

    Parameters
    ----------
    results : list
        Per-trial dicts from :func:`read_band_trial_data` (or ``None``).
    num_channels : int
        Number of acquisition channels (>= 2 adds second-channel arrays).

    Returns
    -------
    dict
        ``lowResData``, ``lowResDataCt``, ``lowResMotionR/C/Z``,
        ``lowResTrialID`` (and ``lowResData2``/``lowResDataCt2`` when
        ``num_channels >= 2``).
    """
    kept = [(i, r) for i, r in enumerate(results) if r is not None]
    arrays = {
        "lowResData": np.concatenate([r["data"] for _, r in kept], axis=1),
        "lowResDataCt": np.concatenate(
            [r["data_count"] for _, r in kept], axis=1
        ),
        "lowResMotionR": np.concatenate(
            [r["alignment"]["motionDSr"] for _, r in kept], axis=0
        ),
        "lowResMotionC": np.concatenate(
            [r["alignment"]["motionDSc"] for _, r in kept], axis=0
        ),
        "lowResMotionZ": np.concatenate(
            [r["alignment"]["motionDSz"] for _, r in kept], axis=0
        ),
        "lowResTrialID": np.concatenate(
            [np.ones_like(r["ds_frames"]) * i for i, r in kept], axis=0
        ),
    }
    if num_channels >= 2:
        arrays["lowResData2"] = np.concatenate(
            [r["data2"] for _, r in kept], axis=1
        )
        arrays["lowResDataCt2"] = np.concatenate(
            [r["data_count2"] for _, r in kept], axis=1
        )
    return arrays


def save_lowres_data(path: str, arrays: dict) -> None:
    """Save assembled low-res arrays to an ``.npz`` cache."""
    np.savez(path, **arrays)


def load_lowres_data(path: str, num_channels: int) -> dict:
    """Load an ``.npz`` low-res cache into a dict of arrays.

    Parameters
    ----------
    path : str
        Path to the ``lowres_data_DMD{N}.npz`` cache.
    num_channels : int
        Number of acquisition channels (>= 2 also loads the second channel).

    Returns
    -------
    dict
        The cached low-res arrays.
    """
    keys = [
        "lowResData",
        "lowResDataCt",
        "lowResMotionR",
        "lowResMotionC",
        "lowResMotionZ",
        "lowResTrialID",
    ]
    if num_channels >= 2:
        keys += ["lowResData2", "lowResDataCt2"]
    with np.load(path) as data_arrays:
        return {k: data_arrays[k] for k in keys}
