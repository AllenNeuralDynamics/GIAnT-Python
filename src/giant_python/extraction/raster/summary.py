"""SILo's output files in GIAnT-MATLAB's schema.

Port of what ``SILo.m`` (GIAnT-MATLAB main 5a0673b) does after extraction:
the kinetics test (:mod:`~giant_python.extraction.raster.kinetics`), the
experiment summary (``saveExperimentSummaryH5.m``), the per-trial summary
(``savePerTrialSummaryH5.m``) and the trial table's ``source_extraction``
fields. The files have MATLAB's names, sizes, orientation (column-major:
h5py shows MATLAB's dimensions reversed), classes and ``/row_major = 0``,
so that MATLAB's ``h5read`` and GIAnT-Python's
:func:`~giant_python.io.hdf5.load_struct_h5` read them as MATLAB's.

Trials are processed one at a time: a path's trials may be given as
:class:`~giant_python.extraction.raster.results.TrialOutcome` objects or
as functions that load them (from results files), so that a session's
traces need not all be in memory. The sources' traces are written to their
datasets trial by trial; per-trial footprints are kept (pixels x sources
per trial) to write the profile images.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import h5py
import numpy as np

from ...io.hdf5 import write_matlab_value
from .kinetics import KineticsTest, reject_fast_sources
from .plan import ExtractionPlan
from .results import TrialOutcome
from .trial_data import upsample_motion

TrialSource = Union[None, TrialOutcome, Callable[[], TrialOutcome]]


@dataclass
class AlignmentRecord:
    """The registration's per-frame motion of one trial (MATLAB's
    ``aData``), as the experiment summary's ``frame_info`` uses it.

    Attributes
    ----------
    ds_frames : ndarray
        One-based line of each registered frame (``DSframes``).
    motion_r, motion_c : ndarray
        Row and column motion of each registered frame (``motionDSr``,
        ``motionDSc``).
    motion_z : ndarray or None
        ``motionDSz``, when the file has it.
    online_x, online_y, online_z : ndarray or None
        The online motion correction (``slap2.onlineMotion?shift``).
    z_depth : float
        ``slap2.Z_depths`` (NaN when missing).
    """

    ds_frames: np.ndarray
    motion_r: np.ndarray
    motion_c: np.ndarray
    motion_z: Optional[np.ndarray] = None
    online_x: Optional[np.ndarray] = None
    online_y: Optional[np.ndarray] = None
    online_z: Optional[np.ndarray] = None
    z_depth: float = math.nan


def read_alignment_record(path) -> AlignmentRecord:
    """Read what the summary needs from a ``*_ALIGNMENTDATA.h5``.

    Parameters
    ----------
    path : str or Path
        The alignment file (MATLAB's or GIAnT-Python's).

    Returns
    -------
    AlignmentRecord
    """

    def vec(f, name):
        """A dataset as a flat float64 vector, or None."""
        if name not in f:
            return None
        return np.asarray(f[name][()], dtype=np.float64).ravel()

    with h5py.File(path, "r") as f:
        z = vec(f, "slap2/Z_depths")
        return AlignmentRecord(
            ds_frames=vec(f, "DSframes"),
            motion_r=vec(f, "motionDSr"),
            motion_c=vec(f, "motionDSc"),
            motion_z=vec(f, "motionDSz"),
            online_x=vec(f, "slap2/onlineMotionXshift"),
            online_y=vec(f, "slap2/onlineMotionYshift"),
            online_z=vec(f, "slap2/onlineMotionZshift"),
            z_depth=float(z[0]) if z is not None and z.size else math.nan,
        )


@dataclass
class PathResult:
    """Everything the summaries need of one imaging path.

    Attributes
    ----------
    plan : ExtractionPlan
        The path's plan (images, alignment, sources, selected pixels,
        per-trial peaks).
    trials : list
        One entry per trial of the trial table: None (not extracted), a
        ``TrialOutcome``, or a function returning it.
    alignment : list
        One ``AlignmentRecord`` (or None) per trial; needed for the trials
        that have an outcome.
    roi_labels : list of str
        The user ROIs' labels.
    roi_masks : list of ndarray of bool
        Their masks in the consensus frame.
    z_depth : float
        The path's ``Z_depths`` (NaN: not written).
    trial_peaks : list or None
        Each trial's own peaks (zero-based ``[amplitude, row, col,
        sigma]`` in its registered frame, or None); None uses the plan's.
    kinetics : KineticsTest or None
        Set by :func:`kinetics_test`.
    """

    plan: ExtractionPlan
    trials: list
    alignment: list
    roi_labels: list = field(default_factory=list)
    roi_masks: list = field(default_factory=list)
    z_depth: float = math.nan
    trial_peaks: Optional[list] = None
    kinetics: Optional[KineticsTest] = None

    def outcome(self, t: int) -> Optional[TrialOutcome]:
        """Trial ``t``'s outcome (loaded if given as a function)."""
        entry = self.trials[t] if t < len(self.trials) else None
        return entry() if callable(entry) else entry


def kinetics_test(
    result: PathResult, tau_s: float, analyze_hz: float, min_score: float
) -> KineticsTest:
    """Run the kinetics test on a path (``rejectFastSources``).

    The activity channel's least-squares traces of every extracted trial
    are scored; the result is stored in ``result.kinetics``.
    """
    traces = []
    for t in range(len(result.trials)):
        out = result.outcome(t)
        if out is not None and out.activity is not None:
            traces.append(np.asarray(out.activity.ls[0], dtype=np.float64))
    result.kinetics = reject_fast_sources(
        traces, result.plan.n_sources, tau_s, analyze_hz, min_score
    )
    return result.kinetics


def _ordered(num_channels: int, activity_channel: int) -> np.ndarray:
    """Zero-based file channels in analysis order (activity first)."""
    a = activity_channel - 1
    return np.r_[np.arange(a, num_channels), np.arange(a)].astype(np.int64)


def _nearest(ds: np.ndarray, values: np.ndarray, lines) -> np.ndarray:
    """MATLAB ``interp1(ds, values, lines, 'nearest', 'extrap')`` (ties to
    the right); one registered frame: its value throughout."""
    lines = np.asarray(lines, dtype=np.float64)
    if ds.size == 1:
        return np.full(lines.shape, float(values[0]))
    right = np.clip(np.searchsorted(ds, lines, side="right"), 1, ds.size - 1)
    left = right - 1
    take_right = (lines - ds[left]) >= (ds[right] - lines)
    return values[np.where(take_right, right, left)]


def _pchip(ds: np.ndarray, values: np.ndarray, lines) -> np.ndarray:
    """``interp1(..., 'pchip', 'extrap')`` as the loader upsamples motion."""
    return upsample_motion(ds, values, np.asarray(lines, dtype=np.float64))


class _Writer:
    """Values written into an HDF5 file as ``saveStructToH5.m`` would."""

    def __init__(self, f: h5py.File) -> None:
        """Wrap an open file."""
        self.f = f

    def put(self, path: str, value) -> None:
        """Write a value (MATLAB orientation) at ``/a/b/name``; empty values
        are skipped and groups exist only above datasets, as in MATLAB."""
        parent, name = path.rsplit("/", 1)
        if np.ndim(value) and np.size(value) == 0:
            return
        grp = self.f.require_group(parent) if parent else self.f
        write_matlab_value(grp, name, value)

    def create(self, path: str, size: tuple, dtype) -> h5py.Dataset:
        """An empty contiguous dataset of MATLAB size ``size``."""
        parent, name = path.rsplit("/", 1)
        grp = self.f.require_group(parent) if parent else self.f
        return grp.create_dataset(name, shape=tuple(size[::-1]), dtype=dtype)


def _profile_blocks(ds, images, cols_per_block: int = 64) -> None:
    """Write images ``(..., rows, cols)`` as MATLAB's ``(..., rows, cols)``
    array (h5py: reversed) in blocks of columns.

    ``images(c0, c1)`` returns the block's array in MATLAB orientation,
    ``(lead..., rows, c1 - c0)``.
    """
    n_cols = ds.shape[0]
    for c0 in range(0, n_cols, cols_per_block):
        c1 = min(n_cols, c0 + cols_per_block)
        block = images(c0, c1)
        ds[c0:c1] = np.ascontiguousarray(block.T)


def _footprint_images(
    footprints: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    shape: tuple,
    c0: int,
    c1: int,
) -> np.ndarray:
    """Footprints ``(pixels, sources)`` as images ``(sources, rows, c1 -
    c0)`` of columns ``c0:c1`` (zero elsewhere)."""
    sel = (cols >= c0) & (cols < c1)
    out = np.zeros((footprints.shape[1], shape[0], c1 - c0), np.float32)
    out[:, rows[sel], cols[sel] - c0] = footprints[sel].T
    return out


def _write_path(  # noqa: C901
    ew: _Writer,
    pw: _Writer,
    name: str,
    result: PathResult,
    n_trials: int,
    num_channels: int,
    activity_channel: int,
) -> None:
    """Write one path's groups of both summaries."""
    plan = result.plan
    kin = result.kinetics
    n_src_all = plan.n_sources
    keep = kin.keep if kin is not None else np.ones(n_src_all, bool)
    kept = np.flatnonzero(keep)
    ordered = _ordered(num_channels, activity_channel)
    h, w = plan.shape
    sel_rows, sel_cols = plan.selected_pixels()

    if not math.isnan(result.z_depth):
        ew.put(f"{name}/Z_depths", np.array([[float(result.z_depth)]]))

    # first pass: frame counts and which trials have sources
    counts = np.zeros(n_trials, dtype=np.int64)
    has_sources = np.zeros(n_trials, bool)
    n_analyzed = 0
    n_global = 0
    for t in range(n_trials):
        out = result.outcome(t)
        if out is None:
            continue
        counts[t] = out.record.n_frames
        n_global = n_global or out.record.global_f.shape[0]
        if out.activity is not None:
            if out.activity.ls.shape[-1] != counts[t]:
                raise ValueError(f"trial {t}: traces and frames differ")
            has_sources[t] = True
            n_analyzed = n_analyzed or out.activity.ls.shape[0]
    total = int(counts.sum())
    n_kept = kept.size
    write_sources = bool(has_sources.any()) and n_kept > 0

    # rejected sources (before any early return, as MATLAB)
    if kin is not None and (~keep).any():
        rej = np.flatnonzero(~keep)
        ew.put(
            f"{name}/sources/rejected/seed_coords",
            np.column_stack(
                [
                    np.zeros(rej.size),
                    plan.source_rows[rej].astype(np.float64),
                    plan.source_cols[rej].astype(np.float64),
                ]
            ),
        )
        ew.put(
            f"{name}/sources/rejected/kinetic_score",
            kin.score[rej].reshape(-1, 1).astype(np.float64),
        )
        ew.put(
            f"{name}/sources/rejected/kinetic_n_events",
            kin.n_events[rej].reshape(-1, 1).astype(np.int32),
        )

    # frame_info, global and user-ROI traces, and the sources' traces
    fi = {
        k: np.full(total, np.nan) for k in ("oX", "oY", "oZ", "nX", "nY", "nZ")
    }
    lines = np.zeros(total)
    discard = np.zeros(total, bool)
    n_rois = len(result.roi_masks)
    gf = np.full((n_global, total), np.nan) if n_global else None
    rf = np.full((n_rois, num_channels, total), np.nan)
    rfs = np.full((n_rois, num_channels, total), np.nan)
    temporal = {}
    if write_sources and total and n_analyzed:
        for key in ("dF_ls", "dF_denoised", "events", "F0"):
            temporal[key] = ew.create(
                f"{name}/sources/temporal/{key}",
                (n_kept, num_channels, total),
                np.float64,
            )
    analyzed = ordered[:n_analyzed]
    snr = np.full((n_kept, n_trials), np.nan)
    profile_sum = np.zeros((sel_rows.size, n_kept))
    n_profile = 0
    per_trial_fp: dict = {}
    off = 0
    for t in range(n_trials):
        n = int(counts[t])
        if n == 0:
            continue
        out = result.outcome(t)
        rec = out.record
        idx = slice(off, off + n)
        discard[idx] = rec.discard
        fl = np.asarray(rec.frame_lines, dtype=np.float64)
        lines[idx] = fl
        al = result.alignment[t] if t < len(result.alignment) else None
        if al is not None and al.ds_frames is not None:
            dr, dc = plan.shifts[t]
            if al.motion_c is not None:
                fi["oX"][idx] = _pchip(al.ds_frames, al.motion_c, fl) + dc
            if al.motion_r is not None:
                fi["oY"][idx] = _pchip(al.ds_frames, al.motion_r, fl) + dr
            if al.motion_z is not None:
                fi["oZ"][idx] = _pchip(al.ds_frames, al.motion_z, fl)
            for key, values in (
                ("nX", al.online_x),
                ("nY", al.online_y),
                ("nZ", al.online_z),
            ):
                if values is not None:
                    fi[key][idx] = _nearest(al.ds_frames, values, fl)
        if gf is not None and rec.global_f.size:
            gf[:, idx] = rec.global_f
        if n_rois:
            if rec.roi_f.size:
                rf[:, ordered[: rec.roi_f.shape[1]], idx] = rec.roi_f
            if rec.roi_fsvd.size:
                rfs[:, ordered[: rec.roi_fsvd.shape[1]], idx] = rec.roi_fsvd
        act = out.activity
        if act is None and temporal:
            # a trial loaded but not extracted: NaN, as MATLAB's arrays
            slab = np.full((n, num_channels, n_kept), np.nan)
            for key in temporal:
                temporal[key][idx] = slab
        if act is not None and write_sources:
            if temporal:
                for key, arr in (
                    ("dF_ls", act.ls),
                    ("dF_denoised", act.denoised),
                    ("events", act.events),
                    ("F0", act.f0),
                ):
                    slab = np.full((n, num_channels, n_kept), np.nan)
                    for c, ch in enumerate(analyzed):
                        slab[:, ch, :] = arr[c][kept].T
                    temporal[key][idx] = slab
            if np.size(act.snr) == n_src_all:
                snr[:, t] = np.asarray(act.snr, dtype=np.float64)[kept]
            fp = np.asarray(act.footprints)[:, kept]
            fp32 = np.where(np.isnan(fp), 0, fp).astype(np.float32)
            profile_sum += fp32
            n_profile += 1
            per_trial_fp[t] = np.asarray(fp, dtype=np.float32)
        off += n

    ew.put(
        f"{name}/frame_info/trial_num_frames",
        counts.reshape(-1, 1).astype(np.int32),
    )
    ew.put(f"{name}/frame_info/offlineXshifts", fi["oX"].reshape(-1, 1))
    ew.put(f"{name}/frame_info/offlineYshifts", fi["oY"].reshape(-1, 1))
    if not np.isnan(fi["oZ"]).all():
        ew.put(f"{name}/frame_info/offlineZshifts", fi["oZ"].reshape(-1, 1))
    ew.put(
        f"{name}/frame_info/frame_line_idxs",
        lines.reshape(-1, 1).astype(np.int32),
    )
    ew.put(f"{name}/frame_info/discard_frames", discard.reshape(-1, 1))
    ew.put(f"{name}/frame_info/onlineXshifts", fi["nX"].reshape(-1, 1))
    ew.put(f"{name}/frame_info/onlineYshifts", fi["nY"].reshape(-1, 1))
    ew.put(f"{name}/frame_info/onlineZshifts", fi["nZ"].reshape(-1, 1))
    affine = np.full((n_trials, 2, 3), np.nan)
    m = min(n_trials, plan.warp.affine.shape[0])
    affine[:m] = plan.warp.affine[:m]
    ew.put(f"{name}/frame_info/trial_alignment_affine", affine)

    # visualizations
    mean_im = np.asarray(plan.mean_im, dtype=np.float64)
    ew.put(
        f"{name}/visualizations/mean_im",
        mean_im.transpose(2, 0, 1)[:, None],
    )
    ew.put(
        f"{name}/visualizations/act_im",
        np.asarray(plan.act_im, dtype=np.float64)[None],
    )
    if n_kept:
        ew.put(
            f"{name}/visualizations/act_im_peaks",
            np.column_stack(
                [
                    np.zeros(n_kept),
                    plan.source_rows[kept].astype(np.float64),
                    plan.source_cols[kept].astype(np.float64),
                ]
            ),
        )
    if gf is not None and total:
        ew.put(f"{name}/global/F", gf)
    if n_rois:
        ew.put(
            f"{name}/user_rois/labels",
            np.array(result.roi_labels, dtype=object).reshape(-1, 1),
        )
        masks = np.zeros((n_rois, 1, h, w), np.uint8)
        for r, mask in enumerate(result.roi_masks):
            masks[r, 0] = mask
        ew.put(f"{name}/user_rois/mask", masks)
        if total:
            ew.put(f"{name}/user_rois/F", rf)
            ew.put(f"{name}/user_rois/Fsvd", rfs)

    if write_sources:
        mean_fp = (profile_sum / n_profile).astype(np.float32)
        ds = ew.create(
            f"{name}/sources/spatial/profiles", (n_kept, 1, h, w), np.float32
        )
        _profile_blocks(
            ds,
            lambda c0, c1: _footprint_images(
                mean_fp, sel_rows, sel_cols, (h, w), c0, c1
            )[:, None],
        )
        ew.put(
            f"{name}/sources/spatial/coords",
            _centroids(mean_fp, sel_rows, sel_cols, zero_if_empty=True),
        )
        if (~np.isnan(snr)).any():
            ok = ~np.isnan(snr)
            mean_snr = np.full(n_kept, np.nan)
            some = ok.any(axis=1)
            mean_snr[some] = (
                np.where(ok, snr, 0).sum(1)[some] / ok.sum(1)[some]
            )
            ew.put(f"{name}/sources/temporal/SNR", mean_snr.reshape(-1, 1))
        if kin is not None and kin.score.size == n_src_all:
            ew.put(
                f"{name}/sources/temporal/kinetic_score",
                kin.score[kept].reshape(-1, 1).astype(np.float64),
            )
            ew.put(
                f"{name}/sources/temporal/kinetic_n_events",
                kin.n_events[kept].reshape(-1, 1).astype(np.int32),
            )

    _write_per_trial(
        pw, name, result, n_trials, kept, per_trial_fp, snr, write_sources
    )


def _centroids(
    fp: np.ndarray, rows: np.ndarray, cols: np.ndarray, zero_if_empty: bool
) -> np.ndarray:
    """Zero-based ``[0, row, col]`` centroid of each footprint
    ``(pixels, sources)``; 0 (experiment summary) or NaN (per trial) where a
    footprint sums to 0 or less."""
    fp = np.where(np.isnan(fp), 0.0, np.asarray(fp, dtype=np.float64))
    wsum = fp.sum(axis=0)
    out = np.zeros((fp.shape[1], 3))
    fill = 0.0 if zero_if_empty else np.nan
    out[:, 1:] = fill
    ok = wsum > 0
    out[ok, 1] = (fp[:, ok] * rows[:, None]).sum(0) / wsum[ok]
    out[ok, 2] = (fp[:, ok] * cols[:, None]).sum(0) / wsum[ok]
    return out


def _write_per_trial(
    pw: _Writer,
    name: str,
    result: PathResult,
    n_trials: int,
    kept: np.ndarray,
    per_trial_fp: dict,
    snr: np.ndarray,
    write_sources: bool,
) -> None:
    """Write one path's group of the per-trial summary."""
    plan = result.plan
    h, w = plan.shape
    act_aligned = np.full((n_trials, h, w), np.nan)
    mean_aligned = np.full(
        (n_trials, h, w, plan.mean_aligned.shape[3]), np.nan
    )
    m = min(n_trials, plan.act_aligned.shape[0])
    act_aligned[:m] = plan.act_aligned[:m]
    mean_aligned[:m] = plan.mean_aligned[:m]
    imaged = ~np.isnan(act_aligned).all(axis=(1, 2))
    mean_aligned[~imaged] = np.nan
    pw.put(
        f"{name}/visualizations/per_trial_mean_im",
        mean_aligned.transpose(0, 3, 1, 2)[:, :, None],
    )
    pw.put(f"{name}/visualizations/per_trial_act_im", act_aligned[:, None])
    peaks = (
        plan.trial_peaks if result.trial_peaks is None else result.trial_peaks
    )
    if any(p is not None for p in peaks):
        n_peaks = np.zeros(n_trials, dtype=np.int32)
        for t in range(min(n_trials, len(peaks))):
            if peaks[t] is not None:
                n_peaks[t] = np.asarray(peaks[t]).reshape(-1, 4).shape[0]
        pw.put(
            f"{name}/visualizations/per_trial_num_peaks",
            n_peaks.reshape(-1, 1),
        )
        most = int(n_peaks.max(initial=0))
        if most:
            out = np.full((n_trials, most, 3), np.nan)
            for t in np.flatnonzero(n_peaks):
                p = np.asarray(peaks[t]).reshape(-1, 4)
                dr, dc = (
                    plan.shifts[t]
                    if t < plan.shifts.shape[0]
                    else (np.nan, np.nan)
                )
                out[t, : p.shape[0], 0] = 0
                out[t, : p.shape[0], 1] = p[:, 1] - dr
                out[t, : p.shape[0], 2] = p[:, 2] - dc
            pw.put(f"{name}/visualizations/per_trial_act_im_peaks", out)
    if not write_sources:
        return
    n_kept = kept.size
    rows, cols = plan.selected_pixels()
    coords = np.full((n_trials, n_kept, 3), np.nan)
    coords[:, :, 0] = 0
    for t, fp in per_trial_fp.items():
        coords[t] = _centroids(fp, rows, cols, zero_if_empty=False)
    ds = pw.create(
        f"{name}/sources/spatial/per_trial_profiles",
        (n_trials, n_kept, 1, h, w),
        np.float32,
    )

    def block(c0, c1):
        """Columns ``c0:c1`` of every trial's profiles."""
        out = np.full((n_trials, n_kept, 1, h, c1 - c0), np.nan, np.float32)
        for t, fp in per_trial_fp.items():
            fp32 = np.where(np.isnan(fp), 0, fp).astype(np.float32)
            out[t, :, 0] = _footprint_images(fp32, rows, cols, (h, w), c0, c1)
        return out

    _profile_blocks(ds, block)
    pw.put(f"{name}/sources/spatial/per_trial_coords", coords)
    pw.put(f"{name}/sources/temporal/per_trial_SNR", snr.T.copy())


def write_summaries(
    experiment_path,
    per_trial_path,
    paths: Sequence[Optional[PathResult]],
    params: dict,
    n_trials: int,
    num_channels: int,
    activity_channel: int,
) -> None:
    """Write ``experiment_summary.h5`` and ``per_trial_summary.h5``.

    Ports of ``saveExperimentSummaryH5.m`` and ``savePerTrialSummaryH5.m``;
    both files are written under temporary names and moved into place when
    complete. Run :func:`kinetics_test` on each path first (a path without
    it keeps every source and gets no kinetics fields).

    Parameters
    ----------
    experiment_path, per_trial_path : str or Path
        The two files (replaced).
    paths : sequence of (PathResult or None)
        One entry per imaging path of the trial table; None for a path
        without valid trials (it is left out, as in MATLAB).
    params : dict
        The ``params`` struct (:meth:`SiloParams.to_matlab`).
    n_trials : int
        Trials of the trial table (the trial axis of the per-trial file).
    num_channels : int
        Channels of the recording.
    activity_channel : int
        One-based activity channel.
    """
    targets = [Path(experiment_path), Path(per_trial_path)]
    tmps = [p.with_name(f"{p.name}.{os.getpid()}.partial") for p in targets]
    try:
        with h5py.File(tmps[0], "w") as fe, h5py.File(tmps[1], "w") as fp:
            ew, pw = _Writer(fe), _Writer(fp)
            for d, result in enumerate(paths):
                if result is None:
                    continue
                _write_path(
                    ew,
                    pw,
                    f"Path{d + 1}",
                    result,
                    n_trials,
                    num_channels,
                    activity_channel,
                )
            for key, value in params.items():
                ew.put(f"params/{key}", value)
            for f in (fe, fp):
                f.create_dataset("row_major", data=np.zeros((1, 1), np.uint8))
        for tmp, target in zip(tmps, targets):
            os.replace(tmp, target)
    except BaseException:
        for tmp in tmps:
            if tmp.exists():
                tmp.unlink()
        raise
