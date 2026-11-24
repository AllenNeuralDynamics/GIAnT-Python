"""SLAP2 trial tables: GIAnT-MATLAB's ``buildTrialTableSLAP2`` in Python.

A trial table lists a session's analysis trials: for each DMD path and trial,
the ``.dat`` file and the first and last line to read, with the reference
stacks and the trial times. :func:`build_trial_table_slap2` writes the same
``trial_table.h5`` as ``buildTrialTableSLAP2(dr, savedr, true)`` (every file
in one epoch), so registration and SILo, in either language, use it
unchanged. :func:`trim_trial_table` keeps the first seconds of a session, as
the GIAnT integration-test capsule's ``trimTrialTableSLAP2`` does.

Files of a path are put in acquisition order by the timestamp and trial
number in their names. Triggered recordings (one file per trial) are lined
up across paths, which also handles SLAP2's trial-numbering bug of Feb 2024
(a path whose trial number failed to increment holds two trials in one
file). Continuous recordings (``-CYCLE-`` files) are cut into trials of at
most 200000 lines. Trial times come from the FPGA line clock when the
``.meta`` file has ``fpgaTimeReference``, otherwise from file modification
times.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

LINE_DIFF_THRESH = 2000
"""Line counts closer than this (about 0.2 s) are the same trial length."""

MULTI_CYCLE_LINES_PER_TRIAL = 200000
"""Continuous recordings are cut into trials of at most this many lines."""

_DATENUM_1970 = 719529.0
"""MATLAB ``datenum(1970, 1, 1)``."""

_MS_PER_DAY = 86400000.0


class TrialTableWarning(UserWarning):
    """File order or trial times may be wrong (as MATLAB's warnings)."""


@dataclass(frozen=True)
class DatInfo:
    """What the trial table needs from one recording (all its files).

    Attributes
    ----------
    num_lines : int
        Lines in the recording (``totalNumLines``).
    first_line_time : float
        MATLAB ``datenum`` of the first line by the FPGA clock, or NaN.
    line_period_s : float
        Line period from the metadata file.
    """

    num_lines: int
    first_line_time: float
    line_period_s: float


def read_dat_info(path: Union[str, Path]) -> DatInfo:
    """Line count, first-line time and line period of a SLAP2 recording.

    A file of a CYCLE-split recording stands for all its files, as with
    GIAnT-MATLAB's ``Slap2DataFile``. The first line is the earliest first
    line of the files; its time is the ``.meta`` file's
    ``fpgaTimeReference`` plus the line's tick count over
    ``fpgaSystemClock_Hz``.

    Parameters
    ----------
    path : str or Path
        A ``.dat`` file.

    Returns
    -------
    DatInfo
        The recording's line count, first-line time and line period.
    """
    from ..io.slap2 import (
        first_line_timestamp,
        open_slap2_file,
        read_mat_datetime_ms,
    )

    reader = open_slap2_file(path)
    files = list(getattr(reader, "hDataFiles", None) or [reader])
    first = files[0]
    lines_per_cycle = int(first.header["linesPerCycle"])
    num_lines = int(sum(int(df.numCycles) for df in files)) * lines_per_cycle
    line_period_s = float(first.metaData.linePeriod_s)
    t = math.nan
    try:
        ticks = min(first_line_timestamp(df) for df in files)
    except (OSError, ValueError):
        ticks = 0
    clock_hz = float(first.header.get("fpgaSystemClock_Hz", 0) or 0)
    ref_ms = read_mat_datetime_ms(first.metaDataFileName, "fpgaTimeReference")
    if ref_ms is not None and ticks > 0 and clock_hz > 0:
        t = _DATENUM_1970 + (ref_ms + 1000.0 * ticks / clock_hz) / _MS_PER_DAY
    return DatInfo(num_lines, t, line_period_s)


def _mtime_datenum(path: Path) -> float:
    """MATLAB ``dir`` datenum: local modification time in days."""
    t = datetime.datetime.fromtimestamp(os.stat(path).st_mtime)
    midnight = t.replace(hour=0, minute=0, second=0, microsecond=0)
    return t.toordinal() + 366 + (t - midnight).total_seconds() / 86400.0


def _later_cycle_file(name: str) -> bool:
    """Whether a file is a later file of a CYCLE-split recording."""
    m = re.search(r"CYCLE-?(\d+)", name)
    return m is not None and int(m.group(1)) > 0


def _recording_mtime(dat_dir: Path, name: str, all_names) -> float:
    """Modification datenum of a recording: of its file, or for a CYCLE-split
    recording of its last file, written when the recording ended.

    MATLAB uses the first file's time for a CYCLE-split recording, which puts
    file-time trial times too early by the length of the later files; only
    recordings without the FPGA line clock use these times.
    """
    m = re.search(r"CYCLE-?\d+", name)
    if m is None:
        return _mtime_datenum(dat_dir / name)
    stem = name[: m.start()]
    parts = [
        n
        for n in all_names
        if n.startswith(stem) and re.match(r"CYCLE-?\d+", n[len(stem) :])
    ]
    return max(_mtime_datenum(dat_dir / n) for n in parts)


def _name_timestamp(stamp: str) -> float:
    """``datenum`` of a yyyyMMddHHmmss string, or NaN if not a valid date."""
    v = [int(stamp[0:4]), int(stamp[4:6]), int(stamp[6:8])]
    v += [int(stamp[8:10]), int(stamp[10:12]), int(stamp[12:14])]
    if not (
        1 <= v[1] <= 12
        and 1 <= v[2] <= 31
        and v[3] < 24
        and v[4] < 60
        and v[5] < 60
    ):
        return math.nan
    try:
        t = datetime.datetime(*v)
    except ValueError:  # e.g. 31 February, which MATLAB's datenum rolls over
        base = datetime.datetime(v[0], v[1], 1)
        t = base + datetime.timedelta(
            days=v[2] - 1, hours=v[3], minutes=v[4], seconds=v[5]
        )
    return (
        t.toordinal()
        + 366
        + (t.hour * 3600 + t.minute * 60 + t.second) / 86400.0
    )


def acquisition_order(names: Sequence[str]) -> Tuple[np.ndarray, bool]:
    """Order SLAP2 ``.dat`` files by acquisition time, then trial number.

    Names look like ``activity_20250314_111223_DMD1-TRIAL000002.dat``; files
    of continuous recordings end in ``-CYCLE-000000`` instead of, or after,
    the trial number.

    Parameters
    ----------
    names : sequence of str
        File names of one path.

    Returns
    -------
    order : ndarray of int
        Zero-based indexes in acquisition order (the identity when not ok).
    ok : bool
        False when files of different acquisitions have no timestamp in
        their names, or when two files get the same position.
    """
    n = len(names)
    acq = np.full(n, np.nan)
    trial = np.zeros(n)
    base = []
    for k, name in enumerate(names):
        stem = re.sub(
            r"-CYCLE-?\d+\.dat$|\.dat$", "", name, flags=re.IGNORECASE
        )
        m = re.search(r"-TRIAL(\d+)$", stem, flags=re.IGNORECASE)
        if m:
            trial[k] = float(m.group(1))
        b = re.sub(r"-TRIAL\d+$", "", stem, flags=re.IGNORECASE)
        base.append(b)
        stamps = re.findall(r"(?<!\d)(\d{8})_(\d{6})(?!\d)", b)
        if stamps:
            acq[k] = _name_timestamp("".join(stamps[-1]))
    identity = np.arange(n)
    if np.all(np.isfinite(acq)):
        key = np.column_stack([acq, trial])
    elif len(set(base)) == 1:
        key = trial[:, None]
    else:
        return identity, False
    ok = np.unique(key, axis=0).shape[0] == n
    if not ok:
        return identity, False
    order = np.lexsort(key.T[::-1])
    return order, True


def _copy_time_reason(
    mtimes: np.ndarray,
    num_lines: np.ndarray,
    line_period_s: float,
    check_order: bool,
) -> str:
    """Why modification times look like copy times rather than write times.

    SLAP2 writes each file as its trial ends, so in acquisition order the
    times increase and span at least the recording time of all files but
    one. Copies (e.g. from S3) carry the copy time instead.
    """
    if mtimes.size < 2:
        return ""
    mtime_s = mtimes * 86400.0
    recorded_s = num_lines * line_period_s
    min_span_s = recorded_s.sum() - recorded_s.max()
    span_s = mtime_s.max() - mtime_s.min()
    if check_order and np.any(np.diff(mtime_s) < -2):
        return "they are out of order relative to the file names"
    if span_s < 0.5 * min_span_s:
        return (
            f"they span {span_s:.0f} s, but the trials add up to at least "
            f"{min_span_s:.0f} s"
        )
    return ""


def matlab_linspace(d1: float, d2: float, n: int) -> np.ndarray:
    """MATLAB ``linspace``, with the same rounding (endpoints exact)."""
    n = int(n)
    if n < 1:
        return np.zeros(0)
    if n == 1:
        return np.array([float(d2)])
    n1 = n - 1
    y = d1 + (np.arange(n, dtype=np.float64) * (d2 - d1)) / n1
    y[0], y[-1] = d1, d2
    return y


def _find_reference(dat_dir: Path, dmd: int) -> Path:
    """The reference stack TIFF of one path, searched under ``dat_dir``."""

    def find(pattern):
        """Files below ``dat_dir`` matching a pattern, sorted."""
        return sorted(p for p in dat_dir.rglob(pattern) if p.is_file())

    found = find(f"*DMD{dmd}*_CONFIG1-REFERENCE*") or find(
        f"*DMD{dmd}*-REFERENCE*"
    )
    if not found:
        raise FileNotFoundError(
            f"Could not find reference image within folder: {dat_dir}"
        )
    if len(found) == 1:
        return found[0]
    tifs = find(f"*DMD{dmd}_CONFIG1-REFERENCE*.tif")
    if not tifs:
        raise ValueError(
            "Too many reference stacks found in the specified directory!"
        )
    if len(tifs) > 1:
        raise ValueError(
            "Too many CONFIG1 reference stacks found in the specified "
            "directory!"
        )
    return tifs[0]


def _json_array(value) -> np.ndarray:
    """A JSON number or array as MATLAB's ``jsondecode`` gives it."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def read_reference_stack(path: Union[str, Path]) -> dict:
    """A SLAP2 reference stack, as ``buildTrialTableSLAP2`` stores it.

    Parameters
    ----------
    path : str or Path
        The ``*-REFERENCE*.tif`` file. Each page's ImageDescription is JSON
        with its ``channel``, ``z`` and ``dmdPixel2SampleTransform``.

    Returns
    -------
    dict
        ``IM``: the pages in MATLAB orientation (columns x rows x pages, as
        ScanImageTiffReader returns them; stored in HDF5 as the pages
        themselves) in their native type; ``channels``: the channels in
        order of appearance; ``Zs``: the z of each page of the first page's
        channel; ``dmdPixel2SampleTransform``: from the last page.
    """
    import tifffile

    with tifffile.TiffFile(path) as tf:
        pages = tf.pages
        first = pages[0]
        n = len(pages)
        stack = np.empty((n,) + tuple(first.shape), dtype=first.dtype)
        descs = []
        for k, page in enumerate(pages):
            stack[k] = page.asarray()
            descs.append(page.description)
    channels: List[float] = []
    zs: List[float] = []
    accum = None
    meta = {}
    for k, desc in enumerate(descs):
        meta = json.loads(desc)
        ch = float(meta["channel"])
        if ch not in channels:
            channels.append(ch)
        if k == 0:
            accum = ch
        if ch == accum:
            zs.append(float(meta["z"]))
    if stack.ndim != 3:
        raise ValueError(f"reference stack pages must be 2-D: {path}")
    im = stack[0].T if n == 1 else stack.transpose(2, 1, 0)
    return {
        "IM": im,
        "channels": np.asarray(channels, dtype=np.float64),
        "Zs": np.asarray(zs, dtype=np.float64),
        "dmdPixel2SampleTransform": _json_array(
            meta["dmdPixel2SampleTransform"]
        ),
    }


@dataclass
class _PathFiles:
    """One path's files in acquisition order, with what trials need."""

    names: List[str]
    lines: np.ndarray
    clock: np.ndarray
    mtime: np.ndarray


@dataclass
class _Trials:
    """Analysis trials as rows (one tuple per trial, path 1 then 2)."""

    files: list = field(default_factory=list)
    first: list = field(default_factory=list)
    last: list = field(default_factory=list)
    start: list = field(default_factory=list)
    end: list = field(default_factory=list)

    def add(self, files, first, last, start, end):
        """Append one trial."""
        self.files.append(files)
        self.first.append(first)
        self.last.append(last)
        self.start.append(start)
        self.end.append(end)


def _path_files(names, order, info, mtime) -> _PathFiles:
    """One path's files in the given order with their counts and times."""
    ordered = [names[i] for i in order]
    return _PathFiles(
        ordered,
        np.array([info[n].num_lines for n in ordered], dtype=np.float64),
        np.array([info[n].first_line_time for n in ordered]),
        np.array([mtime[n] for n in ordered]),
    )


def _paths_in_order(names, info, mtime):
    """Split files into the two paths and order them by acquisition.

    Falls back to modification times (with a warning) when the names don't
    give an order, as MATLAB does. Returns both paths and whether the names
    gave the order.
    """
    d1 = [n for n in names if "DMD1" in n]
    d2 = [n for n in names if "DMD1" not in n]
    o1, ok1 = acquisition_order(d1)
    o2, ok2 = acquisition_order(d2)
    names_ok = ok1 and ok2
    if not names_ok:
        warnings.warn(
            "Could not read acquisition times and trial numbers from the "
            ".dat file names in epoch 1; ordering files by modification time "
            "instead, which is wrong for copied files.",
            TrialTableWarning,
            stacklevel=3,
        )
        o1 = np.argsort([mtime[n] for n in d1], kind="stable")
        o2 = np.argsort([mtime[n] for n in d2], kind="stable")
    p1 = _path_files(d1, o1, info, mtime)
    p2 = _path_files(d2, o2, info, mtime)
    return p1, p2, names_ok


def _warn_copy_times(p1, p2, line_period_s, names_ok) -> None:
    """Warn when modification times, used for trial times, look like copies."""
    reason = _copy_time_reason(
        p1.mtime, p1.lines, line_period_s, names_ok
    ) or _copy_time_reason(p2.mtime, p2.lines, line_period_s, names_ok)
    if reason:
        warnings.warn(
            f"File modification times in epoch 1 do not look like acquisition "
            f"times ({reason}). trial_end_time_from_pc and "
            f"trial_start_time_inferred come from these times and are "
            f"probably copy times.",
            TrialTableWarning,
            stacklevel=3,
        )


def _continuous_trials(p1, p2, use_clock, line_period_s) -> _Trials:
    """Cut each continuous recording into trials of at most 200000 lines."""
    if len(p1.names) != len(p2.names):
        raise ValueError(
            "There was an unequal amount of DMD1 and DMD2 files for an epoch "
            "using continuous acquisitions"
        )
    line_days = line_period_s / 86400.0
    trials = _Trials()
    for f in range(len(p1.names)):
        n_tot = min(p1.lines[f], p2.lines[f])
        n_trials = math.ceil(n_tot / MULTI_CYCLE_LINES_PER_TRIAL)
        edges = matlab_linspace(1.0, n_tot + 1.0, n_trials + 1)
        for a, b in zip(edges[:-1], edges[1:]):
            if use_clock:
                start = p1.clock[f] + (a - 1) * line_days
                end = p1.clock[f] + (b - 1) * line_days
            else:
                start = p1.mtime[f] - (p1.lines[f] - a) * line_days
                end = p1.mtime[f] - (p1.lines[f] - b) * line_days
            trials.add((p1.names[f], p2.names[f]), (a, a), (b, b), start, end)
    return trials


def _triggered_trials(p1, p2, use_clock, line_period_s) -> _Trials:
    """Line up one-file-per-trial recordings across the two paths.

    A trial ends where either path's file ends; a path whose file holds more
    lines (its trial number failed to increment) carries the rest over to
    the next trial.
    """
    line_days = line_period_s / 86400.0
    trials = _Trials()
    i1 = i2 = 0
    acc = [0.0, 0.0]
    while i2 < len(p2.names) and i1 < len(p1.names):
        c1 = np.cumsum(p1.lines[i1 : i1 + 5]) - acc[0]
        c2 = np.cumsum(p2.lines[i2 : i2 + 5]) - acc[1]
        if not np.min(np.abs(c1[:, None] - c2[None, :])) < LINE_DIFF_THRESH:
            raise ValueError("Error lining up trials across DMDs!")
        n = min(c1[0], c2[0])
        files = (p1.names[i1], p2.names[i2])
        first = (acc[0] + 1, acc[1] + 1)
        last = (acc[0] + n, acc[1] + n)
        clock_start = p1.clock[i1] if acc[0] == 0 else p2.clock[i2]
        if abs(c1[0] - c2[0]) < LINE_DIFF_THRESH:
            end_mtime = p1.mtime[i1]
            i1, i2, acc = i1 + 1, i2 + 1, [0.0, 0.0]
        elif c1[0] < c2[0]:
            end_mtime = p1.mtime[i1]
            i1, acc = i1 + 1, [0.0, acc[1] + n]
        else:
            end_mtime = p2.mtime[i2]
            i2, acc = i2 + 1, [acc[0] + n, 0.0]
        if use_clock:
            start, end = clock_start, clock_start + n * line_days
        else:
            start, end = end_mtime - n * line_days, end_mtime
        trials.add(files, first, last, start, end)
    left1 = float(p1.lines[i1:].sum() - acc[0])
    left2 = float(p2.lines[i2:].sum() - acc[1])
    if left1 > 0 or left2 > 0:
        warnings.warn(
            f"After trial {len(trials.files)} the paths could not be lined "
            f"up further, so {left1:.0f} lines of DMD1 "
            f"({len(p1.names) - i1} files) and {left2:.0f} lines of DMD2 "
            f"({len(p2.names) - i2} files) are left out of the trial table, "
            f"as in MATLAB.",
            TrialTableWarning,
            stacklevel=3,
        )
    return trials


def _rows(values, n: int, dtype=np.float64) -> np.ndarray:
    """Per-trial pairs (path 1, path 2) as a 2 x n array."""
    return np.array(values, dtype=dtype).reshape(n, 2).T


def _write_tables(table, out_dir, max_duration_s, line_period_s) -> dict:
    """Write trial_table.h5 (and the untrimmed table when trimming)."""
    from ..io.hdf5 import save_struct_h5

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if max_duration_s is None:
        save_struct_h5(out_dir / "trial_table.h5", table)
        return table
    save_struct_h5(out_dir / "trial_table_untrimmed.h5", table)
    trimmed, _ = trim_trial_table(table, max_duration_s, line_period_s)
    save_struct_h5(out_dir / "trial_table.h5", trimmed)
    return trimmed


def build_trial_table_slap2(
    dat_dir: Union[str, Path],
    save_dir: Optional[Union[str, Path]] = None,
    max_duration_s: Optional[float] = None,
    write: bool = True,
) -> dict:
    """Build (and write) the trial table of a SLAP2 session.

    Python counterpart of GIAnT-MATLAB ``buildTrialTableSLAP2(dr, savedr,
    true)``: every ``.dat`` file of ``dat_dir`` is in one epoch, and the
    first file of each CYCLE-split recording stands for all its files.

    Parameters
    ----------
    dat_dir : str or Path
        Folder with the ``.dat`` and ``.meta`` files; the reference stacks
        are searched for anywhere below it. Stored as ``datadr``, as given.
    save_dir : str or Path, optional
        Where ``trial_table.h5`` is written (``savedr``); default
        ``dat_dir``.
    max_duration_s : float, optional
        Keep only the first seconds (see :func:`trim_trial_table`); the
        whole table is then also written as ``trial_table_untrimmed.h5``.
    write : bool
        Write the HDF5 file(s) (default True).

    Returns
    -------
    dict
        The trial table in MATLAB orientation (as ``load_struct_h5`` reads
        it): ``datadr``, ``savedr``, ``filename`` (2 x trials),
        ``true_trial_ix``, ``epoch``, and ``slap2_info`` with ``ref_stack``,
        ``first_line``, ``last_line``, ``trial_end_time_from_pc`` and
        ``trial_start_time_inferred`` (MATLAB datenums).
    """
    dat_dir_str, dat_dir = str(dat_dir), Path(dat_dir)
    save_dir_str = dat_dir_str if save_dir is None else str(save_dir)
    all_names = sorted(p.name for p in dat_dir.glob("*.dat") if p.is_file())
    names = [n for n in all_names if not _later_cycle_file(n)]
    if not names:
        raise FileNotFoundError(f"No .dat files in {dat_dir}")
    ref_stack = {
        f"Path{d}": read_reference_stack(_find_reference(dat_dir, d))
        for d in (1, 2)
    }
    print(f"Loading metadata from {len(names)} DAT files...")
    info = {n: read_dat_info(dat_dir / n) for n in names}
    mtime = {n: _recording_mtime(dat_dir, n, all_names) for n in names}
    print("done loading")
    line_period_s = info[names[0]].line_period_s

    p1, p2, names_ok = _paths_in_order(names, info, mtime)
    use_clock = all(math.isfinite(info[n].first_line_time) for n in names)
    if not use_clock:
        _warn_copy_times(p1, p2, line_period_s, names_ok)
    cycle = ["-CYCLE" in name for name in p1.names + p2.names]
    if any(cycle) and not all(cycle):
        raise ValueError(
            f"{dat_dir} holds both continuous (-CYCLE-) and triggered "
            f"recordings; build a trial table for each kind separately."
        )
    if all(cycle):
        trials = _continuous_trials(p1, p2, use_clock, line_period_s)
    else:
        trials = _triggered_trials(p1, p2, use_clock, line_period_s)

    n = len(trials.files)
    if n == 0:
        raise ValueError(
            f"No trials could be lined up across the two paths "
            f"({len(p1.names)} DMD1 and {len(p2.names)} DMD2 recordings)."
        )
    table = {
        "datadr": dat_dir_str,
        "savedr": save_dir_str,
        "filename": _rows(trials.files, n, dtype=object),
        "true_trial_ix": np.tile(np.arange(1.0, n + 1), (2, 1)),
        "epoch": np.ones((2, n)),
        "slap2_info": {
            "ref_stack": ref_stack,
            "first_line": _rows(trials.first, n),
            "last_line": _rows(trials.last, n),
            "trial_end_time_from_pc": np.array(trials.end).reshape(1, -1),
            "trial_start_time_inferred": np.array(trials.start).reshape(1, -1),
        },
    }
    if write:
        return _write_tables(
            table, save_dir_str, max_duration_s, line_period_s
        )
    if max_duration_s is None:
        return table
    return trim_trial_table(table, max_duration_s, line_period_s)[0]


def trim_trial_table(
    table: dict, max_duration_s: float, line_period_s: float
) -> Tuple[dict, dict]:
    """Keep only the first seconds of a SLAP2 session's trial table.

    As the GIAnT integration-test capsule's ``trimTrialTableSLAP2``: whole
    trials are kept in order while they start before ``max_duration_s``, and
    the last one is shortened so the total is ``max_duration_s``. Raw data
    are not touched; downstream stages read only ``first_line`` to
    ``last_line``.

    Parameters
    ----------
    table : dict
        Trial table in MATLAB orientation (from
        :func:`build_trial_table_slap2` or ``load_struct_h5``).
    max_duration_s : float
        Seconds to keep.
    line_period_s : float
        Line period of the recording.

    Returns
    -------
    trimmed : dict
        A new table (the reference stacks are shared, not copied).
    info : dict
        ``n_trials_untrimmed``, ``duration_untrimmed_s``, ``n_trials_kept``,
        ``duration_kept_s`` and ``last_trial_shortened``.
    """
    s2 = table["slap2_info"]
    first = np.asarray(s2["first_line"], dtype=np.float64).reshape(2, -1)
    last = np.asarray(s2["last_line"], dtype=np.float64).reshape(2, -1).copy()
    n_lines = last - first + 1
    trial_dur_s = n_lines.max(axis=0) * line_period_s
    n_in = trial_dur_s.size
    start_s = np.concatenate([[0.0], np.cumsum(trial_dur_s[:-1])])
    keep = np.flatnonzero(start_s < max_duration_s)
    if keep.size == 0:
        raise ValueError(f"No trials start before {max_duration_s:g} s.")
    k = keep[-1]
    remaining_s = max_duration_s - start_s[k]
    shortened = bool(remaining_s < trial_dur_s[k])
    if shortened:
        keep_lines = max(1, math.floor(remaining_s / line_period_s))
        last[:, k] = np.minimum(last[:, k], first[:, k] + keep_lines - 1)

    def cols(value):
        """The kept trials (columns) of a per-path, per-trial array."""
        arr = np.asarray(value)
        return arr.reshape(2, -1)[:, keep] if arr.size == 2 * n_in else arr

    out = dict(table)
    out["filename"] = np.asarray(table["filename"], dtype=object).reshape(
        2, -1
    )[:, keep]
    out["true_trial_ix"] = cols(table["true_trial_ix"])
    out["epoch"] = cols(table["epoch"])
    new_s2 = dict(s2)
    new_s2["first_line"] = first[:, keep]
    new_s2["last_line"] = last[:, keep]
    for key in ("trial_end_time_from_pc", "trial_start_time_inferred"):
        if key in s2 and np.asarray(s2[key]).size == n_in:
            new_s2[key] = np.asarray(s2[key]).reshape(1, -1)[:, keep]
    out["slap2_info"] = new_s2
    kept = new_s2["last_line"] - new_s2["first_line"] + 1
    info = {
        "n_trials_untrimmed": int(n_in),
        "duration_untrimmed_s": float(trial_dur_s.sum()),
        "n_trials_kept": int(keep.size),
        "duration_kept_s": float(kept.max(axis=0).sum() * line_period_s),
        "last_trial_shortened": shortened,
    }
    print(
        f"Trimmed trial table: kept {info['n_trials_kept']} of {n_in} trials, "
        f"{info['duration_kept_s']:.1f} of "
        f"{info['duration_untrimmed_s']:.1f} s"
    )
    return out, info
