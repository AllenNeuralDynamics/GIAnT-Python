"""Solved subproblems of one trial-path, in a file: the solve stage's output.

A results file holds the solver's outputs for some or all subproblems of
one container (one trial-path), in the solver's normalized units, and the
trial's record that the collect stage needs besides them (frame lines,
discards, global and user-ROI traces). With it, collecting needs neither the
containers nor the raw data. Several files may cover one container (a
shard per range of its subproblems); :func:`trial_outcome` assembles a
trial from all of them and checks that each subproblem is there once.

File layout (HDF5, row-major, zero-based indices)::

    /                       attributes: format, format_version,
                            code_version, settings_hash (of the solve
                            stage and the subproblem range),
                            solve_settings (the same without the range:
                            equal in every file of one run), container,
                            container_hash, path,
                            trial (both one-based), n_subproblems (of the
                            container), n_pixels, n_sources, photon_scale,
                            extracted (0 for a trial too short to extract)
    /trial/frame_lines      (frames,) one-based lines
    /trial/discard          (frames,) uint8
    /trial/global_f         (channels, frames), file channel order
    /trial/roi_f, roi_fsvd  (rois, channels, frames), analysis order
    /subproblems/index      (k,) the container's subproblems in this file
    /subproblems/<k>/pixels, sources
    /subproblems/<k>/footprints   (pixels, sources)
    /subproblems/<k>/events, ls, f0   (channels, sources, frames)
    /subproblems/<k>/snr    (sources,)
    /subproblems/<k>        attribute info (JSON: each fit's iterations,
                            loss, convergence and seconds)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import h5py
import numpy as np

from ... import __version__
from .solver import (
    SubproblemResult,
    TrialActivity,
    assemble_results,
    process_result,
)
from .solver_setup import SolverParams, Subproblem

FORMAT = "giant-silo-results"
FORMAT_VERSION = 1


@dataclass
class TrialRecord:
    """What collecting needs of a loaded trial besides its sources.

    Attributes
    ----------
    frame_lines : ndarray of int, shape (frames,)
    discard : ndarray of bool, shape (frames,)
    global_f : ndarray of shape (channels, frames)
        File channel order (MATLAB's ``CD.global.F``).
    roi_f, roi_fsvd : ndarray of shape (rois, channels, frames)
        Analysis channel order (``CD.ROIs.F``, ``Fsvd``).
    """

    frame_lines: np.ndarray
    discard: np.ndarray
    global_f: np.ndarray
    roi_f: np.ndarray
    roi_fsvd: np.ndarray

    @property
    def n_frames(self) -> int:
        """Frames of the trial."""
        return int(np.asarray(self.frame_lines).size)

    @classmethod
    def from_trial(cls, trial) -> "TrialRecord":
        """From a loaded trial or a container (``TrialData``,
        ``Container``)."""
        return cls(
            frame_lines=np.asarray(trial.frame_lines),
            discard=np.asarray(trial.discard, dtype=bool),
            global_f=np.asarray(trial.global_f),
            roi_f=np.asarray(trial.roi_f),
            roi_fsvd=np.asarray(trial.roi_fsvd),
        )


@dataclass
class TrialOutcome:
    """One trial-path's extraction (MATLAB's ``E{t}``).

    Attributes
    ----------
    record : TrialRecord
    activity : TrialActivity or None
        The sources' footprints and traces (movie units, discards blanked);
        None when the trial was loaded but not extracted (too short).
    """

    record: TrialRecord
    activity: Optional[TrialActivity] = None


@dataclass
class ResultsFile:
    """The content of one results file.

    Attributes
    ----------
    record : TrialRecord
    results : dict
        Subproblem index (in its container) to ``(pixels, sources,
        SubproblemResult)``, or to None when read without data.
    attrs : dict
        The file's attributes.
    """

    record: TrialRecord
    results: dict
    attrs: dict


def write_results(
    path,
    record: TrialRecord,
    results: dict,
    attrs: dict,
) -> Path:
    """Write a results file (see the module docstring).

    The file is written under a temporary name and moved into place when
    complete.

    Parameters
    ----------
    path : str or Path
        Destination (replaced).
    record : TrialRecord
        The trial's record.
    results : dict
        Subproblem index to ``(pixels, sources, SubproblemResult)``.
    attrs : dict
        Attributes (``container``, ``path``, ``trial``, ``n_subproblems``,
        ``n_pixels``, ``n_sources``, ``photon_scale``, ``extracted``,
        ``settings_hash``, ...); ``format``, ``format_version`` and
        ``code_version`` are added.

    Returns
    -------
    Path
    """
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.partial")
    try:
        with h5py.File(tmp, "w") as f:
            f.attrs["format"] = FORMAT
            f.attrs["format_version"] = FORMAT_VERSION
            f.attrs["code_version"] = __version__
            for key, value in attrs.items():
                f.attrs[key] = value
            t = f.create_group("trial")
            t["frame_lines"] = np.asarray(record.frame_lines, np.int64)
            t["discard"] = np.asarray(record.discard, np.uint8)
            t["global_f"] = np.asarray(record.global_f, np.float64)
            t["roi_f"] = np.asarray(record.roi_f, np.float64)
            t["roi_fsvd"] = np.asarray(record.roi_fsvd, np.float64)
            s = f.create_group("subproblems")
            index = np.array(sorted(results), dtype=np.int64)
            s["index"] = index
            for k in index:
                pixels, sources, r = results[int(k)]
                g = s.create_group(str(int(k)))
                g["pixels"] = np.asarray(pixels, np.int64)
                g["sources"] = np.asarray(sources, np.int64)
                g["footprints"] = np.asarray(r.footprints)
                g["events"] = np.asarray(r.events)
                g["ls"] = np.asarray(r.ls)
                g["f0"] = np.asarray(r.f0)
                g["snr"] = np.asarray(r.snr)
                g.attrs["info"] = json.dumps(r.info, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    return path


def _attr(value):
    """An HDF5 attribute as a Python value."""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_results(path, data: bool = True) -> ResultsFile:
    """Read a results file.

    Parameters
    ----------
    path : str or Path
        The file.
    data : bool
        Read the subproblems' results too (otherwise only the record,
        the attributes and the subproblems' indices: ``results`` maps each
        index to None).

    Returns
    -------
    ResultsFile

    Raises
    ------
    ValueError
        If the file is not a results file of a version this code reads.
    """
    with h5py.File(path, "r") as f:
        if _attr(f.attrs.get("format")) != FORMAT:
            raise ValueError(f"{path} is not a SILo results file")
        if int(f.attrs["format_version"]) > FORMAT_VERSION:
            raise ValueError(
                f"{path} has results format {int(f.attrs['format_version'])}"
                f"; this code reads up to {FORMAT_VERSION}"
            )
        attrs = {k: _attr(v) for k, v in f.attrs.items()}
        t = f["trial"]
        record = TrialRecord(
            frame_lines=t["frame_lines"][()],
            discard=t["discard"][()].astype(bool),
            global_f=t["global_f"][()],
            roi_f=t["roi_f"][()],
            roi_fsvd=t["roi_fsvd"][()],
        )
        results = {}
        for k in f["subproblems/index"][()]:
            if not data:
                results[int(k)] = None
                continue
            g = f[f"subproblems/{int(k)}"]
            results[int(k)] = (
                g["pixels"][()],
                g["sources"][()],
                SubproblemResult(
                    footprints=g["footprints"][()],
                    events=g["events"][()],
                    ls=g["ls"][()],
                    f0=g["f0"][()],
                    snr=g["snr"][()],
                    info=json.loads(g.attrs["info"]),
                ),
            )
        return ResultsFile(record=record, results=results, attrs=attrs)


def trial_outcome(
    files: Sequence[ResultsFile], params: SolverParams
) -> TrialOutcome:
    """Assemble one trial-path from the results files that cover it.

    Each of the container's subproblems must be in exactly one file (or in
    several with equal placement and results), and all files must have
    been solved with the same settings (``solve_settings``). Footprints
    and traces are assembled (``assembleResults``) and processed
    (``processResult``) as MATLAB does after the trial's subproblems are
    solved.

    Parameters
    ----------
    files : sequence of ResultsFile
        All files of one container, read with their data.
    params : SolverParams
        The solver's parameters (kernels of the denoised traces).

    Returns
    -------
    TrialOutcome

    Raises
    ------
    ValueError
        If the files belong to different containers, or a subproblem is
        missing or has different results in two files.
    """
    if not files:
        raise ValueError("no results files for the trial")
    first = files[0].attrs
    record = files[0].record
    merged = _merge(files)
    if not int(first.get("extracted", 1)):
        return TrialOutcome(record=record)
    n_sub = int(first["n_subproblems"])
    if n_sub == 0:
        return TrialOutcome(record=record)
    subs, results = [], []
    for k in range(n_sub):
        pixels, sources, r = merged[k]
        subs.append(_Placement(pixels, sources))
        results.append(r)
    trial = assemble_results(
        subs,
        results,
        int(first["n_pixels"]),
        int(first["n_sources"]),
        float(first["photon_scale"]),
    )
    return TrialOutcome(
        record=record,
        activity=process_result(trial, params, record.discard),
    )


def _merge(files: Sequence[ResultsFile]) -> dict:
    """The subproblems of one container's results files, checked."""
    first = files[0].attrs
    for f in files[1:]:
        for key in ("container", "container_hash", "n_subproblems"):
            if f.attrs.get(key) != first.get(key):
                raise ValueError(
                    f"results of different containers ({key}): "
                    f"{first.get(key)} and {f.attrs.get(key)}"
                )
        if _solve_settings(f.attrs) != _solve_settings(first):
            raise ValueError(
                f"results of {first.get('container')} were solved with "
                "different settings"
            )
    merged: dict = {}
    for f in files:
        for k, item in f.results.items():
            if k in merged and not _same(merged[k], item):
                raise ValueError(f"subproblem {k} has two different results")
            merged[k] = item
    n_sub = int(first["n_subproblems"])
    missing = sorted(set(range(n_sub)) - set(merged))
    if missing and int(first.get("extracted", 1)):
        raise ValueError(
            f"{len(missing)} of {n_sub} subproblems of "
            f"{first.get('container')} have no results (first: {missing[0]})"
        )
    return merged


@dataclass
class _Placement:
    """Where a subproblem's results go (what ``assemble_results`` reads of
    a :class:`Subproblem`)."""

    pixels: np.ndarray
    sources: np.ndarray


def _solve_settings(attrs: dict):
    """The hash of how a results file was solved, without its subproblem
    range (``solve_settings``; None if the file does not record it)."""
    return attrs.get("solve_settings")


def _same(a: tuple, b: tuple) -> bool:
    """Whether two entries ``(pixels, sources, result)`` of one subproblem
    are equal: placement and results (NaN equal)."""
    if not (np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])):
        return False
    return all(
        np.array_equal(getattr(a[2], n), getattr(b[2], n), equal_nan=True)
        for n in ("footprints", "events", "ls", "f0", "snr")
    )


def subproblem_results(
    subs: Sequence[Subproblem],
    results: Sequence[SubproblemResult],
    indices: Optional[Sequence[int]] = None,
) -> dict:
    """Solved subproblems in the form :func:`write_results` takes.

    Parameters
    ----------
    subs : sequence of Subproblem
        The subproblems (with ``pixels`` and ``sources``).
    results : sequence of SubproblemResult
        Their results.
    indices : sequence of int, optional
        Their indices in the container (default: 0, 1, ...).

    Returns
    -------
    dict
        Index to ``(pixels, sources, result)``.
    """
    if indices is None:
        indices = range(len(subs))
    if not len(subs) == len(results) == len(indices):
        raise ValueError("one result and one index are needed per subproblem")
    return {
        int(k): (s.pixels, s.sources, r)
        for k, s, r in zip(indices, subs, results)
    }
