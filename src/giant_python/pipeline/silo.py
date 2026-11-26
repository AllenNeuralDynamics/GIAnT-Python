"""SILo for SLAP2 multi-ROI raster recordings: stages and the streaming driver.

GIAnT-MATLAB's ``SILo.m`` (main 5a0673b) as stages, each a function from
files to files that runs for any subset of its units and skips units whose
outputs exist and were made with the same settings:

=========  ==========  ===========================  ========================
stage      unit        reads                        writes
=========  ==========  ===========================  ========================
localize   trial-path  registered movie, alignment  ``localize/*.h5``
plan       path        localization files,          ``plan/path<d>.h5``
                       annotations, seeds
prepare    trial-path  raw ``.dat``, plan           ``containers/*.h5``
dispatch   session     containers                   ``jobs/job_*.json``
solve      job         container(s), job file       ``results/*.h5``
collect    session     plans, results, localization ``experiment_summary.h5``,
                       and alignment files          ``per_trial_summary.h5``,
                                                    trial table fields
=========  ==========  ===========================  ========================

:func:`run_silo` runs them on one machine the way MATLAB does: per path,
trials are loaded one ahead of the solver and their subproblems solved in a
local process pool; no containers are written unless asked for. Every way of
running gives the same results files, and :func:`collect` makes the
summaries from them.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import math
import os
import threading
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import h5py
import numpy as np

from ..models.params import (
    SiloExecution,
    SiloParams,
    resolve_silo_options,
)
from .silo_session import (
    NO_DATA,
    SiloSession,
    recording_ids,
    attr as _attr,
    file_id as _file_id,
    make_pool,
    run_jobs,
    stage_hash as _hash,
    skip_hash,
    write_skip,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- localize


def _localize_hash(session: SiloSession, params: SiloParams, d, t, baseline):
    """Settings hash of one localization."""
    return _hash(
        {
            "stage": "localize",
            "settings": params.stage_settings("localize"),
            "movie": _file_id(session.movie(d, t)),
            "alignment": _file_id(session.alignment(d, t)),
            "baseline": baseline,
        }
    )


def _baseline_file(baseline_dir, movie: Path) -> Optional[Path]:
    """The network baseline of a registered movie."""
    if baseline_dir is None:
        return None
    stem = movie.name[: movie.name.index("_REGISTERED")]
    return Path(baseline_dir) / f"{stem}_BASELINE-80Hz.h5"


def _localize_job(job: dict) -> dict:
    """Localize one trial-path and write its file (a pool task)."""
    from ..extraction.raster.peaks import get_act_im_peaks
    from ..extraction.raster.stage_files import write_localization
    from ..extraction.raster.trial import localize_trial

    t0 = time.perf_counter()
    p = SiloParams(**job["params"])
    loc, trial = localize_trial(
        job["moco_dir"],
        job["movie"],
        job["num_channels"],
        p.localization_params(job["align_hz"]),
        baseline_dir=job["baseline_dir"],
        activity_channel=p.activity_channel,
        motion_thresh=p.motion_thresh,
        discard_initial_s=p.discard_initial_s,
        find_peaks=False,
    )
    t1 = time.perf_counter()
    peaks = None
    if p.trial_peaks and job["fit_peaks"]:
        loc_p = p.localization_params(job["align_hz"])
        peaks = get_act_im_peaks(
            loc.summary,
            loc_p.peak_th,
            None,
            loc_p.min_peak_distance,
            p.seed_fit,
            p.seed_merge_px,
        )
    write_localization(
        job["out"],
        trial.mean_im,
        loc.act_im,
        loc.summary,
        trial.discard,
        peaks,
        job["attrs"],
    )
    return {
        "unit": job["unit"],
        "seconds": time.perf_counter() - t0,
        "peak_seconds": time.perf_counter() - t1 if peaks is not None else 0,
    }


def _peaks_job(job: dict) -> dict:
    """Fit one localized trial's peaks and add them to its file."""
    from ..extraction.raster.peaks import get_act_im_peaks
    from ..extraction.raster.stage_files import add_peaks, read_localization

    t0 = time.perf_counter()
    rec = read_localization(job["file"])
    peaks = get_act_im_peaks(
        rec.summary,
        job["peak_th"],
        None,
        job["min_peak_distance"],
        job["seed_fit"],
        job["seed_merge_px"],
    )
    add_peaks(job["file"], peaks)
    return {"unit": job["unit"], "seconds": time.perf_counter() - t0}


def _units(session: SiloSession, units, paths=None) -> list:
    """Kept trial-paths ``(d, t)``, zero-based, of ``units`` or all."""
    if units is None:
        units = [
            (d, t)
            for d in range(session.n_paths)
            for t in range(session.n_trials)
        ]
    out = []
    for d, t in units:
        if paths is not None and d not in paths:
            continue
        if session.keep[d, t]:
            out.append((int(d), int(t)))
    return out


def localize_jobs(
    session: SiloSession,
    params: SiloParams,
    units=None,
    baseline_dir=None,
    fit_peaks: bool = True,
    overwrite: bool = False,
) -> list:
    """The localization tasks still to do (stage ``localize``).

    Parameters
    ----------
    session : SiloSession
    params : SiloParams
    units : iterable of (int, int), optional
        Zero-based ``(path, trial)`` pairs (default: every kept one).
    baseline_dir : str or Path, optional
        Folder of the network baseline files (``localization_baseline =
        'network'``).
    fit_peaks : bool
        Fit the per-trial peaks in the same task (otherwise
        :func:`peaks_jobs` does it later).
    overwrite : bool
        Redo units whose files were made with the same settings.

    Returns
    -------
    list of dict
        Tasks for :func:`_localize_job`.
    """
    network = params.localization_baseline == "network"
    if network and baseline_dir is None:
        raise ValueError(
            "localization_baseline 'network' needs the folder of the "
            "*_BASELINE-80Hz.h5 files"
        )
    jobs = []
    for d, t in _units(session, units):
        movie = session.movie(d, t)
        base = _baseline_file(baseline_dir, movie) if network else None
        if base is not None and not base.exists():
            raise FileNotFoundError(f"no network baseline: {base}")
        h = _localize_hash(
            session, params, d, t, None if base is None else _file_id(base)
        )
        out = session.localize_file(d, t)
        if not overwrite and _attr(out, "settings_hash") == h:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        jobs.append(
            {
                "unit": (d, t),
                "params": asdict(params),
                "moco_dir": str(session.moco_dir),
                "movie": movie.name,
                "num_channels": session.num_channels,
                "align_hz": session.align_hz,
                "baseline_dir": (
                    str(baseline_dir) if base is not None else None
                ),
                "fit_peaks": fit_peaks,
                "out": str(out),
                "attrs": {
                    "settings_hash": h,
                    "movie": movie.name,
                    "path": d + 1,
                    "trial": t + 1,
                    "align_hz": session.align_hz,
                    "num_channels": session.num_channels,
                },
            }
        )
    return jobs


def peaks_jobs(session: SiloSession, params: SiloParams, units=None) -> list:
    """Peak fits still to do: localization files without peaks (in the
    file or its ``.peaks.h5`` sidecar, where :func:`_peaks_job` writes
    them so that the localization file is not written while the plan stage
    reads it)."""
    from ..extraction.raster.stage_files import localization_attrs

    if not params.trial_peaks:
        return []
    loc = params.localization_params(session.align_hz)
    jobs = []
    for d, t in _units(session, units):
        f = session.localize_file(d, t)
        attrs = localization_attrs(f)
        if attrs is not None and not int(attrs.get("peaks_fitted", 0)):
            jobs.append(
                {
                    "unit": (d, t),
                    "file": str(f),
                    "peak_th": loc.peak_th,
                    "min_peak_distance": loc.min_peak_distance,
                    "seed_fit": params.seed_fit,
                    "seed_merge_px": params.seed_merge_px,
                }
            )
    return jobs


def localize(
    session: SiloSession,
    params: SiloParams,
    execution: Optional[SiloExecution] = None,
    units=None,
    baseline_dir=None,
) -> list:
    """Stage ``localize``: one localization file per trial-path.

    Units whose files exist with the same settings are kept, but their
    per-trial peaks are fitted if they are missing.

    Returns
    -------
    list of dict
        Per localized unit (and per peak fit): ``unit`` and ``seconds``.
    """
    ex = execution or SiloExecution()
    jobs = localize_jobs(
        session, params, units, baseline_dir, True, ex.overwrite
    )
    threads = max(1, (os.cpu_count() or 1) // max(1, ex.workers))
    done = run_jobs(_localize_job, jobs, ex.workers, threads)
    # peaks still missing (e.g. a streaming run stopped before its
    # background fits ended): fit them now
    done += run_jobs(
        _peaks_job, peaks_jobs(session, params, units), ex.workers, threads
    )
    return done


# ---------------------------------------------------------------- plan


def _plan_inputs(session: SiloSession, d: int):
    """The localization records of path ``d`` (None for trials without)."""
    from ..extraction.raster.stage_files import read_localization

    recs = []
    for t in range(session.n_trials):
        f = session.localize_file(d, t)
        recs.append(
            read_localization(f, peaks=False) if session.keep[d, t] else None
        )
    return recs


def _plan_hash(session, params, d, seeds, annotations) -> str:
    """Settings hash of a path's plan."""
    locs = {}
    for t in range(session.n_trials):
        if session.keep[d, t]:
            f = session.localize_file(d, t)
            locs[t] = _attr(f, "settings_hash")
            if locs[t] is None:
                raise FileNotFoundError(f"not localized: {f}")
    return _hash(
        {
            "stage": "plan",
            "settings": params.stage_settings("plan"),
            "localization": locs,
            "seeds": None if seeds is None else np.asarray(seeds).tolist(),
            "annotations": annotations,
        }
    )


def write_user_rois(path: Path, labels: Sequence[str], masks) -> None:
    """Add the user ROIs in the consensus frame to a plan file
    (``/user_rois``: ``labels`` JSON attribute, ``masks`` (rois, rows,
    cols) uint8)."""
    with h5py.File(path, "a") as f:
        if "user_rois" in f:
            del f["user_rois"]
        g = f.create_group("user_rois")
        g.attrs["labels"] = json.dumps(list(labels))
        m = np.asarray(masks, dtype=np.uint8)
        g.create_dataset(
            "masks",
            data=m,
            **(dict(compression="gzip", compression_opts=4) if m.size else {}),
        )


def read_user_rois(path: Path) -> tuple[list, list]:
    """The user ROIs of a plan file: labels and masks."""
    with h5py.File(path, "r") as f:
        if "user_rois" not in f:
            return [], []
        g = f["user_rois"]
        labels = json.loads(g.attrs["labels"])
        masks = [m.astype(bool) for m in g["masks"][()]]
    return labels, masks


def plan_one(
    session: SiloSession,
    params: SiloParams,
    d: int,
    seeds: Optional[np.ndarray] = None,
    overwrite: bool = False,
) -> Optional[Path]:
    """Stage ``plan`` for path ``d``: ``plan/path<d>.h5``.

    Parameters
    ----------
    session : SiloSession
    params : SiloParams
    d : int
        Zero-based path.
    seeds : ndarray of shape (n, 4), optional
        Zero-based ``[amplitude, row, col, sigma]`` used instead of the
        plan's own seeding (MATLAB's ``thetaf``, for parity runs).
    overwrite : bool
        Redo a plan made with the same settings.

    Returns
    -------
    Path or None
        The plan file; None for a path without kept trials.
    """
    from ..extraction.raster.annotations import (
        annotation_trial,
        rois_to_consensus,
        session_annotations,
    )
    from ..extraction.raster.plan import plan_path, write_plan

    if not session.keep[d].any():
        warnings.warn(f"path {d + 1}: no trial passed file verification")
        return None
    ann = None
    ann_id = None
    if params.user_rois and session.annotations is not None:
        ann = session_annotations(session.annotations, session.n_paths)[d]
        ann_id = _file_id(session.annotations)
    h = _plan_hash(session, params, d, seeds, ann_id)
    out = session.plan_file(d)
    if not overwrite and _attr(out, "settings_hash") == h:
        return out
    recs = _plan_inputs(session, d)
    trials = [None if r is None else r.trial_images() for r in recs]
    soma, soma_trial = None, None
    if ann is not None and ann.rois:
        soma_trial = annotation_trial(ann.fn, session.fn_reg_ds[d])
        somas = [r.mask for r in ann.rois if r.is_soma and r.mask.size]
        if somas:
            shape = np.max([m.shape for m in somas], axis=0)
            soma = np.zeros(tuple(shape), bool)
            for m in somas:
                soma[: m.shape[0], : m.shape[1]] |= m
    plan = plan_path(
        trials,
        soma_mask=soma,
        params=params.plan_params(),
        seeds=seeds,
        soma_mask_trial=soma_trial,
    )
    rois = (
        rois_to_consensus(ann, session.fn_reg_ds[d], plan.warp, plan.shape)
        if ann is not None
        else []
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.{os.getpid()}.partial")
    try:
        write_plan(tmp, plan)
        write_user_rois(
            tmp,
            [r.label for r in rois],
            np.array([r.mask for r in rois]).reshape(
                (len(rois),) + plan.shape
            ),
        )
        with h5py.File(tmp, "a") as f:
            f.attrs["settings_hash"] = h
            f.attrs["path"] = d + 1
            f.attrs["z_depth"] = _z_depth(session, d)
        os.replace(tmp, out)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    return out


def _z_depth(session: SiloSession, d: int) -> float:
    """``Z_depths`` of path ``d`` (first kept trial's alignment file)."""
    from ..extraction.raster.summary import read_alignment_record

    kept = np.flatnonzero(session.keep[d])
    if not kept.size:
        return math.nan
    return read_alignment_record(session.alignment(d, int(kept[0]))).z_depth


def _plan_job(job: dict):
    """:func:`plan_one` as a pool task."""
    session = job["session"]
    t0 = time.perf_counter()
    out = plan_one(
        session,
        SiloParams(**job["params"]),
        job["d"],
        job["seeds"],
        job["overwrite"],
    )
    return {"path": job["d"], "file": out, "seconds": time.perf_counter() - t0}


def plan(
    session: SiloSession,
    params: SiloParams,
    execution: Optional[SiloExecution] = None,
    paths: Optional[Iterable[int]] = None,
    seeds: Optional[list] = None,
) -> list:
    """Stage ``plan`` for several paths (in parallel, one process each).

    Parameters
    ----------
    paths : iterable of int, optional
        Zero-based paths (default: all).
    seeds : list of (ndarray or None), optional
        Per path, seeds to use instead of seeding (``read_seeds``).

    Returns
    -------
    list of dict
        ``path``, ``file`` and ``seconds`` of each path.
    """
    ex = execution or SiloExecution()
    paths = list(range(session.n_paths)) if paths is None else list(paths)
    jobs = [
        {
            "session": session,
            "params": asdict(params),
            "d": d,
            "seeds": None if seeds is None else seeds[d],
            "overwrite": ex.overwrite,
        }
        for d in paths
    ]
    threads = max(1, (os.cpu_count() or 1) // max(1, len(jobs)))
    return run_jobs(_plan_job, jobs, min(ex.workers, len(jobs)), threads)


# ---------------------------------------------------------------- prepare


def load_trial_for(
    session: SiloSession, params: SiloParams, plan, d, t, rois, device="cpu"
):
    """Trial ``t`` of path ``d`` at the plan's selected pixels.

    Returns
    -------
    TrialData
    """
    from ..extraction.raster.plan import loader_warp
    from ..extraction.raster.trial_data import load_trial

    return load_trial(
        session.raw(d, t),
        session.first_line[d, t],
        session.last_line[d, t],
        session.alignment(d, t),
        plan.sel,
        loader_warp(plan.warp, t),
        plan.trial_discard[t],
        plan.mean_aligned[t],
        roi_masks=rois,
        params=params.load_params(),
        device=device,
    )


def _prepare_hash(
    session: SiloSession, params: SiloParams, plan_hash: str, d, t
) -> str:
    """Settings hash of a container: the prepare settings, the plan, the
    raw recording's files, the trial's lines and its alignment file."""
    return _hash(
        {
            "stage": "prepare",
            "settings": params.stage_settings("prepare"),
            "plan": plan_hash,
            "unit": [d, t],
            "raw": recording_ids(session.raw(d, t)),
            "lines": [
                float(session.first_line[d, t]),
                float(session.last_line[d, t]),
            ],
            "alignment": _file_id(session.alignment(d, t)),
        }
    )


def prepare(
    session: SiloSession,
    params: SiloParams,
    execution: Optional[SiloExecution] = None,
    units=None,
) -> list:
    """Stage ``prepare``: a container per extracted trial-path.

    The valid trials of each planned path are loaded at the plan's
    selected pixels (200 Hz, from the raw files) and written with the
    plan's subproblems. A trial in which no selected pixel has data gets a
    ``*.skip.json`` marker instead.

    Parameters
    ----------
    units : iterable of (int, int), optional
        Zero-based ``(path, trial)`` pairs (default: all valid trials).

    Returns
    -------
    list of dict
        Per unit: ``unit``, ``file`` and ``seconds``.
    """
    from ..extraction.raster.container import write_container
    from ..extraction.raster.plan import read_plan

    ex = execution or SiloExecution()
    done = []
    plans = {}
    for d, t in _valid_units(session, units):
        if d not in plans:
            f = session.plan_file(d)
            plans[d] = (
                read_plan(f),
                read_user_rois(f)[1],
                _attr(f, "settings_hash"),
            )
        p, rois, ph = plans[d]
        h = _prepare_hash(session, params, ph, d, t)
        out = session.container_file(d, t)
        skip = session.skip_file(d, t)
        if not ex.overwrite and (
            _attr(out, "settings_hash") == h or skip_hash(skip) == h
        ):
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        try:
            trial = load_trial_for(
                session, params, p, d, t, rois, ex.loader_device
            )
        except ValueError as exc:
            if NO_DATA not in str(exc):
                raise
            _mark_skipped(session, d, t, str(exc), h, ph)
            continue
        if skip.exists():
            skip.unlink()
        write_container(
            out,
            trial,
            np.column_stack([p.source_rows, p.source_cols]),
            [g.pixels for g in p.groups],
            [g.sources for g in p.groups],
            p.shape,
            params=_container_params(params, ph, d, t),
            source={
                "raw": session.raw(d, t).name,
                "path": d + 1,
                "trial": t + 1,
            },
        )
        # the hash of the settings (not of the params JSON, see settings_hash)
        with h5py.File(out, "a") as f:
            f.attrs["settings_hash"] = h
        done.append(
            {"unit": (d, t), "file": out, "seconds": time.perf_counter() - t0}
        )
    return done


def _mark_skipped(session, d, t, reason, prepare_hash, plan_hash) -> None:
    """Mark a trial-path as without data; remove its earlier container and
    results files, so that no stage uses them."""
    write_skip(session.skip_file(d, t), reason, prepare_hash, plan_hash)
    stale = [session.container_file(d, t)]
    stem = f"path{d + 1}_trial{t + 1:04d}"
    if session.results_dir.exists():
        stale += [
            p
            for p in session.results_dir.glob(f"{stem}*.h5")
            if p.stem == stem or p.stem.startswith(stem + "_s")
        ]
    for p in stale:
        if p.exists():
            p.unlink()


def _container_params(
    params: SiloParams, plan_hash: str, d: int, t: int
) -> dict:
    """What a container records of its settings."""
    return {
        "prepare": params.stage_settings("prepare"),
        "plan": plan_hash,
        "path": d + 1,
        "trial": t + 1,
    }


def _valid_units(session: SiloSession, units=None) -> list:
    """Planned paths' valid trials ``(d, t)`` (of ``units`` or all)."""
    wanted = None if units is None else {(int(d), int(t)) for d, t in units}
    out = []
    for d in range(session.n_paths):
        f = session.plan_file(d)
        if not f.exists():
            continue
        with h5py.File(f, "r") as h:
            valid = h["alignment/valid_trials"][()].astype(int)
        for t in valid:
            if wanted is None or (d, int(t)) in wanted:
                out.append((d, int(t)))
    return out


# ---------------------------------------------------------------- streaming


@dataclass
class _Pending:
    """A loaded trial whose subproblems are being solved."""

    d: int
    t: int
    record: object
    placements: list
    futures: list
    results: list
    attrs: dict
    out: Path
    submitted: float = 0.0
    done: Optional[_Done] = None


class _Done:
    """Completion times of a trial's subproblems, from done callbacks."""

    def __init__(self, n: int) -> None:
        """Expect ``n`` completions."""
        self.n, self.times = n, []
        self.cond = threading.Condition()

    def add(self, _future=None) -> None:
        """Record one completion now (a done callback)."""
        with self.cond:
            self.times.append(time.perf_counter())
            self.cond.notify_all()

    def last(self, timeout: float = 60.0):
        """The last completion time once all ``n`` are in (a future's
        result can be read before its callbacks have run), or None."""
        with self.cond:
            self.cond.wait_for(lambda: len(self.times) >= self.n, timeout)
            return max(self.times) if self.times else None


def _track(futures: list) -> _Done:
    """Track when each future is done; for no futures (solved here), the
    solve is done now."""
    done = _Done(max(1, len(futures)))
    if not futures:
        done.add()
    for f in futures:
        f.add_done_callback(done.add)
    return done


def _finish(p: _Pending) -> float:
    """Wait for a trial's subproblems and write its results file."""
    from ..extraction.raster.results import write_results

    t0 = time.perf_counter()
    results = p.results or [f.result() for f in p.futures]
    solved = {
        k: (pix, src, r)
        for k, ((pix, src), r) in enumerate(zip(p.placements, results))
    }
    write_results(p.out, p.record, solved, p.attrs)
    return time.perf_counter() - t0


@dataclass
class _ContainerView:
    """What :func:`~.silo_solve.results_attrs` reads of a container, from a
    loaded trial."""

    trial: object
    plan: object
    d: int
    t: int
    plan_hash: str = ""

    @property
    def pixel_rc(self):
        """The selected pixels."""
        return self.trial.pixel_rc

    @property
    def sources_rc(self):
        """The path's sources."""
        return np.column_stack([self.plan.source_rows, self.plan.source_cols])

    @property
    def photon_scale(self):
        """Counts per unit of the data."""
        return self.trial.photon_scale

    @property
    def source(self):
        """Path and trial, one-based."""
        return {"path": self.d + 1, "trial": self.t + 1}

    @property
    def params(self):
        """What the container would record: its plan's hash."""
        return {"plan": self.plan_hash}


def _submit(subs, sp, options, pool):
    """Solve a trial's subproblems: futures from the pool (largest first),
    or results solved here."""
    from ..extraction.raster.solver import solve_subproblems
    from .silo_solve import _solve_one, effective_batch

    if not subs:
        return [], []
    if pool is None or effective_batch(options) is not None:
        return [], solve_subproblems(subs, sp, options)
    order = np.argsort(
        [-s.n_pixels * s.n_frames * (s.n_sources + 1) for s in subs],
        kind="stable",
    )
    by_index = {
        int(i): pool.submit(_solve_one, subs[i], sp, options) for i in order
    }
    return [by_index[i] for i in range(len(subs))], []


def _save_container(folder, session, params, plan, ph, chash, trial, d, t):
    """Write a loaded trial's container (``--save-subproblems``)."""
    from ..extraction.raster.container import write_container

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / session.container_file(d, t).name
    write_container(
        path,
        trial,
        np.column_stack([plan.source_rows, plan.source_cols]),
        [g.pixels for g in plan.groups],
        [g.sources for g in plan.groups],
        plan.shape,
        params=_container_params(params, ph, d, t),
        source={"raw": session.raw(d, t).name, "path": d + 1, "trial": t + 1},
    )
    with h5py.File(path, "a") as f:
        f.attrs["settings_hash"] = chash


def stream(  # noqa: C901
    session: SiloSession,
    params: SiloParams,
    execution: SiloExecution,
    paths: Optional[Iterable[int]] = None,
    pool=None,
    save_subproblems: Optional[Path] = None,
    clock0: Optional[float] = None,
) -> list:
    """Load the paths' valid trials one ahead of the solver.

    As MATLAB's ``processAllTrials_Async``: while one trial's subproblems
    are solved in ``pool``, the next trial is loaded here (also across
    paths: a path's first trial is loaded while the previous path's last
    trial is solved). Each trial's results go to the results file that the
    staged run writes (``results/path<d>_trial<tttt>.h5``, with the same
    settings hash), so that :func:`~.silo_collect.collect` makes the
    summaries either way. Containers are written only to
    ``save_subproblems``, if given.

    Parameters
    ----------
    paths : iterable of int, optional
        Zero-based paths, in the order to extract them; may be a generator
        that yields each path once it is planned (default: every planned
        path).
    clock0 : float, optional
        ``time.perf_counter()`` that ``submitted`` and ``solved`` count
        from (default: when this starts).

    Returns
    -------
    list of dict
        Per trial: ``unit``, ``load`` and ``wait`` seconds (the wait is
        for the previous trial's solve), ``submitted`` and ``solved``
        (seconds since ``clock0`` when its subproblems went to the solver
        and when the last was solved; a trial solved in this process, with
        one solver process or batches, is solved when that returns), or
        ``skipped``.
    """
    clock0 = time.perf_counter() if clock0 is None else clock0
    solve_times = {}
    from ..extraction.raster.plan import read_plan
    from ..extraction.raster.results import TrialRecord
    from .silo_solve import results_attrs, solve_hash

    sp = params.solver_params()
    options = execution.solver_options()
    report = []
    pending: Optional[_Pending] = None
    if paths is None:
        paths = range(session.n_paths)
    for d in paths:
        pf = session.plan_file(d)
        if not pf.exists():
            continue
        plan = read_plan(pf)
        rois = read_user_rois(pf)[1]
        ph = _attr(pf, "settings_hash")
        placements = [(g.pixels, g.sources) for g in plan.groups]
        n_sub = len(plan.groups)
        for t in (int(x) for x in plan.valid_trials):
            chash = _prepare_hash(session, params, ph, d, t)
            rhash = solve_hash(params, options, chash, (0, n_sub))
            out = session.results_file(d, t)
            skip = session.skip_file(d, t)
            if not execution.overwrite and (
                _attr(out, "settings_hash") == rhash
                or skip_hash(skip) == chash
            ):
                report.append({"unit": (d, t), "skipped": "up to date"})
                continue
            t0 = time.perf_counter()
            try:
                trial = load_trial_for(
                    session, params, plan, d, t, rois, execution.loader_device
                )
            except ValueError as exc:
                if NO_DATA not in str(exc):
                    raise
                _mark_skipped(session, d, t, str(exc), chash, ph)
                report.append({"unit": (d, t), "skipped": str(exc)})
                continue
            if skip.exists():
                skip.unlink()
            load_s = time.perf_counter() - t0
            record = TrialRecord.from_trial(trial)
            if save_subproblems is not None:
                _save_container(
                    save_subproblems,
                    session,
                    params,
                    plan,
                    ph,
                    chash,
                    trial,
                    d,
                    t,
                )
            extracted = trial.frame_lines.size >= params.min_frames
            subs = (
                [g.subproblem(trial.y, trial.finv) for g in plan.groups]
                if extracted
                else []
            )
            attrs = results_attrs(
                _ContainerView(trial, plan, d, t, ph),
                session.container_file(d, t).name,
                chash,
                (0, n_sub),
                n_sub,
                extracted,
                rhash,
                params,
                options,
            )
            del trial
            submitted = time.perf_counter() - clock0
            futures, results = _submit(subs, sp, options, pool)
            # before waiting for the previous trial, so that a trial solved
            # meanwhile gets its own completion time
            done = _track(futures)
            del subs
            if pending is not None:
                wait_s = _finish(pending)
                solve_times[(pending.d, pending.t)] = _solve_span(
                    pending, clock0
                )
            else:
                wait_s = 0.0
            out.parent.mkdir(parents=True, exist_ok=True)
            pending = _Pending(
                d,
                t,
                record,
                placements if extracted else [],
                futures,
                results,
                attrs,
                out,
                submitted,
                done,
            )
            report.append({"unit": (d, t), "load": load_s, "wait": wait_s})
            if execution.verbose:
                print(
                    f"SILo path {d + 1} trial {t + 1}: loaded in "
                    f"{load_s:.1f} s (waited {wait_s:.1f} s for the "
                    "previous trial)",
                    flush=True,
                )
    if pending is not None:
        report.append(
            {"unit": (pending.d, pending.t), "wait": _finish(pending)}
        )
        solve_times[(pending.d, pending.t)] = _solve_span(pending, clock0)
    for entry in report:
        if "load" in entry and entry["unit"] in solve_times:
            entry["submitted"], entry["solved"] = solve_times[entry["unit"]]
    return report


def _solve_span(p: _Pending, clock0: float) -> tuple:
    """When a finished trial went to the solver and when its last
    subproblem was solved, in seconds since ``clock0``."""
    end = p.done.last() if p.done is not None else None
    if end is None:
        end = time.perf_counter()
    return p.submitted, end - clock0


def _ready_paths(  # noqa: C901
    session,
    params,
    ex,
    seeds,
    baseline_dir,
    timer,
    peak_pool,
    peak_futures,
    pools,
):
    """Localize and plan the paths; yield each path once it is planned.

    With several workers a path's plan starts as soon as its own trials
    are localized, and its per-trial peak fits go to ``peak_pool``
    (``peak_futures``) then; paths are yielded in the order their plans
    finish, so the first planned path is extracted while the other is
    still planned. With one worker everything runs here, in order.
    """
    t0 = time.perf_counter()
    jobs = localize_jobs(
        session, params, None, baseline_dir, False, ex.overwrite
    )
    timer["localize_units"] = len(jobs)
    cpus = os.cpu_count() or 1
    paths = [d for d in range(session.n_paths) if session.keep[d].any()]
    seeds_of = (lambda d: None) if seeds is None else (lambda d: seeds[d])
    if ex.workers <= 1:
        timer["localize_trials"] = run_jobs(_localize_job, jobs, 1, cpus)
        timer["localize"] = time.perf_counter() - t0
        timer["plan_paths"] = {}
        for d in paths:
            t1 = time.perf_counter()
            plan_one(session, params, d, seeds_of(d), ex.overwrite)
            timer["plan_paths"][d + 1] = time.perf_counter() - t1
            yield d
        return
    loc_pool = None
    if jobs:
        loc_pool = make_pool(
            min(ex.workers, len(jobs)), max(1, cpus // ex.workers)
        )
        pools.append(loc_pool)
    plan_pool = make_pool(
        max(1, len(paths)), max(1, cpus // max(1, len(paths)))
    )
    pools.append(plan_pool)
    pending = {d: 0 for d in paths}
    futures = {}
    for job in jobs:
        pending[job["unit"][0]] += 1
        futures[loc_pool.submit(_localize_job, job)] = job["unit"][0]
    planned = {}
    timer["plan_paths"] = {}

    def start(d):
        """Plan path ``d`` and fit its trials' peaks in the background."""
        planned[
            plan_pool.submit(
                _plan_job,
                {
                    "session": session,
                    "params": asdict(params),
                    "d": d,
                    "seeds": seeds_of(d),
                    "overwrite": ex.overwrite,
                },
            )
        ] = d
        if peak_pool is not None:
            units = [(d, t) for t in range(session.n_trials)]
            for job in peaks_jobs(session, params, units):
                peak_futures.append(peak_pool.submit(_peaks_job, job))

    for d in paths:
        if not pending[d]:
            start(d)
    timer["localize_trials"] = []
    for f in cf.as_completed(futures):
        timer["localize_trials"].append(f.result())
        d = futures[f]
        pending[d] -= 1
        if not pending[d]:
            start(d)
    timer["localize"] = time.perf_counter() - t0
    if loc_pool is not None:
        loc_pool.shutdown()
    for f in cf.as_completed(planned):
        res = f.result()
        timer["plan_paths"][res["path"] + 1] = res["seconds"]
        timer.setdefault("first_plan_ready", time.perf_counter() - t0)
        yield res["path"]
    timer["plan"] = time.perf_counter() - t0


def run_silo(
    table,
    params=None,
    execution=None,
    *,
    moco_dir=None,
    data_dir=None,
    out_dir=None,
    work_dir=None,
    annotations=None,
    seeds_file=None,
    baseline_dir=None,
    save_subproblems=None,
    update_table: bool = True,
    peak_workers: Optional[int] = None,
) -> dict:
    """SILo on one machine (``giant silo``), streaming as MATLAB's.

    1. Localize every kept trial-path (``workers`` processes).
    2. Plan each path (one process per path) as soon as its trials are
       localized; its per-trial peak fits, which only the per-trial
       summary uses, start then in the background (``peak_workers``
       processes, one thread each).
    3. Per path, in the order the plans finish, load each valid trial at
       the plan's selected pixels while the previous trial's subproblems
       are solved (``solver_workers`` processes of ``solver_threads``
       threads); write each trial's results file.
    4. Collect: kinetics test, ``experiment_summary.h5``,
       ``per_trial_summary.h5``, the trial table's fields.

    Units whose outputs exist and were made with the same settings are
    kept (unless ``execution.overwrite``).

    Parameters
    ----------
    table : str or Path
        ``trial_table.h5`` (or its folder).
    params : SiloParams or dict, optional
        Scientific settings (default: MATLAB's SILo for SLAP2).
    execution : SiloExecution or dict, optional
        How to run.
    moco_dir, data_dir, out_dir, work_dir, annotations : optional
        See :meth:`SiloSession.open`.
    seeds_file : str or Path, optional
        Seeds to use instead of the plan's seeding
        (:func:`~giant_python.extraction.raster.stage_files.read_seeds`).
    baseline_dir : str or Path, optional
        The network baselines (``localization_baseline='network'``).
    save_subproblems : str or Path, optional
        Also write each trial-path's container to this folder.
    update_table : bool
        Write the trial table's ``source_extraction`` fields.
    peak_workers : int, optional
        Processes fitting per-trial peaks in the background (default: half
        of ``workers``, at least 1).

    Returns
    -------
    dict
        Seconds of each stage, per path and per trial.
    """
    from ..extraction.raster.stage_files import read_seeds
    from .silo_collect import collect
    from .silo_session import now

    params, ex = resolve_silo_options(params, execution)
    start = now()
    timer: dict = {"start": start}
    t_all = time.perf_counter()
    session = SiloSession.open(
        table, moco_dir, data_dir, out_dir, work_dir, annotations
    )
    seeds = read_seeds(seeds_file, session.n_paths) if seeds_file else None

    peak_futures: list = []
    peak_pool = None
    pools: list = []
    try:
        if ex.workers > 1:
            n_peak = peak_workers or max(1, ex.workers // 2)
            peak_pool = make_pool(n_peak, 1) if params.trial_peaks else None
        paths = _ready_paths(
            session,
            params,
            ex,
            seeds,
            baseline_dir,
            timer,
            peak_pool,
            peak_futures,
            pools,
        )
        t0 = time.perf_counter()
        from .silo_solve import _pool

        pool = _pool(ex, ex.solver_options())
        try:
            timer["trials"] = stream(
                session, params, ex, paths, pool, save_subproblems, t_all
            )
        finally:
            if pool is not None:
                pool.shutdown()
        timer["extract"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        timer["peaks"] = [f.result() for f in peak_futures]
        if peak_pool is None and params.trial_peaks:
            timer["peaks"] = run_jobs(
                _peaks_job, peaks_jobs(session, params), 1, 1
            )
        timer["peaks_wait"] = time.perf_counter() - t0
    finally:
        for p in pools + ([peak_pool] if peak_pool is not None else []):
            p.shutdown(cancel_futures=True)

    t0 = time.perf_counter()
    timer["collect_detail"] = collect(
        session,
        params,
        ex,
        update_table=update_table,
        start_time=start,
        save_subproblems_dir=str(save_subproblems or ""),
    )
    timer["collect"] = time.perf_counter() - t0
    timer["total"] = time.perf_counter() - t_all
    return timer
