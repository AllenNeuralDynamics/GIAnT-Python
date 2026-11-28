"""SILo for SLAP2 multi-ROI raster recordings: stages and the streaming driver.

GIAnT-MATLAB's ``SILo.m`` (main 5a0673b) as stages, each a function from
files to files that runs for any subset of its units and skips units whose
outputs exist and were made with the same settings:

=========  ==========  ===========================  ========================
stage      unit        reads                        writes
=========  ==========  ===========================  ========================
baseline   trial-path  registered movie             ``*_BASELINE-80Hz.h5``
                                                    (slap2-f0, optional)
localize   trial-path  registered movie, alignment  ``localize/*.h5``
plan       path        localization files,          ``plan/path<d>.h5``
                       annotations, seeds
prepare    trial-path  raw ``.dat``, plan,          ``containers/*.h5``
                       network baseline
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

With ``SiloParams.baseline_source='network'`` the F0 network's baselines
(``--baseline-dir``, or predicted with ``--f0-ckpt``) are used to localize
and, mapped onto each trial's selected pixels
(:mod:`~giant_python.extraction.raster.baseline_map`), to extract. Each
stage checks that the stage before it used the same baseline.
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
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePath
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


def baseline_folder(session: SiloSession, baseline_dir=None) -> Path:
    """The folder of the network's baselines: ``baseline_dir``, or the
    session's default (``<work-dir>/baselines``, where ``--f0-ckpt`` puts
    them)."""
    return Path(baseline_dir) if baseline_dir else session.baselines_dir


def registered_name(session: SiloSession, d: int, t: int) -> Path:
    """The registered movie's file name that the trial table records
    (``fn_reg_ds``), whether or not the movie is here."""
    return Path(PurePath(str(session.fn_reg_ds[d, t]).replace("\\", "/")).name)


def _network_baseline(session, params, d, t, baseline_dir) -> Path:
    """The baseline file of trial-path ``(d, t)``; it must exist and have
    been predicted from the movie that SILo reads (and, if it has a record,
    with this run's settings: ``check_baseline_record``)."""
    from .silo_baseline import check_baseline_record

    base = _baseline_file(
        baseline_folder(session, baseline_dir), registered_name(session, d, t)
    )
    if not base.exists():
        raise FileNotFoundError(
            f"no network baseline: {base} (predict it with --f0-ckpt, or "
            "give the baselines' folder with --baseline-dir)"
        )
    source = _attr(base, "source")
    if isinstance(source, bytes):
        source = source.decode()
    movie = session.movie(d, t)
    if source and movie is not None and str(source) != movie.name:
        raise ValueError(
            f"{base.name} was predicted from {source}, SILo reads "
            f"{movie.name}: predict it again from that movie (or remove "
            "the other one)"
        )
    check_baseline_record(base, movie, session.alignment(d, t), params)
    return base


def _recorded(path, name: str, default: str = "median") -> str:
    """A stage file's record of the baseline it used (files made before
    the network baseline was wired in used the moving median)."""
    value = _attr(path, name)
    if isinstance(value, bytes):
        value = value.decode()
    return default if value is None else str(value)


def _check_same(
    what: str, used, given, option: str, flag: str = "--baseline"
) -> None:
    """Refuse a stage whose settings differ from the stage before it."""
    if used != given:
        raise ValueError(
            f"{what} was made with {option}={used!r}, these settings have "
            f"{given!r}: give every stage the same settings (e.g. "
            f"{flag} {used})"
        )


def _needs_halves(params: SiloParams) -> bool:
    """Whether the plan needs each trial's half images (the patchwise
    alignment weights its patches with them)."""
    return params.inter_trial_alignment == "patchwise"


def _has_halves(path) -> bool:
    """Whether a localization file holds the trial's half images."""
    return bool(_attr(path, "half_ims"))


def check_plan_alignment(what: str, plan, params: SiloParams) -> None:
    """Refuse to extract or collect with another across-trial alignment
    than the plan's: the trials are read through the plan's maps, and the
    summaries record these settings."""
    _check_same(
        what,
        plan.params.inter_trial_alignment,
        params.inter_trial_alignment,
        "inter_trial_alignment",
        "--alignment",
    )
    if params.inter_trial_alignment == "patchwise":
        _check_same(
            what,
            float(plan.params.align_patch_px),
            float(params.align_patch_px),
            "align_patch_px",
            "--align-patch-px",
        )


def check_plan_seeding(what: str, plan, params: SiloParams) -> None:
    """Refuse to collect with another seed threshold than the plan's: the
    summary records ``peakth``, and the evidence test judges the sources
    by their height above it (a folder made with the network's earlier
    default, 8, is not collected with today's, 3)."""
    used, given = float(plan.params.peak_th), float(params.peak_th)
    if used != given:
        raise ValueError(
            f"{what} was seeded with peak_th={used:g}, these settings have "
            f"{given:g}: run the stages again from localization with these "
            f"settings (giant silo redoes them), or collect with "
            f"--peak-th {used:g}"
        )


def check_plan_support(what: str, plan, params: SiloParams) -> None:
    """Refuse to extract with a footprint support other than the one the
    plan's zones were made for (``valid_kernel``: one per run)."""
    _check_same(
        what,
        int(plan.params.valid_kernel),
        params.valid_kernel,
        "valid_kernel",
        "--valid-kernel",
    )


def trial_null_seed(base: int, d: int, t: int, draws: int = 1) -> int:
    """The first null seed of trial-path ``(d, t)`` (zero-based) when the
    run keeps its null images: ``base`` plus a trial-path offset, so that
    the trials' null movies are independent draws (with one seed for all
    trials the same random stream recurs in every trial, and the combined
    null keeps much of one trial's spread). A trial-path's ``draws`` seeds
    (``seed .. seed + draws - 1``) never overlap another's: offsets are
    ``max(1000, draws)`` apart."""
    width = max(1000, int(draws))
    return int(base) + width * (1 + int(t) + 100000 * int(d))


def _localize_job(job: dict) -> dict:
    """Localize one trial-path and write its file (a pool task)."""
    from ..extraction.raster.peaks import get_act_im_peaks
    from ..extraction.raster.stage_files import write_localization
    from ..extraction.raster.trial import localize_trial

    t0 = time.perf_counter()
    p = SiloParams(**job["params"])
    lp = p.localization_params(job["align_hz"])
    if p.localization_keep_null:
        lp = replace(
            lp,
            null_seed=trial_null_seed(
                lp.null_seed, *job["unit"], draws=lp.null_draws
            ),
        )
    loc, trial = localize_trial(
        job["moco_dir"],
        job["movie"],
        job["num_channels"],
        lp,
        baseline_dir=job["baseline_dir"],
        activity_channel=p.activity_channel,
        motion_thresh=p.motion_thresh,
        discard_initial_s=p.discard_initial_s,
        half_ims=_needs_halves(p),
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
    null_im = None
    if p.localization_keep_null:
        null_im = loc.info.get("null", {}).get("null_im")
    write_localization(
        job["out"],
        trial.mean_im,
        loc.act_im,
        loc.summary,
        trial.discard,
        peaks,
        job["attrs"],
        trial.half_ims,
        null_im,
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
    rec = read_localization(job["file"], half_ims=False)
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
        'network'``; default: the session's ``baselines`` folder).
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
    jobs = []
    for d, t in _units(session, units):
        movie = session.movie(d, t)
        base = (
            _network_baseline(session, params, d, t, baseline_dir)
            if network
            else None
        )
        h = _localize_hash(
            session, params, d, t, None if base is None else _file_id(base)
        )
        out = session.localize_file(d, t)
        if (
            not overwrite
            and _attr(out, "settings_hash") == h
            # files made before the baseline was recorded: those made with
            # the network are redone (the plan stage would refuse them)
            and _recorded(out, "localization_baseline")
            == params.localization_baseline
            # the patchwise alignment needs the half images, which files
            # localized for another alignment lack
            and (not _needs_halves(params) or _has_halves(out))
        ):
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
                    str(base.parent) if base is not None else None
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
                    "localization_baseline": params.localization_baseline,
                    # the file localized with (extraction must use it too)
                    # and its network, for the summary
                    "network_baseline": (
                        json.dumps(baseline_provenance(base), sort_keys=True)
                        if base is not None
                        else ""
                    ),
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


def _plan_inputs(session: SiloSession, d: int, half_ims: bool = False):
    """The localization records of path ``d`` (None for trials without);
    the half images only with ``half_ims``."""
    from ..extraction.raster.stage_files import read_localization

    recs = []
    for t in range(session.n_trials):
        f = session.localize_file(d, t)
        recs.append(
            read_localization(f, peaks=False, half_ims=half_ims)
            if session.keep[d, t]
            else None
        )
    return recs


def _localization_hashes(session, d) -> dict:
    """The settings hashes of path ``d``'s localization files, by trial."""
    locs = {}
    for t in range(session.n_trials):
        if session.keep[d, t]:
            f = session.localize_file(d, t)
            locs[t] = _attr(f, "settings_hash")
            if locs[t] is None:
                raise FileNotFoundError(f"not localized: {f}")
    return locs


def check_plan_localizations(session, d) -> None:
    """Refuse a plan whose trials were localized again since it was made
    (their peaks and baselines would not be the plan's): the plan records
    its localizations' settings hashes (plans made before that are not
    checked)."""
    made = _attr(session.plan_file(d), "localization_hashes")
    if not made:
        return
    if isinstance(made, bytes):
        made = made.decode()
    made = {int(k): v for k, v in json.loads(made).items()}
    for t, h in made.items():
        now = _attr(session.localize_file(d, t), "settings_hash")
        if now != h:
            raise ValueError(
                f"path {d + 1}, trial {t + 1} was localized again since its "
                "plan was made: plan again and redo the stages after it"
            )


def _plan_hash(session, params, d, seeds, annotations) -> str:
    """Settings hash of a path's plan."""
    locs = _localization_hashes(session, d)
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
    _check_localizations(session, params, d)
    h = _plan_hash(session, params, d, seeds, ann_id)
    out = session.plan_file(d)
    if not overwrite and _attr(out, "settings_hash") == h:
        return out
    recs = _plan_inputs(session, d, _needs_halves(params))
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
            f.attrs["baseline_source"] = params.baseline_source
            f.attrs["localization_hashes"] = json.dumps(
                _localization_hashes(session, d), sort_keys=True
            )
        os.replace(tmp, out)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    return out


def _check_localizations(session: SiloSession, params: SiloParams, d: int):
    """Refuse to plan path ``d`` from localizations made with another
    baseline than these settings', or without the half images that a
    patchwise alignment needs."""
    for t in np.flatnonzero(session.keep[d]):
        f = session.localize_file(d, int(t))
        if f.exists():
            _check_same(
                f"the localization {f.name}",
                _recorded(f, "localization_baseline"),
                params.localization_baseline,
                "localization_baseline",
            )
            if _needs_halves(params) and not _has_halves(f):
                raise ValueError(
                    f"the localization {f.name} has no half images, which "
                    "the patchwise alignment needs: run the localize stage "
                    "with these settings (it redoes such files)"
                )


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
    session: SiloSession,
    params: SiloParams,
    plan_hash: str,
    d,
    t,
    baseline: Optional[Path] = None,
) -> str:
    """Settings hash of a container: the prepare settings, the plan, the
    raw recording's files, the trial's lines and its alignment file, and
    the network baseline file mapped into it (when the solver uses one)."""
    from ..extraction.raster.baseline_map import MIN_WEIGHT

    record = {
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
    if baseline is not None:
        record["baseline"] = {
            "file": _file_id(baseline),
            "min_weight": MIN_WEIGHT,
        }
    return _hash(record)


def _extraction_baseline(session, params, d, t, baseline_dir):
    """The baseline file the solver's mode needs for ``(d, t)``, or None.

    It must be the file the trial was localized with, when it was localized
    with the network's baseline (one baseline per run, whichever folders
    the stages were given).
    """
    if params.baseline_mode == "fit":
        return None
    base = _network_baseline(session, params, d, t, baseline_dir)
    loc = localization_baseline(session, d, t)
    if loc is not None and loc["file"] != _file_id(base):
        raise ValueError(
            f"path {d + 1}, trial {t + 1} was localized with another "
            f"baseline file than {base} ({loc['file']}): give the stages "
            "the same baselines (--baseline-dir)"
        )
    return base


def baseline_provenance(base: Path) -> dict:
    """A baseline file's identity and its network: the checkpoint's path
    and step that slap2-f0 records, and its identity at prediction time
    from the file's record (when it was predicted here)."""
    from .silo_baseline import record_path

    network = {}
    for name in ("checkpoint", "step"):
        v = _attr(base, name)
        if isinstance(v, bytes):
            v = v.decode()
        if v is not None:
            network[name] = v.item() if hasattr(v, "item") else v
    out = {"file": _file_id(base), "network": network}
    rec = record_path(base)
    if rec.exists():
        out["checkpoint"] = json.loads(rec.read_text()).get("checkpoint")
    return out


def localization_baseline(session: SiloSession, d: int, t: int):
    """The network baseline trial-path ``(d, t)`` was localized with
    (:func:`baseline_provenance`), or None."""
    v = _attr(session.localize_file(d, t), "network_baseline")
    if isinstance(v, bytes):
        v = v.decode()
    return json.loads(v) if v else None


def load_trial_baseline(
    session: SiloSession, params: SiloParams, trial, base: Path, d, t
):
    """The network's baseline at a loaded trial's selected pixels.

    :func:`~giant_python.extraction.raster.baseline_map.read_trial_baseline`,
    with the checkpoint's identity and the prediction's settings added from
    the file's record (when it was predicted here).
    """
    from ..extraction.raster.baseline_map import read_trial_baseline
    from .silo_baseline import check_baseline_record

    with h5py.File(session.alignment(d, t), "r") as f:
        # the registered frames' lines (DSframes, as read_alignment)
        ds_lines = np.asarray(f["DSframes"], np.float64).ravel()
    tb = read_trial_baseline(
        base, trial.trial_rc, ds_lines, trial.photon_scale
    )
    rec = check_baseline_record(base, None, session.alignment(d, t), params)
    if rec is not None:
        # identified when it was predicted; a file without a record names
        # its checkpoint only by path and step (tb.identity["network"])
        tb.identity["checkpoint"] = rec["checkpoint"]
        tb.identity["prediction"] = rec["settings"]
    return tb


def trial_subproblems(plan, trial, solver_params, baseline=None) -> list:
    """The plan's subproblems on a loaded trial.

    With a solver mode that uses an external baseline, each gets the
    network's (``baseline``, a
    :class:`~giant_python.extraction.raster.baseline_map.TrialBaseline`):
    the same arrays as a container of the trial gives
    (:meth:`~giant_python.extraction.raster.container.Container.subproblem`).
    """
    from ..extraction.raster.baseline_map import subproblem_baseline

    if solver_params.baseline_mode == "fit":
        return [g.subproblem(trial.y, trial.finv) for g in plan.groups]
    if baseline is None:
        raise ValueError(
            f"the solver's baseline mode {solver_params.baseline_mode!r} "
            "needs the network's baseline"
        )
    values = np.asarray(baseline.values, dtype=np.float32)
    return [
        g.subproblem(
            trial.y,
            trial.finv,
            subproblem_baseline(
                values[g.pixels],
                baseline.lines,
                trial.frame_lines,
                trial.y[0, g.pixels],
                solver_params,
            ),
        )
        for g in plan.groups
    ]


def prepare(
    session: SiloSession,
    params: SiloParams,
    execution: Optional[SiloExecution] = None,
    units=None,
    baseline_dir=None,
) -> list:
    """Stage ``prepare``: a container per extracted trial-path.

    The valid trials of each planned path are loaded at the plan's
    selected pixels (200 Hz, from the raw files) and written with the
    plan's subproblems, and, when the solver's ``baseline_mode`` uses it,
    the network's baseline at those pixels. A trial in which no selected
    pixel has data gets a ``*.skip.json`` marker instead.

    Parameters
    ----------
    units : iterable of (int, int), optional
        Zero-based ``(path, trial)`` pairs (default: all valid trials).
    baseline_dir : str or Path, optional
        Folder of the network baseline files (default: the session's
        ``baselines`` folder).

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
            check_plan_localizations(session, d)
            _check_same(
                f"the plan {f.name}",
                _recorded(f, "baseline_source"),
                params.baseline_source,
                "baseline_source",
            )
            plans[d] = (
                read_plan(f),
                read_user_rois(f)[1],
                _attr(f, "settings_hash"),
            )
            check_plan_support(f"the plan {f.name}", plans[d][0], params)
            check_plan_alignment(f"the plan {f.name}", plans[d][0], params)
        p, rois, ph = plans[d]
        base = _extraction_baseline(session, params, d, t, baseline_dir)
        h = _prepare_hash(session, params, ph, d, t, base)
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
        tb = (
            None
            if base is None
            else load_trial_baseline(session, params, trial, base, d, t)
        )
        write_container(
            out,
            trial,
            np.column_stack([p.source_rows, p.source_cols]),
            [g.pixels for g in p.groups],
            [g.sources for g in p.groups],
            p.shape,
            params=_container_params(params, ph, d, t, tb),
            source={
                "raw": session.raw(d, t).name,
                "path": d + 1,
                "trial": t + 1,
            },
            baseline=tb,
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
    params: SiloParams, plan_hash: str, d: int, t: int, baseline=None
) -> dict:
    """What a container records of its settings (with the network: the
    baseline source, and the identity of the baseline mapped into it)."""
    out = {
        "prepare": params.stage_settings("prepare"),
        "plan": plan_hash,
        "path": d + 1,
        "trial": t + 1,
    }
    if params.baseline_source != "median":
        out["baseline_source"] = params.baseline_source
    if params.valid_kernel != 3:
        # the plan's zones were made for it; the solve stage checks it
        out["valid_kernel"] = params.valid_kernel
    if baseline is not None:
        out["baseline"] = baseline.identity
    return out


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

    def __init__(self, n: int = 1) -> None:
        """Expect ``n`` completions (set by :func:`_submit`)."""
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
    baseline: Optional[dict] = None

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
        """What the container would record: its plan's hash and the
        baseline's identity."""
        out = {"plan": self.plan_hash}
        if self.baseline is not None:
            out["baseline"] = self.baseline
        return out


def _submit(subs, sp, options, pool, done: Optional[_Done] = None):
    """Solve a trial's subproblems: futures from the pool (largest first),
    or results solved here.

    ``done`` gets each future's completion time (its done callback is
    attached as soon as the future exists, so a subproblem solved while the
    rest are still submitted keeps its own time), or the time the solve
    here returned.
    """
    from ..extraction.raster.solver import solve_subproblems
    from .silo_solve import _solve_one, effective_batch

    done = done if done is not None else _Done()
    if not subs:
        done.add()
        return [], []
    if pool is None or effective_batch(options) is not None:
        results = solve_subproblems(subs, sp, options)
        done.add()
        return [], results
    order = np.argsort(
        [-s.n_pixels * s.n_frames * (s.n_sources + 1) for s in subs],
        kind="stable",
    )
    done.n = len(subs)
    by_index = {}
    for i in order:
        f = pool.submit(_solve_one, subs[i], sp, options)
        f.add_done_callback(done.add)
        by_index[int(i)] = f
    return [by_index[i] for i in range(len(subs))], []


def _save_container(
    folder, session, params, plan, ph, chash, trial, d, t, baseline=None
):
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
        params=_container_params(params, ph, d, t, baseline),
        source={"raw": session.raw(d, t).name, "path": d + 1, "trial": t + 1},
        baseline=baseline,
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
    baseline_dir=None,
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
    baseline_dir : str or Path, optional
        Folder of the network baseline files, when the solver's mode uses
        them (default: the session's ``baselines`` folder).

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
        check_plan_localizations(session, d)
        _check_same(
            f"the plan {pf.name}",
            _recorded(pf, "baseline_source"),
            params.baseline_source,
            "baseline_source",
        )
        check_plan_support(f"the plan {pf.name}", plan, params)
        check_plan_alignment(f"the plan {pf.name}", plan, params)
        placements = [(g.pixels, g.sources) for g in plan.groups]
        n_sub = len(plan.groups)
        for t in (int(x) for x in plan.valid_trials):
            base = _extraction_baseline(session, params, d, t, baseline_dir)
            chash = _prepare_hash(session, params, ph, d, t, base)
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
            tb = (
                None
                if base is None
                else load_trial_baseline(session, params, trial, base, d, t)
            )
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
                    tb,
                )
            extracted = trial.frame_lines.size >= params.min_frames
            subs = trial_subproblems(plan, trial, sp, tb) if extracted else []
            attrs = results_attrs(
                _ContainerView(
                    trial,
                    plan,
                    d,
                    t,
                    ph,
                    None if tb is None else tb.identity,
                ),
                session.container_file(d, t).name,
                chash,
                (0, n_sub),
                n_sub,
                extracted,
                rhash,
                params,
                options,
            )
            del trial, tb
            submitted = time.perf_counter() - clock0
            done = _Done()
            futures, results = _submit(subs, sp, options, pool, done)
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
    f0_ckpt=None,
    f0_device: str = "cuda",
    f0_command: Optional[str] = None,
    f0_out_frames: Optional[int] = None,
) -> dict:
    """SILo on one machine (``giant silo``), streaming as MATLAB's.

    0. With ``f0_ckpt``: predict the network's baselines of the trial-paths
       that have none (``slap2-f0 predict-registered``, one process).
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
        Scientific settings (default: MATLAB's SILo for SLAP2). Given as a
        dict or None, ``baseline_source`` defaults to 'network' when
        ``baseline_dir`` or ``f0_ckpt`` is given (moving median otherwise).
    execution : SiloExecution or dict, optional
        How to run.
    moco_dir, data_dir, out_dir, work_dir, annotations : optional
        See :meth:`SiloSession.open`.
    seeds_file : str or Path, optional
        Seeds to use instead of the plan's seeding
        (:func:`~giant_python.extraction.raster.stage_files.read_seeds`).
    baseline_dir : str or Path, optional
        Folder of the network's baselines (default
        ``<work_dir>/baselines``).
    save_subproblems : str or Path, optional
        Also write each trial-path's container to this folder.
    update_table : bool
        Write the trial table's ``source_extraction`` fields.
    peak_workers : int, optional
        Processes fitting per-trial peaks in the background (default: half
        of ``workers``, at least 1).
    f0_ckpt : str or Path, optional
        A slap2-f0 checkpoint: predict the missing baselines into the
        baselines' folder first (needs ``slap2-f0``).
    f0_device : str
        Device of the prediction ('cuda' by default, 'cpu', 'auto').
    f0_command : str, optional
        The slap2-f0 executable (default: ``slap2-f0`` on the PATH).
    f0_out_frames : int, optional
        Frames per tile of the prediction (see
        :func:`~giant_python.pipeline.silo_baseline.predict_baselines`).

    Returns
    -------
    dict
        Seconds of each stage, per path and per trial.
    """
    from ..extraction.raster.stage_files import read_seeds
    from .silo_collect import collect
    from .silo_session import now

    params = default_baseline_source(
        params, baseline_dir is not None or f0_ckpt is not None
    )
    params, ex = resolve_silo_options(params, execution)
    start = now()
    timer: dict = {"start": start}
    t_all = time.perf_counter()
    session = SiloSession.open(
        table, moco_dir, data_dir, out_dir, work_dir, annotations
    )
    seeds = read_seeds(seeds_file, session.n_paths) if seeds_file else None
    bdir = baseline_folder(session, baseline_dir)
    if not params.needs_network_baseline and (baseline_dir or f0_ckpt):
        warnings.warn(
            "the network's baselines are not used: these settings localize "
            "and extract with the moving median (baseline_source='median')"
        )
    elif f0_ckpt is not None:
        from .silo_baseline import predict_baselines

        timer["predict_detail"] = predict_baselines(
            session,
            params,
            bdir,
            f0_ckpt,
            device=f0_device,
            command=f0_command,
            out_frames=f0_out_frames,
            verbose=ex.verbose,
        )
        timer["predict"] = timer["predict_detail"]["seconds"]

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
            bdir,
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
                session,
                params,
                ex,
                paths,
                pool,
                save_subproblems,
                t_all,
                bdir,
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
        baseline_dir=bdir,
    )
    timer["collect"] = time.perf_counter() - t0
    timer["total"] = time.perf_counter() - t_all
    return timer


def default_baseline_source(params, network: bool):
    """``params`` with ``baseline_source`` 'network' when the network's
    baselines are given (``network``) and ``params`` (a dict or None)
    does not set it. A :class:`SiloParams` is returned as it is: its
    ``baseline_source`` was chosen when it was made."""
    if not network or isinstance(params, SiloParams):
        return params
    values = dict(params or {})
    values.setdefault("baseline_source", "network")
    return values
