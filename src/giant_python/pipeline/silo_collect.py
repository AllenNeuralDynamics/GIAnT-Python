"""SILo's collect stage: the summaries and the trial table from the results.

Reads every planned path's plan, results files, localization files (the
trials' own peaks) and alignment files (the motion records), runs the
kinetics test, and writes ``experiment_summary.h5`` and
``per_trial_summary.h5`` in GIAnT-MATLAB's schema, then the trial table's
``source_extraction`` fields (``analysis_params``, ``fn_raw``), as
``SILo.m`` does at its end.

With the network's baseline, ``experiment_summary.h5`` also gets the
attributes ``baseline_source``, ``baseline_mode``, ``localization_baseline``
and ``network_baseline`` (JSON: per trial-path, the baseline file and the
network's checkpoint that extraction used); the ``params`` struct records
the settings that differ from MATLAB's (``baselineSource``,
``baselineMode``, ``peakth``, ``localizationSurround``, ...).
"""

from __future__ import annotations

import json
import os
import time
import warnings
from typing import Optional

import h5py
import numpy as np

from ..models.params import SiloExecution, SiloParams
from .silo_session import SiloSession, attr, now, skip_hash
from .silo_solve import solve_science


def _results_files(session: SiloSession, d: int, t: int) -> list:
    """The results files of trial-path ``(d, t)``."""
    stem = f"path{d + 1}_trial{t + 1:04d}"
    folder = session.results_dir
    if not folder.exists():
        return []
    return sorted(
        p
        for p in folder.glob(f"{stem}*.h5")
        if p.stem == stem or p.stem.startswith(stem + "_s")
    )


def _container_hash(
    session, params, plan_hash, d, t, baseline_dir=None
) -> Optional[str]:
    """The prepare hash of trial-path ``(d, t)`` from its current inputs;
    None if its raw recording or its network baseline is not here (a
    collect machine without them: results are then checked against plan
    and settings only)."""
    from .silo import (
        _baseline_file,
        _prepare_hash,
        baseline_folder,
        registered_name,
    )

    base = None
    if params.baseline_mode != "fit":
        # named after the table's registered movie, which need not be here
        base = _baseline_file(
            baseline_folder(session, baseline_dir),
            registered_name(session, d, t),
        )
        if not base.exists():
            return None
    try:
        return _prepare_hash(session, params, plan_hash, d, t, base)
    except OSError:
        return None


def _current(
    files: list,
    plan_hash: str,
    science: str,
    container_hash: Optional[str] = None,
) -> list:
    """The results files made from the plan ``plan_hash`` (and, when
    given, the container ``container_hash``: the trial's current raw data,
    lines and alignment) with the scientific solve settings ``science``;
    others are left out, with a warning (they are from earlier runs)."""
    keep, stale = [], []
    for f in files:
        ok = (
            attr(f, "plan_hash") == plan_hash
            and attr(f, "solve_science") == science
            and (
                container_hash is None
                or attr(f, "container_hash") == container_hash
            )
        )
        (keep if ok else stale).append(f)
    if stale:
        warnings.warn(
            f"{len(stale)} results file(s) from other settings or another "
            f"plan are left out: {', '.join(p.name for p in stale)}"
        )
    return keep


def _outcome_loader(files: list, solver_params):
    """A function that assembles a trial from its results files."""

    def load():
        """Read the files and assemble the trial."""
        from ..extraction.raster.results import read_results, trial_outcome

        return trial_outcome([read_results(f) for f in files], solver_params)

    return load


def path_result(
    session: SiloSession,
    params: SiloParams,
    d: int,
    baseline_dir=None,
    provenance: Optional[dict] = None,
) -> Optional[object]:
    """Everything the summaries need of path ``d``, read from its files.

    ``provenance``, if given, gets the network baseline (its identity, as
    the results files record it) of each trial-path assembled here, keyed
    ``path<d>_trial<tttt>``: from the files that are used, not from earlier
    runs' files left in the folder.

    Returns
    -------
    PathResult or None
        None for a path without a plan (no valid trials).

    Raises
    ------
    FileNotFoundError
        If a valid trial has neither a skip marker of the current plan nor
        results files made from the current plan with the current
        scientific solve settings (``solve_science``).
    """
    from ..extraction.raster.plan import read_plan
    from ..extraction.raster.stage_files import read_localization
    from ..extraction.raster.summary import PathResult, read_alignment_record
    from .silo import _check_same, _recorded, read_user_rois

    pf = session.plan_file(d)
    if not pf.exists():
        return None
    _check_same(
        f"the plan {pf.name}",
        _recorded(pf, "baseline_source"),
        params.baseline_source,
        "baseline_source",
    )
    plan = read_plan(pf)
    labels, masks = read_user_rois(pf)
    sp = params.solver_params()
    n = session.n_trials
    trials: list = [None] * n
    alignment: list = [None] * n
    peaks: list = [None] * n
    for t in range(n):
        f = session.localize_file(d, t)
        if session.keep[d, t] and f.exists():
            peaks[t] = read_localization(f).peaks
    plan_hash = attr(pf, "settings_hash")
    science = solve_science(params)
    for t in plan.valid_trials:
        t = int(t)
        if skip_hash(session.skip_file(d, t), "plan_hash") == plan_hash:
            continue  # no data at the selected pixels
        files = _current(
            _results_files(session, d, t),
            plan_hash,
            science,
            _container_hash(session, params, plan_hash, d, t, baseline_dir),
        )
        if not files:
            raise FileNotFoundError(
                f"path {d + 1}, trial {t + 1}: no results files made from "
                f"the current plan with the current solve settings in "
                f"{session.results_dir}"
            )
        trials[t] = _outcome_loader(files, sp)
        if provenance is not None:
            used = _baseline_identity(files)
            if used is not None:
                provenance[f"path{d + 1}_trial{t + 1:04d}"] = used
        alignment[t] = read_alignment_record(session.alignment(d, t))
    z = attr(pf, "z_depth")
    return PathResult(
        plan=plan,
        trials=trials,
        alignment=alignment,
        roi_labels=labels,
        roi_masks=masks,
        z_depth=float(z) if z is not None else float("nan"),
        trial_peaks=peaks if params.trial_peaks else [None] * n,
    )


def update_trial_table(session: SiloSession, params_matlab: dict) -> None:
    """Write the trial table's ``source_extraction`` fields as SILo does
    (``analysis_params``, ``fn_raw``); the rest of the table is left as it
    is."""
    from .register import _update_trial_table

    fn_raw = np.array(
        [[str(x or "") for x in row] for row in session.filename],
        dtype=object,
    )
    _update_trial_table(
        session.table_path,
        {"analysis_params": params_matlab, "fn_raw": fn_raw},
        group="source_extraction",
    )


def collect(
    session: SiloSession,
    params: SiloParams,
    execution: Optional[SiloExecution] = None,
    update_table: bool = True,
    start_time: Optional[str] = None,
    save_subproblems_dir: str = "",
    baseline_dir=None,
) -> dict:
    """Stage ``collect``: the summaries and the trial table's fields.

    Parameters
    ----------
    session : SiloSession
    params : SiloParams
    execution : SiloExecution, optional
        Recorded as ``nWorkers`` (``workers``).
    update_table : bool
        Write ``source_extraction`` into the trial table.
    start_time : str, optional
        ``startTime`` to record (default: now).
    save_subproblems_dir : str
        ``saveSubproblemsDir`` to record.
    baseline_dir : str or Path, optional
        Folder of the network's baselines (default: the session's), to
        check the results against them; without the files there the
        results are checked against plan and settings only.

    Returns
    -------
    dict
        Seconds of the kinetics test and of writing; the sources kept per
        path.
    """
    from ..extraction.raster.summary import kinetics_test, write_summaries

    ex = execution or SiloExecution()
    start_time = start_time or now()
    t0 = time.perf_counter()
    used: dict = {}
    results = [
        path_result(session, params, d, baseline_dir, used)
        for d in range(session.n_paths)
    ]
    kept = {}
    for d, r in enumerate(results):
        if r is None:
            continue
        kin = kinetics_test(
            r, params.tau_s, params.analyze_hz, params.min_kinetic_score
        )
        kept[d + 1] = (int(kin.keep.sum()), int(kin.keep.size))
    t1 = time.perf_counter()
    session.out_dir.mkdir(parents=True, exist_ok=True)
    pm = params.to_matlab(
        session.num_channels,
        session.align_hz,
        ex.workers,
        start_time,
        "",
        save_subproblems_dir,
    )
    pm["endTime"] = now()
    write_summaries(
        session.out_dir / "experiment_summary.h5",
        session.out_dir / "per_trial_summary.h5",
        results,
        pm,
        session.n_trials,
        session.num_channels,
        params.activity_channel,
    )
    if params.needs_network_baseline or params.baseline_source != "median":
        # MATLAB's algorithm leaves the file as MATLAB writes it
        with h5py.File(session.out_dir / "experiment_summary.h5", "a") as f:
            f.attrs["baseline_source"] = params.baseline_source
            f.attrs["baseline_mode"] = params.baseline_mode
            f.attrs["localization_baseline"] = params.localization_baseline
            if used:
                f.attrs["network_baseline"] = json.dumps(used, sort_keys=True)
    t2 = time.perf_counter()
    if update_table:
        update_trial_table(session, pm)
    record = {
        "created": pm["endTime"],
        "baseline_source": params.baseline_source,
        "baseline_mode": params.baseline_mode,
        "network_baseline": used,
        "kept_sources": kept,
        "kinetics_seconds": t1 - t0,
        "write_seconds": t2 - t1,
    }
    path = session.work_dir / "collect.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.partial")
    tmp.write_text(json.dumps(record, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    return record


def _baseline_identity(files: list) -> Optional[dict]:
    """The network baseline recorded in a trial's results files (they are
    of one container, so they record the same), or None."""
    for f in files:
        value = attr(f, "network_baseline")
        if value:
            if isinstance(value, bytes):
                value = value.decode()
            return json.loads(value)
    return None


def read_kept(path) -> dict:
    """Sources per path in an experiment summary (debugging aid)."""
    out = {}
    with h5py.File(path, "r") as f:
        for name in f:
            if name.startswith("Path") and "sources/temporal/SNR" in f[name]:
                out[name] = f[f"{name}/sources/temporal/SNR"].shape[-1]
    return out
