"""SILo's collect stage: the summaries and the trial table from the results.

Reads every planned path's plan, results files, localization files (the
trials' own peaks) and alignment files (the motion records), runs the
kinetics test, and writes ``experiment_summary.h5`` and
``per_trial_summary.h5`` in GIAnT-MATLAB's schema, then the trial table's
``source_extraction`` fields (``analysis_params``, ``fn_raw``), as
``SILo.m`` does at its end.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import h5py
import numpy as np

from ..models.params import SiloExecution, SiloParams
from .silo_session import SiloSession, attr, now


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


def _outcome_loader(files: list, solver_params):
    """A function that assembles a trial from its results files."""

    def load():
        """Read the files and assemble the trial."""
        from ..extraction.raster.results import read_results, trial_outcome

        return trial_outcome([read_results(f) for f in files], solver_params)

    return load


def path_result(
    session: SiloSession, params: SiloParams, d: int
) -> Optional[object]:
    """Everything the summaries need of path ``d``, read from its files.

    Returns
    -------
    PathResult or None
        None for a path without a plan (no valid trials).

    Raises
    ------
    FileNotFoundError
        If a valid trial has neither results nor a skip marker.
    """
    from ..extraction.raster.plan import read_plan
    from ..extraction.raster.stage_files import read_localization
    from ..extraction.raster.summary import PathResult, read_alignment_record
    from .silo import read_user_rois

    pf = session.plan_file(d)
    if not pf.exists():
        return None
    plan = read_plan(pf)
    labels, masks = read_user_rois(pf)
    sp = params.solver_params()
    n = session.n_trials
    trials: list = [None] * n
    alignment: list = [None] * n
    peaks: list = [None] * n
    for t in range(n):
        if session.keep[d, t]:
            rec = read_localization(session.localize_file(d, t))
            peaks[t] = rec.peaks
    for t in plan.valid_trials:
        t = int(t)
        files = _results_files(session, d, t)
        if not files:
            if session.skip_file(d, t).exists():
                continue
            raise FileNotFoundError(
                f"path {d + 1}, trial {t + 1}: no results files in "
                f"{session.results_dir}"
            )
        trials[t] = _outcome_loader(files, sp)
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
    results = [path_result(session, params, d) for d in range(session.n_paths)]
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
    t2 = time.perf_counter()
    if update_table:
        update_trial_table(session, pm)
    record = {
        "created": pm["endTime"],
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


def read_kept(path) -> dict:
    """Sources per path in an experiment summary (debugging aid)."""
    out = {}
    with h5py.File(path, "r") as f:
        for name in f:
            if name.startswith("Path") and "sources/temporal/SNR" in f[name]:
                out[name] = f[f"{name}/sources/temporal/SNR"].shape[-1]
    return out
