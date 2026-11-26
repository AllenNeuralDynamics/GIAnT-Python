"""SILo's solve stage: containers' subproblems solved into results files.

A solver instance needs a job file and the containers it names, nothing
else (no raw data, no SLAP2 reader, no trial table). :func:`dispatch`
writes the job files: shards of about equal estimated work, as contiguous
ranges of each container's subproblems. :func:`solve_job` runs one;
:func:`solve_containers` runs whole containers on this machine.

Each subproblem is solved on its own (MATLAB's algorithm, converged fits),
so the results do not depend on how the subproblems were split into jobs
or ranges, or on how many processes solved them (given the same threads
per process, precision and device).

Job file (JSON)::

    {"format": "giant-silo-job", "format_version": 1,
     "job": 0, "n_jobs": 4,
     "params": {... SiloParams ...},
     "solver": {"device": "cpu", "dtype": "float64", "batch_bytes": null},
     "items": [{"container": "path1_trial0001.h5",
                "container_hash": "...", "span": [0, 46],
                "results": "path1_trial0001_s0000-0045.h5",
                "work": 1.2e9}, ...],
     "work": 5.0e9, "estimated_seconds": 400.0}

``span`` is ``[first, stop)`` in the container's subproblem order; names
are relative to the containers' and results' folders given at run time.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

from ..models.params import SiloExecution, SiloParams
from .silo_session import (
    SiloSession,
    attr,
    make_pool,
    stage_hash,
)

JOB_FORMAT = "giant-silo-job"
JOB_VERSION = 1

SECONDS_PER_WORK = 2.5e-8
"""Estimated solve time per unit of :func:`subproblem_work` on one process
with 4 threads (measured on 780125; a guide for shard sizes only)."""


def subproblem_work(n_pixels, n_sources, n_frames) -> np.ndarray:
    """Estimated cost of subproblems: pixels x frames x (sources + 1).

    The fits' arrays are pixels x frames, and the footprint and activity
    fits grow with the sources.
    """
    return (
        np.asarray(n_pixels, np.float64)
        * float(n_frames)
        * (np.asarray(n_sources, np.float64) + 1.0)
    )


def solve_hash(params: SiloParams, options, container_hash: str, span) -> str:
    """Settings hash of a results file."""
    return stage_hash(
        {
            "stage": "solve",
            "settings": params.stage_settings("solve"),
            "solver": {
                "device": str(options.device).split(":")[0],
                "dtype": options.dtype,
                "batch_bytes": options.batch_bytes,
                "max_iter": options.max_iter,
                "fit_max_iter": options.fit_max_iter,
                "ftol": options.ftol,
                "cg_iter": options.cg_iter,
            },
            "container": container_hash,
            "span": [int(span[0]), int(span[1])],
        }
    )


def _solve_one(sub, params, options):
    """Solve one subproblem (a pool task)."""
    from ..extraction.raster.solver import solve_subproblem

    return solve_subproblem(sub, params, options)


def solve_subproblem_list(subs, params, options, pool=None) -> list:
    """Solve subproblems, in a pool if given; results in input order.

    With ``options.batch_bytes`` (GPU batches) they are solved together
    here instead. Larger subproblems are submitted first.
    """
    from ..extraction.raster.solver import solve_subproblems

    if not subs:
        return []
    if options.batch_bytes is not None or pool is None:
        return solve_subproblems(list(subs), params, options)
    order = np.argsort(
        [-s.n_pixels * s.n_frames * (s.n_sources + 1) for s in subs],
        kind="stable",
    )
    futures = {
        int(i): pool.submit(_solve_one, subs[i], params, options)
        for i in order
    }
    return [futures[i].result() for i in range(len(subs))]


def results_attrs(
    container, name: str, container_hash: str, span, n_sub, extracted, h
) -> dict:
    """Attributes of a results file."""
    src = getattr(container, "source", {}) or {}
    return {
        "settings_hash": h,
        "container": name,
        "container_hash": container_hash,
        "path": int(src.get("path", 0)),
        "trial": int(src.get("trial", 0)),
        "n_subproblems": int(n_sub),
        "n_pixels": int(container.pixel_rc.shape[0]),
        "n_sources": int(container.sources_rc.shape[0]),
        "photon_scale": float(container.photon_scale),
        "extracted": int(extracted),
        "span": np.asarray(span, np.int64),
    }


def solve_container(
    container_path,
    out_path,
    params: SiloParams,
    options,
    span=None,
    pool=None,
    overwrite: bool = False,
) -> Optional[dict]:
    """Solve subproblems of one container into a results file.

    Parameters
    ----------
    container_path : str or Path
        The container (read once, whole).
    out_path : str or Path
        The results file.
    params : SiloParams
        Scientific settings (the solver's are derived from them).
    options : SolverOptions
        How the fits run.
    span : (int, int), optional
        Subproblems ``first`` to ``stop - 1`` (default: all).
    pool : Executor, optional
        Workers to solve subproblems in.
    overwrite : bool
        Redo a results file made with the same settings.

    Returns
    -------
    dict or None
        ``seconds`` and the number solved; None if skipped (up to date).
    """
    from ..extraction.raster.container import read_container
    from ..extraction.raster.results import (
        TrialRecord,
        subproblem_results,
        write_results,
    )

    container_path, out_path = Path(container_path), Path(out_path)
    chash = attr(container_path, "settings_hash")
    t0 = time.perf_counter()
    c = read_container(container_path, data=False)
    n_sub = c.n_subproblems
    span = (0, n_sub) if span is None else (int(span[0]), int(span[1]))
    if not 0 <= span[0] <= span[1] <= n_sub:
        raise ValueError(f"span {span} is outside 0..{n_sub}")
    h = solve_hash(params, options, chash, span)
    if not overwrite and attr(out_path, "settings_hash") == h:
        return None
    extracted = c.frame_lines.size >= params.min_frames
    subs, idx = [], []
    if extracted and span[1] > span[0]:
        c = read_container(container_path, data=True)
        idx = list(range(*span))
        subs = [c.subproblem(k) for k in idx]
    results = solve_subproblem_list(
        subs, params.solver_params(), options, pool
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_results(
        out_path,
        TrialRecord.from_trial(c),
        subproblem_results(subs, results, idx),
        results_attrs(
            c, container_path.name, chash, span, n_sub, extracted, h
        ),
    )
    return {"seconds": time.perf_counter() - t0, "solved": len(subs)}


def _containers(session: SiloSession) -> list:
    """Containers of the session's planned, valid trial-paths."""
    from .silo import _valid_units

    out = []
    for d, t in _valid_units(session):
        f = session.container_file(d, t)
        if f.exists():
            out.append(((d, t), f))
        elif not session.skip_file(d, t).exists():
            raise FileNotFoundError(f"not prepared: {f}")
    return out


def _partition(costs: np.ndarray, n: int) -> list:
    """Cut a sequence into at most ``n`` contiguous parts of about equal
    total cost; returns the parts' ``[start, stop)``."""
    total = float(costs.sum())
    if costs.size == 0:
        return []
    n = max(1, min(n, costs.size))
    edges = [0]
    csum = np.cumsum(costs)
    for j in range(1, n):
        target = total * j / n
        cut = int(np.searchsorted(csum, target))
        # the nearer of the two cuts around the target
        if cut < costs.size and cut > edges[-1]:
            below = csum[cut - 1] if cut > 0 else 0.0
            if target - below > csum[cut] - target:
                cut += 1
        cut = max(cut, edges[-1] + 1)
        if cut >= costs.size:
            break
        edges.append(cut)
    edges.append(costs.size)
    return list(zip(edges[:-1], edges[1:]))


def dispatch(
    session: SiloSession,
    params: SiloParams,
    machines: int,
    execution: Optional[SiloExecution] = None,
) -> list:
    """Write job files: shards of about equal estimated work.

    The session's subproblems (containers in path and trial order, each
    container's in its order) are cut into ``machines`` contiguous parts
    of about equal :func:`subproblem_work`; each part is a job, holding a
    range of subproblems of each container it touches. Earlier job files
    in the folder are removed.

    Parameters
    ----------
    session : SiloSession
    params : SiloParams
    machines : int
        Number of jobs (solver instances).
    execution : SiloExecution, optional
        Device, precision and batch size recorded for the solvers.

    Returns
    -------
    list of Path
        The job files.
    """
    from ..extraction.raster.container import read_container

    if machines < 1:
        raise ValueError("machines must be at least 1")
    ex = execution or SiloExecution()
    options = ex.solver_options()
    entries = []  # (container file, container hash, k, work)
    for _, f in _containers(session):
        c = read_container(f, data=False)
        h = attr(f, "settings_hash")
        n_px = [p.size for p in c.subproblem_pixels]
        n_src = [s.size for s in c.subproblem_sources]
        work = subproblem_work(n_px, n_src, c.frame_lines.size)
        if c.frame_lines.size < params.min_frames or not len(work):
            # nothing to solve: one empty item writes the trial's record
            entries.append((f, h, c.n_subproblems, None, 0.0))
        for k, w in enumerate(work):
            entries.append((f, h, c.n_subproblems, k, float(w)))
    costs = np.array([e[4] for e in entries]) + 1.0
    parts = _partition(costs, machines)
    jobs_dir = session.jobs_dir
    jobs_dir.mkdir(parents=True, exist_ok=True)
    for old in jobs_dir.glob("job_*.json"):
        old.unlink()
    files = []
    for j, (a, b) in enumerate(parts):
        items: list = []
        for f, h, n_sub, k, w in entries[a:b]:
            if items and items[-1]["container"] == f.name and k is not None:
                items[-1]["span"][1] = k + 1
                items[-1]["work"] += w
                continue
            span = [0, n_sub] if k is None else [k, k + 1]
            items.append(
                {
                    "container": f.name,
                    "container_hash": h,
                    "span": span,
                    "work": w,
                }
            )
        for it in items:
            whole = it["span"] == [0, _n_sub(entries, it["container"])]
            it["results"] = (
                Path(it["container"]).stem
                + (
                    ""
                    if whole
                    else "_s%04d-%04d" % (it["span"][0], it["span"][1] - 1)
                )
                + ".h5"
            )
        work = sum(it["work"] for it in items)
        job = {
            "format": JOB_FORMAT,
            "format_version": JOB_VERSION,
            "job": j,
            "n_jobs": len(parts),
            "params": asdict(params),
            "solver": {
                "device": options.device,
                "dtype": options.dtype,
                "batch_bytes": options.batch_bytes,
            },
            "items": items,
            "work": work,
            "estimated_seconds": work * SECONDS_PER_WORK,
        }
        path = jobs_dir / f"job_{j:04d}.json"
        path.write_text(json.dumps(job, indent=1), encoding="utf-8")
        files.append(path)
    return files


def _n_sub(entries: list, name: str) -> int:
    """Subproblems of a container in the dispatch list."""
    for f, _, n_sub, _, _ in entries:
        if f.name == name:
            return n_sub
    return 0


def read_job(path) -> dict:
    """Read and check a job file."""
    job = json.loads(Path(path).read_text(encoding="utf-8"))
    if job.get("format") != JOB_FORMAT:
        raise ValueError(f"{path} is not a SILo job file")
    if int(job.get("format_version", 0)) > JOB_VERSION:
        raise ValueError(
            f"{path}: job format {job['format_version']} is newer"
        )
    return job


def solve_job(
    job_file,
    containers_dir=None,
    results_dir=None,
    execution: Optional[SiloExecution] = None,
) -> dict:
    """Run one job file (stage ``solve`` on one machine).

    Parameters
    ----------
    job_file : str or Path
        The job.
    containers_dir, results_dir : str or Path, optional
        Where the containers are and the results go (default: the job
        folder's siblings ``containers`` and ``results``).
    execution : SiloExecution, optional
        Processes and threads to solve with (the job's device, precision
        and batch size are used).

    Returns
    -------
    dict
        Per item: seconds (None if up to date); total seconds.
    """
    from ..extraction.raster.solver import SolverOptions

    job_file = Path(job_file)
    job = read_job(job_file)
    root = job_file.parent.parent
    containers_dir = Path(containers_dir or root / "containers")
    results_dir = Path(results_dir or root / "results")
    ex = execution or SiloExecution()
    params = SiloParams(**job["params"])
    options = SolverOptions(**job["solver"])
    t0 = time.perf_counter()
    done = {}
    pool = None
    try:
        if ex.solver_workers > 1 and options.batch_bytes is None:
            pool = make_pool(ex.solver_workers, ex.solver_threads)
        for it in job["items"]:
            f = containers_dir / it["container"]
            h = attr(f, "settings_hash")
            if h != it["container_hash"]:
                raise ValueError(
                    f"{f} is not the container the job was made for "
                    "(settings hash differs)"
                )
            done[it["results"]] = solve_container(
                f,
                results_dir / it["results"],
                params,
                options,
                it["span"],
                pool,
                ex.overwrite,
            )
    finally:
        if pool is not None:
            pool.shutdown()
    return {"items": done, "seconds": time.perf_counter() - t0}


def solve_containers(
    session: SiloSession,
    params: SiloParams,
    execution: Optional[SiloExecution] = None,
) -> dict:
    """Solve every container of a session here, one results file each."""
    ex = execution or SiloExecution()
    options = ex.solver_options()
    done = {}
    pool = None
    try:
        if ex.solver_workers > 1 and options.batch_bytes is None:
            pool = make_pool(ex.solver_workers, ex.solver_threads)
        for (d, t), f in _containers(session):
            done[f.name] = solve_container(
                f,
                session.results_file(d, t),
                params,
                options,
                None,
                pool,
                ex.overwrite,
            )
    finally:
        if pool is not None:
            pool.shutdown()
    return done


def thread_env(threads: int) -> None:
    """Set this process's thread counts (before the solver runs)."""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(threads)
    import torch

    torch.set_num_threads(threads)
