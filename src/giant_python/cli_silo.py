"""``giant silo`` and its staged commands (raster SILo for SLAP2).

``giant silo TABLE`` runs SILo on one machine, streaming as GIAnT-MATLAB's
``SILo.m``. The stages also run on their own, for any subset of their units,
so that they can be scattered over machines:

* ``giant silo-localize TABLE``: per-trial localization files;
* ``giant silo-plan TABLE``: one extraction plan per path;
* ``giant silo-prepare TABLE``: one container per trial-path;
* ``giant silo-dispatch TABLE --machines N``: solver job files;
* ``giant silo-solve JOB``: one job's results files (needs only the job file
  and the containers it names);
* ``giant silo-collect TABLE``: the kinetics test, ``experiment_summary.h5``,
  ``per_trial_summary.h5`` and the trial table's fields.

Scientific settings are :class:`~giant_python.models.params.SiloParams`
fields: a few have their own options, any can be set with ``--set
name=value`` or a JSON file (``--params``).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Optional

from .models.params import SiloExecution, SiloParams

SILO_COMMANDS = (
    "silo",
    "silo-localize",
    "silo-plan",
    "silo-prepare",
    "silo-dispatch",
    "silo-solve",
    "silo-collect",
)


def parse_numbers(text: Optional[str]) -> Optional[list]:
    """One-based numbers ``"1,3-5"`` as a list; None for None."""
    if text is None:
        return None
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = (int(x) for x in part.split("-", 1))
            if b < a:
                raise ValueError(f"empty range {part!r}")
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    if any(n < 1 for n in out):
        raise ValueError("path and trial numbers start at 1")
    return out


def _add_science(parser: argparse.ArgumentParser) -> None:
    """Scientific settings (SiloParams)."""
    g = parser.add_argument_group("scientific settings (SiloParams)")
    g.add_argument(
        "--params",
        default=None,
        help="JSON file of SiloParams fields (others keep their defaults).",
    )
    g.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Set a SiloParams field (value as JSON, else as text); may be "
        "repeated, and applies after --params.",
    )
    g.add_argument(
        "--activity-channel",
        type=int,
        default=None,
        help="One-based activity channel (default 1).",
    )
    g.add_argument(
        "--peak-th",
        type=float,
        default=None,
        help="Seed threshold, robust SDs (MATLAB peakth; default 3).",
    )
    g.add_argument(
        "--min-kinetic-score",
        type=float,
        default=None,
        help="Kinetics test threshold (default 0.2; 0 keeps all sources).",
    )
    g.add_argument(
        "--footprint-variance",
        choices=("model", "fixed"),
        default=None,
        help="Variance term of the footprint fit (default 'model', "
        "MATLAB's).",
    )
    g.add_argument(
        "--localization-baseline",
        choices=("median", "network"),
        default=None,
        help="Localization baseline: MATLAB's moving median (default) or "
        "the F0 network's *_BASELINE-80Hz.h5 files (--baseline-dir).",
    )
    g.add_argument(
        "--no-trial-peaks",
        action="store_true",
        help="Do not fit each trial's peaks (they feed only the per-trial "
        "summary's per_trial_act_im_peaks and per_trial_num_peaks).",
    )


def _add_session(parser: argparse.ArgumentParser) -> None:
    """The trial table and folder overrides."""
    parser.add_argument(
        "trial_table", help="trial_table.h5 (or the folder holding it)."
    )
    g = parser.add_argument_group("folders")
    g.add_argument(
        "--moco-dir",
        default=None,
        help="Motion-correction folder (default <savedr>/motion_correction).",
    )
    g.add_argument(
        "--data-dir",
        default=None,
        help="Raw data folder (default: the table's datadr).",
    )
    g.add_argument(
        "--out-dir",
        default=None,
        help="Folder of the summaries (default <savedr>/source_extraction).",
    )
    g.add_argument(
        "--work-dir",
        default=None,
        help="Folder of the stages' files (default <out-dir>/silo).",
    )
    g.add_argument(
        "--annotations",
        default=None,
        help="annotations.h5 (default <savedr>/annotations/annotations.h5).",
    )


def _add_execution(
    parser: argparse.ArgumentParser, solver: bool, fits: bool = True
) -> None:
    """How to run (SiloExecution); ``fits``: the solver's device,
    precision and batches (a job file brings its own)."""
    d = SiloExecution
    g = parser.add_argument_group("execution (results do not depend on it)")
    g.add_argument(
        "--workers",
        type=int,
        default=d.workers,
        help=f"Trials localized at once (default {d.workers}).",
    )
    if solver:
        g.add_argument(
            "--solver-workers",
            type=int,
            default=d.solver_workers,
            help=f"Solver processes (default {d.solver_workers}).",
        )
        g.add_argument(
            "--solver-threads",
            type=int,
            default=d.solver_threads,
            help=f"Threads per solver process (default {d.solver_threads}).",
        )
    if solver and fits:
        g.add_argument(
            "--device",
            default=d.device,
            help="Solver device: cpu (default), cuda, cuda:N.",
        )
        g.add_argument(
            "--dtype",
            choices=("float64", "float32"),
            default=d.dtype,
            help="Solver precision (default float64).",
        )
        g.add_argument(
            "--batch-bytes",
            type=float,
            default=None,
            help="Solve subproblems in padded batches of this size (GPU).",
        )
    g.add_argument(
        "--loader-device",
        default=d.loader_device,
        help="Trial loader device (default cpu).",
    )
    g.add_argument(
        "--overwrite",
        action="store_true",
        help="Redo units made with the same settings.",
    )
    g.add_argument("--verbose", action="store_true", help="Print progress.")


def add_silo_commands(subparsers) -> None:
    """Add ``silo`` and the staged commands to ``giant``'s subparsers."""
    run = subparsers.add_parser(
        "silo",
        help="SILo source extraction of SLAP2 raster recordings on one "
        "machine (MATLAB's SILo.m).",
    )
    _add_session(run)
    _add_science(run)
    _add_execution(run, solver=True)
    run.add_argument(
        "--seeds",
        default=None,
        help="Seeds file used instead of the plan's seeding (parity runs).",
    )
    run.add_argument(
        "--baseline-dir",
        default=None,
        help="Folder of the network's *_BASELINE-80Hz.h5 files.",
    )
    run.add_argument(
        "--save-subproblems",
        default=None,
        metavar="DIR",
        help="Also write each trial-path's container to DIR.",
    )
    run.add_argument(
        "--peak-workers",
        type=int,
        default=None,
        help="Processes fitting per-trial peaks in the background.",
    )
    run.add_argument(
        "--no-table-update",
        action="store_true",
        help="Leave the trial table's source_extraction fields as they are.",
    )

    loc = subparsers.add_parser(
        "silo-localize", help="SILo stage: per-trial localization files."
    )
    _add_session(loc)
    _add_science(loc)
    _add_execution(loc, solver=False)
    _add_units(loc)
    loc.add_argument(
        "--baseline-dir",
        default=None,
        help="Folder of the network's *_BASELINE-80Hz.h5 files.",
    )

    pl = subparsers.add_parser(
        "silo-plan", help="SILo stage: one extraction plan per path."
    )
    _add_session(pl)
    _add_science(pl)
    _add_execution(pl, solver=False)
    pl.add_argument("--paths", default=None, help="Paths, e.g. 1,2.")
    pl.add_argument(
        "--seeds",
        default=None,
        help="Seeds file used instead of the plan's seeding (parity runs).",
    )

    prep = subparsers.add_parser(
        "silo-prepare", help="SILo stage: one container per trial-path."
    )
    _add_session(prep)
    _add_science(prep)
    _add_execution(prep, solver=False)
    _add_units(prep)

    disp = subparsers.add_parser(
        "silo-dispatch", help="SILo stage: solver job files."
    )
    _add_session(disp)
    _add_science(disp)
    _add_execution(disp, solver=True)
    disp.add_argument(
        "--machines",
        type=int,
        required=True,
        help="Number of jobs (solver instances).",
    )

    solve = subparsers.add_parser(
        "silo-solve",
        help="SILo stage: solve one job's subproblems (needs only the job "
        "file and its containers).",
    )
    solve.add_argument("job", help="A job file (silo-dispatch).")
    solve.add_argument(
        "--containers",
        default=None,
        help="Folder of the containers (default: ../containers of the job).",
    )
    solve.add_argument(
        "--results",
        default=None,
        help="Folder of the results files (default: ../results of the job).",
    )
    _add_execution(solve, solver=True, fits=False)

    col = subparsers.add_parser(
        "silo-collect",
        help="SILo stage: kinetics test, experiment and per-trial "
        "summaries, trial table fields.",
    )
    _add_session(col)
    _add_science(col)
    _add_execution(col, solver=False)
    col.add_argument(
        "--no-table-update",
        action="store_true",
        help="Leave the trial table's source_extraction fields as they are.",
    )


def _add_units(parser: argparse.ArgumentParser) -> None:
    """Selection of trial-paths."""
    parser.add_argument(
        "--paths", default=None, help="Paths, e.g. 1,2 (default: all)."
    )
    parser.add_argument(
        "--trials",
        default=None,
        help="Trials, one-based, e.g. 1-10,12 (default: all).",
    )


def _value(text: str):
    """A ``--set`` value: JSON if it parses, else the text."""
    try:
        return json.loads(text)
    except ValueError:
        return text


def params_from_args(args: argparse.Namespace) -> SiloParams:
    """SiloParams from ``--params``, ``--set`` and the explicit options."""
    values: dict = {}
    if getattr(args, "params", None):
        values.update(
            json.loads(Path(args.params).read_text(encoding="utf-8"))
        )
    for item in getattr(args, "set", []) or []:
        if "=" not in item:
            raise ValueError(f"--set needs NAME=VALUE, not {item!r}")
        name, text = item.split("=", 1)
        values[name.strip()] = _value(text.strip())
    for name in (
        "activity_channel",
        "peak_th",
        "min_kinetic_score",
        "footprint_variance",
        "localization_baseline",
    ):
        v = getattr(args, name, None)
        if v is not None:
            values[name] = v
    if getattr(args, "no_trial_peaks", False):
        values["trial_peaks"] = False
    known = {f.name for f in fields(SiloParams)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown SiloParams fields: {sorted(unknown)}")
    return SiloParams(**values)


def execution_from_args(args: argparse.Namespace) -> SiloExecution:
    """SiloExecution from the execution options."""
    values = {"workers": args.workers, "overwrite": args.overwrite}
    values["verbose"] = args.verbose
    values["loader_device"] = args.loader_device
    for name in (
        "solver_workers",
        "solver_threads",
        "device",
        "dtype",
        "batch_bytes",
    ):
        if hasattr(args, name):
            values[name] = getattr(args, name)
    return SiloExecution(**values)


def _session(args: argparse.Namespace):
    """The session of the command's trial table and folders."""
    from .pipeline.silo_session import SiloSession

    return SiloSession.open(
        args.trial_table,
        args.moco_dir,
        args.data_dir,
        args.out_dir,
        args.work_dir,
        args.annotations,
    )


def _selected_units(session, args) -> Optional[list]:
    """Zero-based ``(path, trial)`` pairs of ``--paths`` and ``--trials``."""
    paths = parse_numbers(args.paths)
    trials = parse_numbers(args.trials)
    if paths is None and trials is None:
        return None
    paths = paths or list(range(1, session.n_paths + 1))
    trials = trials or list(range(1, session.n_trials + 1))
    return [(d - 1, t - 1) for d in paths for t in trials]


def _print(record) -> None:
    """Print a stage's report as JSON."""
    print(json.dumps(record, indent=1, default=str))


def run_silo_command(args: argparse.Namespace) -> int:  # noqa: C901
    """Run one of the SILo commands; returns the exit code."""
    from .pipeline import silo, silo_collect, silo_solve

    ex = execution_from_args(args)
    if args.command == "silo-solve":
        _print(
            silo_solve.solve_job(args.job, args.containers, args.results, ex)
        )
        return 0
    params = params_from_args(args)
    if args.command == "silo":
        _print(
            silo.run_silo(
                args.trial_table,
                params,
                ex,
                moco_dir=args.moco_dir,
                data_dir=args.data_dir,
                out_dir=args.out_dir,
                work_dir=args.work_dir,
                annotations=args.annotations,
                seeds_file=args.seeds,
                baseline_dir=args.baseline_dir,
                save_subproblems=args.save_subproblems,
                update_table=not args.no_table_update,
                peak_workers=args.peak_workers,
            )
        )
        return 0
    session = _session(args)
    if args.command == "silo-localize":
        _print(
            silo.localize(
                session,
                params,
                ex,
                _selected_units(session, args),
                args.baseline_dir,
            )
        )
    elif args.command == "silo-plan":
        from .extraction.raster.stage_files import read_seeds

        seeds = read_seeds(args.seeds, session.n_paths) if args.seeds else None
        paths = parse_numbers(args.paths)
        _print(
            silo.plan(
                session,
                params,
                ex,
                None if paths is None else [p - 1 for p in paths],
                seeds,
            )
        )
    elif args.command == "silo-prepare":
        _print(
            silo.prepare(session, params, ex, _selected_units(session, args))
        )
    elif args.command == "silo-dispatch":
        files = silo_solve.dispatch(session, params, args.machines, ex)
        _print([str(f) for f in files])
    elif args.command == "silo-collect":
        _print(
            silo_collect.collect(
                session, params, ex, update_table=not args.no_table_update
            )
        )
    return 0
