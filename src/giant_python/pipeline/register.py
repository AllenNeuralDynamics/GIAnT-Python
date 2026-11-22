"""Register every trial of a SLAP2 raster trial table (MultiRoiRegistration).

Outputs go to ``<savedr>/motion_correction`` with GIAnT-MATLAB's names, and
the registration fields of the trial table's ``motion_correction`` group are
replaced in place (other groups and other ``motion_correction`` entries are
kept), as ``MultiRoiRegistration.m`` does. The table must be MATLAB-written
(column-major).

A request is validated before anything is written: the table's layout, the
device, and every recording that will be registered. A rejected request
leaves the outputs and the table as they were.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np

from ..io.hdf5 import write_matlab_value
from ..io.trial_table import read_trial_table
from ..models.params import (
    ExecutionInput,
    MultiRoiRegistrationParams,
    RegistrationParamsInput,
    resolve_registration_options,
)
from ..parallel import map_trials

log = logging.getLogger(__name__)


def _now() -> str:
    """Local time as MATLAB's ``yyyy-MM-dd'T'HH:mm:ss.SSSZZZZZ``."""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def output_names(
    epoch: int, trial: int, dmd: int, params: MultiRoiRegistrationParams
) -> tuple[str, str]:
    """Registered-movie and alignment-data filenames of one trial.

    Parameters
    ----------
    epoch : int
        The trial's epoch.
    trial, dmd : int
        One-based trial and DMD indices.
    params : MultiRoiRegistrationParams
        Selects ``.tif`` or ``.h5`` and the rate in the name.

    Returns
    -------
    tuple of str
        ``E<epoch>T<trial>DMD<dmd>_REGISTERED_DOWNSAMPLED-<hz>Hz.tif`` (or
        ``.h5``) and ``E<epoch>T<trial>DMD<dmd>_ALIGNMENTDATA.h5``.
    """
    stem = f"E{int(round(epoch))}T{trial}DMD{dmd}"
    ext = ".tif" if params.save_tif else ".h5"
    hz = int(round(params.align_hz))
    return (
        f"{stem}_REGISTERED_DOWNSAMPLED-{hz}Hz{ext}",
        f"{stem}_ALIGNMENTDATA.h5",
    )


def _register_job(job: dict) -> dict:
    """Register one trial; failures are logged and reported, not raised."""
    import torch

    from ..registration.multiroi import register_trial

    if job["threads"]:
        torch.set_num_threads(job["threads"])
    try:
        res = register_trial(
            job["dat"],
            job["first"],
            job["last"],
            job["movie"],
            job["adata"],
            job["params"],
            device=job["device"],
        )
        return {"failed": res.registration_failed, "error": None}
    except Exception as exc:  # reported per trial, as MATLAB continues
        log.exception("registration failed for %s", job["dat"])
        return {"failed": True, "error": repr(exc)}


def _reusable(movie: Path, adata: Path, settings: dict) -> Optional[bool]:
    """Whether a trial's existing outputs can be kept, and their status.

    Outputs are kept only if their alignment file records the requested
    settings, so that ``align_params`` describes every output in the
    table. Outputs made with other settings, or that do not record theirs
    (e.g. MATLAB's), are registered again.

    Parameters
    ----------
    movie, adata : Path
        The trial's registered movie and alignment file.
    settings : dict
        :func:`~giant_python.registration.outputs.settings_record` of the
        requested registration.

    Returns
    -------
    bool or None
        The alignment file's ``registrationFailed`` if the outputs can be
        kept; otherwise None (register the trial again).
    """
    from ..registration.outputs import SETTINGS_ATTR

    if not (movie.exists() and adata.exists()):
        return None
    try:
        with h5py.File(adata, "r") as f:
            saved = f.attrs.get(SETTINGS_ATTR)
            failed = bool(np.asarray(f["registrationFailed"][()]).ravel()[0])
        saved = None if saved is None else json.loads(saved)
    except (OSError, KeyError, IndexError, ValueError):
        log.info("%s cannot be read; registering again", adata.name)
        return None
    if saved != settings:
        log.info(
            "%s does not record the requested settings; registering again",
            adata.name,
        )
        return None
    return failed


def _record(results, where, failed, fn_reg, fn_adata) -> None:
    """Enter each job's outcome in the trial table's grids.

    A trial whose registration raised published nothing; files left from
    an earlier run were made with other settings, so its filenames are
    cleared (the files stay on disk).
    """
    for (d, t), res in zip(where, results):
        failed[d, t] = res["failed"]
        if res["error"] is not None:
            fn_reg[d, t] = fn_adata[d, t] = ""


def _check_column_major(f: h5py.File) -> None:
    """Reject trial tables that are not MATLAB-written (column-major)."""
    flag = f.get("row_major")
    if flag is not None and int(np.asarray(flag[()]).ravel()[0]) != 0:
        raise ValueError(
            "only MATLAB-written (column-major) trial tables can be updated"
        )


def _check_recordings(jobs: list) -> None:
    """Reject the request if a recording to register is not supported.

    Each data file is checked once, before anything is written.
    """
    from ..registration.multiroi import check_recording

    problems = {}
    for dat in dict.fromkeys(job["dat"] for job in jobs):
        try:
            check_recording(dat)
        except (OSError, ValueError) as exc:
            problems[dat] = exc
    if problems:
        listed = "; ".join(f"{p.name}: {e}" for p, e in problems.items())
        raise ValueError(
            f"{len(problems)} recording(s) cannot be registered, so nothing "
            f"was written: {listed}"
        )


def _stage(grp: h5py.Group, fields: dict) -> dict:
    """Write each field under a staged name; remove them all on failure.

    Returns
    -------
    dict
        The staged name of each field.
    """
    # '.' cannot occur in MATLAB field names
    staged = {key: f"{key}.staged" for key in fields}
    try:
        for key, value in fields.items():
            if staged[key] in grp:
                del grp[staged[key]]
            write_matlab_value(grp, staged[key], value)
    except BaseException:
        for name in staged.values():
            if name in grp:
                del grp[name]
        raise
    return staged


def _update_trial_table(path: Path, fields: dict) -> None:
    """Replace ``fields`` in the table's ``motion_correction`` group.

    Other entries of the group (and their attributes) are left as they
    are. Every new value is first written under a staged name; old values
    are deleted only once all are written, so a failure while writing
    leaves the table unchanged.
    """
    with h5py.File(path, "r+") as f:
        _check_column_major(f)
        grp = f.require_group("motion_correction")
        staged = _stage(grp, fields)
        for key, name in staged.items():
            if key in grp:
                del grp[key]
            if name in grp:  # empty values are not written, as in MATLAB
                grp.move(name, key)


def register_multiroi(
    trial_table: Union[str, Path],
    params: RegistrationParamsInput = None,
    *,
    execution: ExecutionInput = None,
) -> dict:
    """Register all trials of a trial table (``MultiRoiRegistration``).

    Parameters
    ----------
    trial_table : str or Path
        A MATLAB-written ``trial_table.h5``; it is updated in place.
    params : MultiRoiRegistrationParams, dict or None
        Scientific settings.
    execution : ExecutionOptions, dict or None
        ``device`` selects CPU or CUDA. On the CPU, up to ``max_workers``
        trials run in parallel processes; on CUDA, trials run one at a time.
        ``max_trials`` is ignored.

    Returns
    -------
    dict
        The ``motion_correction`` fields written: ``fn_reg_ds``,
        ``fn_adata``, ``registration_failed`` and ``align_params``. Trials
        whose registration raised are failed and have empty filenames.

    Raises
    ------
    ValueError
        Before anything is written, for a row-major trial table, one
        without ``slap2_info``, an unavailable device, or a recording to
        register that is not supported (or cannot be read).
    """
    from ..registration.multiroi import resolve_device
    from ..registration.outputs import settings_record

    params, execution = resolve_registration_options(params, execution)
    path = Path(trial_table)
    start = _now()
    with h5py.File(path, "r") as f:
        _check_column_major(f)
    tt = read_trial_table(path)
    if tt.slap2_info is None:
        raise ValueError("the trial table has no slap2_info")
    files = np.atleast_2d(np.asarray(tt.filename, dtype=object))
    n_dmd, n_trials = files.shape
    epoch = np.asarray(tt.epoch, dtype=float).reshape(n_dmd, n_trials)
    first = np.asarray(tt.slap2_info.first_line, dtype=float).reshape(
        n_dmd, n_trials
    )
    last = np.asarray(tt.slap2_info.last_line, dtype=float).reshape(
        n_dmd, n_trials
    )
    moco = Path(tt.savedr) / "motion_correction"
    device = resolve_device(execution.device)
    settings = settings_record(params)
    fn_reg = np.empty((n_dmd, n_trials), dtype=object)
    fn_adata = np.empty((n_dmd, n_trials), dtype=object)
    failed = np.zeros((n_dmd, n_trials), dtype=bool)
    jobs, where = [], []
    for d in range(n_dmd):
        for t in range(n_trials):
            movie, adata = output_names(epoch[d, t], t + 1, d + 1, params)
            fn_reg[d, t], fn_adata[d, t] = movie, adata
            if not params.overwrite_existing:
                saved = _reusable(moco / movie, moco / adata, settings)
                if saved is not None:
                    log.info("%s is already registered; skipping", adata)
                    failed[d, t] = saved  # as recorded when registered
                    continue
            jobs.append(
                {
                    "dat": Path(tt.datadr) / str(files[d, t]),
                    # fractional for continuous recordings; MATLAB keeps
                    # them unrounded (ceil(first:dt:last))
                    "first": float(first[d, t]),
                    "last": float(last[d, t]),
                    "movie": moco / movie,
                    "adata": moco / adata,
                    "params": params,
                    "device": str(device),
                    "threads": None,
                }
            )
            where.append((d, t))
    _check_recordings(jobs)
    moco.mkdir(parents=True, exist_ok=True)
    workers = 1
    if device.type == "cpu":
        workers = max(
            1, min(execution.max_workers, len(jobs), os.cpu_count() or 1)
        )
    if workers > 1:
        for job in jobs:
            job["threads"] = max(1, (os.cpu_count() or 1) // workers)
    results = map_trials(
        _register_job,
        jobs,
        n_workers=workers,
        desc="Registering" if execution.verbose else None,
    )
    _record(results, where, failed, fn_reg, fn_adata)
    align = params.to_matlab()
    align["nWorkers"] = float(workers)
    align["device"] = str(device)
    align["startTime"] = start
    align["endTime"] = _now()
    from .. import __version__

    align["implementation"] = f"giant_python {__version__}"
    fields = {
        "registration_failed": failed,
        "fn_reg_ds": fn_reg,
        "fn_adata": fn_adata,
        "align_params": align,
    }
    _update_trial_table(path, fields)
    return fields
