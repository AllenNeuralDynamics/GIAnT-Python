"""Register every trial of a SLAP2 raster trial table (MultiRoiRegistration).

Outputs go to ``<savedr>/motion_correction`` with GIAnT-MATLAB's names, and
the trial table's ``motion_correction`` group is rewritten in place (other
groups and other ``motion_correction`` fields are kept), as
``MultiRoiRegistration.m`` does. The table must be MATLAB-written
(column-major).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Union

import h5py
import numpy as np

from ..io.hdf5 import _read_group, write_matlab_value
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


def _update_trial_table(path: Path, fields: dict) -> None:
    """Rewrite the table's ``motion_correction`` group, keeping its others."""
    with h5py.File(path, "r+") as f:
        flag = f.get("row_major")
        if flag is not None and int(np.asarray(flag[()]).ravel()[0]) != 0:
            raise ValueError(
                "only MATLAB-written (column-major) trial tables can be "
                "updated"
            )
        existing = {}
        if "motion_correction" in f:
            existing = _read_group(f["motion_correction"], row_major=False)
            del f["motion_correction"]
        existing.update(fields)
        write_matlab_value(f, "motion_correction", existing)


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
        ``fn_adata``, ``registration_failed`` and ``align_params``.
    """
    from ..registration.multiroi import resolve_device

    params, execution = resolve_registration_options(params, execution)
    path = Path(trial_table)
    start = _now()
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
    moco.mkdir(parents=True, exist_ok=True)
    device = resolve_device(execution.device)
    fn_reg = np.empty((n_dmd, n_trials), dtype=object)
    fn_adata = np.empty((n_dmd, n_trials), dtype=object)
    failed = np.zeros((n_dmd, n_trials), dtype=bool)
    jobs, where = [], []
    for d in range(n_dmd):
        for t in range(n_trials):
            movie, adata = output_names(epoch[d, t], t + 1, d + 1, params)
            fn_reg[d, t], fn_adata[d, t] = movie, adata
            if (
                not params.overwrite_existing
                and (moco / movie).exists()
                and (moco / adata).exists()
            ):
                log.info("%s is already registered; skipping", adata)
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
    for (d, t), res in zip(where, results):
        failed[d, t] = res["failed"]
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
