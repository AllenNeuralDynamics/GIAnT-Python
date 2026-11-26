"""A SILo session: its trial table, folders and stage files.

What every SILo stage needs to find its inputs and outputs: the trial
table's fields (read without its reference stack), the folders (raw data,
motion correction, source extraction, the stages' work folder), which
trial-paths pass file verification (``verifyFiles.m``), and the name of
every stage file.

Work folder layout (default ``<savedr>/source_extraction/silo``; trial and
path numbers one-based, as in MATLAB's file names)::

    localize/path<d>_trial<tttt>.h5     per-trial localization
    plan/path<d>.h5                     the path's extraction plan
    containers/path<d>_trial<tttt>.h5   a trial-path's container
    containers/path<d>_trial<tttt>.skip.json   a trial-path not extracted
    jobs/job_<jjjj>.json                solver jobs (dispatch)
    results/path<d>_trial<tttt>[_s<a>-<b>].h5   solved subproblems
    collect.json                        what the summaries were made from
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePath
from typing import Optional

import h5py
import numpy as np

STAGE_VERSION = 1
"""Changes when a stage's outputs change for the same settings."""

NO_DATA = "none of the selected pixels has a sample"
"""The trial loader's error for a trial without data (the trial is skipped)."""

_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)


def stage_hash(record: dict) -> str:
    """Settings hash of a stage's unit."""
    from ..extraction.raster.container import settings_hash

    return settings_hash({"stage_version": STAGE_VERSION, **record})


def file_id(path: Path) -> dict:
    """A file's identity for a settings hash: name, size and modification
    time (nanoseconds).

    A changed or replaced input gets a new identity; so does a copied one
    (a copy has a new modification time), which only costs a rerun.
    Reading every input for a content hash would cost minutes per trial.
    """
    st = Path(path).stat()
    return {
        "name": Path(path).name,
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }


def recording_ids(dat: Path) -> list:
    """Identities (:func:`file_id`) of the files a raw recording is read
    from: its ``.dat`` file (all files of a CYCLE-split recording) and the
    ``.meta`` files whose names begin its name."""
    import re

    dat = Path(dat)
    folder = dat.parent
    files = [dat]
    base = re.sub(r"-?CYCLE-?\d+$", "", dat.stem, flags=re.IGNORECASE)
    if base != dat.stem:
        files = sorted(
            p
            for p in folder.glob(f"{base}*.dat")
            if re.fullmatch(
                re.escape(base) + r"-?CYCLE-?\d+", p.stem, re.IGNORECASE
            )
        )
    meta = sorted(
        p for p in folder.glob("*.meta") if dat.stem.startswith(p.stem)
    )
    return [file_id(p) for p in files + meta]


def now() -> str:
    """The time as MATLAB's SILo records it."""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def attr(path: Path, name: str):
    """An attribute of an HDF5 file, or None if absent or unreadable."""
    try:
        with h5py.File(path, "r") as f:
            v = f.attrs.get(name)
    except OSError:
        return None
    return v.decode() if isinstance(v, bytes) else v


# ---------------------------------------------------------------- pools


def _worker_init(threads: int) -> None:
    """Set a pool worker's thread counts."""
    for name in _THREAD_VARS:
        os.environ[name] = str(threads)
    import torch

    torch.set_num_threads(threads)


def _warm(_):
    """A first task, so that the pool starts all its workers at once."""
    time.sleep(0.5)
    return os.getpid()


def make_pool(workers: int, threads: int) -> cf.ProcessPoolExecutor:
    """A process pool whose workers use ``threads`` threads each.

    The thread variables are set while the workers start, so that the
    numerical libraries see them when they load.
    """
    saved = {k: os.environ.get(k) for k in _THREAD_VARS}
    for k in _THREAD_VARS:
        os.environ[k] = str(threads)
    try:
        pool = cf.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(threads,),
        )
        list(pool.map(_warm, range(workers)))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return pool


def run_jobs(func, jobs: list, workers: int, threads: int) -> list:
    """Run tasks in a process pool (or here, for one worker)."""
    if not jobs:
        return []
    if workers <= 1 or len(jobs) == 1:
        return [func(j) for j in jobs]
    with make_pool(min(workers, len(jobs)), threads) as pool:
        return list(pool.map(func, jobs))


def _decode(value) -> str:
    """A string element of an HDF5 dataset."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _read(f: h5py.File, name: str, row_major: bool):
    """A dataset in MATLAB orientation (strings decoded), or None."""
    if name not in f:
        return None
    ds = f[name]
    value = ds[()]
    if ds.dtype.kind in ("O", "S", "U"):
        value = np.vectorize(_decode, otypes=[object])(np.asarray(value))
    value = np.asarray(value)
    if not row_major and value.ndim >= 2:
        value = value.T
    return value


def _grid(value, shape) -> np.ndarray:
    """A per-path, per-trial grid of the table's shape."""
    a = np.asarray(value)
    if a.size == shape[0] * shape[1]:
        return a.reshape(shape)
    raise ValueError(f"a trial-table grid has {a.size} entries, not {shape}")


@dataclass
class SiloSession:
    """One SLAP2 session as SILo's stages see it.

    Attributes
    ----------
    table_path : Path
        The trial table.
    datadr : Path
        Folder of the raw ``.dat`` files.
    savedr : Path
        The session's output folder (trial table ``savedr``).
    moco_dir : Path
        Motion-correction folder (``<savedr>/motion_correction``).
    out_dir : Path
        Where the summaries go (``<savedr>/source_extraction``).
    work_dir : Path
        The stages' files (see the module docstring).
    annotations : Path or None
        ``annotations.h5`` (``<savedr>/annotations/annotations.h5``).
    filename, fn_reg_ds, fn_adata : ndarray of object, (paths, trials)
        Raw, registered-movie and alignment file names.
    first_line, last_line : ndarray of float, (paths, trials)
        Each trial's lines in its raw file (``slap2_info``).
    registration_failed : ndarray of bool, (paths, trials)
    keep : ndarray of bool, (paths, trials)
        Trial-paths that pass file verification.
    num_channels : int
    align_hz : float
        Of the registered movies (first kept trial's alignment file).
    """

    table_path: Path
    datadr: Path
    savedr: Path
    moco_dir: Path
    out_dir: Path
    work_dir: Path
    annotations: Optional[Path]
    filename: np.ndarray
    fn_reg_ds: np.ndarray
    fn_adata: np.ndarray
    first_line: np.ndarray
    last_line: np.ndarray
    registration_failed: np.ndarray
    keep: np.ndarray
    num_channels: int
    align_hz: float

    @property
    def n_paths(self) -> int:
        """Imaging paths (DMDs)."""
        return self.filename.shape[0]

    @property
    def n_trials(self) -> int:
        """Trials of the table."""
        return self.filename.shape[1]

    @classmethod
    def open(
        cls,
        table,
        moco_dir=None,
        data_dir=None,
        out_dir=None,
        work_dir=None,
        annotations=None,
    ) -> "SiloSession":
        """Read a trial table and verify the session's files.

        Parameters
        ----------
        table : str or Path
            ``trial_table.h5``, or the folder holding it.
        moco_dir, data_dir, out_dir, work_dir : str or Path, optional
            Overrides of ``<savedr>/motion_correction``, the table's
            ``datadr``, ``<savedr>/source_extraction`` and
            ``<out_dir>/silo``.
        annotations : str or Path, optional
            The annotations file (default:
            ``<savedr>/annotations/annotations.h5`` if it exists).

        Returns
        -------
        SiloSession

        Raises
        ------
        ValueError
            If no trial-path passes file verification, or the table is
            not a SLAP2 table.
        """
        table = Path(table)
        if table.is_dir():
            table = table / "trial_table.h5"
        with h5py.File(table, "r") as f:
            row_major = False
            if "row_major" in f:
                row_major = bool(np.asarray(f["row_major"][()]).ravel()[0])
            filename = np.atleast_2d(_read(f, "filename", row_major))
            shape = filename.shape
            fields = {
                k: _read(f, f"motion_correction/{k}", row_major)
                for k in ("fn_reg_ds", "fn_adata", "registration_failed")
            }
            first = _read(f, "slap2_info/first_line", row_major)
            last = _read(f, "slap2_info/last_line", row_major)
            datadr = _read(f, "datadr", row_major)
            savedr = _read(f, "savedr", row_major)
        if first is None or last is None:
            raise ValueError(f"{table} is not a SLAP2 trial table")
        if fields["fn_reg_ds"] is None or fields["fn_adata"] is None:
            raise ValueError(f"{table} has no registration (giant register)")
        savedr = Path(
            str(savedr.ravel()[0]) if savedr is not None else table.parent
        )
        datadr = Path(
            data_dir
            if data_dir is not None
            else (
                str(datadr.ravel()[0]) if datadr is not None else table.parent
            )
        )
        moco_dir = Path(moco_dir or savedr / "motion_correction")
        out_dir = Path(out_dir or savedr / "source_extraction")
        work_dir = Path(work_dir or out_dir / "silo")
        if annotations is None:
            default = savedr / "annotations" / "annotations.h5"
            annotations = default if default.exists() else None
        failed = fields["registration_failed"]
        failed = (
            np.zeros(shape, bool)
            if failed is None or np.size(failed) != shape[0] * shape[1]
            else _grid(failed, shape).astype(bool)
        )
        session = cls(
            table_path=table,
            datadr=datadr,
            savedr=savedr,
            moco_dir=moco_dir,
            out_dir=out_dir,
            work_dir=work_dir,
            annotations=Path(annotations) if annotations else None,
            filename=filename,
            fn_reg_ds=_grid(fields["fn_reg_ds"], shape),
            fn_adata=_grid(fields["fn_adata"], shape),
            first_line=_grid(first, shape).astype(np.float64),
            last_line=_grid(last, shape).astype(np.float64),
            registration_failed=failed,
            keep=np.zeros(shape, bool),
            num_channels=0,
            align_hz=0.0,
        )
        session.verify()
        return session

    def movie(self, d: int, t: int) -> Optional[Path]:
        """Trial-path ``(d, t)``'s registered movie (``.tif`` or ``.h5``),
        if it exists."""
        name = str(self.fn_reg_ds[d, t] or "")
        if not name:
            return None
        stem = PurePath(name.replace("\\", "/")).stem
        for ext in (".tif", ".h5"):
            p = self.moco_dir / (stem + ext)
            if p.exists():
                return p
        return None

    def alignment(self, d: int, t: int) -> Path:
        """Trial-path ``(d, t)``'s alignment file."""
        name = PurePath(str(self.fn_adata[d, t]).replace("\\", "/")).stem
        return self.moco_dir / (name + ".h5")

    def raw(self, d: int, t: int) -> Path:
        """Trial-path ``(d, t)``'s raw ``.dat`` file."""
        return self.datadr / str(self.filename[d, t])

    def verify(self) -> None:
        """Set :attr:`keep` as ``verifyFiles.m``: the registered movie, the
        alignment file and the raw file exist, and registration did not
        fail; read the channels and frame rate of the first kept trial."""
        keep = np.zeros(self.filename.shape, bool)
        for d in range(self.n_paths):
            for t in range(self.n_trials):
                if not str(self.filename[d, t] or ""):
                    continue
                keep[d, t] = (
                    self.movie(d, t) is not None
                    and self.alignment(d, t).exists()
                    and not self.registration_failed[d, t]
                    and self.raw(d, t).exists()
                )
        if not keep.any():
            raise ValueError(
                "no trial passes file verification (registered movie, "
                "alignment file, raw file, registration not failed)"
            )
        self.keep = keep
        d, t = (int(i[0]) for i in np.nonzero(keep))
        with h5py.File(self.alignment(d, t), "r") as f:
            self.num_channels = int(
                np.asarray(f["numChannels"][()]).ravel()[0]
            )
            self.align_hz = float(np.asarray(f["alignHz"][()]).ravel()[0])

    # stage files
    def localize_file(self, d: int, t: int) -> Path:
        """Localization file of trial-path ``(d, t)`` (zero-based)."""
        return self.work_dir / "localize" / f"path{d + 1}_trial{t + 1:04d}.h5"

    def plan_file(self, d: int) -> Path:
        """Plan file of path ``d`` (zero-based)."""
        return self.work_dir / "plan" / f"path{d + 1}.h5"

    def container_file(self, d: int, t: int) -> Path:
        """Container of trial-path ``(d, t)``."""
        return (
            self.work_dir / "containers" / f"path{d + 1}_trial{t + 1:04d}.h5"
        )

    def skip_file(self, d: int, t: int) -> Path:
        """Marker of a trial-path that is not extracted."""
        return self.container_file(d, t).with_suffix(".skip.json")

    def results_file(self, d: int, t: int, span=None) -> Path:
        """Results file of trial-path ``(d, t)`` (or of subproblems
        ``span = (a, b)``, ``a`` to ``b - 1``)."""
        stem = f"path{d + 1}_trial{t + 1:04d}"
        if span is not None:
            stem += f"_s{span[0]:04d}-{span[1] - 1:04d}"
        return self.work_dir / "results" / f"{stem}.h5"

    @property
    def results_dir(self) -> Path:
        """Folder of the results files."""
        return self.work_dir / "results"

    @property
    def jobs_dir(self) -> Path:
        """Folder of the solver jobs."""
        return self.work_dir / "jobs"


def write_skip(
    path: Path, reason: str, settings_hash: str, plan_hash: str = ""
) -> None:
    """Mark a trial-path as not extracted (``*.skip.json``: the reason, the
    settings hash of the prepare stage and the plan's hash)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.partial")
    record = {
        "reason": reason,
        "settings_hash": settings_hash,
        "plan_hash": plan_hash,
    }
    tmp.write_text(json.dumps(record), encoding="utf-8")
    os.replace(tmp, path)


def skip_hash(path: Path, key: str = "settings_hash") -> Optional[str]:
    """A hash recorded in a skip marker (``settings_hash`` or
    ``plan_hash``; None if there is no marker)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))[key]
    except (OSError, ValueError, KeyError, TypeError):
        return None
