"""The F0 network's baselines for SILo (stage ``baseline``).

``slap2-f0 predict-registered`` (slap2-F0-network) writes one
``<trial>_BASELINE-80Hz.h5`` per registered trial-path. GIAnT-Python does
not import slap2-f0: :func:`predict_baselines` runs its command line, when
it is installed, for the trial-paths whose baseline file is missing
(``giant silo --f0-ckpt PATH``). Elsewhere (Code Ocean) the prediction is
its own step and the files are given with ``--baseline-dir``.

slap2-f0's files do not record how the trial was prepared for the network
(channel, thresholds). :func:`predict_baselines` therefore writes a record
beside each file it makes (``<trial>_BASELINE-80Hz.json``: those settings,
the movie, alignment and baseline files' identities, and the checkpoint's
SHA-256 at prediction time), and :func:`check_baseline_record` refuses a
baseline whose record does not fit the run. A file without a record (made
elsewhere) is used as it is: its checkpoint is then known only by the
path and step the file names.

Files are identified by name, size and modification time. A record written
with ``digests=True`` also holds each file's SHA-256, so that it still
identifies the files after they are copied or moved (a Code Ocean data
asset's files carry their upload times): a file whose modification time
differs is then accepted if its content has the recorded SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
import warnings
from pathlib import Path
from typing import Optional

from ..models.params import SiloParams
from .silo_session import SiloSession

F0_COMMAND = "slap2-f0"
"""The slap2-F0-network command line (found on the PATH)."""

RECORD_FORMAT = "giant-silo-baseline-record"

# the run's settings that the prediction prepares the trial with
_PREDICTION_SETTINGS = (
    "activity_channel",
    "nan_thresh",
    "motion_thresh",
    "discard_initial_s",
)

_SHA_CACHE: dict = {}


def file_sha256(path) -> str:
    """A file's SHA-256 (cached per path, size and modification time)."""
    path = Path(path)
    st = path.stat()
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns)
    if key not in _SHA_CACHE:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        _SHA_CACHE[key] = h.hexdigest()
    return _SHA_CACHE[key]


def checkpoint_identity(path) -> dict:
    """A checkpoint's name, size and SHA-256 (cached per path, size and
    modification time)."""
    path = Path(path)
    sha = file_sha256(path)
    return {"name": path.name, "size": path.stat().st_size, "sha256": sha}


def baseline_path(baseline_dir, movie: Path) -> Path:
    """The network baseline file of a registered movie."""
    from ..extraction.raster.baseline_map import baseline_file

    return baseline_file(baseline_dir, Path(movie).name)


def f0_command(command: Optional[str] = None) -> str:
    """The slap2-f0 executable: ``command`` or ``slap2-f0`` on the PATH.

    Raises
    ------
    FileNotFoundError
        If it is not found.
    """
    found = shutil.which(command or F0_COMMAND)
    if found is None:
        raise FileNotFoundError(
            f"{command or F0_COMMAND!r} not found: install slap2-F0-network "
            "(slap2-f0) or give its executable (--f0-command), or give the "
            "baselines' folder (--baseline-dir)"
        )
    return found


def record_path(baseline: Path) -> Path:
    """The record of a baseline file (``<trial>_BASELINE-80Hz.json``)."""
    return Path(baseline).with_suffix(".json")


def _file_id(path: Path) -> dict:
    """A file's identity (name, size, modification time)."""
    from .silo_session import file_id

    return file_id(Path(path))


def _identity(path: Path, digest: bool) -> dict:
    """A file's identity for a record: :func:`_file_id`, with its SHA-256
    when ``digest``."""
    out = _file_id(path)
    if digest:
        out["sha256"] = file_sha256(path)
    return out


def same_file(recorded: Optional[dict], path: Path) -> bool:
    """Whether a recorded identity is ``path``'s: the same name and size,
    and the same modification time, or (a file moved or copied since, e.g.
    into a data asset, whose files carry their upload times) the same
    SHA-256 when the record has one."""
    if not recorded:
        return False
    now = _file_id(path)
    if (
        recorded.get("name") != now["name"]
        or recorded.get("size") != now["size"]
    ):
        return False
    if recorded.get("mtime_ns") == now["mtime_ns"]:
        return True
    sha = recorded.get("sha256")
    return bool(sha) and sha == file_sha256(path)


def prediction_settings(params: SiloParams) -> dict:
    """The run's settings that a baseline's prediction depends on."""
    return {n: getattr(params, n) for n in _PREDICTION_SETTINGS}


def write_baseline_record(
    baseline: Path,
    movie: Path,
    alignment: Path,
    params,
    checkpoint: dict,
    done: bool = True,
    digests: bool = False,
) -> Path:
    """Record how a baseline file was predicted (see the module).

    Written before the prediction runs with ``done=False`` (the baseline
    file's identity left empty), so that a file written by a prediction
    that then fails or is stopped still has its settings; and again with
    the file's identity once it is there.

    With ``digests`` the baseline, movie and alignment files' identities
    also hold their SHA-256, so that the record still identifies them after
    they are copied or moved to another place (:func:`same_file`), for
    example when the baselines and the registration become data assets.
    """
    path = record_path(baseline)
    rec = {
        "format": RECORD_FORMAT,
        "baseline": _identity(baseline, digests) if done else None,
        "movie": _identity(movie, digests),
        "alignment": _identity(alignment, digests),
        "settings": prediction_settings(params),
        "checkpoint": checkpoint,
    }
    path.write_text(json.dumps(rec, indent=1, sort_keys=True))
    return path


def check_baseline_record(
    baseline: Path, movie: Optional[Path], alignment: Path, params
) -> Optional[dict]:
    """The record of a baseline file, checked against the run.

    Returns
    -------
    dict or None
        The record; None for a file without one (made elsewhere).

    Raises
    ------
    ValueError
        If the record is not of this file (replaced since), or the file was
        predicted with other settings, or from another movie or alignment
        than the run's. Files are compared by :func:`same_file`: a file
        moved or copied since (another modification time) passes only if
        the record has its SHA-256 and the content still has it.
    """
    path = record_path(baseline)
    if not path.exists():
        return None
    rec = json.loads(path.read_text())
    again = f"predict {Path(baseline).name} again (remove it and its record)"
    # a record without the file's identity was written before a prediction
    # that did not finish: the file (written whole by slap2-f0) is its
    if rec.get("baseline") is not None and not same_file(
        rec["baseline"], baseline
    ):
        raise ValueError(
            f"{path.name} does not describe {Path(baseline).name} (replaced "
            f"since?): remove the record, or {again}"
        )
    given = prediction_settings(params)
    if rec.get("settings") != given:
        raise ValueError(
            f"{Path(baseline).name} was predicted with {rec.get('settings')}, "
            f"this run has {given}: {again}"
        )
    for name, f in (("movie", movie), ("alignment", alignment)):
        if f is not None and not same_file(rec.get(name), f):
            raise ValueError(
                f"{Path(baseline).name} was predicted from another {name} "
                f"file than {Path(f).name} is now: {again}"
            )
    return rec


def _units(session: SiloSession, units=None) -> list:
    """Kept trial-paths ``(d, t)`` of ``units`` or all."""
    from .silo import _units as kept_units

    return kept_units(session, units)


def missing_baselines(session: SiloSession, baseline_dir, units=None) -> list:
    """Kept trial-paths without a baseline file: ``[(d, t, movie), ...]``."""
    out = []
    for d, t in _units(session, units):
        movie = session.movie(d, t)
        if not baseline_path(baseline_dir, movie).exists():
            out.append((d, t, movie))
    return out


def f0_movie(moco_dir, stem: str) -> Optional[Path]:
    """The registered movie of a trial that ``predict-registered`` reads:
    ``.tif`` or ``.h5``, the newer where both exist (slap2-f0's
    ``registered_movies``)."""
    chosen = None
    for ext in (".tif", ".h5"):
        p = Path(moco_dir) / f"{stem}_REGISTERED_DOWNSAMPLED-80Hz{ext}"
        if p.exists() and (
            chosen is None or p.stat().st_mtime > chosen.stat().st_mtime
        ):
            chosen = p
    return chosen


def _check_movies(session: SiloSession, todo: list) -> None:
    """Refuse to predict from another movie than SILo reads (a trial with
    both a TIFF and an HDF5 movie whose newer one is not SILo's)."""
    for _, _, movie in todo:
        stem = movie.name[: movie.name.index("_REGISTERED")]
        other = f0_movie(session.moco_dir, stem)
        if other is not None and other.name != movie.name:
            raise ValueError(
                f"slap2-f0 would predict {stem} from {other.name}, SILo "
                f"reads {movie.name}: remove one of them"
            )


def _check_kept(session, params, baseline_dir, units, todo, identity):
    """The baseline files that are kept must fit the run: their records
    (:func:`check_baseline_record`), and the checkpoint they were predicted
    with must be this one (by its SHA-256), so that one run does not mix
    the predictions of two models."""
    for d, t in _units(session, units):
        if (d, t) in todo:
            continue
        base = baseline_path(baseline_dir, session.movie(d, t))
        rec = check_baseline_record(
            base, session.movie(d, t), session.alignment(d, t), params
        )
        made = (rec or {}).get("checkpoint") or {}
        if made.get("sha256") and made["sha256"] != identity["sha256"]:
            raise ValueError(
                f"{base.name} was predicted with another checkpoint "
                f"({made.get('name')}, SHA-256 {made['sha256'][:12]}...) "
                f"than {identity['name']} ({identity['sha256'][:12]}...): "
                "remove it and its record to predict it again, or give "
                "that checkpoint"
            )


def _check_checkpoint(session, baseline_dir, units, ckpt: Path) -> None:
    """Warn when existing baseline files name another checkpoint."""
    import h5py

    others = set()
    for d, t in _units(session, units):
        f = baseline_path(baseline_dir, session.movie(d, t))
        if not f.exists():
            continue
        with h5py.File(f, "r") as h:
            name = h.attrs.get("checkpoint")
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")
        if name is not None and Path(str(name)).name != ckpt.name:
            others.add(Path(str(name)).name)
    if others:
        warnings.warn(
            f"baseline files in {baseline_dir} were made with "
            f"{sorted(others)}, not {ckpt.name}; they are used as they are "
            "(remove them to predict them again)"
        )


def _run(args: list, log: Path, verbose: bool):
    """Run a command, its output shown or appended to ``log``."""
    env = dict(os.environ)
    if verbose:
        return subprocess.run(args, env=env)
    with open(log, "a", encoding="utf-8") as out:
        out.write(" ".join(args) + "\n")
        out.flush()
        return subprocess.run(
            args, stdout=out, stderr=subprocess.STDOUT, env=env
        )


def _predict_args(
    exe, session, params, ckpt, baseline_dir, todo, device, out_frames
) -> list:
    """The ``predict-registered`` command line for the trial-paths
    ``todo`` (the run's trial preparation settings)."""
    keys = sorted({m.name[: m.name.index("_REGISTERED")] for _, _, m in todo})
    args = [
        exe,
        "predict-registered",
        str(session.moco_dir),
        "--ckpt",
        str(ckpt),
        "--out",
        str(baseline_dir),
        "--files",
        ",".join(f"{k}_REGISTERED" for k in keys),
        "--channel",
        str(params.activity_channel),
        "--nan_thresh",
        repr(float(params.nan_thresh)),
        "--motion_thresh",
        repr(float(params.motion_thresh)),
        "--discard_initial_s",
        repr(float(params.discard_initial_s)),
        "--device",
        device,
    ]
    if out_frames:
        args += ["--out_frames", str(int(out_frames))]
    return args


def predict_baselines(
    session: SiloSession,
    params: SiloParams,
    baseline_dir,
    ckpt,
    units=None,
    device: str = "cuda",
    command: Optional[str] = None,
    verbose: bool = False,
    out_frames: Optional[int] = None,
    digests: bool = False,
) -> dict:
    """Predict the missing baselines with ``slap2-f0 predict-registered``.

    One process predicts every missing trial-path (the model is loaded
    once). The prediction prepares each trial as SILo does, with the run's
    ``activity_channel``, ``nan_thresh``, ``motion_thresh`` and
    ``discard_initial_s``.

    Parameters
    ----------
    session : SiloSession
    params : SiloParams
    baseline_dir : str or Path
        Where the files are (and are written).
    ckpt : str or Path
        The slap2-f0 checkpoint.
    units : iterable of (int, int), optional
        Zero-based ``(path, trial)`` pairs (default: every kept one).
    device : str
        ``predict-registered``'s device ('cuda' by default; 'cpu' works,
        at many times the time; 'auto' takes the GPU if there is one).
    command : str, optional
        The slap2-f0 executable (default: ``slap2-f0`` on the PATH).
    verbose : bool
        Show its output (otherwise it goes to ``predict.log`` in
        ``baseline_dir``).
    out_frames : int, optional
        Frames per tile of the network (``--out_frames``; default: as many
        as fit in 85% of the free GPU memory). On a GPU that also drives a
        display, memory beyond what is free spills into shared system
        memory instead of failing, several times slower; a fixed size
        (512 on an 8 GB card) avoids that. Tiles overlap by the network's
        reach, so the baseline is the same (to float16 rounding).
    digests : bool
        Also record the SHA-256 of the baseline, movie and alignment files
        (:func:`write_baseline_record`), so that the records still identify
        them after the files are copied or moved (data assets). Costs one
        read of each file.

    Returns
    -------
    dict
        ``predicted`` (one-based ``[path, trial]`` pairs), ``seconds``,
        ``checkpoint`` (:func:`checkpoint_identity`).

    Raises
    ------
    RuntimeError
        If the command fails or leaves a file missing.
    """
    ckpt = Path(ckpt)
    if not ckpt.is_file():
        raise FileNotFoundError(f"no checkpoint {ckpt}")
    baseline_dir = Path(baseline_dir)
    units = None if units is None else list(units)  # read more than once
    identity = checkpoint_identity(ckpt)
    _check_checkpoint(session, baseline_dir, units, ckpt)
    todo = missing_baselines(session, baseline_dir, units)
    have = {(d, t) for d, t, _ in todo}
    _check_kept(session, params, baseline_dir, units, have, identity)
    record = {"predicted": [], "seconds": 0.0, "checkpoint": identity}
    if not todo:
        return record
    _check_movies(session, todo)
    exe = f0_command(command)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    for d, t, movie in todo:  # the settings, before any file is written
        write_baseline_record(
            baseline_path(baseline_dir, movie),
            movie,
            session.alignment(d, t),
            params,
            identity,
            done=False,
            digests=digests,
        )
    args = _predict_args(
        exe, session, params, ckpt, baseline_dir, todo, device, out_frames
    )
    t0 = time.perf_counter()
    log = baseline_dir / "predict.log"
    proc = _run(args, log, verbose)
    record["seconds"] = time.perf_counter() - t0
    for d, t, movie in todo:  # the files written, also by a failed run
        if baseline_path(baseline_dir, movie).exists():
            write_baseline_record(
                baseline_path(baseline_dir, movie),
                movie,
                session.alignment(d, t),
                params,
                identity,
                digests=digests,
            )
    if proc.returncode != 0:
        raise RuntimeError(
            f"slap2-f0 predict-registered exited with {proc.returncode}"
            + ("" if verbose else f" (see {log})")
        )
    left = missing_baselines(session, baseline_dir, units)
    if left:
        raise RuntimeError(
            "slap2-f0 predict-registered wrote no baseline for "
            + ", ".join(m.name for _, _, m in left)
        )
    record["predicted"] = [[d + 1, t + 1] for d, t, _ in todo]
    return record
