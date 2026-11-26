"""The F0 network's baselines for SILo (stage ``baseline``).

``slap2-f0 predict-registered`` (slap2-F0-network) writes one
``<trial>_BASELINE-80Hz.h5`` per registered trial-path. GIAnT-Python does
not import slap2-f0: :func:`predict_baselines` runs its command line, when
it is installed, for the trial-paths whose baseline file is missing
(``giant silo --f0-ckpt PATH``). Elsewhere (Code Ocean) the prediction is
its own step and the files are given with ``--baseline-dir``.
"""

from __future__ import annotations

import hashlib
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

_SHA_CACHE: dict = {}


def checkpoint_identity(path) -> dict:
    """A checkpoint's name, size and SHA-256 (cached per path, size and
    modification time)."""
    path = Path(path)
    st = path.stat()
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns)
    if key not in _SHA_CACHE:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        _SHA_CACHE[key] = h.hexdigest()
    return {"name": path.name, "size": st.st_size, "sha256": _SHA_CACHE[key]}


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


def predict_baselines(
    session: SiloSession,
    params: SiloParams,
    baseline_dir,
    ckpt,
    units=None,
    device: str = "cuda",
    command: Optional[str] = None,
    verbose: bool = False,
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
    identity = checkpoint_identity(ckpt)
    _check_checkpoint(session, baseline_dir, units, ckpt)
    todo = missing_baselines(session, baseline_dir, units)
    record = {"predicted": [], "seconds": 0.0, "checkpoint": identity}
    if not todo:
        return record
    exe = f0_command(command)
    baseline_dir.mkdir(parents=True, exist_ok=True)
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
    t0 = time.perf_counter()
    log = baseline_dir / "predict.log"
    env = dict(os.environ)
    if verbose:
        proc = subprocess.run(args, env=env)
    else:
        with open(log, "a", encoding="utf-8") as out:
            out.write(" ".join(args) + "\n")
            out.flush()
            proc = subprocess.run(
                args, stdout=out, stderr=subprocess.STDOUT, env=env
            )
    record["seconds"] = time.perf_counter() - t0
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
