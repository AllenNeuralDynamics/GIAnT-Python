"""``giant reg-stats``: summary statistics of trials registered earlier.

The same statistics that registration computes while it writes a movie
(:mod:`giant_python.registration.stats`), for movies that already exist:
GIAnT-Python's (TIFF or HDF5) and GIAnT-MATLAB's. Each trial's raw recording
is read again for the photon-noise model (gain curves and the kernel
geometry of its frames); the frames themselves come from the registered
movie and ``varFacDS``, and pass through the same accumulator in the same
order, so the maps equal registration's (bit for bit on the same device).

The frame times, motion, registration window and online compensation are
taken from the alignment file and checked against the frame source built
from the raw recording; a mismatch (other settings than the file's) is an
error, not a silent difference.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional, Sequence, Union

import h5py
import numpy as np

from ..io.trial_table import read_trial_table
from ..parallel import map_trials
from .register import recorded_cells

log = logging.getLogger(__name__)

FRAMES_PER_READ = 64
"""Frames read from the movie at a time."""


def _scalar(f: h5py.File, name: str, default=None):
    """A scalar of a MATLAB-layout file (``default`` if absent)."""
    if name not in f:
        return default
    return np.asarray(f[name][()]).ravel()[0]


def _vector(f: h5py.File, name: str) -> np.ndarray:
    """A vector of a MATLAB-layout file, flattened."""
    return np.asarray(f[name][()], dtype=np.float64).ravel()


def read_alignment(path: Union[str, Path]) -> dict:
    """What the statistics need from an alignment file.

    Returns
    -------
    dict
        ``ds`` (int64 frame times), ``motion`` ``(frames, 2)``,
        ``n_channels``, ``align_hz``, ``trim_rows``, ``trim_cols`` (one-based),
        ``out_shape`` (rows, cols), ``compensated``, ``compensation`` (the
        saved ``slap2/onlineCompensation`` record when enabled, else None)
        and ``settings`` (the recorded registration settings, MATLAB names,
        or None).
    """
    with h5py.File(path, "r") as f:
        row_major = "row_major" in f and bool(np.asarray(f["row_major"]).any())
        ds = _vector(f, "DSframes")
        out = {
            "ds": ds.astype(np.int64),
            "motion": np.stack(
                [_vector(f, "motionDSr"), _vector(f, "motionDSc")], 1
            ),
            "n_channels": int(_scalar(f, "numChannels")),
            "align_hz": float(_scalar(f, "alignHz")),
            "trim_rows": _vector(f, "slap2/trimRows").astype(np.int64),
            "trim_cols": _vector(f, "slap2/trimCols").astype(np.int64),
            "out_shape": _out_shape(f["slap2"], row_major),
            "compensated": bool(
                _scalar(f, "slap2/onlineCompensation/enabled", 0)
            ),
            "settings": None,
        }
        out["compensation"] = (
            _compensation_record(f["slap2/onlineCompensation"], row_major)
            if out["compensated"]
            else None
        )
        saved = f.attrs.get("align_params")
    if saved is not None:
        out["settings"] = json.loads(saved)
    if not np.array_equal(out["ds"], ds):
        raise ValueError(f"{path}: DSframes are not whole lines")
    return out


def _matlab_2d(ds: h5py.Dataset, row_major: bool) -> np.ndarray:
    """A 2-D dataset in MATLAB's orientation."""
    a = np.asarray(ds[()], dtype=np.float64)
    return a if row_major else a.T


def _out_shape(grp: h5py.Group, row_major: bool) -> tuple:
    """``(rows, cols)`` of the registered frame, from ``viewR`` and
    ``viewC`` (as the SILo loader reads them)."""
    rows = _matlab_2d(grp["viewR"], row_major)[:, 0]
    cols = _matlab_2d(grp["viewC"], row_major)[0, :]
    return rows.size, cols.size


def _compensation_record(grp: h5py.Group, row_major: bool) -> dict:
    """The fields of a saved ``onlineCompensation`` record that decide
    where samples are placed."""
    return {
        "refOffset": _vector(grp, "refOffset"),
        "frameRefX": _vector(grp, "frameRefX"),
        "frameRefY": _vector(grp, "frameRefY"),
        "mapping": _matlab_2d(grp["mapping"], row_major),
        "yLead": _vector(grp, "yLead"),
    }


def _check_compensation(dat, source, ds, saved: Optional[dict]) -> None:
    """Refuse a trial whose online compensation, rebuilt from the raw
    recording, is not the one its registration saved (other offsets,
    references, mapping or y lead would place the samples elsewhere)."""
    from ..registration.compensation import compensation_record

    if source.compensation is None:
        return
    rec = compensation_record(source.compensation, source.references(ds))
    for key, value in saved.items():
        mine = np.asarray(rec[key], dtype=np.float64).ravel()
        if not np.array_equal(mine, np.asarray(value).ravel()):
            raise ValueError(
                f"{dat}: online compensation ({key}) differs from the "
                "alignment file's record"
            )


def registration_params(alignment: dict, table_settings: Optional[dict]):
    """The settings that formed the trial's frames.

    The alignment file's own record (GIAnT-Python), else the trial table's
    ``align_params``, else the defaults; online compensation always as the
    alignment file says.
    """
    from ..models.params import MultiRoiRegistrationParams

    rec = alignment["settings"] or table_settings or {}
    kw = {}
    for field, key in (
        ("init_frames", "initFrames"),
        ("maxshift", "maxshift"),
    ):
        if key in rec and rec[key] is not None:
            kw[field] = int(np.asarray(rec[key]).ravel()[0])
    kw["align_hz"] = alignment["align_hz"]
    kw["compensate_online"] = alignment["compensated"]
    return MultiRoiRegistrationParams(**kw)


def _source(dat, first, last, alignment, params, device):
    """The trial's frame source, checked against the alignment file."""
    from ..registration.multiroi import open_trial
    from ..registration.sources import make_source

    layout, former, init, ds, dtk = open_trial(
        dat, first, last, params, device
    )
    if not np.array_equal(ds, alignment["ds"]):
        raise ValueError(
            f"{dat}: frame times differ from the alignment file's "
            f"({ds.size} vs {alignment['ds'].size} frames)"
        )
    source = make_source(
        layout, former, first, last, init, ds, dtk, params.compensate_online
    )
    win = source.window
    if not (
        np.array_equal(win.trim_rows, alignment["trim_rows"])
        and np.array_equal(win.trim_cols, alignment["trim_cols"])
    ):
        raise ValueError(f"{dat}: registration window differs from trimRows")
    _check_compensation(dat, source, ds, alignment.get("compensation"))
    return layout, source, ds


def _movie_reader(path: Path, n_channels: int):
    """Iterator over ``(frame, (channels, rows, cols))`` float32 frames of
    a registered movie (HDF5 ``/data`` or TIFF pages)."""
    if path.suffix == ".h5":
        with h5py.File(path, "r") as f:
            data = f[next(iter(f.keys()))]
            n = data.shape[0] // n_channels
            for a in range(0, n, FRAMES_PER_READ):
                b = min(n, a + FRAMES_PER_READ)
                pages = data[a * n_channels : b * n_channels]
                pages = pages.reshape(b - a, n_channels, *pages.shape[1:])
                for i in range(b - a):
                    yield a + i, np.swapaxes(pages[i], -1, -2)
        return
    import tifffile

    with tifffile.TiffFile(path) as tf:
        pages = tf.pages
        n = len(pages) // n_channels
        for k in range(n):
            frame = [
                pages[k * n_channels + ch].asarray().T
                for ch in range(n_channels)
            ]
            yield k, np.stack(frame).astype(np.float32, copy=False)


def _var_factor_reader(path: Path):
    """Iterator over each frame's ``(rows, cols)`` ``varFacDS``."""
    with h5py.File(path, "r") as f:
        vf = f["slap2/varFacDS"]
        n = vf.shape[0]
        for a in range(0, n, FRAMES_PER_READ):
            block = vf[a : min(n, a + FRAMES_PER_READ)]
            for page in block:
                yield np.ascontiguousarray(page.T, dtype=np.float32)


def trial_stats(
    dat,
    first: float,
    last: float,
    movie: Union[str, Path],
    adata: Union[str, Path],
    out: Union[str, Path],
    device="cpu",
    table_settings: Optional[dict] = None,
) -> dict:
    """Compute and write one registered trial's statistics.

    Parameters
    ----------
    dat : str or Path
        The raw recording.
    first, last : float
        The trial's line range (trial table ``slap2_info``).
    movie, adata : str or Path
        Registered movie and alignment file.
    out : str or Path
        The statistics file to write.
    device : str or torch.device
        Where the statistics are formed.
    table_settings : dict, optional
        The trial table's ``align_params`` (MATLAB names).

    Returns
    -------
    dict
        The statistics (:func:`~giant_python.registration.stats.maps.
        finish_maps`).
    """
    import torch

    from ..registration.stats import (
        alignment_digest,
        new_accumulator,
        write_stats,
    )

    movie, adata, out = Path(movie), Path(adata), Path(out)
    align = read_alignment(adata)
    params = registration_params(align, table_settings)
    layout, source, ds = _source(dat, first, last, align, params, device)
    h, w = source.window.shape
    margin = (align["out_shape"][0] - h) // 2
    if align["out_shape"] != (h + 2 * margin, w + 2 * margin):
        raise ValueError(f"{adata}: meanIM does not fit the window")
    acc = new_accumulator(source, ds, margin, align["n_channels"])
    dev = source.former.device
    vfs = _var_factor_reader(adata)
    for k, frame in _movie_reader(movie, align["n_channels"]):
        vf = next(vfs)
        mr, mc = align["motion"][k]
        # as registration: a frame without data is missing, whatever motion
        # an alignment file (GIAnT-MATLAB's) gives it
        if not (
            np.isfinite(mr) and np.isfinite(mc) and np.isfinite(frame).any()
        ):
            acc.add_missing(k)
            continue
        x = torch.from_numpy(np.ascontiguousarray(frame)).to(dev)
        acc.add(k, list(x), torch.from_numpy(vf).to(dev), (mr, mc))
    result = acc.finish()
    attrs = {
        "source": "reg-stats",
        "movie": movie.name,
        "movie_size": int(movie.stat().st_size),
        "alignment": adata.name,
        "alignment_digest": alignment_digest(
            align["ds"], align["motion"][:, 0], align["motion"][:, 1]
        ),
        "align_hz": align["align_hz"],
        "device": str(dev),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".partial")
    write_stats(tmp, result, attrs)
    os.replace(tmp, out)
    return result


def _up_to_date(out: Path, adata: Path, movie: Path) -> bool:
    """Whether ``out`` holds this version's statistics of ``adata`` and
    ``movie`` (same frame times and motion, same movie size)."""
    from ..registration.stats import STATS_VERSION, alignment_digest

    if not (out.exists() and adata.exists() and movie.exists()):
        return False
    try:
        with h5py.File(out, "r") as f:
            version = int(f.attrs.get("stats_version", -1))
            digest = f.attrs.get("alignment_digest", "")
            size = int(f.attrs.get("movie_size", -1))
        align = read_alignment(adata)
    except (OSError, KeyError, ValueError):
        return False
    want = alignment_digest(
        align["ds"], align["motion"][:, 0], align["motion"][:, 1]
    )
    return (
        version == STATS_VERSION
        and digest == want
        and size == movie.stat().st_size
    )


def _stats_job(job: dict) -> dict:
    """One trial; failures are logged and reported, not raised."""
    import torch

    if job["threads"]:
        torch.set_num_threads(job["threads"])
    try:
        trial_stats(
            job["dat"],
            job["first"],
            job["last"],
            job["movie"],
            job["adata"],
            job["out"],
            job["device"],
            job["settings"],
        )
        return {"out": str(job["out"]), "error": None}
    except Exception as exc:  # reported per trial
        log.exception("statistics failed for %s", job["movie"])
        return {"out": str(job["out"]), "error": repr(exc)}


def _table_settings(path: Path) -> Optional[dict]:
    """The trial table's ``motion_correction/align_params`` (or None)."""
    from ..io.hdf5 import load_struct_h5

    try:
        mc = load_struct_h5(path).get("motion_correction", {})
    except (OSError, KeyError, ValueError):
        return None
    ap = mc.get("align_params")
    return ap if isinstance(ap, dict) else None


def _grid(value, shape) -> np.ndarray:
    """A trial-table field as an object grid of ``shape``."""
    return np.asarray(value, dtype=object).reshape(shape)


def _selected(shape, paths, trials) -> np.ndarray:
    """Cells of the requested one-based paths and trials (all if None)."""
    keep = np.ones(shape, dtype=bool)
    if paths is not None:
        keep &= np.isin(np.arange(shape[0]) + 1, list(paths))[:, None]
    if trials is not None:
        keep &= np.isin(np.arange(shape[1]) + 1, list(trials))[None, :]
    return keep


def _jobs(tt, files, opts: dict):
    """One job per registered trial-path that needs statistics, and the
    results of those whose movie or alignment file is missing (failed)."""
    from ..registration.stats import stats_path

    shape = files.shape
    mc = tt.motion_correction or {}
    fn_reg = _grid(mc.get("fn_reg_ds", np.full(shape, "")), shape)
    fn_ad = _grid(mc.get("fn_adata", np.full(shape, "")), shape)
    first = np.asarray(tt.slap2_info.first_line, float).reshape(shape)
    last = np.asarray(tt.slap2_info.last_line, float).reshape(shape)
    present = recorded_cells(files) & recorded_cells(fn_reg)
    present &= _selected(shape, opts["paths"], opts["trials"])
    moco = opts["moco"]
    jobs, missing = [], []
    for d, t in zip(*np.nonzero(present)):
        movie, adata = moco / str(fn_reg[d, t]), moco / str(fn_ad[d, t])
        out = stats_path(movie)
        if opts["out_dir"] is not None:
            out = Path(opts["out_dir"]) / out.name
        if not (movie.exists() and adata.exists()):
            log.warning("%s or its alignment file is missing", movie.name)
            error = f"{movie.name} or its alignment file is missing"
            missing.append({"out": out, "error": error})
            continue
        if not opts["overwrite"] and _up_to_date(out, adata, movie):
            log.info("%s is up to date; skipping", out.name)
            continue
        jobs.append(
            dict(
                dat=opts["datadr"] / str(files[d, t]),
                first=float(first[d, t]),
                last=float(last[d, t]),
                movie=movie,
                adata=adata,
                out=out,
                device=str(opts["device"]),
                settings=opts["settings"],
                threads=None,
            )
        )
    return jobs, missing


def reg_stats(
    trial_table: Union[str, Path],
    out_dir: Optional[Union[str, Path]] = None,
    *,
    datadr: Optional[Union[str, Path]] = None,
    moco: Optional[Union[str, Path]] = None,
    device: str = "auto",
    max_workers: int = 1,
    overwrite: bool = False,
    paths: Optional[Sequence[int]] = None,
    trials: Optional[Sequence[int]] = None,
) -> list:
    """Statistics of every registered trial of a trial table.

    Parameters
    ----------
    trial_table : str or Path
        A trial table with ``motion_correction`` (it is not changed).
    out_dir : str or Path, optional
        Folder for the statistics files (default: next to each movie).
    datadr, moco : str or Path, optional
        The raw-data folder and the motion-correction folder, if they moved
        since the table was written (default: ``datadr`` and
        ``<savedr>/motion_correction``).
    device : str
        ``'auto'``, ``'cpu'``, ``'cuda'`` or ``'cuda:N'``. On the CPU up to
        ``max_workers`` trials run in parallel; on CUDA one at a time.
    max_workers : int
        Parallel trials on the CPU.
    overwrite : bool
        Recompute statistics that are up to date (same version, same
        registration).
    paths, trials : sequence of int, optional
        One-based paths (DMDs) and trials to do (default: all).

    Returns
    -------
    list of dict
        Per trial: ``out`` and ``error`` (None if it succeeded; a requested
        trial whose movie or alignment file is missing has failed).
    """
    from ..registration.multiroi import resolve_device

    path = Path(trial_table)
    tt = read_trial_table(path)
    if tt.slap2_info is None:
        raise ValueError("the trial table has no slap2_info")
    files = np.atleast_2d(np.asarray(tt.filename, dtype=object))
    dev = resolve_device(device)
    opts = dict(
        moco=Path(moco) if moco else Path(tt.savedr) / "motion_correction",
        datadr=Path(datadr) if datadr else Path(tt.datadr),
        out_dir=out_dir,
        device=dev,
        settings=_table_settings(path),
        overwrite=overwrite,
        paths=paths,
        trials=trials,
    )
    jobs, missing = _jobs(tt, files, opts)
    workers = 1
    if dev.type == "cpu":
        workers = max(1, min(max_workers, len(jobs), os.cpu_count() or 1))
    if workers > 1:
        for job in jobs:
            job["threads"] = max(1, (os.cpu_count() or 1) // workers)
    return missing + map_trials(_stats_job, jobs, n_workers=workers)
