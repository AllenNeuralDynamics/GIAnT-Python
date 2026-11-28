"""A SILo run's outputs as the viewer reads them.

A run is a folder with ``experiment_summary.h5`` and ``per_trial_summary.h5``
(directly or in ``source_extraction/``), as ``giant silo`` writes them
(GIAnT-MATLAB's schema). The viewer reads only these two files, so runs
whose work folder is gone (Code Ocean results) work too; the plan files
(``work/plan/path<d>.h5``, or ``source_extraction/silo/plan``), when
present, add the seeds that did not become sources.

The summaries are written in MATLAB's orientation (h5py shows the
dimensions reversed); everything here is returned in MATLAB's orientation
with zero-based pixel coordinates. Arrays of one path are read when the
path is first used and kept.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from .render import TrialMap


def _mat(f, name: str):
    """A summary dataset in MATLAB orientation, or None."""
    if name not in f:
        return None
    a = f[name][()]
    return a.T if getattr(a, "ndim", 0) > 1 else a


def _scalar(v):
    """An HDF5 value as a Python scalar or string."""
    a = np.asarray(v)
    if a.dtype.kind in "SO":
        x = a.ravel()[0] if a.size else ""
        return x.decode() if isinstance(x, bytes) else str(x)
    if a.dtype.kind in "U":
        return str(a.ravel()[0]) if a.size else ""
    return a.ravel()[0].item() if a.size == 1 else a.tolist()


def find_summaries(run_dir) -> Path:
    """The folder holding a run's ``experiment_summary.h5``."""
    run_dir = Path(run_dir)
    for d in (run_dir, run_dir / "source_extraction"):
        if (d / "experiment_summary.h5").exists():
            return d
    found = sorted(run_dir.glob("**/experiment_summary.h5"))
    if len(found) == 1:
        return found[0].parent
    raise FileNotFoundError(
        f"{run_dir}: no experiment_summary.h5"
        + (" (several below it; give one)" if found else "")
    )


def _plan_file(run_dir: Path, summaries: Path, d: int) -> Optional[Path]:
    """The plan file of path ``d`` (one-based), if the run kept it."""
    for p in (
        run_dir / "work" / "plan" / f"path{d}.h5",
        summaries / "silo" / "plan" / f"path{d}.h5",
    ):
        if p.exists():
            return p
    return None


def footprint_width(profile: np.ndarray, r0: float, c0: float, radius=6):
    """RMS width per axis (px) of a footprint image over the disk of
    ``radius`` px around its source pixel ``(r0, c0)``, values clipped at
    0 (the measure of the footprint-support comparisons)."""
    r0, c0 = int(round(r0)), int(round(c0))
    rr, cc = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    m = rr**2 + cc**2 <= radius**2
    rr, cc = rr[m] + r0, cc[m] + c0
    ok = (
        (rr >= 0)
        & (rr < profile.shape[0])
        & (cc >= 0)
        & (cc < profile.shape[1])
    )
    rr, cc = rr[ok], cc[ok]
    x = np.clip(np.nan_to_num(profile[rr, cc]), 0, None)
    if x.sum() <= 0:
        return float("nan")
    mr, mc = (x * rr).sum() / x.sum(), (x * cc).sum() / x.sum()
    var = (x * ((rr - mr) ** 2 + (cc - mc) ** 2)).sum() / x.sum() / 2
    return float(math.sqrt(var))


@dataclass
class Sources:
    """One path's sources in a run (consensus canvas, zero-based).

    Attributes
    ----------
    rc : ndarray (n, 2)
        Kept sources' footprint centroids (``sources/spatial/coords``).
    pixel : ndarray (n, 2)
        Their source pixels (``visualizations/act_im_peaks``).
    snr, kinetic_score, kinetic_n_events, width : ndarray (n,)
    rejected_rc : ndarray (m, 2)
        Sources the kinetics test rejected (their pixels).
    rejected_score, rejected_n_events : ndarray (m,)
    unused_seeds : ndarray (k, 2) or None
        Seeds without a source (from the plan file; None without it).
    """

    rc: np.ndarray
    pixel: np.ndarray
    snr: np.ndarray
    kinetic_score: np.ndarray
    kinetic_n_events: np.ndarray
    width: np.ndarray
    rejected_rc: np.ndarray
    rejected_score: np.ndarray
    rejected_n_events: np.ndarray
    unused_seeds: Optional[np.ndarray] = None

    @property
    def n(self) -> int:
        """Number of kept sources."""
        return int(self.rc.shape[0])


def _vec(a, n: int, fill=np.nan) -> np.ndarray:
    """A summary vector as float64 of length ``n``."""
    if a is None:
        return np.full(n, fill)
    return np.asarray(a, dtype=np.float64).ravel()[:n]


def _rc(a) -> np.ndarray:
    """``[0, row, col]`` rows as ``(n, 2)`` (empty when missing)."""
    if a is None:
        return np.zeros((0, 2))
    return np.asarray(a, dtype=np.float64).reshape(-1, 3)[:, 1:3]


@dataclass
class PathData:
    """One path of a run: images, sources, traces, footprints, maps.

    Attributes
    ----------
    path : int
        One-based imaging path.
    shape : tuple of int
        ``(rows, cols)`` of the consensus canvas.
    act_im, mean_im : ndarray (rows, cols)
        Combined activity image; mean image of the activity channel.
    trial_act, trial_mean : ndarray (trials, rows, cols) or None
    n_frames : ndarray of int (trials,)
        SILo's 200 Hz frames per trial (0: not analysed).
    lines, discard : ndarray (total,)
        SILo's frame lines and discarded frames, trials concatenated.
    maps : list of (TrialMap or None)
        Each trial's map from the canvas to its registered frame.
    sources : Sources
    """

    path: int
    shape: tuple
    act_im: np.ndarray
    mean_im: np.ndarray
    trial_act: Optional[np.ndarray]
    trial_mean: Optional[np.ndarray]
    n_frames: np.ndarray
    lines: np.ndarray
    discard: np.ndarray
    maps: list
    sources: Sources
    alignment: str = "rigid"
    _traces: dict = field(default_factory=dict, repr=False)
    _footprints: Optional[list] = field(default=None, repr=False)
    _trial_rc: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def offsets(self) -> np.ndarray:
        """First 200 Hz frame of each trial in the concatenated traces."""
        return np.concatenate([[0], np.cumsum(self.n_frames)]).astype(int)

    def trial_span(self, t: int) -> slice:
        """The concatenated frames of trial ``t`` (zero-based)."""
        o = self.offsets
        return slice(int(o[t]), int(o[t + 1]))


TRACES = ("dF_denoised", "dF_ls", "events", "F0")


class SiloRun:
    """A SILo run's summaries, read path by path on demand.

    Parameters
    ----------
    run_dir : str or Path
        The run folder (or its ``source_extraction`` folder).
    name : str, optional
        Label shown on the page (default: the folder's name).
    """

    def __init__(self, run_dir, name: Optional[str] = None):
        """Open the run and read its parameters."""
        self.run_dir = Path(run_dir)
        self.summaries = find_summaries(self.run_dir)
        self.experiment = self.summaries / "experiment_summary.h5"
        self.per_trial = self.summaries / "per_trial_summary.h5"
        self.name = name or self.run_dir.name
        with h5py.File(self.experiment, "r") as f:
            # scalars and short vectors; validKernel is the kernel itself
            # (a width x width matrix), kept whatever its size
            self.params = {
                k: _scalar(f["params"][k][()])
                for k in f.get("params", {})
                if f["params"][k].size <= 64 or k == "validKernel"
            }
            self.attrs = {k: _scalar(v) for k, v in f.attrs.items()}
            self.paths = sorted(
                int(k[4:])
                for k in f
                if k.startswith("Path") and k[4:].isdigit()
            )
        self._paths: dict = {}
        self._lock = threading.Lock()

    def param(self, name: str, default=None):
        """A value of the run's ``params`` (MATLAB names)."""
        v = self.params.get(name, default)
        return default if v is None or v == "" else v

    @property
    def activity_channel(self) -> int:
        """Zero-based file channel of the activity signal."""
        return int(self.param("activityChannel", 1)) - 1

    def describe(self) -> dict:
        """The run's settings that the page shows."""
        keys = (
            "baselineSource",
            "baselineMode",
            "localizationBaseline",
            "interTrialAlignment",
            "validKernel",
            "peakth",
            "footprintVariance",
            "seedFit",
            "seedMergePx",
            "minKineticScore",
            "photonScale",
            "alignHz",
            "analyzeHz",
        )
        out = {k: self.params[k] for k in keys if k in self.params}
        out.setdefault("baselineSource", "median")
        vk = out.get("validKernel")
        if isinstance(vk, list):  # the kernel itself: report its width
            out["validKernel"] = int(round(math.sqrt(len(np.ravel(vk)))))
        return out

    def path(self, d: int) -> PathData:
        """Path ``d`` (one-based), read on first use."""
        with self._lock:
            if d not in self._paths:
                self._paths[d] = self._read_path(d)
            return self._paths[d]

    def _read_path(self, d: int) -> PathData:
        """Read one path's images, frames, maps and sources."""
        g = f"Path{d}"
        ch = self.activity_channel
        with h5py.File(self.experiment, "r") as f:
            act = _mat(f, f"{g}/visualizations/act_im")[0]
            mean = _mat(f, f"{g}/visualizations/mean_im")
            mean = mean[min(ch, mean.shape[0] - 1), 0]
            fi = f"{g}/frame_info"
            n_frames = np.asarray(_mat(f, f"{fi}/trial_num_frames")).ravel()
            lines = _vec(_mat(f, f"{fi}/frame_line_idxs"), int(n_frames.sum()))
            disc = _vec(_mat(f, f"{fi}/discard_frames"), lines.size, 0)
            maps, mode = _read_maps(f, fi, n_frames.size)
            sources = _read_sources(f, g, act.shape)
        trial_act = trial_mean = None
        if self.per_trial.exists():
            with h5py.File(self.per_trial, "r") as f:
                ta = _mat(f, f"{g}/visualizations/per_trial_act_im")
                tm = _mat(f, f"{g}/visualizations/per_trial_mean_im")
            trial_act = None if ta is None else ta[:, 0].astype(np.float32)
            if tm is not None:
                trial_mean = tm[:, min(ch, tm.shape[1] - 1), 0]
                trial_mean = trial_mean.astype(np.float32)
        plan = _plan_file(self.run_dir, self.summaries, d)
        if plan is not None:
            sources.unused_seeds = _unused_seeds(plan)
        return PathData(
            path=d,
            shape=tuple(act.shape),
            act_im=act.astype(np.float32),
            mean_im=mean.astype(np.float32),
            trial_act=trial_act,
            trial_mean=trial_mean,
            n_frames=n_frames.astype(int),
            lines=lines,
            discard=disc.astype(bool),
            maps=maps,
            sources=sources,
            alignment=mode,
        )

    def traces(self, d: int) -> dict:
        """The activity channel's traces of path ``d``: name to array
        ``(total frames, kept sources)`` (dF_denoised, dF_ls, events, F0)."""
        p = self.path(d)
        with self._lock:
            if not p._traces:
                p._traces = self._read_traces(d)
            return p._traces

    def _read_traces(self, d: int) -> dict:
        """Read the four trace datasets of one path (activity channel)."""
        out = {}
        ch = self.activity_channel
        with h5py.File(self.experiment, "r") as f:
            for name in TRACES:
                key = f"Path{d}/sources/temporal/{name}"
                if key in f:  # h5py: (frames, channels, sources)
                    ds = f[key]
                    c = min(ch, ds.shape[1] - 1)
                    out[name] = np.asarray(ds[:, c, :], np.float32)
        return out

    def profile_patch(self, d: int, s: int, half: int = 7):
        """Source ``s``'s mean footprint (``sources/spatial/profiles``)
        around its pixel: ``(patch, row0, col0)``."""
        p = self.path(d)
        r0, c0 = (int(v) for v in p.sources.pixel[s])
        ra = slice(max(0, r0 - half), min(p.shape[0], r0 + half + 1))
        ca = slice(max(0, c0 - half), min(p.shape[1], c0 + half + 1))
        with h5py.File(self.experiment, "r") as f:
            prof = f[f"Path{d}/sources/spatial/profiles"]  # (c, r, 1, n)
            patch = np.asarray(prof[ca, ra, 0, s], np.float32).T
        return patch, ra.start, ca.start

    def footprints(self, d: int) -> list:
        """Each trial's footprints of path ``d`` on the canvas, as
        ``(rows, cols, sources, values)`` (None for a trial without)."""
        p = self.path(d)
        with self._lock:
            if p._footprints is None:
                p._footprints, p._trial_rc = self._read_footprints(d, p)
            return p._footprints

    def trial_centroids(self, d: int) -> np.ndarray:
        """Per-trial footprint centroids ``(trials, sources, 2)`` (NaN
        where a trial has none)."""
        self.footprints(d)
        return self.path(d)._trial_rc

    def _read_footprints(self, d: int, p: PathData):
        """Read ``per_trial_profiles`` in blocks of columns."""
        n_t, n_s = p.n_frames.size, p.sources.n
        parts: list = [[] for _ in range(n_t)]
        rc = np.full((n_t, n_s, 2), np.nan)
        if not self.per_trial.exists() or n_s == 0:
            return [None] * n_t, rc
        g = f"Path{d}/sources"
        with h5py.File(self.per_trial, "r") as f:
            if f"{g}/spatial/per_trial_profiles" not in f:
                return [None] * n_t, rc
            ds = f[f"{g}/spatial/per_trial_profiles"]  # (cols, rows, 1, s, t)
            step = max(1, int(2e8 // max(1, ds.shape[1] * n_s * n_t * 4)))
            for c0 in range(0, ds.shape[0], step):
                blk = np.nan_to_num(ds[c0 : c0 + step, :, 0])
                c, r, s, t = np.nonzero(blk)
                v = blk[c, r, s, t]
                for k in np.unique(t):
                    m = t == k
                    parts[k].append((r[m], c[m] + c0, s[m], v[m]))
            coords = _mat(f, f"{g}/spatial/per_trial_coords")
        if coords is not None:  # MATLAB (trials, sources, 3)
            rc[:] = np.asarray(coords, np.float64)[:n_t, :n_s, 1:3]
        out = []
        for k in range(n_t):
            if not parts[k]:
                out.append(None)
                continue
            out.append(tuple(np.concatenate(x) for x in zip(*parts[k])))
        return out, rc


def _read_maps(f, fi: str, n_trials: int):
    """Each trial's ``TrialMap`` and the alignment mode."""
    affine = _mat(f, f"{fi}/trial_alignment_affine")
    disp = _mat(f, f"{fi}/trial_alignment_displacement")
    if affine is None:  # only offline shifts: unknown
        return [None] * n_trials, "unknown"
    affine = np.asarray(affine, np.float64).reshape(-1, 2, 3)
    maps = []
    for t in range(n_trials):
        a = affine[t] if t < affine.shape[0] else np.full((2, 3), np.nan)
        fld = None
        if disp is not None and t < disp.shape[0]:
            fld = np.asarray(disp[t], np.float64)  # (2, rows, cols)
        m = TrialMap(a, fld)
        maps.append(m if m.valid else None)
    if disp is not None:
        return maps, "patchwise"
    eye = np.isfinite(affine[:, 0, 0])
    rigid = np.allclose(affine[eye][:, :, :2], np.eye(2)) if eye.any() else 1
    return maps, "rigid" if rigid else "affine"


def _read_sources(f, g: str, shape) -> Sources:
    """A path's kept and rejected sources from the experiment summary."""
    rc = _rc(_mat(f, f"{g}/sources/spatial/coords"))
    n = rc.shape[0]
    pixel = _rc(_mat(f, f"{g}/visualizations/act_im_peaks"))
    if pixel.shape[0] != n:
        pixel = np.round(rc)
    temporal = f"{g}/sources/temporal"
    width = np.full(n, np.nan)
    if n and f"{g}/sources/spatial/profiles" in f:
        prof = f[f"{g}/sources/spatial/profiles"]  # (cols, rows, 1, n)
        for s in range(n):
            r0, c0 = pixel[s]
            ra = slice(max(0, int(r0) - 7), min(shape[0], int(r0) + 8))
            ca = slice(max(0, int(c0) - 7), min(shape[1], int(c0) + 8))
            img = prof[ca, ra, 0, s].T
            width[s] = footprint_width(img, r0 - ra.start, c0 - ca.start)
    rej = f"{g}/sources/rejected"
    rej_rc = _rc(_mat(f, f"{rej}/seed_coords"))
    m = rej_rc.shape[0]
    return Sources(
        rc=rc,
        pixel=pixel,
        snr=_vec(_mat(f, f"{temporal}/SNR"), n),
        kinetic_score=_vec(_mat(f, f"{temporal}/kinetic_score"), n),
        kinetic_n_events=_vec(_mat(f, f"{temporal}/kinetic_n_events"), n),
        width=width,
        rejected_rc=rej_rc,
        rejected_score=_vec(_mat(f, f"{rej}/kinetic_score"), m),
        rejected_n_events=_vec(_mat(f, f"{rej}/kinetic_n_events"), m),
    )


def _unused_seeds(plan_path: Path) -> np.ndarray:
    """Seeds of a plan whose rounded pixel has no source."""
    with h5py.File(plan_path, "r") as f:
        g = f["sources"] if "sources" in f else f
        seeds = np.asarray(g["seeds"][()], np.float64).reshape(-1, 4)
        rows = np.asarray(g["rows"][()]).ravel()
        cols = np.asarray(g["cols"][()]).ravel()
    have = set(zip(rows.tolist(), cols.tolist()))
    pix = np.floor(seeds[:, 1:3] + 0.5).astype(int)
    keep = [(int(r), int(c)) not in have for r, c in pix]
    return seeds[np.asarray(keep, bool)][:, 1:3]
