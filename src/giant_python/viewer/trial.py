"""One trial-path in the viewer's movie view.

The registered 80 Hz movie of the trial (the activity channel, in the
trial's own registered frame, not resampled), the runs' fitted activity
rendered into it (:mod:`~giant_python.viewer.render`), and the F0
network's baseline when a baseline folder is given. Arrays are held for
the valid pixels only, as the exporter defines them: missing in less than
``nanThresh`` of the frames with SILo's motion discards counted as
missing. The frames blanked afterwards because their kernel weight lies
mostly on discarded 200 Hz frames do not count (they are blank at every
pixel, and the exporter's training targets cover the same pixels). The
activity is not stored per frame: a frame or a trace is the footprints
(sparse) times the sources' 80 Hz traces.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Optional

import h5py
import numpy as np

from ..extraction.raster.trial import discard_frames
from ..pipeline.silo_session import _grid, _read
from .render import frame_weights, kernel_dt, place_footprints
from .run import SiloRun


class Session:
    """The trial table and folders the runs of one session share.

    Parameters
    ----------
    table : str or Path
        ``trial_table.h5`` (or its folder).
    moco_dir : str or Path, optional
        Folder of the registered movies and alignment files (default:
        ``<savedr>/motion_correction``, else next to the table).
    baseline_dir : str or Path, optional
        Folder of ``<stem>_BASELINE-80Hz.h5`` (``slap2-f0
        predict-registered``); optional.
    data_dir : str or Path, optional
        Folder of the raw files, to read the lines per cycle (default: the
        table's ``datadr``; without them the kernel reaches 3 dt).
    """

    def __init__(self, table, moco_dir=None, baseline_dir=None, data_dir=None):
        """Read the table's file names."""
        table = Path(table)
        if table.is_dir():
            table = table / "trial_table.h5"
        self.table = table
        with h5py.File(table, "r") as f:
            rm = "row_major" in f and bool(np.ravel(f["row_major"][()])[0])
            filename = np.atleast_2d(_read(f, "filename", rm))
            shape = filename.shape
            reg = _read(f, "motion_correction/fn_reg_ds", rm)
            adata = _read(f, "motion_correction/fn_adata", rm)
            savedr = _read(f, "savedr", rm)
            datadr = _read(f, "datadr", rm)
        if reg is None or adata is None:
            raise ValueError(f"{table} has no registration")
        self.filename = filename
        self.fn_reg_ds = _grid(reg, shape)
        self.fn_adata = _grid(adata, shape)
        savedr = Path(str(np.ravel(savedr)[0])) if savedr is not None else None
        if moco_dir is None:
            moco_dir = (savedr / "motion_correction") if savedr else None
            if moco_dir is None or not moco_dir.exists():
                moco_dir = table.parent / "motion_correction"
        self.moco_dir = Path(moco_dir)
        self.baseline_dir = Path(baseline_dir) if baseline_dir else None
        if data_dir is None and datadr is not None:
            data_dir = str(np.ravel(datadr)[0])
        self.data_dir = Path(data_dir) if data_dir else None
        self._ds: dict = {}
        self._lpc: dict = {}

    @staticmethod
    def _name(value) -> str:
        """A table file name without its folder."""
        return PurePath(str(value).replace("\\", "/")).name

    def movie_file(self, d: int, t: int) -> Path:
        """The registered movie of path ``d``, trial ``t`` (zero-based)."""
        return self.moco_dir / self._name(self.fn_reg_ds[d, t])

    def alignment_file(self, d: int, t: int) -> Path:
        """The alignment file of path ``d``, trial ``t`` (zero-based)."""
        return self.moco_dir / self._name(self.fn_adata[d, t])

    def has_movie(self, d: int, t: int) -> bool:
        """Whether the trial's movie and alignment file are here."""
        if not str(self.fn_reg_ds[d, t]).strip():
            return False
        return (
            self.movie_file(d, t).exists()
            and self.alignment_file(d, t).exists()
        )

    def baseline_file(self, d: int, t: int) -> Optional[Path]:
        """The trial's network baseline, if there is one."""
        if self.baseline_dir is None:
            return None
        name = self._name(self.fn_reg_ds[d, t])
        if "_REGISTERED" not in name:
            return None
        stem = name[: name.index("_REGISTERED")]
        p = self.baseline_dir / f"{stem}_BASELINE-80Hz.h5"
        return p if p.exists() else None

    def ds_lines(self, d: int, t: int) -> np.ndarray:
        """The registered frames' lines (``DSframes``) of a trial."""
        key = (d, t)
        if key not in self._ds:
            with h5py.File(self.alignment_file(d, t), "r") as f:
                self._ds[key] = np.asarray(f["DSframes"][()], float).ravel()
        return self._ds[key]

    def lines_per_cycle(self, d: int, t: int) -> int:
        """Lines per cycle from the raw file's header (0 if unreadable)."""
        key = (d, t)
        if key not in self._lpc:
            self._lpc[key] = 0
            if self.data_dir is not None:
                try:
                    from ..io.slap2_raster import raster_layout

                    fn = self.data_dir / self._name(self.filename[d, t])
                    self._lpc[key] = int(raster_layout(fn).lines_per_cycle)
                except Exception:  # noqa: BLE001  (no raw data: 3 dt)
                    self._lpc[key] = 0
        return self._lpc[key]


def read_channel(path: Path, num_channels: int, channel: int) -> np.ndarray:
    """One channel of a registered movie as ``(frames, rows, cols)``
    float32 (zero-based channel; channels interleaved by frame)."""
    if path.suffix == ".h5":
        with h5py.File(path, "r") as f:
            ds = f[next(iter(f.keys()))]  # (pages, MATLAB cols, rows)
            n = ds.shape[0] // num_channels
            # whole frames at a time: twice as fast as a strided selection
            # of the (one page per chunk, compressed) dataset
            step = 128 * num_channels
            raw = np.concatenate(
                [
                    ds[a : min(a + step, n * num_channels)][
                        channel::num_channels
                    ]
                    for a in range(0, n * num_channels, step)
                ]
            )
    else:
        import tifffile

        with tifffile.TiffFile(path) as tf:
            pages = tf.pages
            n = len(pages) // num_channels
            first = pages[channel].asarray()
            raw = np.empty((n,) + first.shape, first.dtype)
            raw[0] = first
            for k in range(1, n):
                raw[k] = pages[channel + k * num_channels].asarray()
    return np.ascontiguousarray(np.asarray(raw, np.float32).transpose(0, 2, 1))


@dataclass
class RunLayer:
    """One run's activity in a trial.

    Attributes
    ----------
    name : str
    footprints : scipy.sparse.csr_matrix (valid pixels, sources)
    x80 : dict
        Time kernel (:data:`KERNELS`) to the sources' denoised traces at
        the movie's frames, ``(sources, frames)``.
    kept_rc, rejected_rc, seeds_rc : ndarray (n, 2)
        Marker positions in the trial's registered frame.
    rendered : bool
        False when the run has no footprints or no map for the trial.
    """

    name: str
    footprints: object
    x80: dict
    kept_rc: np.ndarray
    rejected_rc: np.ndarray
    seeds_rc: np.ndarray
    rendered: bool = True


@dataclass
class TrialMovie:
    """A loaded trial-path: movie, baseline and each run's activity.

    ``kernel`` is the time kernel (:data:`KERNELS`) that :meth:`frame` and
    :meth:`trace` use unless they are given another.
    """

    path: int
    trial: int
    shape: tuple
    rows: np.ndarray
    cols: np.ndarray
    mean: np.ndarray
    discard: np.ndarray
    movie: np.ndarray
    baseline: Optional[np.ndarray]
    runs: list
    info: dict = field(default_factory=dict)
    kernel: str = "matched"
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def T(self) -> int:
        """Frames."""
        return int(self.movie.shape[0])

    @property
    def N(self) -> int:
        """Valid pixels."""
        return int(self.movie.shape[1])

    @property
    def layers(self) -> list:
        """Layer names in the order of :meth:`frame` and :meth:`trace`."""
        out = ["movie"]
        for r in self.runs:
            out += [f"activity:{r.name}", f"residual:{r.name}"]
        if self.baseline is not None:
            out.append("baseline")
            out += [f"residual-baseline:{r.name}" for r in self.runs]
        return out

    def span(self, t: int, avg: int) -> tuple:
        """The frames ``[lo, hi)`` a mean of ``avg`` frames at ``t`` uses."""
        lo = min(max(t - (avg - 1) // 2, 0), self.T - 1)
        return lo, min(max(lo + avg, lo + 1), self.T)

    def _stack(self, frames, pixels, kernel=None) -> list:
        """Every layer's values ``(frames, pixels)`` at the given frames
        (a slice) and pixels (an index array or a slice)."""
        kernel = kernel or self.kernel
        if kernel not in KERNELS:
            raise KeyError(f"kernel {kernel}")
        m = self.movie[frames][:, pixels]
        miss = np.isnan(m)
        out = [m]
        acts = []
        for r in self.runs:
            h = r.footprints[pixels] if r.rendered else None
            a = (
                np.asarray((h @ r.x80[kernel][:, frames]).T, np.float32)
                if h is not None
                else np.zeros_like(m)
            )
            a[miss] = np.nan
            acts.append(a)
            out += [a, m - a]
        if self.baseline is not None:
            b = self.baseline[frames][:, pixels].astype(np.float32)
            out.append(b)
            out += [m - a - b for a in acts]
        return out

    def frame(self, t: int, avg: int, kernel=None) -> np.ndarray:
        """Each layer's mean over ``avg`` frames at ``t``: ``(layers,
        valid pixels)`` float32, NaN-aware."""
        lo, hi = self.span(t, avg)
        blocks = self._stack(slice(lo, hi), slice(None), kernel)
        out = np.empty((len(blocks), self.N), np.float32)
        with np.errstate(invalid="ignore"), _quiet():
            for i, b in enumerate(blocks):
                out[i] = b[0] if hi - lo == 1 else np.nanmean(b, 0)
        return out

    def trace(self, r: int, c: int, rad: int, kernel=None) -> tuple:
        """Each layer's mean over the valid pixels within ``rad`` rows and
        columns of ``(r, c)``: ``(layers, frames)`` float32, and the
        number of pixels."""
        near = (np.abs(self.rows - r) <= rad) & (np.abs(self.cols - c) <= rad)
        k = np.flatnonzero(near)
        out = np.full((len(self.layers), self.T), np.nan, np.float32)
        if k.size:
            with _quiet():
                for i, b in enumerate(self._stack(slice(None), k, kernel)):
                    out[i] = np.nanmean(b, 1)
        return out, int(k.size)

    def busiest_frame(self, run: int = 0) -> int:
        """The frame with the most rendered activity of a run."""
        r = self.runs[run]
        x = r.x80[self.kernel]
        if not r.rendered or x.size == 0:
            return 0
        total = np.asarray(r.footprints.sum(0)).ravel() @ x
        total[self.discard] = -np.inf
        return int(np.argmax(total))


class _quiet:
    """Silence NumPy's all-NaN warnings inside a block."""

    def __enter__(self):
        """Start ignoring RuntimeWarnings."""
        import warnings

        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        """Restore the warning filters."""
        self._w.__exit__(*exc)


def _marker_rc(tmap, rc: np.ndarray) -> np.ndarray:
    """Canvas positions moved into the trial frame (NaN rows stay NaN)."""
    rc = np.asarray(rc, np.float64).reshape(-1, 2)
    if tmap is None or rc.size == 0:
        return rc.copy()
    r, c = tmap.forward(np.nan_to_num(rc[:, 0]), np.nan_to_num(rc[:, 1]))
    out = np.column_stack([r, c])
    out[np.isnan(rc).any(1)] = np.nan
    return out


def run_layer(run: SiloRun, session: Session, d: int, t: int, ctx: dict):
    """One run's activity in trial ``t`` of path ``d`` (one-based path,
    zero-based trial). ``ctx`` holds the trial's ``shape``, valid flat
    pixel indices ``flat``, ``ds_lines``, ``kernels`` (name to ``(dt,
    reach)``) and ``blur``; the run's 80 Hz discard share under the
    exporter's kernel is added to it."""
    p = run.path(d)
    n = p.sources.n
    tmap = p.maps[t] if t < len(p.maps) else None
    fps = run.footprints(d)
    fp = fps[t] if t < len(fps) else None
    span = p.trial_span(t) if t < p.n_frames.size else slice(0, 0)
    lines200 = p.lines[span]
    rendered = bool(tmap is not None and fp is not None and lines200.size)
    T = ctx["ds_lines"].size
    x80 = {k: np.zeros((n, T), np.float32) for k in ctx["kernels"]}
    h = None
    if rendered:
        x200 = np.nan_to_num(run.traces(d)["dF_denoised"][span])  # (n200, n)
        for k, (dtr, reach) in ctx["kernels"].items():
            w, share = frame_weights(
                lines200, p.discard[span], ctx["ds_lines"], dtr, reach
            )
            if k == "exporter":
                ctx.setdefault("shares", []).append(share)
            x80[k] = np.asarray((w.T @ x200).T, np.float32)
        h = place_footprints(fp, n, tmap, ctx["shape"], ctx["blur"])
        h = h[ctx["flat"]]
    trc = run.trial_centroids(d)
    rc = trc[t] if t < trc.shape[0] else np.full((n, 2), np.nan)
    rc = np.where(np.isnan(rc), p.sources.rc, rc)
    seeds = p.sources.unused_seeds
    return RunLayer(
        name=run.name,
        footprints=h,
        x80=x80,
        kept_rc=_marker_rc(tmap, rc),
        rejected_rc=_marker_rc(tmap, p.sources.rejected_rc),
        seeds_rc=_marker_rc(tmap, seeds if seeds is not None else []),
        rendered=rendered,
    )


def _settings(run: SiloRun) -> dict:
    """SILo's frame settings of a run (defaults: SILo's for SLAP2)."""
    return {
        "num_channels": int(run.param("numChannels", 2)),
        "channel": run.activity_channel,
        "motion_thresh": float(run.param("motionThresh", 2.0)),
        "discard_initial_s": float(run.param("discardInitial_s", 0.0)),
        "nan_thresh": float(run.param("nanThresh", 0.33)),
        "ratio": float(run.param("analyzeHz", 200.0))
        / float(run.param("alignHz", 80.0)),
    }


def _read_alignment(path: Path, n_frames: int):
    """``alignHz`` and ``recNegErr`` of a trial's alignment file."""
    with h5py.File(path, "r") as f:
        hz = float(np.ravel(f["alignHz"][()])[0])
        rne = np.asarray(f["recNegErr"][()], np.float64).ravel()[:n_frames]
    return hz, rne


KERNELS = ("exporter", "matched")


def render_dt(kernel: str, dt80: int, lines200: np.ndarray) -> float:
    """The time constant (lines) the 200 Hz traces are averaged with.

    ``exporter``: the movie's own ``dt``, as the exporter does. The 200 Hz
    frames are already averaged with their own kernel (``dt200 =
    ceil(mean(diff(lines)))``), so this smooths fast events twice;
    ``matched`` uses ``sqrt(dt80**2 - dt200**2)``, the kernel whose
    variance added to the 200 Hz frames' gives the movie's.
    """
    if kernel == "exporter":
        return float(dt80)
    if kernel != "matched":
        raise ValueError(f"kernel must be one of {KERNELS}, not {kernel!r}")
    lines200 = np.asarray(lines200, dtype=np.float64)
    if lines200.size < 2:
        return float(dt80)
    dt200 = math.ceil(float(np.mean(np.diff(lines200))))
    return math.sqrt(max(dt80**2 - dt200**2, 1.0))


def load_trial(
    runs: list,
    session: Session,
    d: int,
    t: int,
    blur: float = 0.0,
    kernel: str = "matched",
) -> TrialMovie:
    """Load trial ``t`` (zero-based) of path ``d`` (one-based) with every
    run's activity.

    The movie is NaN at SILo's motion discards (``recNegErr``, as
    ``loadAndProcessTrialAsync``) and at the frames whose kernel weight
    falls mostly on SILo's discarded 200 Hz frames (the exporter's rule).
    Valid pixels are decided on the motion discards alone, as the
    exporter decides them (see the module docstring).
    Both time kernels (:func:`render_dt`) are rendered; ``kernel`` is the
    one :meth:`TrialMovie.frame` uses by default.
    """
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}, not {kernel!r}")
    st = _settings(runs[0])
    movie = read_channel(
        session.movie_file(d - 1, t), st["num_channels"], st["channel"]
    )
    T, R, C = movie.shape
    hz, rne = _read_alignment(session.alignment_file(d - 1, t), T)
    disc = discard_frames(
        rne, hz, st["motion_thresh"], st["discard_initial_s"]
    )
    movie[disc] = np.nan
    with np.errstate(invalid="ignore"):
        valid = np.isnan(movie).mean(0) < st["nan_thresh"]
    rows, cols = np.nonzero(valid)
    flat = rows * C + cols
    ds = session.ds_lines(d - 1, t)[:T]
    lines0 = runs[0].path(d).lines
    dt = kernel_dt(ds, lines0, st["ratio"])
    seg = lines0[runs[0].path(d).trial_span(t)]
    lpc = session.lines_per_cycle(d - 1, t)
    kernels = {}
    for k in KERNELS:
        dtr = render_dt(k, dt, seg)
        kernels[k] = (dtr, max(3 * dtr, lpc))
    ctx = {
        "shape": (R, C),
        "flat": flat,
        "ds_lines": ds,
        "kernels": kernels,
        "blur": float(blur),
    }
    layers = [run_layer(r, session, d, t, ctx) for r in runs]
    for share in ctx.get("shares", []):
        disc = disc | (share > 0.5)
    mov = movie.reshape(T, R * C)[:, flat]
    del movie
    mov[disc] = np.nan
    base = _read_baseline(session.baseline_file(d - 1, t), (T, R, C), flat)
    if base is not None:
        base[disc] = np.nan
    with np.errstate(invalid="ignore"), _quiet():
        mean = np.nanmean(mov, 0)
    info = {
        "dt_lines": dt,
        "render_dt_lines": {k: round(v[0], 2) for k, v in kernels.items()},
        "reach_lines": {k: round(v[1], 1) for k, v in kernels.items()},
        "lines_per_cycle": lpc,
        "blur_px": float(blur),
        "align_hz": hz,
        "movie": session.movie_file(d - 1, t).name,
        "baseline": (session.baseline_file(d - 1, t) or Path("")).name,
        "discarded": int(disc.sum()),
    }
    return TrialMovie(
        path=d,
        trial=t,
        shape=(R, C),
        rows=rows,
        cols=cols,
        mean=mean,
        discard=disc,
        movie=mov,
        baseline=base,
        runs=layers,
        info=info,
        kernel=kernel,
    )


def _read_baseline(path: Optional[Path], shape: tuple, flat: np.ndarray):
    """A network baseline at the valid pixels, float16 ``(frames, N)``."""
    if path is None:
        return None
    T, R, C = shape
    with h5py.File(path, "r") as f:
        b = np.asarray(f["baseline"][()])  # (frames, MATLAB cols, rows)
    if b.shape != (T, C, R):
        raise ValueError(f"{path.name}: baseline {b.shape}, movie {shape}")
    b = b.transpose(0, 2, 1).reshape(T, R * C)[:, flat]
    return np.ascontiguousarray(b.astype(np.float16))


def largest_events(x: np.ndarray, n: int, gap: int) -> np.ndarray:
    """Indices of the ``n`` largest local maxima of ``x`` at least ``gap``
    samples apart (largest first)."""
    x = np.nan_to_num(np.asarray(x, np.float64), nan=-np.inf)
    if x.size < 3:
        return np.zeros(0, int)
    peak = np.flatnonzero((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:])) + 1
    peak = peak[x[peak] > 0]
    order = peak[np.argsort(-x[peak], kind="stable")]
    taken: list = []
    for i in order:
        if all(abs(i - j) >= gap for j in taken):
            taken.append(int(i))
            if len(taken) >= n:
                break
    return np.asarray(taken, int)


def source_events(run: SiloRun, session: Session, d: int, s: int, n=30):
    """A source's largest denoised events over all trials, as movie
    frames: list of ``{trial, frame, line, value}`` (largest first)."""
    p = run.path(d)
    x = run.traces(d)["dF_denoised"][:, s]
    hz = float(run.param("analyzeHz", 200.0))
    gap = max(1, int(math.ceil(0.25 * hz)))
    out = []
    for t in range(p.n_frames.size):
        span = p.trial_span(t)
        if span.stop <= span.start or not session.has_movie(d - 1, t):
            continue
        xs = x[span]
        ds = session.ds_lines(d - 1, t)
        for i in largest_events(xs, n, gap):
            line = float(p.lines[span][i])
            frame = int(np.argmin(np.abs(ds - line)))
            out.append(
                {
                    "trial": t,
                    "frame": frame,
                    "value": float(xs[i]),
                    "line": line,
                }
            )
    out.sort(key=lambda e: -e["value"])
    return out[:n]
