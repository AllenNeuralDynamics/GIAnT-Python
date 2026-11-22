"""Register one SLAP2 raster trial (``alignAsync`` of MultiRoiRegistration).

The trial's frames are registered sequentially, as in GIAnT-MATLAB: each DS
frame is matched to the mean of the initial and running templates, starting
from the previous frame's rounded motion with a search radius of
``max(|init|, clip_shift)``; its registered image is then added to the
running template. With ``compensate_online``, frames are formed with the
online motion correction undone line by line (see :mod:`.compensation`).

Frames without valid samples (e.g. before a late recording start) are
missing frames: they get NaN motion and an all-NaN movie frame and are not
matched, so they change neither the template nor the next frame's start,
nor ``registrationFailed``. MATLAB gives them an arbitrary finite motion
(the edge of the search range); their registered images are NaN there too.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from ..io.slap2_raster import (
    RasterLayout,
    raster_layout,
    read_line_offsets,
    read_raster_samples,
)
from ..models.params import MultiRoiRegistrationParams
from .compensation import compensation_record
from .frames import FrameFormer
from .growth import FrameCache, Grower, GrowthQC, seeds_from_consensus
from .interpolation import interp_frame
from .matching import RegistrationError, match_maps
from .outputs import (
    AlignmentFile,
    MovieWriter,
    downsample2,
    matlab_round,
    publish,
    rec_neg_err,
    settings_record,
)
from .sources import has_samples, make_source
from .template import TemplateState, consensus, initial_template

log = logging.getLogger(__name__)


def matlab_colon(a: float, d: float, b: float) -> np.ndarray:
    """``a:d:b`` as MATLAB computes it, symmetrically from both ends."""
    n = int(np.floor((b - a) / d + 1e-10))
    k = np.arange(n + 1)
    return np.where(k <= n / 2, a + k * d, (a + n * d) - (n - k) * d)


def matlab_linspace(a: float, b: float, n: int) -> np.ndarray:
    """``linspace(a, b, n)`` with exact end points."""
    y = a + np.arange(n) * ((b - a) / (n - 1))
    y[0], y[-1] = a, b
    return y


@dataclass
class TrialRegistration:
    """Summary of one registered trial.

    Attributes
    ----------
    ds_frames : ndarray
        Line times of the registered frames.
    motion_r, motion_c : ndarray
        Row and column motion of each frame.
    peak : ndarray
        Peak correlation of each frame's match.
    registration_failed : bool
        Whether the motion's standard deviation exceeded 1.5 pixels.
    growth : GrowthQC or None
        How each frame was registered, with ``schedule='grow'``.
    timings : dict
        Wall time in seconds of each phase.
    """

    ds_frames: np.ndarray
    motion_r: np.ndarray
    motion_c: np.ndarray
    peak: np.ndarray
    registration_failed: bool
    growth: Optional[GrowthQC] = None
    timings: dict = field(default_factory=dict)


class Timer:
    """Wall time of consecutive phases (synchronizing CUDA at each mark)."""

    def __init__(self, device):
        """Start timing."""
        self.device = torch.device(device)
        self.times = {}
        self._last = time.perf_counter()

    def mark(self, name: str) -> None:
        """Record the time since the previous mark under ``name``."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        now = time.perf_counter()
        self.times[name] = self.times.get(name, 0.0) + now - self._last
        self._last = now


def resolve_device(device: str) -> torch.device:
    """``'auto'`` selects CUDA when available, else the CPU."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def frame_times(first_line: float, last_line: float, dt: float, n_init: int):
    """Line times of the initial frames and of the registered (DS) frames.

    Parameters
    ----------
    first_line, last_line : float
        One-based line range of the trial (may be fractional).
    dt : float
        Lines per registered frame (line rate / alignHz).
    n_init : int
        Number of initial frames.

    Returns
    -------
    init, ds : ndarray of int
        ``round(linspace(first, last, n_init + 2))`` without its ends, and
        ``ceil(first:dt:last)``.
    """
    init = matlab_round(matlab_linspace(first_line, last_line, n_init + 2))
    ds = np.ceil(matlab_colon(first_line, dt, last_line))
    return init[1:-1].astype(np.int64), ds.astype(np.int64)


def build_template(source, init, params, dtype):
    """Build the initial template from the ``init`` frames of a source.

    Returns
    -------
    state : TemplateState
        The initial template and its sums.
    cons : ConsensusResult
        The pairwise matching that chose its frames.
    """
    window = source.window
    vals, fresh = source.frames(init)
    y = vals.sum(0)
    images = window.image(y, float("nan"))
    cons = consensus(
        y,
        fresh,
        images,
        window.rows,
        window.cols,
        params.init_search,
        dtype,
        keep=has_samples(vals),
    )
    state = initial_template(
        images,
        window.image(fresh, 0.0),
        cons,
        params.maxshift,
        params.min_samples,
    )
    log.info(
        "template from %d of %d frames (reference %d)",
        cons.members.size,
        init.size,
        cons.reference,
    )
    return state, cons


class OutputSink:
    """Accumulate per-frame outputs: movie pages, variance factors, means.

    The movie and the alignment file are written under temporary names and
    replace any existing pair only in :meth:`finish`.

    Parameters
    ----------
    movie_path, adata_path : str or Path
        Destinations.
    out_shape : (int, int)
        Registered frame shape.
    n_frames, n_channels : int
        Frames and channels.
    save_tif : bool
        Movie format.
    device : torch.device
        Where the mean images accumulate.
    movie_deflate, var_deflate : int
        Deflate levels of an HDF5 movie and of ``varFacDS`` (0: none).
    """

    def __init__(
        self,
        movie_path,
        adata_path,
        out_shape,
        n_frames,
        n_channels,
        save_tif,
        device,
        movie_deflate: int = 1,
        var_deflate: int = 1,
    ):
        """Open the movie and the alignment file."""
        ho, wo = out_shape
        self.sum = torch.zeros(n_channels, ho, wo, device=device)
        self.count = torch.zeros(n_channels, ho, wo, device=device)
        self.movie_ds = np.full(
            (n_frames, ho // 4, wo // 4), np.nan, dtype=np.float32
        )
        self.movie = MovieWriter(
            movie_path,
            out_shape,
            n_frames * n_channels,
            save_tif,
            movie_deflate,
        )
        try:
            self.adata = AlignmentFile(
                adata_path, out_shape, n_frames, var_deflate
            )
        except BaseException:
            self.movie.abort()
            raise

    def add_missing(self, k: int) -> None:
        """Record frame ``k`` as missing: all-NaN pages and variance."""
        blank = torch.full(
            self.sum.shape[1:], float("nan"), device=self.sum.device
        )
        self.add(k, [blank] * self.sum.shape[0], blank)

    def add(self, k: int, regs, var_factor: torch.Tensor) -> torch.Tensor:
        """Record frame ``k``'s registered channels; return their sum."""
        self.adata.write_var_factor(k, var_factor.cpu().numpy())
        total = None
        for ch, reg in enumerate(regs):
            self.movie.write(reg.cpu().numpy())
            good = ~torch.isnan(reg)
            self.sum[ch] += torch.where(good, reg, 0.0)
            self.count[ch] += good
            total = reg if total is None else total + reg
        self.movie_ds[k] = downsample2(downsample2(total)).cpu().numpy()
        return total

    def mean_image(self) -> np.ndarray:
        """Per-channel mean of the registered frames (meanIM)."""
        mean = self.sum / torch.clamp(self.count, min=1)
        return mean.masked_fill(self.count < 1, float("nan")).cpu().numpy()

    def finish(self, fields: dict, settings: Optional[dict] = None) -> None:
        """Complete the alignment file and publish it with the movie."""
        self.movie.close()
        self.adata.finish(fields, settings)
        publish(self.movie, self.adata)

    def abort(self) -> None:
        """Discard the partial movie and alignment file."""
        try:
            self.movie.abort()
        finally:
            self.adata.abort()


def register_sequential(
    source,
    state: TemplateState,
    ds,
    params,
    sink: OutputSink,
    dtype,
    chunk: int = 32,
):
    """MATLAB's frame loop: match, resample, add to the template.

    Frames without valid samples are missing (NaN motion, all-NaN output)
    and skipped. A frame with samples whose correlation is undefined at
    every shift keeps the previous frame's motion, as the growth schedule
    gives such a frame its anchor's.

    Returns
    -------
    motion_r, motion_c, peak : ndarray
        Per-frame motion and peak correlation.
    """
    m = int(params.maxshift)
    window = source.window
    h, w = window.shape
    n = ds.size
    motion = np.zeros((3, n))
    init_r = init_c = 0
    prev = (0.0, 0.0)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        vals, fresh = source.frames(ds[a:b])
        present = has_samples(vals)
        mov = torch.sqrt(vals.sum(0))
        for i in range(b - a):
            if not present[i]:
                motion[:, a + i] = np.nan
                sink.add_missing(a + i)
                continue
            tmpl = state.matching_template()[
                m - init_r : m - init_r + h, m - init_c : m - init_c + w
            ]
            search = max(abs(init_c), abs(init_r), params.clip_shift)
            mot, pk, _ = match_maps(
                mov[i : i + 1],
                fresh[i : i + 1],
                tmpl,
                window.rows,
                window.cols,
                search,
                dtype,
            )
            if torch.isnan(pk[0]):
                mr, mc = prev
            elif torch.isnan(mot).any():
                raise RegistrationError("Computed motion contains NaN.")
            else:
                mr, mc = init_r + float(mot[0, 0]), init_c + float(mot[0, 1])
            motion[:, a + i] = mr, mc, float(pk[0])
            prev = (mr, mc)
            fimg = window.image(fresh[i], 0.0)
            regs = []
            for ch in range(vals.shape[0]):
                img = window.image(vals[ch, i], float("nan"))
                regs.append(interp_frame(img, fimg, mr, mc, m))
            state.add(sink.add(a + i, [r for r, _ in regs], regs[0][1]))
            init_r = int(np.clip(matlab_round(mr), -m, m))
            init_c = int(np.clip(matlab_round(mc), -m, m))
        log.info("%d of %d frames", b, n)
    return motion[0], motion[1], motion[2]


def register_grow(
    source,
    state,
    cons,
    init,
    ds,
    params,
    sink: OutputSink,
    dtype,
    timer: Optional[Timer] = None,
):
    """Batched growth from seed frames, then rendering in frame order.

    Returns
    -------
    motion_r, motion_c, peak : ndarray
        Per-frame motion and peak correlation.
    qc : GrowthQC
        How each frame was registered.
    """
    timer = timer or Timer(source.former.device)
    cache = FrameCache(source, ds)
    present = cache.present()
    if not present.all():  # frames without samples are left out
        cache.keep(present)
    timer.mark("frames")
    motion = np.full((ds.size, 2), np.nan)
    qc = GrowthQC.missing(ds.size)
    if present.any():
        reach = 2 * max(3 * source.dt, source.former.layout.lines_per_cycle)
        seeds, candidates, independent = seeds_from_consensus(
            cons, init, ds[present], reach
        )
        grower = Grower(
            cache, state, params, seeds, candidates, dtype, independent
        )
        grown, qc = grower.run()
        timer.mark("grow")
        if params.polish:
            grown = grower.polish(params.polish_radius)
            timer.mark("polish")
        motion[present] = grown
        qc = qc.expanded(present)
    m = int(params.maxshift)
    j = np.cumsum(present) - 1  # index of each frame in the cache
    for k in range(ds.size):
        if not present[k]:
            sink.add_missing(k)
            continue
        regs = cache.registered(int(j[k]), motion[k], m)
        sink.add(k, [r for r, _ in regs], regs[0][1])
    timer.mark("render")
    return motion[:, 0].copy(), motion[:, 1].copy(), qc.peak.copy(), qc


def motion_failed(motion_r, motion_c) -> bool:
    """``registrationFailed``: motion SD above 1.5 pixels on either axis.

    Missing frames (NaN motion) are left out. A trial without any frame
    that has samples counts as failed, so that SILo skips it (MATLAB's
    drifting search-edge motion fails it too).
    """
    ok = np.isfinite(motion_r) & np.isfinite(motion_c)
    if not ok.any():
        return True
    return bool(
        np.std(motion_c[ok], ddof=1) > 1.5
        or np.std(motion_r[ok], ddof=1) > 1.5
    )


def alignment_fields(
    layout: RasterLayout,
    source,
    ds,
    motion_r,
    motion_c,
    sink: OutputSink,
    params,
    failed: bool,
):
    """The ALIGNMENTDATA struct, as MultiRoiRegistration saves it."""
    m = int(params.maxshift)
    window = source.window
    h, w = window.shape
    record = {"enabled": np.int8(0)}
    if source.compensation is not None:
        record = compensation_record(
            source.compensation, source.references(ds)
        )
    ho, wo = h + 2 * m, w + 2 * m
    x_off, y_off, z_off = read_line_offsets(layout, ds)
    return {
        "numChannels": float(layout.num_channels),
        "meanIM": sink.mean_image().astype(np.float32),
        "frametime": 1.0 / params.align_hz,
        "alignHz": float(params.align_hz),
        "DSframes": ds.astype(np.float64),
        "motionDSc": motion_c,
        "motionDSr": motion_r,
        "recNegErr": rec_neg_err(sink.movie_ds, params.align_hz),
        "registrationFailed": failed,
        "slap2": {
            "Z_depths": layout.fast_zs.astype(np.float32),
            "cropRow": float(window.trim_rows[0] - m),
            "cropCol": float(window.trim_cols[0] - m),
            "viewC": np.repeat((np.arange(wo) + 1.0 - m)[None, :], ho, 0),
            "viewR": np.repeat((np.arange(ho) + 1.0 - m)[:, None], wo, 1),
            "trimRows": window.trim_rows.astype(np.float64),
            "trimCols": window.trim_cols.astype(np.float64),
            "onlineMotionXshift": x_off[:, None],
            "onlineMotionYshift": y_off[:, None],
            "onlineMotionZshift": z_off[:, None],
            "onlineCompensation": record,
        },
    }


def open_trial(dat_path, first_line, last_line, params, device):
    """Read a trial's samples and frame times.

    Returns
    -------
    layout : RasterLayout
        The recording's layout.
    former : FrameFormer
        Frame former over every cycle the trial's frames reach.
    init, ds : ndarray of int
        Initial and registered frame times.
    dtk : int
        Kernel time constant in lines.
    """
    layout = raster_layout(dat_path)
    if layout.fast_zs.size != 1:
        raise ValueError("only single-plane acquisitions are supported")
    if layout.num_channels not in (1, 2):
        raise ValueError("only one- or two-channel recordings are supported")
    p = layout.lines_per_cycle
    dt = (1.0 / layout.line_period_s) / params.align_hz
    dtk = int(np.ceil(dt))
    init, ds = frame_times(first_line, last_line, dt, params.init_frames)
    dt_read = max(3 * dtk, p)
    lo = max(1, int(min(init.min(), ds.min())) - dt_read)
    hi = min(layout.total_lines, int(max(init.max(), ds.max())) + dt_read)
    c0, c1 = (lo - 1) // p, (hi - 1) // p + 1
    samples = read_raster_samples(layout, (c0, c1))
    return layout, FrameFormer(layout, samples, c0, 1, device), init, ds, dtk


def register_trial(
    dat_path: Union[str, Path],
    first_line: float,
    last_line: float,
    movie_path: Union[str, Path],
    adata_path: Union[str, Path],
    params: Optional[MultiRoiRegistrationParams] = None,
    device="cpu",
    dtype=torch.float32,
) -> TrialRegistration:
    """Register the lines ``first_line .. last_line`` of a recording.

    Parameters
    ----------
    dat_path : str or Path
        The trial's ``.dat`` file.
    first_line, last_line : float
        One-based line range of the trial. Kept fractional, as in MATLAB:
        trial tables of continuous recordings cut trials at ``linspace``
        edges, and the frame times are ``ceil(first:dt:last)``.
    movie_path, adata_path : str or Path
        Destinations of the registered movie and the alignment data.
    params : MultiRoiRegistrationParams, optional
        Scientific settings (default: MATLAB's defaults).
    device : str or torch.device
        Where frames are formed and matched.
    dtype : torch.dtype
        Precision of the correlation sums.

    Returns
    -------
    TrialRegistration
        Motion and quality of the trial.

    Raises
    ------
    ValueError
        For recordings with more than one fast-z plane or channel counts
        other than 1 or 2.
    """
    if params is None:
        params = MultiRoiRegistrationParams()
    device = torch.device(device)
    timer = Timer(device)
    layout, former, init, ds, dtk = open_trial(
        dat_path, float(first_line), float(last_line), params, device
    )
    timer.mark("read")
    log.info("%s: %d frames", Path(dat_path).name, ds.size)
    source = make_source(
        layout,
        former,
        float(first_line),
        float(last_line),
        init,
        ds,
        dtk,
        params.compensate_online,
    )
    if source.compensation is not None:
        log.info(
            "undoing the online motion correction (%d x/y switches)",
            source.compensation.switch_lines.size,
        )
    state, cons = build_template(source, init, params, dtype)
    timer.mark("template")
    m = int(params.maxshift)
    h, w = source.window.shape
    sink = OutputSink(
        movie_path,
        adata_path,
        (h + 2 * m, w + 2 * m),
        ds.size,
        layout.num_channels,
        params.save_tif,
        device,
        params.movie_deflate,
        params.var_factor_deflate,
    )
    qc = None
    try:
        if params.schedule == "grow":
            motion_r, motion_c, peak, qc = register_grow(
                source, state, cons, init, ds, params, sink, dtype, timer
            )
        else:
            motion_r, motion_c, peak = register_sequential(
                source, state, ds, params, sink, dtype
            )
            timer.mark("register")
        sink.movie.close()
        timer.mark("movie")
        failed = motion_failed(motion_r, motion_c)
        if not np.isfinite(motion_r).any():
            log.warning("no frame of %s has valid samples", dat_path)
        elif failed:
            log.warning("too much motion in %s; saving anyway", dat_path)
        fields = alignment_fields(
            layout, source, ds, motion_r, motion_c, sink, params, failed
        )
        if qc is not None:
            fields["growthQC"] = qc.fields()
        sink.finish(fields, settings_record(params))
    except BaseException:
        sink.abort()
        raise
    timer.mark("alignment file")
    log.info(
        "timings (s): %s", {k: round(v, 2) for k, v in timer.times.items()}
    )
    return TrialRegistration(
        ds, motion_r, motion_c, peak, failed, qc, timer.times
    )
