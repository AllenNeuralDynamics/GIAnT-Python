"""Register one SLAP2 raster trial (``alignAsync`` of MultiRoiRegistration).

The trial's frames are registered sequentially, as in GIAnT-MATLAB: each DS
frame is matched to the mean of the initial and running templates, starting
from the previous frame's rounded motion with a search radius of
``max(|init|, clip_shift)``; its registered image is then added to the
running template. With ``compensate_online``, frames are formed with the
online motion correction undone line by line (see :mod:`.compensation`).
"""

import logging
from dataclasses import dataclass
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
from .interpolation import interp_frame
from .matching import match
from .outputs import (
    AlignmentFile,
    MovieWriter,
    downsample2,
    matlab_round,
    rec_neg_err,
)
from .sources import make_source
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
    """

    ds_frames: np.ndarray
    motion_r: np.ndarray
    motion_c: np.ndarray
    peak: np.ndarray
    registration_failed: bool


def resolve_device(device: str) -> torch.device:
    """``'auto'`` selects CUDA when available, else the CPU."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def frame_times(first_line: int, last_line: int, dt: float, n_init: int):
    """Line times of the initial frames and of the registered (DS) frames.

    Parameters
    ----------
    first_line, last_line : int
        One-based line range of the trial.
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
    TemplateState
        The initial template and its sums.
    """
    window = source.window
    vals, fresh = source.frames(init)
    y = vals.sum(0)
    images = window.image(y, float("nan"))
    cons = consensus(
        y, fresh, images, window.rows, window.cols, params.init_search, dtype
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
    return state


class OutputSink:
    """Accumulate per-frame outputs: movie pages, variance factors, means.

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
    ):
        """Open the movie and the alignment file."""
        ho, wo = out_shape
        self.sum = torch.zeros(n_channels, ho, wo, device=device)
        self.count = torch.zeros(n_channels, ho, wo, device=device)
        self.movie_ds = np.full(
            (n_frames, ho // 4, wo // 4), np.nan, dtype=np.float32
        )
        self.movie = MovieWriter(
            movie_path, out_shape, n_frames * n_channels, save_tif
        )
        self.adata = AlignmentFile(adata_path, out_shape, n_frames)

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

    def abort(self) -> None:
        """Close the movie and discard the partial alignment file."""
        self.movie.close()
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
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        vals, fresh = source.frames(ds[a:b])
        mov = torch.sqrt(vals.sum(0))
        for i in range(b - a):
            tmpl = state.matching_template()[
                m - init_r : m - init_r + h, m - init_c : m - init_c + w
            ]
            search = max(abs(init_c), abs(init_r), params.clip_shift)
            mot, pk = match(
                mov[i : i + 1],
                fresh[i : i + 1],
                tmpl,
                window.rows,
                window.cols,
                search,
                dtype,
            )
            mr, mc = init_r + float(mot[0, 0]), init_c + float(mot[0, 1])
            motion[:, a + i] = mr, mc, float(pk[0])
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
    first_line: int,
    last_line: int,
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
    first_line, last_line : int
        One-based line range of the trial.
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
    layout, former, init, ds, dtk = open_trial(
        dat_path, int(first_line), int(last_line), params, device
    )
    log.info("%s: %d frames", Path(dat_path).name, ds.size)
    source = make_source(
        layout,
        former,
        int(first_line),
        int(last_line),
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
    state = build_template(source, init, params, dtype)
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
    )
    try:
        motion_r, motion_c, peak = register_sequential(
            source, state, ds, params, sink, dtype
        )
    except Exception:
        sink.abort()
        raise
    sink.movie.close()
    failed = bool(
        np.std(motion_c, ddof=1) > 1.5 or np.std(motion_r, ddof=1) > 1.5
    )
    if failed:
        log.warning("too much motion in %s; saving anyway", dat_path)
    sink.adata.finish(
        alignment_fields(
            layout, source, ds, motion_r, motion_c, sink, params, failed
        )
    )
    return TrialRegistration(ds, motion_r, motion_c, peak, failed)
