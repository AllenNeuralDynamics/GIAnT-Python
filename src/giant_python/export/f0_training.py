"""Training data for the SLAP2 F0 network from a ``giant silo`` run.

The Python counterpart of slap2-F0-network's ``exportF0TrainingData.m``,
built on the renderer of ``giant view`` (:mod:`giant_python.viewer.render`).
For every imaging path and every trial that SILo analysed it writes
``<out>/Path<k>/trial_<NNN>.h5`` in the layout ``slap2_f0.io.TrialFile``
reads, plus ``<out>/export_summary.json`` with each trial's QC. Arrays are
written so that h5py reads them in C order, on the trial's 80 Hz registered
grid cropped to the box around the valid pixels (pixel coordinates zero-based
in the crop, MATLAB's row and column orientation):

``movie`` (channel, frame, row, col) float32
    The registered movie, NaN at discarded frames and where not imaged.
``activity`` (channel, frame, row, col) float32
    SILo's fitted activity on the same grid and timing; the training target
    is ``movie - activity``. Zero outside the footprints.
``mean_image`` (channel, row, col) float32
    Mean over all frames of each written channel (the MATLAB exporter wrote
    every channel of the file, which the reader indexes like ``movie``).
``mask/valid``, ``mask/labeled``, ``mask/footprint_support`` (row, col) uint8
    Pixels missing in less than ``nanThresh`` of the frames (motion discards
    counted); pixels brighter than 3 x the 25th percentile of the 3 x 3
    median-filtered mean image; pixels any source's footprint covers.
``frame/discard``, ``frame/motion_row``, ``frame/motion_col``,
``frame/rec_neg_err``, ``frame/onlineMotion{X,Y,Z}shift``,
``frame/valid_mean`` (frame,)
    Discarded frames (SILo's motion discards and frames whose kernel weight
    lies mostly on SILo's discarded 200 Hz frames), the registration's
    motion and error, the online motion record, the mean of the valid
    pixels.
``varfac`` (row, col), or (frame, row, col) with ``var_fac="full"``
    SLAP2's variance factor (``slap2/varFacDS``), its per-pixel mean.
``sources/peak_row``, ``sources/peak_col`` (source,) int32
    Each kept source's peak pixel in the trial (-1: no footprint there).

The root attributes hold the QC (:mod:`~giant_python.export.f0_qc`) and the
provenance under the MATLAB exporter's names, plus ``exporter``,
``exporterVersion``, ``kernel``, ``kernelR``, ``kernelDt200Lines``,
``alignmentMode``, ``siloRun``, ``baselineSource``, ``baselineFile`` and
``f0Network`` (SILo's record of the network baseline of the trial).

How the activity is rendered, and what differs from the MATLAB exporter:

* **Space.** Each trial's footprints are moved from SILo's consensus canvas
  into the trial's registered frame through SILo's own across-trial map
  (rigid, affine or patchwise; :func:`~giant_python.viewer.render.
  place_footprints`), not by an offset searched for the best movie-activity
  correlation.
* **Time.** The 200 Hz traces become 80 Hz frames with the ``exact`` kernel
  by default (:func:`~giant_python.viewer.render.exact_weights`); the MATLAB
  exporter's weights (``kernel="exporter"``) smooth fast events twice and
  keep 0.93-0.95 of their 80 Hz peak.
* **Blur.** None by default (a fixed ``blur`` is an option); the MATLAB
  exporter chose a blur per trial by matching the movie's core and ring
  slopes. ``blurShapeRatio`` still reports that core / ring slope ratio for
  the activity as written.
* **Scale.** None: footprints (each summing to 1) times ``dF_denoised``
  (``activityGain`` 1), as the MATLAB exporter now writes it.
"""

from __future__ import annotations

import json
import math
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from scipy.ndimage import median_filter

from .. import __version__
from ..extraction.raster.localization import matlab_prctile
from ..extraction.raster.trial import _read_tiff_pages, discard_frames
from ..viewer.render import (
    TrialMap,
    kernel_dt,
    kernel_weights,
    place_footprints,
)
from ..viewer.run import SiloRun
from ..viewer.trial import Session
from . import f0_qc

EXPORTER = "giant_python.export.f0_training"
INTERP_ORDER = {"linear": 1, "cubic": 3}


@dataclass
class ExportOptions:
    """Settings of an F0 training export.

    Attributes
    ----------
    kernel : str
        Time kernel (``exact``, ``matched`` or ``exporter``; see
        :func:`~giant_python.viewer.render.kernel_weights`).
    blur : float
        Gaussian blur (px) of the moved footprints (0: none).
    interp : str
        How footprints are moved into the trial (``linear`` or ``cubic``;
        :func:`~giant_python.viewer.render.place_footprints`).
    channels : tuple of int, optional
        One-based channels whose movie and activity are written (default:
        SILo's activity channel).
    var_fac : str
        ``mean`` (per-pixel mean of ``varFacDS``) or ``full`` (every frame).
    deflate : int
        HDF5 gzip level of the arrays (with shuffle).
    label : str
        Session label stored with the data (default: the run folder's name).
    giant_ref : str
        Version or commit of the GIAnT that made the run (metadata).
    paths, trials : tuple of int, optional
        One-based paths and trials to export (default: all SILo analysed).
    workers : int
        Trials exported at once in separate processes (each needs a few GB).
    ignore_sources : bool
        Export as if SILo had found no sources (zero activity).
    """

    kernel: str = "exact"
    blur: float = 0.0
    interp: str = "linear"
    channels: Optional[tuple] = None
    var_fac: str = "mean"
    deflate: int = 1
    label: str = ""
    giant_ref: str = ""
    paths: Optional[tuple] = None
    trials: Optional[tuple] = None
    workers: int = 1
    ignore_sources: bool = False


@dataclass
class TrialJob:
    """Everything one trial's export needs (picklable for the workers).

    ``trial`` and ``path`` are one-based. ``x200`` is ``(n200, channels,
    sources)``; ``footprints`` the trial's canvas footprints ``(rows, cols,
    sources, values)``; both None for a path without sources.
    """

    path: int
    trial: int
    out_file: Path
    movie_file: Path
    alignment_file: Path
    baseline_file: Optional[Path]
    lines_per_cycle: int
    settings: dict
    options: ExportOptions
    n_sources: int = 0
    footprints: Optional[tuple] = None
    tmap: Optional[TrialMap] = None
    x200: Optional[np.ndarray] = None
    lines200: Optional[np.ndarray] = None
    discard200: Optional[np.ndarray] = None
    provenance: dict = field(default_factory=dict)


class TrialSkipped(RuntimeError):
    """A trial that cannot be exported (no map or footprints in SILo)."""


def _settings(run: SiloRun) -> dict:
    """SILo's frame settings of a run (SILo's SLAP2 defaults)."""
    return {
        "num_channels": int(run.param("numChannels", 2)),
        "activity_channel": int(run.param("activityChannel", 1)),
        "motion_thresh": float(run.param("motionThresh", 2.0)),
        "discard_initial_s": float(run.param("discardInitial_s", 0.0)),
        "nan_thresh": float(run.param("nanThresh", 0.33)),
        "ratio": float(run.param("analyzeHz", 200.0))
        / float(run.param("alignHz", 80.0)),
        "align_hz": float(run.param("alignHz", 80.0)),
        "photon_scale": float(run.param("photonScale", np.nan)),
    }


def read_stack(path: Path, num_channels: int) -> np.ndarray:
    """A registered movie as ``(frames, channels, rows, cols)`` float32
    (MATLAB's orientation; channels interleaved by frame)."""
    path = Path(path)
    if path.suffix == ".h5":
        with h5py.File(path, "r") as f:
            raw = f[next(iter(f.keys()))][()]  # (pages, cols, rows)
    else:
        raw = _read_tiff_pages(path)  # (pages, MATLAB col, MATLAB row)
    raw = np.asarray(raw, np.float32).transpose(0, 2, 1)
    n = raw.shape[0] // num_channels
    return raw[: n * num_channels].reshape(
        n, num_channels, raw.shape[1], raw.shape[2]
    )


def _vec(f, name: str, n: Optional[int] = None):
    """A dataset of the alignment file as a flat float64 vector (None if
    missing)."""
    if name not in f:
        return None
    v = np.asarray(f[name][()], np.float64).ravel()
    return v if n is None else v[:n]


def read_alignment(path: Path, n_frames: int) -> dict:
    """The alignment file's frame lines, rate, error and motion records."""
    with h5py.File(path, "r") as f:
        out = {
            "ds_lines": _vec(f, "DSframes", n_frames),
            "align_hz": float(_vec(f, "alignHz")[0]),
            "rec_neg_err": _vec(f, "recNegErr"),
            "motion_row": _vec(f, "motionDSr"),
            "motion_col": _vec(f, "motionDSc"),
            "online": {
                k: _vec(f, f"slap2/{k}")
                for k in (
                    "onlineMotionXshift",
                    "onlineMotionYshift",
                    "onlineMotionZshift",
                )
                if f"slap2/{k}" in f
            },
        }
    return out


def read_varfac(path: Path, crop: tuple, mode: str) -> Optional[np.ndarray]:
    """``slap2/varFacDS`` in the crop: per-pixel mean ``(rows, cols)`` or
    every frame ``(frames, rows, cols)``; None without it."""
    r0, r1, c0, c1 = crop
    with h5py.File(path, "r") as f:
        if "slap2/varFacDS" not in f:
            return None
        ds = f["slap2/varFacDS"]  # (frames, cols, rows): MATLAB R x C x T
        if mode == "full":
            return np.ascontiguousarray(
                np.asarray(ds[:, c0:c1, r0:r1], np.float32).transpose(0, 2, 1)
            )
        tot = np.zeros((c1 - c0, r1 - r0))
        cnt = np.zeros((c1 - c0, r1 - r0))
        for a in range(0, ds.shape[0], 256):
            blk = np.asarray(ds[a : a + 256, c0:c1, r0:r1], np.float64)
            ok = np.isfinite(blk)
            tot += np.where(ok, blk, 0).sum(0)
            cnt += ok.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, tot / cnt, np.nan).T.astype(np.float32)


def labeled_mask(mean_act: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Labeled pixels: the 3 x 3 median-filtered mean image (zero padding,
    as MATLAB's ``medfilt2``) above 3 x its 25th percentile; for a dim trial
    where nearly every valid pixel would count, the percentile over the
    valid pixels instead."""
    nan = np.isnan(mean_act)
    mf = median_filter(np.where(nan, np.inf, mean_act), 3, mode="constant")
    mf[~np.isfinite(mf)] = np.nan
    lab = ~nan & (mf > 3 * matlab_prctile(mf, 25))
    if (valid & ~lab).sum() < 0.05 * valid.sum():
        lab = ~nan & (mf > 3 * matlab_prctile(mf[valid], 25))
    return lab


def _nanmedian(x) -> float:
    """Median ignoring NaN (NaN when empty), without warnings."""
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def _crop_box(valid: np.ndarray) -> tuple:
    """``(r0, r1, c0, c1)`` (end exclusive) of the box around the valid
    pixels."""
    rows = np.flatnonzero(valid.any(1))
    cols = np.flatnonzero(valid.any(0))
    if rows.size == 0:
        raise TrialSkipped("no valid pixels")
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def _render(job: TrialJob, ds_lines, shape):
    """The trial's footprints in its frame, the sources' 80 Hz traces per
    SILo channel, the 80 Hz discard share and the kernel's record."""
    if job.tmap is None:
        raise TrialSkipped("SILo has no across-trial map for this trial")
    if job.footprints is None or not len(job.footprints[0]):
        raise TrialSkipped("SILo has no footprints for this trial")
    o, st = job.options, job.settings
    dt80 = kernel_dt(ds_lines, job.lines200, st["ratio"])
    w, share, info = kernel_weights(
        o.kernel,
        job.lines200,
        job.discard200,
        ds_lines,
        dt80,
        job.lines_per_cycle,
    )
    if o.kernel == "matched":  # the discard rule is the movie kernel's
        _, share, _ = kernel_weights(
            "exporter",
            job.lines200,
            job.discard200,
            ds_lines,
            dt80,
            job.lines_per_cycle,
        )
    x200 = np.nan_to_num(np.asarray(job.x200, np.float64))
    x80 = [
        np.asarray((w.T @ x200[:, c, :]).T, np.float64)
        for c in range(x200.shape[1])
    ]
    H = place_footprints(
        job.footprints,
        job.n_sources,
        job.tmap,
        shape,
        o.blur,
        INTERP_ORDER[o.interp],
    )
    info["dt80"] = dt80
    return H, x80, share, info


def _qc(job, H, Xa, Y, valid, crop, discard):
    """The QC values of the activity as written (see
    :mod:`~giant_python.export.f0_qc`)."""
    shape = valid.shape
    pk = f0_qc.peak_pixels(H)
    sl, r = f0_qc.peak_fit(H, Xa, Y, pk)
    dsl, dr = f0_qc.disk_fit(H, Xa, Y, pk, shape)
    pk0, act_slope = pk, _nanmedian(dsl)
    if job.options.blur > 0:  # the slope and the shape cores at blur 0
        H0 = place_footprints(
            job.footprints,
            job.n_sources,
            job.tmap,
            shape,
            0.0,
            INTERP_ORDER[job.options.interp],
        )
        pk0 = f0_qc.peak_pixels(H0)
        act_slope = _nanmedian(f0_qc.disk_fit(H0, Xa, Y, pk0, shape)[0])
    rho = f0_qc.shape_fit(H, Xa, Y, valid, pk0)
    frac, n_ev = f0_qc.event_residual(H, Xa, Y, pk)
    ring = _ring_qc(H, Xa, Y, valid, crop, pk, discard)
    return pk, {
        "blurShapeRatio": _nanmedian(rho),
        "activitySlope": act_slope,
        "peakSlope": _nanmedian(sl),
        "peakR": _nanmedian(r),
        "diskSlope": _nanmedian(dsl),
        "diskR": _nanmedian(dr),
        "eventResidualFraction": frac,
        "nEvents": int(n_ev),
        "eventRatioCore": ring.core,
        "eventRatio2px": ring.near,
        "eventRatioSources": ring.n_sources,
        "eventRatioEvents": ring.n_events,
        "eventRatioArtifactSources": ring.n_artifact_sources,
    }


def _ring_qc(H, Xa, Y, valid, crop, pk, discard):
    """The ring-event ratios on the cropped arrays (a source counts only
    when its 17 x 17 px box lies inside the crop, as in the MATLAB
    exporter)."""
    r0, r1, c0, c1 = crop
    R, C = valid.shape
    cc = c1 - c0

    def act_at(flat):
        """Activity at the crop's flat pixels, ``(frames, n)``."""
        full = (flat // cc + r0) * C + flat % cc + c0
        return np.asarray(H[full] @ Xa).T

    mean_act = np.asarray(H @ Xa.mean(1)).reshape(R, C)[r0:r1, c0:c1]
    movie = Y.reshape(Y.shape[0], R, C)[:, r0:r1, c0:c1]
    far = -(10**6)
    pr = np.where(pk >= 0, pk // C - r0, far)
    pc = np.where(pk >= 0, pk % C - c0, far)
    return f0_qc.ring_event_ratios(
        movie, act_at, mean_act, valid[r0:r1, c0:c1], pr, pc, discard
    )


NO_QC = {
    "blurShapeRatio": np.nan,
    "activitySlope": np.nan,
    "peakSlope": np.nan,
    "peakR": np.nan,
    "diskSlope": np.nan,
    "diskR": np.nan,
    "eventResidualFraction": np.nan,
    "nEvents": 0,
    "eventRatioCore": np.nan,
    "eventRatio2px": np.nan,
    "eventRatioSources": 0,
    "eventRatioEvents": 0,
    "eventRatioArtifactSources": 0,
}


def _create(f, name: str, data, deflate: int, chunks=None) -> None:
    """Write a dataset with gzip and shuffle (one chunk unless given)."""
    data = np.asarray(data)
    if data.size == 0:
        f.create_dataset(name, data=data)
        return
    f.create_dataset(
        name,
        data=data,
        chunks=chunks if chunks is not None else data.shape,
        compression="gzip",
        compression_opts=deflate,
        shuffle=True,
    )


def _movie_chunks(shape: tuple) -> tuple:
    """Chunks of a (channel, frame, row, col) array, as the MATLAB
    exporter's (one channel, 256 frames, 64 x 64 px)."""
    return (1, min(256, shape[1]), min(64, shape[2]), min(64, shape[3]))


def _load(job: TrialJob):
    """The movie stack, its alignment record, the motion discards and the
    valid pixels."""
    st = job.settings
    stack = read_stack(job.movie_file, st["num_channels"])
    T = stack.shape[0]
    al = read_alignment(job.alignment_file, T)
    disc = discard_frames(
        al["rec_neg_err"][:T],
        al["align_hz"],
        st["motion_thresh"],
        st["discard_initial_s"],
    )
    act = stack[:, st["activity_channel"] - 1]
    miss = np.isnan(act) | disc[:, None, None]
    valid = miss.mean(0) < st["nan_thresh"]
    return stack, al, disc, valid


def _activity(H, x80, chans: list, crop: tuple, shape: tuple, T: int):
    """The written channels' activity in the crop ``(channels, frames,
    rows, cols)`` float32, computed at the footprint pixels (zero
    elsewhere)."""
    r0, r1, c0, c1 = crop
    C = shape[1]
    out = np.zeros((len(chans), T, r1 - r0, c1 - c0), np.float32)
    if H is None:
        return out
    qr, qc = np.mgrid[r0:r1, c0:c1]
    flat = (qr * C + qc).ravel()
    Hc = H[flat]
    sup = np.flatnonzero(np.asarray(Hc.getnnz(axis=1)) > 0)
    for k, ch in enumerate(chans):
        x = x80[min(ch, len(x80)) - 1]
        a = np.asarray(Hc[sup] @ x, np.float32)  # (pixels, frames)
        view = out[k].reshape(T, -1)
        view[:, sup] = a.T
    return out


def _write(job: TrialJob, arrays: dict, attrs: dict) -> None:
    """Write one trial file (through a temporary name, replacing an
    existing file)."""
    dfl = job.options.deflate
    out = Path(job.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with h5py.File(tmp, "w") as f:
        for name, a in arrays.items():
            chunks = None
            if name in ("movie", "activity"):
                chunks = _movie_chunks(a.shape)
            elif name == "varfac" and np.ndim(a) == 3:
                chunks = (min(256, a.shape[0]),) + tuple(
                    min(64, n) for n in a.shape[1:]
                )
            _create(f, name, a, dfl, chunks)
        for k, v in attrs.items():
            f.attrs[k] = v
    tmp.replace(out)


def _provenance(job: TrialJob) -> dict:
    """String attributes naming the run, the baseline and the exporter."""
    p = job.provenance
    return {
        "label": p.get("label", ""),
        "giantRef": job.options.giant_ref,
        "exporter": EXPORTER,
        "exporterVersion": __version__,
        "siloRun": p.get("silo_run", ""),
        "baselineSource": p.get("baseline_source", ""),
        "alignmentMode": p.get("alignment", ""),
        "baselineFile": job.baseline_file.name if job.baseline_file else "",
        "f0Network": json.dumps(p.get("network", {}), sort_keys=True),
    }


def _attr_value(v):
    """An HDF5 attribute value (lists as float64 arrays)."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return np.asarray(v, np.float64)
    return v


def _map_shift(job: TrialJob) -> list:
    """The map's mean offset (trial minus canvas) over the footprint
    pixels: the counterpart of the MATLAB exporter's searched ``shiftRC``."""
    if job.tmap is None or job.footprints is None:
        return [np.nan, np.nan]
    r = np.asarray(job.footprints[0], float)
    c = np.asarray(job.footprints[1], float)
    if r.size == 0:
        return [np.nan, np.nan]
    rt, ct = job.tmap.forward(r, c)
    return [float(np.mean(rt - r)), float(np.mean(ct - c))]


def _support(H, shape) -> np.ndarray:
    """Pixels any footprint covers, ``(rows, cols)`` bool."""
    if H is None:
        return np.zeros(shape, bool)
    return (np.asarray(H.getnnz(axis=1)) > 0).reshape(shape)


def _frame_arrays(al: dict, disc, Y, valid) -> dict:
    """The per-frame datasets."""
    out = {"frame/discard": disc.astype(np.uint8)}
    for key in ("motion_row", "motion_col", "rec_neg_err"):
        if al[key] is not None:
            out[f"frame/{key}"] = al[key].astype(np.float32)
    for k, v in al["online"].items():
        out[f"frame/{k}"] = v.astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        gm = np.nanmean(Y[:, valid.ravel()], axis=1)
    out["frame/valid_mean"] = gm.astype(np.float32)
    return out


def _baseline_qc(job: TrialJob, arrays: dict, crop: tuple) -> dict:
    """With the trial's network baseline: the time-mean of the training
    target (movie - activity) over that of the baseline, at valid
    footprint pixels and at valid labeled pixels outside the footprints
    (``targetOverBaselineSupport``, ``targetOverBaselineLabeled``; NaN
    without a baseline)."""
    out = {
        "targetOverBaselineSupport": np.nan,
        "targetOverBaselineLabeled": np.nan,
    }
    ach = job.settings["activity_channel"]
    chans = list(job.options.channels or (ach,))
    if job.baseline_file is None or ach not in chans:
        return out
    k = chans.index(ach)
    r0, r1, c0, c1 = crop
    with h5py.File(job.baseline_file, "r") as f:
        b = np.asarray(f["baseline"][:, c0:c1, r0:r1], np.float32)
    b = b.transpose(0, 2, 1)  # (frames, rows, cols)
    if b.shape != arrays["movie"].shape[1:]:
        return out
    target = arrays["movie"][k] - arrays["activity"][k]
    valid = arrays["mask/valid"].astype(bool)
    sup = arrays["mask/footprint_support"].astype(bool) & valid
    lab = arrays["mask/labeled"].astype(bool) & valid & ~sup
    ok = np.isfinite(target) & np.isfinite(b)
    for key, m in (
        ("targetOverBaselineSupport", sup),
        ("targetOverBaselineLabeled", lab),
    ):
        sel = ok & m[None]
        if sel.any():
            out[key] = float(target[sel].mean() / b[sel].mean())
    return out


def _arrays(job, stack, Y, H, x80, valid, crop, disc, pk) -> dict:
    """Every dataset of the trial file but the per-frame ones."""
    st, o = job.settings, job.options
    ach = st["activity_channel"]
    chans = list(o.channels or (ach,))
    T, _, R, C = stack.shape
    r0, r1, c0, c1 = crop
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_all = np.nanmean(stack, axis=0)  # (channels, rows, cols)
    movie = np.stack([stack[:, ch - 1, r0:r1, c0:c1] for ch in chans], 0)
    movie[:, disc] = np.nan
    lab = labeled_mask(mean_all[ach - 1], valid)
    out = {
        "movie": movie,
        "activity": _activity(H, x80, chans, crop, (R, C), T),
        "mean_image": mean_all[[ch - 1 for ch in chans], r0:r1, c0:c1].astype(
            np.float32
        ),
        "mask/valid": valid[r0:r1, c0:c1].astype(np.uint8),
        "mask/labeled": lab[r0:r1, c0:c1].astype(np.uint8),
        "mask/footprint_support": _support(H, (R, C))[r0:r1, c0:c1].astype(
            np.uint8
        ),
    }
    vf = read_varfac(job.alignment_file, crop, o.var_fac)
    if vf is None:
        vf = np.ones((r1 - r0, c1 - c0), np.float32)
    out["varfac"] = vf
    if H is not None:
        out["sources/peak_row"] = np.where(pk >= 0, pk // C - r0, -1).astype(
            np.int32
        )
        out["sources/peak_col"] = np.where(pk >= 0, pk % C - c0, -1).astype(
            np.int32
        )
    return out


NO_KERNEL = {
    "dt80": np.nan,
    "dt": np.nan,
    "reach": np.nan,
    "dt200": np.nan,
    "r": np.nan,
}


def _record(job, T, crop, valid, H, kern, qc, disc) -> dict:
    """The trial's summary record (also its attributes, without
    ``file``)."""
    o, st = job.options, job.settings
    r0, r1, c0, c1 = crop
    ach = st["activity_channel"]
    return {
        "path": job.path,
        "trial": job.trial,
        "file": str(job.out_file),
        "frames": int(T),
        "rows": r1 - r0,
        "cols": c1 - c0,
        "cropOriginRC": [r0, c0],
        "nSources": int(job.n_sources if H is not None else 0),
        "validPixels": int(valid.sum()),
        "kernel": o.kernel,
        "kernelDtLines": kern["dt"],
        "kernelHalfWidthLines": kern["reach"],
        "kernelDt200Lines": kern["dt200"],
        "kernelR": kern["r"],
        "movieDtLines": kern["dt80"],
        "linesPerCycle": int(job.lines_per_cycle),
        "shiftRC": _map_shift(job) if H is not None else [np.nan, np.nan],
        "blurSigma": float(o.blur),
        "footprintInterp": o.interp,
        "activityGain": 1.0,
        **qc,
        "discardedFrames": int(disc.sum()),
        "channels": list(o.channels or (ach,)),
        "activityChannel": ach,
        "alignHz": st["align_hz"],
        "photonScale": st["photon_scale"],
        **_provenance(job),
    }


def _export_trial(job: TrialJob) -> dict:
    """The work of :func:`export_trial`."""
    o, st = job.options, job.settings
    stack, al, disc, valid = _load(job)
    T, _, R, C = stack.shape
    ach = st["activity_channel"]
    H, x80, kern = None, None, dict(NO_KERNEL)
    if job.n_sources and not o.ignore_sources:
        H, x80, share, kern = _render(job, al["ds_lines"], (R, C))
        disc = disc | (share > 0.5)
    crop = _crop_box(valid)
    Y = stack[:, ach - 1].reshape(T, R * C).copy()
    Y[disc] = np.nan
    pk, qc = np.zeros(0, np.int64), dict(NO_QC)
    if H is not None:
        pk, qc = _qc(job, H, x80[min(ach, len(x80)) - 1], Y, valid, crop, disc)
    arrays = _arrays(job, stack, Y, H, x80, valid, crop, disc, pk)
    del stack
    arrays.update(_frame_arrays(al, disc, Y, valid))
    q = _record(job, T, crop, valid, H, kern, qc, disc)
    q.update(_baseline_qc(job, arrays, crop))
    attrs = {k: _attr_value(v) for k, v in q.items() if k != "file"}
    _write(job, arrays, attrs)
    return q


def export_trial(job: TrialJob) -> dict:
    """Export one trial; returns its summary record (``error`` instead
    when it could not be exported)."""
    t0 = time.time()
    try:
        q = _export_trial(job)
    except (TrialSkipped, OSError, ValueError, KeyError) as exc:
        q = {"path": job.path, "trial": job.trial, "error": str(exc)}
        out = Path(job.out_file)
        if out.exists():  # the reader finds files, not summaries
            out.unlink()
            q["error"] += "; the earlier file was removed"
        return q
    q["seconds"] = round(time.time() - t0, 1)
    return q


# --------------------------------------------------------------- the run


def _denoised(run: SiloRun, d: int) -> Optional[np.ndarray]:
    """``dF_denoised`` of path ``d``, ``(frames, channels, sources)``."""
    with h5py.File(run.experiment, "r") as f:
        key = f"Path{d}/sources/temporal/dF_denoised"
        if key not in f:
            return None
        return np.asarray(f[key][()], np.float32)


def _network_records(run: SiloRun) -> dict:
    """SILo's per-trial records of the network baseline (empty without)."""
    try:
        return json.loads(run.attrs.get("network_baseline") or "{}")
    except (TypeError, ValueError):
        return {}


def _path_jobs(run, session, d, out_dir, options, prov) -> list:
    """The trial jobs of one path."""
    p = run.path(d)
    st = _settings(run)
    n = 0 if options.ignore_sources else p.sources.n
    fps = run.footprints(d) if n else [None] * p.n_frames.size
    x = _denoised(run, d) if n else None
    if n and x is None:
        n = 0
    if options.trials:
        trials = [t - 1 for t in options.trials]
    elif (p.n_frames > 0).any():  # the trials SILo analysed
        trials = [t for t in range(p.n_frames.size) if p.n_frames[t] > 0]
    else:  # no frame counts (a path without sources): registered movies
        trials = [
            t
            for t in range(session.fn_reg_ds.shape[1])
            if session.has_movie(d - 1, t)
        ]
    recs = _network_records(run)
    jobs = []
    for t in trials:
        span = p.trial_span(t) if t < p.n_frames.size else slice(0, 0)
        job = TrialJob(
            path=d,
            trial=t + 1,
            out_file=Path(out_dir) / f"Path{d}" / f"trial_{t + 1:03d}.h5",
            movie_file=session.movie_file(d - 1, t),
            alignment_file=session.alignment_file(d - 1, t),
            baseline_file=session.baseline_file(d - 1, t),
            lines_per_cycle=session.lines_per_cycle(d - 1, t),
            settings=st,
            options=options,
            provenance=dict(
                prov,
                alignment=p.alignment,
                network=recs.get(f"path{d}_trial{t + 1:04d}", {}),
            ),
        )
        if n:
            job.n_sources = n
            job.footprints = fps[t] if t < len(fps) else None
            job.tmap = p.maps[t] if t < len(p.maps) else None
            job.x200 = x[span]
            job.lines200 = p.lines[span]
            job.discard200 = p.discard[span]
        jobs.append(job)
    return jobs


def _json_safe(v):
    """NaN and infinities as None, NumPy values as Python ones."""
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_json_safe(x) for x in np.asarray(v, dtype=object).tolist()]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(v) else None
    return v


def write_summary(out_dir: Path, summary: dict) -> None:
    """``export_summary.json`` (NaN written as null, as MATLAB's
    ``jsonencode``)."""
    path = Path(out_dir) / "export_summary.json"
    path.write_text(json.dumps(_json_safe(summary), indent=1))


def export_run(
    run_dir,
    out_dir,
    table=None,
    moco_dir=None,
    baseline_dir=None,
    data_dir=None,
    options: Optional[ExportOptions] = None,
    log=print,
) -> dict:
    """Export a SILo run as F0 training data.

    Parameters
    ----------
    run_dir : str or Path
        The ``giant silo`` run (its folder or ``source_extraction``).
    out_dir : str or Path
        Output folder (``Path<k>/trial_<NNN>.h5``, ``export_summary.json``).
    table, moco_dir, baseline_dir, data_dir : str or Path, optional
        As ``giant view``: the trial table (default: in or above the run),
        the registered movies and alignment files (default: the table's
        ``savedr/motion_correction``), the network baselines (optional;
        provenance and the target-over-baseline QC), the raw files (for
        the lines per cycle; without them the kernels reach 3 dt).
    options : ExportOptions, optional
    log : callable
        Progress lines.

    Returns
    -------
    dict
        The summary written to ``export_summary.json``.
    """
    options = options or ExportOptions()
    if options.kernel not in ("exact", "matched", "exporter"):
        raise ValueError(f"unknown kernel {options.kernel!r}")
    if options.interp not in INTERP_ORDER:
        raise ValueError(f"unknown interpolation {options.interp!r}")
    run = SiloRun(run_dir)
    session = Session(
        table or find_table(Path(run_dir)), moco_dir, baseline_dir, data_dir
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = options.label or Path(run_dir).name
    prov = {
        "label": label,
        "silo_run": str(Path(run_dir)),
        "baseline_source": str(run.param("baselineSource", "median")),
    }
    summary = {
        "label": label,
        "giantRef": options.giant_ref,
        "exporter": EXPORTER,
        "exporterVersion": __version__,
        "options": asdict(options),
        "trials": [],
    }
    jobs = []
    for d in options.paths or run.paths:
        jobs += _path_jobs(run, session, d, out_dir, options, prov)
    for q in _run_jobs(jobs, options.workers):
        summary["trials"].append(q)
        write_summary(out_dir, summary)
        log(_line(label, q))
    summary["trials"].sort(key=lambda q: (q["path"], q["trial"]))
    write_summary(out_dir, summary)
    return summary


def _run_jobs(jobs: list, workers: int):
    """Each job's record, in order of completion with several workers."""
    if workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            yield export_trial(job)
        return
    ctx = get_context("spawn")
    with ProcessPoolExecutor(min(workers, len(jobs)), mp_context=ctx) as ex:
        yield from ex.map(export_trial, jobs)


def _line(label: str, q: dict) -> str:
    """One progress line for a trial record."""
    head = f"[export-f0] {label} path {q['path']} trial {q['trial']}: "
    if "error" in q:
        return head + q["error"]
    return (
        f"{head}{q['frames']} frames, {q['nSources']} sources, peak r "
        f"{q['peakR']:.2f}, disk slope {q['diskSlope']:.2f}, event "
        f"residual {q['eventResidualFraction']:.2f}, ring core "
        f"{q['eventRatioCore']:.2f} ({q['seconds']} s)"
    )


def find_table(run_dir: Path) -> Path:
    """The ``trial_table.h5`` in or above a run folder."""
    for d in (run_dir, *run_dir.parents):
        if (d / "trial_table.h5").exists():
            return d / "trial_table.h5"
    raise FileNotFoundError(
        f"no trial_table.h5 in or above {run_dir}: give --table"
    )


# ------------------------------------------------------------------ CLI


def add_arguments(p) -> None:
    """``giant export-f0``'s arguments."""
    p.add_argument("run", help="the giant silo run folder")
    p.add_argument("out", help="output folder (Path<k>/trial_<NNN>.h5)")
    p.add_argument(
        "--table",
        default=None,
        help="the session's trial_table.h5 (default: in or above the run)",
    )
    p.add_argument(
        "--moco-dir",
        default=None,
        help="registered movies and alignment files (default: the "
        "table's savedr/motion_correction)",
    )
    p.add_argument(
        "--baseline-dir",
        default=None,
        help="F0 network baselines (<stem>_BASELINE-80Hz.h5): provenance "
        "and the target-over-baseline QC",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help="raw .dat/.meta folder, for the lines per cycle (default: "
        "the table's datadr; without it the kernels reach 3 dt)",
    )
    p.add_argument(
        "--kernel",
        choices=("exact", "matched", "exporter"),
        default="exact",
        help="200 Hz traces to 80 Hz frames: exact (default; r x200 + "
        "(1 - r) exporter's, r = (dt200/dt80)^2), matched "
        "(sqrt(dt80^2 - dt200^2)) or exporter (the MATLAB exporter's dt80, "
        "smooths fast events twice)",
    )
    p.add_argument(
        "--blur",
        type=float,
        default=0.0,
        help="Gaussian blur (px) of the footprints (default 0)",
    )
    p.add_argument(
        "--interp",
        choices=tuple(INTERP_ORDER),
        default="linear",
        help="how footprints move into the trial: linear "
        "(default) or cubic",
    )
    p.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=None,
        help="one-based channels to write (default: SILo's "
        "activity channel)",
    )
    p.add_argument(
        "--var-fac",
        choices=("mean", "full"),
        default="mean",
        help="varFacDS: per-pixel mean (default) or every frame",
    )
    p.add_argument(
        "--deflate", type=int, default=1, help="HDF5 gzip level (default 1)"
    )
    p.add_argument(
        "--label",
        default="",
        help="session label (default: the run folder's name)",
    )
    p.add_argument(
        "--giant-ref",
        default="",
        help="GIAnT version or commit of the run (metadata)",
    )
    p.add_argument(
        "--paths",
        type=int,
        nargs="+",
        default=None,
        help="one-based paths (default all)",
    )
    p.add_argument(
        "--trials",
        type=int,
        nargs="+",
        default=None,
        help="one-based trials (default: all SILo analysed)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="trials exported at once (default 1)",
    )
    p.add_argument(
        "--ignore-sources",
        action="store_true",
        help="export with zero activity (raw residual checks)",
    )


def run_command(args) -> int:
    """Handle ``giant export-f0``; nonzero if any trial failed."""
    options = ExportOptions(
        kernel=args.kernel,
        blur=args.blur,
        interp=args.interp,
        channels=tuple(args.channels) if args.channels else None,
        var_fac=args.var_fac,
        deflate=args.deflate,
        label=args.label,
        giant_ref=args.giant_ref,
        paths=tuple(args.paths) if args.paths else None,
        trials=tuple(args.trials) if args.trials else None,
        workers=args.workers,
        ignore_sources=args.ignore_sources,
    )
    summary = export_run(
        args.run,
        args.out,
        args.table,
        args.moco_dir,
        args.baseline_dir,
        args.data_dir,
        options,
    )
    return int(any("error" in q for q in summary["trials"]))
