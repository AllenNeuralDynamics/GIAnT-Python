"""BandSILo driver: compose Phases 1-7 and write experiment_summary.h5.

Ported from ``main()`` in ``extractSLAP2IntegrationSources.py`` (ref
1360-3108). :func:`extract_band_sources` runs the whole band-scan
source-extraction pipeline for one ``trial_table.h5``: load inputs
and geometry, read low-res trial data, estimate background/noise/rho, build the
activity image and detect peaks, localize sources with NMF, extract high-res
traces, assemble dF/F, and write the byte-compatible ``experiment_summary.h5``
before returning a populated :class:`~giant_python.models.ExperimentSummary`.

The orchestration driver itself touches ``slap2_utils`` reads, torch NMF, and
disk IO end-to-end, so it cannot be exercised without a full dataset and is
marked ``# pragma: no cover`` (its constituent Phase 1-7 kernels are each
unit-tested and cross-checked against the reference). The output assembly and
HDF5 writer are pure and fully tested here, since the on-disk schema is a
frozen contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import h5py
import numpy as np
import torch

from ..models.experiment import (
    ExperimentSummary,
    Source,
    UserRoi,
    Visualizations,
)
from ..models.params import SiloParams
from ..parallel import map_trials
from . import background as bg
from . import geometry as geo
from . import summary_images as si
from . import trial_data as td
from .annotate import (
    build_user_roi_geometry,
    resolve_interactivity,
    resolve_user_rois,
)
from .baseline import assemble_dff
from .hdf5 import (
    compute_keep_trials,
    load_trial_table,
    read_align_info,
    to_serializable,
    write_dict_to_h5group,
)
from .nmf import fit_sources
from .peaks import get_act_im_peaks
from .progress import log, progress
from .traces import get_high_res_traces

_FRAME_INFO_KEYS = (
    "trial_num_frames",
    "discard_frames",
    "frame_line_idxs",
    "offlineXshifts",
    "offlineYshifts",
    "offlineZshifts",
    "onlineXshifts",
    "onlineYshifts",
    "onlineZshifts",
)


@dataclass
class PathResult:
    """Per-DMD (``Path{N}``) results feeding the output writer.

    Bundles the localized sources, per-trial trace results, and summary images
    for one imaging path, plus the geometry and windows needed to assemble the
    ``experiment_summary.h5`` datasets.

    Attributes
    ----------
    a : torch.Tensor
        Spatial profiles ``(n_pixels, n_sources)``.
    source_params : torch.Tensor
        Source parameters ``(n_sources, 6)``.
    source_snr : ndarray
        Per-source low-resolution SNR ``(n_sources,)``.
    act_im, mean_im, act_im_peaks : ndarray
        Activity image, mean image, and raw activity-image peaks.
    trace_results : list
        Per-trial 8-tuples from
        :func:`giant_python.bandsilo.traces.get_high_res_traces`.
    z_depths : ndarray
        Per-fast-Z reference-stack depth (``Path{N}/Z_depths``).
    num_fast_zs, dmd_pixels_per_column, dmd_pixels_per_row, num_channels : int
        Geometry / channel count.
    denoise_window, baseline_window : int
        Windows passed to :func:`giant_python.bandsilo.baseline.compute_f0`.
    draw_user_rois : bool
        Whether user ROIs were drawn (gates writing the user-ROI group).
    soma_masks : list or None
        Per-ROI boolean masks (``(fastz, rows, cols)`` each).
    soma_labels : list or None
        Per-ROI text labels.
    yx_shape : tuple
        ``(rows, cols)`` plane shape (for the empty-ROI mask shape).
    """

    a: torch.Tensor
    source_params: torch.Tensor
    source_snr: np.ndarray
    act_im: np.ndarray
    mean_im: np.ndarray
    act_im_peaks: np.ndarray
    trace_results: list
    z_depths: np.ndarray
    num_fast_zs: int
    dmd_pixels_per_column: int
    dmd_pixels_per_row: int
    num_channels: int
    denoise_window: int
    baseline_window: int
    draw_user_rois: bool = False
    soma_masks: Optional[list] = None
    soma_labels: Optional[list] = None
    yx_shape: tuple = (0, 0)


def _with_nan_channel(arr: np.ndarray, num_channels: int) -> np.ndarray:
    """Reshape ``(frames, sources)`` to ``(sources, channels, frames)``.

    Only the activity channel (index 0) is filled; the remaining channels are
    NaN (reserved until multi-channel source extraction exists).

    Parameters
    ----------
    arr : ndarray of shape (n_frames, n_sources)
        Per-source trace.
    num_channels : int
        Number of acquisition channels.

    Returns
    -------
    ndarray of shape (n_sources, num_channels, n_frames)
        The reshaped trace with NaN in the non-activity channels.
    """
    out = np.full(
        (arr.shape[1], num_channels, arr.shape[0]), np.nan, dtype=arr.dtype
    )
    out[:, 0, :] = arr.T
    return out


def _assemble_user_rois(pr: PathResult) -> Optional[dict]:
    """Assemble the ``user_rois`` datasets for one path, or ``None``.

    Parameters
    ----------
    pr : PathResult
        The path's results (uses ``draw_user_rois``/``soma_masks``/
        ``soma_labels`` and the per-trial ``F_soma`` from ``trace_results``).

    Returns
    -------
    dict or None
        ``{"mask", "labels", "F"}`` when user ROIs were drawn, else ``None``.
    """
    if not pr.draw_user_rois or pr.soma_masks is None:
        return None
    if len(pr.soma_masks) > 0:
        masks_by_roi = np.stack(
            [np.asarray(m, dtype=bool) for m in pr.soma_masks], axis=0
        )
    else:
        masks_by_roi = np.zeros((0, pr.num_fast_zs, *pr.yx_shape), dtype=bool)
    f_soma_all = np.concatenate(
        [r[7] for r in pr.trace_results], axis=0
    ).transpose(1, 2, 0)
    return {"mask": masks_by_roi, "labels": pr.soma_labels, "F": f_soma_all}


def _dmd_user_rois(user_rois: Optional[dict], key: str) -> tuple:
    """Return ``(soma_masks, soma_labels, soma_sps)`` for one DMD path.

    Slices a resolved session selection (from
    :func:`giant_python.bandsilo.annotate.resolve_user_rois`) down to one DMD,
    or the neutral ``(None, None, [])`` when user ROIs are disabled.

    Parameters
    ----------
    user_rois : dict or None
        The resolved session selection, or ``None`` when ``draw_user_rois``
        is unset.
    key : str
        The DMD key (``"DMD{N}"``).

    Returns
    -------
    tuple
        ``(soma_masks, soma_labels, soma_sps)`` -- the per-ROI masks and labels
        (or ``None``) and the per-ROI superpixel index lists (``[]`` when
        absent).
    """
    if user_rois is None:
        return None, None, []
    return (
        user_rois["user_roi_masks"].get(key),
        user_rois["user_roi_labels"].get(key),
        user_rois["user_roi_superpixels"].get(key, []),
    )


def assemble_path_outputs(pr: PathResult) -> dict:
    """Assemble one path's HDF5-ready datasets from its results.

    Concatenates the per-trial traces, re-estimates ``F0`` and forms the
    per-source dF/F, and shapes every ``Path{N}`` dataset (sources, frame info,
    global fluorescence, visualizations, optional user ROIs).

    Parameters
    ----------
    pr : PathResult
        The path's results.

    Returns
    -------
    dict
        Keys mirror the ``experiment_summary.h5`` ``Path{N}`` datasets:
        ``z_depths``, ``profiles``, ``coords``, ``dF_ls``, ``F0``, ``SNR``,
        the frame-info arrays, ``global_f``, ``act_im``, ``mean_im``,
        ``act_im_peaks``, and ``user_rois`` (or ``None``).
    """
    results = pr.trace_results
    d_f = np.concatenate([r[0] for r in results], axis=0)
    f0_ls = np.concatenate([r[1] for r in results], axis=0)
    f_full, f0, d_f, _ = assemble_dff(
        d_f, f0_ls, pr.denoise_window, pr.baseline_window
    )

    profiles = pr.a.numpy().T.reshape(
        -1, pr.num_fast_zs, pr.dmd_pixels_per_column, pr.dmd_pixels_per_row
    )

    return {
        "z_depths": pr.z_depths,
        "profiles": profiles,
        "coords": pr.source_params[:, :3].numpy(),
        "dF_ls": _with_nan_channel(d_f, pr.num_channels),
        "F0": _with_nan_channel(f0, pr.num_channels),
        "SNR": pr.source_snr.reshape(-1, 1),
        "trial_num_frames": np.concatenate(
            [[len(r[2])] for r in results]
        ).reshape(-1, 1),
        "discard_frames": np.any(np.isnan(f_full), axis=1).reshape(-1, 1),
        "frame_line_idxs": np.concatenate(
            [r[2] for r in results], axis=0
        ).reshape(-1, 1),
        "offlineXshifts": np.concatenate(
            [r[5][1] for r in results], axis=0
        ).reshape(-1, 1),
        "offlineYshifts": np.concatenate(
            [r[5][0] for r in results], axis=0
        ).reshape(-1, 1),
        "offlineZshifts": np.concatenate(
            [r[5][2] for r in results], axis=0
        ).reshape(-1, 1),
        "onlineXshifts": np.concatenate(
            [r[6][1] for r in results], axis=0
        ).reshape(-1, 1),
        "onlineYshifts": np.concatenate(
            [r[6][0] for r in results], axis=0
        ).reshape(-1, 1),
        "onlineZshifts": np.concatenate(
            [r[6][2] for r in results], axis=0
        ).reshape(-1, 1),
        "global_f": np.concatenate([r[4] for r in results], axis=0).T,
        "act_im": pr.act_im,
        "mean_im": pr.mean_im,
        "act_im_peaks": pr.act_im_peaks,
        "user_rois": _assemble_user_rois(pr),
    }


def _write_path_group(f: h5py.File, dmd_ix: int, assembled: dict) -> None:
    """Write one ``Path{N}`` group from assembled outputs (overwriting)."""
    group_name = f"Path{dmd_ix + 1}"
    if group_name in f:
        del f[group_name]
    dmd_group = f.create_group(group_name)
    dmd_group.create_dataset("Z_depths", data=assembled["z_depths"])

    sources = dmd_group.create_group("sources")
    spatial = sources.create_group("spatial")
    spatial.create_dataset("profiles", data=assembled["profiles"])
    spatial.create_dataset("coords", data=assembled["coords"])
    temporal = sources.create_group("temporal")
    temporal.create_dataset("dF_ls", data=assembled["dF_ls"])
    temporal.create_dataset("F0", data=assembled["F0"])
    temporal.create_dataset("SNR", data=assembled["SNR"])

    frame_group = dmd_group.create_group("frame_info")
    for key in _FRAME_INFO_KEYS:
        frame_group.create_dataset(key, data=assembled[key])

    global_group = dmd_group.create_group("global")
    global_group.create_dataset("F", data=assembled["global_f"])

    vis = dmd_group.create_group("visualizations")
    vis.create_dataset("act_im", data=assembled["act_im"])
    vis.create_dataset("mean_im", data=assembled["mean_im"])
    vis.create_dataset("act_im_peaks", data=assembled["act_im_peaks"])

    if assembled["user_rois"] is not None:
        rois = assembled["user_rois"]
        user_roi_group = dmd_group.create_group("user_rois")
        user_roi_group.create_dataset("mask", data=rois["mask"])
        user_roi_group.create_dataset(
            "labels",
            data=np.array(rois["labels"], dtype=object).reshape(-1, 1),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        user_roi_group.create_dataset("F", data=rois["F"])


def write_experiment_summary(
    output_path: Union[str, Path],
    params_serializable: dict,
    assembled_paths: List[dict],
) -> None:
    """Write the byte-compatible ``experiment_summary.h5``.

    Writes the root ``row_major`` flag and ``params`` group once, then one
    ``Path{N}`` group per DMD (overwriting any existing group of that name).

    Parameters
    ----------
    output_path : str or Path
        Destination ``experiment_summary.h5`` path.
    params_serializable : dict
        The analysis parameters (already run through
        :func:`giant_python.bandsilo.hdf5.to_serializable`).
    assembled_paths : list of dict
        Per-path outputs from :func:`assemble_path_outputs`.
    """
    with h5py.File(output_path, "a") as f:
        if "row_major" not in f:
            f["row_major"] = 1
        if "params" not in f:
            write_dict_to_h5group(
                f.create_group("params"), params_serializable
            )
        for dmd_ix, assembled in enumerate(assembled_paths):
            _write_path_group(f, dmd_ix, assembled)


def _sources_from_assembled(assembled: dict) -> List[Source]:
    """Build the per-source :class:`Source` list for one path."""
    profiles = assembled["profiles"]
    coords = assembled["coords"]
    d_f = assembled["dF_ls"]
    f0 = assembled["F0"]
    snr = assembled["SNR"]
    return [
        Source(
            profile=profiles[j],
            coords=coords[j],
            df_ls=d_f[j],
            f0=f0[j],
            snr=float(snr[j, 0]),
        )
        for j in range(profiles.shape[0])
    ]


def _user_rois_from_assembled(assembled: dict) -> List[UserRoi]:
    """Build the per-ROI :class:`UserRoi` list for one path."""
    rois = assembled["user_rois"]
    if rois is None:
        return []
    labels = rois["labels"]
    masks = rois["mask"]
    f_soma = rois["F"]
    return [
        UserRoi(
            label=str(labels[k]),
            mask=masks[k] if k < masks.shape[0] else None,
            f=f_soma[k] if k < f_soma.shape[0] else None,
        )
        for k in range(len(labels))
    ]


def build_experiment_summary(
    params_serializable: dict, assembled_paths: List[dict]
) -> ExperimentSummary:
    """Build the returned :class:`ExperimentSummary` from assembled outputs.

    Populates per-path sources and user ROIs; the single-view visualization /
    z-depth / global-F / frame-info fields hold the last path's values (the
    per-path structure lives in ``experiment_summary.h5``).

    Parameters
    ----------
    params_serializable : dict
        The analysis parameters.
    assembled_paths : list of dict
        Per-path outputs from :func:`assemble_path_outputs`.

    Returns
    -------
    ExperimentSummary
        The populated summary.
    """
    summary = ExperimentSummary(params=params_serializable)
    for assembled in assembled_paths:
        summary.sources.append(_sources_from_assembled(assembled))
        summary.user_rois.append(_user_rois_from_assembled(assembled))
        summary.visualizations = Visualizations(
            mean_im=assembled["mean_im"],
            act_im=assembled["act_im"],
            act_im_peaks=assembled["act_im_peaks"],
        )
        summary.z_depths = assembled["z_depths"]
        summary.global_f = assembled["global_f"]
        summary.frame_info = {k: assembled[k] for k in _FRAME_INFO_KEYS}
    return summary


def _resolve_params(
    params_in,
) -> SiloParams:  # pragma: no cover - trivial glue
    """Coerce ``params_in`` (None / dict / SiloParams) to a SiloParams."""
    if params_in is None:
        return SiloParams(scan_mode="band")
    if isinstance(params_in, SiloParams):
        return params_in
    return SiloParams(**params_in)


def _resolve_session_user_rois(
    result_dr: str,
    trial_table: dict,
    lookup: dict,
    params: SiloParams,
) -> Optional[
    dict
]:  # pragma: no cover - reference-stack/aData IO + optional GUI
    """Resolve per-DMD user ROIs for the session, or ``None`` if disabled.

    When ``draw_user_rois`` is set, builds the per-DMD annotation geometry and
    loads ``<result_dr>/annotations/annotations.h5`` (drawing it interactively,
    or failing fast when headless, via
    :func:`giant_python.bandsilo.annotate.resolve_user_rois`). Extraction
    stays non-interactive: drawing only happens as the resolver's fallback.

    Parameters
    ----------
    result_dr : str
        Session results directory (holds the ``annotations`` subdir).
    trial_table : dict
        Normalized trial table.
    lookup : dict
        Band-registration lookup table.
    params : SiloParams
        Extraction parameters (``draw_user_rois``/``interactive``).

    Returns
    -------
    dict or None
        The resolved user-ROI selection, or ``None`` when ``draw_user_rois``
        is unset.
    """
    if not params.draw_user_rois:
        return None
    user_roi_geo, ref_files = build_user_roi_geometry(trial_table, lookup)
    return resolve_user_rois(
        Path(result_dr) / "annotations",
        trial_table["n_dmds"],
        user_roi_geo,
        interactive=resolve_interactivity(params.interactive),
        ref_files=ref_files,
    )


def extract_band_sources(
    path_to_trial_table: Union[str, Path],
    params_in=None,
) -> ExperimentSummary:  # pragma: no cover - full IO/slap2/torch orchestration
    """Run the band source-extraction pipeline for one trial table.

    Loads the ``trial_table.h5`` and its geometry, then for each DMD path runs
    the low-res read, background/rho estimation, activity-image peak detection,
    NMF localization, and high-res trace extraction, writes
    ``experiment_summary.h5``, and returns a populated
    :class:`~giant_python.models.ExperimentSummary`.

    This driver reads SLAP2 files (``slap2_utils``), runs the torch NMF, and
    performs disk IO, so it is not unit-tested (its Phase 1-7 kernels are).

    Parameters
    ----------
    path_to_trial_table : str or Path
        Path to the ``trial_table.h5`` produced by BandRegistration.
    params_in : SiloParams or dict, optional
        Parameter overrides.

    Returns
    -------
    ExperimentSummary
        The extracted sources and summary (also written to disk).
    """
    params = _resolve_params(params_in)
    log(f"Loading trial table {path_to_trial_table}", params.verbose)
    trial_table = load_trial_table(path_to_trial_table)
    result_dr = str(trial_table["savedr"])
    src_extr_dr = Path(result_dr) / "source_extraction"
    src_extr_dr.mkdir(parents=True, exist_ok=True)
    output_path = src_extr_dr / "experiment_summary.h5"

    lookup = geo.load_lookup_table(
        Path(result_dr) / "motion_correction" / "bandRegLookupTable.h5",
        trial_table["n_dmds"],
    )
    psf = geo.load_psf(params.psf_dilation, trial_table["n_dmds"])

    trial_table["keep_trials"] = compute_keep_trials(
        trial_table["fn_adata"], trial_table["filename"], trial_table["datadr"]
    )
    align_hz, num_channels = read_align_info(
        trial_table["fn_adata"],
        trial_table["keep_trials"],
        trial_table["n_dmds"],
    )
    trial_table["align_hz"] = align_hz
    if params.num_channels is None:
        params.num_channels = num_channels
    if params.num_channels is None:
        raise ValueError(
            "num_channels could not be determined from the alignment data "
            "(no kept trials?); set SiloParams.num_channels explicitly."
        )

    user_rois = _resolve_session_user_rois(
        result_dr, trial_table, lookup, params
    )

    n_dmds = trial_table["n_dmds"]
    log(f"Extracting sources for {n_dmds} DMD path(s)", params.verbose)
    assembled_paths = []
    for dmd_ix in range(n_dmds):
        assembled_paths.append(
            assemble_path_outputs(
                _process_dmd(
                    dmd_ix, trial_table, params, lookup, psf, user_rois
                )
            )
        )

    params_serializable = to_serializable(_params_dict(params))
    log(f"Writing {output_path}", params.verbose)
    write_experiment_summary(output_path, params_serializable, assembled_paths)
    return build_experiment_summary(params_serializable, assembled_paths)


def _params_dict(params: SiloParams) -> dict:  # pragma: no cover - glue
    """Return the on-disk params dict written to ``experiment_summary.h5``."""
    return {
        "numChannels": params.num_channels,
        "analyzeHz": params.analyze_hz,
        "denoiseWindow_s": params.denoise_window_s,
        "baselineWindow_s": params.baseline_window_s,
        "decayTau_s": params.decay_tau_s,
        "dXY": params.d_xy,
        "sparse_fac": params.sparse_fac,
        "vif": params.vif,
        "peakth": params.peakth,
        "peak_buffer": params.peak_buffer,
        "draw_user_rois": params.draw_user_rois,
        "operator": params.operator,
    }


def _process_dmd(
    dmd_ix: int,
    trial_table: dict,
    params: SiloParams,
    lookup: dict,
    psf: dict,
    user_rois: Optional[dict] = None,
) -> PathResult:  # pragma: no cover - full per-DMD IO/slap2/torch compute
    """Run Phases 3-7 for one DMD path and bundle the results.

    Reads the low-res trial data, estimates background/noise/rho, builds the
    activity image and detects peaks, localizes sources with NMF, and extracts
    high-res traces per trial (fanned out via
    :func:`giant_python.parallel.map_trials`). When ``user_rois`` is provided,
    this DMD's ROI masks/labels are attached and its per-ROI superpixels feed
    the per-trial ``F_soma``.
    """
    key = f"DMD{dmd_ix + 1}"
    log(f"Processing {key}", params.verbose)
    soma_masks, soma_labels, soma_sps = _dmd_user_rois(user_rois, key)
    datadr = str(trial_table["datadr"])
    n_trials = trial_table["filename"].shape[1]
    if params.max_trials is not None:
        n_trials = min(n_trials, params.max_trials)
        log(f"Debug: limiting to first {n_trials} trial(s)", params.verbose)
    num_channels = params.num_channels
    align_hz = trial_table["align_hz"][key]
    keep_trials = trial_table["keep_trials"]

    ref_stack, _, _ = geo.load_reference_stack(
        datadr, trial_table["ref_stack"], dmd_ix
    )
    dmd_pixels_per_column = ref_stack.shape[2]
    dmd_pixels_per_row = ref_stack.shape[3]
    yx_shape = (dmd_pixels_per_column, dmd_pixels_per_row)
    fastz_to_refz = lookup["fastZ2RefZ"][key]
    num_fast_zs = fastz_to_refz.shape[0]
    all_sp_ids = lookup["allSuperPixelIDs"][key]
    num_super_pixels = all_sp_ids.shape[0]
    n_pixels = num_fast_zs * dmd_pixels_per_column * dmd_pixels_per_row

    smi = geo.build_subsample_matrix_inds(
        all_sp_ids, lookup["sparseMaskInds"][key]
    )
    psf2d = psf[key]
    sparse_h_inds, sparse_h_vals = geo.build_sparse_h(
        smi, psf2d, dmd_pixels_per_column, dmd_pixels_per_row
    )
    ref_d, ref_c, ref_r = geo.ref_pixs_to_drc(
        smi[:, 0], dmd_pixels_per_column, dmd_pixels_per_row
    )

    def _read(trial_ix):
        """Read one trial's low-res data (per-trial worker)."""
        return td.read_band_trial_data(
            trial_ix,
            bool(keep_trials[dmd_ix, trial_ix]),
            dmd_ix,
            align_hz,
            all_sp_ids,
            datadr,
            trial_table,
            num_channels,
        )

    low = td.assemble_lowres_data(
        map_trials(
            _read,
            range(n_trials),
            params.max_workers,
            desc=f"{key} reading trials" if params.verbose else None,
        ),
        num_channels,
    )
    low_res_data_norm = low["lowResData"] / low["lowResDataCt"]
    v_im = 1.0 / low["lowResDataCt"]
    low2 = (
        low["lowResData2"] / low["lowResDataCt2"]
        if num_channels >= 2
        else None
    )

    unique_motion, mot_inds = bg.bin_motion(
        low["lowResMotionR"], low["lowResMotionC"], low["lowResMotionZ"]
    )
    median_z = np.median(low["lowResMotionZ"])
    _, frames_to_keep = bg.select_motion_bins(
        unique_motion, mot_inds, low["lowResMotionZ"]
    )
    mean_im = si.compute_mean_image(
        low_res_data_norm,
        unique_motion,
        mot_inds,
        frames_to_keep,
        ref_d,
        ref_r,
        ref_c,
        num_fast_zs,
        dmd_pixels_per_column,
        dmd_pixels_per_row,
        num_channels,
        low2,
    )
    umyx, mot_inds_yx = bg.bin_motion_yx(
        low["lowResMotionR"], low["lowResMotionC"], frames_to_keep
    )
    _, sel_pix_idxs = bg.build_selected_pixel_mask(
        umyx,
        ref_d,
        ref_r,
        ref_c,
        num_fast_zs,
        dmd_pixels_per_column,
        dmd_pixels_per_row,
        psf2d,
    )
    pixel_coords = bg.pixel_coords_from_idxs(
        sel_pix_idxs, dmd_pixels_per_column, dmd_pixels_per_row
    )

    background = _estimate_background(
        low_res_data_norm,
        umyx,
        mot_inds_yx,
        sel_pix_idxs,
        ref_d,
        ref_r,
        ref_c,
        num_fast_zs,
        dmd_pixels_per_column,
        dmd_pixels_per_row,
        psf2d,
        num_super_pixels,
        align_hz,
        params,
    )
    data_std, _, _ = bg.fit_noise_variance_model(
        low_res_data_norm, background, v_im, params.vif
    )
    residual = bg.compute_residual(low_res_data_norm, background, data_std)

    act_im, act_im_peaks, a, source_params, source_snr = _localize(
        residual,
        umyx,
        mot_inds_yx,
        sel_pix_idxs,
        pixel_coords,
        n_pixels,
        ref_d,
        ref_r,
        ref_c,
        num_fast_zs,
        dmd_pixels_per_column,
        dmd_pixels_per_row,
        psf2d,
        num_super_pixels,
        sparse_h_inds,
        sparse_h_vals,
        align_hz,
        params,
    )

    def _traces(trial_ix):
        """Extract one trial's high-res traces (per-trial worker)."""
        return get_high_res_traces(
            (
                trial_ix,
                bool(keep_trials[dmd_ix, trial_ix]),
                background[:, low["lowResTrialID"] == trial_ix],
            ),
            dmd_ix,
            params.analyze_hz,
            all_sp_ids,
            datadr,
            trial_table,
            smi,
            sparse_h_inds,
            sparse_h_vals,
            a,
            umyx,
            np.arange(umyx.shape[0]),
            median_z,
            psf2d,
            num_super_pixels,
            num_fast_zs,
            dmd_pixels_per_column,
            dmd_pixels_per_row,
            num_channels,
            soma_sps,
        )

    trace_results = map_trials(
        _traces,
        range(n_trials),
        params.max_workers,
        desc=f"{key} extracting traces" if params.verbose else None,
    )
    trace_results = [r for r in trace_results if r[0].shape[0] > 0]

    return PathResult(
        a=a,
        source_params=source_params,
        source_snr=source_snr,
        act_im=act_im,
        mean_im=mean_im,
        act_im_peaks=act_im_peaks,
        trace_results=trace_results,
        z_depths=fastz_to_refz,
        num_fast_zs=num_fast_zs,
        dmd_pixels_per_column=dmd_pixels_per_column,
        dmd_pixels_per_row=dmd_pixels_per_row,
        num_channels=num_channels,
        denoise_window=int(
            np.ceil(params.denoise_window_s * params.analyze_hz)
        ),
        baseline_window=int(
            np.ceil(params.baseline_window_s * params.analyze_hz)
        ),
        draw_user_rois=params.draw_user_rois,
        soma_masks=soma_masks,
        soma_labels=soma_labels,
        yx_shape=yx_shape,
    )


def _estimate_background(
    low_res_data_norm,
    umyx,
    mot_inds_yx,
    sel_pix_idxs,
    ref_d,
    ref_r,
    ref_c,
    num_fast_zs,
    dmd_pixels_per_column,
    dmd_pixels_per_row,
    psf2d,
    num_super_pixels,
    align_hz,
    params,
):  # pragma: no cover - Phase-4 background composition over real data
    """Interpolated + rolling-baseline background on the superpixel grid."""
    log("Estimating background", params.verbose)
    psf_tensor, psf_center, psf_exp, psf_center_exp = bg.expand_psf(psf2d)
    interp_data = np.full(
        (sel_pix_idxs.shape[0], low_res_data_norm.shape[1]),
        np.nan,
        dtype=np.float32,
    )
    for z in progress(
        range(num_fast_zs), desc="Interpolating planes", verbose=params.verbose
    ):
        z_idxs, sel_2d = bg.selected_pixels_2d_for_plane(
            sel_pix_idxs, z, dmd_pixels_per_column, dmd_pixels_per_row
        )
        if z_idxs.size == 0:
            continue
        ref_z = np.flatnonzero(ref_d == z)
        interp_data[z_idxs], _, _ = bg.build_interp_data(
            low_res_data_norm[ref_z],
            ref_r[ref_z],
            ref_c[ref_z],
            sel_2d,
            umyx,
            mot_inds_yx,
        )
    baseline_window = bg.baseline_window_frames(
        align_hz, params.baseline_window_s
    )
    interp_background = bg.compute_rolling_baseline(
        interp_data, baseline_window
    )
    return bg.assemble_background(
        interp_background,
        umyx,
        mot_inds_yx,
        sel_pix_idxs,
        ref_d,
        ref_r,
        ref_c,
        low_res_data_norm.shape[0],
        low_res_data_norm.shape[1],
        dmd_pixels_per_column,
        dmd_pixels_per_row,
    )


def _localize(
    residual,
    umyx,
    mot_inds_yx,
    sel_pix_idxs,
    pixel_coords,
    n_pixels,
    ref_d,
    ref_r,
    ref_c,
    num_fast_zs,
    dmd_pixels_per_column,
    dmd_pixels_per_row,
    psf2d,
    num_super_pixels,
    sparse_h_inds,
    sparse_h_vals,
    align_hz,
    params,
):  # pragma: no cover - Phase 4-6 rho/activity/NMF over real data
    """Compute rho + activity image, detect peaks, and NMF-fit sources."""
    log("Computing activity image and localizing sources", params.verbose)
    h_mots = bg.build_motion_h_matrices(
        sparse_h_inds,
        sparse_h_vals,
        umyx,
        sel_pix_idxs,
        num_super_pixels,
        dmd_pixels_per_row,
    )
    # TEMPORARY: build the activity-image D matrices from an isotropic
    # difference-of-Gaussians instead of the PSF / expanded-PSF pair.
    # ``compute_rho`` column-normalizes each and subtracts them, so the
    # effective filter is H @ (G_center - G_surround). ``psf2d`` is still used
    # there for the valid-column dilation/erosion mask.
    #
    # To revert, delete the Gaussian kernels below, restore these two lines
    # (the ``shrink_psf`` call was itself a temporary hack and is optional),
    # and swap the commented-out ``build_convolution_matrix`` calls back in
    # inside the loop:
    # psf2d = bg.shrink_psf(psf2d, scale_y=0.9, scale_x=0.75)
    # psf_tensor, psf_center, psf_exp, psf_center_exp = bg.expand_psf(psf2d)
    dog_center_sd = 0.9
    dog_surround_sd = 4.5
    center_kernel, center_kernel_ctr = bg.gaussian_kernel_2d(dog_center_sd)
    surround_kernel, surround_kernel_ctr = bg.gaussian_kernel_2d(
        dog_surround_sd
    )
    d_mats, d_mats_exp = [], []
    for z in progress(
        range(num_fast_zs),
        desc="Computing D matrices",
        verbose=params.verbose,
    ):
        _, sel_2d = bg.selected_pixels_2d_for_plane(
            sel_pix_idxs, z, dmd_pixels_per_column, dmd_pixels_per_row
        )
        # d_mats.append(
        #     bg.build_convolution_matrix(sel_2d, psf_tensor, psf_center)
        # )
        # d_mats_exp.append(
        #     bg.build_convolution_matrix(sel_2d, psf_exp, psf_center_exp)
        # )
        d_mats.append(
            bg.build_convolution_matrix(
                sel_2d, center_kernel, center_kernel_ctr
            )
        )
        d_mats_exp.append(
            bg.build_convolution_matrix(
                sel_2d, surround_kernel, surround_kernel_ctr
            )
        )
    rho = bg.compute_rho(
        residual,
        mot_inds_yx,
        umyx,
        h_mots,
        d_mats,
        d_mats_exp,
        sel_pix_idxs,
        ref_d,
        ref_r,
        ref_c,
        num_fast_zs,
        dmd_pixels_per_column,
        dmd_pixels_per_row,
        psf2d,
        verbose=params.verbose,
    )
    nan_ct = bg.mask_high_nan_rho(rho)
    rho = bg.smooth_rho(
        rho,
        bg.decay_kernel_1d(params.decay_tau_s, align_hz),
        verbose=params.verbose,
    )

    act_im = si.accumulate_activity_image(
        rho,
        sel_pix_idxs,
        num_fast_zs,
        dmd_pixels_per_column,
        dmd_pixels_per_row,
        verbose=params.verbose,
    )
    act_im = si.finalize_activity_image(act_im, sel_pix_idxs, nan_ct)

    peak_th = params.peakth
    source_seeds = get_act_im_peaks(
        act_im, peak_th=peak_th, buffer_size=params.peak_buffer
    )
    act_im_peaks = source_seeds.copy()

    result = fit_sources(
        source_seeds,
        residual,
        h_mots,
        umyx,
        mot_inds_yx,
        sel_pix_idxs,
        pixel_coords,
        n_pixels,
        params.d_xy,
        params.sparse_fac,
        verbose=params.verbose,
    )
    return (
        act_im,
        act_im_peaks,
        result["A"],
        result["source_params"],
        result["source_snr"],
    )
