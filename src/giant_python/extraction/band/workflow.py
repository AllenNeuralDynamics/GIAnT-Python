"""Canonical band extraction driver, from prepared inputs to eager results.

Numerical ordering follows the original BandSILo driver. Acquisition metadata,
execution policy and annotations are separate from scientific configuration;
result assembly and persistence belong to their canonical owners.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from ...io.experiment_summary import write_summary
from ...io.hdf5 import to_serializable
from ...models.experiment_summary import ExperimentSummary, PathSummary
from ...models.params import (
    AnnotationInput,
    AnnotationOptions,
    BandParamsInput,
    BandSiloParams,
    ExecutionInput,
    ExecutionOptions,
    resolve_band_options,
)
from ...models.trial_table import TrialTable
from ...numerics.peaks import get_act_im_peaks
from ...parallel import map_trials
from ...progress import log, progress
from . import activity
from . import background as bg
from . import geometry as geo
from . import inputs, motion_binning, operators
from . import summary_images as si
from . import trial_data as td
from .annotation import (
    build_user_roi_geometry,
    resolve_interactivity,
    resolve_user_rois,
)
from .inputs import compute_keep_trials, load_trial_table, read_align_info
from .localization import fit_sources
from .result_assembly import (
    PathResult,
    _dmd_user_rois,
    assemble_path_summary,
)
from .traces import get_high_res_traces
from .types import ResolvedAcquisition


def _resolve_session_user_rois(
    result_dr: str,
    trial_table: dict,
    lookup: dict,
    annotations: AnnotationOptions,
) -> Optional[dict]:
    """Load session annotations or invoke the configured drawing fallback."""
    if not annotations.enabled:
        return None
    user_roi_geo, ref_files = build_user_roi_geometry(trial_table, lookup)
    return resolve_user_rois(
        Path(result_dr) / "annotations",
        trial_table["n_dmds"],
        user_roi_geo,
        interactive=resolve_interactivity(annotations.interactive),
        ref_files=ref_files,
    )


def extract_band_sources(
    input: Union[str, Path, TrialTable],
    params: BandParamsInput = None,
    *,
    execution: ExecutionInput = None,
    annotations: AnnotationInput = None,
) -> ExperimentSummary:
    """Extract all paths from an exact filename or an already loaded table.

    Loaded models are prepared directly, without inferring or reopening a
    conventional filename. Channel inference is local; caller scientific and
    policy objects remain unchanged. Results are persisted under the table's
    savedr/source_extraction directory and returned with every path intact.
    """
    params, run, roi = resolve_band_options(params, execution, annotations)
    log(f"Loading trial table {input}", run.verbose)
    trial_table = load_trial_table(input)
    result_dr = str(trial_table["savedr"])
    src_extr_dr = Path(result_dr) / "source_extraction"
    src_extr_dr.mkdir(parents=True, exist_ok=True)
    output_path = src_extr_dr / "experiment_summary.h5"

    lookup = inputs.load_lookup_table(
        Path(result_dr) / "motion_correction" / "bandRegLookupTable.h5",
        trial_table["n_dmds"],
    )
    psf = inputs.load_psf(params.psf_dilation, trial_table["n_dmds"])
    trial_table["keep_trials"] = compute_keep_trials(
        trial_table["fn_adata"], trial_table["filename"], trial_table["datadr"]
    )
    align_hz, num_channels = read_align_info(
        trial_table["fn_adata"],
        trial_table["keep_trials"],
        trial_table["n_dmds"],
    )
    if num_channels is None:
        raise ValueError(
            "num_channels could not be determined from the alignment data "
            "(no kept trials?); provide acquisition metadata for a kept trial."
        )
    acquisition = ResolvedAcquisition(num_channels, align_hz)
    user_rois = _resolve_session_user_rois(result_dr, trial_table, lookup, roi)

    n_dmds = trial_table["n_dmds"]
    log(f"Extracting sources for {n_dmds} DMD path(s)", run.verbose)
    paths: list[PathSummary] = []
    for dmd_ix in range(n_dmds):
        paths.append(
            assemble_path_summary(
                _process_dmd(
                    dmd_ix,
                    trial_table,
                    params,
                    lookup,
                    psf,
                    user_rois,
                    acquisition=acquisition,
                    execution=run,
                    annotations=roi,
                ),
                path_index=dmd_ix,
            )
        )

    params_serializable = to_serializable(
        _params_dict(params, acquisition, roi)
    )
    log(f"Writing {output_path}", run.verbose)
    summary = ExperimentSummary(paths=paths, params=params_serializable)
    write_summary(summary, output_path)
    return summary


def _params_dict(
    params: BandSiloParams,
    acquisition: ResolvedAcquisition,
    annotations: AnnotationOptions,
) -> dict:
    """Historical persisted parameter names, with locally resolved metadata."""
    return {
        "numChannels": acquisition.num_channels,
        "analyzeHz": params.analyze_hz,
        "denoiseWindow_s": params.denoise_window_s,
        "baselineWindow_s": params.baseline_window_s,
        "backgroundInterpolation": "cubic",
        "decayTau_s": params.decay_tau_s,
        "dXY": params.d_xy,
        "sparse_fac": params.sparse_fac,
        "vif": params.vif,
        "peakth": params.peakth,
        "peak_buffer": params.peak_buffer,
        "draw_user_rois": annotations.enabled,
        "operator": annotations.operator,
    }


def _process_dmd(
    dmd_ix: int,
    trial_table: dict,
    params: BandSiloParams,
    lookup: dict,
    psf: dict,
    user_rois: Optional[dict] = None,
    *,
    acquisition: ResolvedAcquisition,
    execution: ExecutionOptions,
    annotations: AnnotationOptions,
) -> PathResult:
    """Read, localize and extract one path without changing numerical order."""
    key = f"DMD{dmd_ix + 1}"
    log(f"Processing {key}", execution.verbose)
    soma_masks, soma_labels, soma_sps = _dmd_user_rois(user_rois, key)
    datadr = str(trial_table["datadr"])
    n_trials = trial_table["filename"].shape[1]
    if execution.max_trials is not None:
        n_trials = min(n_trials, execution.max_trials)
        log(f"Debug: limiting to first {n_trials} trial(s)", execution.verbose)
    num_channels = acquisition.num_channels
    align_hz = acquisition.align_hz[key]
    keep_trials = trial_table["keep_trials"]

    ref_stack, _, _ = inputs.load_reference_stack(
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
    sparse_h_inds, sparse_h_vals = operators.build_sparse_h(
        smi, psf2d, dmd_pixels_per_column, dmd_pixels_per_row
    )
    ref_d, ref_c, ref_r = geo.ref_pixs_to_drc(
        smi[:, 0], dmd_pixels_per_column, dmd_pixels_per_row
    )

    def _read(trial_ix):
        """Read one low-resolution trial in acquisition order."""
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
            execution.max_workers,
            desc=f"{key} reading trials" if execution.verbose else None,
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
    unique_motion, mot_inds = motion_binning.bin_motion(
        low["lowResMotionR"], low["lowResMotionC"], low["lowResMotionZ"]
    )
    median_z = np.median(low["lowResMotionZ"])
    _, frames_to_keep = motion_binning.select_motion_bins(
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
    umyx, mot_inds_yx = motion_binning.bin_motion_yx(
        low["lowResMotionR"], low["lowResMotionC"], frames_to_keep
    )
    _, sel_pix_idxs = geo.build_selected_pixel_mask(
        umyx,
        ref_d,
        ref_r,
        ref_c,
        num_fast_zs,
        dmd_pixels_per_column,
        dmd_pixels_per_row,
        psf2d,
    )
    pixel_coords = geo.pixel_coords_from_idxs(
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
        execution=execution,
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
        execution=execution,
    )

    def _traces(trial_ix):
        """Extract high-res traces using this trial's background slice."""
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
        execution.max_workers,
        desc=f"{key} extracting traces" if execution.verbose else None,
    )
    trace_results = [r for r in trace_results if r.d_f.shape[0] > 0]
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
        draw_user_rois=annotations.enabled,
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
    *,
    execution=None,
):
    """Compose interpolation and rolling baseline on the original grid."""
    execution = ExecutionOptions() if execution is None else execution
    log("Estimating background", execution.verbose)
    # Preserve the original preparation order, even though these kernels are
    # not consumed by the interpolation-based background reconstruction.
    operators.expand_psf(psf2d)
    interp_data = np.full(
        (sel_pix_idxs.shape[0], low_res_data_norm.shape[1]),
        np.nan,
        dtype=np.float32,
    )
    for z in progress(
        range(num_fast_zs),
        desc="Interpolating planes",
        verbose=execution.verbose,
    ):
        z_idxs, sel_2d = geo.selected_pixels_2d_for_plane(
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
            method="cubic",
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
    *,
    execution=None,
):
    """Compose rho, activity and localization; preserve DoG and RNG order."""
    execution = ExecutionOptions() if execution is None else execution
    log("Computing activity image and localizing sources", execution.verbose)
    h_mots = operators.build_motion_h_matrices(
        sparse_h_inds,
        sparse_h_vals,
        umyx,
        sel_pix_idxs,
        num_super_pixels,
        dmd_pixels_per_row,
    )
    # compute_rho normalizes center/surround columns separately. The PSF still
    # determines its valid-column dilation/erosion mask, not the DoG widths.
    dog_center_sd = 0.9
    dog_surround_sd = 4.5
    center_kernel, center_kernel_ctr = operators.gaussian_kernel_2d(
        dog_center_sd
    )
    surround_kernel, surround_kernel_ctr = operators.gaussian_kernel_2d(
        dog_surround_sd
    )
    d_mats, d_mats_exp = [], []
    for z in progress(
        range(num_fast_zs),
        desc="Computing D matrices",
        verbose=execution.verbose,
    ):
        _, sel_2d = geo.selected_pixels_2d_for_plane(
            sel_pix_idxs, z, dmd_pixels_per_column, dmd_pixels_per_row
        )
        d_mats.append(
            operators.build_convolution_matrix(
                sel_2d, center_kernel, center_kernel_ctr
            )
        )
        d_mats_exp.append(
            operators.build_convolution_matrix(
                sel_2d, surround_kernel, surround_kernel_ctr
            )
        )
    rho = activity.compute_rho(
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
        verbose=execution.verbose,
    )
    nan_ct = activity.mask_high_nan_rho(rho)
    rho = activity.smooth_rho(
        rho,
        activity.decay_kernel_1d(params.decay_tau_s, align_hz),
        verbose=execution.verbose,
    )
    act_im = si.accumulate_activity_image(
        rho,
        sel_pix_idxs,
        num_fast_zs,
        dmd_pixels_per_column,
        dmd_pixels_per_row,
        verbose=execution.verbose,
    )
    # Only final median geometry follows the dominant retained motion bin.
    # Dropped frames (-1) do not vote; rho/NMF retain original scan centers.
    motion_counts = np.bincount(mot_inds_yx[mot_inds_yx >= 0])
    dominant_motion = umyx[np.argmax(motion_counts)]
    act_im = si.finalize_activity_image(
        act_im,
        sel_pix_idxs,
        nan_ct,
        ref_r=ref_r + int(dominant_motion[0]),
        ref_c=ref_c + int(dominant_motion[1]),
        ref_d=ref_d,
    )
    source_seeds = get_act_im_peaks(
        act_im, peak_th=params.peakth, buffer_size=params.peak_buffer
    )
    act_im_peaks = source_seeds.copy()
    print(f"Number of detected sources: {act_im_peaks.shape[0]}")
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
        verbose=execution.verbose,
    )
    return (
        act_im,
        act_im_peaks,
        result["A"],
        result["source_params"],
        result["source_snr"],
    )
