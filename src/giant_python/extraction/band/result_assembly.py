"""Canonical band numerical output assembly; persistence belongs to shared IO.

Internal reference offsets become MATLAB displacement once here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from torch import Tensor

from ...models.experiment_summary import (
    FrameInfo,
    PathSummary,
    Source,
    UserRoi,
    Visualizations,
)
from ...numerics.baseline import assemble_dff
from .types import TrialTraceResult


@dataclass
class PathResult:
    """One path's numerical outputs, geometry, and assembly window lengths.

    ``a`` is (pixels, sources), ``source_params`` is (sources, 6), and
    trace_results contains named trial records.
    Reference offsets use the INTERNAL band convention;
    online shifts retain their saved displacement convention.
    No tensor copies or numerical normalization occur in this container.
    """

    a: Tensor
    source_params: Tensor
    source_snr: np.ndarray
    act_im: np.ndarray
    mean_im: np.ndarray
    act_im_peaks: np.ndarray
    trace_results: list[TrialTraceResult]
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
    """Map frames/sources to sources/channels/frames, filling channel 0."""
    out = np.full(
        (arr.shape[1], num_channels, arr.shape[0]), np.nan, dtype=arr.dtype
    )
    out[:, 0, :] = arr.T
    return out


def _dmd_user_rois(user_rois: Optional[dict], key: str) -> tuple:
    """Select one DMD's resolved ROI masks, labels and superpixels."""
    if user_rois is None:
        return None, None, []
    return (
        user_rois["user_roi_masks"].get(key),
        user_rois["user_roi_labels"].get(key),
        user_rois["user_roi_superpixels"].get(key, []),
    )


def assemble_path_summary(pr: PathResult, path_index: int = 0) -> PathSummary:
    """Concatenate trials and assemble dF/F with unchanged numerical ordering.

    Offline reference offsets are negated once into MATLAB displacement.
    Online shifts, frame lines, input arrays, and interpolation are unchanged.
    Populated model entries share array views; empty templates retain layout.
    """
    results = pr.trace_results
    d_f = np.concatenate([r.d_f for r in results], axis=0)
    f0_ls = np.concatenate([r.f0_ls for r in results], axis=0)
    f_full, f0, d_f, _ = assemble_dff(
        d_f, f0_ls, pr.denoise_window, pr.baseline_window
    )
    profiles = pr.a.numpy().T.reshape(
        -1, pr.num_fast_zs, pr.dmd_pixels_per_column, pr.dmd_pixels_per_row
    )
    coords = pr.source_params[:, :3].numpy()
    df_channels = _with_nan_channel(d_f, pr.num_channels)
    f0_channels = _with_nan_channel(f0, pr.num_channels)
    snr = pr.source_snr.reshape(-1, 1)
    sources = [
        Source(
            profile=profiles[j],
            coords=coords[j],
            df_ls=df_channels[j],
            f0=f0_channels[j],
            snr=snr[j, 0],
        )
        for j in range(profiles.shape[0])
    ]
    frame_info = FrameInfo(
        trial_num_frames=np.concatenate(
            [[len(r.frame_line_idxs)] for r in results]
        ).reshape(-1, 1),
        discard_frames=np.any(np.isnan(f_full), axis=1).reshape(-1, 1),
        frame_line_idxs=np.concatenate(
            [r.frame_line_idxs for r in results], axis=0
        ).reshape(-1, 1),
        offlineXshifts=-np.concatenate(
            [r.reference_offsets[1] for r in results], axis=0
        ).reshape(-1, 1),
        offlineYshifts=-np.concatenate(
            [r.reference_offsets[0] for r in results], axis=0
        ).reshape(-1, 1),
        offlineZshifts=-np.concatenate(
            [r.reference_offsets[2] for r in results], axis=0
        ).reshape(-1, 1),
        onlineXshifts=np.concatenate(
            [r.online_shifts[1] for r in results], axis=0
        ).reshape(-1, 1),
        onlineYshifts=np.concatenate(
            [r.online_shifts[0] for r in results], axis=0
        ).reshape(-1, 1),
        onlineZshifts=np.concatenate(
            [r.online_shifts[2] for r in results], axis=0
        ).reshape(-1, 1),
    )
    annotation_enabled = False
    user_rois = []
    empty_roi_arrays = {}
    if pr.draw_user_rois and pr.soma_masks is not None:
        annotation_enabled = True
        masks = (
            np.stack([np.asarray(m, dtype=bool) for m in pr.soma_masks])
            if len(pr.soma_masks) > 0
            else np.zeros((0, pr.num_fast_zs, *pr.yx_shape), dtype=bool)
        )
        fluorescence = np.concatenate(
            [r.user_roi_f for r in results], axis=0
        ).transpose(1, 2, 0)
        labels = np.asarray(pr.soma_labels, dtype=object).reshape(-1)
        user_rois = [
            UserRoi(
                label=str(label),
                mask=masks[j] if j < masks.shape[0] else None,
                f=fluorescence[j] if j < fluorescence.shape[0] else None,
            )
            for j, label in enumerate(labels)
        ]
        if not user_rois:
            empty_roi_arrays = {"mask": masks, "F": fluorescence}
    return PathSummary(
        path_index=path_index,
        sources=sources,
        user_rois=user_rois,
        visualizations=Visualizations(
            act_im=pr.act_im, mean_im=pr.mean_im, act_im_peaks=pr.act_im_peaks
        ),
        frame_info=frame_info,
        global_f=np.concatenate([r.global_f for r in results], axis=0).T,
        z_depths=pr.z_depths,
        annotation_enabled=annotation_enabled,
        _empty_source_arrays=(
            {}
            if sources
            else {
                "profiles": profiles,
                "coords": coords,
                "dF_ls": df_channels,
                "F0": f0_channels,
                "SNR": snr,
            }
        ),
        _empty_roi_arrays=empty_roi_arrays,
    )
