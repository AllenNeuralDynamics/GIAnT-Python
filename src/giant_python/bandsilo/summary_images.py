"""Summary/visualization images: the mean image and the activity image.

Ported from ``extractSLAP2IntegrationSources.py``. Groups the two per-DMD
summary images written to ``experiment_summary.h5`` / consumed downstream:

* :func:`compute_mean_image` (ref 1913-1918) scatters the kept-frame mean
  trace onto the reference-pixel grid, producing the ``mean_im`` visualization.
* :func:`accumulate_activity_image` / :func:`finalize_activity_image` (ref
  2358-2504) turn the smoothed ``rho`` from
  :mod:`giant_python.bandsilo.background` into the ``act_im`` activity image
  that Phase-5 peak detection localizes sources in.

The activity image is built by sweeping ``rho`` in temporal batches, finding
spatio-temporal local maxima (a voxel greater than its 6 spatial neighbors in
the same frame and its temporal neighbors), and accumulating their squared
value; it is then masked where data was never valid and has a local
(1x11x11) NaN-median subtracted.

The reference's optional ``profile_activity_map`` timing instrumentation is
intentionally dropped (it only prints timings and does not affect output).
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndimage

from .trial_data import fast_dilation


def compute_mean_image(
    low_res_data_norm: np.ndarray,
    unique_motion: np.ndarray,
    mot_inds: np.ndarray,
    frames_to_keep: np.ndarray,
    ref_d: np.ndarray,
    ref_r: np.ndarray,
    ref_c: np.ndarray,
    num_fast_zs: int,
    dmd_pixels_per_column: int,
    dmd_pixels_per_row: int,
    num_channels: int,
    low_res_data2_norm: np.ndarray = None,
) -> np.ndarray:
    """Scatter the kept-frame mean trace onto the reference pixel grid.

    The mean over kept frames is placed at each superpixel's reference pixel,
    shifted by the most common motion bin, producing the ``mean_im``
    visualization ``(channels, z, rows, cols)``.

    Parameters
    ----------
    low_res_data_norm : ndarray of shape (n_superpixels, n_frames)
        Count-normalized channel-1 low-res traces.
    unique_motion : ndarray of shape (n_bins, 3)
        Unique motion vectors from
        :func:`giant_python.bandsilo.background.bin_motion`.
    mot_inds : ndarray
        Per-frame motion-bin indices (used to pick the most common bin).
    frames_to_keep : ndarray of bool
        Frames contributing to the mean.
    ref_d, ref_r, ref_c : ndarray
        Per-superpixel reference depth/row/column indices.
    num_fast_zs, dmd_pixels_per_column, dmd_pixels_per_row : int
        Output grid geometry.
    num_channels : int
        Number of acquisition channels (>= 2 fills the second channel).
    low_res_data2_norm : ndarray, optional
        Count-normalized channel-2 traces (used when ``num_channels >= 2``).

    Returns
    -------
    ndarray of float32
        ``mean_im`` with shape
        ``(num_channels, num_fast_zs, dmd_pixels_per_column,
        dmd_pixels_per_row)``.
    """
    mean_im = np.full(
        (num_channels, num_fast_zs, dmd_pixels_per_column, dmd_pixels_per_row),
        np.nan,
        dtype=np.float32,
    )
    most_common_mot = np.argmax(np.bincount(mot_inds))
    sel_r = ref_r + int(unique_motion[most_common_mot, 0])
    sel_c = ref_c + int(unique_motion[most_common_mot, 1])
    mean_im[0, ref_d, sel_r, sel_c] = np.nanmean(
        low_res_data_norm[:, frames_to_keep], axis=1
    )
    if num_channels >= 2:
        mean_im[1, ref_d, sel_r, sel_c] = np.nanmean(
            low_res_data2_norm[:, frames_to_keep], axis=1
        )
    return mean_im


def accumulate_activity_image(
    rho: np.ndarray,
    sel_pix_idxs: np.ndarray,
    num_fast_zs: int,
    dmd_pixels_per_column: int,
    dmd_pixels_per_row: int,
    batch_size: int = 1000,
) -> np.ndarray:
    """Accumulate squared spatio-temporal local maxima of ``rho``.

    ``rho`` (selected pixels x frames) is scattered into a padded spatial grid
    one temporal batch at a time. Within each batch, a voxel is a local maximum
    if it strictly exceeds its two temporal neighbors and its four in-plane
    neighbors and is not adjacent (after a 3x3 dilation) to any NaN voxel. The
    squared values of local maxima are summed into ``act_im``.

    Parameters
    ----------
    rho : ndarray of shape (n_sel, n_frames)
        Smoothed rho response (NaN allowed).
    sel_pix_idxs : ndarray of int
        Flattened selected-pixel indices into the spatial grid.
    num_fast_zs, dmd_pixels_per_column, dmd_pixels_per_row : int
        Spatial grid geometry.
    batch_size : int
        Number of frames processed per batch (capped at ``n_frames``).

    Returns
    -------
    ndarray of float32
        ``act_im`` with shape
        ``(num_fast_zs, dmd_pixels_per_column, dmd_pixels_per_row)``.
    """
    n_frames = rho.shape[1]
    batch_size = min(batch_size, n_frames)
    num_batches = int(np.ceil(n_frames / batch_size))

    spatial_coords = np.unravel_index(
        sel_pix_idxs, (num_fast_zs, dmd_pixels_per_column, dmd_pixels_per_row)
    )
    depth_indices = spatial_coords[0][None, :].astype(np.intp, copy=False)
    row_indices = spatial_coords[1][None, :].astype(np.intp, copy=False)
    col_indices = spatial_coords[2][None, :].astype(np.intp, copy=False)

    dilation_struct = np.ones((3, 3), dtype=np.uint8)
    act_im = np.zeros(
        (num_fast_zs, dmd_pixels_per_column, dmd_pixels_per_row),
        dtype=np.float32,
    )
    temporal_pad = 1

    # Crop to a padded ROI containing all selected pixels; all-NaN regions
    # outside it can never produce a local maximum.
    row_min = int(np.min(row_indices))
    row_max = int(np.max(row_indices))
    col_min = int(np.min(col_indices))
    col_max = int(np.max(col_indices))
    row_start = max(0, row_min - 1)
    row_stop = min(dmd_pixels_per_column, row_max + 2)
    col_start = max(0, col_min - 1)
    col_stop = min(dmd_pixels_per_row, col_max + 2)
    roi_h = row_stop - row_start
    roi_w = col_stop - col_start

    row_indices_roi = (row_indices - row_start).astype(np.intp, copy=False)
    col_indices_roi = (col_indices - col_start).astype(np.intp, copy=False)

    prealloc_size = int(min(batch_size + 2 * temporal_pad, n_frames))
    batch_rho = np.empty(
        (prealloc_size, num_fast_zs, roi_h, roi_w), dtype=np.float32
    )
    nan_mask = np.empty_like(batch_rho, dtype=bool)
    interior_shape = (
        max(prealloc_size - 2, 0),
        num_fast_zs,
        max(roi_h - 2, 0),
        max(roi_w - 2, 0),
    )
    local_maxima_core = np.empty(interior_shape, dtype=bool)
    compare_tmp_core = np.empty(interior_shape, dtype=bool)
    batch_rho_pow2_core = np.empty(interior_shape, dtype=np.float32)
    time_indices_pre = np.arange(prealloc_size)[:, None]

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_frames)
        padded_start = max(0, batch_start - temporal_pad)
        padded_end = min(n_frames, batch_end + temporal_pad)
        curr_size = padded_end - padded_start

        br = batch_rho[:curr_size]
        nm = nan_mask[:curr_size]

        br.fill(0)
        nm.fill(True)
        time_indices = time_indices_pre[:curr_size]
        batch_vals = rho[:, padded_start:padded_end].T
        br[
            time_indices, depth_indices, row_indices_roi, col_indices_roi
        ] = batch_vals
        nm[
            time_indices, depth_indices, row_indices_roi, col_indices_roi
        ] = np.isnan(batch_vals)

        br[nm] = 0
        core_t = curr_size - 2
        if core_t <= 0:
            continue

        dilated_nan_mask = fast_dilation(nm, dilation_struct)

        center = br[1:-1, :, 1:-1, 1:-1]
        lmc = local_maxima_core[:core_t]
        ctmp = compare_tmp_core[:core_t]
        b3c = batch_rho_pow2_core[:core_t]

        np.greater(center, br[:-2, :, 1:-1, 1:-1], out=lmc)
        np.greater(center, br[2:, :, 1:-1, 1:-1], out=ctmp)
        np.logical_and(lmc, ctmp, out=lmc)
        np.greater(center, br[1:-1, :, :-2, 1:-1], out=ctmp)
        np.logical_and(lmc, ctmp, out=lmc)
        np.greater(center, br[1:-1, :, 2:, 1:-1], out=ctmp)
        np.logical_and(lmc, ctmp, out=lmc)
        np.greater(center, br[1:-1, :, 1:-1, :-2], out=ctmp)
        np.logical_and(lmc, ctmp, out=lmc)
        np.greater(center, br[1:-1, :, 1:-1, 2:], out=ctmp)
        np.logical_and(lmc, ctmp, out=lmc)
        np.logical_not(dilated_nan_mask[1:-1, :, 1:-1, 1:-1], out=ctmp)
        np.logical_and(lmc, ctmp, out=lmc)

        np.multiply(center, center, out=b3c)
        np.multiply(b3c, lmc, out=b3c, casting="unsafe")
        r_lo, r_hi = row_start + 1, row_stop - 1
        c_lo, c_hi = col_start + 1, col_stop - 1
        act_im[:, r_lo:r_hi, c_lo:c_hi] += np.sum(
            b3c, axis=0, dtype=np.float32
        )

    return act_im


def finalize_activity_image(
    act_im: np.ndarray,
    sel_pix_idxs: np.ndarray,
    nan_ct: np.ndarray,
) -> np.ndarray:
    """Mask never-valid pixels and subtract a local NaN-median.

    Pixels whose rho was mostly NaN (``nan_ct > 0.5``) or that were never
    selected are set to NaN; the remaining pixels have a local ``1x11x11``
    NaN-median subtracted, and the mask is re-applied.

    Parameters
    ----------
    act_im : ndarray
        Accumulated activity image (modified in place before subtraction).
    sel_pix_idxs : ndarray of int
        Flattened selected-pixel indices into ``act_im``'s grid.
    nan_ct : ndarray of shape (n_sel,)
        Per-selected-pixel rho NaN fraction from
        :func:`giant_python.bandsilo.background.mask_high_nan_rho`.

    Returns
    -------
    ndarray
        The finalized activity image (NaN outside the valid support).
    """
    nan_mask = np.full_like(act_im, True, dtype=bool)
    valid_sel_pix = np.flatnonzero(nan_ct <= 0.5)
    nan_mask[
        np.unravel_index(sel_pix_idxs[valid_sel_pix], nan_mask.shape)
    ] = False
    act_im[nan_mask] = np.nan

    med_act_im = ndimage.generic_filter(act_im, np.nanmedian, size=(1, 11, 11))
    act_im = act_im - med_act_im
    act_im[nan_mask] = np.nan
    return act_im
