"""Motion binning, background/noise estimation, and rho computation.

Ported from the background/noise block of
``extractSLAP2IntegrationSources.py`` (ref 1901-2356). This module owns the
kernels that turn the per-superpixel low-res traces into:

* motion bins (3-D and 2-D) and the selected-pixel support,
* the PSF convolution matrices ``D``/``D_expanded`` and the sparse per-motion
  ``H`` matrices,
* the interpolated background estimate (``build_interp_data`` +
  rolling-window baseline) and its reassembly onto the superpixel grid,
* an affine noise-variance model producing ``data_std`` and the z-scored
  ``residual``,
* the ``rho`` matched-filter response and its NaN-aware temporal smoothing.

The summary images that consume these outputs (the mean image, and the
activity image built from ``rho``) live in
:mod:`giant_python.bandsilo.summary_images`. The full threading of these
kernels over the per-DMD arrays is done by the Phase-8 pipeline driver; here
each kernel is a standalone, testable function.

The commented-out soft-impute background completion and rolling-MAD rho
normalization in the reference are intentionally dropped (dead code).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import scipy.ndimage as ndimage
import torch
from scipy import signal
from scipy.interpolate import RectBivariateSpline

from .progress import progress


def bin_motion(
    motion_r: np.ndarray, motion_c: np.ndarray, motion_z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Group frames into unique rounded (row, col, z) motion bins.

    Parameters
    ----------
    motion_r, motion_c, motion_z : ndarray
        Per-frame downsampled motion in rows, columns, and z.

    Returns
    -------
    unique_motion : ndarray of shape (n_bins, 3)
        The unique rounded ``[row, col, z]`` motion vectors.
    mot_inds : ndarray of shape (n_frames,)
        Index of each frame's motion bin into ``unique_motion``.
    """
    unique_motion, mot_inds = np.unique(
        np.round(np.stack((motion_r, motion_c, motion_z), axis=1)),
        axis=0,
        return_inverse=True,
    )
    # np.unique's inverse is 1-D on numpy<2 and 2.1+, but 2-D on 2.0; flatten
    # so downstream bincount/isin behave identically across versions.
    return unique_motion, np.reshape(mot_inds, -1)


def select_motion_bins(
    unique_motion: np.ndarray,
    mot_inds: np.ndarray,
    motion_z: np.ndarray,
    z_thresh: float = 1.5,
    min_frames: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep motion bins near the median z with enough frames.

    A bin is kept when its z is within ``z_thresh`` of the median z (over all
    frames) and it contains more than ``min_frames`` frames. After filtering,
    the remaining frames are treated as having no z motion.

    Parameters
    ----------
    unique_motion : ndarray of shape (n_bins, 3)
        Unique motion vectors from :func:`bin_motion`.
    mot_inds : ndarray of shape (n_frames,)
        Per-frame motion-bin indices.
    motion_z : ndarray
        Per-frame z motion (used for the median).
    z_thresh : float
        Maximum absolute z deviation from the median, in microns.
    min_frames : int
        Minimum frame count (strictly greater) for a bin to be kept.

    Returns
    -------
    mot_inds_to_keep : ndarray
        Indices of the kept bins into ``unique_motion``.
    frames_to_keep : ndarray of bool
        Per-frame mask of frames belonging to a kept bin.
    """
    median_z = np.median(motion_z)
    bin_counts = np.bincount(mot_inds, minlength=unique_motion.shape[0])
    keep_mask = (np.abs(unique_motion[:, 2] - median_z) <= z_thresh) & (
        bin_counts > min_frames
    )
    mot_inds_to_keep = np.nonzero(keep_mask)[0]
    frames_to_keep = np.isin(mot_inds, mot_inds_to_keep)
    return mot_inds_to_keep, frames_to_keep


def bin_motion_yx(
    motion_r: np.ndarray,
    motion_c: np.ndarray,
    frames_to_keep: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bin kept frames into unique rounded (row, col) motion bins.

    Parameters
    ----------
    motion_r, motion_c : ndarray
        Per-frame downsampled row/column motion.
    frames_to_keep : ndarray of bool
        Frames to bin; dropped frames get index ``-1``.

    Returns
    -------
    unique_motion_to_keep_yx : ndarray of shape (n_yx_bins, 2)
        Unique rounded ``[row, col]`` motion vectors over the kept frames.
    mot_inds_yx : ndarray of shape (n_frames,), int32
        Per-frame bin index into ``unique_motion_to_keep_yx``; ``-1`` for
        frames not kept.
    """
    mot_inds_yx = -1 * np.ones((frames_to_keep.shape[0],), dtype=np.int32)
    stacked = np.round(
        np.stack((motion_r, motion_c), axis=1)[frames_to_keep, :]
    )
    unique_motion_to_keep_yx, inv = np.unique(
        stacked, axis=0, return_inverse=True
    )
    mot_inds_yx[frames_to_keep] = np.reshape(inv, -1)
    return unique_motion_to_keep_yx, mot_inds_yx


def build_selected_pixel_mask(
    unique_motion_to_keep_yx: np.ndarray,
    ref_d: np.ndarray,
    ref_r: np.ndarray,
    ref_c: np.ndarray,
    num_fast_zs: int,
    dmd_pixels_per_column: int,
    dmd_pixels_per_row: int,
    psf2d: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the selected-pixel support mask over all kept motion shifts.

    Marks every reference pixel shifted by each kept 2-D motion bin, then
    dilates in-plane by the PSF footprint.

    Parameters
    ----------
    unique_motion_to_keep_yx : ndarray of shape (n_yx_bins, 2)
        Kept 2-D motion vectors.
    ref_d, ref_r, ref_c : ndarray
        Per-superpixel reference depth/row/column indices.
    num_fast_zs, dmd_pixels_per_column, dmd_pixels_per_row : int
        Grid geometry.
    psf2d : ndarray
        The (cropped) PSF for this DMD; its shape sets the dilation footprint.

    Returns
    -------
    sel_pix_mask : ndarray of bool
        ``(num_fast_zs, dmd_pixels_per_column, dmd_pixels_per_row)`` support.
    sel_pix_idxs : ndarray of int
        Flattened indices of the selected pixels.
    """
    sel_pix_mask = np.zeros(
        (num_fast_zs, dmd_pixels_per_column, dmd_pixels_per_row), dtype=bool
    )
    for i in range(len(unique_motion_to_keep_yx)):
        sel_pix_mask[
            ref_d,
            ref_r + int(unique_motion_to_keep_yx[i, 0]),
            ref_c + int(unique_motion_to_keep_yx[i, 1]),
        ] = True
    sel_pix_mask = ndimage.binary_dilation(
        sel_pix_mask,
        structure=np.ones((1, psf2d.shape[0], psf2d.shape[1]), dtype=bool),
    )
    sel_pix_idxs = np.flatnonzero(sel_pix_mask)
    return sel_pix_mask, sel_pix_idxs


def pixel_coords_from_idxs(
    sel_pix_idxs: np.ndarray,
    dmd_pixels_per_column: int,
    dmd_pixels_per_row: int,
) -> np.ndarray:
    """Convert flat selected-pixel indices to ``[z, row, col]`` coordinates.

    Parameters
    ----------
    sel_pix_idxs : ndarray of int
        Flattened selected-pixel indices.
    dmd_pixels_per_column, dmd_pixels_per_row : int
        Grid geometry.

    Returns
    -------
    ndarray of shape (n_sel, 3), int32
        ``[z, row, col]`` per selected pixel.
    """
    plane_size = dmd_pixels_per_column * dmd_pixels_per_row
    pixel_coords = np.empty((len(sel_pix_idxs), 3), dtype=np.int32)
    pixel_coords[:, 0] = sel_pix_idxs // plane_size
    remainder = sel_pix_idxs % plane_size
    pixel_coords[:, 1] = remainder // dmd_pixels_per_row
    pixel_coords[:, 2] = remainder % dmd_pixels_per_row
    return pixel_coords


def expand_psf(
    psf2d: np.ndarray, ex_fac: int = 2
) -> Tuple[torch.Tensor, Tuple[int, int], torch.Tensor, Tuple[int, int]]:
    """Normalize a PSF and build a center-aligned higher-resolution copy.

    The expanded PSF is an ``ex_fac``x up-sampled cubic-spline interpolation
    that keeps the center aligned, used to form the ``HD - HD_expanded``
    matched filter in :func:`compute_rho`.

    Parameters
    ----------
    psf2d : ndarray
        The (cropped) PSF for this DMD.
    ex_fac : int
        Expansion factor along each axis.

    Returns
    -------
    psf_tensor : torch.Tensor
        Sum-normalized PSF.
    psf_center : tuple of int
        ``(row, col)`` center of ``psf_tensor``.
    psf_tensor_expanded : torch.Tensor
        Sum-normalized expanded PSF.
    psf_center_expanded : tuple of int
        ``(row, col)`` center of ``psf_tensor_expanded``.
    """
    psf_arr = np.asarray(psf2d, dtype=np.float32)
    psf_tensor = torch.from_numpy(psf_arr).float()
    psf_tensor = psf_tensor / torch.sum(psf_tensor)
    psf_shape = psf_arr.shape
    psf_center = (psf_shape[0] // 2, psf_shape[1] // 2)

    expanded_center_y = (psf_shape[0] * ex_fac - 1) / 2
    expanded_center_x = (psf_shape[1] * ex_fac - 1) / 2
    orig_y = torch.linspace(
        -expanded_center_y, expanded_center_y, psf_shape[0]
    )
    orig_x = torch.linspace(
        -expanded_center_x, expanded_center_x, psf_shape[1]
    )
    expanded_y = torch.linspace(
        -expanded_center_y, expanded_center_y, psf_shape[0] * ex_fac
    )
    expanded_x = torch.linspace(
        -expanded_center_x, expanded_center_x, psf_shape[1] * ex_fac
    )
    interp_spline = RectBivariateSpline(
        orig_y.numpy(), orig_x.numpy(), psf_tensor.numpy()
    )
    psf_tensor_expanded = torch.from_numpy(
        interp_spline(expanded_y.numpy(), expanded_x.numpy())
    ).float()
    psf_tensor_expanded = psf_tensor_expanded / torch.sum(psf_tensor_expanded)
    psf_shape_expanded = psf_tensor_expanded.shape
    psf_center_expanded = (
        psf_shape_expanded[0] // 2,
        psf_shape_expanded[1] // 2,
    )
    return psf_tensor, psf_center, psf_tensor_expanded, psf_center_expanded


def selected_pixels_2d_for_plane(
    sel_pix_idxs: np.ndarray,
    plane_z: int,
    dmd_pixels_per_column: int,
    dmd_pixels_per_row: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the in-plane ``[row, col]`` coords of one z-plane's pixels.

    Parameters
    ----------
    sel_pix_idxs : ndarray of int
        Flattened selected-pixel indices (all planes).
    plane_z : int
        The z-plane to select.
    dmd_pixels_per_column, dmd_pixels_per_row : int
        Grid geometry.

    Returns
    -------
    z_idxs : ndarray
        Positions (into ``sel_pix_idxs``) of pixels in ``plane_z``.
    sel_pixels_2d : ndarray of shape (n_plane, 2)
        ``[row, col]`` coordinates of those pixels.
    """
    plane_size = dmd_pixels_per_column * dmd_pixels_per_row
    sel_pix_z = sel_pix_idxs // plane_size
    sel_pix_remainder = sel_pix_idxs % plane_size
    z_idxs = np.flatnonzero(sel_pix_z == plane_z)
    sel_pixels_2d = np.column_stack(
        [
            sel_pix_remainder[z_idxs] // dmd_pixels_per_row,
            sel_pix_remainder[z_idxs] % dmd_pixels_per_row,
        ]
    )
    return z_idxs, sel_pixels_2d


def build_convolution_matrix(
    sel_pixels_2d: np.ndarray,
    psf_tensor: torch.Tensor,
    psf_center: Tuple[int, int],
) -> torch.Tensor:
    """Build a dense PSF-convolution matrix for one plane's selected pixels.

    Entry ``(t, s)`` is the PSF weight for the offset from source pixel ``s``
    to target pixel ``t`` (zero where the offset falls outside the PSF). Used
    with both the base and expanded PSFs to form ``D``/``D_expanded``.

    Parameters
    ----------
    sel_pixels_2d : ndarray of shape (n_plane, 2)
        In-plane ``[row, col]`` coordinates.
    psf_tensor : torch.Tensor
        Sum-normalized PSF (base or expanded).
    psf_center : tuple of int
        ``(row, col)`` center of ``psf_tensor``.

    Returns
    -------
    torch.Tensor
        Dense ``(n_plane, n_plane)`` convolution matrix.
    """
    n = sel_pixels_2d.shape[0]
    if n == 0:
        return torch.zeros((0, 0), dtype=torch.float32)
    src_rows = sel_pixels_2d[:, 0][np.newaxis, :]
    src_cols = sel_pixels_2d[:, 1][np.newaxis, :]
    tgt_rows = sel_pixels_2d[:, 0][:, np.newaxis]
    tgt_cols = sel_pixels_2d[:, 1][:, np.newaxis]
    rel_rows = tgt_rows - src_rows + psf_center[0]
    rel_cols = tgt_cols - src_cols + psf_center[1]
    psf_np = psf_tensor.numpy()
    psf_shape = psf_np.shape
    valid_mask = (
        (rel_rows >= 0)
        & (rel_rows < psf_shape[0])
        & (rel_cols >= 0)
        & (rel_cols < psf_shape[1])
    )
    out = np.zeros((n, n), dtype=np.float32)
    out[valid_mask] = psf_np[rel_rows[valid_mask], rel_cols[valid_mask]]
    return torch.from_numpy(out)


def _collect_interp_columns(
    cols, s_c, s_r, sel_idxs_by_col_all, sel_rows, r0, r1
):
    """Group a motion bin's reference pixels by column for interpolation.

    Columns with a single reference pixel become ``single_points`` (copied
    verbatim); columns with several become sorted ``(rows, source-index)``
    entries for linear interpolation.
    """
    rows_by_col = {}
    idxs_by_col = {}
    single_points = []
    for c in cols:
        c_int = int(c)
        if c_int not in sel_idxs_by_col_all:
            continue
        data_ix = np.flatnonzero(s_c == c)
        if data_ix.size == 1:
            rr = int(s_r[data_ix[0]])
            col_sel = sel_idxs_by_col_all[c_int]
            rr_match = col_sel[sel_rows[col_sel] == rr]
            if rr_match.size > 0 and (r0 <= rr <= r1):
                single_points.append((rr_match, int(data_ix[0])))
            continue
        rows = s_r[data_ix]
        order = rows.argsort(kind="mergesort")
        rows_by_col[c_int] = rows[order]
        idxs_by_col[c_int] = data_ix[order]
    return rows_by_col, idxs_by_col, single_points


def _interp_columns(
    out,
    rows_by_col,
    idxs_by_col,
    sel_idxs_by_col_all,
    sel_rows,
    data,
    frames,
    dtype,
):
    """Linearly interpolate each multi-point column onto its selected pixels.

    Selected rows that coincide with a reference row are copied exactly; those
    strictly between two reference rows are linearly interpolated; those
    outside the sampled row range are left untouched (NaN).
    """
    for c_int, rows in rows_by_col.items():
        col_sel_pix_idxs = sel_idxs_by_col_all[c_int]
        target_rows = sel_rows[col_sel_pix_idxs]
        src_idx = idxs_by_col[c_int]

        pos = np.searchsorted(rows, target_rows, side="left")
        in_bounds = pos < rows.size
        exact = np.zeros_like(pos, dtype=bool)
        exact[in_bounds] = rows[pos[in_bounds]] == target_rows[in_bounds]

        if np.any(exact):
            exact_rows = np.flatnonzero(exact)
            exact_src = src_idx[pos[exact]]
            out[np.ix_(col_sel_pix_idxs[exact_rows], frames)] = data[
                exact_src
            ][:, frames]

        lo = pos - 1
        hi = pos
        interp_mask = (~exact) & (lo >= 0) & (hi < rows.size)
        if not np.any(interp_mask):
            continue

        interp_rows = np.flatnonzero(interp_mask)
        lo_v = lo[interp_mask]
        hi_v = hi[interp_mask]
        row_lo = rows[lo_v].astype(dtype, copy=False)
        row_hi = rows[hi_v].astype(dtype, copy=False)
        denom = row_hi - row_lo
        nonzero = denom != 0
        if not np.any(nonzero):
            # Unreachable defensive guard (mirrors the reference): sorted
            # distinct rows make searchsorted never straddle equal rows, so
            # denom is always non-zero here.
            continue  # pragma: no cover

        interp_rows = interp_rows[nonzero]
        lo_v = lo_v[nonzero]
        hi_v = hi_v[nonzero]
        row_lo = row_lo[nonzero]
        row_hi = row_hi[nonzero]
        alpha = (
            (target_rows[interp_rows].astype(dtype) - row_lo)
            / (row_hi - row_lo)
        )[:, None]

        vals_lo = data[src_idx[lo_v]][:, frames]
        vals_hi = data[src_idx[hi_v]][:, frames]
        out[np.ix_(col_sel_pix_idxs[interp_rows], frames)] = (
            1.0 - alpha
        ) * vals_lo + alpha * vals_hi


def build_interp_data(
    data: np.ndarray,
    ref_r: np.ndarray,
    ref_c: np.ndarray,
    sel_pixels_2d: np.ndarray,
    unique_motion_to_keep_yx: np.ndarray,
    mot_inds_yx: np.ndarray,
    height: int = 800,
    width: int = 1280,
    dtype=np.float32,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """Interpolate reference-pixel traces onto the selected-pixel grid.

    For each kept motion bin, the reference pixels (shifted by the bin's
    motion) are linearly interpolated along rows, per column, onto the selected
    pixels of the plane, filling one background estimate per selected pixel and
    frame. Selected pixels outside the sampled row range remain NaN.

    Parameters
    ----------
    data : ndarray of shape (n_ref, n_frames)
        Reference-pixel traces for one plane.
    ref_r, ref_c : ndarray
        Reference-pixel row/column indices for this plane.
    sel_pixels_2d : ndarray of shape (n_sel, 2)
        Selected-pixel ``[row, col]`` coordinates for this plane.
    unique_motion_to_keep_yx : ndarray of shape (n_yx_bins, 2)
        Kept 2-D motion vectors.
    mot_inds_yx : ndarray of shape (n_frames,)
        Per-frame motion-bin index (``-1`` for dropped frames).
    height, width : int
        Sensor row/column extents used to window valid shifts.
    dtype : numpy dtype
        Working/output dtype.

    Returns
    -------
    out : ndarray of shape (n_sel, n_frames)
        Interpolated background estimate (NaN where unsampled).
    row_bounds : tuple of int
        ``(r0, r1)`` inclusive row window.
    col_bounds : tuple of int
        ``(c0, c1)`` inclusive column window.
    """
    ref_r = np.asarray(ref_r, dtype=np.int32)
    ref_c = np.asarray(ref_c, dtype=np.int32)
    data = np.asarray(data, dtype=dtype)

    r0 = int(max(0, ref_r.min()) + unique_motion_to_keep_yx[:, 0].min())
    r1 = int(min(height, ref_r.max()) + unique_motion_to_keep_yx[:, 0].max())
    c0 = int(max(0, ref_c.min()) + unique_motion_to_keep_yx[:, 1].min())
    c1 = int(min(width, ref_c.max()) + unique_motion_to_keep_yx[:, 1].max())

    out = np.full((sel_pixels_2d.shape[0], data.shape[1]), np.nan, dtype=dtype)

    sel_cols = sel_pixels_2d[:, 1]
    sel_rows = sel_pixels_2d[:, 0]
    unique_sel_cols = np.unique(sel_cols)
    sel_idxs_by_col_all = {
        int(c): np.flatnonzero(sel_cols == c) for c in unique_sel_cols
    }

    frames_by_motion = [
        np.flatnonzero(mot_inds_yx == idx)
        for idx in range(len(unique_motion_to_keep_yx))
    ]

    for m_idx, frames in enumerate(frames_by_motion):
        if frames.size == 0:
            continue

        s_r = ref_r + int(unique_motion_to_keep_yx[m_idx, 0])
        s_c = ref_c + int(unique_motion_to_keep_yx[m_idx, 1])

        in_win = (s_r >= r0) & (s_r <= r1) & (s_c >= c0) & (s_c <= c1)
        if not np.any(in_win):
            continue
        cols = np.unique(s_c[in_win])

        rows_by_col, idxs_by_col, single_points = _collect_interp_columns(
            cols, s_c, s_r, sel_idxs_by_col_all, sel_rows, r0, r1
        )
        _interp_columns(
            out,
            rows_by_col,
            idxs_by_col,
            sel_idxs_by_col_all,
            sel_rows,
            data,
            frames,
            dtype,
        )
        for rr_match, src_pos in single_points:
            out[np.ix_(rr_match, frames)] = data[src_pos, frames]

    return out, (r0, r1), (c0, c1)


def baseline_window_frames(align_hz: float, baseline_window_s: float) -> int:
    """Return the rolling-baseline window length in frames.

    Parameters
    ----------
    align_hz : float
        Alignment/analysis rate, in Hz.
    baseline_window_s : float
        Baseline window, in seconds.

    Returns
    -------
    int
        ``int(align_hz * baseline_window_s)``.
    """
    return int(align_hz * baseline_window_s)


def compute_rolling_baseline(
    interp_data: np.ndarray, baseline_window: int
) -> np.ndarray:
    """Compute a NaN-aware rolling-mean baseline of the interpolated data.

    Uses running window sums of values and of the valid-sample count, then
    divides, so NaNs contribute nothing and windows with no valid samples stay
    NaN. ``interp_data`` is modified in place (NaNs replaced by 0).

    Parameters
    ----------
    interp_data : ndarray of shape (n_sel, n_frames)
        Interpolated background estimate (from :func:`build_interp_data`).
    baseline_window : int
        Rolling window length, in frames.

    Returns
    -------
    ndarray
        The rolling-mean background estimate (NaN where no valid samples).
    """
    valid = ~np.isnan(interp_data)
    np.nan_to_num(interp_data, copy=False, nan=0.0)

    sum_vals = (
        ndimage.uniform_filter1d(
            interp_data, size=baseline_window, axis=1, mode="nearest"
        )
        * baseline_window
    )
    valid_f = valid.astype(np.float32, copy=False)
    count_vals = (
        ndimage.uniform_filter1d(
            valid_f, size=baseline_window, axis=1, mode="nearest"
        )
        * baseline_window
    )

    interp_data_background = np.empty_like(sum_vals, dtype=interp_data.dtype)
    interp_data_background.fill(np.nan)
    np.divide(
        sum_vals, count_vals, out=interp_data_background, where=count_vals > 0
    )
    return interp_data_background


def assemble_background(
    interp_data_background: np.ndarray,
    unique_motion_to_keep_yx: np.ndarray,
    mot_inds_yx: np.ndarray,
    sel_pix_idxs: np.ndarray,
    ref_d: np.ndarray,
    ref_r: np.ndarray,
    ref_c: np.ndarray,
    n_super_pixels: int,
    n_frames: int,
    dmd_pixels_per_column: int,
    dmd_pixels_per_row: int,
) -> np.ndarray:
    """Map the selected-pixel background back onto superpixels per motion bin.

    For each kept motion bin and its frames, each superpixel's motion-shifted
    reference pixel is looked up in ``sel_pix_idxs`` and its background trace
    is copied onto that superpixel.

    Parameters
    ----------
    interp_data_background : ndarray of shape (n_sel, n_frames)
        Rolling-baseline background on the selected-pixel grid.
    unique_motion_to_keep_yx : ndarray of shape (n_yx_bins, 2)
        Kept 2-D motion vectors.
    mot_inds_yx : ndarray of shape (n_frames,)
        Per-frame motion-bin index (``-1`` for dropped frames).
    sel_pix_idxs : ndarray of int
        Sorted flat selected-pixel indices.
    ref_d, ref_r, ref_c : ndarray
        Per-superpixel reference depth/row/column indices.
    n_super_pixels, n_frames : int
        Output shape.
    dmd_pixels_per_column, dmd_pixels_per_row : int
        Grid geometry.

    Returns
    -------
    ndarray of shape (n_super_pixels, n_frames), float32
        Per-superpixel background (NaN where unmapped).
    """
    plane_size = dmd_pixels_per_column * dmd_pixels_per_row
    background = np.full((n_super_pixels, n_frames), np.nan, dtype=np.float32)

    for motion_idx in range(len(unique_motion_to_keep_yx)):
        motion_frames = np.flatnonzero(mot_inds_yx == motion_idx)
        d_r = int(unique_motion_to_keep_yx[motion_idx, 0])
        d_c = int(unique_motion_to_keep_yx[motion_idx, 1])
        s_d = ref_d
        s_r = ref_r + d_r
        s_c = ref_c + d_c

        shifted_indices = s_d * plane_size + s_r * dmd_pixels_per_row + s_c

        bg_idxs = np.searchsorted(sel_pix_idxs, shifted_indices)
        idxs_mask = bg_idxs < sel_pix_idxs.size
        idxs_mask[idxs_mask] &= (
            sel_pix_idxs[bg_idxs[idxs_mask]] == shifted_indices[idxs_mask]
        )

        if np.any(idxs_mask):
            valid_rows = np.flatnonzero(idxs_mask)
            background[
                np.ix_(valid_rows, motion_frames)
            ] = interp_data_background[
                np.ix_(bg_idxs[valid_rows], motion_frames)
            ]

    return background


def fit_noise_variance_model(
    low_res_data_norm: np.ndarray,
    background: np.ndarray,
    v_im: np.ndarray,
    vif: float = 1.38,
) -> Tuple[np.ndarray, float, float]:
    """Calibrate an affine noise-variance model and return per-sample std.

    Fits ``Var ~= Vk * (background * vIM) + Vb`` from the brightest pixels over
    the first valid frames, then returns the predicted standard deviation for
    every sample.

    Parameters
    ----------
    low_res_data_norm : ndarray of shape (n_superpixels, n_frames)
        Count-normalized channel-1 traces.
    background : ndarray of shape (n_superpixels, n_frames)
        Per-superpixel background estimate.
    v_im : ndarray of shape (n_superpixels, n_frames)
        Inverse count image (``1 / lowResDataCt``).
    vif : float
        Variance inflation factor.

    Returns
    -------
    data_std : ndarray
        Per-sample predicted standard deviation.
    v_k : float
        Fitted multiplicative variance coefficient.
    v_b : float
        Fitted additive variance floor.
    """
    first_valid_frames = np.flatnonzero(
        np.any(~np.isnan(background[:, :1000]), axis=0)
    )
    var_im = np.nanvar(low_res_data_norm[:, first_valid_frames], axis=1)
    v_b = np.nanpercentile(var_im, 5) * vif
    var_pred = np.nanmean(
        background[:, first_valid_frames], axis=1
    ) * np.nanmean(v_im[:, first_valid_frames], axis=1)
    sel_bright = var_pred > np.nanpercentile(var_pred, 90)
    v_k = np.nanpercentile(
        (var_im[sel_bright] - (v_b / vif)) / var_pred[sel_bright], 10
    )
    data_std = np.sqrt(np.clip(v_k * background * v_im, 0, None) + v_b)
    return data_std, v_k, v_b


def compute_residual(
    low_res_data_norm: np.ndarray,
    background: np.ndarray,
    data_std: np.ndarray,
) -> np.ndarray:
    """Return the background-subtracted, noise-normalized residual.

    Parameters
    ----------
    low_res_data_norm : ndarray
        Count-normalized traces.
    background : ndarray
        Per-superpixel background estimate.
    data_std : ndarray
        Per-sample predicted standard deviation.

    Returns
    -------
    ndarray
        ``(low_res_data_norm - background) / data_std``.
    """
    return (low_res_data_norm - background) / data_std


def build_motion_h_matrices(
    sparse_h_inds: np.ndarray,
    sparse_h_vals: np.ndarray,
    unique_motion_to_keep_yx: np.ndarray,
    sel_pix_idxs: np.ndarray,
    num_super_pixels: int,
    dmd_pixels_per_row: int,
) -> List[torch.Tensor]:
    """Build one sparse superpixel<-selected-pixel ``H`` per motion bin.

    Each motion bin shifts the base ``H`` columns (image pixels) by its 2-D
    motion, then remaps them into selected-pixel column space.

    Parameters
    ----------
    sparse_h_inds : ndarray of shape (2, nnz)
        Base ``H`` COO row/column indices (from ``geometry.build_sparse_h``).
    sparse_h_vals : ndarray of shape (nnz,)
        Base ``H`` values.
    unique_motion_to_keep_yx : ndarray of shape (n_yx_bins, 2)
        Kept 2-D motion vectors.
    sel_pix_idxs : ndarray of int
        Sorted flat selected-pixel indices.
    num_super_pixels : int
        Number of superpixels (``H`` row count).
    dmd_pixels_per_row : int
        Grid geometry (column-shift stride).

    Returns
    -------
    list of torch.Tensor
        One sparse COO ``(num_super_pixels, n_sel)`` tensor per motion bin.
    """
    n_mot = len(unique_motion_to_keep_yx)
    h_mots: List[torch.Tensor] = [None] * n_mot
    base_sparse_rows = sparse_h_inds[0]
    base_sparse_cols = sparse_h_inds[1]
    motion_shifts = unique_motion_to_keep_yx[:, 0].astype(
        np.int64, copy=False
    ) * dmd_pixels_per_row + unique_motion_to_keep_yx[:, 1].astype(
        np.int64, copy=False
    )
    shifted_inds = np.empty((2, base_sparse_cols.shape[0]), dtype=np.int64)
    shifted_inds[0] = base_sparse_rows
    for i, pix_shift in enumerate(motion_shifts):
        shifted_cols = base_sparse_cols + pix_shift
        shifted_inds[1] = np.searchsorted(sel_pix_idxs, shifted_cols)
        h_mots[i] = torch.sparse_coo_tensor(
            shifted_inds,
            sparse_h_vals,
            (num_super_pixels, sel_pix_idxs.shape[0]),
            dtype=torch.float32,
        )
    return h_mots


def _plane_column_maps(
    sel_pix_idxs: np.ndarray, num_fast_zs: int, plane_size: int
) -> Tuple[list, list, list]:
    """Precompute per-z column masks, indices, and remaps into local space.

    Returns three lists (one entry per z): the boolean column mask, the column
    indices of that plane's selected pixels, and a remap from global to local
    (per-plane) column index (``-1`` outside the plane).
    """
    sel_pix_z = sel_pix_idxs // plane_size
    z_masks = [torch.from_numpy(sel_pix_z == z) for z in range(num_fast_zs)]
    z_col_idxs = [torch.nonzero(m, as_tuple=False).squeeze(1) for m in z_masks]
    z_remaps = []
    for z in range(num_fast_zs):
        remap = torch.full((sel_pix_idxs.shape[0],), -1, dtype=torch.long)
        if z_col_idxs[z].numel() > 0:
            remap[z_col_idxs[z]] = torch.arange(
                z_col_idxs[z].numel(), dtype=torch.long
            )
        z_remaps.append(remap)
    return z_masks, z_col_idxs, z_remaps


def _valid_sel_cols_for_motion(
    motion_idx: int,
    unique_motion_to_keep_yx: np.ndarray,
    ref_d: np.ndarray,
    ref_r: np.ndarray,
    ref_c: np.ndarray,
    sel_pix_idxs: np.ndarray,
    num_fast_zs: int,
    dmd_pixels_per_column: int,
    dmd_pixels_per_row: int,
    psf2d: np.ndarray,
) -> np.ndarray:
    """Return a per-selected-pixel mask of columns valid for one motion bin.

    The valid region is the PSF-dilated reference support minus its
    horizontally-eroded border (erosion width ``2 * psf_w - 1``), evaluated at
    this bin's motion shift.
    """
    plane_size = dmd_pixels_per_column * dmd_pixels_per_row
    valid_pix_mask = np.zeros(
        (num_fast_zs, dmd_pixels_per_column, dmd_pixels_per_row), dtype=bool
    )
    valid_pix_mask[
        ref_d,
        ref_r + int(unique_motion_to_keep_yx[motion_idx, 0]),
        ref_c + int(unique_motion_to_keep_yx[motion_idx, 1]),
    ] = True
    valid_pix_mask = ndimage.binary_dilation(
        valid_pix_mask,
        structure=np.ones((1, psf2d.shape[0], psf2d.shape[1]), dtype=bool),
    )
    valid_pix_mask = ndimage.binary_erosion(
        valid_pix_mask,
        # TODO: check if dimensions here should change
        structure=np.ones(
            (1, psf2d.shape[0], psf2d.shape[1] * 2 - 1), dtype=bool
        ),
    )
    valid_pix_idxs = np.flatnonzero(valid_pix_mask)
    valid_lookup = np.zeros(num_fast_zs * plane_size, dtype=bool)
    valid_lookup[valid_pix_idxs] = True
    return valid_lookup[sel_pix_idxs]


def compute_rho(
    residual: np.ndarray,
    mot_inds_yx: np.ndarray,
    unique_motion_to_keep_yx: np.ndarray,
    h_mots: List[torch.Tensor],
    d_mats: List[torch.Tensor],
    d_mats_expanded: List[torch.Tensor],
    sel_pix_idxs: np.ndarray,
    ref_d: np.ndarray,
    ref_r: np.ndarray,
    ref_c: np.ndarray,
    num_fast_zs: int,
    dmd_pixels_per_column: int,
    dmd_pixels_per_row: int,
    psf2d: np.ndarray,
    verbose: bool = False,
) -> np.ndarray:
    """Compute the ``rho`` matched-filter response on the selected pixels.

    For each motion bin and z-plane, restricts ``H`` to that plane, forms the
    normalized ``HD`` and ``HD_expanded`` projections, and projects the
    residual onto ``HD - HD_expanded`` over the plane's valid columns.

    Parameters
    ----------
    residual : ndarray of shape (n_superpixels, n_frames)
        Noise-normalized residual from :func:`compute_residual`.
    mot_inds_yx : ndarray of shape (n_frames,)
        Per-frame motion-bin index (``-1`` for dropped frames).
    unique_motion_to_keep_yx : ndarray of shape (n_yx_bins, 2)
        Kept 2-D motion vectors.
    h_mots : list of torch.Tensor
        Per-motion sparse ``H`` from :func:`build_motion_h_matrices`.
    d_mats, d_mats_expanded : list of torch.Tensor
        Per-plane base/expanded convolution matrices from
        :func:`build_convolution_matrix`.
    sel_pix_idxs : ndarray of int
        Sorted flat selected-pixel indices.
    ref_d, ref_r, ref_c : ndarray
        Per-superpixel reference depth/row/column indices.
    num_fast_zs, dmd_pixels_per_column, dmd_pixels_per_row : int
        Grid geometry.
    psf2d : ndarray
        The (cropped) PSF for this DMD.
    verbose : bool
        Show a per-motion-bin progress bar when set.

    Returns
    -------
    ndarray of shape (n_sel, n_frames), float32
        The ``rho`` response (NaN where not computed).
    """
    plane_size = dmd_pixels_per_column * dmd_pixels_per_row
    z_masks, z_col_idxs, z_remaps = _plane_column_maps(
        sel_pix_idxs, num_fast_zs, plane_size
    )
    z_col_idxs_np = [zi.numpy() for zi in z_col_idxs]

    rho = np.full(
        (len(sel_pix_idxs), residual.shape[1]), np.nan, dtype=np.float32
    )

    for i in progress(
        range(len(unique_motion_to_keep_yx)),
        desc="Computing rho",
        verbose=verbose,
    ):
        motion_frames = np.flatnonzero(mot_inds_yx == i)
        if motion_frames.size == 0:
            continue

        h_mot = h_mots[i].coalesce()
        h_idxs = h_mot.indices()
        h_vals = h_mot.values()
        nrows, _ = h_mot.size()
        residual_motion = residual[:, motion_frames].T

        valid_sel_cols = _valid_sel_cols_for_motion(
            i,
            unique_motion_to_keep_yx,
            ref_d,
            ref_r,
            ref_c,
            sel_pix_idxs,
            num_fast_zs,
            dmd_pixels_per_column,
            dmd_pixels_per_row,
            psf2d,
        )

        for z in range(num_fast_zs):
            new_ncols = int(z_col_idxs[z].numel())
            if new_ncols == 0:
                continue

            z_mask = z_masks[z]
            keep_mask = z_mask[h_idxs[1]]
            if keep_mask.sum().item() == 0:
                # Unreachable defensive guard (mirrors the reference): a plane
                # with selected pixels (new_ncols > 0) always has H columns in
                # it, since both derive from the same in-plane PSF footprint.
                continue  # pragma: no cover

            remap = z_remaps[z]
            new_rows = h_idxs[0, keep_mask]
            new_cols = remap[h_idxs[1, keep_mask]]
            new_idxs = torch.stack([new_rows, new_cols], dim=0)
            new_vals = h_vals[keep_mask]
            h_sub = torch.sparse_coo_tensor(
                new_idxs, new_vals, (nrows, new_ncols), dtype=torch.float32
            ).coalesce()

            hd = torch.sparse.mm(h_sub, d_mats[z])
            hd = hd / torch.sum(hd, dim=0, keepdim=True)
            hd_expanded = torch.sparse.mm(h_sub, d_mats_expanded[z])
            hd_expanded = hd_expanded / torch.sum(
                hd_expanded, dim=0, keepdim=True
            )
            hd_diff = hd - hd_expanded

            z_cols = z_col_idxs_np[z]
            z_valid_mask = valid_sel_cols[z_cols]
            if not np.any(z_valid_mask):
                continue
            z_valid_cols = z_cols[z_valid_mask]
            local_valid_cols = remap[torch.from_numpy(z_valid_cols)].long()

            rho[np.ix_(z_valid_cols, motion_frames)] = (
                residual_motion @ hd_diff[:, local_valid_cols].numpy()
            ).T

    return rho


def mask_high_nan_rho(rho: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """NaN-out rows of ``rho`` that are mostly NaN and return the NaN fraction.

    ``rho`` is modified in place. Rows whose NaN fraction exceeds ``thresh``
    are set entirely to NaN.

    Parameters
    ----------
    rho : ndarray of shape (n_sel, n_frames)
        The rho response.
    thresh : float
        NaN-fraction threshold above which a row is dropped.

    Returns
    -------
    ndarray of shape (n_sel,)
        Per-row NaN fraction (before dropping).
    """
    nan_ct = np.mean(np.isnan(rho), axis=1)
    rho[nan_ct > thresh] = np.nan
    return nan_ct


def decay_kernel_1d(decay_tau_s: float, align_hz: float) -> np.ndarray:
    """Build the normalized causal exponential-decay smoothing kernel.

    Parameters
    ----------
    decay_tau_s : float
        Calcium-decay time constant, in seconds.
    align_hz : float
        Alignment/analysis rate, in Hz.

    Returns
    -------
    ndarray
        A 1-D kernel spanning ``3 * tau`` frames, summing to 1.
    """
    decay_tau_frames = decay_tau_s * align_hz
    k1d = np.exp(
        np.linspace(
            -np.ceil(decay_tau_frames * 3),
            0,
            int(np.ceil(decay_tau_frames * 3) + 1),
        )
        / decay_tau_frames
    )
    return k1d / np.sum(k1d)


def smooth_rho(
    rho: np.ndarray, k1d: np.ndarray, verbose: bool = False
) -> np.ndarray:
    """NaN-aware temporal smoothing of ``rho`` with a 1-D decay kernel.

    Convolves along time only, dividing the smoothed values by the smoothed
    valid-sample count and NaN-ing samples whose effective count drops below
    ``0.75``. Rows are processed in chunks to bound temporary memory.
    ``rho`` is modified in place and also returned.

    Parameters
    ----------
    rho : ndarray of shape (n_sel, n_frames)
        The rho response (NaN allowed).
    k1d : ndarray
        1-D smoothing kernel from :func:`decay_kernel_1d`.
    verbose : bool
        Show a per-row-chunk progress bar when set.

    Returns
    -------
    ndarray
        The smoothed ``rho``.
    """
    k2d = np.expand_dims(k1d, 0)
    n_time = rho.shape[1]
    bytes_per_row = max(1, n_time) * np.dtype(np.float32).itemsize * 3
    row_chunk = max(64, min(4096, int(200_000_000 // bytes_per_row)))

    for r0 in progress(
        range(0, rho.shape[0], row_chunk),
        desc="Smoothing rho",
        verbose=verbose,
    ):
        r1 = min(r0 + row_chunk, rho.shape[0])
        rc = rho[r0:r1].copy()
        row_has_data = np.any(np.isfinite(rc), axis=1)
        if not np.any(row_has_data):
            rho[r0:r1] = rc
            continue

        rc_valid = rc[row_has_data]
        rho_num = np.nan_to_num(rc_valid, nan=0.0)
        rho_den = signal.convolve(
            np.isfinite(rc_valid).astype(np.float32), k2d, mode="same"
        )
        rho_num = signal.convolve(rho_num, k2d, mode="same")
        valid_den = rho_den > 0.75
        np.divide(rho_num, rho_den, out=rho_num, where=valid_den)
        rho_num[~valid_den] = np.nan
        rc[row_has_data] = rho_num
        rho[r0:r1] = rc

    return rho
