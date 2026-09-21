"""Matched-filter activity, valid-column masking, and temporal smoothing."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import scipy.ndimage as ndimage
import torch
from scipy import signal

from ...progress import progress


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
            # Center-surround matched filter. Both projections are
            # column-normalized before subtracting, so ``hd_diff`` has
            # zero-sum columns and is blind to spatially flat residual.
            hd_expanded = torch.sparse.mm(h_sub, d_mats_expanded[z])
            hd_expanded = hd_expanded / torch.sum(
                hd_expanded, dim=0, keepdim=True
            )
            hd_diff = hd - hd_expanded
            # TEMPORARY: whiten each filter column to unit L2 norm so every
            # selected pixel's rho has the same noise variance. The L1
            # normalization above pins each column's *sum*, not its norm, so
            # noise gain varies with where a column sits relative to the
            # sparse superpixel reference-pixel lattice. Since the activity
            # image accumulates rho^2 only at local maxima, that bias is
            # amplified into a visible grid of bright reference pixels.
            # Whitening also makes ``peak_th`` a true sigma threshold.
            # To revert, delete the two lines below.
            hd_norm = torch.linalg.norm(hd_diff, dim=0, keepdim=True)
            hd_diff = hd_diff / hd_norm.clamp_min(1e-12)

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
