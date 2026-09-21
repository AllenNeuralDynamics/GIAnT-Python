"""Pure DMD pixel geometry, selected-pixel support and PSF padding/cropping.

Loading belongs to band.inputs; sparse projection to band.operators.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.ndimage as ndimage


def ref_pixs_to_drc(
    ref_pixs: np.ndarray,
    dmd_pixels_per_column: int,
    dmd_pixels_per_row: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map flat reference-pixel indices to DMD (depth, column, row) indices.

    Parameters
    ----------
    ref_pixs : ndarray of int
        Flat reference-pixel indices.
    dmd_pixels_per_column, dmd_pixels_per_row : int
        DMD geometry.

    Returns
    -------
    ref_d, ref_c, ref_r : ndarray of int32
        Depth, column, and row indices.
    """
    ref_pixs = np.asarray(ref_pixs, dtype=np.int64)
    plane = int(dmd_pixels_per_column) * int(dmd_pixels_per_row)
    npc = int(dmd_pixels_per_column)
    ref_d = np.floor_divide(ref_pixs, plane).astype(np.int32)
    ref_c = np.floor_divide(
        ref_pixs - ref_d.astype(np.int64) * plane, npc
    ).astype(np.int32)
    ref_r = np.mod(ref_pixs, npc).astype(np.int32)
    return ref_d, ref_c, ref_r


def build_subsample_matrix_inds(
    all_super_pixel_ids: np.ndarray,
    sparse_mask_inds: np.ndarray,
) -> np.ndarray:
    """Build the superpixel -> reference-pixel index map for one DMD.

    For each superpixel, picks the element at index ``len(open_pixs) // 2``
    in sparse-mask order, for both odd and even counts; values are not sorted.

    Parameters
    ----------
    all_super_pixel_ids : ndarray of shape (n_superpixels, 1)
        Superpixel ids for this DMD.
    sparse_mask_inds : ndarray of shape (N, 2)
        ``[open_pixel (1-based), superpixel_id (1-based)]`` rows.

    Returns
    -------
    ndarray of shape (n_superpixels, 2), int32
        ``[ref_pixel (0-based), superpixel_id (1-based)]`` per superpixel.
    """
    num_super_pixels = all_super_pixel_ids.shape[0]
    out = np.zeros((num_super_pixels, 2), dtype=np.int32)
    for sp in range(num_super_pixels):
        inds = np.where(sparse_mask_inds[:, 1] == sp + 1)[0]
        open_pixs = sparse_mask_inds[inds, 0] - 1
        ref_pix = open_pixs[int(np.floor(len(open_pixs) / 2))]
        out[sp, 0] = ref_pix
        out[sp, 1] = sp + 1
    return out


def _pad_to(psf2d: np.ndarray, height: int, width: int) -> np.ndarray:
    """Center-pad a 2-D PSF to ``(height, width)`` with its minimum value."""
    ph, pw = psf2d.shape
    pad_r = (height - ph) // 2
    pad_c = (width - pw) // 2
    return np.pad(
        psf2d,
        ((pad_r, pad_r), (pad_c, pad_c)),
        constant_values=np.min(psf2d),
    )


def build_combined_psf(psfs: list) -> np.ndarray:
    """Stack per-DMD PSFs into a ``(n_dmds, H, W)`` array, center-padded."""
    height = max(p.shape[0] for p in psfs)
    width = max(p.shape[1] for p in psfs)
    combined = np.zeros((len(psfs), height, width), dtype=np.float32)
    for i, p in enumerate(psfs):
        combined[i] = _pad_to(p, height, width)
    return combined


def threshold_and_crop_psf(psf2d: np.ndarray) -> np.ndarray:
    """Zero values below ``max * exp(-3)`` and crop boundary zeros.

    Parameters
    ----------
    psf2d : ndarray
        A single-DMD PSF image.

    Returns
    -------
    ndarray of float32
        The thresholded, tightly-cropped PSF.
    """
    psf2d = psf2d.astype(np.float32, copy=True)
    psf2d[psf2d < np.max(psf2d) * np.exp(-3)] = 0
    non_zero_rows = np.any(psf2d != 0, axis=1)
    non_zero_cols = np.any(psf2d != 0, axis=0)
    row_start, row_end = np.where(non_zero_rows)[0][[0, -1]]
    col_start, col_end = np.where(non_zero_cols)[0][[0, -1]]
    row_stop = row_end + 1
    col_stop = col_end + 1
    return psf2d[row_start:row_stop, col_start:col_stop]


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
