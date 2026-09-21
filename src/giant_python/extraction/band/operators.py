"""PSF kernels and sparse/dense band projection operators."""

from __future__ import annotations

from typing import Any, List, Tuple, cast

import numpy as np
import torch
from scipy.interpolate import RectBivariateSpline

from .geometry import ref_pixs_to_drc


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


def shrink_psf(
    psf2d: np.ndarray, scale_y: float = 1.0, scale_x: float = 0.75
) -> np.ndarray:
    """Spatially shrink a PSF on the same grid (TEMPORARY).

    Resamples the PSF on a grid whose coordinates are divided by the per-axis
    scale so a ``scale < 1`` narrows the PSF's spatial footprint along that
    axis while keeping the same array shape and center; the result is
    re-normalized to sum to 1. Used only for the temporary activity-image
    experiment.

    Parameters
    ----------
    psf2d : ndarray
        The (cropped) PSF for this DMD.
    scale_y : float
        Spatial shrink factor along the row (y) axis (``< 1`` shrinks,
        ``1`` is a no-op).
    scale_x : float
        Spatial shrink factor along the column (x) axis (``< 1`` shrinks,
        ``1`` is a no-op).

    Returns
    -------
    ndarray of shape ``psf2d.shape``, float32
        The shrunk, sum-normalized PSF.
    """
    psf_arr = np.asarray(psf2d, dtype=np.float32)
    psf_shape = psf_arr.shape
    center_y = (psf_shape[0] - 1) / 2
    center_x = (psf_shape[1] - 1) / 2
    orig_y = np.arange(psf_shape[0], dtype=np.float64) - center_y
    orig_x = np.arange(psf_shape[1], dtype=np.float64) - center_x
    interp_spline = RectBivariateSpline(orig_y, orig_x, psf_arr)
    shrunk = interp_spline(orig_y / scale_y, orig_x / scale_x)
    shrunk = np.clip(shrunk, 0.0, None).astype(np.float32)
    total = shrunk.sum()
    if total > 0:
        shrunk = shrunk / total
    return shrunk


def gaussian_kernel_2d(
    sigma: float, truncate: float = 4.0
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Build a sum-normalized isotropic 2-D Gaussian kernel (TEMPORARY).

    Used by the temporary difference-of-Gaussians activity-image experiment,
    which replaces the PSF-derived ``D``/``D_expanded`` convolution matrices
    with an isotropic center/surround pair. The subtraction itself still
    happens in :func:`compute_rho`, after per-column normalization.

    Parameters
    ----------
    sigma : float
        Gaussian standard deviation, in pixels.
    truncate : float
        Kernel half-width, in standard deviations. Truncation lowers the
        realized standard deviation slightly, so the default is generous.

    Returns
    -------
    kernel : torch.Tensor
        Sum-normalized ``(2 * radius + 1, 2 * radius + 1)`` kernel.
    center : tuple of int
        ``(row, col)`` center index of ``kernel``.
    """
    radius = int(np.ceil(truncate * sigma))
    coords = np.arange(-radius, radius + 1, dtype=np.float64)
    g_1d = np.exp(-(coords**2) / (2.0 * sigma**2))
    kernel = np.outer(g_1d, g_1d).astype(np.float32)
    kernel /= kernel.sum()
    return torch.from_numpy(kernel), (radius, radius)


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
    h_mots: List[torch.Tensor | None] = [None] * n_mot
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
        # PyTorch accepts NumPy COO arrays despite its narrower stubs.
        h_mots[i] = torch.sparse_coo_tensor(
            cast(Any, shifted_inds),
            cast(Any, sparse_h_vals),
            (num_super_pixels, sel_pix_idxs.shape[0]),
            dtype=torch.float32,
        )
    # Every slot is populated by the corresponding motion shift above.
    return cast(List[torch.Tensor], h_mots)


def build_sparse_h(
    subsample_matrix_inds: np.ndarray,
    psf2d: np.ndarray,
    dmd_pixels_per_column: int,
    dmd_pixels_per_row: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the sparse ``H`` PSF-convolution matrix (COO index/value arrays).

    ``H`` projects image space (flattened ``z*(rows*cols) + row*cols + col``)
    into superpixel space by convolving each superpixel's reference pixel with
    the PSF.

    Parameters
    ----------
    subsample_matrix_inds : ndarray of shape (n_superpixels, 2)
        ``[ref_pixel (0-based), superpixel_id (1-based)]`` per superpixel.
    psf2d : ndarray
        The (thresholded/cropped) PSF for this DMD.
    dmd_pixels_per_column, dmd_pixels_per_row : int
        DMD plane geometry.

    Returns
    -------
    sparse_h_inds : ndarray of shape (2, nnz), int32
        Row (superpixel) / column (image-pixel) indices of non-zero entries.
    sparse_h_vals : ndarray of shape (nnz,), float32
        The corresponding PSF weights.
    """
    ref_d, ref_c, ref_r = ref_pixs_to_drc(
        subsample_matrix_inds[:, 0], dmd_pixels_per_column, dmd_pixels_per_row
    )
    psf2d = np.asarray(psf2d, dtype=np.float32)
    psf_h, psf_w = psf2d.shape
    filter_size = psf_h * psf_w
    plane_size = dmd_pixels_per_column * dmd_pixels_per_row
    num_super_pixels = subsample_matrix_inds.shape[0]

    row_offsets = np.arange(psf_h, dtype=np.int32) - (psf_h // 2)
    col_offsets = np.arange(psf_w, dtype=np.int32) - (psf_w // 2)
    row_offsets_flat = np.broadcast_to(
        row_offsets.reshape(-1, 1), (psf_h, psf_w)
    ).ravel()
    col_offsets_flat = np.broadcast_to(
        col_offsets.reshape(1, -1), (psf_h, psf_w)
    ).ravel()
    psf_vals_flat = psf2d.ravel()

    sparse_h_inds = np.zeros(
        (2, num_super_pixels * filter_size), dtype=np.int32
    )
    sparse_h_vals = np.zeros(
        (num_super_pixels * filter_size,), dtype=np.float32
    )
    sparse_h_inds[0] = np.repeat(subsample_matrix_inds[:, 1] - 1, filter_size)
    for sp in range(num_super_pixels):
        start = sp * filter_size
        end = (sp + 1) * filter_size
        rows = ref_r[sp] + row_offsets_flat
        cols = ref_c[sp] + col_offsets_flat
        sparse_h_inds[1, start:end] = (
            ref_d[sp] * plane_size + rows * dmd_pixels_per_row + cols
        ).astype(np.int32, copy=False)
        sparse_h_vals[start:end] = psf_vals_flat

    non_zero_mask = sparse_h_vals != 0
    return sparse_h_inds[:, non_zero_mask], sparse_h_vals[non_zero_mask]
