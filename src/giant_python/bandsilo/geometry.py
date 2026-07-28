"""DMD geometry, reference-stack/PSF loading, and sparse projection matrices.

Ported from the setup blocks of ``extractSLAP2IntegrationSources.py``:
``ref_pixs_to_drc`` (flat pixel -> depth/column/row), the band-registration
lookup-table reader, ``subsampleMatrixInds`` construction, reference-stack
loading, PSF loading/cropping (from bundled ``assets/psfs/dil-NN.tif``), and
the sparse ``H`` PSF-convolution matrix used to project image space into
superpixel space.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import tifffile

from .hdf5 import load_struct_from_h5


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


def load_lookup_table(path: Union[str, Path], n_dmds: int) -> dict:
    """Load the band-registration lookup table (``bandRegLookupTable.h5``).

    Parameters
    ----------
    path : str or Path
        Path to the lookup-table HDF5 file.
    n_dmds : int
        Number of DMD paths.

    Returns
    -------
    dict
        ``{"allSuperPixelIDs": {...}, "sparseMaskInds": {...},
        "fastZ2RefZ": {...}}``, each an inner dict keyed ``DMD{N}``.
    """
    lt = load_struct_from_h5(path)
    all_super_pixel_ids = {}
    sparse_mask_inds = {}
    fastz_to_refz = {}
    for d in range(n_dmds):
        p = lt.get(f"Path{d + 1}", lt.get(f"DMD{d + 1}"))
        key = f"DMD{d + 1}"
        all_super_pixel_ids[key] = (
            np.asarray(p["allSuperPixelIDs"]).reshape(-1, 1).astype(np.int32)
        )
        sparse_mask_inds[key] = np.asarray(p["sparseMaskInds"]).astype(
            np.int32
        )
        fastz_to_refz[key] = (
            np.asarray(p["fastZ2RefZ"]).reshape(-1, 1).astype(np.int32)
        )
    return {
        "allSuperPixelIDs": all_super_pixel_ids,
        "sparseMaskInds": sparse_mask_inds,
        "fastZ2RefZ": fastz_to_refz,
    }


def build_subsample_matrix_inds(
    all_super_pixel_ids: np.ndarray,
    sparse_mask_inds: np.ndarray,
) -> np.ndarray:
    """Build the superpixel -> reference-pixel index map for one DMD.

    For each superpixel, picks a reference open pixel: if the number of open
    pixels is odd, the median pixel value is used; if even, the middle element
    (by sparse-mask order) is used.

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
        if len(open_pixs) % 2 == 1:
            ref_pix = int(np.median(open_pixs))
        else:
            ref_pix = open_pixs[int(np.floor(len(open_pixs) / 2))]
        out[sp, 0] = ref_pix
        out[sp, 1] = sp + 1
    return out


def _ref_stack_group(ref_stack: dict, dmd_ix: int):
    """Return the ref_stack subgroup for a DMD (``Path{N}`` or ``DMD{N}``)."""
    return ref_stack.get(
        f"Path{dmd_ix + 1}", ref_stack.get(f"DMD{dmd_ix + 1}")
    )


def find_reference_file(
    datadr: Union[str, Path], dmd_ix: int
) -> Optional[str]:
    """Find the REFERENCE tif for a DMD under ``datadr`` (recursive glob).

    Parameters
    ----------
    datadr : str or Path
        Raw-data directory.
    dmd_ix : int
        0-based DMD index.

    Returns
    -------
    str or None
        Path to the first matching REFERENCE tif, or ``None`` if none found.
    """
    patterns = (
        f"**/*DMD{dmd_ix + 1}_CONFIG2-REFERENCE*",
        f"**/*DMD{dmd_ix + 1}-REFERENCE*",
    )
    for pattern in patterns:
        matches = list(Path(datadr).glob(pattern))
        if matches:
            return str(matches[0])
    return None


def load_reference_stack(
    datadr: Union[str, Path],
    ref_stack_meta: dict,
    dmd_ix: int,
) -> Tuple[Optional[np.ndarray], np.ndarray, Optional[str]]:
    """Load the band reference stack for one DMD.

    Reads the REFERENCE tif from ``datadr`` but takes the channel list from the
    trial table's embedded ref_stack metadata. The raw counts are scaled by
    ``1/100`` and reshaped to ``[channels, z, y, x]``.

    Parameters
    ----------
    datadr : str or Path
        Raw-data directory.
    ref_stack_meta : dict
        The trial table's ``ref_stack`` sub-struct.
    dmd_ix : int
        0-based DMD index.

    Returns
    -------
    ref_stack : ndarray or None
        ``[channels, z, y, x]`` stack, or ``None`` if no REFERENCE file found.
    channels : ndarray
        Channel indices for this DMD.
    ref_file : str or None
        The REFERENCE file used, or ``None``.
    """
    grp = _ref_stack_group(ref_stack_meta, dmd_ix)
    channels = np.asarray(grp["channels"]).reshape(-1)
    num_channels = len(channels)

    ref_file = find_reference_file(datadr, dmd_ix)
    if ref_file is None:
        return None, channels, None

    ref = tifffile.imread(ref_file) / 100
    ref = ref.reshape(-1, num_channels, ref.shape[1], ref.shape[2])
    ref = ref.transpose(1, 0, 2, 3)
    return ref, channels, ref_file


def default_psf(dilation: int) -> np.ndarray:
    """Load a bundled default PSF template (``assets/psfs/dil-NN.tif``).

    Parameters
    ----------
    dilation : int
        Dilation size selecting ``dil-{dilation:02d}.tif`` (e.g. ``17``).

    Returns
    -------
    ndarray of float32
        The PSF template image.
    """
    name = f"dil-{int(dilation):02d}.tif"
    resource = (
        resources.files("giant_python")
        .joinpath("assets")
        .joinpath("psfs")
        .joinpath(name)
    )
    with resources.as_file(resource) as p:
        return tifffile.imread(str(p)).astype(np.float32)


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


def load_psf(dilation: int, n_dmds: int) -> dict:
    """Load per-DMD PSFs from bundled assets, thresholded and cropped.

    Each DMD uses the same ``dil-NN.tif`` template from
    ``assets/psfs/``, center-padded to a common size when shapes differ.

    Parameters
    ----------
    dilation : int
        Dilation size selecting the bundled PSF template.
    n_dmds : int
        Number of DMD paths.

    Returns
    -------
    dict
        ``{"DMD{N}": psf2d}`` thresholded/cropped PSFs.
    """
    psfs = [default_psf(dilation) for _ in range(n_dmds)]
    psf_combined = build_combined_psf(psfs)
    return {
        f"DMD{i + 1}": threshold_and_crop_psf(psf_combined[i])
        for i in range(n_dmds)
    }


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
