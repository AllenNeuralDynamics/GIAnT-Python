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
value; it is then masked where data was never valid, has a local NaN-median
subtracted, and is divided by the raw local median absolute deviation (MAD).

The reference's optional ``profile_activity_map`` timing instrumentation is
intentionally dropped (it only prints timings and does not affect output).
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndimage
from scipy.stats import median_abs_deviation

from .progress import progress
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
    verbose: bool = False,
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
    verbose : bool
        Show a per-batch progress bar when set.

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

    for batch_idx in progress(
        range(num_batches), desc="Creating activity map", verbose=verbose
    ):
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
    ref_r: np.ndarray = None,
    ref_c: np.ndarray = None,
    normalize_mad: bool = True,
    ref_d: np.ndarray = None,
    phase_window_height: int = 7,
    phase_window_width: int = 11,
    phase_tolerance: float = 0.0,
    phase_min_samples: int = 5,
) -> np.ndarray:
    """Mask never-valid pixels and normalize by local NaN-median and MAD.

    Pixels whose rho was mostly NaN (``nan_ct > 0.5``) or that were never
    selected are set to NaN. When superpixel reference centers are supplied,
    an axis-aligned window includes only valid pixels with similar signed
    vertical offsets from their nearest same-column reference center.
    No snake identity or shared trajectory is assumed. Without reference
    centers, a fixed 5x11 footprint is used. First-pass statistics use
    the same local samples: ``MAD = nanmedian(abs(samples - median))``.
    The output is ``(act_im - median) / MAD``, with no Gaussian scale factor.
    Pixels with zero or non-finite MAD are set to NaN, as are masked pixels.
    With ``normalize_mad=False``, only the median is subtracted; MAD is not
    computed and zero-MAD pixels remain valid. With geometry, neighborhoods
    below ``phase_min_samples`` use the mean of finite first-pass medians
    (and MADs) in the full window, regardless of phase. Empty fallback
    windows and pixels without reference geometry remain NaN.

    Parameters
    ----------
    act_im : ndarray
        Accumulated activity image (masked in place before normalization).
    sel_pix_idxs : ndarray of int
        Flattened selected-pixel indices into ``act_im``'s grid.
    nan_ct : ndarray of shape (n_sel,)
        Per-selected-pixel rho NaN fraction from
        :func:`giant_python.bandsilo.background.mask_high_nan_rho`.
    ref_r, ref_c : ndarray, optional
        Reference-center row and column for each superpixel, expressed in
        the activity-image coordinate frame. The pipeline shifts these by
        the dominant retained motion bin as an approximation for the
        multi-motion image. Multiple bands can occupy the same column.
    normalize_mad : bool
        Divide the median-subtracted image by raw local MAD (default True).
        False skips MAD computation and normalization, retaining masking and
        median subtraction.
    ref_d : ndarray, optional
        Depth of each reference center. Required with geometry for multi-plane
        images; omitted centers belong to plane zero in single-plane images.
    phase_window_height, phase_window_width : int
        Positive odd spatial window dimensions for phase matching (7x11).
    phase_tolerance : float
        Maximum signed-phase difference, in row pixels (default 0, exact
        equality). Midpoint ties choose the smaller reference row.
    phase_min_samples : int
        Minimum finite phase-matched samples for first-pass median/MAD
        (default 5); smaller neighborhoods use the full-window fallback.

    Returns
    -------
    ndarray
        Activity in raw local-MAD units (NaN outside the valid support or
        where the local MAD is zero or non-finite), or median-subtracted
        activity in original units when ``normalize_mad=False``.
    """
    nan_mask = np.full_like(act_im, True, dtype=bool)
    valid_sel_pix = np.flatnonzero(nan_ct <= 0.5)
    nan_mask[
        np.unravel_index(sel_pix_idxs[valid_sel_pix], nan_mask.shape)
    ] = False
    act_im[nan_mask] = np.nan

    if ref_r is not None and ref_c is not None:
        # Legacy trajectory-following filter: uncomment this call and comment
        # out the phase-matched call below to switch back. The original helper
        # is retained unchanged below for comparison and regression tests.
        # local_stats = snake_aligned_nanmedian(
        #     act_im, np.flatnonzero(~nan_mask), ref_r, ref_c,
        #     return_mad=normalize_mad,
        # )
        local_stats = phase_matched_nanmedian(
            act_im,
            np.flatnonzero(~nan_mask),
            ref_r,
            ref_c,
            ref_d=ref_d,
            height=phase_window_height,
            width=phase_window_width,
            phase_tolerance=phase_tolerance,
            min_samples=phase_min_samples,
            return_mad=normalize_mad,
        )
        if normalize_mad:
            med_act_im, mad_act_im = local_stats
        else:
            med_act_im = local_stats
    else:
        med_act_im = ndimage.generic_filter(
            act_im, np.nanmedian, size=(1, 5, 11)
        )
        if normalize_mad:
            # Use each footprint's own median, not the spatial median image.
            mad_act_im = ndimage.generic_filter(
                act_im,
                median_abs_deviation,
                size=(1, 5, 11),
                extra_keywords={"nan_policy": "omit"},
            )
    act_im = act_im - med_act_im
    if not normalize_mad:
        act_im[nan_mask] = np.nan
        return act_im
    valid_scale = (~nan_mask) & np.isfinite(mad_act_im) & (mad_act_im > 0)
    np.divide(act_im, mad_act_im, out=act_im, where=valid_scale)
    act_im[~valid_scale] = np.nan
    return act_im


def phase_matched_nanmedian(
    image: np.ndarray,
    valid_pix_idxs: np.ndarray,
    ref_r: np.ndarray,
    ref_c: np.ndarray,
    ref_d: np.ndarray = None,
    height: int = 7,
    width: int = 11,
    phase_tolerance: float = 0.0,
    min_samples: int = 5,
    chunk_size: int = 4096,
    return_mad: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Local median/MAD conditioned on signed reference-band phase.

    For each selected pixel, phase is ``row - nearest_reference_row`` using
    only reference centers in the same column and depth. Distances are
    vertical, not Euclidean, so adjacent-column center spacing cannot change
    the sign/phase. Different bands may overlap in column space and have
    different curvature. Reference coordinates must already include motion.

    All nearest-center assignments are retained; exact distance ties choose
    the smaller reference row. Duplicate centers are deduplicated; exact
    coincident branches cannot be distinguished from these coordinates alone.
    Columns without centers have no phase: no interpolation across gaps or
    extrapolation beyond scanned columns is attempted. Multi-plane images
    require ``ref_d`` to avoid borrowing geometry from a different plane.

    Within a clipped, axis-aligned ``height x width`` window, take only finite
    pixels in ``valid_pix_idxs`` whose signed phase differs by at most
    ``phase_tolerance`` (default 0, exact equality). The center itself is
    included if finite. There is no
    branch-membership restriction: nearby different snakes can contribute
    when their phases match. At least ``min_samples`` (default 5) are required
    for first-pass statistics. Otherwise, use the nanmean of finite first-pass
    medians in the full spatial window, regardless of phase. Fallback values
    are never reused by other windows; an empty fallback remains NaN.
    This is a phase-conditioned spatial background, not a
    reconstruction or separation of contributions at physical crossings.

    With ``return_mad=True``, first-pass raw MAD uses exactly the median's
    samples, without Gaussian scaling. Fallback MAD is the nanmean of finite
    first-pass MADs in the full window, not a pooled-sample MAD. Return both
    images; otherwise skip MAD entirely. Non-selected/nonfinite pixels and
    pixels without geometry remain NaN. Sparse phase/support arrays and chunked
    gathers bound working memory; only output images are dense.
    """
    image = np.asarray(image)
    if image.ndim != 3 or any(size == 0 for size in image.shape):
        raise ValueError("image must have non-empty shape (depth, row, column)")
    for name, value in (("height", height), ("width", width)):
        if not isinstance(value, (int, np.integer)) or value < 1 or value % 2 == 0:
            raise ValueError(f"{name} must be a positive odd integer")
    for name, value in (("min_samples", min_samples), ("chunk_size", chunk_size)):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(phase_tolerance) or phase_tolerance < 0:
        raise ValueError("phase_tolerance must be finite and nonnegative")

    ref_r = np.asarray(ref_r, dtype=float).reshape(-1)
    ref_c = np.asarray(ref_c).reshape(-1)
    if not ref_r.size or ref_r.size != ref_c.size:
        raise ValueError("ref_r and ref_c must be non-empty and equally sized")
    if not np.all(np.isfinite(ref_r)):
        raise ValueError("reference rows must be finite")
    if ref_d is None:
        if image.shape[0] != 1:
            raise ValueError("ref_d is required for multi-plane images")
        ref_d = np.zeros(ref_r.size, dtype=np.intp)
    else:
        ref_d = np.asarray(ref_d).reshape(-1)
    if ref_d.size != ref_r.size:
        raise ValueError("ref_d must have the same size as ref_r")
    for coordinates in (ref_c, ref_d):
        if not np.all(np.isfinite(coordinates)) or np.any(coordinates != np.rint(coordinates)):
            raise ValueError("reference columns and depths must be finite integers")
    ref_c, ref_d = ref_c.astype(np.intp), ref_d.astype(np.intp)

    selected = np.asarray(valid_pix_idxs)
    if selected.ndim != 1 or (selected.size and not np.issubdtype(selected.dtype, np.integer)):
        raise ValueError("valid_pix_idxs must be a one-dimensional integer array")
    if np.any(selected < 0) or np.any(selected >= image.size):
        raise ValueError("valid_pix_idxs contains an out-of-bounds index")
    selected = np.unique(selected.astype(np.intp))
    result = np.full(image.shape, np.nan, dtype=np.result_type(image.dtype, np.float32))
    mad_result = np.full_like(result, np.nan) if return_mad else None
    if not selected.size:
        return (result, mad_result) if return_mad else result

    nz, nr, nc = image.shape
    in_bounds = (ref_d >= 0) & (ref_d < nz) & (ref_c >= 0) & (ref_c < nc)
    # Group references by (depth, column); unlike the legacy filter, do not
    # collapse all rows in a column to one shared trajectory.
    references = {}
    for depth, column, row in zip(ref_d[in_bounds], ref_c[in_bounds], ref_r[in_bounds]):
        references.setdefault(int(depth * nc + column), []).append(row)
    depths, rows, cols = np.unravel_index(selected, image.shape)
    keys = depths * nc + cols
    order = np.argsort(keys, kind="stable")
    boundaries = np.r_[0, np.flatnonzero(np.diff(keys[order])) + 1, selected.size]
    phase = np.full(selected.size, np.nan)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        positions = order[start:stop]
        centers = references.get(int(keys[positions[0]]))
        if centers is None:
            continue
        centers = np.unique(centers)
        pixel_rows = rows[positions]
        insertion = np.searchsorted(centers, pixel_rows)
        left = np.clip(insertion - 1, 0, centers.size - 1)
        right = np.clip(insertion, 0, centers.size - 1)
        nearest = np.where(
            np.abs(pixel_rows - centers[left]) <= np.abs(pixel_rows - centers[right]),
            left, right,
        )
        phase[positions] = pixel_rows - centers[nearest]

    # Lookup both support membership and phase using sorted selected indices;
    # finite unselected image pixels must never enter the sample population.
    row_offsets, col_offsets = np.meshgrid(
        np.arange(-(height // 2), height // 2 + 1),
        np.arange(-(width // 2), width // 2 + 1), indexing="ij",
    )
    row_offsets, col_offsets = row_offsets.ravel(), col_offsets.ravel()
    values = image.ravel()[selected]
    outputs = np.flatnonzero(np.isfinite(phase) & np.isfinite(values))

    def windows(output_positions):
        for start in range(0, output_positions.size, chunk_size):
            positions = output_positions[start:start + chunk_size]
            sample_r = rows[positions, None] + row_offsets
            sample_c = cols[positions, None] + col_offsets
            inside = (sample_r >= 0) & (sample_r < nr) & (sample_c >= 0) & (sample_c < nc)
            sample_flat = (depths[positions, None] * nr + sample_r) * nc + sample_c
            lookup = np.minimum(np.searchsorted(selected, sample_flat), selected.size - 1)
            yield positions, lookup, inside & (selected[lookup] == sample_flat)

    for positions, lookup, support in windows(outputs):
        accept = (
            support
            & np.isfinite(values[lookup])
            & (np.abs(phase[lookup] - phase[positions, None]) <= phase_tolerance)
        )
        enough = np.count_nonzero(accept, axis=1) >= min_samples
        if not np.any(enough):
            continue
        samples = np.where(accept[enough], values[lookup[enough]], np.nan).astype(result.dtype)
        medians = np.nanmedian(samples, axis=1)
        flat = selected[positions[enough]]
        result.ravel()[flat] = medians
        if return_mad:
            np.subtract(samples, medians[:, None], out=samples)
            np.abs(samples, out=samples)
            mad_result.ravel()[flat] = np.nanmedian(samples, axis=1)

    # Advanced indexing freezes first-pass values: fallback estimates must
    # not propagate or depend on chunk size / iteration order.
    first_medians = result.ravel()[selected]
    first_mads = mad_result.ravel()[selected] if return_mad else None
    missing = outputs[~np.isfinite(first_medians[outputs])]
    statistics = [(first_medians, result)]
    if return_mad:
        statistics.append((first_mads, mad_result))
    for positions, lookup, support in windows(missing):
        for source, destination in statistics:
            accept = support & np.isfinite(source[lookup])
            counts = np.count_nonzero(accept, axis=1)
            enough = counts > 0
            # Equivalent to nanmean, without empty-window warnings.
            totals = np.sum(np.where(accept, source[lookup], 0), axis=1, dtype=np.float64)
            destination.ravel()[selected[positions[enough]]] = totals[enough] / counts[enough]
    return (result, mad_result) if return_mad else result


def snake_aligned_nanmedian(
    image: np.ndarray,
    valid_pix_idxs: np.ndarray,
    ref_r: np.ndarray,
    ref_c: np.ndarray,
    height: int = 5,
    width: int = 11,
    chunk_size: int = 100_000,
    return_mad: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return local medians and optionally MADs along the band trajectory.

    The parallel snakes have the same trajectory and differ by a nearly fixed
    vertical spacing. The trajectory is unwrapped modulo that spacing so its
    row does not jump when an outer snake is absent from a column. Disconnected
    column runs are kept separate. For each valid output pixel, this function
    gathers ``height * width`` samples translated by the local trajectory.
    Work is restricted to valid pixels and chunked to bound temporary memory.
    ``ref_r`` and ``ref_c`` must already be expressed in ``image`` coordinates;
    this helper does not apply motion correction itself.

    With ``return_mad=True``, return ``(median_image, mad_image)``. The raw
    MAD is ``nanmedian(abs(samples - median))`` over exactly the same samples
    used for each output median, without Gaussian scaling. The sample buffer
    is reused for absolute deviations. Both images are NaN at unrequested
    pixels and all-NaN footprints; constant footprints have MAD zero.
    The default returns only the median image for compatibility.
    """
    if height < 1 or width < 1 or height % 2 == 0 or width % 2 == 0:
        raise ValueError("height and width must be positive odd integers")

    ref_r = np.asarray(ref_r).reshape(-1)
    ref_c = np.asarray(ref_c).reshape(-1).astype(np.intp, copy=False)
    if ref_r.size == 0 or ref_r.size != ref_c.size:
        raise ValueError("ref_r and ref_c must be non-empty and equally sized")

    n_rows, n_cols = image.shape[-2:]
    in_bounds = (ref_c >= 0) & (ref_c < n_cols)
    columns = np.unique(ref_c[in_bounds])
    if columns.size == 0:
        raise ValueError("no reference centers fall inside the image")

    rows_by_column = [np.sort(ref_r[ref_c == column]) for column in columns]
    spacing_samples = [
        np.diff(rows) for rows in rows_by_column if rows.size > 1
    ]
    row_spacings = (
        np.concatenate(spacing_samples)
        if spacing_samples
        else np.empty(0, dtype=float)
    )
    snake_spacing = (
        float(np.median(row_spacings[row_spacings > 0]))
        if np.any(row_spacings > 0)
        else None
    )

    run_starts = np.r_[0, np.flatnonzero(np.diff(columns) > 1) + 1]
    run_stops = np.r_[run_starts[1:], columns.size]
    center_rows = np.array([rows[0] for rows in rows_by_column], dtype=float)
    segment_at_reference = np.empty(columns.size, dtype=np.intp)
    for segment, (start, stop) in enumerate(zip(run_starts, run_stops)):
        segment_at_reference[start:stop] = segment
        if snake_spacing is None:
            center_rows[start:stop] = [
                np.median(rows) for rows in rows_by_column[start:stop]
            ]
            continue
        for index in range(start + 1, stop):
            delta = center_rows[index] - center_rows[index - 1]
            center_rows[index] -= np.rint(delta / snake_spacing) * snake_spacing

    all_columns = np.arange(n_cols)
    insertions = np.searchsorted(columns, all_columns)
    right = np.clip(insertions, 0, columns.size - 1)
    left = np.clip(insertions - 1, 0, columns.size - 1)
    nearest = np.where(
        np.abs(all_columns - columns[left])
        <= np.abs(columns[right] - all_columns),
        left,
        right,
    )
    trajectory = center_rows[nearest]
    trajectory_segment = segment_at_reference[nearest]

    result = np.full_like(image, np.nan, dtype=np.result_type(image, np.float32))
    mad_result = np.full_like(result, np.nan) if return_mad else None
    col_offsets = np.arange(-(width // 2), width // 2 + 1)
    row_offsets = np.arange(-(height // 2), height // 2 + 1)

    for start in range(0, len(valid_pix_idxs), chunk_size):
        flat = valid_pix_idxs[start : start + chunk_size]
        depths, rows, cols = np.unravel_index(flat, image.shape)
        sample_cols = cols[:, None] + col_offsets[None, :]
        valid_cols = (sample_cols >= 0) & (sample_cols < n_cols)
        clipped_cols = np.clip(sample_cols, 0, n_cols - 1)
        valid_cols &= (
            trajectory_segment[clipped_cols]
            == trajectory_segment[cols, None]
        )
        path_delta = np.rint(
            trajectory[clipped_cols] - trajectory[cols, None]
        ).astype(np.intp)
        path_rows = rows[:, None] + path_delta

        sample_rows = path_rows[:, :, None] + row_offsets[None, None, :]
        sample_cols = np.broadcast_to(
            clipped_cols[:, :, None], sample_rows.shape
        )
        sample_depths = np.broadcast_to(
            depths[:, None, None], sample_rows.shape
        )
        valid = (
            valid_cols[:, :, None]
            & (sample_rows >= 0)
            & (sample_rows < n_rows)
        )
        samples = np.full(sample_rows.shape, np.nan, dtype=result.dtype)
        samples[valid] = image[
            sample_depths[valid], sample_rows[valid], sample_cols[valid]
        ]
        medians = np.nanmedian(samples, axis=(1, 2))
        result.ravel()[flat] = medians
        if return_mad:
            np.subtract(samples, medians[:, None, None], out=samples)
            np.abs(samples, out=samples)
            mad_result.ravel()[flat] = np.nanmedian(samples, axis=(1, 2))

    return (result, mad_result) if return_mad else result
