"""Band background interpolation, rolling baseline, and noise estimation."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

try:  # pragma: no cover - exercised via the installed extra
    import bottleneck as bn
except ImportError:  # pragma: no cover
    bn = None


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


# Preserve scientific parity with the reference interpolation branches.
def _interp_columns(  # noqa: C901
    out,
    rows_by_col,
    idxs_by_col,
    sel_idxs_by_col_all,
    sel_rows,
    data,
    frames,
    dtype,
    method="linear",
    spline_weight_cache=None,
):
    """Interpolate each multi-point column onto its selected pixels.

    Selected rows that coincide with a reference row are copied exactly; those
    strictly between reference rows use linear or natural-cubic interpolation;
    those outside the sampled row range are left untouched (NaN). Cubic spline
    weights depend only on row geometry, so they are computed once and applied
    to all complete frames in one matrix multiplication. Linear interpolation
    is retained for columns with fewer than three samples and frames containing
    missing source values.
    """
    if method not in {"linear", "cubic"}:
        raise ValueError(
            "interpolation method must be 'linear' or 'cubic', "
            f"got {method!r}"
        )
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

        target_indices = col_sel_pix_idxs[interp_rows]
        linear_frame_mask = np.ones(frames.size, dtype=bool)
        has_cubic_geometry = rows.size >= 3 and np.all(np.diff(rows) > 0)
        if method == "cubic" and has_cubic_geometry:
            source_values = data[src_idx][:, frames]
            complete_frames = np.all(np.isfinite(source_values), axis=0)
            if np.any(complete_frames):
                source_rows = rows.astype(dtype, copy=False)
                target_interp_rows = target_rows[interp_rows].astype(
                    dtype, copy=False
                )
                origin = source_rows[0]
                cache_key = (
                    tuple(source_rows - origin),
                    tuple(target_interp_rows - origin),
                    np.dtype(dtype).str,
                )
                spline_weights = (
                    spline_weight_cache.get(cache_key)
                    if spline_weight_cache is not None
                    else None
                )
                if spline_weights is None:
                    spline_weights = CubicSpline(
                        source_rows,
                        np.eye(rows.size, dtype=dtype),
                        axis=0,
                        bc_type="natural",
                        extrapolate=False,
                    )(target_interp_rows).astype(dtype, copy=False)
                    if spline_weight_cache is not None:
                        spline_weight_cache[cache_key] = spline_weights
                cubic_values = (
                    spline_weights @ source_values[:, complete_frames]
                )
                out[np.ix_(target_indices, frames[complete_frames])] = (
                    cubic_values
                )
                linear_frame_mask = ~complete_frames

        if not np.any(linear_frame_mask):
            continue
        linear_frames = frames[linear_frame_mask]
        vals_lo = data[src_idx[lo_v]][:, linear_frames]
        vals_hi = data[src_idx[hi_v]][:, linear_frames]
        linear_values = (1.0 - alpha) * vals_lo + alpha * vals_hi
        out[np.ix_(target_indices, linear_frames)] = linear_values


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
    method: str = "cubic",
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
    method : {"linear", "cubic"}
        Interpolation along each column (default ``"cubic"``). Cubic mode uses
        a natural spline with precomputed geometry weights and falls back to
        linear for columns with fewer than three samples or frames with missing
        samples.

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
    if method not in {"linear", "cubic"}:
        raise ValueError(
            "interpolation method must be 'linear' or 'cubic', "
            f"got {method!r}"
        )

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
    spline_weight_cache = {} if method == "cubic" else None

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
            method,
            spline_weight_cache,
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
    """Compute a NaN-aware rolling-median baseline of the interpolated data.

    Each frame's baseline is the median of its centered window, ignoring NaNs,
    so windows with no valid samples stay NaN and edge windows are truncated to
    the available samples. This is the fast moving-median analogue of the
    former moving-mean baseline; the median is more robust to transient
    activity spikes when estimating the background.

    The heavy lifting uses :func:`bottleneck.move_median` (a C sliding-window
    median whose cost is close to the moving mean) when available, falling back
    to pandas' skip-list rolling median otherwise. ``bottleneck`` computes a
    *trailing* window, so the data is right-padded with NaNs by ``window // 2``
    and the result is sliced back to center each window on its frame.

    Parameters
    ----------
    interp_data : ndarray of shape (n_sel, n_frames)
        Interpolated background estimate (from :func:`build_interp_data`).
    baseline_window : int
        Rolling window length, in frames.

    Returns
    -------
    ndarray
        The rolling-median background estimate (NaN where no valid samples).
    """
    n_frames = interp_data.shape[1]
    window = int(min(max(baseline_window, 1), n_frames))

    if bn is not None:
        pad = window // 2
        if pad:
            padded = np.concatenate(
                [
                    interp_data,
                    np.full(
                        (interp_data.shape[0], pad),
                        np.nan,
                        dtype=interp_data.dtype,
                    ),
                ],
                axis=1,
            )
        else:
            padded = interp_data
        trailing = bn.move_median(padded, window=window, min_count=1, axis=1)
        background = trailing[:, pad : pad + n_frames]
        return background.astype(interp_data.dtype, copy=False)

    # Fallback (no bottleneck): pandas rolls along the row axis, so transpose
    # to (n_frames, n_sel), roll each column, and transpose back.
    background = (
        pd.DataFrame(interp_data.T)
        .rolling(window=window, center=True, min_periods=1)
        .median()
        .to_numpy()
        .T
    )
    return background.astype(interp_data.dtype, copy=False)


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
            background[np.ix_(valid_rows, motion_frames)] = (
                interp_data_background[
                    np.ix_(bg_idxs[valid_rows], motion_frames)
                ]
            )

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
