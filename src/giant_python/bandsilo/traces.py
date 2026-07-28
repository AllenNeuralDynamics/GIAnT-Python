"""High-resolution per-trial trace extraction.

Ported from ``get_high_res_traces`` in
``extractSLAP2IntegrationSources.py`` (ref 844-1011). For one trial this reads
the high-res superpixel activity, interpolates the alignment motion/background
onto the trial's downsample grid, bins the frames by motion (matching the
low-res motion bins used during NMF), and — per motion bin — projects the fixed
spatial profiles ``A_final`` into superpixel space and least-squares-solves the
per-source temporal traces (``phi``, the least-squares dF) and the background
projection (``F0``). It also computes the global and per-user-ROI fluorescence.

The pure compute (:func:`compute_high_res_traces`) is factored out of the IO
wrapper (:func:`get_high_res_traces`, which reads the SLAP2 file lazily via
:mod:`giant_python.bandsilo.trial_data`) so the numerics are testable without
``slap2_utils``. Per-trial ``.npz`` caching and the ``mp.Pool`` fan-out are
deferred to the Phase-8 pipeline driver.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from .background import build_motion_h_matrices, build_selected_pixel_mask
from .geometry import ref_pixs_to_drc
from .nmf import solve_phi_motion
from .trial_data import nearest_interp, read_band_trial_data


def _empty_trace_result(
    n_sources: int, num_channels: int, n_soma: int
) -> tuple:
    """Return the all-empty result tuple for a skipped trial.

    Parameters
    ----------
    n_sources : int
        Number of sources (column count of the empty ``phi``/``F0``).
    num_channels : int
        Number of acquisition channels.
    n_soma : int
        Number of user/soma ROIs.

    Returns
    -------
    tuple
        Same 8-element shape as :func:`compute_high_res_traces`, with zero
        frames.
    """
    return (
        np.full((0, n_sources), np.nan, dtype=np.float32),
        np.full((0, n_sources), np.nan, dtype=np.float32),
        np.full((0,), np.nan),
        np.full((0,), np.nan),
        np.full((0, num_channels), np.nan, dtype=np.float32),
        (np.full((0,), np.nan), np.full((0,), np.nan), np.full((0,), np.nan)),
        (
            np.full((0,), 0, dtype=np.int16),
            np.full((0,), 0, dtype=np.int16),
            np.full((0,), 0, dtype=np.int16),
        ),
        np.full((0, n_soma, num_channels), np.nan, dtype=np.float32),
    )


def _interp_trial_alignment(
    frames: np.ndarray, a_data: dict, background_ds: np.ndarray
):
    """Interpolate alignment motion/background onto the trial's frame grid.

    Motion and per-superpixel background are linearly interpolated from the
    alignment downsample grid to ``frames``; the online shifts use
    nearest-neighbor interpolation.

    Parameters
    ----------
    frames : ndarray
        The trial's high-res downsample frame line positions.
    a_data : dict
        Alignment data (``DSframes``, ``motionDSr/c/z``,
        ``onlineYshift/Xshift/Zshift``).
    background_ds : ndarray of shape (n_superpixels, n_ds)
        Per-superpixel low-res background for this trial.

    Returns
    -------
    tuple
        ``(motion_r, motion_c, motion_z, background, online_y, online_x,
        online_z)``.
    """
    ds = a_data["DSframes"]
    motion_r = np.interp(frames, ds, a_data["motionDSr"])
    motion_c = np.interp(frames, ds, a_data["motionDSc"])
    motion_z = np.interp(frames, ds, a_data["motionDSz"])
    background = np.array(
        [
            np.interp(frames, ds, background_ds[i])
            for i in range(background_ds.shape[0])
        ]
    )
    online_y = nearest_interp(frames, ds, a_data["onlineYshift"])
    online_x = nearest_interp(frames, ds, a_data["onlineXshift"])
    online_z = nearest_interp(frames, ds, a_data["onlineZshift"])
    return (
        motion_r,
        motion_c,
        motion_z,
        background,
        online_y,
        online_x,
        online_z,
    )


def _bin_trial_motion(
    motion_r: np.ndarray,
    motion_c: np.ndarray,
    motion_z: np.ndarray,
    median_z: float,
    unique_motion_ds: np.ndarray,
    mot_inds_to_keep_ds: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bin trial frames by motion and match them to the low-res motion bins.

    Frames within 1.5 um of the median z are rebinned into 2-D (row, col)
    motion bins; each retained low-res bin is matched to its trial bin so the
    same PSF geometry is reused.

    Parameters
    ----------
    motion_r, motion_c, motion_z : ndarray
        Per-frame interpolated motion.
    median_z : float
        Median z across the recording (from the low-res pass).
    unique_motion_ds : ndarray of shape (n_ds_bins, 2+)
        The low-res kept 2-D motion vectors.
    mot_inds_to_keep_ds : ndarray
        Indices of the low-res bins to keep.

    Returns
    -------
    unique_motion : ndarray of shape (n_bins, 2)
        Trial 2-D motion bins (over the z-kept frames).
    mot_inds : ndarray of shape (n_frames,), int32
        Per-frame trial-bin index (``-1`` for dropped frames).
    mot_inds_to_keep : ndarray
        Trial-bin indices matching a kept low-res bin.
    frames_to_keep : ndarray of bool
        Frames belonging to a matched bin.
    """
    unique_motion3, mot_inds = np.unique(
        np.round(np.stack((motion_r, motion_c, motion_z), axis=1)),
        axis=0,
        return_inverse=True,
    )
    mot_inds = np.reshape(mot_inds, -1)
    frames_to_keep = np.isin(
        mot_inds,
        np.flatnonzero(np.abs(unique_motion3[:, 2] - median_z) <= 1.5),
    )

    mot_inds = -1 * np.ones((len(mot_inds),), dtype=np.int32)
    stacked = np.round(
        np.stack((motion_r, motion_c), axis=1)[frames_to_keep, :]
    )
    unique_motion, inv = np.unique(stacked, axis=0, return_inverse=True)
    mot_inds[frames_to_keep] = np.reshape(inv, -1)

    mot_inds_to_keep = -1 * np.ones_like(mot_inds_to_keep_ds, dtype=np.int64)
    for i, motion_idx_ds in enumerate(mot_inds_to_keep_ds):
        matches = np.flatnonzero(
            np.all(
                unique_motion[:, :2] == unique_motion_ds[motion_idx_ds, :2],
                axis=1,
            )
        )
        if len(matches) > 0:
            mot_inds_to_keep[i] = matches[0]
    mot_inds_to_keep = mot_inds_to_keep[mot_inds_to_keep != -1]

    frames_to_keep = np.isin(mot_inds, mot_inds_to_keep)
    return unique_motion, mot_inds, mot_inds_to_keep, frames_to_keep


def _solve_trial_phi_f0(
    data: np.ndarray,
    background: np.ndarray,
    unique_motion: np.ndarray,
    mot_inds: np.ndarray,
    mot_inds_to_keep: np.ndarray,
    sparse_h_inds: np.ndarray,
    sparse_h_vals: np.ndarray,
    sel_pix_idxs: np.ndarray,
    a_final: torch.Tensor,
    num_super_pixels: int,
    dmd_pixels_per_row: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Least-squares-solve per-source traces and background projection.

    For each kept motion bin, shifts ``H`` by the bin's motion, projects
    ``A_final`` into superpixel space, and solves for the source temporal
    weights (from the background-subtracted residual) and the background
    projection ``F0``.

    Parameters
    ----------
    data : ndarray of shape (n_superpixels, n_frames)
        Count-normalized activity.
    background : ndarray of shape (n_superpixels, n_frames)
        Interpolated background.
    unique_motion : ndarray of shape (n_bins, 2)
        Trial 2-D motion bins.
    mot_inds : ndarray of shape (n_frames,)
        Per-frame trial-bin index.
    mot_inds_to_keep : ndarray
        Trial-bin indices to solve.
    sparse_h_inds, sparse_h_vals : ndarray
        Base sparse ``H`` (from ``geometry.build_sparse_h``).
    sel_pix_idxs : ndarray of int
        Sorted flat selected-pixel indices.
    a_final : torch.Tensor of shape (n_pixels, n_sources)
        Fixed source spatial profiles.
    num_super_pixels : int
        Superpixel count (``H`` row count).
    dmd_pixels_per_row : int
        Grid geometry (column-shift stride).

    Returns
    -------
    phi, f0 : torch.Tensor of shape (n_frames, n_sources)
        Per-source least-squares dF and background projection (NaN where no
        motion bin applies).
    """
    n_sources = a_final.shape[1]
    n_frames = data.shape[1]
    phi = torch.full((n_frames, n_sources), float("nan"), dtype=torch.float32)
    f0 = torch.full((n_frames, n_sources), float("nan"), dtype=torch.float32)
    residual = data - background

    # One shifted superpixel<-pixel H per trial motion bin (same builder the
    # NMF pass uses); only the matched bins are solved below.
    h_mots = build_motion_h_matrices(
        sparse_h_inds,
        sparse_h_vals,
        unique_motion,
        sel_pix_idxs,
        num_super_pixels,
        dmd_pixels_per_row,
    )

    for motion_idx in mot_inds_to_keep:
        motion_frames = np.flatnonzero(mot_inds == motion_idx)
        x = torch.sparse.mm(h_mots[motion_idx], a_final[sel_pix_idxs, :])
        # phi from the background-subtracted residual; F0 from the background,
        # both via the shared regularized normal-equations solve.
        phi[motion_frames, :] = solve_phi_motion(
            x, torch.from_numpy(residual[:, motion_frames].astype(np.float32))
        )
        f0[motion_frames, :] = solve_phi_motion(
            x,
            torch.from_numpy(background[:, motion_frames].astype(np.float32)),
        )

    return phi, f0


def _compute_global_f(
    data: np.ndarray,
    data2: Optional[np.ndarray],
    frames_to_keep: np.ndarray,
    num_channels: int,
) -> np.ndarray:
    """Sum fluorescence over all superpixels per channel and frame.

    Parameters
    ----------
    data : ndarray of shape (n_superpixels, n_frames)
        Channel-1 activity.
    data2 : ndarray or None
        Channel-2 activity (used when ``num_channels >= 2``).
    frames_to_keep : ndarray of bool
        Dropped frames are set to NaN.
    num_channels : int
        Number of acquisition channels.

    Returns
    -------
    ndarray of shape (n_frames, num_channels)
        Global fluorescence.
    """
    channel_sums = [np.sum(data, axis=0)]
    if num_channels >= 2:
        channel_sums.append(np.sum(data2, axis=0))
    global_f = np.stack(channel_sums, axis=-1)
    global_f[~frames_to_keep, :] = np.nan
    return global_f


def _compute_f_soma(
    data: np.ndarray,
    data2: Optional[np.ndarray],
    soma_sps: List[np.ndarray],
    frames_to_keep: np.ndarray,
    num_channels: int,
) -> np.ndarray:
    """Sum fluorescence over each user-ROI's superpixels per channel.

    Parameters
    ----------
    data : ndarray of shape (n_superpixels, n_frames)
        Channel-1 activity.
    data2 : ndarray or None
        Channel-2 activity (used when ``num_channels >= 2``).
    soma_sps : list of ndarray
        Per-ROI superpixel index arrays.
    frames_to_keep : ndarray of bool
        Dropped frames are set to NaN.
    num_channels : int
        Number of acquisition channels.

    Returns
    -------
    ndarray of shape (n_frames, n_rois, num_channels)
        Per-ROI fluorescence.
    """
    n_frames = data.shape[1]
    f_soma = np.full(
        (n_frames, len(soma_sps), num_channels), np.nan, dtype=np.float32
    )
    for i, roi_sps in enumerate(soma_sps):
        f_soma[:, i, 0] = np.nansum(data[roi_sps, :], axis=0)
        if num_channels >= 2:
            f_soma[:, i, 1] = np.nansum(data2[roi_sps, :], axis=0)
    f_soma[~frames_to_keep, :, :] = np.nan
    return f_soma


def compute_high_res_traces(
    data: np.ndarray,
    data2: Optional[np.ndarray],
    background_ds: np.ndarray,
    frames: np.ndarray,
    a_data: dict,
    subsample_matrix_inds: np.ndarray,
    sparse_h_inds: np.ndarray,
    sparse_h_vals: np.ndarray,
    a_final: torch.Tensor,
    unique_motion_ds: np.ndarray,
    mot_inds_to_keep_ds: np.ndarray,
    median_z: float,
    psf2d: np.ndarray,
    num_super_pixels: int,
    num_fast_zs: int,
    dmd_pixels_per_column: int,
    dmd_pixels_per_row: int,
    num_channels: int,
    soma_sps: List[np.ndarray],
) -> tuple:
    """Compute one trial's high-res source traces from loaded arrays.

    Parameters
    ----------
    data : ndarray of shape (n_superpixels, n_frames)
        Count-normalized channel-1 activity.
    data2 : ndarray or None
        Count-normalized channel-2 activity.
    background_ds : ndarray of shape (n_superpixels, n_ds)
        Per-superpixel low-res background for this trial.
    frames : ndarray
        The trial's downsample frame line positions.
    a_data : dict
        Alignment data for this trial.
    subsample_matrix_inds : ndarray
        Superpixel reference-pixel map (from ``geometry``).
    sparse_h_inds, sparse_h_vals : ndarray
        Base sparse ``H``.
    a_final : torch.Tensor of shape (n_pixels, n_sources)
        Fixed source spatial profiles.
    unique_motion_ds : ndarray
        Low-res kept 2-D motion vectors.
    mot_inds_to_keep_ds : ndarray
        Indices of the low-res bins to keep.
    median_z : float
        Median z (from the low-res pass).
    psf2d : ndarray
        The (cropped) PSF for this DMD.
    num_super_pixels, num_fast_zs, dmd_pixels_per_column, dmd_pixels_per_row \
: int
        Geometry.
    num_channels : int
        Number of acquisition channels.
    soma_sps : list of ndarray
        Per-ROI superpixel index arrays.

    Returns
    -------
    tuple
        ``(phi, F0, frames, sel_pix_idxs, global_f, (motion_r, motion_c,
        motion_z), (online_y, online_x, online_z), F_soma)``. ``phi`` and
        ``F0`` are ``(n_frames, n_sources)`` ndarrays.
    """
    (
        motion_r,
        motion_c,
        motion_z,
        background,
        online_y,
        online_x,
        online_z,
    ) = _interp_trial_alignment(frames, a_data, background_ds)

    (
        unique_motion,
        mot_inds,
        mot_inds_to_keep,
        frames_to_keep,
    ) = _bin_trial_motion(
        motion_r,
        motion_c,
        motion_z,
        median_z,
        unique_motion_ds,
        mot_inds_to_keep_ds,
    )

    ref_d, ref_c, ref_r = ref_pixs_to_drc(
        subsample_matrix_inds[:, 0], dmd_pixels_per_column, dmd_pixels_per_row
    )
    _, sel_pix_idxs = build_selected_pixel_mask(
        unique_motion,
        ref_d,
        ref_r,
        ref_c,
        num_fast_zs,
        dmd_pixels_per_column,
        dmd_pixels_per_row,
        psf2d,
    )

    phi, f0 = _solve_trial_phi_f0(
        data,
        background,
        unique_motion,
        mot_inds,
        mot_inds_to_keep,
        sparse_h_inds,
        sparse_h_vals,
        sel_pix_idxs,
        a_final,
        num_super_pixels,
        dmd_pixels_per_row,
    )

    global_f = _compute_global_f(data, data2, frames_to_keep, num_channels)
    f_soma = _compute_f_soma(
        data, data2, soma_sps, frames_to_keep, num_channels
    )

    return (
        phi.numpy(),
        f0.numpy(),
        frames,
        sel_pix_idxs,
        global_f,
        (motion_r, motion_c, motion_z),
        (online_y, online_x, online_z),
        f_soma,
    )


def _load_high_res_trial_data(
    trial_ix: int,
    dmd_ix: int,
    samp_freq: float,
    super_pixel_ids: np.ndarray,
    datadr: str,
    trial_table: dict,
    num_channels: int,
):  # pragma: no cover - requires slap2_utils
    """Read one trial's high-res activity via the SLAP2 reader.

    Returns
    -------
    tuple
        ``(data, data2, a_data, frames)`` with count-normalized activity.
    """
    result = read_band_trial_data(
        trial_ix,
        True,
        dmd_ix,
        samp_freq,
        super_pixel_ids,
        datadr,
        trial_table,
        num_channels,
        all_channels=True,
    )
    data = result["data"] / result["data_count"]
    data2 = None
    if num_channels >= 2:
        data2 = result["data2"] / result["data_count2"]
    return data, data2, result["alignment"], result["ds_frames"]


def get_high_res_traces(
    trial_info: tuple,
    dmd_ix: int,
    samp_freq: float,
    super_pixel_ids: np.ndarray,
    datadr: str,
    trial_table: dict,
    subsample_matrix_inds: np.ndarray,
    sparse_h_inds: np.ndarray,
    sparse_h_vals: np.ndarray,
    a_final,
    unique_motion_ds: np.ndarray,
    mot_inds_to_keep_ds: np.ndarray,
    median_z: float,
    psf2d: np.ndarray,
    num_super_pixels: int,
    num_fast_zs: int,
    dmd_pixels_per_column: int,
    dmd_pixels_per_row: int,
    num_channels: int,
    soma_sps: List[np.ndarray],
) -> tuple:
    """Read and extract one trial's high-res source traces.

    Skipped trials (``keep_trial`` False) return the empty result immediately;
    otherwise the trial is read (via ``slap2_utils``) and passed to
    :func:`compute_high_res_traces`.

    Parameters
    ----------
    trial_info : tuple
        ``(trial_ix, keep_trial, background_ds)`` for this trial.
    dmd_ix : int
        0-based DMD index.
    samp_freq : float
        Analysis sample rate, in Hz.
    super_pixel_ids : ndarray
        Superpixel id lookup for this DMD.
    datadr : str
        Raw-data directory.
    trial_table : dict
        Normalized trial table.
    subsample_matrix_inds, sparse_h_inds, sparse_h_vals : ndarray
        Geometry / sparse ``H``.
    a_final : ndarray or torch.Tensor
        Fixed source spatial profiles ``(n_pixels, n_sources)``.
    unique_motion_ds, mot_inds_to_keep_ds : ndarray
        Low-res motion bins / kept indices.
    median_z : float
        Median z (from the low-res pass).
    psf2d : ndarray
        The (cropped) PSF for this DMD.
    num_super_pixels, num_fast_zs, dmd_pixels_per_column, dmd_pixels_per_row \
: int
        Geometry.
    num_channels : int
        Number of acquisition channels.
    soma_sps : list of ndarray
        Per-ROI superpixel index arrays.

    Returns
    -------
    tuple
        The 8-element result of :func:`compute_high_res_traces` (empty for
        skipped trials).
    """
    trial_ix, keep_trial, background_ds = trial_info

    if not isinstance(a_final, torch.Tensor):
        a_final = torch.tensor(a_final, dtype=torch.float32)
    n_sources = a_final.shape[1]

    if not keep_trial:
        return _empty_trace_result(n_sources, num_channels, len(soma_sps))

    data, data2, a_data, frames = _load_high_res_trial_data(
        trial_ix,
        dmd_ix,
        samp_freq,
        super_pixel_ids,
        datadr,
        trial_table,
        num_channels,
    )

    return compute_high_res_traces(
        data,
        data2,
        background_ds,
        frames,
        a_data,
        subsample_matrix_inds,
        sparse_h_inds,
        sparse_h_vals,
        a_final,
        unique_motion_ds,
        mot_inds_to_keep_ds,
        median_z,
        psf2d,
        num_super_pixels,
        num_fast_zs,
        dmd_pixels_per_column,
        dmd_pixels_per_row,
        num_channels,
        soma_sps,
    )
