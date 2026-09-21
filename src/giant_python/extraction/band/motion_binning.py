"""Rounded motion bins and low-resolution frame-selection policy."""

from __future__ import annotations

from typing import Tuple

import numpy as np


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
