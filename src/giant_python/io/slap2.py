"""SLAP2 binary data I/O (``.dat`` / ``.meta``) and online-motion retrieval.

Intended home for the SLAP2 data-file reader (equivalent of MATLAB's
``slap2.Slap2DataFile``) plus the online motion-offset retrieval ported from
getOnlineMotion.m. Also hosts the band-scan superpixel readers
used by the band source-extraction backend.
"""

from typing import Optional, Tuple

import numpy as np


def ref_pixs_to_drc(
    ref_pixs: np.ndarray,
    dmd_pixels_per_column: int,
    dmd_pixels_per_row: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map flat reference pixel indices to DMD (depth, column, row) indices.

    Port of ``ref_pixs_to_drc`` from ``extractSLAP2IntegrationSources.py``.

    Parameters
    ----------
    ref_pixs : ndarray of int
        Flat reference-pixel indices.
    dmd_pixels_per_column, dmd_pixels_per_row : int
        DMD geometry.

    Returns
    -------
    ref_d, ref_c, ref_r : ndarray of int
        Depth, column, and row indices.
    """
    raise NotImplementedError


def read_band_trial_data(
    data_file: object,
    ds_frames: np.ndarray,
    super_pixel_ids: np.ndarray,
    activity_channel: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Read superpixel-binned activity for one band-scan trial.

    Accumulates weighted line data into per-superpixel time series over the
    downsampled frame grid. Port of ``get_trial_data`` from
    ``extractSLAP2IntegrationSources.py``.

    Parameters
    ----------
    data_file : object
        SLAP2 data-file object exposing line/superpixel access.
    ds_frames : ndarray
        Downsampled frame (line-index) grid.
    super_pixel_ids : ndarray
        Superpixel id lookup for the DMD being read.
    activity_channel : int, optional
        Channel to read; ``None`` reads all channels.

    Returns
    -------
    data : ndarray of shape (n_superpixels, n_ds_frames)
        Weighted-mean superpixel activity.
    data_count : ndarray of shape (n_superpixels, n_ds_frames)
        Accumulated weights (for normalization).
    """
    raise NotImplementedError


def get_online_motion(
    data_file: object,
    ds_frames: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retrieve online motion correction offsets from a SLAP2 data file.

    Reads the per-frame X, Y, and Z motion correction shifts that were
    computed online during SLAP2 acquisition by inspecting per-line headers
    in the data file. Corresponds to getOnlineMotion.m in GIAnT-MATLAB.

    Parameters
    ----------
    data_file : object
        SLAP2 data file object exposing line-header access (e.g. a
        ``Slap2DataFile`` instance).
    ds_frames : ndarray of int
        Line indices of the downsampled frames for which to retrieve
        offsets.

    Returns
    -------
    online_x_shift : ndarray of shape (n_frames,)
        Online motion correction X (column) shifts, in pixels.
    online_y_shift : ndarray of shape (n_frames,)
        Online motion correction Y (row) shifts, in pixels.
    online_z_shift : ndarray of shape (n_frames,)
        Online motion correction Z (axial) shifts, in micrometres.
    """
    raise NotImplementedError
