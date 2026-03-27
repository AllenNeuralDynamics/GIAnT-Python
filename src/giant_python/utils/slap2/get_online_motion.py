"""Online motion correction offset retrieval for SLAP2 data."""

from typing import Tuple

import numpy as np


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
