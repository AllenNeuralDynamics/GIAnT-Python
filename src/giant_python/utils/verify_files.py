"""Trial table file verification utilities."""

from pathlib import Path
from typing import Tuple, Union

import numpy as np


def verify_files(
    fn: Union[str, Path],
    dr: Union[str, Path],
    params: dict,
) -> Tuple[dict, np.ndarray]:
    """Verify that all files referenced by a trial table exist on disk.

    Loads the trial table and checks for the registered TIFF or HDF5 movies,
    alignment data files, and raw source data files for each trial and DMD.
    Returns the updated trial table and a boolean keep-mask. Corresponds to
    verifyFiles.m in GIAnT-MATLAB.

    Parameters
    ----------
    fn : str or Path
        Filename of the trial table file (relative to *dr*).
    dr : str or Path
        Directory containing the trial table and associated data.
    params : dict
        Parameter dictionary, as returned by
        :func:`giant_python.utils.set_params`.

    Returns
    -------
    trial_table : dict
        Loaded (and possibly path-corrected) trial table.
    keep_trials : ndarray of bool, shape (n_dmds, n_trials)
        ``True`` for each trial that has all required files present.
    """
    raise NotImplementedError
