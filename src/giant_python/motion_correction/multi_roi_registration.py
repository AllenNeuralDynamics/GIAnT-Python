"""Multi-ROI motion correction for SLAP2 recordings."""

from pathlib import Path
from typing import Optional, Union


def multi_roi_registration(
    full_path_to_trial_table: Optional[Union[str, Path]] = None,
    params_in: Optional[dict] = None,
) -> None:
    """Perform motion correction for SLAP2 multi-ROI recordings.

    Registers each trial to a common template computed from initial frames,
    producing downsampled registered TIFF movies and alignment metadata for
    each DMD. Corresponds to MultiRoiRegistration.m in GIAnT-MATLAB.

    Parameters
    ----------
    full_path_to_trial_table : str or Path, optional
        Full path to the trial table file.
    params_in : dict, optional
        Parameter overrides. Keys and default values are defined in
        :func:`giant_python.utils.set_params` under ``"MultiRoiRegistration"``.
    """
    raise NotImplementedError
