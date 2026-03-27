"""Strip-based motion correction for Bergamo recordings."""

from pathlib import Path
from typing import Optional, Union


def strip_registration(
    dr: Optional[Union[str, Path]] = None,
    fns: Optional[Union[str, list]] = None,
    params_in: Optional[dict] = None,
) -> None:
    """Perform strip-based motion correction for Bergamo recordings.

    Registers each trial using DFT-based rigid registration, producing
    downsampled and full-resolution registered TIFF or HDF5 movies together
    with per-trial alignment metadata. Corresponds to StripRegistration.m in
    GIAnT-MATLAB.

    Parameters
    ----------
    dr : str or Path, optional
        Directory containing TIFF or HDF5 files, or the path to a trial
        table file.
    fns : str or list, optional
        Filename(s) to register, or path to a trial table file.
    params_in : dict, optional
        Parameter overrides. Keys and default values are defined in
        :func:`giant_python.utils.set_params` under ``"StripRegistration"``.
    """
    raise NotImplementedError
