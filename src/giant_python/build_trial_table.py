"""Build trial table for Bergamo recordings."""

from pathlib import Path
from typing import Optional, Union


def build_trial_table(
    dr: Optional[Union[str, Path]] = None,
    fns: Optional[Union[str, list]] = None,
    save_dr: Optional[Union[str, Path]] = None,
) -> dict:
    """Organize multi-trial recordings and metadata for Bergamo recordings.

    Groups TIFF or HDF5 recording files into epochs and trials, populating
    a trial table that is saved to disk. Corresponds to buildTrialTable.m
    in GIAnT-MATLAB.

    Parameters
    ----------
    dr : str or Path, optional
        Directory containing TIFF or HDF5 recording files.
    fns : str or list, optional
        Filename or list of filenames to include. Pass ``True`` to
        auto-select all non-registered TIFF files in *dr*.
    save_dr : str or Path, optional
        Directory in which to save the trial table. Defaults to *dr*.

    Returns
    -------
    dict
        Trial table with keys ``filename``, ``trueTrialIx``, ``epoch``,
        ``trialEndTimeFromPC``, and ``trialStartTimeInferred``.
    """
    raise NotImplementedError
