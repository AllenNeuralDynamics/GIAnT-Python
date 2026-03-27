"""Build trial table for SLAP2 recordings."""

from pathlib import Path
from typing import Optional, Union


def build_trial_table_slap2(
    dr: Optional[Union[str, Path]] = None,
    save_dr: Optional[Union[str, Path]] = None,
) -> dict:
    """Organize multi-trial recordings for the SLAP2 data processing pipeline.

    Groups DMD1 and DMD2 DAT files into epochs and trials, loads reference
    stacks, and populates a trial table saved to disk. Corresponds to
    buildTrialTableSLAP2.m in GIAnT-MATLAB.

    Parameters
    ----------
    dr : str or Path, optional
        Directory containing DAT recording files and reference TIFF stacks.
    save_dr : str or Path, optional
        Directory in which to save the trial table. Defaults to *dr*.

    Returns
    -------
    dict
        Trial table with keys ``filename``, ``firstLine``, ``lastLine``,
        ``trueTrialIx``, ``epoch``, ``refStack``, ``trialEndTimeFromPC``,
        ``trialStartTimeInferred``, ``datadr``, and ``savedr``.
    """
    raise NotImplementedError
