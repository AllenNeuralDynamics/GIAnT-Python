"""Stage 1 - trial organization (ports of buildTrialTable[SLAP2].m).

Builds the trial table that threads through the rest of the pipeline, plus the
``verify_files`` integrity check. Exposes both the porting-faithful functions
and the :class:`TrialTableBuilder` stage class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from .base import Stage


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
        :func:`giant_python.models.params.set_params`.

    Returns
    -------
    trial_table : dict
        Loaded (and possibly path-corrected) trial table.
    keep_trials : ndarray of bool, shape (n_dmds, n_trials)
        ``True`` for each trial that has all required files present.
    """
    raise NotImplementedError


class TrialTableBuilder(Stage):
    """Stage 1: build the trial table for a microscope's raw data.

    Dispatches to :func:`build_trial_table` (Bergamo) or
    :func:`build_trial_table_slap2` (SLAP2) based on *microscope*.

    Parameters
    ----------
    microscope : str
        ``"slap2"`` or ``"bergamo"``.
    """

    def __init__(self, microscope: str = "slap2") -> None:
        self.microscope = microscope

    def run(
        self,
        data_dir: Union[str, Path],
        save_dir: Union[str, Path],
    ) -> "object":
        """Organize trials and write trial_table.h5.

        Parameters
        ----------
        data_dir : str or Path
            Directory containing the raw recording files.
        save_dir : str or Path
            Directory in which to write trial_table.h5.

        Returns
        -------
        TrialTable
            The organized trial table (also persisted to disk).
        """
        raise NotImplementedError
