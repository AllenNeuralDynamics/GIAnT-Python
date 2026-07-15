"""Trial table model (``trialTable`` struct / trial_table.h5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union


@dataclass
class Slap2Info:
    """SLAP2-specific trial metadata stored alongside the trial table.

    Corresponds to the ``slap2_info`` sub-struct in GIAnT-MATLAB. Holds the
    reference stack(s), per-trial line ranges, and inferred timing.

    Attributes
    ----------
    ref_stack : dict
        Reference stack(s), keyed by DMD path, each with image, channels,
        Zs, and the DMD-pixel-to-sample transform.
    first_line, last_line : list of int
        First/last scanned line index for each trial.
    trial_start_time_inferred, trial_end_time_from_pc : list of float
        Inferred and PC-reported trial timing.
    """

    ref_stack: dict = field(default_factory=dict)
    first_line: list = field(default_factory=list)
    last_line: list = field(default_factory=list)
    trial_start_time_inferred: list = field(default_factory=list)
    trial_end_time_from_pc: list = field(default_factory=list)


@dataclass
class TrialTable:
    """Trial table: the metadata that threads through the whole pipeline.

    Mirrors the ``trialTable`` struct and its trial_table.h5 serialization in
    GIAnT-MATLAB. The motion-correction and source-extraction stages append
    their groups (``motion_correction``, ``source_extraction``) to this same
    structure, so it is the shared state held by :class:`~giant_python.\
pipeline.pipeline.Pipeline`.

    Attributes
    ----------
    datadr : Path or None
        Directory containing the raw data files.
    savedr : Path or None
        Directory in which results are written.
    filename : list
        File names, shaped ``(n_dmds, n_trials)``.
    true_trial_ix : list
        Trial indices, shaped ``(n_dmds, n_trials)``.
    epoch : list
        Epoch numbers, shaped ``(n_dmds, n_trials)``.
    slap2_info : Slap2Info or None
        SLAP2-only metadata (``None`` for Bergamo).
    motion_correction : dict or None
        Populated by the registration stage (registered filenames, alignment
        data filenames, per-trial failure flags, and the ``AlignParams`` used).
    source_extraction : dict or None
        Populated by the extraction stage (analysis params, raw input files).
    """

    datadr: Optional[Path] = None
    savedr: Optional[Path] = None
    filename: list = field(default_factory=list)
    true_trial_ix: list = field(default_factory=list)
    epoch: list = field(default_factory=list)
    slap2_info: Optional[Slap2Info] = None
    motion_correction: Optional[dict] = None
    source_extraction: Optional[dict] = None

    @classmethod
    def from_h5(cls, path: Union[str, Path]) -> "TrialTable":
        """Load a trial table from a MATLAB-compatible trial_table.h5 file.

        Parameters
        ----------
        path : str or Path
            Path to the trial_table.h5 file.

        Returns
        -------
        TrialTable
            The deserialized trial table.
        """
        raise NotImplementedError

    def to_h5(self, path: Union[str, Path]) -> None:
        """Write this trial table to a MATLAB-compatible trial_table.h5 file.

        Parameters
        ----------
        path : str or Path
            Destination path for the trial_table.h5 file.
        """
        raise NotImplementedError
