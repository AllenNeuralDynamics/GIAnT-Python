"""Trial table model (``trialTable`` struct / trial_table.h5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from ..io import load_struct_h5

_SLAP2_FIELDS = (
    "ref_stack",
    "first_line",
    "last_line",
    "trial_start_time_inferred",
    "trial_end_time_from_pc",
)


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

        Faithful deserialization: the on-disk group hierarchy (README
        ``trial_table.h5`` schema) is mirrored into the model with no path
        resolution or file probing. ``slap2_info`` becomes a
        :class:`Slap2Info`; ``motion_correction`` / ``source_extraction`` stay
        as nested dicts (present only once those stages have run). Callers that
        need a runtime view (absolute ``fn_adata`` paths, kept-trial mask, per-
        DMD ``alignHz``) resolve it separately.

        Parameters
        ----------
        path : str or Path
            Path to the trial_table.h5 file.

        Returns
        -------
        TrialTable
            The deserialized trial table.
        """
        raw = load_struct_h5(path)
        slap2 = raw.get("slap2_info")
        return cls(
            datadr=Path(raw["datadr"]) if raw.get("datadr") else None,
            savedr=Path(raw["savedr"]) if raw.get("savedr") else None,
            filename=raw.get("filename"),
            true_trial_ix=raw.get("true_trial_ix"),
            epoch=raw.get("epoch"),
            slap2_info=cls._slap2_info_from_raw(slap2),
            motion_correction=raw.get("motion_correction"),
            source_extraction=raw.get("source_extraction"),
        )

    @staticmethod
    def _slap2_info_from_raw(slap2: Optional[dict]) -> Optional["Slap2Info"]:
        """Build a :class:`Slap2Info` from the raw ``slap2_info`` dict.

        Parameters
        ----------
        slap2 : dict or None
            The faithfully-loaded ``slap2_info`` group (or ``None`` for a
            non-SLAP2 experiment).

        Returns
        -------
        Slap2Info or None
            The populated metadata (only recognized fields), or ``None``.
        """
        if not slap2:
            return None
        kwargs = {k: slap2[k] for k in _SLAP2_FIELDS if k in slap2}
        return Slap2Info(**kwargs)

    def to_h5(self, path: Union[str, Path]) -> None:
        """Write this trial table to a MATLAB-compatible trial_table.h5 file.

        Parameters
        ----------
        path : str or Path
            Destination path for the trial_table.h5 file.
        """
        raise NotImplementedError
