"""The :class:`Pipeline` facade: the full-pipeline orchestration object.

This is a thin *coordinator*, deliberately not a god-object. It holds only
lightweight session state (config, paths, and the ``TrialTable`` metadata),
delegates all real work to the stage classes (which remain independently
usable for partial re-runs), and owns the cross-cutting concerns that belong
to no single stage: ``run_all``, resume/checkpointing, logging, and config
loading.

Crucially it does **not** cache heavy pixel arrays, mean images, or source
matrices in memory; those stay on disk in HDF5/TIFF and are loaded lazily by
each stage. ``self.summary`` references an ``ExperimentSummary`` whose heavy
arrays are themselves disk-backed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from ..models.params import AlignParams, SiloParams
from .extract import SourceExtractor
from .organize import TrialTableBuilder
from .register import MotionCorrector


class Pipeline:
    """Coordinator for the full organize -> register -> extract pipeline.

    Parameters
    ----------
    microscope : str
        ``"slap2"`` or ``"bergamo"``.
    save_dir : str or Path
        Directory in which all results are written.
    align_params : AlignParams, optional
        Motion-correction parameters.
    silo_params : SiloParams, optional
        Source-extraction parameters.

    Attributes
    ----------
    tt : TrialTable or None
        The session's trial table (lightweight metadata), threaded between
        stages.
    summary : ExperimentSummary or None
        The extraction result (heavy arrays disk-backed).
    """

    def __init__(
        self,
        microscope: str,
        save_dir: Union[str, Path],
        align_params: Optional[AlignParams] = None,
        silo_params: Optional[SiloParams] = None,
    ) -> None:
        """Store session config; ``tt``/``summary`` start empty."""
        self.microscope = microscope
        self.save_dir = Path(save_dir)
        self.align_params = align_params or AlignParams()
        self.silo_params = silo_params or SiloParams(microscope=microscope)
        self.tt: Optional[object] = None
        self.summary: Optional[object] = None

    @classmethod
    def from_trial_table(
        cls,
        path: Union[str, Path],
        microscope: str,
        **kwargs: object,
    ) -> "Pipeline":
        """Resume a session from an existing trial_table.h5.

        Parameters
        ----------
        path : str or Path
            Path to an existing trial_table.h5 file.
        microscope : str
            ``"slap2"`` or ``"bergamo"``.
        **kwargs
            Forwarded to :class:`Pipeline` (e.g. ``align_params``).

        Returns
        -------
        Pipeline
            A pipeline with ``tt`` loaded from disk.
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, path: Union[str, Path]) -> "Pipeline":
        """Construct a pipeline from a declarative config file.

        Hook for the future config-driven (YAML/TOML) run mode.

        Parameters
        ----------
        path : str or Path
            Path to the config file.

        Returns
        -------
        Pipeline
            A configured pipeline.
        """
        raise NotImplementedError

    def organize(self, data_dir: Union[str, Path]) -> "Pipeline":
        """Run stage 1 (trial organization), storing the result in ``self.tt``.

        Parameters
        ----------
        data_dir : str or Path
            Directory containing the raw recording files.

        Returns
        -------
        Pipeline
            ``self`` (supports method chaining).
        """
        self.tt = TrialTableBuilder(self.microscope).run(
            data_dir, self.save_dir
        )
        return self

    def register(self) -> "Pipeline":
        """Run stage 2 (motion correction), updating ``self.tt``.

        Returns
        -------
        Pipeline
            ``self`` (supports method chaining).
        """
        corrector = MotionCorrector.for_microscope(
            self.microscope, self.align_params
        )
        self.tt = corrector.run(self.tt)
        return self

    def annotate(self) -> "Pipeline":
        """Run the optional ROI-annotation GUI step.

        Imported lazily so headless installs never import the GUI extra.

        Returns
        -------
        Pipeline
            ``self`` (supports method chaining).
        """
        from ..gui import annotate_rois

        annotate_rois(self.tt)
        return self

    def extract(self) -> "Pipeline":
        """Run stage 3 (source extraction), storing ``self.summary``.

        Returns
        -------
        Pipeline
            ``self`` (supports method chaining).
        """
        extractor = SourceExtractor.for_scan_mode(
            self.silo_params.scan_mode, self.silo_params
        )
        self.summary = extractor.run(self.tt)
        return self

    def run_all(
        self,
        data_dir: Union[str, Path],
        *,
        annotate: bool = False,
        resume: bool = True,
    ) -> "Pipeline":
        """Run the whole pipeline end-to-end.

        Parameters
        ----------
        data_dir : str or Path
            Directory containing the raw recording files.
        annotate : bool
            Whether to run the GUI annotation step between registration and
            extraction.
        resume : bool
            Skip stages whose HDF5 outputs already exist (mirrors the
            ``overwriteExisting`` logic in GIAnT-MATLAB).

        Returns
        -------
        Pipeline
            ``self`` (with ``tt`` and ``summary`` populated).
        """
        raise NotImplementedError
