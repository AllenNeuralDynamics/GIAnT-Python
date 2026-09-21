"""Chainable coordinator over the same services as standalone functions.

Band annotation/extraction starts from existing registered trial metadata.
Returned summaries currently contain eager arrays.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

from ..models.experiment_summary import ExperimentSummary
from ..models.params import (
    AnnotationInput,
    BandParamsInput,
    BandSiloParams,
    ExecutionInput,
    resolve_band_options,
)
from ..models.trial_table import TrialTable
from .annotate import annotate_rois
from .extract import TrialTableInput, extract_sources


class Pipeline:
    """Coordinator for annotation and extraction from a registered trial table.

    Parameters
    ----------
    microscope : str
        Acquisition type; only ``"slap2"`` band extraction is implemented.
    params : BandSiloParams, dict or None
        Scientific settings only; defaults to band.
    execution : ExecutionOptions, dict or None
        Run policy, separate from science.
    annotations : AnnotationOptions, dict or None
        ROI inclusion and GUI policy.
    scan_mode : str or None
        Optional explicit routing override; unsupported modes fail honestly.

    Attributes
    ----------
    tt : TrialTable or None
        The session's trial table (lightweight metadata), threaded between
        stages.
    summary : ExperimentSummary or None
        The extraction result (eager arrays, not a disk-backed proxy).
    trial_table_path : Path or None
        Exact source filename, when known. Never inferred from ``savedr``.
    """

    def __init__(
        self,
        microscope: str = "slap2",
        params: BandParamsInput = None,
        *,
        execution: ExecutionInput = None,
        annotations: AnnotationInput = None,
        scan_mode: Optional[str] = "band",
    ) -> None:
        """Store session config; ``tt``/``summary`` start empty."""
        self.microscope = microscope
        self.params = BandSiloParams() if params is None else params
        self.execution = execution
        self.annotations = annotations
        self.scan_mode = scan_mode
        self.tt: Optional[TrialTable] = None
        self.summary: Optional[ExperimentSummary] = None
        self.trial_table_path: Optional[Path] = None
        self.annotation_path: Optional[str] = None

    @classmethod
    def from_trial_table(
        cls,
        path: TrialTableInput,
        microscope: str = "slap2",
        **kwargs,
    ) -> "Pipeline":
        """Start a session from a trial-table path or loaded metadata.

        Parameters
        ----------
        path : str, Path or TrialTable
            Exact filename or an already loaded table. A path is read once;
            in-memory inputs are not copied, serialized or reloaded.
        microscope : str
            ``"slap2"``; other acquisition types are unsupported.
        **kwargs
            Forwarded to :class:`Pipeline` (e.g. ``params`` or ``execution``).

        Returns
        -------
        Pipeline
            A pipeline retaining ``tt`` and its known source filename.
        """
        if isinstance(path, TrialTable):
            table = path
            source_path = table.source_path
        elif isinstance(path, (str, Path)):
            source_path = Path(path)
            table = TrialTable.from_h5(source_path)
            # Keep the supplied filename even when savedr points elsewhere.
            table.source_path = source_path
        else:
            raise TypeError("Expected a trial-table path or TrialTable")
        pipeline = cls(microscope=microscope, **kwargs)
        pipeline.tt = table
        pipeline.trial_table_path = (
            Path(source_path) if source_path is not None else None
        )
        return pipeline

    def _require_trial_table(self) -> TrialTable:
        """Return loaded metadata or explain how to initialize a session."""
        if self.tt is None:
            raise ValueError(
                "No trial table is loaded; use Pipeline.from_trial_table() "
                "before annotation or extraction"
            )
        return self.tt

    def annotate(self) -> "Pipeline":
        """Load or draw band ROIs, retaining the same trial-table object.

        Uses the public annotation service and enables ROI inclusion for a
        subsequent ``extract()`` after success. Drawing is still subject to
        explicit/headless/automatic GUI policy. Caller options are not edited.

        Returns
        -------
        Pipeline
            ``self`` (supports method chaining).
        """
        table = self._require_trial_table()
        self.annotation_path = annotate_rois(
            table,
            self.params,
            microscope=self.microscope,
            scan_mode=self.scan_mode,
            execution=self.execution,
            annotations=self.annotations,
        )
        _, _, policy = resolve_band_options(
            self.params, self.execution, self.annotations
        )
        self.annotations = replace(policy, enabled=True)
        return self

    def extract(self) -> "Pipeline":
        """Run source extraction, storing ``self.summary``.

        Returns
        -------
        Pipeline
            ``self`` (supports method chaining).
        """
        table = self._require_trial_table()
        self.summary = extract_sources(
            table,
            self.params,
            microscope=self.microscope,
            scan_mode=self.scan_mode,
            execution=self.execution,
            annotations=self.annotations,
        )
        return self
