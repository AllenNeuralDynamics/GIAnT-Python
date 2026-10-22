"""Canonical standalone extraction services.

Only SLAP2 band extraction is supported. Backend imports are lazy; no input
table is reloaded here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from ..models.experiment_summary import ExperimentSummary
from ..models.params import (
    AnnotationInput,
    BandParamsInput,
    ExecutionInput,
)
from ..models.trial_table import TrialTable

TrialTableInput = Union[str, Path, TrialTable]


def _require_band_mode(
    microscope: str,
    scan_mode: Optional[str],
) -> None:
    """Reject unsupported routing before loading data or a backend."""
    if microscope != "slap2":
        raise ValueError(
            "Only microscope='slap2', scan_mode='band' is supported"
        )
    mode = "band" if scan_mode is None else scan_mode
    if mode != "band":
        raise ValueError(
            f"Source extraction for scan_mode={mode!r} is unsupported; "
            "only 'band' is supported"
        )


def extract_sources(
    input: TrialTableInput,
    params: BandParamsInput = None,
    *,
    microscope: str = "slap2",
    scan_mode: Optional[str] = "band",
    execution: ExecutionInput = None,
    annotations: AnnotationInput = None,
) -> ExperimentSummary:
    """Extract sources through the supported workflow service.

    Parameters
    ----------
    input : str, Path or TrialTable
        Exact trial-table filename or a loaded table, passed through intact.
    params : BandSiloParams, dict or None
        Scientific settings only.
    microscope : str
        Currently only ``'slap2'`` is supported.
    scan_mode : str or None
        Explicit routing, defaulting to band. ``None`` also selects band.
    execution : ExecutionOptions, dict or None
        Run policy, independent of science.
    annotations : AnnotationOptions, dict or None
        ROI inclusion and interactivity policy.

    Returns
    -------
    ExperimentSummary
        Eager extraction results, also persisted by the backend.

    Raises
    ------
    ValueError
        If an unsupported microscope or scan mode is requested.
    """
    _require_band_mode(microscope, scan_mode)
    return extract_band_sources(
        input, params, execution=execution, annotations=annotations
    )


def extract_band_sources(
    input: TrialTableInput,
    params: BandParamsInput = None,
    *,
    execution: ExecutionInput = None,
    annotations: AnnotationInput = None,
) -> ExperimentSummary:
    """Perform band-scan source extraction for SLAP2.

    The band-scan variant of SILo: operates on superpixel / DMD-geometry
    data and the externally produced band registration lookup table
    (``fnAdataInt``), rather than on a reconstructed pixel movie. Corresponds
    to ``extractSLAP2IntegrationSources.py`` in ophys-slap2-analysis.

    Delegates lazily to ``giant_python.extraction.band.workflow``. All option
    resolution belongs to that service.

    Parameters
    ----------
    input : str, Path or TrialTable
        Trial-table path or loaded metadata with band alignment data.
    params : BandSiloParams, dict or None
        Scientific settings only.
    execution : ExecutionOptions, dict or None
        Run policy, independent of scientific settings.
    annotations : AnnotationOptions, dict or None
        ROI inclusion and GUI policy.

    Returns
    -------
    ExperimentSummary
        The extracted sources and summary (also written to disk).
    """
    if input is None:
        raise ValueError("A trial-table path or TrialTable is required")
    from ..extraction.band.workflow import extract_band_sources as extract

    return extract(input, params, execution=execution, annotations=annotations)
