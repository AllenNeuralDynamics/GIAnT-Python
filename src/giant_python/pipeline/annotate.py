"""Public ROI annotation service with lazy band/GUI imports."""

from __future__ import annotations

from typing import Optional

from ..models.params import (
    AnnotationInput,
    BandParamsInput,
    ExecutionInput,
)
from .extract import TrialTableInput, _require_band_mode


def annotate_rois(
    input: TrialTableInput,
    params: BandParamsInput = None,
    *,
    microscope: str = "slap2",
    scan_mode: Optional[str] = "band",
    execution: ExecutionInput = None,
    annotations: AnnotationInput = None,
) -> str:
    """Load or draw and save band ROIs through the shared annotation service.

    Parameters
    ----------
    input : str, Path or TrialTable
        Exact filename or loaded trial table; never reconstructed from a
        results directory.
    params : BandSiloParams, dict or None
        Scientific configuration only.
    microscope : str
        Currently only ``'slap2'`` is supported.
    scan_mode : str or None
        Same routing policy as ``extract_sources``.
    execution : ExecutionOptions, dict or None
        Run policy, including logging.
    annotations : AnnotationOptions, dict or None
        GUI policy and operator metadata. Calling this explicit annotation
        step enables ROI handling for this call even if ``enabled=False``.

    Returns
    -------
    str
        Path to the written or existing annotation file.
    """
    _require_band_mode(microscope, scan_mode)
    if input is None:
        raise ValueError("A trial-table path or TrialTable is required")
    from ..extraction.band.annotation import annotate_band_rois

    return annotate_band_rois(
        input, params, execution=execution, annotations=annotations
    )
