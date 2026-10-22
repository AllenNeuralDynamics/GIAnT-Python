"""GIAnT-Python: band extraction functions and a chainable ``Pipeline``.

Use ``BandSiloParams`` for science, ``ExecutionOptions`` for run policy and
``AnnotationOptions`` for ROI/GUI policy. ``extract_sources`` defaults to the
implemented SLAP2 band workflow; explicitly requested unsupported modes raise.
Public exports are lazy: importing GIAnT does not load acquisition or GUI
toolkits. Summary arrays are currently eager, not disk-backed.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import (
        AnnotationOptions,
        BandSiloParams,
        ExecutionOptions,
        ExperimentSummary,
        FrameInfo,
        PathSummary,
        TrialTable,
        resolve_band_options,
    )
    from .pipeline import (
        Pipeline,
        annotate_rois,
        extract_band_sources,
        extract_sources,
    )

__version__ = "0.1.10"

_MODEL_EXPORTS = {
    "AnnotationOptions",
    "BandSiloParams",
    "ExecutionOptions",
    "ExperimentSummary",
    "FrameInfo",
    "PathSummary",
    "TrialTable",
    "resolve_band_options",
}
_PIPELINE_EXPORTS = {
    "Pipeline",
    "annotate_rois",
    "extract_band_sources",
    "extract_sources",
}


def __getattr__(name: str):
    """Load a public model or workflow export on first access."""
    if name in _MODEL_EXPORTS:
        module = import_module(".models", __name__)
    elif name in _PIPELINE_EXPORTS:
        module = import_module(".pipeline", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public exports in interactive completion."""
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "AnnotationOptions",
    "BandSiloParams",
    "ExecutionOptions",
    "ExperimentSummary",
    "FrameInfo",
    "PathSummary",
    "Pipeline",
    "TrialTable",
    "annotate_rois",
    "extract_band_sources",
    "extract_sources",
    "resolve_band_options",
]
