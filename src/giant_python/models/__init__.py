"""Typed data models for GIAnT-Python.

Scientific configuration is distinct from run and annotation policy.
Implemented model codecs delegate to shared IO.
"""

from .experiment_summary import (
    ExperimentSummary,
    FrameInfo,
    PathSummary,
    Source,
    UserRoi,
    Visualizations,
)
from .params import (
    AnnotationOptions,
    BandSiloParams,
    ExecutionOptions,
    resolve_band_options,
)
from .trial_table import Slap2Info, TrialTable

__all__ = [
    "AnnotationOptions",
    "BandSiloParams",
    "ExecutionOptions",
    "ExperimentSummary",
    "FrameInfo",
    "PathSummary",
    "resolve_band_options",
    "Slap2Info",
    "Source",
    "TrialTable",
    "UserRoi",
    "Visualizations",
]
