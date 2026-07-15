"""Typed data models for GIAnT-Python.

These dataclasses are the single source of truth for the on-disk HDF5 schema
shared with GIAnT-MATLAB. Each model knows how to (de)serialize itself via
``from_h5`` / ``to_h5`` so that files written by either toolbox remain
interchangeable.
"""

from .alignment import AlignmentData
from .experiment import (
    ExperimentSummary,
    Source,
    UserRoi,
    Visualizations,
)
from .params import AlignParams, SiloParams, set_params
from .trial_table import Slap2Info, TrialTable

__all__ = [
    "AlignmentData",
    "AlignParams",
    "ExperimentSummary",
    "set_params",
    "SiloParams",
    "Slap2Info",
    "Source",
    "TrialTable",
    "UserRoi",
    "Visualizations",
]
