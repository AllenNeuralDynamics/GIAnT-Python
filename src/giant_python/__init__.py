"""GIAnT-Python: Python translation of the GIAnT-MATLAB analysis package.

Public API
----------
The recommended entry point for the full pipeline is :class:`Pipeline`. Power
users can drive individual stages (``TrialTableBuilder``, ``MotionCorrector``,
``SourceExtractor``) or call the underlying functions directly. Typed data
models (``TrialTable``, ``AlignmentData``, ``ExperimentSummary``) own the
HDF5 serialization shared with GIAnT-MATLAB.

The GUI lives in :mod:`giant_python.gui` and is imported lazily, so importing
this package never pulls in GUI dependencies.
"""

from .models import (
    AlignmentData,
    AlignParams,
    ExperimentSummary,
    SiloParams,
    TrialTable,
)
from .pipeline import (
    IntegrationSourceExtractor,
    MotionCorrector,
    Pipeline,
    SourceExtractor,
    StandardSourceExtractor,
    TrialTableBuilder,
)

__version__ = "0.1.3"

__all__ = [
    "AlignmentData",
    "AlignParams",
    "ExperimentSummary",
    "IntegrationSourceExtractor",
    "MotionCorrector",
    "Pipeline",
    "SiloParams",
    "SourceExtractor",
    "StandardSourceExtractor",
    "TrialTable",
    "TrialTableBuilder",
    "__version__",
]
