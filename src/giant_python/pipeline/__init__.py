"""Pipeline orchestration for GIAnT-Python.

Thin stage classes (the "C" layer) that compose the functional/dataclass core
into the three pipeline stages, plus the :class:`Pipeline` facade that owns the
shared session state and exposes ``run_all``. The underlying stage *functions*
are also exported for power users who want to skip the classes.
"""

from .base import Stage
from .extract import (
    BandSourceExtractor,
    SourceExtractor,
    StandardSourceExtractor,
    extract_band_sources,
    silo,
)
from .organize import (
    TrialTableBuilder,
    build_trial_table,
    build_trial_table_slap2,
    verify_files,
)
from .pipeline import Pipeline
from .register import (
    BandRegistration,
    MotionCorrector,
    MultiRoiRegistration,
    StripRegistration,
    band_registration,
    multi_roi_registration,
    strip_registration,
)

__all__ = [
    "band_registration",
    "BandRegistration",
    "BandSourceExtractor",
    "build_trial_table",
    "build_trial_table_slap2",
    "extract_band_sources",
    "MotionCorrector",
    "MultiRoiRegistration",
    "multi_roi_registration",
    "Pipeline",
    "silo",
    "SourceExtractor",
    "StandardSourceExtractor",
    "Stage",
    "StripRegistration",
    "strip_registration",
    "TrialTableBuilder",
    "verify_files",
]
