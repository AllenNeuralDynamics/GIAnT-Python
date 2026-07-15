"""Pipeline orchestration for GIAnT-Python.

Thin stage classes (the "C" layer) that compose the functional/dataclass core
into the three pipeline stages, plus the :class:`Pipeline` facade that owns the
shared session state and exposes ``run_all``. The underlying stage *functions*
are also exported for power users who want to skip the classes.
"""

from .base import Stage
from .extract import (
    IntegrationSourceExtractor,
    SourceExtractor,
    StandardSourceExtractor,
    extract_integration_sources,
    sel_act,
)
from .organize import (
    TrialTableBuilder,
    build_trial_table,
    build_trial_table_slap2,
    verify_files,
)
from .pipeline import Pipeline
from .register import (
    IntegrationRegistration,
    MotionCorrector,
    MultiRoiRegistration,
    StripRegistration,
    integration_registration,
    multi_roi_registration,
    strip_registration,
)

__all__ = [
    "build_trial_table",
    "build_trial_table_slap2",
    "extract_integration_sources",
    "integration_registration",
    "IntegrationRegistration",
    "IntegrationSourceExtractor",
    "MotionCorrector",
    "MultiRoiRegistration",
    "multi_roi_registration",
    "Pipeline",
    "sel_act",
    "SourceExtractor",
    "StandardSourceExtractor",
    "Stage",
    "StripRegistration",
    "strip_registration",
    "TrialTableBuilder",
    "verify_files",
]
