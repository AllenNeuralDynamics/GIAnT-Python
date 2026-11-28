"""Trial tables, registration, band annotation/extraction and a facade.

``register_multiroi`` imports PyTorch only when called.
"""

from .annotate import annotate_rois
from .extract import extract_band_sources, extract_sources
from .pipeline import Pipeline
from .register import register_multiroi
from .trial_table import build_trial_table_slap2, trim_trial_table

__all__ = [
    "annotate_rois",
    "build_trial_table_slap2",
    "extract_band_sources",
    "extract_sources",
    "Pipeline",
    "register_multiroi",
    "trim_trial_table",
]
