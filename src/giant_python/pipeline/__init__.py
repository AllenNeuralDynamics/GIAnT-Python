"""Registration, band annotation/extraction services and a chainable facade.

``register_multiroi`` imports PyTorch only when called.
"""

from .annotate import annotate_rois
from .extract import extract_band_sources, extract_sources
from .pipeline import Pipeline
from .register import register_multiroi

__all__ = [
    "annotate_rois",
    "extract_band_sources",
    "extract_sources",
    "Pipeline",
    "register_multiroi",
]
