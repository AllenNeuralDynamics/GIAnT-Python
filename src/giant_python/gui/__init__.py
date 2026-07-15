"""Interactive GUI for ROI annotation.

Import-isolated: the headless pipeline never imports this package, so
``pip install giant-python`` works on a server without GUI dependencies. The
GUI extras (e.g. napari) are pulled in via ``pip install giant-python[gui]``.
"""

from .draw_rois import DrawROIs, annotate_rois

__all__ = ["annotate_rois", "DrawROIs"]
