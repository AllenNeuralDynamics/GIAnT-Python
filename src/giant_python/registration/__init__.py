"""Motion registration of SLAP2 raster recordings (MultiRoiRegistration).

A port of GIAnT-MATLAB ``motion_correction/MultiRoiRegistration.m``: frames
are formed from the raw samples with getImage's exponential kernel, matched
to a growing template by freshness-weighted normalized cross-correlation and
resampled onto a common grid. Outputs use GIAnT-MATLAB's file layout, so its
SILo reads them unchanged.

Computations run in PyTorch on the CPU or a CUDA device.
"""

from .multiroi import register_trial

__all__ = ["register_trial"]
