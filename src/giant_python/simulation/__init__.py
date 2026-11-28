"""Synthetic ground truth for SLAP2 raster recordings.

:mod:`.injection` adds photons from synthetic sources to a copy of a
recording's raw samples and writes what was added (the truth file);
:mod:`.render` forms registered frames of any raw samples (for example the
copy minus the original) with a recording's own registration, to check what
was injected.
"""
