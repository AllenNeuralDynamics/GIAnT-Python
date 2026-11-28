"""Spine size from a SILo run: background brightness in fixed windows.

A spine's size is read out as its background brightness (F0), averaged
over each trial and measured with the same window in every trial of a
session (Wright, Hedrick & Komiyama, Science 2025). The readout tracks
changes within a session; absolute sizes, sizes relative to other spines
and changes across sessions are outside what it measures.

The per-trial images that the readout projects come from:

* the F0 network's baseline (``<stem>_BASELINE-80Hz.h5`` of ``slap2-f0
  predict-registered``), averaged over each trial's kept frames and
  resampled into the run's consensus frame (:mod:`.images`);
* the run's own per-trial images (``per_trial_mean_im``, and MATLAB's
  ``per_trial_F0_im`` where a run has it).

:mod:`.summary_io` reads a run's summaries (MATLAB's schema, written by
GIAnT-MATLAB or by GIAnT-Python's raster SILo), :mod:`.rois` builds the
windows (sources' mean footprints, user ROIs, disks), :mod:`.readout`
projects images on them, and :mod:`.product` writes everything a viewer
or a later analysis needs into one compact ``spine_summary`` file.
"""
