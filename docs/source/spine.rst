Spine size from background brightness
=====================================

``giant spine`` reads out spine size as background brightness (F0) in fixed
windows, trial by trial, from a SILo run. The readout follows Wright,
Hedrick & Komiyama (Science 2025): a spine's resting brightness, measured
with the same window in every trial of a session, tracks changes in its
size within the session. It is not meant for absolute sizes, for sizes of
spines relative to each other, or for changes across sessions.

Inputs
------

- A run folder with ``experiment_summary.h5`` and ``per_trial_summary.h5``,
  written by GIAnT-Python's raster SILo or by GIAnT-MATLAB.
- The session's ``trial_table.h5``: the registered movies' names (which give
  the baseline files) and the raw files' acquisition trial numbers.
- Optionally, the F0 network's baselines: the folder of
  ``<stem>_BASELINE-80Hz.h5`` files written by ``slap2-f0
  predict-registered`` (or by ``giant silo --f0-ckpt``), and the
  motion-correction folder they were predicted from.
- Optionally, the work folder of a run with the 'gain' baseline mode (N1,
  the default with the network's baseline), whose results files hold the
  solver's per-pixel gains.

Per-trial images
----------------

Every image is a stack ``(trials, rows, cols)`` in the run's consensus
frame:

``mean_ch<c>``
    The run's per-trial mean images (``per_trial_mean_im``), ``<c>`` the
    one-based file channel. The run's activity channel
    (``params/activityChannel``, 1 if the run does not record it; for
    example 2 in some external recordings) includes activity.
``f0_silo_ch<c>``
    The run's own per-trial F0 images, where the run has them
    (``per_trial_F0_im``, GIAnT-MATLAB's spine-size branch).
``f0_net``
    The network's baseline averaged over each trial's kept frames: frames
    SiLo discards for motion (from the alignment file's ``recNegErr``) and
    the first and last ``--trim-s`` seconds are left out. Each trial's image
    is resampled into the consensus frame through the run's across-trial
    map (``frame_info/trial_alignment_*``), bilinearly. These images take
    minutes (every trial's baseline is read), so they are cached under a key
    of their inputs and settings; ``giant spine images`` builds only them.
``f0_gain``
    ``f0_net`` times the solver's per-pixel gains (with ``--gain-work-dir``),
    at the selected pixels: the network's baseline as SILo rescaled it. A
    pixel in overlapping subproblems takes the mean of its gains.

Windows and readout
-------------------

Windows are the sources' mean footprints (``sources/spatial/profiles``) and
the run's user ROIs. All images are read out in the same windows: pixels
with a value in every trial used, in every image that covers the field of
view (the mean images and ``f0_net``). ``f0_silo_ch<c>`` and ``f0_gain``
have values only at the pixels SILo fitted, which hold the sources'
footprints but often not all of a user ROI; they are read out in the same
windows, with NaN where they do not cover one, and the command reports how
many windows that leaves without them. For a footprint ``h``
normalized to sum 1 over those pixels and a trial's image ``x``, the size is
``(h . x) / (h . h)``. User ROIs are read out as the mean over their
pixels in the windows, and also as GIAnT-MATLAB's ``user_rois/per_trial_F0``:
the mean over their pixels with a value in that trial, with the share of
such pixels as coverage. The normalized size
is ``log`` of the readout, minus each trial's median over the sources
(bleaching), minus each source's own mean. GIAnT-MATLAB's ``spineTrends``
uses the same readout and normalization. The median is over one set of
sources (``ok``: positive readouts in every trial used, in every image that
covers them); an image with values only at the fitted pixels takes it over
the sources of that set it covers, which is the whole set unless a
footprint reaches past the fitted pixels.

Trend tests
-----------

Each image's normalized source sizes go through the tests of GIAnT-MATLAB's
``spineTrends.m`` and ``spineTrendsResid.m``
(:mod:`giant_python.spine.trends`), over the trials in which the image has
values, for the sources in ``ok`` that it covers:

``step``
    For each split of the trials into the first ``k`` and the rest, the t
    statistic of the difference of their means (pooled SD); the source's
    statistic is the largest ``|t|`` over splits, with the split and the
    size of the step.
``brief``
    A change of 1 to ``--max-brief`` (3) consecutive trials that comes
    back: each such run, with trials on both sides in the same block of
    consecutive acquisition numbers, against the up to 2 trials before it
    and the up to 2 after it, with the noise SD from the median
    trial-to-trial difference away from the run. The run's statistic is
    the smaller of the two t's when they agree in sign; the source's is the
    largest over runs.
``resid`` (beyond the shared trend)
    The trend shared by the sources is the first right singular vector of
    their normalized sizes (signed so that it ends positive). Each source's
    part of it is removed and the step test is run on what is left.

The p-values come from ``--n-perm`` (20000) random orders of each source's
trials (seeded per source with ``--seed``, so a source's p does not depend
on which other sources are tested), counting a permuted statistic within a
relative 1e-9 of the observed one as a tie: the step at the first or last
split depends only on which trial stands alone, so ties are common there.
q-values come from Benjamini-Hochberg across the tested sources of a path;
the report counts the sources with ``q <= --fdr-q`` (0.1) per test and
direction. ``--n-perm 0`` gives the statistics without p.

On GIAnT-MATLAB's own normalized sizes, the statistics, splits, brief runs
and the shared trend agree with ``spineTrends.m`` and
``spineTrendsResid.m`` at single precision, and the p-values within Monte
Carlo error, except at splits within 3 trials of an end, where MATLAB's
rounding decided whether a tie counted and its p could be far too small.

The output
----------

``giant spine readout`` writes ``spine_summary_<key>.h5`` into
``--out-dir``. The key covers the inputs' identities and the settings
(including those of the trend tests), so a rerun with unchanged inputs
reuses the file; the network images' cache does not depend on the trend
settings. It holds, per path, the images, the windows (sparse), the
readouts, the normalized sizes, their trend tests, each source's
per-trial activity in the activity channel (mean ``dF_ls`` and event
frames, to check whether activity moves the readout), which file channel
that is (``activity_channel``), and the acquisition trial numbers. See
:mod:`giant_python.spine.product` for the layout.

Example::

    giant spine readout RUN_DIR --table trial_table.h5 \
        --baseline-dir baselines --moco-dir motion_correction \
        --gain-work-dir RUN_DIR/silo --out-dir spine
