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
    The run's per-trial mean images (``per_trial_mean_im``). Channel 1
    includes activity.
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
with a value in every trial used, in every image. For a footprint ``h``
normalized to sum 1 over those pixels and a trial's image ``x``, the size is
``(h . x) / (h . h)``. User ROIs are read out as the mean over their pixels
with a value, with the share of such pixels as coverage. The normalized size
is ``log`` of the readout, minus each trial's median over the sources
(bleaching), minus each source's own mean. GIAnT-MATLAB's ``spineTrends``
uses the same readout and normalization.

The output
----------

``giant spine readout`` writes ``spine_summary_<key>.h5`` into
``--out-dir``. The key covers the inputs' identities and the settings, so a
rerun with unchanged inputs reuses the file. It holds, per path, the images,
the windows (sparse), the readouts, the normalized sizes, each source's
per-trial activity (mean ``dF_ls`` and event frames, to check whether
activity moves the readout) and the acquisition trial numbers. See
:mod:`giant_python.spine.product` for the layout.

Example::

    giant spine readout RUN_DIR --table trial_table.h5 \
        --baseline-dir baselines --moco-dir motion_correction \
        --gain-work-dir RUN_DIR/silo --out-dir spine
