Training data for the F0 network
================================

``giant export-f0`` writes training data for the SLAP2 F0 network
(`slap2-F0-network <https://github.com/AllenNeuralDynamics/slap2-F0-network>`_)
from a ``giant silo`` run: per trial and imaging path, the registered 80 Hz
movie and SILo's fitted activity on the same grid, so that the network's
training target is ``movie - activity``. It replaces slap2-F0-network's MATLAB
exporter (``code/matlab/exportF0TrainingData.m``) for runs made with
GIAnT-Python, and writes the same file layout, which ``slap2_f0.io.TrialFile``
reads unchanged.

The exporter lives in GIAnT-Python because it reads SILo's outputs (footprints,
traces, the across-trial maps, the 200 Hz frame lines and discards) and uses
the renderer of ``giant view``; GIAnT-Python does not import slap2-f0, and
slap2-f0 needs only the HDF5 layout below.

Usage
-----

.. code-block:: text

   giant export-f0 RUN OUT [--table TRIAL_TABLE] [--moco-dir MOTION_CORRECTION]
                   [--baseline-dir BASELINES] [--data-dir RAW]
                   [--kernel exact|matched|exporter] [--interp linear|cubic]
                   [--blur SIGMA] [--channels C ...] [--var-fac mean|full]
                   [--paths D ...] [--trials T ...] [--workers N]
                   [--label TEXT] [--giant-ref REF] [--deflate N]

``RUN`` is the run folder (``experiment_summary.h5`` and
``per_trial_summary.h5``, directly or in ``source_extraction/``). The trial
table, registered movies and raw folder are found as ``giant view`` finds them.
``--baseline-dir`` names the F0 network baselines the run used; they are not
needed for the target, but give the file its provenance (``baselineFile``,
``f0Network``) and the QC values ``targetOverBaselineSupport`` and
``targetOverBaselineLabeled``. Each worker needs a few times the size of one
trial's movie in memory (14 GB for a two-channel 804 x 382 px trial of 28 s).

Output
------

``OUT/Path<k>/trial_<NNN>.h5`` (trial numbers of the trial table) and
``OUT/export_summary.json`` (every trial's record, NaN as ``null``; a trial
that could not be exported, for example one without an across-trial map, is
listed with ``error`` and not written). Arrays as h5py reads them, on the
trial's registered grid cropped to the box around the valid pixels, MATLAB's
row and column orientation, pixel coordinates zero-based in the crop
(``cropOriginRC`` gives the box):

=============================  ========================  =====================================================
dataset                        shape, type               content
=============================  ========================  =====================================================
``movie``                      (channel, frame, row,     registered movie; NaN at discarded frames and where
                               col) float32              not imaged
``activity``                   same                      SILo's footprints times ``dF_denoised`` at the
                                                         movie's frames; zero outside the footprints
``mean_image``                 (channel, row, col)       mean over all frames of each written channel
``mask/valid``                 (row, col) uint8          missing in less than ``nanThresh`` of the frames,
                                                         motion discards counted
``mask/labeled``               (row, col) uint8          3 x 3 median-filtered mean image above 3 x its 25th
                                                         percentile
``mask/footprint_support``     (row, col) uint8          pixels any footprint covers
``frame/discard``              (frame,) uint8            SILo's motion discards and frames whose kernel weight
                                                         lies mostly on discarded 200 Hz frames
``frame/motion_row``, ...      (frame,) float32          registration motion, ``rec_neg_err``, online motion
                                                         (``onlineMotion{X,Y,Z}shift``), mean of valid pixels
``varfac``                     (row, col) float32        per-pixel mean of ``slap2/varFacDS`` (every frame
                                                         with ``--var-fac full``)
``sources/peak_row``, ``_col`` (source,) int32           each kept source's peak pixel (-1: none in the trial)
=============================  ========================  =====================================================

Root attributes: the MATLAB exporter's names (``path``, ``trial``,
``frames``, ``nSources``, ``validPixels``, ``kernelDtLines``,
``kernelHalfWidthLines``, ``linesPerCycle``, ``shiftRC``, ``blurSigma``,
``activityGain`` = 1, the QC values below, ``discardedFrames``,
``channels``, ``activityChannel``, ``alignHz``, ``photonScale``, ``label``,
``giantRef``) and ``exporter``, ``exporterVersion``, ``kernel``,
``kernelR``, ``kernelDt200Lines``, ``movieDtLines``, ``footprintInterp``,
``alignmentMode``, ``siloRun``, ``baselineSource``, ``baselineFile``,
``f0Network`` (SILo's record of the trial's network baseline, JSON).
``shiftRC`` is the across-trial map's mean offset over the footprint pixels.

How the activity is rendered
----------------------------

*Space.* Each trial's footprints are moved from SILo's consensus canvas into
the trial's registered frame through SILo's own across-trial map (rigid,
affine or patchwise): the footprint at trial pixel ``q`` is the canvas
footprint at the map's inverse of ``q``. SILo read the trial's data through
the same map. ``--interp linear`` (default, as ``giant view``) interpolates
bilinearly, which blurs a footprint moved by a fraction of a pixel (up to
about 10% off a 1.3 px footprint's peak pixel); ``--interp cubic`` uses a
cubic spline, which keeps the shape (small negative lobes at the edge are
kept, so the flux is unchanged).

*Time.* The 80 Hz movie frame is a weighted mean of the raw samples within
``max(3 dt80, P)`` lines of its line, weights ``exp(-|dline| / dt80)``; SILo's
200 Hz frames are such means with ``dt200``. ``--kernel exact`` (default)
renders ``x80 = r x200(line) + (1 - r) avg80(x200)``, ``r = (dt200 /
dt80)**2``, where ``x200(line)`` is the 200 Hz trace interpolated at the 80
Hz frame's line from kept frames and ``avg80`` the MATLAB exporter's average
(exact for densely sampled pixels; see
:func:`giant_python.viewer.render.exact_weights`). ``--kernel exporter`` is
the MATLAB exporter's average alone, which smooths fast events twice;
``--kernel matched`` narrows it to ``sqrt(dt80**2 - dt200**2)``. Both ``dt``
values come from the data (``DSframes`` and SILo's frame lines).

*Blur and scale.* No blur unless ``--blur`` is given; no scaling (footprints
sum to 1, traces in movie units).

QC attributes
-------------

As the MATLAB exporter computes them, for the activity as written, at the
activity channel, high-passed by an 81-frame running median: ``peakSlope`` and
``peakR`` (movie on activity at each source's peak pixel, medians over
sources), ``diskSlope`` and ``diskR`` (flux in the footprint pixels within 4
px of the peak), ``activitySlope`` (``diskSlope`` at blur 0),
``blurShapeRatio`` (core over 1.5-3.5 px slope; above 1 when the activity is
broader than the movie's events), ``eventResidualFraction`` and ``nEvents``
(event-triggered residual at the peaks over the event-triggered movie, events
on each source's own trace), and the selection-safe ring-event ratios
``eventRatioCore``, ``eventRatio2px``, ``eventRatioSources``,
``eventRatioEvents``, ``eventRatioArtifactSources`` (events picked on the
movie 3-5 px from each source; ``slap2_f0.ringqc`` reproduces them from the
written arrays). New: ``targetOverBaselineSupport`` and
``targetOverBaselineLabeled``, the time-mean of ``movie - activity`` over that
of the network baseline at valid footprint pixels and at valid labeled pixels
outside the footprints.

Differences from the MATLAB exporter
------------------------------------

* SILo's across-trial map instead of a rigid offset searched per trial for the
  best movie-activity correlation at the source peaks (the search could
  prefer an offset that fits noise; patchwise maps cannot be expressed as one
  offset).
* The exact time kernel instead of the double-smoothing average (fast events
  kept 0.93-0.95 of their 80 Hz peak in the activity, so the targets kept
  5-7% of every fast event).
* No per-trial shape-matching blur; ``blurShapeRatio`` still reports the core
  / ring slope ratio.
* ``mean_image`` holds the written channels (the MATLAB exporter wrote every
  channel of the movie, which the reader indexes like ``movie``).
* A trial that fails on a rerun loses its earlier file (the reader finds
  files, not summaries).
* The labeled mask's median filter treats unimaged (NaN) pixels as the
  largest values; MATLAB's ``medfilt2`` behaviour with NaN is not documented.
