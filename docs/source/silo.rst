SILo for SLAP2 raster recordings
================================

``giant silo`` extracts sources from registered SLAP2 multi-ROI (raster)
recordings as GIAnT-MATLAB's ``SILo.m`` (main 5a0673b) does, and writes the
same files: ``source_extraction/experiment_summary.h5``,
``source_extraction/per_trial_summary.h5`` and the trial table's
``source_extraction`` group, in MATLAB's schema (names, sizes as MATLAB's
``h5read`` sees them, classes, ``/row_major = 0``). MATLAB's algorithm is the
default (moving-median localization, the fitted baseline,
``getActImPeaks`` seeding, the kinetics test), with these differences: the
trials are aligned with patchwise displacement fields
(``inter_trial_alignment='patchwise'``, an option of GIAnT-MATLAB main whose
default is ``'rigid'``); the solver's fits run to convergence, and by default
the footprint fit holds its variance term at its start (the solver's
default, ``footprint_variance='fixed'``; ``'model'`` is MATLAB's loss);
the seeding's Gaussian fits are local and converged, and a residual peak
that ends within 1.5 pixels of another is rejected (``seed_fit='local'``,
``seed_merge_px=1.5``; ``'matlab'`` and ``0`` are MATLAB's); and the
footprints' support is 5 x 5 pixels (``valid_kernel=5``; ``3`` is
MATLAB's).

.. code-block:: bash

   giant register path/to/trial_table.h5
   giant silo path/to/trial_table.h5 --verbose

or, from Python:

.. code-block:: python

   from giant_python.models.params import SiloExecution, SiloParams
   from giant_python.pipeline.silo import run_silo

   if __name__ == "__main__":
       run_silo(
           "path/to/trial_table.h5",
           SiloParams(),  # the SLAP2 defaults (see Settings)
           SiloExecution(workers=5, solver_workers=4, solver_threads=4),
       )

The worker processes are spawned (a fresh interpreter each, on every
platform, so that their thread counts apply), and a spawned worker imports
the calling script: in a script, call ``run_silo`` under
``if __name__ == "__main__":`` as above.

Settings
--------

:class:`~giant_python.models.params.SiloParams` is the one parameter set of
every stage; its defaults are MATLAB's ``setParams('SILo')`` for SLAP2 with
the modality defaults resolved (``peakth`` 3, ``minKineticScore`` 0.2,
``photonScale`` 25), the footprint fit's variance term is the solver's
default, and
``to_matlab()`` gives the ``params`` struct that the summary and the trial
table record. On the command line a few fields
have their own options (``--peak-th``, ``--min-kinetic-score``,
``--footprint-variance``, ``--valid-kernel``, ``--localization-baseline``,
``--no-trial-peaks``, ``--activity-channel``); any field can be set with
``--set name=value`` or a JSON file (``--params``).
:class:`~giant_python.models.params.SiloExecution` says how to run (worker
processes and threads, device, precision); it does not change results.

Settings that MATLAB does not have are fields too: ``footprint_variance``
(default: the solver's, now ``'fixed'``), the seeding options below
(defaults: the plan stage's), the baseline (``baseline_source`` and the
settings it sets, below), and ``trial_peaks``. They are added to the saved
``params`` (in camelCase) when they differ from MATLAB's algorithm: a
default run records
``footprintVariance = 'fixed'``, ``seedFit = 'local'`` and
``seedMergePx = 1.5`` (and MATLAB's own fields ``validKernel`` at 5 x 5
and ``interTrialAlignment = 'patchwise'``), and a run with ``--set
footprint_variance=model --set seed_fit=matlab --set seed_merge_px=0
--valid-kernel 3 --alignment rigid`` has MATLAB's fields exactly.

Two seeding options are not MATLAB's (both on by default since the user's
decision of 2026-11-26):

* ``seed_fit='local'`` runs the Gaussian fits of the seeding and of the
  per-trial peaks locally: each Gaussian only within 10 pixels of where it
  started, every fit to convergence, and a residual peak refitted with the
  peaks within 6 pixels of it rather than with its whole connected region.
  The search is MATLAB's otherwise. On 780125 (5 trials) a path's seeding
  takes about 5-10 s instead of 3-8 minutes, and its sources are closer to
  MATLAB's own than those of the port of MATLAB's capped fit (path 1: 300
  sources, 282 on MATLAB's pixel, against 261 and 249; MATLAB has 296).
* ``seed_merge_px=1.5`` rejects a residual peak whose fit ends within
  1.5 pixels of another peak. An activity image's peak is narrower than a
  pixel; a source between pixels is otherwise fitted as two or three peaks
  less than a pixel apart, and each becomes a source. On injected ground
  truth (GFP and 780125 copies) this removes the duplicate sources (a
  quarter to a third of the sources); every injected source that had a
  seed within 2 px keeps one, but with elongated or wide injected
  footprints 3 of 60 end 1-2 px from the truth instead of within 1 px.

Either one reruns the localization and plan stages of outputs made before
these options existed (with MATLAB's fit); ``seed_fit='matlab'`` with
``seed_merge_px=0`` keeps those current. The ported function
:func:`~giant_python.extraction.raster.peaks.get_act_im_peaks`, and the
standalone per-trial localization, keep MATLAB's fit as their own default.

Footprint support
~~~~~~~~~~~~~~~~~

A footprint is the solver's ``hfilter`` (a Gaussian of 0.9 ``sigma_px``)
blurring a nonnegative image that may be nonzero only on the
``valid_kernel`` x ``valid_kernel`` pixels around the source (MATLAB's
``validKernel``, which MATLAB hard-codes as 3 x 3). Its width therefore
lies between 1.20 and 1.56 px with 3 at ``sigma_px`` 1.33, and up to
2.33 px with 5. ``valid_kernel`` (``--valid-kernel``, odd) is one setting
for the whole run: it sets the plan's grouping zones (sources whose disks
of radius ``ceil(1.5 sigma_px + (valid_kernel - 1) / 2)``, 3 px with 3
and 4 px with 5, touch are solved together) and the solver's support
together. The default is 5 (decision 11 of the SILo port): MATLAB's
3 x 3 cannot represent footprints wider than about 1.56 px. On injected
ground truth (GFP and 780125 copies, close pairs) 5 x 5 halves the width
error of wide and elongated footprints and has 3-9% fewer false events;
on typical footprints it finds slightly fewer of the smallest events (a
1-2% share, about half of it chance detections) and sizes events about 1%
larger; close pairs, traces and F0 are unchanged; the solve stage takes
about 6-10% more CPU. It is a Python setting only: 3 is the setting to
compare with MATLAB, and the saved ``params`` struct records
``validKernel`` at the run's size. A changed value redoes the plan and the stages after it (so
outputs made with 3 before the default changed are redone); the stages
after the plan refuse a plan or container made with another value.

The F0 network's baseline
-------------------------

``baseline_source='network'`` (``--baseline network``) localizes and
extracts with the baselines of the F0 network (slap2-F0-network,
``slap2-f0 predict-registered``: one ``<trial>_BASELINE-80Hz.h5`` per
registered trial-path, on its registered 80 Hz frame, in movie counts)
instead of MATLAB's moving median and baseline fit. It is the default when
the baselines are given:

.. code-block:: bash

   # baselines made beforehand (e.g. a GPU step of a pipeline)
   giant silo path/to/trial_table.h5 --baseline-dir path/to/baselines
   # or predicted here for the trial-paths without one (needs slap2-f0)
   giant silo path/to/trial_table.h5 --f0-ckpt path/to/slap2_f0_s6_ft.pt

``--f0-ckpt`` runs ``slap2-f0 predict-registered`` (found on the PATH, or
``--f0-command``) once for every trial-path without a baseline file,
with the run's ``activity_channel``, ``nan_thresh``, ``motion_thresh`` and
``discard_initial_s``, on the GPU (``--f0-device cpu`` works, many times
slower), into ``--baseline-dir`` or ``<work-dir>/baselines``.
On a GPU that also drives a display, give ``--f0-out-frames 512`` (on an
8 GB card): the default tile size fills the free memory, and Windows spills
what does not fit into shared system memory instead of failing, which runs
several times slower (the baselines are the same, to float16 rounding).
GIAnT-Python does not import slap2-f0. Beside each file it predicts it
writes a record (``<trial>_BASELINE-80Hz.json``: those settings, the movie,
alignment and baseline files, the checkpoint's SHA-256 at prediction
time); a baseline whose record does not fit the run, or that another
checkpoint (by its SHA-256) made, is refused, and the
summary takes the checkpoint's identity from the record. Files made
elsewhere (without a record) are used as they are; slap2-f0 records only
the checkpoint's path and step in them. The record identifies files by
name, size and modification time, so a copied file does not match it.
A prediction step that runs elsewhere (slap2-F0-network's Code Ocean
``predict_registered`` mode) writes the same record with each file's
SHA-256 as well (``predict_baselines(..., digests=True)``); a copy whose
content has the recorded SHA-256 is then accepted, for example baselines
and registration attached as data assets.

``baseline_source`` sets the defaults of the settings that go with the
baseline; any of them can still be set:

============================  ===============  ======================
setting                       ``'median'``     ``'network'``
============================  ===============  ======================
``localization_baseline``     ``'median'``     ``'network'``
``localization_surround``     True             False
``localization_null``         ``'none'``       ``'bootstrap'``
``peak_th``                   3                8
``baseline_mode``             ``'fit'`` (R0)   ``'gain'`` (N1)
============================  ===============  ======================

The localization settings are those recommended for the network's baseline
(slap2-F0-network#6: no surround, the bootstrap null, the variance floor
kept, threshold 8). ``baseline_mode`` is the solver's: 'gain' holds the
network's baseline times one gain per pixel, fitted once after the first
activity fit (N1 of the SILo port's baseline test, within its tolerances
of refitting MATLAB's baseline model starting from the network's, at
fewer parameters); 'fixed' holds it as it is (N0), 'fit_from' refits it
(N3), 'fit' is MATLAB's own fit.

``prepare`` (and the streaming run) maps each trial's baseline onto the
selected pixels: bilinear at the pixels' positions in the trial's
registered frame (the across-trial map the data are read through), divided
by ``photonScale``, kept at the registration's frames in the container's
``/baseline``; each subproblem's baseline is interpolated from it to the
extraction's frames (:mod:`~giant_python.extraction.raster.baseline_map`).
Pixels without a network value start from MATLAB's ``splitFreq`` baseline.
A container's settings hash covers its baseline file, so new baselines
redo the containers made from them. ``experiment_summary.h5`` records
``baseline_source``, ``baseline_mode`` and, per trial-path, the baseline
file and the checkpoint that extraction used (``network_baseline``, JSON)
and that localization used (``localization_network_baseline``); the ``params``
struct records the settings that differ from MATLAB's.

The staged commands take the same settings as ``giant silo``. Those that
read the baselines (``silo-localize``, which also takes ``--f0-ckpt``,
and ``silo-prepare``) default to the network with ``--baseline-dir``; the
others (``silo-plan``, ``silo-dispatch``, ``silo-collect``) need
``--baseline network`` (or the same ``--params``). Each stage refuses files
made with another baseline than its settings say.

Stages
------

``giant silo`` runs these stages on one machine; each also runs on its own,
for any subset of its units, so that a pipeline can scatter them:

=====================  ==========  =======================================
command                unit        writes (under the work folder)
=====================  ==========  =======================================
``silo-localize``      trial-path  ``localize/path<d>_trial<tttt>.h5``
``silo-plan``          path        ``plan/path<d>.h5``
``silo-prepare``       trial-path  ``containers/path<d>_trial<tttt>.h5``
``silo-dispatch``      session     ``jobs/job_<jjjj>.json``
``silo-solve JOB``     job         ``results/path<d>_trial<tttt>[_s..].h5``
``silo-collect``       session     the summaries, the trial table's fields
=====================  ==========  =======================================

The work folder is ``<savedr>/source_extraction/silo`` unless
``--work-dir`` says otherwise; ``--moco-dir``, ``--data-dir``,
``--out-dir`` and ``--annotations`` override the other folders.
``--paths`` and ``--trials`` (one-based, e.g. ``1-10,12``) choose the units
of ``silo-localize`` and ``silo-prepare``, ``--paths`` those of
``silo-plan``.

Every output records a settings hash: the stage's settings
(``SiloParams.stage_settings``) and the identity of its inputs (the
localization files' hashes for a plan, the plan's for a container, the
container's and the subproblem range for a results file). A unit whose
output exists with the same hash is skipped, so a run that stopped can be
started again; ``--overwrite`` redoes it.

**localize**: per trial-path, the registered 80 Hz movie is localized
(``loadAndProcessTrialAsync``, ``localizeSources_vIM``). The file holds the
trial's mean image, activity image, the image its peaks are fitted in, the
discarded frames, and the trial's own peaks. The peaks
(``getActImPeaks`` per trial) feed only the per-trial summary's
``per_trial_act_im_peaks``; ``giant silo`` fits them in the background while
the other stages run. With ``--alignment patchwise`` the file also holds
the mean images of two halves of the trial's frames (alternating 0.5 s
blocks, ``trialHalves``), which the patchwise alignment weights its
patches with; files without them are localized again for such a run.

**plan**: per path, the trials are aligned, the combined activity image is
seeded, pixels are selected and sources grouped into subproblems
(:func:`~giant_python.extraction.raster.plan.plan_path`). The alignment
(``--alignment``, ``SiloParams.inter_trial_alignment``, MATLAB's
``interTrialAlignment``) is a rigid shift per trial, refined for the valid
trials as GIAnT-MATLAB main does:

* ``rigid``: the shift;
* ``affine``: an affine map per trial, fitted groupwise to the mean of the
  warped trials (``alignTrialsAffine``);
* ``patchwise``: a smooth displacement field per trial, set by its values
  at the centres of patches of about ``--align-patch-px`` (1500) imaged
  pixels on top of the field's plane, fitted groupwise with robust weights;
  a patch's bend is kept only as far as it exceeds its noise, measured
  between the two halves of each trial (``alignTrialsPatchwise``). The
  default.

Trials that are aligned but not valid keep their shift. Every per-trial
image, the extraction's sampling of each trial (each selected pixel at its
own mapped position), the network baseline's mapping, the user ROIs (masks
and geometry, through the annotation trial's full map) and the per-trial
peaks (through the map's inverse) use the maps; the summaries hold
``frame_info/trial_alignment_affine`` and, for ``patchwise``,
``frame_info/trial_alignment_displacement``. The motion records add each
trial's displacement at the canvas centre. Soma ROIs of
``annotations.h5`` mask the seeding; the user ROIs, moved into the
consensus frame, are stored with the plan. ``--seeds FILE`` uses given
seeds instead of the seeding (``/Path<n>/seeds`` ``(n, 4)`` ``[amplitude,
row, col, sigma]``, root attribute ``index_base`` 0 or 1; MATLAB's
``thetaf`` is one-based), for parity runs with MATLAB.

**prepare**: per valid trial-path, the raw data are read at the selected
pixels at 200 Hz (``loadTrial``) and written with the subproblems as a
container (:mod:`giant_python.extraction.raster.container`). A trial in
which no selected pixel has a sample gets a ``.skip.json`` marker instead
(MATLAB has no such case handled).

**dispatch**: the session's subproblems, in path, trial and subproblem
order, are cut into ``--machines`` contiguous parts of about equal
estimated work (pixels x frames x (sources + 1)); each part is a job file
naming ranges of subproblems of the containers it touches.

**solve**: a job needs nothing but its job file and the containers it
names. Each subproblem is solved on its own, so results do not depend on
how the work was split (with the same threads per process, precision and
device). ``--containers`` and ``--results`` say where the containers are and
where the results go.

**collect**: the results files of each trial are assembled
(``assembleResults``, ``processResult``), the kinetics test is run
(``rejectFastSources``), and the summaries and the trial table's fields are
written.

``giant silo`` streams as MATLAB does: per path, each valid trial is loaded
while the previous trial's subproblems are solved in a pool of
``--solver-workers`` processes of ``--solver-threads`` threads. It writes the
same results files as the staged run (and containers only with
``--save-subproblems DIR``), and ends with the collect stage.

File contracts
--------------

All stage files are HDF5 (row-major, zero-based indices, images in MATLAB's
row and column orientation) or JSON; each has ``format`` and
``format_version`` attributes.

* Localization file
  (:mod:`giant_python.extraction.raster.stage_files`): ``mean_im``
  ``(rows, cols, channels)``, ``act_im``, ``summary``, ``discard``, ``peaks``
  ``(n, 4)`` when fitted, ``half_ims`` ``(2, rows, cols, channels)`` for a
  patchwise alignment; attributes ``settings_hash``, ``movie``, ``path``,
  ``trial``, ``align_hz``, ``num_channels``, ``peaks_fitted``,
  ``half_ims``.
* Plan file (:func:`~giant_python.extraction.raster.plan.write_plan`), plus
  ``/user_rois`` (``labels`` JSON attribute, ``masks`` (rois, rows, cols))
  and the attributes ``settings_hash``, ``path``, ``z_depth``.
* Container (:mod:`giant_python.extraction.raster.container`).
* Job file (:mod:`giant_python.pipeline.silo_solve`): JSON with the
  ``SiloParams``, the solver's device, precision and batch size, and items
  ``{container, container_hash, span: [first, stop), results, work}``.
* Results file (:mod:`giant_python.extraction.raster.results`): the
  solver's outputs per subproblem in normalized units (footprints, events,
  least squares, F0, SNR, each fit's iterations) and the trial's record
  (frame lines, discards, global and user-ROI traces).

Differences from MATLAB
-----------------------

* The iterative fits run to convergence (MATLAB's fmincon stops at
  iteration caps), and the footprint fit holds its variance at its start
  by default; see :mod:`giant_python.extraction.raster.solver`.
* The Gaussian seed fit does not reproduce MATLAB's seeds, which are where
  MATLAB's ``lsqcurvefit`` stops at its evaluation limit; parity runs pass
  MATLAB's seeds (``--seeds``). The capped fit's result also depends on the
  number of BLAS threads (on 780125 path 1: 262 seeds with the pipeline's
  threads, 280 with one); ``seed_fit='local'`` much less so. By default the
  seeding is the local fit with merging (above), which seeds fewer sources
  than MATLAB where MATLAB seeds one source two or three times.
* The per-trial peak fits run in the background; ``--no-trial-peaks``
  leaves ``per_trial_act_im_peaks`` and ``per_trial_num_peaks`` out.
* A trial without any sample at the selected pixels is skipped (MATLAB
  would fail on it).
* The patchwise alignment's patches are k-means clusters drawn with numpy's
  generator, not MATLAB's, so they differ from MATLAB's; given MATLAB's
  patches the fit agrees with MATLAB's to rounding (1e-12 px on 780125).
* Not ported: WMF seeding, recordings other
  than SLAP2, the ROI-drawing GUI and the import of post-acquisition
  annotations (an existing ``annotations.h5`` is read).

Viewing a run
-------------

``giant view RUN [RUN ...]`` serves a page (http://127.0.0.1:8766 by
default) for inspecting SILo runs by eye. A run is a folder with
``experiment_summary.h5`` and ``per_trial_summary.h5`` (directly or in
``source_extraction/``); the page reads only these files, the trial table,
the registered movies and, optionally, the F0 network's baselines, so runs
whose work folder is gone (Code Ocean results) can be viewed too. Two or
more runs of the same session (network against median baseline, new
against old defaults) are compared; ``NAME=FOLDER`` names a run.

.. code-block:: bash

   giant view network=runs/net median=runs/med \
       --table trial_table.h5 --moco-dir motion_correction \
       --baseline-dir baselines

``--table`` defaults to a ``trial_table.h5`` in or above the first run's
folder, ``--moco-dir`` to the table's ``savedr/motion_correction``,
``--data-dir`` (raw ``.dat``/``.meta``, for the lines per cycle) to the
table's ``datadr``; ``--port``, ``--no-browser``, ``--cache-trials``
(trial-paths kept in memory, default 2), ``--kernel`` and ``--blur``
(below).

**Sources view.** Per path, two panels side by side, each showing a run's
combined activity image, one trial's activity image or the mean image
(``visualizations/act_im``, ``per_trial_act_im``, ``mean_im``), with zoom
and pan in step, contrast, and a flicker between the panels (hold, or
alternating). Markers: kept sources at their footprint centroids,
kinetics-rejected sources at their pixels and, when the run kept its plan
files, seeds whose pixel got no source. With two runs, kept sources are
marked by whether the other run has one within 1.5 px (one to one, closest
first). Clicking a source shows its mean footprint, SNR, kinetic score
and footprint width (RMS per axis over the 6 px disk around its pixel),
its traces over all trials at 200 Hz (``dF_ls``, ``dF_denoised``,
``events``, ``F0``) and its largest denoised events, each of which opens
the movie view at the source's position in that trial and frame; the
movie view then steps through the events.

**Movie view.** One trial-path at a time, frame by frame (scroll, play,
average 1-64 frames), all panels in step, with traces of a clicked pixel
or small region below:

* *Aligned movie*: the trial's registered 80 Hz movie
  (``*_REGISTERED_DOWNSAMPLED-80Hz``, activity channel) in the trial's own
  registered frame, not resampled. It is blank at SILo's motion discards
  (``recNegErr``, as the trial loader) and at the frames whose time kernel
  falls mostly on SILo's discarded 200 Hz frames; only the pixels missing
  in less than ``nanThresh`` of the frames are shown (held in memory: the
  movie as float32 and the baseline as float16, the activity computed per
  request from sparse footprints).
* *Fitted activity*: every kept source's footprint times its denoised
  trace, rendered in the movie's pixels and frames. Space: each trial's
  footprints (``per_trial_profiles``, on the consensus canvas) are moved
  into the trial's frame through the trial's across-trial map
  (``frame_info/trial_alignment_affine``, or
  ``trial_alignment_displacement`` for patchwise runs): the trial-frame
  footprint at pixel ``q`` is the canvas footprint at the map's inverse of
  ``q``, interpolated bilinearly, the map SILo read the trial's data
  through. Time: each 80 Hz frame averages the 200 Hz traces at SILo's
  frame lines with weights ``exp(-|dline| / dt)`` within ``max(3 dt, P)``
  lines (``P`` lines per cycle from the raw header, else ``3 dt``), leaving
  out SILo's discarded frames. With ``--kernel exporter`` (the default)
  ``dt`` is the movie's own, ``ceil(mean(diff(DSframes)))``, as the F0
  network's training-data exporter (slap2-F0-network
  ``exportF0TrainingData.m``) renders training targets. The 200 Hz frames
  are themselves averages (``dt200 = ceil(mean(diff(lines)))``), so this
  smooths fast events twice: on noiseless injected GFP events it keeps
  0.93-0.95 of an event's 80 Hz peak. ``--kernel matched`` uses
  ``sqrt(dt**2 - dt200**2)`` (123 lines on these rigs) and keeps
  0.97-0.99. The page switches between the two (Time kernel).
  Kinetics-rejected sources are not rendered (the summaries do not hold
  their traces).
* *Movie - activity*: the residual, what a training target made from the
  run would be; *movie - activity - baseline* with ``--baseline-dir``
  (the network's ``<stem>_BASELINE-80Hz.h5``), and the baseline itself.

Units: raw counts of the registered movie (each pixel's weighted mean of
its raw SLAP2 samples); SILo's traces and F0 are in the same units, and
footprints sum to 1, so a trace is the source's total over its footprint.

Differences from the exporter's rendering: the trial's footprints are
moved through SILo's own across-trial map instead of an offset searched
per trial by correlating movie and activity (an integer offset of the mean
images plus a sub-pixel search), and they are not blurred to match the
movie's shape unless ``--blur SIGMA`` is given (the exporter picks a blur
per trial); the ``matched`` kernel is not the exporter's.
