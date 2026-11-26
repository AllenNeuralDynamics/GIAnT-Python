SILo for SLAP2 raster recordings
================================

``giant silo`` extracts sources from registered SLAP2 multi-ROI (raster)
recordings as GIAnT-MATLAB's ``SILo.m`` (main 5a0673b) does, and writes the
same files: ``source_extraction/experiment_summary.h5``,
``source_extraction/per_trial_summary.h5`` and the trial table's
``source_extraction`` group, in MATLAB's schema (names, sizes as MATLAB's
``h5read`` sees them, classes, ``/row_major = 0``). MATLAB's algorithm is the
default (moving-median localization, the fitted baseline, rigid
across-trial alignment, ``getActImPeaks`` seeding, the kinetics test), with
these differences: the solver's fits run to convergence, and by default
the footprint fit holds its variance term at its start (the solver's
default, ``footprint_variance='fixed'``; ``'model'`` is MATLAB's loss);
the seeding's Gaussian fits are local and converged, and a residual peak
that ends within 1.5 pixels of another is rejected (``seed_fit='local'``,
``seed_merge_px=1.5``; ``'matlab'`` and ``0`` are MATLAB's).

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
``--footprint-variance``, ``--localization-baseline``,
``--no-trial-peaks``, ``--activity-channel``); any field can be set with
``--set name=value`` or a JSON file (``--params``).
:class:`~giant_python.models.params.SiloExecution` says how to run (worker
processes and threads, device, precision); it does not change results.

Settings that MATLAB does not have are fields too: ``footprint_variance``
(default: the solver's, now ``'fixed'``), the seeding options below
(defaults: the plan stage's), the localization options of the F0-network
work (``localization_baseline='network'`` with ``--baseline-dir``,
``localization_surround``, ``localization_null``, ...; their defaults are
MATLAB's), and ``trial_peaks``. They are added to the saved ``params`` (in
camelCase) when they differ from MATLAB's algorithm: a default run records
``footprintVariance = 'fixed'``, ``seedFit = 'local'`` and
``seedMergePx = 1.5``, and a run with ``--set footprint_variance=model
--set seed_fit=matlab --set seed_merge_px=0`` has MATLAB's fields exactly.

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
the other stages run.

**plan**: per path, the trials are aligned (rigid), the combined activity
image is seeded, pixels are selected and sources grouped into subproblems
(:func:`~giant_python.extraction.raster.plan.plan_path`). Soma ROIs of
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
  ``(n, 4)`` when fitted; attributes ``settings_hash``, ``movie``,
  ``path``, ``trial``, ``align_hz``, ``num_channels``, ``peaks_fitted``.
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
* Not ported: affine and patchwise alignment, WMF seeding, recordings other
  than SLAP2, the ROI-drawing GUI and the import of post-acquisition
  annotations (an existing ``annotations.h5`` is read).
