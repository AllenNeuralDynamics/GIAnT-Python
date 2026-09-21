API and data conventions
========================

Public workflow and options
---------------------------

Use the lazy package-root exports ``extract_band_sources``,
``extract_sources``, ``annotate_rois``, ``Pipeline``, ``TrialTable``,
``ExperimentSummary``, ``BandSiloParams``, ``ExecutionOptions`` and
``AnnotationOptions``. Both extraction functions accept an exact trial-table
filename or a loaded model. Do not reconstruct a filename from ``savedr``.
``Pipeline.from_trial_table(..., params=BandSiloParams()).extract()`` returns
the same pipeline; read the extraction result through ``pipeline.summary``.

The band service signature is
``extract_band_sources(input, params=None, *, execution=None, annotations=None)``.
``Pipeline(..., params=...)`` uses the same scientific option scope. The chain
method is ``extract()``; ``extract_sources()`` is a standalone generic service,
not a second chain method. Generic extraction and annotation services, and
``Pipeline``, take explicit ``microscope`` / ``scan_mode`` routing arguments;
only ``microscope='slap2'`` with band mode is implemented. ``scan_mode=None``
also selects band. Unsupported routing through standalone services raises
``ValueError`` before input/backend loading, not ``NotImplementedError``.

``Pipeline`` has only ``from_trial_table``, ``annotate`` and ``extract`` as its
public workflow methods. ``from_config``, ``organize``, ``register`` and
``run_all`` are removed, together with constructor options/attributes
``save_dir`` and ``align_params``. Use the loaded table's ``datadr`` / ``savedr``
for explicit input/output directory overrides. Missing loaded state raises
``ValueError`` when an operation is requested.

The ``giant`` CLI offers only ``annotate`` and ``extract``, each with a trial-table
filename. Its only microscope choice is ``slap2`` and its only scan-mode choice
is ``band``. Unsupported commands/options are argument errors (exit code 2).
Use ``--draw-user-rois --headless`` to extract with existing annotations without
drawing; missing required annotations fail explicitly.

.. list-table:: Independent configuration scopes
   :header-rows: 1

   * - Argument
     - Dataclass
     - Scope
   * - ``params``
     - ``BandSiloParams``
     - Band science only
   * - ``execution``
     - ``ExecutionOptions``
     - ``max_workers``, ``verbose``, ``max_trials``
   * - ``annotations``
     - ``AnnotationOptions``
     - ``enabled``, ``interactive``, ``operator``

``resolve_band_options(params=None, execution=None, annotations=None)`` returns
fresh validated objects in the order shown above. Each input accepts **only
its corresponding dataclass, a dictionary of that scope's fields, or None**.
Mixed dictionaries, cross-scope fields, unknown keys and unrelated dataclasses
are rejected; fields are not moved between scopes or converted from a combined
carrier. Caller-owned inputs are not mutated, even when acquisition channel
counts or GUI policy are later resolved.

Defaults remain peak threshold 7, analysis rate 100 Hz, decay 0.15 seconds,
baseline window 4 seconds, denoise window 1 second, VIF 1.38 and PSF dilation 17.
Execution defaults to six workers, no trial limit and verbosity off. Annotation
defaults to disabled with automatic interactivity and operator ``SLAP2 User``.
``num_channels`` and ``activity_channel`` have been removed from
``BandSiloParams``. Channel count is inferred from acquisition metadata and
retained in output metadata; source activity extraction always uses channel
zero (zero-based). Supplying either removed option raises ``TypeError``.
If channel count cannot be inferred, extraction fails with ``ValueError``;
provide acquisition metadata for a kept trial rather than a manual override.
``AlignParams`` has been removed; registration has no Python configuration API.

Canonical owner map
-------------------

The preceding Python API was unreleased and unused. Compatibility-only
namespaces, combined options, extraction adapters and result/IO aliases have
been removed, not deprecated. No old Python import support or production
Python migration procedure is promised. MATLAB file-format compatibility is
retained independently of those removed Python surfaces.

.. list-table:: Canonical modules under giant_python
   :header-rows: 1
   :widths: 44 56

   * - Owner
     - Interface or responsibility
   * - ``pipeline.extract`` / ``pipeline.annotate`` / ``pipeline.pipeline``
     - Root-exported extraction/annotation functions and ``Pipeline``;
       common services also used by the CLI
   * - ``models.params``
     - Split configuration and ``resolve_band_options``; no registration options
   * - ``models.experiment_summary``
     - ``ExperimentSummary``, ``PathSummary``, ``FrameInfo``, ``Source``,
       ``UserRoi`` and ``Visualizations``
   * - ``io.experiment_summary``
     - ``read_summary(path)`` / ``write_summary(summary, path)``;
       also exported by ``io``; model-only summary interface
   * - ``io.hdf5`` / ``io.trial_table``
     - ``load_struct_h5`` and parameter-group writing; ``read_trial_table``
       builds portable metadata and serves ``TrialTable.from_h5``
   * - ``io.slap2`` / ``io.tiff``
     - Raw acquisition/alignment reading and TIFF decoding, without private
       band motion-sign normalization
   * - ``extraction.band.inputs``
     - Runtime paths, kept trials, private reference offsets,
       lookup/reference loading and PSF discovery (including ``default_psf``)
   * - ``extraction.band.workflow`` / ``extraction.band.result_assembly``
     - Scientific orchestration; ``assemble_path_summary`` constructs the
       public path model, with persistence delegated to shared IO
   * - ``extraction.band.geometry`` / ``extraction.band.operators``
     - Pure pixel support, PSF padding/cropping and sparse projection operators
   * - ``extraction.band.motion_binning`` / ``extraction.band.background`` /
       ``extraction.band.activity``
     - Band motion policies, background estimation and activity calculations
   * - ``extraction.band.localization``
     - Implemented band-specific NMF/localization; the generic NMF stub is removed
   * - ``extraction.band.traces`` / ``extraction.band.summary_images`` /
       ``extraction.band.trial_data``
     - Band trace extraction, image assembly and superpixel data reduction
   * - ``numerics.baseline`` / ``numerics.peaks`` / ``numerics.interpolation`` /
       ``numerics.morphology``
     - Shared implemented numerical kernels and helpers
   * - ``gui.roi_editor`` / ``gui.parameters_gui``
     - Interactive presentation with lazy toolkit imports
   * - ``io.annotations`` / ``extraction.band.rois`` / ``extraction.band.annotation``
     - Annotation persistence, pure band ROI geometry and annotation policy
   * - ``parallel`` / ``progress``
     - Ordered trial mapping and opt-in logging/progress

Placeholder removal — 2026-09-20
--------------------------------

The earlier requirement to retain future placeholders is explicitly superseded.
Unimplemented functions/files and corresponding exports are removed, not
deprecated or replaced by forwarding modules. This is an API-scope cleanup,
not an implementation of the omitted GIAnT-MATLAB stages.

Removed modules under ``giant_python``:

* ``numerics.deconv``, ``numerics.nmf``, ``numerics.registration`` and
  ``numerics.variance`` (generic numerical stubs).
* ``pipeline.base``, ``pipeline.organize`` and ``pipeline.register`` (including
  ``Stage``, ``TrialTableBuilder`` and ``MotionCorrector``).
* ``gui.draw_rois`` (not the working ROI editor) and ``models.alignment``
  (including ``AlignmentData``).

Removed members in retained modules include ``AlignParams``,
``TrialTable.to_h5``, ``numerics.interpolation.interp_frame``,
``io.hdf5.save_struct_h5``, ``io.slap2.get_online_motion``,
``io.tiff.scanimagetiff_wrapper``, ``io.tiff.scanimagetiff_data_wrapper``,
and ``pipeline.extract.silo``.
The unsupported Pipeline methods/options listed above and their related exports
are also gone. Config-file construction, checkpoint/resume, organization,
registration, standard SILo, deconvolution and trial-table writing remain
unavailable; calling a removed API is not a supported capability check.

Retained functionality includes ``nearest_interp``, peak detection/fitting,
baseline/morphology kernels, the working band NMF/localization solver, native
TIFF/SLAP2 readers, raw alignment decoding, ``write_dict_to_h5group``, annotation
persistence and experiment-summary codecs. ``Slap2Reader`` is a structural
typing protocol for real vendor readers, not a workflow stub to delete.
Scientific algorithms, defaults, ordering and external file conventions are
unchanged by placeholder removal. The README retains full MATLAB schema
descriptions, including unsupported fields and external registration outputs,
as reference documentation rather than promised Python functionality.

Result, IO and GUI contracts
------------------------------

All imaging results belong to ``ExperimentSummary.paths``. Each ``PathSummary``
owns its sources, ROIs, visualizations, ``FrameInfo``, depths and global
fluorescence. There are no flat or aggregate summary properties. ``path_index``
is zero-based and maps to disk group ``Path{path_index + 1}``; a list position
is not necessarily that identity when reading selected paths.

``FrameInfo`` exposes attributes and a shallow ``to_dict()`` only, not mapping
methods. Results are eager; per-source/ROI entries may share array storage.
They are neither lazy HDF5 proxies nor independent immutable snapshots.
Disabled annotations omit the ROI group; enabled-but-empty is a distinct state.

``io.experiment_summary.read_summary(path)`` returns ``ExperimentSummary`` and
``write_summary(summary, path)`` accepts that model and returns ``None``.
``ExperimentSummary.from_h5`` / ``to_h5`` delegate to these canonical functions.
``extraction.band.result_assembly.assemble_path_summary`` directly builds
``PathSummary``; no public assembled-dictionary result adapter or writer remains.

The codec supports the emitted band subset, not the full reference schema.
It does not emit denoised traces, events, ROI SVD fluorescence or per-trial
summaries. New files set ``row_major=1``. MATLAB layout decoding remains
supported (missing or zero flag reverses multidimensional axes as needed).
Rank-two frame vectors, one-based acquisition line indexes, zero-based spatial
coordinates and annotation coordinate flags are unchanged. The HDF5 field
spelling is exactly ``Z_depths``; ``z_depths`` is the path model attribute.

Writing an existing summary replaces supplied path groups while retaining
unrelated paths and existing root parameters/layout flags. Required source/ROI
arrays are prepared before replacing each path, but the operation is not a
file-wide transaction or resume mechanism. Prefer a fresh destination;
unknown reference fields are not guaranteed a lossless model round-trip.

``gui.parameters_gui.run_parameter_gui(params=None, *, execution=None,
annotations=None)`` returns ``(BandSiloParams, ExecutionOptions,
AnnotationOptions)`` or ``None`` on cancellation. The form retains its
natural-log sparse-factor control and returns validated options without
mutating inputs; it does not combine them into a legacy carrier.

Gaussian parameter convention
-----------------------------

``numerics.peaks.gaussian_peaks_integrated`` accepts only ``(N, 5)`` parameters
``[amp, mu_y, mu_x, sigma_y, sigma_x]``. Supply both widths explicitly, even
for isotropic Gaussians; there is no four-column adapter.
Peak fitting includes an isotropic warm start followed by anisotropic
refinement. Both stages retain five-column parameters; the first stage ties
width updates internally using their combined derivative, then the second
allows independent widths. Removing the alternate representation does not
retune scientific defaults or this implemented fitting policy.

Motion sign and data provenance
-------------------------------

Public ``FrameInfo.offlineXshifts``, ``offlineYshifts`` and
``offlineZshifts`` contain **alignment displacement**, matching standard
MATLAB SILo's saved sign. At matching sample locations this is the sign of
the original alignment ``motionDSc``, ``motionDSr`` and ``motionDSz``.
Displacement is added to reference/view-grid coordinates when sampling raw
images; mapping raw measurement positions into reference coordinates uses
its negative. These two mapping directions must not be conflated.

The raw alignment reader preserves saved values. Band input normalization
negates offline displacement into private reference offsets. Band result
assembly in ``assemble_path_summary`` converts those offsets back exactly once
before building public ``PathSummary``/``FrameInfo`` objects or saving them.
The summary codec itself does not flip signs. No inter-trial XY correction is
invented where the band workflow has not computed one.

The preceding Python version was never released or used, so no production
Python summary migration is required. **If an experimental file from an
unreleased checkout exists**, its offline sign needs producer/alignment-data
provenance. Do not auto-correct on read, use the installed package version
alone to infer file history, or infer provenance from zero-valued shifts.
Retain originals before any explicit transformation. MATLAB-produced and
already correct summaries must not be negated again. Online shifts are
unchanged by the Python API cleanup.

This is sign compatibility, not a claim of identical interpolation: the band
code keeps its existing linear/clamped interpolation, whereas MATLAB's
standard summary builder uses PCHIP/extrapolation. Scientific defaults,
center/surround kernels, motion-bin ordering, numerical/random call order,
trial ordering, and internal extraction calculations are unchanged.

Staging example
---------------

The repository's ``examples/extract_band.py`` is safe to import:
there are no import-time filesystem operations, environment assignments,
matplotlib imports or heavy GIAnT imports.

Provide ``--trial-table``, ``--data-dir``, ``--motion-correction`` and a **new**
``--results-dir``. The example copies the exact supplied table and copies
existing motion-correction assets by default. ``--stage-motion symlink``
selects directory linking (Windows may require Developer Mode or link
privileges). ``--stage-data copy`` or ``symlink`` stages the complete raw
directory, including sidecars and reference TIFFs; ``none`` uses it in place.
This is staging of externally generated motion correction, not execution of
Python registration (which is not implemented and has no placeholder API).

Crucially, the example overrides ``datadr`` and ``savedr`` on the loaded
``TrialTable`` and passes that model directly to extraction. It does not
reload the copied file and lose those overrides. The copied file is an
unaltered provenance artifact: its stored directories are not rewritten,
and there is no ``TrialTable.to_h5`` method to call. Input filename fields
must be relative without parent traversal; absolute names are rejected rather
than silently bypassing staging or being guessed from basenames.

The example refuses any existing results file, directory or symlink, and
refuses output nested inside an input tree. It never deletes arbitrary
destinations or silently retries by replacing a link. Partial output after
failure is left for inspection; choose a new directory for another run.
Copied source symlinks are retained, so verify their targets when relocating
data between machines. Symlink mode is intentionally not a portable copy.

``--joblib-temp`` optionally selects an existing directory with sufficient
space for worker memory maps. Paths, loaded metadata and scientific/run
options are validated before any staging or environment changes. Its
environment override is scoped to staging/extraction, before importing the
lazy extraction service, and restored even when extraction fails. Without
the flag, the existing environment and joblib defaults are left alone.
No fixed container paths or operator names are embedded. Dependencies must
already be installed; the example does not install packages.

Scientific and execution flags are explicit run choices, not changes to
library defaults. Use ``--use-rois --annotations-dir`` for existing headless
annotations, or add ``--interactive`` to permit drawing. Inspect ``--help``
for all flags.

Dependency note
---------------

Verbose progress imports ``tqdm`` lazily with no missing-package fallback;
it is declared directly. The existing band editor uses OpenCV and tkinter,
not napari. Removing Python compatibility code is not a GUI redesign and
changes no numerical dependency, quality threshold, coverage exclusion or
execution backend.