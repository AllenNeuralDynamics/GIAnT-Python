API and data conventions
========================

Public workflow and options
---------------------------

The lazy package-root exports provide ``extract_band_sources``,
``extract_sources``, ``annotate_rois``, ``Pipeline`` and the input, option and
result models. Both extraction functions accept an exact trial-table filename
or a loaded ``TrialTable`` and return ``ExperimentSummary``.
``Pipeline.from_trial_table(...).extract()`` returns the same pipeline;
read its result through ``pipeline.summary``. ``Pipeline.annotate()`` and
``annotate_rois(...)`` load existing annotations or draw them when permitted.

The band service signature is
``extract_band_sources(input, params=None, *, execution=None, annotations=None)``.
``Pipeline`` uses the same option scopes. Generic services and ``Pipeline``
take explicit ``microscope`` / ``scan_mode`` routing arguments; only
``microscope='slap2'`` with ``scan_mode='band'`` (or ``None``) is supported.
Unsupported standalone routing raises ``ValueError`` before input/backend
loading. Pipeline operations require a loaded table.

The ``giant`` CLI offers ``annotate`` and ``extract``, each with a trial-table
filename and ``slap2`` / ``band`` routing choices. Invalid commands/options
are argument errors (exit code 2). For existing headless ROIs use
``--draw-user-rois --headless``, or in Python
``AnnotationOptions(enabled=True, interactive=False)``. Missing required
annotations fail explicitly rather than silently disabling ROIs.

Configuration
-------------

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
     - ``enabled``, ``interactive``

``resolve_band_options(params=None, execution=None, annotations=None)`` returns
fresh validated objects in the order shown above. Each input accepts **only
its corresponding dataclass, a dictionary of that scope's fields, or None**.
Mixed dictionaries, cross-scope fields, unknown keys and unrelated dataclasses
raise ``TypeError``. Caller-owned inputs are not mutated, even when acquisition
channel counts or GUI policy are later resolved.

Defaults include peak threshold 7, analysis rate 100 Hz, decay 0.15 seconds,
baseline window 4 seconds, denoise window 1 second, VIF 1.38 and PSF dilation 17.
Execution defaults to six workers, no trial limit and verbosity off. Annotation
defaults to disabled with automatic interactivity. See :doc:`traces` for
temporal-fit controls, motion-bin tolerance and random-seed semantics.

Channel count is inferred from acquisition metadata and retained in output
metadata. Source activity extraction always uses **channel zero**; other
source-activity channels contain NaNs. Neither channel count nor activity
channel is a scientific option. If channel count cannot be inferred,
extraction raises ``ValueError``; supply acquisition metadata for a kept trial.

``gui.parameters_gui.run_parameter_gui(params=None, *, execution=None,
annotations=None)`` returns a validated ``(BandSiloParams, ExecutionOptions,
AnnotationOptions)`` triple, or ``None`` on cancellation, without mutating
inputs. Its sparse-factor control is in natural-log units. GUI toolkit imports
are lazy; see :doc:`architecture` for module ownership and numerical boundaries.

Input identity and output location
----------------------------------

``Pipeline.from_trial_table`` loads an exact filename once and retains it as
``trial_table_path``; a loaded table is retained directly. ``source_path`` is
provenance, not a filename reconstructed from ``savedr``. Set ``datadr`` and
``savedr`` on the loaded ``TrialTable`` for explicit directory overrides and
pass that model to extraction. Copying the table file does not rewrite its
stored directories.

Extraction writes beneath ``savedr`` to
``source_extraction/experiment_summary.h5``. Use a fresh output location:
updating an existing summary replaces supplied path groups but retains
unrelated paths and root parameters/layout flags. Array preparation before
each path replacement does not make writes file-wide atomic or provide
checkpoint/resume. Unknown external fields need not survive a model round-trip.
See :doc:`file_formats` for the persistence contract.

Results API
-----------

All imaging results belong to ``ExperimentSummary.paths``. Each ``PathSummary``
owns its sources, ROIs, visualizations, ``FrameInfo``, depths and global
fluorescence. There are no flat or aggregate summary properties. ``path_index``
is zero-based and maps to disk group ``Path{path_index + 1}``; a list position
is not necessarily that identity when reading selected paths.

``FrameInfo`` exposes attributes and a shallow ``to_dict()`` only, not mapping
methods. Results are eager; per-source/ROI entries may share array storage.
They are neither lazy HDF5 proxies nor independent immutable snapshots.
Mutating shared storage can affect other views; serialization may stack
arrays and does not promise zero-copy writing.

``io.experiment_summary.read_summary(path)`` returns ``ExperimentSummary`` and
``write_summary(summary, path)`` accepts that model and returns ``None``.
Both functions are also exported by ``giant_python.io``.
``ExperimentSummary.from_h5`` / ``to_h5`` delegate to these canonical functions.

Production summaries emit ``dF_ls``, ``dF_denoised`` and ``events`` alongside
source profiles, ``F0``, ``SNR``, visualizations, depths, global fluorescence
and frame information. ROI ``Fsvd`` is emitted as NaN unless explicitly
supplied through the model; BandSILo does not calculate it. Disabled and
enabled-but-empty annotations both emit zero-ROI datasets, distinguished by
the ``annotation_enabled`` attribute on the ``user_rois`` group.

New summaries set ``row_major=1`` and preserve rank-two frame vectors.
Acquisition line indexes are one-based; summary spatial coordinates are
zero-based ``[z, y, x]``. ``PathSummary.z_depths`` maps to ``Z_depths`` on disk.
See :doc:`file_formats` for complete shapes, MATLAB layout decoding, annotation
coordinate flags and motion-sign provenance; see :doc:`traces` for temporal
output interpretation. External per-trial summaries are not Python outputs.

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
Python registration.

Crucially, the example overrides ``datadr`` and ``savedr`` on the loaded
``TrialTable`` and passes that model directly to extraction. It does not
reload the copied file and lose those overrides. The copied file is an
unaltered provenance artifact: its stored directories are not rewritten,
and Python does not write trial tables. Input filename fields
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
Dependencies must already be installed; the example does not install packages.

Scientific and execution flags are explicit run choices, not changes to
library defaults. Use ``--use-rois --annotations-dir`` for existing headless
annotations, or add ``--interactive`` to permit drawing. Inspect ``--help``
for all flags.