Architecture and ownership
==========================

Scope
-----

The working path starts with existing registered SLAP2 band inputs and ends
with annotation and BandSILo extraction. Standard SILo, organization,
registration, deconvolution, config loading and checkpoint/resume are not
implemented. As of **2026-09-20**, the earlier requirement to retain future
placeholders is superseded: unimplemented placeholder files, functions and
exports have been removed rather than maintained as future contracts.
MATLAB external-format schemas remain documented separately from Python API
availability; see the repository README and :doc:`migration`.

Directory owners
----------------

All paths below are relative to ``src/giant_python``.

.. list-table:: Canonical responsibilities
   :header-rows: 1
   :widths: 24 76

   * - Owner
     - Responsibility
   * - ``pipeline/``
     - Public extraction/annotation services, explicit capability routing,
       and the chainable ``Pipeline``. Only annotation and extraction are
       supported. The CLI and standalone calls use these same services.
   * - ``models/``
     - Portable metadata, split configuration and eager public results.
       Convenience codecs delegate to shared IO; construction does not
       perform band file discovery or motion normalization.
   * - ``io/``
     - Shared HDF5 decoding, trial-table reading, emitted experiment-summary
       codecs, raw annotation persistence and acquisition adapters. Schema
       reading must not silently change the sign of alignment displacement.
   * - ``extraction/band/``
     - Working scientific backend: input preparation, geometry, motion bins,
       operators, background, activity, localization, summary images, trial
       data reduction, traces, ROI geometry and result assembly. ``workflow``
       composes these stages. ``inputs`` owns runtime path resolution,
       private motion normalization, lookup/reference/PSF loading and asset
       discovery. Pure pixel support and PSF padding/cropping belong to
       ``geometry``; numerical assembly belongs to ``result_assembly``.
       TIFF decoding stays in ``io.tiff``.
   * - ``numerics/``
     - Acquisition-independent baseline, peaks, interpolation and morphology.
       Generic registration/deconvolution/NMF/variance stubs were removed;
       the working band-specific NMF solver remains in ``localization``.
   * - ``gui/``
     - Presentation: the existing ROI editor and parameter form. OpenCV and
       tkinter are loaded for interactive work, not model deserialization.
       Napari is not the implemented band editor.
   * - ``parallel.py`` / ``progress.py``
     - Shared ordered trial mapping and opt-in logging/progress. ``tqdm`` is
       imported lazily for verbose progress and is a declared dependency.
   * - ``assets/psfs/``
     - Bundled PSF templates; no asset relocation or new cache is required.

The former Python API was unreleased and unused. Compatibility-only
``bandsilo`` / ``math`` imports and extraction adapters are removed, not
alternative owners or supported aliases. MATLAB file-format support is a
separate IO responsibility. See :doc:`migration` for the canonical owner map.

Input identity and configuration
--------------------------------

``extract_band_sources(input, params=None, *, execution=None, annotations=None)``
accepts an exact filename or a loaded ``TrialTable``.
``Pipeline.from_trial_table`` loads a filename once and retains it as
``trial_table_path``; a loaded table is retained directly. ``source_path`` is
provenance, not a filename reconstructed from ``savedr``. Runtime input
preparation derives raw and alignment locations from the model's explicit
``datadr`` and ``savedr``. Copying the table itself does not relocate them.

``BandSiloParams`` owns scientific settings, ``ExecutionOptions`` owns worker
count, verbose reporting and the optional trial limit, and
``AnnotationOptions`` owns ROI inclusion, interactivity and operator metadata.
Resolution does not mutate the caller's options to store inferred channel
counts. ``ResolvedAcquisition`` carries resolved backend metadata separately.
Annotation-disabled and enabled-but-empty are distinct states.

``Pipeline(..., params=...)`` and ``Pipeline.from_trial_table(..., params=...)``
use the same split options. ``resolve_band_options`` accepts only the matching
dataclass, a dictionary of fields belonging to that scope, or ``None`` for
each argument, and returns fresh validated objects. It does not accept mixed
configuration dictionaries or redistribute routing/run/ROI fields from science.
Routing belongs to explicit ``microscope`` / ``scan_mode`` arguments on the
generic services and ``Pipeline``. Only ``microscope='slap2'`` and band mode
are supported; ``scan_mode=None`` also selects band. Standalone routing checks
raise ``ValueError`` before loading a backend or input, not
``NotImplementedError``. ``AlignParams`` has been removed.

The public workflow methods on ``Pipeline`` are ``from_trial_table``,
``annotate`` and ``extract``. The constructor accepts ``microscope``, ``params``,
``execution``, ``annotations`` and ``scan_mode``; ``save_dir`` and ``align_params``
options/attributes no longer exist. Output location comes from ``tt.savedr``.
There are no ``from_config``, ``organize``, ``register`` or ``run_all`` methods,
nor generic stage/builder/corrector classes. The ``giant`` CLI exposes only
``annotate`` and ``extract``, with ``slap2`` / ``band`` as the only routing
choices; invalid commands/options are argument errors.

``gui.parameters_gui.run_parameter_gui(params=None, *, execution=None,
annotations=None)`` uses these same scopes and returns their validated triple
or ``None`` on cancellation. The existing form, including its natural-log
sparse-factor control, is retained without a combined parameter carrier.

Typed results and array ownership
---------------------------------

``ExperimentSummary.paths`` contains ``PathSummary`` records. Each path owns
its sources, user ROIs, visualizations, depths, global fluorescence and typed
``FrameInfo``. ``path_index`` is **zero-based**; HDF5 names are **one-based**:
``path_index=0`` maps to ``Path1``. Do not infer identity from a path's list
position when reading a subset of saved paths.

``Source`` and ``UserRoi`` entries can be views of assembled arrays, sharing
data. ``FrameInfo.to_dict()`` is shallow. These are **eager NumPy-backed
results**, not lazy HDF5 proxies, immutable snapshots, or independent copies.
Mutating shared storage can affect other views. Persistence may stack
per-source arrays; the API does not promise zero-copy serialization.

``ExperimentSummary.paths`` is the only per-path result interface; there are
no flat convenience or aggregate source/ROI properties. ``FrameInfo`` exposes
attributes and ``to_dict()``, not mapping methods or a mapping adapter.

Inside the backend, ``TrialTraceResult`` names the historical eight-tuple
fields without changing their order. Source traces are frames by sources;
global fluorescence is frames by channels, and ROI traces are frames by ROIs
by channels. Its ``reference_offsets`` are private row/column/Z offsets,
not the public saved displacement. ``PathResult`` collects numerical outputs
before their public summary assembly.

``result_assembly.assemble_path_summary(PathResult, path_index=0)`` directly
constructs a ``PathSummary``. It is the only band result assembly interface; the former
assembled-dictionary entry point and writer adapters are removed. Private
schema dictionaries inside IO remain implementation details, not a second
public result API.

Persistence boundaries
----------------------

The shared experiment codec supports the **emitted band subset**, not every
field in the README's reference schemas. Future ``dF_denoised``, ``events``,
``Fsvd`` and per-trial summaries are not emitted. Model fields reserved for
those concepts do not establish codec support. Trial-table saving and the
``AlignmentData`` model codecs are unavailable: ``TrialTable.to_h5`` and the
alignment model were removed. Raw alignment reading is implemented separately
by ``io.slap2.load_alignment_data_h5`` and retains saved displacement signs.

``io.hdf5.load_struct_h5`` and ``write_dict_to_h5group`` are working decoder and
parameter-group writer helpers. Removing the unimplemented ``save_struct_h5``
does not remove HDF5 writing, annotation persistence or the summary codec.
Native ``io.tiff.read_tiff`` and ``io.slap2.open_slap2_file`` remain; the generic
ScanImage wrappers and ``get_online_motion`` stub do not. ``Slap2Reader`` remains
a structural typing protocol for the vendor reader's existing attributes and
batched ``getLineData`` method; it promises neither a context manager nor a
close method and is not an executable placeholder stage.

``io.experiment_summary.read_summary(path)`` returns ``ExperimentSummary``;
``write_summary(summary, path)`` accepts that model and returns ``None``.
Both are also exported by ``giant_python.io``. Model convenience methods
delegate to these functions. There are no alternate summary codec aliases or
public assembled-dictionary writers.

The decoder honors ``row_major`` (absent or zero means MATLAB layout) and
reverses multidimensional axis order when needed. The band writer preserves
rank-two frame vectors and writes new files with ``row_major=1``. Numeric
line indexes remain one-based; output spatial coordinates are zero-based
``[z, y, x]``. Annotation coordinate flags distinguish current zero-based
``[y, x]`` from legacy MATLAB one-based ``[x, y]``.
The HDF5 depth field remains exactly ``Z_depths``; its model attribute is
``PathSummary.z_depths``. These format and coordinate conventions are unaffected
by removal of Python compatibility code.

Writing an existing summary retains root parameters/layout flags and unrelated
paths, while replacing supplied path groups. This existing behavior is not
transactional output, a fresh-file guarantee, or resume support. Unknown
reference fields are not guaranteed a lossless model round-trip.

Scientific invariants
---------------------

Removing compatibility code and placeholders does not retune defaults, remove
the working band NMF solver, reorder kernels/motion bins/random calls/trials,
change interpolation or scheduling, add caching, or introduce lazy loading.
The public offline-motion contract established by the architecture refactor
matches standard MATLAB SILo: internal reference offsets become alignment
displacement once in ``assemble_path_summary``. IO does not apply another
conversion; online shifts are unchanged. See :doc:`migration` for sign and
provenance cautions, not a production Python migration procedure.

Shared Gaussian peak evaluation accepts only five-column parameters
``[amp, mu_y, mu_x, sigma_y, sigma_x]``. Equal widths must be explicit in both
columns. Peak fitting retains the implemented isotropic warm start followed
by anisotropic refinement: the first stage ties the two width columns
internally while keeping the canonical five-column representation. Removing
four-column compatibility is not a change to thresholds or fitting policy.