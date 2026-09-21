GIAnT-Python
============

The implemented workflow is SLAP2 BandSILo extraction and ROI annotation
from a **pre-registered trial table**. Organization, registration, standard
SILo and deconvolution are not implemented; their placeholders have been removed.
Existing raw recordings, reference TIFFs, alignment files and the band
registration lookup table are required; this is not a full GIAnT-MATLAB
pipeline replacement.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   architecture
   migration

Standalone extraction
---------------------

Both an exact trial-table filename and a loaded ``TrialTable`` are accepted.
The filename is never guessed from the results directory. For example, given
``table_or_filename`` supplied by the caller:

.. code-block:: python

   from giant_python import (
       AnnotationOptions, BandSiloParams, ExecutionOptions,
       extract_band_sources,
   )

   summary = extract_band_sources(
       table_or_filename,
       BandSiloParams(),
       execution=ExecutionOptions(max_workers=6, verbose=True),
       annotations=AnnotationOptions(enabled=False, interactive=False),
   )

For a chainable session using the same extraction service:

.. code-block:: python

   from giant_python import (
      AnnotationOptions, BandSiloParams, ExecutionOptions, Pipeline,
   )

   pipeline = Pipeline.from_trial_table(
      table_or_filename,
      params=BandSiloParams(),
      execution=ExecutionOptions(max_workers=6),
      annotations=AnnotationOptions(enabled=False, interactive=False),
   ).extract()
   summary = pipeline.summary

The band signature is
``extract_band_sources(input, params=None, *, execution=None, annotations=None)``.
``Pipeline`` also uses ``params=...`` for science. Each option scope accepts
only its corresponding dataclass, a dictionary of its own fields, or ``None``.
``resolve_band_options`` returns a fresh validated triple; mixed dictionaries
and cross-scope fields are rejected, not translated into other option scopes.
Explicit ``microscope`` / ``scan_mode`` routing belongs to the generic services
and ``Pipeline``; only SLAP2 band is implemented (``scan_mode=None`` also selects
band). Unsupported standalone routing raises ``ValueError``, not
``NotImplementedError``.

``Pipeline`` exposes ``from_trial_table``, ``annotate`` and ``extract``;
``from_config``, ``organize``, ``register`` and ``run_all`` have been removed.
There are no constructor options/attributes named ``save_dir`` or
``align_params``. Directory overrides belong on the loaded trial table.
The ``giant`` CLI has only ``annotate`` and ``extract`` subcommands, with
``--microscope slap2`` and ``--scan-mode band`` as its sole routing choices.
Unsupported commands or choices fail argument parsing.

An explicit ``annotate_rois(...)`` call or ``Pipeline.annotate()`` loads
existing annotations or draws them when permitted. For headless extraction
with existing ROIs, use ``AnnotationOptions(enabled=True, interactive=False)``;
missing required annotations fail rather than silently disabling ROIs.

Output is written beneath the loaded table's ``savedr`` in
``source_extraction/experiment_summary.h5``. Copying a table does not rewrite
its stored directories. The repository example ``examples/extract_band.py``
stages inputs and overrides ``datadr``/``savedr`` on a loaded model explicitly.
It refuses existing output directories and never changes the environment at
import time. See :doc:`migration` for canonical owners, data conventions and
deployment notes. The preceding Python API was unreleased and unused;
compatibility-only imports and adapters are removed, not supported aliases.

Results and parameter editing
-----------------------------

Use ``summary.paths`` exclusively for per-path data. Each ``PathSummary`` has
its own sources, ROIs, visualizations, frame information, depths and global
fluorescence. ``path_index`` is zero-based; disk groups begin at ``Path1``.
Access ``FrameInfo`` fields as attributes or use its shallow ``to_dict()``;
there is no mapping adapter or flat summary convenience API.

``giant_python.gui.parameters_gui.run_parameter_gui(params=None, *, execution=None,
annotations=None)`` returns a validated ``(BandSiloParams, ExecutionOptions,
AnnotationOptions)`` triple, or ``None`` on cancellation, without mutating inputs.
Toolkit imports remain lazy.

Supported schema subset
-----------------------

The repository README preserves extensive MATLAB reference trees and field
tables. They are reference material, **not** a claim that Python emits every
listed field. Band output includes source profiles/coordinates, ``dF_ls``,
``F0``, ``SNR``, visualizations, depths, global fluorescence, frame information
and optional user ROI masks/labels/``F``.

``dF_denoised``, ``events``, ROI ``Fsvd`` and the entire
``per_trial_summary.h5`` are future/reference-only and **not emitted**.
``ExperimentSummary.from_h5`` and ``to_h5`` support the emitted band subset
eagerly. Generic HDF5 decoding and parameter-group writing are shared working
helpers. ``TrialTable.to_h5`` and the ``AlignmentData`` model have been removed;
the raw band alignment reader remains implemented. Native TIFF/SLAP2 readers,
annotation persistence and summary writing are also retained.

The summary codec is model-only: ``giant_python.io.read_summary(path)`` returns
an ``ExperimentSummary`` and ``giant_python.io.write_summary(summary, path)``
returns ``None``. Band assembly uses
``giant_python.extraction.band.result_assembly.assemble_path_summary`` to construct
``PathSummary`` directly; no public assembled-dictionary writer remains.
Removing Python compatibility surfaces does not change MATLAB layout handling,
the exact ``Z_depths`` field spelling, coordinate conversions or shift signs.

Placeholder removal — 2026-09-20
--------------------------------

The earlier requirement to retain future placeholders is superseded. All
unimplemented placeholder modules/functions and corresponding exports have
been removed, including ``AlignParams``, future pipeline stages, generic
numerical stubs and unsupported IO wrappers. Working band science, peak fitting,
the band-specific NMF solver and HDF5 codecs are unchanged. ``Slap2Reader`` is
retained as a typing protocol for real vendor readers, not a placeholder stage.
The README's MATLAB schemas remain external-format reference documentation.
See :doc:`migration` for the removal inventory and :doc:`architecture` for
current owners.

Public API reference
--------------------

These modules expose models, canonical IO and lazy workflow services.
Autodoc deliberately does not traverse the numerical backend or GUI toolkit
modules.

.. automodule:: giant_python.pipeline.extract
   :members: extract_band_sources, extract_sources

.. automodule:: giant_python.pipeline.annotate
   :members: annotate_rois

.. autoclass:: giant_python.pipeline.pipeline.Pipeline
   :members: from_trial_table, annotate, extract

.. automodule:: giant_python.models.params
   :members: BandSiloParams, ExecutionOptions, AnnotationOptions, resolve_band_options

.. autodata:: giant_python.models.params.BandParamsInput

.. autodata:: giant_python.models.params.ExecutionInput

.. autodata:: giant_python.models.params.AnnotationInput

.. autodata:: giant_python.pipeline.extract.TrialTableInput

.. automodule:: giant_python.models.experiment_summary
   :members: ExperimentSummary, PathSummary, FrameInfo, Source, UserRoi, Visualizations

.. automodule:: giant_python.io.experiment_summary
   :members: read_summary, write_summary

.. autoclass:: giant_python.models.trial_table.TrialTable
   :members: from_h5

.. automodule:: giant_python.io.slap2
   :members: Slap2Reader, open_slap2_file, load_alignment_data_h5

.. automodule:: giant_python.io.tiff
   :members: read_tiff

.. automodule:: giant_python.io.hdf5
   :members: load_struct_h5, write_dict_to_h5group

The package export lists below include only the current public surfaces;
removed placeholder names are not aliases or deprecated entry points.

.. autodata:: giant_python.__all__

.. autodata:: giant_python.io.__all__


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
