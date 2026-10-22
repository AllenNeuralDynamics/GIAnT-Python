GIAnT-Python
============

GIAnT-Python is a Python package for processing glutamate imaging data,
with advanced capabilities for SLAP2 recordings. Currently, only **SLAP2
Band Scan processing** is implemented: **BandSILo source extraction and
Band Scan ROI annotation** from a pre-registered trial table.

Additional functionality is available in
`GIAnT-MATLAB <https://github.com/AllenNeuralDynamics/GIAnT-MATLAB>`_,
and efforts are underway to port it into GIAnT-Python. Organization,
registration, standard SILo and general-purpose deconvolution stages are
not yet implemented in Python. The current workflow requires existing raw
recordings, reference TIFFs, alignment files and the band registration
lookup table.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   migration
   traces
   file_formats
   architecture

Getting started
---------------

Both an exact trial-table filename and a loaded ``TrialTable`` are accepted.
Given ``table_or_filename`` supplied by the caller:

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

For a chainable session, use ``Pipeline.from_trial_table(...).extract()`` and
read ``pipeline.summary``. The ``giant`` CLI offers ``annotate`` and ``extract``.
For headless extraction with existing ROIs, use
``AnnotationOptions(enabled=True, interactive=False)``; missing required
annotations fail explicitly.

Output is written beneath the loaded table's ``savedr`` in
``source_extraction/experiment_summary.h5``. Copying a table does not rewrite
its stored directories. Use a **fresh output location**: summary updates are
not atomic replacement or resume. See :doc:`migration` for configuration,
the results API, parameter editing and safe staging with the repository's
``examples/extract_band.py``.

Current outputs
---------------

``summary.paths`` holds eager per-path sources, ROIs, visualizations, frame
information, depths and global fluorescence. BandSILo emits ``dF_ls``,
``dF_denoised`` and ``events``, together with source profiles, coordinates,
``F0`` and ``SNR``. ROI ``Fsvd`` is written as NaN unless supplied through the
model. Disabled or enabled-but-empty annotations produce zero-ROI datasets;
the ``user_rois`` group's ``annotation_enabled`` attribute distinguishes them.

See :doc:`traces` for estimation and interpretation, :doc:`file_formats` for
schemas and indexing, and :doc:`architecture` for module ownership and
numerical boundaries. External reference formats, including per-trial
summaries and registration products, do not imply Python output support.

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

.. autoclass:: giant_python.models.trial_table.Slap2Info

.. automodule:: giant_python.io.slap2
   :members: Slap2Reader, open_slap2_file, load_alignment_data_h5

.. automodule:: giant_python.io.tiff
   :members: read_tiff

.. automodule:: giant_python.io.hdf5
   :members: load_struct_h5, write_dict_to_h5group

Package exports:

.. autodata:: giant_python.__all__

.. autodata:: giant_python.io.__all__


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
