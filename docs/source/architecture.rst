Architecture and ownership
==========================

Scope
-----

GIAnT-Python is a package for processing glutamate imaging data, with
advanced capabilities for SLAP2 recordings. The current implementation
supports only SLAP2 Band Scan processing: BandSILo source extraction and
Band Scan ROI annotation from pre-registered inputs. Additional functionality
is available in
`GIAnT-MATLAB <https://github.com/AllenNeuralDynamics/GIAnT-MATLAB>`_,
with porting efforts underway.

The architecture below describes the implemented functionality. Public usage,
configuration, results and staging are documented in :doc:`migration`; saved
schemas and coordinate conventions are in :doc:`file_formats`; trace
estimation is in :doc:`traces`.

Module owners
-------------

All module names below are relative to ``giant_python``.

.. list-table:: Canonical responsibilities
   :header-rows: 1
   :widths: 44 56

   * - Owner
     - Responsibility
   * - ``pipeline.extract`` / ``pipeline.annotate`` / ``pipeline.pipeline``
     - Lazy public services, explicit capability routing and the chainable
       ``Pipeline``. The CLI and standalone calls use the same services.
   * - ``models.params`` / ``models.trial_table``
     - Split configuration and portable input metadata; construction does not
       discover band assets or normalize motion.
   * - ``models.experiment_summary``
     - Eager ``ExperimentSummary``, ``PathSummary``, ``FrameInfo``, ``Source``,
       ``UserRoi`` and ``Visualizations`` models.
   * - ``io.experiment_summary`` / ``io.hdf5`` / ``io.trial_table``
     - Summary codecs, shared HDF5 decoding and parameter-group writing,
       and trial-table reading. Model convenience codecs delegate here.
   * - ``io.slap2`` / ``io.tiff``
     - Raw acquisition/alignment reading and TIFF decoding, without private
       band motion-sign normalization.
   * - ``extraction.band.inputs``
     - Runtime paths, kept trials, resolved acquisition metadata, private
       reference offsets, lookup/reference loading and PSF discovery.
   * - ``extraction.band.workflow`` / ``extraction.band.result_assembly``
     - Scientific orchestration and construction of public path models from
       numerical results; persistence is delegated to shared IO.
   * - ``extraction.band.geometry`` / ``extraction.band.operators``
     - Pure pixel support, PSF padding/cropping and sparse projection operators.
   * - ``extraction.band.motion_binning`` / ``extraction.band.background`` /
       ``extraction.band.activity``
     - Band motion policies, background estimation and activity calculations.
   * - ``extraction.band.localization``
     - Band-specific NMF/localization and spatial profile fitting.
   * - ``extraction.band.traces`` / ``extraction.band.transient``
     - Trace extraction and fixed-profile temporal optimization.
   * - ``extraction.band.summary_images`` / ``extraction.band.trial_data``
     - Image assembly and superpixel data reduction.
   * - ``numerics.baseline`` / ``numerics.peaks`` /
       ``numerics.interpolation`` / ``numerics.morphology``
     - Acquisition-independent numerical kernels and helpers.
   * - ``io.annotations`` / ``extraction.band.rois`` /
       ``extraction.band.annotation``
     - Annotation persistence, pure band ROI geometry and annotation policy.
   * - ``gui.roi_editor`` / ``gui.parameters_gui``
     - Interactive presentation with lazy toolkit imports.
   * - ``parallel`` / ``progress``
     - Ordered trial mapping and opt-in logging/progress; the declared
       ``tqdm`` dependency is imported lazily for verbose progress.
   * - ``assets/psfs/``
     - Bundled PSF templates.

Backend and IO boundaries
-------------------------

Runtime input preparation resolves raw and alignment locations from the
loaded table's explicit ``datadr`` and ``savedr``. ``source_path`` records
provenance, not a location reconstructed from ``savedr``. Copying a table
does not relocate its inputs; see :doc:`migration` for staging instructions.
``ResolvedAcquisition`` carries inferred channel metadata separately from
caller-owned scientific options.

``PathResult`` collects numerical outputs;
``result_assembly.assemble_path_summary(PathResult, path_index=0)`` constructs
the public ``PathSummary``. Schema dictionaries inside IO are implementation
details, not a second public result API. The summary codec supports the
emitted band subset, not lossless round-tripping of every external field.
See :doc:`file_formats` for output coverage and non-atomic write behavior.

``Slap2Reader`` is a structural typing protocol for the vendor reader's
attributes and batched ``getLineData`` method. It promises neither a context
manager nor a close method. OpenCV and tkinter are loaded for interactive
work, not model deserialization; the implemented ROI editor does not use
napari.

Numerical boundaries
--------------------

Spatial localization/NMF and trace-only temporal optimization are separate
stages. Source activity uses channel zero. See :doc:`traces` for the trace
objective, validity masks, trial boundaries, convergence and reproducibility
controls rather than treating denoising as a selectable extraction mode.

``numerics.peaks.gaussian_peaks_integrated`` accepts ``(N, 5)`` parameters
``[amp, mu_y, mu_x, sigma_y, sigma_x]``. Both widths must be explicit, even
for isotropic Gaussians. Fitting uses an isotropic warm start followed by
anisotropic refinement: the first stage ties width updates using their
combined derivative while retaining the five-column representation.

Raw alignment reading preserves saved displacement. Band input preparation
negates it into private reference offsets; ``assemble_path_summary`` converts
those offsets back exactly once for public offline shifts. Generic IO does
not flip signs. Displacement maps reference/view-grid coordinates into raw
sampling coordinates; the inverse mapping uses its negative. No inter-trial
XY correction is invented where the workflow has not computed one. See
:doc:`file_formats` for saved motion signs, producer-provenance cautions and
the linear/clamped versus MATLAB PCHIP/extrapolation distinction.