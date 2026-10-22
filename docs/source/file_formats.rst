File formats
============

This reference describes Python BandSILo outputs and the external
`GIAnT-MATLAB <https://github.com/AllenNeuralDynamics/GIAnT-MATLAB>`_
schemas used to interpret pre-registered inputs. The full external field
reference is retained even where Python does not consume or emit a field.
Schema descriptions do not imply that every reference file round-trips
losslessly through ``ExperimentSummary``.

Python outputs and write behavior
---------------------------------

BandSILo emits ``experiment_summary.h5`` with source ``profiles``, ``coords``,
``dF_ls``, ``dF_denoised``, ``events``, ``F0``, and ``SNR``; visualizations;
column-vector ``Z_depths``; global ``F``; frame bookkeeping and shifts; and
ROI ``labels``, ``mask``, ``Fsvd``, and ``F``. See :doc:`traces` for trace
estimation and interpretation.

Disabled or enabled-but-empty annotations produce correctly shaped zero-ROI
datasets. The ``annotation_enabled`` attribute on each ``Path{n}/user_rois``
group distinguishes these states. ROI ``Fsvd`` is **not calculated by
BandSILo**: the dataset contains NaN unless values are supplied explicitly
through the model. It is never copied from raw ``F``. Older summaries without
the optional temporal datasets remain readable; rewriting missing optional
data emits NaNs, not fabricated fits.

``per_trial_summary.h5`` is an **unsupported, external-reference format**;
Python BandSILo does not emit it. Registration products and the band lookup
table are **external inputs**, not Python-generated outputs. Python loads
existing trial tables but does not write them.

.. warning::

   Summary writes replace supplied path groups but retain unrelated path
   groups and existing root parameters and layout flags. This is neither an
   atomic replacement nor a resume mechanism. Prefer a **fresh output
   location** for a new analysis, and do not mix producers in place.

HDF5 layout
-----------

The implemented codecs use `h5py <https://www.h5py.org/>`_. New Python band
summary and annotation files set ``row_major`` to ``1``: the logical
dimensions below then match each dataset's h5py ``shape``. MATLAB files set
``row_major`` to ``0`` or omit it; their dimension tuples follow MATLAB
``size()``. The shared decoder inspects the flag and reverses multidimensional
axes as needed to recover logical orientation.

.. list-table:: Layout flag and logical dimensions
   :header-rows: 1
   :widths: 18 32 50

   * - ``row_major``
     - Layout
     - Relation to the dimensions listed below
   * - ``1``
     - Row-major (h5py / NumPy C order)
     - h5py ``shape`` matches the listed logical dimensions.
   * - ``0`` or absent
     - Column-major (MATLAB ``size()``)
     - h5py axes are reversed relative to the listed logical dimensions.

**Emitted vector shapes are preserved on disk.** Band summary frame vectors
retain rank two rather than being squeezed to ``(nFrames,)``. Reference-only
vector shapes describe the external schema, not a Python writer guarantee.
Layout conversion requires schema-specific interpretation and does not imply
full interchangeability. Updating an existing summary preserves its layout
flag; use a fresh destination when changing producers.

Indices, coordinates, and motion signs
--------------------------------------

* Line/frame indices (``first_line``, ``last_line``, ``DSframes``, and
  ``frame_line_idxs``) are **1-indexed**, retaining the SLAP2 convention.
* Summary spatial coordinates (``sources/spatial/coords``, ``act_im_peaks``,
  and per-trial coordinate fields) are **0-indexed**
  ``[z_loc, y_loc, x_loc]``, matching image axes ``(fastz, rows, cols)``.
  ``z_loc`` indexes the fast-z axis; it is not a physical depth.
* Annotation ``position`` and ``center`` use **0-indexed**
  ``[y_loc, x_loc]`` when ``coords_zero_indexed=1``. Absent or zero flags
  denote legacy MATLAB ``images.roi`` coordinates: **1-indexed** ``[x, y]``.
* Images use ``rows x cols`` (row = Y, column = X). ``motionDSr`` and
  ``motionR`` are row shifts; ``motionDSc`` and ``motionC`` are column shifts.
* Python source activity extraction always uses **channel zero**, with NaNs
  in other source-activity channels. Channel count comes from acquisition
  metadata, not ``BandSiloParams``. The external MATLAB parameter
  ``params/activityChannel``, when present, is **1-indexed** into
  ``numChannels``; it is not a Python activity-channel selector or a required
  emitted parameter.

.. warning::

   Public ``FrameInfo.offlineXshifts``, ``offlineYshifts``, and
   ``offlineZshifts`` contain alignment displacement with the standard MATLAB
   SILo sign. Private band reference offsets have the opposite sign and are
   converted once by ``assemble_path_summary()``, not by generic IO.
   Establish producer/alignment provenance before interpreting or changing
   shifts from experimental files. Never negate MATLAB-produced or
   already-correct summaries again. Online shifts are unchanged. Matching
   the sign does not make Python's linear/clamped interpolation equivalent
   to MATLAB's PCHIP/extrapolation.

Schema trees
------------

Trees show files, groups (names ending in ``/``), and datasets. Dimensions
are logical dimensions under the layout convention above. ``Path{n}`` means
one group per imaging path in trial-table order, with disk numbering starting
at one; ``PathSummary.path_index`` is zero-based. Typical two-path recordings
use ``Path1`` and ``Path2``.

Trial table: external input
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A trial table summarizes file locations and analysis-trial structure. Python
loads an existing pre-registered table through ``TrialTable`` / ``Slap2Info``;
it does not create/save the model or append reference stage groups.
``slap2_info`` is SLAP2-specific. In the external pipeline,
``motion_correction`` and ``source_extraction`` are populated downstream::

   trial_table.h5
   ├── datadr
   ├── savedr
   ├── filename
   ├── true_trial_ix
   ├── epoch
   ├── row_major
   ├── slap2_info/
   │   ├── ref_stack/
   │   │   └── Path{n}/
   │   │       ├── IM
   │   │       ├── channels
   │   │       ├── Zs
   │   │       └── dmdPixel2SampleTransform
   │   ├── first_line
   │   ├── last_line
   │   ├── trial_start_time_inferred
   │   └── trial_end_time_from_pc
   ├── motion_correction/
   │   ├── fn_reg_ds
   │   ├── fn_adata
   │   ├── fn_raw
   │   ├── registration_failed
   │   ├── first_line_original
   │   └── align_params/
   └── source_extraction/
       ├── analysis_params/
       └── fn_raw

Alignment data: external registration input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

External motion correction saves one alignment HDF5 file per trial. Python
reads the band fields through
``giant_python.io.slap2.load_alignment_data_h5``; Python registration is not
implemented. The tree includes external fields not consumed by BandSILo::

   <trial_stem>_ALIGNMENTDATA.h5
   ├── row_major
   ├── numChannels
   ├── frametime
   ├── alignHz
   ├── motionDSc
   ├── motionDSr
   ├── motionDSz             (BandRegistration; MultiRoiRegistration with refStackTemplate)
   ├── meanIM               (StripRegistration and MultiRoiRegistration)
   ├── recNegErr            (StripRegistration and MultiRoiRegistration)
   ├── motionC              (StripRegistration / Bergamo only)
   ├── motionR              (StripRegistration / Bergamo only)
   ├── motionZ              (reserved; not written by current external scripts)
   ├── brightnessDS         (BandRegistration only)
   ├── logLikelihoodDS      (BandRegistration only)
   ├── DSframes             (MultiRoiRegistration and BandRegistration)
   ├── registrationFailed   (MultiRoiRegistration and BandRegistration)
   └── slap2/               (SLAP2 only)
       ├── onlineMotionXshift
       ├── onlineMotionYshift
       ├── onlineMotionZshift
       ├── varFacDS         (MultiRoiRegistration only)
       ├── Z_depths         (MultiRoiRegistration only)
       ├── cropRow          (MultiRoiRegistration only)
       ├── cropCol          (MultiRoiRegistration only)
       ├── viewC            (MultiRoiRegistration only)
       ├── viewR            (MultiRoiRegistration only)
       ├── trimRows         (MultiRoiRegistration only)
       └── trimCols         (MultiRoiRegistration only)

Manual annotations
~~~~~~~~~~~~~~~~~~

Manual annotations identify pixels to exclude from analysis or soma pixels
whose signals should be extracted. ROI information is saved in
``annotations.h5``. String fields use UTF-16 code units (``uint16``) for
MATLAB/Python compatibility::

   annotations.h5
   ├── row_major
   ├── coords_zero_indexed
   └── Path{n}/
       ├── dr
       ├── fn
       ├── n_rois
       └── roi_###/
           ├── type
           ├── label
           ├── mask
           ├── position        (polygon; nVertices x 2 [y_loc, x_loc] when flagged)
           ├── center          (circle/ellipse; 1 x 2 [y_loc, x_loc] when flagged)
           ├── semi_axes       (ellipse)
           ├── rotation_angle  (ellipse)
           └── radius          (circle)

Experiment summary
~~~~~~~~~~~~~~~~~~

Source extraction (Source Identification by Activity Localization; SILo)
writes ``experiment_summary.h5`` with extracted sources and experiment data.
The ``total frames`` axis stitches trials from that path in time. The models
are ``ExperimentSummary``, ``Source``, ``UserRoi``, and ``Visualizations``::

   experiment_summary.h5
   ├── row_major
   ├── params/
   └── Path{n}/
       ├── Z_depths                         (fastz x 1)
       ├── sources/
       │   ├── temporal/
       │   │   ├── dF_ls                    (sources x channels x total frames)
       │   │   ├── dF_denoised              (sources x channels x total frames)
       │   │   ├── events                   (sources x channels x total frames)
       │   │   ├── F0                       (sources x channels x total frames)
       │   │   └── SNR                      (sources x 1)
       │   └── spatial/
       │       ├── profiles                 (sources x fastz x rows x cols)
       │       └── coords                   (sources x 3 [z_loc, y_loc, x_loc])
       ├── user_rois/                       [attribute: annotation_enabled]
       │   ├── labels                       (rois x 1)
       │   ├── mask                         (rois x fastz x rows x cols)
       │   ├── Fsvd                         (rois x channels x total frames; NaN unless supplied)
       │   └── F                            (rois x channels x total frames)
       ├── visualizations/
       │   ├── mean_im                      (channels x fastz x rows x cols)
       │   ├── act_im                       (fastz x rows x cols)
       │   └── act_im_peaks                 (sources x 3 [z_loc, y_loc, x_loc])
       ├── global/
       │   └── F                            (channels x total frames)
       └── frame_info/
           ├── offlineXshifts               (total frames x 1)
           ├── offlineYshifts               (total frames x 1)
           ├── offlineZshifts               (total frames x 1)
           ├── onlineXshifts                (total frames x 1)
           ├── onlineYshifts                (total frames x 1)
           ├── onlineZshifts                (total frames x 1)
           ├── trial_num_frames             (trials x 1)
           ├── frame_line_idxs              (total frames x 1)
           └── discard_frames               (total frames x 1)

Per-trial summary: unsupported external reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Python BandSILo does not emit this file.** The external format holds fields
that vary across analysis trials. Its trial axis matches the reference trial
table (all analysis trials); trials without alignment or source-extraction
data are represented by NaN slices::

   per_trial_summary.h5
   ├── row_major
   └── Path{n}/
       ├── sources/
       │   ├── temporal/
       │   │   └── per_trial_SNR            (trials x sources)
       │   └── spatial/
       │       ├── per_trial_profiles       (trials x sources x fastz x rows x cols)
       │       └── per_trial_coords         (trials x sources x 3 [z_loc, y_loc, x_loc])
       └── visualizations/
           ├── per_trial_mean_im            (trials x channels x fastz x rows x cols)
           ├── per_trial_act_im             (trials x fastz x rows x cols)
           ├── per_trial_act_im_peaks       (trials x max_peaks x 3 [z_loc, y_loc, x_loc])
           └── per_trial_num_peaks          (trials x 1)

Band registration lookup table: external input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

External MATLAB ``BandRegistration`` builds
``motion_correction/bandRegLookupTable.h5``. Python consumes this file;
registration and lookup-table generation are not implemented. XY search
limits are shared across paths; superpixel and reference-stack data are
per-path::

   bandRegLookupTable.h5
   ├── xPre
   ├── xPost
   ├── yPre
   ├── yPost
   ├── row_major
   └── Path{n}/
       ├── likelihood_means  (Y x X x Z x C x nSP)
       ├── allSuperPixelIDs  (nSP x 1)
       ├── sparseMaskInds    (N x 2)
       ├── zPre
       ├── zPost
       └── fastZ2RefZ

Field descriptions
------------------

These tables retain the full external reference fields alongside Python
outputs. Registration outputs are inputs to Python; all per-trial-summary
fields are external-reference only. The layout flag has the same meaning in
each table: ``1`` means the listed dimensions match h5py ``shape``; ``0`` or
absence means MATLAB layout.

Trial table fields
~~~~~~~~~~~~~~~~~~

.. list-table:: ``trial_table.h5`` (external input)
   :header-rows: 1
   :widths: 30 20 12 38

   * - Field
     - Size
     - Data type
     - Description
   * - ``row_major``
     - 1 x 1
     - uint8
     - Layout flag; if absent, assume column-major (``0``).
   * - ``datadr``
     - 1 x 1
     - string
     - Data directory location.
   * - ``savedr``
     - 1 x 1
     - string
     - Results directory location.
   * - ``filename``
     - nPaths x total trials
     - string (ragged)
     - Relative file name from ``datadr``.
   * - ``true_trial_ix``
     - nPaths x total trials
     - integer
     - Trial indices unraveled by epochs.
   * - ``epoch``
     - nPaths x total trials
     - integer
     - Epoch numbers.
   * - ``slap2_info``
     - —
     - group
     - Only saved for SLAP2 experiments.
   * - ``slap2_info/ref_stack/Path{n}/IM``
     - image dims
     - numeric
     - Reference stack image.
   * - ``slap2_info/ref_stack/Path{n}/channels``
     - 1 x nChannels
     - numeric
     - Color channels.
   * - ``slap2_info/ref_stack/Path{n}/Zs``
     - 1 x nZ
     - numeric
     - Z positions.
   * - ``slap2_info/ref_stack/Path{n}/dmdPixel2SampleTransform``
     - 3 x 3
     - numeric
     - Transformation matrix.
   * - ``slap2_info/first_line``
     - nPaths x total trials
     - integer
     - First line of each trial (1-indexed).
   * - ``slap2_info/last_line``
     - nPaths x total trials
     - integer
     - Last line of each trial (1-indexed).
   * - ``slap2_info/trial_start_time_inferred``
     - 1 x total trials
     - integer
     - Inferred trial start times.
   * - ``slap2_info/trial_end_time_from_pc``
     - 1 x total trials
     - integer
     - Trial end times from PC.
   * - ``motion_correction``
     - —
     - group
     - Written by the external motion-correction stage.
   * - ``motion_correction/fn_reg_ds``
     - nPaths x total trials
     - string
     - Registered and downsampled TIFF filename.
   * - ``motion_correction/fn_adata``
     - nPaths x total trials
     - string
     - Alignment metadata ``_ALIGNMENTDATA.h5`` filename.
   * - ``motion_correction/fn_raw``
     - nPaths x total trials
     - string
     - Registered raw-resolution file (Bergamo only).
   * - ``motion_correction/registration_failed``
     - nPaths x total trials
     - bool
     - Whether registration failed.
   * - ``motion_correction/first_line_original``
     - nPaths x total trials
     - integer
     - Original ``slap2_info/first_line`` before reVolt adjustment.
   * - ``motion_correction/align_params``
     - —
     - group/struct
     - Alignment parameters used.
   * - ``source_extraction``
     - —
     - group
     - Written by the external source-extraction stage.
   * - ``source_extraction/analysis_params``
     - —
     - group/struct
     - Analysis parameters used.
   * - ``source_extraction/fn_raw``
     - nPaths x total trials
     - string
     - Raw file source extraction reads from per trial.

Alignment data fields
~~~~~~~~~~~~~~~~~~~~~

All three external motion-correction backends write ``numChannels``,
``frametime``, ``alignHz``, ``motionDSc``, and ``motionDSr``.
``meanIM`` and ``recNegErr`` are written by StripRegistration and
MultiRoiRegistration, but not BandRegistration. ``motionC`` and ``motionR``
are StripRegistration/Bergamo-only. Both SLAP2 backends write ``DSframes``
and ``registrationFailed``. ``brightnessDS`` and ``logLikelihoodDS`` are
BandRegistration-only. The ``slap2`` group is populated only for SLAP2.

.. list-table:: ``<trial_stem>_ALIGNMENTDATA.h5`` (external registration input)
   :header-rows: 1
   :widths: 28 22 12 38

   * - Field
     - Size
     - Data type
     - Description
   * - ``row_major``
     - 1 x 1
     - uint8
     - Layout flag.
   * - ``numChannels``
     - 1 x 1
     - integer
     - Number of channels in the recording.
   * - ``meanIM``
     - channels x rows x cols
     - single
     - Per-channel mean of motion-corrected frames; not written by BandRegistration.
   * - ``frametime``
     - 1 x 1
     - numeric
     - Seconds per downsampled frame.
   * - ``alignHz``
     - 1 x 1
     - numeric
     - Frame rate (Hz) at which alignment was performed.
   * - ``motionDSc``
     - 1 x nDSframes
     - numeric
     - Inferred column shift per downsampled frame.
   * - ``motionDSr``
     - 1 x nDSframes
     - numeric
     - Inferred row shift per downsampled frame.
   * - ``motionDSz``
     - 1 x nDSframes
     - numeric
     - Inferred Z shift per downsampled frame. Always written by BandRegistration;
       written by MultiRoiRegistration only with ``refStackTemplate`` enabled;
       never written by StripRegistration.
   * - ``recNegErr``
     - 1 x nDSframes
     - numeric
     - Per-frame reconstruction error; alignment QC metric used for motion
       censoring. Not written by BandRegistration.
   * - ``brightnessDS``
     - nDSframes x channels
     - numeric
     - BandRegistration-only per-channel brightness/scaling factor at the
       selected motion shift.
   * - ``logLikelihoodDS``
     - nDSframes x 1
     - numeric
     - BandRegistration-only peak log-likelihood of the motion match per
       downsampled frame.
   * - ``motionC``
     - 1 x nFrames
     - numeric
     - Column shift upsampled to raw frame rate (Bergamo only).
   * - ``motionR``
     - 1 x nFrames
     - numeric
     - Row shift upsampled to raw frame rate (Bergamo only).
   * - ``motionZ``
     - 1 x nFrames
     - numeric
     - Reserved Z shift upsampled to raw frame rate; not written by any
       current external script.
   * - ``DSframes``
     - 1 x nDSframes
     - integer
     - Line indices of each downsampled frame (1-indexed); SLAP2 only,
       MultiRoiRegistration and BandRegistration.
   * - ``registrationFailed``
     - 1 x 1
     - bool
     - Whether registration failed for this trial; SLAP2 only,
       MultiRoiRegistration and BandRegistration.
   * - ``slap2``
     - —
     - group
     - Only saved for SLAP2 experiments.
   * - ``slap2/varFacDS``
     - rows x cols x nDSframes
     - numeric
     - MultiRoiRegistration-only variance factor; multiply pixel intensity
       by it to obtain a value proportional to its variance.
   * - ``slap2/Z_depths``
     - fastz x 1
     - numeric
     - MultiRoiRegistration-only imaged Z depths from microscope metadata.
   * - ``slap2/cropRow``
     - 1 x 1
     - integer
     - MultiRoiRegistration-only row offset to add to ROIs to index the
       original recording.
   * - ``slap2/cropCol``
     - 1 x 1
     - integer
     - MultiRoiRegistration-only column offset to add to ROIs to index the
       original recording.
   * - ``slap2/viewC``
     - (rows+2·maxshift) x (cols+2·maxshift)
     - numeric
     - MultiRoiRegistration-only column interpolation grid for remapping
       into saved TIFF space.
   * - ``slap2/viewR``
     - (rows+2·maxshift) x (cols+2·maxshift)
     - numeric
     - MultiRoiRegistration-only row interpolation grid for remapping
       into saved TIFF space.
   * - ``slap2/trimRows``
     - 1 x nTrimRows
     - integer
     - MultiRoiRegistration-only row indices used to remap images from
       the datafile into saved TIFF space.
   * - ``slap2/trimCols``
     - 1 x nTrimCols
     - integer
     - MultiRoiRegistration-only column indices used to remap images from
       the datafile into saved TIFF space.
   * - ``slap2/onlineMotionXshift``
     - 1 x nDSframes
     - numeric
     - Online motion-correction X shift from the microscope.
   * - ``slap2/onlineMotionYshift``
     - 1 x nDSframes
     - numeric
     - Online motion-correction Y shift from the microscope.
   * - ``slap2/onlineMotionZshift``
     - 1 x nDSframes
     - numeric
     - Online motion-correction Z shift from the microscope.

Band registration lookup fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

External ``BandRegistration`` writes the cached lookup table into
``motion_correction/`` on its first run and loads it on later runs. XY search
limits (``xPre``, ``yPre``, etc.) are shared across paths; each ``Path{n}``
holds superpixel and reference-stack data for one DMD in trial-table order.
``Y``, ``X``, and ``Z`` denote row, column, and reference-stack Z dimensions
of the search cube (``yPre + yPost + 1``, etc.).

.. list-table:: ``bandRegLookupTable.h5`` (external registration input)
   :header-rows: 1
   :widths: 28 22 12 38

   * - Field
     - Size
     - Data type
     - Description
   * - ``row_major``
     - 1 x 1
     - uint8
     - Layout flag.
   * - ``xPre``
     - 1 x 1
     - numeric
     - Maximum column shift searched before the reference position (pixels);
       equals ``align_params.maxshiftXY``.
   * - ``xPost``
     - 1 x 1
     - numeric
     - Maximum column shift searched after the reference position (pixels);
       equals ``align_params.maxshiftXY``.
   * - ``yPre``
     - 1 x 1
     - numeric
     - Maximum row shift searched before the reference position (pixels);
       equals ``align_params.maxshiftXY``.
   * - ``yPost``
     - 1 x 1
     - numeric
     - Maximum row shift searched after the reference position (pixels);
       equals ``align_params.maxshiftXY``.
   * - ``Path{n}``
     - —
     - group
     - One group per imaging path (DMD).
   * - ``Path{n}/likelihood_means``
     - Y x X x Z x C x nSP
     - single
     - Precomputed expected superpixel mean intensity in the padded reference
       stack at each search-cube displacement, per channel and superpixel;
       template for Poisson or correlation motion inference.
   * - ``Path{n}/allSuperPixelIDs``
     - nSP x 1
     - numeric
     - Unique keys ``superPixIdx * 100 + zIdx`` for this path; band-scan pixels
       only when ``bandOnly`` is true.
   * - ``Path{n}/sparseMaskInds``
     - N x 2
     - numeric
     - Sparse ROI definition. Column 1: linear DMD pixel index in
       ``rows x cols x numFastZs`` layout. Column 2: superpixel index (1 … nSP).
   * - ``Path{n}/zPre``
     - 1 x 1
     - numeric
     - Maximum reference-stack Z shift searched before the matched plane
       (planes); capped by ``align_params.maxshiftZ`` and available reference
       Z planes.
   * - ``Path{n}/zPost``
     - 1 x 1
     - numeric
     - Maximum reference-stack Z shift searched after the matched plane
       (planes); capped as for ``zPre``.
   * - ``Path{n}/fastZ2RefZ``
     - numFastZs x 1
     - numeric
     - Maps each imaged fast-Z index to its nearest reference-stack Z plane
       index, used when sampling ``likelihood_means``.

Annotation fields
~~~~~~~~~~~~~~~~~

New files set ``coords_zero_indexed=1`` and store ``position`` / ``center``
as **0-indexed** ``[y_loc, x_loc]`` (row, column). Legacy files without the
flag use MATLAB ``images.roi`` convention: **1-indexed** ``[x, y]``
(column, row).

.. list-table:: ``annotations.h5``
   :header-rows: 1
   :widths: 28 22 12 38

   * - Field
     - Size
     - Data type
     - Description
   * - ``row_major``
     - 1 x 1
     - uint8
     - Layout flag.
   * - ``coords_zero_indexed``
     - 1 x 1
     - uint8
     - When ``1``, ``position`` / ``center`` are 0-indexed
       ``[y_loc, x_loc]``; when absent or ``0``, legacy 1-indexed ``[x, y]``.
   * - ``Path{n}``
     - —
     - group
     - One group per imaging path in trial-table order.
   * - ``Path{n}/dr``
     - 1 x nChars
     - uint16
     - Motion-correction directory used while drawing these ROIs.
   * - ``Path{n}/fn``
     - 1 x nChars
     - uint16
     - Trial stem used when displaying the ROI GUI.
   * - ``Path{n}/n_rois``
     - 1 x 1
     - uint32
     - Number of saved ROI entries for this path.
   * - ``Path{n}/roi_###/type``
     - 1 x nChars
     - uint16
     - ROI geometry: ``polygon``, ``circle``, or ``ellipse``.
   * - ``Path{n}/roi_###/label``
     - 1 x nChars
     - uint16
     - User label, for example ``SOMA``.
   * - ``Path{n}/roi_###/mask``
     - rows x cols
     - uint8
     - Binary ROI mask in image coordinates (1 = included pixel).
   * - ``Path{n}/roi_###/position``
     - nVertices x 2
     - double
     - Polygon vertices ``[y_loc, x_loc]`` when ``coords_zero_indexed=1``;
       otherwise legacy ``[x, y]``.
   * - ``Path{n}/roi_###/center``
     - 1 x 2
     - double
     - Circle/ellipse center ``[y_loc, x_loc]`` when
       ``coords_zero_indexed=1``; otherwise legacy ``[x, y]``.
   * - ``Path{n}/roi_###/semi_axes``
     - 1 x 2
     - double
     - Ellipse semi-axis lengths (ellipse only).
   * - ``Path{n}/roi_###/rotation_angle``
     - 1 x 1
     - double
     - Ellipse rotation angle in degrees (ellipse only).
   * - ``Path{n}/roi_###/radius``
     - 1 x 1
     - double
     - Circle radius (circle only).

Experiment summary fields
~~~~~~~~~~~~~~~~~~~~~~~~~

Pixel/plane coordinates are **0-indexed** ``[z_loc, y_loc, x_loc]``;
``frame_line_idxs`` remains **1-indexed**. The Python writer emits every
dataset in the summary tree, including ``SNR`` and ``offlineZshifts``.
External summaries may omit SNR or Z shifts when unavailable. Missing model
SNR values are serialized as NaN.

.. list-table:: ``experiment_summary.h5``
   :header-rows: 1
   :widths: 30 22 10 38

   * - Field
     - Size
     - Data type
     - Description
   * - ``row_major``
     - 1 x 1
     - uint8
     - Layout flag.
   * - ``params``
     - —
     - group/struct
     - Analysis parameters. External MATLAB ``activityChannel``, if present,
       is 1-indexed into ``numChannels``. Python activity extraction uses
       channel zero, not this reference selector.
   * - ``Path{n}``
     - —
     - group
     - One group per imaging path.
   * - ``Path{n}/Z_depths``
     - fastz x 1
     - numeric
     - Z depths per imaging plane (SLAP2).
   * - ``Path{n}/frame_info/offlineXshifts``
     - total frames x 1
     - numeric
     - Offline registration X displacement per frame, using the public
       alignment sign convention above.
   * - ``Path{n}/frame_info/offlineYshifts``
     - total frames x 1
     - numeric
     - Offline registration Y displacement per frame.
   * - ``Path{n}/frame_info/offlineZshifts``
     - total frames x 1
     - numeric
     - Offline registration Z displacement per frame. Emitted by Python
       BandSILo; optional in the external schema when 3D alignment is absent.
   * - ``Path{n}/frame_info/onlineXshifts``
     - total frames x 1
     - numeric
     - SLAP2 online X shift per frame.
   * - ``Path{n}/frame_info/onlineYshifts``
     - total frames x 1
     - numeric
     - SLAP2 online Y shift per frame.
   * - ``Path{n}/frame_info/onlineZshifts``
     - total frames x 1
     - numeric
     - SLAP2 online Z shift per frame.
   * - ``Path{n}/frame_info/trial_num_frames``
     - trials x 1
     - integer
     - Number of frames contributed by each analysis trial.
   * - ``Path{n}/frame_info/frame_line_idxs``
     - total frames x 1
     - integer
     - Raw line index (SLAP2) or frame index (other microscopes in the
       external schema) for each stitched-series frame; 1-indexed.
   * - ``Path{n}/frame_info/discard_frames``
     - total frames x 1
     - bool or uint8
     - Frame excluded from analysis, for example by motion censoring.
   * - ``Path{n}/visualizations/mean_im``
     - channels x fastz x rows x cols
     - numeric
     - Mean registered image per channel / Z slice.
   * - ``Path{n}/visualizations/act_im``
     - fastz x rows x cols
     - numeric
     - Activity / localization summary image.
   * - ``Path{n}/visualizations/act_im_peaks``
     - sources x 3
     - numeric
     - Activity-image peak locations ``[z_loc, y_loc, x_loc]``, 0-indexed;
       Python uses source row/column coordinates with ``z_loc`` fixed at ``0``.
   * - ``Path{n}/global/F``
     - channels x total frames
     - numeric
     - Whole-field fluorescence traces, one row per channel.
   * - ``Path{n}/user_rois/labels``
     - rois x 1
     - string
     - User-defined ROI labels (UTF-8 strings in Python summary output).
   * - ``Path{n}/user_rois/mask``
     - rois x fastz x rows x cols
     - uint8 or bool
     - Stacked binary masks for each user ROI.
   * - ``Path{n}/user_rois/Fsvd``
     - rois x channels x total frames
     - numeric
     - ROI SVD / projection signals. Emitted as NaN unless explicitly
       supplied through the model; not computed by BandSILo or copied from ``F``.
   * - ``Path{n}/user_rois/F``
     - rois x channels x total frames
     - numeric
     - ROI fluorescence (raw or baseline-corrected according to the producer).
   * - ``Path{n}/user_rois`` attribute ``annotation_enabled``
     - scalar attribute
     - bool
     - Python annotation policy: distinguishes disabled from enabled-but-empty
       annotations. Both produce zero-ROI ``labels``, ``mask``, ``Fsvd``, and
       ``F`` datasets with the remaining dimensions intact.
   * - ``Path{n}/sources/spatial/profiles``
     - sources x fastz x rows x cols
     - numeric
     - Spatial component / pixel weights per source, averaged across trials
       with footprints.
   * - ``Path{n}/sources/spatial/coords``
     - sources x 3
     - numeric
     - Source centers, one ``[z_loc, y_loc, x_loc]`` per row, 0-indexed.
   * - ``Path{n}/sources/temporal/dF_ls``
     - sources x channels x total frames
     - numeric
     - Least-squares fluorescence change (absolute or scaled); Python uses
       baseline re-estimation and processes activity channel zero only.
   * - ``Path{n}/sources/temporal/dF_denoised``
     - sources x channels x total frames
     - numeric
     - Fitted convolution fluorescence change, saved alongside least squares
       without a second baseline subtraction. Missing optional model data
       serialize as NaN.
   * - ``Path{n}/sources/temporal/events``
     - sources x channels x total frames
     - numeric
     - Nonnegative fitted event amplitudes; excludes each trial's initial
       state. Discarded frames and non-activity channels are NaN.
   * - ``Path{n}/sources/temporal/F0``
     - sources x channels x total frames
     - numeric
     - Baseline estimate used for normalization; based on QR, not the
       transient model.
   * - ``Path{n}/sources/temporal/SNR``
     - sources x 1
     - numeric
     - Per-source signal-to-noise ratio metric. Python emits this dataset
       (NaN for missing model values); optional in external summaries.

Per-trial summary fields: unsupported external reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Python does not emit this file.** The reference trial axis includes all
analysis trials in the trial table. Trials without alignment or extraction
data have NaN slices. Coordinates use **0-indexed**
``[z_loc, y_loc, x_loc]``.

.. list-table:: ``per_trial_summary.h5`` (external-reference only)
   :header-rows: 1
   :widths: 30 22 10 38

   * - Field
     - Size
     - Data type
     - Description
   * - ``row_major``
     - 1 x 1
     - uint8
     - Layout flag.
   * - ``Path{n}``
     - —
     - group
     - One group per imaging path.
   * - ``Path{n}/visualizations/per_trial_mean_im``
     - trials x channels x fastz x rows x cols
     - numeric
     - Trial-aligned mean registered image per channel / Z slice.
   * - ``Path{n}/visualizations/per_trial_act_im``
     - trials x fastz x rows x cols
     - numeric
     - Trial-aligned activity / localization summary image.
   * - ``Path{n}/visualizations/per_trial_act_im_peaks``
     - trials x max_peaks x 3
     - numeric
     - Per-trial detected peak locations ``[z_loc, y_loc, x_loc]``, 0-indexed;
       NaN-padded when a trial has fewer than ``max_peaks``.
   * - ``Path{n}/visualizations/per_trial_num_peaks``
     - trials x 1
     - integer
     - Number of valid peaks per trial.
   * - ``Path{n}/sources/spatial/per_trial_profiles``
     - trials x sources x fastz x rows x cols
     - numeric
     - Spatial component / pixel weights per source per trial.
   * - ``Path{n}/sources/spatial/per_trial_coords``
     - trials x sources x 3
     - numeric
     - Source centers per trial, ``[z_loc, y_loc, x_loc]``, 0-indexed.
   * - ``Path{n}/sources/temporal/per_trial_SNR``
     - trials x sources
     - numeric
     - Per-source SNR for each analysis trial.