# GIAnT-python

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-100.0%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![Python](https://img.shields.io/badge/python->=3.10-blue?logo=python)

Python translation of the [GIAnT-MATLAB](https://github.com/AllenNeuralDynamics/GIAnT-MATLAB) analysis package.

## Level of Support
 - [ ] Supported: We are releasing this code to the public as a tool we expect others to use. Issues are welcomed, and we expect to address them promptly; pull requests will be vetted by our staff before inclusion.
 - [ ] Occasional updates: We are planning on occasional updating this tool with no fixed schedule. Community involvement is encouraged through both issues and pull requests.
 - [ ] Unsupported: We are not currently supporting this code, but simply releasing it to the community AS IS but are not able to provide any guarantees of support. The community is welcome to submit issues, but you should not expect an active response.

## Installation
To use the software, in the root directory, run
```bash
pip install -e .
```

To develop the code, run
```bash
pip install -e . --group dev
```
Note: --group flag is available only in pip versions >=25.1

Alternatively, if using `uv`, run
```bash
uv sync
```

## Epoch and Analysis Trial

For each experiment we run through the pipeline, we break down the data into epochs and analysis trials.

Epochs are full experimental sessions that can be aligned with each other (i.e. the same field of view and regions of interest are being imaged). Analysis trials are generally contiguous subsets (in time) of an epoch. These analysis trials may not align exactly with experimental trials.

For SLAP2, analysis trials are the experimental trials if the data was collected using the multi-trial functions of the SLAP2 and each trial is saved off the microscope in a different file. If data was continuously collected on SLAP2, the experiment will be split up into analysis trials of length 200000 lines (~20 sec) to help parallelize processing.

For data not collected on SLAP2, the current GIAnT pipeline sets Epochs to be 1 and each file that is selected to be processed is an analysis trial. These analysis trials must be able to be aligned to one another.

## Pipeline Outputs

GIAnT-python writes the **same set of HDF5 output files with the same field names, shapes, dtypes, and conventions** as [GIAnT-MATLAB](https://github.com/AllenNeuralDynamics/GIAnT-MATLAB), so files produced by either toolbox are interchangeable. The schemas below are the on-disk contract; the typed data models in `giant_python.models` (`TrialTable`, `AlignmentData`, `ExperimentSummary`, ...) mirror these structures and own the `from_h5` / `to_h5` (de)serialization.

### Reading and writing H5

GIAnT-python reads and writes these files with [`h5py`](https://www.h5py.org/), which stores arrays in **row-major (C) order**. Files written by GIAnT-python therefore set the `row_major` flag to `1`, and **every dimension tuple listed below matches the h5py `shape` directly** (i.e. read the tuples left-to-right as the axis order returned by `h5py`).

Files produced by GIAnT-**MATLAB** instead set `row_major` to `0` (or omit it), because MATLAB stores arrays column-major and its dimension tuples follow MATLAB `size()`. GIAnT-python's loaders inspect the `row_major` flag and **permute axes as needed**, so a MATLAB-written file and a Python-written file deserialize to the same logical arrays. When in doubt:

| `row_major` value | Layout | Axis order of the tuples below |
| --- | --- | --- |
| `1` | row-major (h5py / NumPy C order) | matches h5py `shape` as written here |
| `0` or absent | column-major (MATLAB `size()`) | reverse of the tuples here |

**Vector shapes are preserved on disk.** Fields documented as `1 x N` (row vectors) or `N x 1` (column vectors) are written as rank-2 datasets with those exact dimensions — e.g. `motionDSc` is shape `(1, nDSframes)`, not a squeezed `(nDSframes,)`. GIAnT-python does **not** collapse singleton dimensions when saving. Downstream code can therefore use the same size checks and indexing on MATLAB- or Python-generated files after handling the `row_major` flag (and any required axis permutation for multi-D arrays).

### Index and coordinate conventions

These conventions are shared with GIAnT-MATLAB and preserved on both read and write:

- **Line/frame indices** (`first_line`, `last_line`, `DSframes`, `frame_line_idxs`) are **1-indexed** to retain the SLAP2 line-indexing convention.
- **Spatial coordinate fields** in the summaries (`sources/spatial/coords`, `act_im_peaks`, `per_trial_*_coords`, etc.) are **0-indexed** `[z_loc, y_loc, x_loc]`, matching image axis order (`fastz`, rows, cols). `z_loc` is the 0-based index into the `fastz` axis.
- **Annotation coordinate fields** (`position`, `center`) are **0-indexed** `[y_loc, x_loc]` when `coords_zero_indexed = 1`; legacy files without the flag use MATLAB `images.roi` convention (1-indexed `[x, y]`).
- **Images** are stored `rows x cols` (row = Y, col = X). Motion shifts: `motionDSr` / `motionR` are **row** shifts, `motionDSc` / `motionC` are **column** shifts.
- `params/activityChannel` is **1-indexed** into the recording's `numChannels` channels.

Legend for the trees below (🗄️ file · 📁 group · 🔤 string · 🔢 integer · 📈 numeric · 🖼️ image · ☑️ bool).

### Trial Table

Each experiment processed with GIAnT first gets a `trial_table.h5` file that summarizes relevant file locations and analysis trial structures. The `slap2_info` group is only populated for SLAP2 experiments. The `motion_correction` and `source_extraction` groups are populated by downstream pipeline stages and will only be present once those stages have run. (Model: `giant_python.models.TrialTable` / `Slap2Info`.)

```
🗄️ trial_table.h5
 ├ 🔤 datadr
 ├ 🔤 savedr
 ├ 🔤 filename
 ├ 🔢 true_trial_ix
 ├ 🔢 epoch
 ├ ☑️ row_major
 ├ 📁 slap2_info
 |  ├ 📁 ref_stack
 |  |  └ 📁 Path{1,2}
 |  |     ├ 🖼️ IM
 |  |     ├ 🔢 channels
 |  |     ├ 📈 Zs
 |  |     └ 📈 dmdPixel2SampleTransform
 |  ├ 🔢 first_line
 |  ├ 🔢 last_line
 |  ├ 🔢 trial_start_time_inferred
 |  └ 🔢 trial_end_time_from_pc
 ├ 📁 motion_correction
 |  ├ 🔤 fn_reg_ds
 |  ├ 🔤 fn_adata
 |  ├ 🔤 fn_raw
 |  ├ ☑️ registration_failed
 |  ├ 🔢 first_line_original
 |  └ 📁 align_params
 └ 📁 source_extraction
    ├ 📁 analysis_params
    └ 🔤 fn_raw
```

### Alignment Data

The motion correction stage saves a H5 file ending in `_ALIGNMENTDATA.h5` that contains the alignment data for each trial. (Model: `giant_python.models.AlignmentData`.)

```
🗄️ <trial_stem>_ALIGNMENTDATA.h5
 ├ ☑️ row_major
 ├ 📈 numChannels
 ├ 📈 frametime
 ├ 📈 alignHz
 ├ 📈 motionDSc
 ├ 📈 motionDSr
 ├ 📈 motionDSz           (BandRegistration always; MultiRoiRegistration when refStackTemplate is enabled)
 ├ 🖼️ meanIM              (StripRegistration and MultiRoiRegistration only; not written by BandRegistration)
 ├ 📈 recNegErr           (StripRegistration and MultiRoiRegistration only; not written by BandRegistration)
 ├ 📈 motionC             (StripRegistration / Bergamo only)
 ├ 📈 motionR             (StripRegistration / Bergamo only)
 ├ 📈 motionZ             (reserved; not written by any current script)
 ├ 📈 brightnessDS        (BandRegistration only)
 ├ 📈 logLikelihoodDS     (BandRegistration only)
 ├ 🔢 DSframes            (SLAP2 only: MultiRoiRegistration and BandRegistration)
 ├ ☑️ registrationFailed  (SLAP2 only: MultiRoiRegistration and BandRegistration)
 └ 📁 slap2               (SLAP2 only)
    ├ 📈 onlineMotionXshift
    ├ 📈 onlineMotionYshift
    ├ 📈 onlineMotionZshift
    ├ 🖼️ varFacDS          (MultiRoiRegistration only)
    ├ 📈 Z_depths          (MultiRoiRegistration only)
    ├ 🔢 cropRow           (MultiRoiRegistration only)
    ├ 🔢 cropCol           (MultiRoiRegistration only)
    ├ 🖼️ viewC             (MultiRoiRegistration only)
    ├ 🖼️ viewR             (MultiRoiRegistration only)
    ├ 🔢 trimRows          (MultiRoiRegistration only)
    └ 🔢 trimCols          (MultiRoiRegistration only)
```

### Manual Annotations

Users can manually annotate pixels to exclude from analysis, or pixels that correspond to soma whose signals should be extracted. When ROIs are annotated, information about the ROIs is saved in the `annotations.h5` file. String fields are stored as UTF-16 code units (`uint16`) for robust MATLAB/Python compatibility.

```
🗄️ annotations.h5
 ├ ☑️ row_major
 ├ ☑️ coords_zero_indexed
 └ 📁 Path{1,2}
    ├ 🔤 dr
    ├ 🔤 fn
    ├ 🔢 n_rois
    └ 📁 roi_###
       ├ 🔤 type
       ├ 🔤 label
       ├ 🖼️ mask
       ├ 📈 position (polygon only; nVertices x 2 [y_loc, x_loc] when flagged)
       ├ 📈 center (circle/ellipse; 1 x 2 [y_loc, x_loc] when flagged)
       ├ 📈 semi_axes (ellipse)
       ├ 📈 rotation_angle (ellipse)
       └ 📈 radius (circle)
```

### Experiment Summary

The final step of the pipeline, source extraction (Source Identification by Activity Localization; SILo), outputs an `experiment_summary.h5` file which contains the extracted sources as well as other useful data about the experiment. Dimensions use one `total frames` axis for all trials from that path stitched in time. (Model: `giant_python.models.ExperimentSummary` / `Source` / `UserRoi` / `Visualizations`.)

```
🗄️ experiment_summary.h5
 ├ ☑️ row_major
 ├ 📁 params
 └ 📁 Path{1,2}
    ├ 📈 Z_depths (fastz x 1)
    ├ 📁 sources
    |  ├ 📁 temporal
    |  |  ├ 📈 dF_ls (sources x channels x total frames)
    |  |  ├ 📈 dF_denoised (sources x channels x total frames)
    |  |  ├ 📈 events (sources x channels x total frames)
    |  |  ├ 📈 F0 (sources x channels x total frames)
    |  |  └ 📈 SNR (sources x 1)
    |  └ 📁 spatial
    |     ├ 🖼️ profiles (sources x fastz x rows x cols)
    |     └ 📈 coords (sources x 3 [z_loc, y_loc, x_loc])
    ├ 📁 user_rois
    |  ├ 🔤 labels (rois x 1)
    |  ├ 🖼️ mask (rois x fastz x rows x cols)
    |  ├ 📈 Fsvd (rois x channels x total frames)
    |  └ 📈 F (rois x channels x total frames)
    ├ 📁 visualizations
    |  ├ 🖼️ mean_im (channels x fastz x rows x cols)
    |  ├ 🖼️ act_im (fastz x rows x cols)
    |  └ 🖼️ act_im_peaks (sources x 3 [z_loc, y_loc, x_loc])
    ├ 📁 global
    |  └ 📈 F (channels x total frames)
    └ 📁 frame_info
       ├ 📈 offlineXshifts (total frames x 1)
       ├ 📈 offlineYshifts (total frames x 1)
       ├ 📈 offlineZshifts (total frames x 1)
       ├ 📈 onlineXshifts (total frames x 1)
       ├ 📈 onlineYshifts (total frames x 1)
       ├ 📈 onlineZshifts (total frames x 1)
       ├ 🔢 trial_num_frames (trials x 1)
       ├ 🔢 frame_line_idxs (total frames x 1)
       └ ☑️ discard_frames (total frames x 1)
```

A summary file of per-trial data is also saved as `per_trial_summary.h5` for any fields that may vary across analysis trials. The trial axis matches `trial_table.h5` (all analysis trials); trials without alignment or source-extraction data are left as NaN in the corresponding slices.

```
🗄️ per_trial_summary.h5
 ├ ☑️ row_major
 └ 📁 Path{1,2}
    ├ 📁 sources
    |  ├ 📁 temporal
    |  |  └ 📈 per_trial_SNR (trials x sources)
    |  └ 📁 spatial
    |     ├ 🖼️ per_trial_profiles (trials x sources x fastz x rows x cols)
    |     └ 📈 per_trial_coords (trials x sources x 3 [z_loc, y_loc, x_loc])
    └ 📁 visualizations
       ├ 🖼️ per_trial_mean_im (trials x channels x fastz x rows x cols)
       ├ 🖼️ per_trial_act_im (trials x fastz x rows x cols)
       ├ 🖼️ per_trial_act_im_peaks (trials x max_peaks x 3 [z_loc, y_loc, x_loc])
       └ 🔢 per_trial_num_peaks (trials x 1)
```

### Band Registration Lookup Table (intermediate file)

`BandRegistration` builds this file once per experiment under `motion_correction/bandRegLookupTable.h5` and reuses it on subsequent runs. XY search limits (`xPre`, `yPre`, etc.) are shared across paths; per-path superpixel and reference-stack data live under `Path{n}`.

```
🗄️ bandRegLookupTable.h5
 ├ 🔢 xPre
 ├ 🔢 xPost
 ├ 🔢 yPre
 ├ 🔢 yPost
 ├ ☑️ row_major
 └ 📁 Path{1,2}
    ├ 📈 likelihood_means (Y x X x Z x C x nSP)
    ├ 🔢 allSuperPixelIDs (nSP x 1)
    ├ 🔢 sparseMaskInds (N x 2)
    ├ 🔢 zPre
    ├ 🔢 zPost
    └ 📈 fastZ2RefZ
```

## File Field Descriptions

### `trial_table.h5`

| Field | Size | Data type | Description |
| --- | --- | --- | --- |
| `row_major` | 1 x 1 | uint8 | Layout flag: `1` = row-major (sizes match h5py `shape`; written by GIAnT-python); `0` = column-major (MATLAB `size()`). **If absent, assume column-major (`0`).** |
| `datadr` | 1 x 1 | string | Data directory location |
| `savedr` | 1 x 1 | string | Results directory location |
| `filename` | nPaths x total trials | string (ragged) | Relative file name from `datadr` |
| `true_trial_ix` | nPaths x total trials | integer | Trial indices unraveled by epochs |
| `epoch` | nPaths x total trials | integer | Epoch numbers |
| `slap2_info` | — | group | Only saved for SLAP2 experiments |
| `slap2_info/ref_stack/Path{1,2}/IM` | image dims | numeric | Reference stack image |
| `slap2_info/ref_stack/Path{1,2}/channels` | 1 x nChannels | numeric | Color channels |
| `slap2_info/ref_stack/Path{1,2}/Zs` | 1 x nZ | numeric | Z positions |
| `slap2_info/ref_stack/Path{1,2}/dmdPixel2SampleTransform` | 3 x 3 | numeric | Transformation matrix |
| `slap2_info/first_line` | nPaths x total trials | integer | First line of each trial (1-indexed) |
| `slap2_info/last_line` | nPaths x total trials | integer | Last line of each trial (1-indexed) |
| `slap2_info/trial_start_time_inferred` | 1 x total trials | integer | Inferred trial start times |
| `slap2_info/trial_end_time_from_pc` | 1 x total trials | integer | Trial end times from PC |
| `motion_correction` | — | group | Written by motion correction stage |
| `motion_correction/fn_reg_ds` | nPaths x total trials | string | Registered + downsampled tif filename |
| `motion_correction/fn_adata` | nPaths x total trials | string | Alignment metadata `_ALIGNMENTDATA.h5` filename |
| `motion_correction/fn_raw` | nPaths x total trials | string | Registered raw-resolution file (Bergamo only) |
| `motion_correction/registration_failed` | nPaths x total trials | bool | Whether registration failed |
| `motion_correction/first_line_original` | nPaths x total trials | integer | Original `slap2_info/first_line` before reVolt adjustment |
| `motion_correction/align_params` | — | group/struct | Alignment parameters used |
| `source_extraction` | — | group | Written by source extraction stage |
| `source_extraction/analysis_params` | — | group/struct | Analysis parameters used |
| `source_extraction/fn_raw` | nPaths x total trials | string | Raw file source extraction reads from per trial |

### `<trial_stem>_ALIGNMENTDATA.h5`

Top-level fields written by **all three** motion correction backends: `numChannels`, `frametime`, `alignHz`, `motionDSc`, `motionDSr`. `meanIM` and `recNegErr` are written by StripRegistration and MultiRoiRegistration but **not** by BandRegistration. `motionC`/`motionR` are written only by StripRegistration (Bergamo); `DSframes`/`registrationFailed` by both SLAP2 backends (MultiRoiRegistration and BandRegistration); `brightnessDS`/`logLikelihoodDS` by BandRegistration only. The `slap2` group is only populated for SLAP2 experiments.

| Field | Size | Data type | Description |
| --- | --- | --- | --- |
| `row_major` | 1 x 1 | uint8 | Layout flag (see above) |
| `numChannels` | 1 x 1 | integer | Number of channels in the recording |
| `meanIM` | channels x rows x cols | single | Per-channel mean of motion-corrected frames (not written by BandRegistration) |
| `frametime` | 1 x 1 | numeric | Seconds per downsampled frame |
| `alignHz` | 1 x 1 | numeric | Frame rate (Hz) at which alignment was performed |
| `motionDSc` | 1 x nDSframes | numeric | Inferred column shift per downsampled frame |
| `motionDSr` | 1 x nDSframes | numeric | Inferred row shift per downsampled frame |
| `motionDSz` | 1 x nDSframes | numeric | Inferred Z shift per downsampled frame; always written by BandRegistration; written by MultiRoiRegistration only when `refStackTemplate` is enabled; never written by StripRegistration |
| `recNegErr` | 1 x nDSframes | numeric | Per-frame reconstruction error; alignment QC metric and used for motion censoring (not written by BandRegistration) |
| `brightnessDS` | nDSframes x channels | numeric | (BandRegistration only) Per-channel brightness/scaling factor at the selected motion shift |
| `logLikelihoodDS` | nDSframes x 1 | numeric | (BandRegistration only) Peak log-likelihood of the motion match per downsampled frame |
| `motionC` | 1 x nFrames | numeric | Column shift upsampled to raw frame rate (Bergamo only) |
| `motionR` | 1 x nFrames | numeric | Row shift upsampled to raw frame rate (Bergamo only) |
| `motionZ` | 1 x nFrames | numeric | (reserved; not written by any current script) Z shift upsampled to raw frame rate |
| `DSframes` | 1 x nDSframes | integer | Line indices of each downsampled frame (SLAP2 only: MultiRoiRegistration and BandRegistration; 1-indexed) |
| `registrationFailed` | 1 x 1 | bool | Whether registration failed for this trial (SLAP2 only: MultiRoiRegistration and BandRegistration) |
| `slap2` | — | group | Only saved for SLAP2 experiments |
| `slap2/varFacDS` | rows x cols x nDSframes | numeric | (MultiRoiRegistration only) Variance factor; multiply pixel intensity to get a value proportional to its variance |
| `slap2/Z_depths` | fastz x 1 | numeric | (MultiRoiRegistration only) Imaged Z depths from microscope metadata |
| `slap2/cropRow` | 1 x 1 | integer | (MultiRoiRegistration only) Row offset to add to ROIs to index into original recording |
| `slap2/cropCol` | 1 x 1 | integer | (MultiRoiRegistration only) Column offset to add to ROIs to index into original recording |
| `slap2/viewC` | (rows+2·maxshift) x (cols+2·maxshift) | numeric | (MultiRoiRegistration only) Column interpolation grid for remapping into saved tiff space |
| `slap2/viewR` | (rows+2·maxshift) x (cols+2·maxshift) | numeric | (MultiRoiRegistration only) Row interpolation grid for remapping into saved tiff space |
| `slap2/trimRows` | 1 x nTrimRows | integer | (MultiRoiRegistration only) Row indices used to remap images from the datafile into saved tiff space |
| `slap2/trimCols` | 1 x nTrimCols | integer | (MultiRoiRegistration only) Column indices used to remap images from the datafile into saved tiff space |
| `slap2/onlineMotionXshift` | 1 x nDSframes | numeric | Online motion-correction X shift from the microscope |
| `slap2/onlineMotionYshift` | 1 x nDSframes | numeric | Online motion-correction Y shift from the microscope |
| `slap2/onlineMotionZshift` | 1 x nDSframes | numeric | Online motion-correction Z shift from the microscope |

### `bandRegLookupTable.h5`

`BandRegistration` writes this cached lookup table to `motion_correction/` on the first run and loads it on later runs. XY search limits (`xPre`, `yPre`, etc.) are shared across paths; per-path superpixel and reference-stack data live under `Path{n}` (one group per DMD, in trial-table path order). `Y`, `X`, and `Z` are the row, column, and reference-stack Z dimensions of the motion search cube (`yPre + yPost + 1`, etc.).

| Field | Size | Data type | Description |
| --- | --- | --- | --- |
| `row_major` | 1 x 1 | uint8 | Layout flag (see above) |
| `xPre` | 1 x 1 | numeric | Maximum column shift searched **before** the reference position (pixels); equals `align_params.maxshiftXY` |
| `xPost` | 1 x 1 | numeric | Maximum column shift searched **after** the reference position (pixels); equals `align_params.maxshiftXY` |
| `yPre` | 1 x 1 | numeric | Maximum row shift searched **before** the reference position (pixels); equals `align_params.maxshiftXY` |
| `yPost` | 1 x 1 | numeric | Maximum row shift searched **after** the reference position (pixels); equals `align_params.maxshiftXY` |
| `Path{n}` | — | group | One group per imaging path (DMD) |
| `Path{n}/likelihood_means` | Y x X x Z x C x nSP | single | Precomputed expected superpixel mean intensity in the padded reference stack at each displacement in the search cube, per channel and superpixel; used as the template for Poisson or correlation motion inference |
| `Path{n}/allSuperPixelIDs` | nSP x 1 | numeric | Unique superpixel keys for this path: `superPixIdx * 100 + zIdx` (integration-mode pixels only when `integrationOnly` is true) |
| `Path{n}/sparseMaskInds` | N x 2 | numeric | Sparse ROI definition: column 1 = linear DMD pixel index (`rows x cols x numFastZs` layout); column 2 = superpixel index (1 … nSP) |
| `Path{n}/zPre` | 1 x 1 | numeric | Maximum reference-stack Z shift searched **before** the matched plane (planes); capped by `align_params.maxshiftZ` and available reference Z planes |
| `Path{n}/zPost` | 1 x 1 | numeric | Maximum reference-stack Z shift searched **after** the matched plane (planes); capped similarly to `zPre` |
| `Path{n}/fastZ2RefZ` | numFastZs x 1 | numeric | Maps each imaged fast-Z index to the nearest reference-stack Z plane index (used when sampling `likelihood_means`) |

### `annotations.h5`

**Indexing conventions.** New files set `coords_zero_indexed` to `1` and store `position` / `center` as **0-indexed `[y_loc, x_loc]`** (row, column). Legacy files without this flag use MATLAB `images.roi` convention: **1-indexed `[x, y]`** (column, row).

| Field | Size | Data type | Description |
| --- | --- | --- | --- |
| `row_major` | 1 x 1 | uint8 | Layout flag (see above) |
| `coords_zero_indexed` | 1 x 1 | uint8 | When `1`, `position` / `center` are 0-indexed `[y_loc, x_loc]`; when absent or `0`, legacy 1-indexed `[x, y]` |
| `Path{n}` | — | group | One group per imaging path in trial-table order |
| `Path{n}/dr` | 1 x nChars | uint16 | Motion-correction directory used while drawing these ROIs |
| `Path{n}/fn` | 1 x nChars | uint16 | Trial stem used when displaying ROI GUI |
| `Path{n}/n_rois` | 1 x 1 | uint32 | Number of saved ROI entries for this path |
| `Path{n}/roi_###/type` | 1 x nChars | uint16 | ROI geometry type: `polygon`, `circle`, or `ellipse` |
| `Path{n}/roi_###/label` | 1 x nChars | uint16 | User label (e.g., `SOMA`) |
| `Path{n}/roi_###/mask` | rows x cols | uint8 | Binary ROI mask in image coordinates (1 = included pixel) |
| `Path{n}/roi_###/position` | nVertices x 2 | double | Polygon vertices `[y_loc, x_loc]` when `coords_zero_indexed=1`, else legacy `[x, y]` |
| `Path{n}/roi_###/center` | 1 x 2 | double | Circle/ellipse center `[y_loc, x_loc]` when `coords_zero_indexed=1`, else legacy `[x, y]` |
| `Path{n}/roi_###/semi_axes` | 1 x 2 | double | Ellipse semi-axes lengths (ellipse only) |
| `Path{n}/roi_###/rotation_angle` | 1 x 1 | double | Ellipse rotation angle in degrees (ellipse only) |
| `Path{n}/roi_###/radius` | 1 x 1 | double | Circle radius (circle only) |

### `experiment_summary.h5`

**Indexing conventions.** Pixel/plane coordinates in `sources/spatial/coords` and related peak/coordinate fields are written **0-indexed** as `[z_loc, y_loc, x_loc]`, matching image axis order (`fastz`, rows, cols). `frame_line_idxs` is kept **1-indexed** to retain the SLAP2 line-indexing convention.

| Field | Size | Data type | Description |
| --- | --- | --- | --- |
| `row_major` | 1 x 1 | uint8 | Layout flag (see above) |
| `params` | — | group/struct | Analysis parameters. `params/activityChannel` is **1-indexed** into the recording's `numChannels` channels — use it to pick the glutamate channel from any `channels x …` dataset (e.g., `global/F`, `sources/temporal/dF_ls`) |
| `Path{n}` | — | group | One group per imaging path |
| `Path{n}/Z_depths` | fastz x 1 | numeric | Z depths per imaging plane (SLAP2 only) |
| `Path{n}/frame_info/offlineXshifts` | total frames x 1 | numeric | Offline registration X shift per frame |
| `Path{n}/frame_info/offlineYshifts` | total frames x 1 | numeric | Offline registration Y shift per frame |
| `Path{n}/frame_info/offlineZshifts` | total frames x 1 | numeric | (optional) Offline registration Z shift per frame; written only when 3D alignment was performed |
| `Path{n}/frame_info/onlineXshifts` | total frames x 1 | numeric | (SLAP2 only) online X shift per frame |
| `Path{n}/frame_info/onlineYshifts` | total frames x 1 | numeric | (SLAP2 only) online Y shift per frame |
| `Path{n}/frame_info/onlineZshifts` | total frames x 1 | numeric | (SLAP2 only) online Z shift per frame |
| `Path{n}/frame_info/trial_num_frames` | trials x 1 | integer | Number of frames contributed by each analysis trial |
| `Path{n}/frame_info/frame_line_idxs` | total frames x 1 | integer | Raw line (SLAP2) or frame (other microscopes) index for each frame in the stitched series. **1-indexed** |
| `Path{n}/frame_info/discard_frames` | total frames x 1 | bool or uint8 | Frame excluded from analysis (e.g., motion censoring) |
| `Path{n}/visualizations/mean_im` | channels x fastz x rows x cols | numeric | Mean registered image per channel / Z slice |
| `Path{n}/visualizations/act_im` | fastz x rows x cols | numeric | Activity / localization summary image |
| `Path{n}/visualizations/act_im_peaks` | sources x 3 | numeric | Activity image peak locations `[z_loc, y_loc, x_loc]`, **0-indexed**; from source row/column coordinates, `z_loc` fixed at `0` |
| `Path{n}/global/F` | channels x total frames | numeric | Fluorescence traces over the whole field (one row per channel) |
| `Path{n}/user_rois/labels` | rois x 1 | string | User-defined ROI labels |
| `Path{n}/user_rois/mask` | rois x fastz x rows x cols | uint8 or bool | Stacked binary masks for each user ROI |
| `Path{n}/user_rois/Fsvd` | rois x channels x total frames | numeric | ROI signals after SVD / projection step (if used) |
| `Path{n}/user_rois/F` | rois x channels x total frames | numeric | Raw or baseline-corrected ROI fluorescence |
| `Path{n}/sources/spatial/profiles` | sources x fastz x rows x cols | numeric | Spatial component / pixel weights per source, averaged across trials with footprints |
| `Path{n}/sources/spatial/coords` | sources x 3 | numeric | Source centers per row: `[z_loc, y_loc, x_loc]`, **0-indexed** |
| `Path{n}/sources/temporal/dF_ls` | sources x channels x total frames | numeric | Least-squares ΔF (absolute or scaled) |
| `Path{n}/sources/temporal/dF_denoised` | sources x channels x total frames | numeric | Denoised ΔF |
| `Path{n}/sources/temporal/events` | sources x channels x total frames | numeric | Deconvolved source events |
| `Path{n}/sources/temporal/F0` | sources x channels x total frames | numeric | Baseline estimate used for normalization |
| `Path{n}/sources/temporal/SNR` | sources x 1 | numeric | (optional) Signal-to-noise ratio metric; only written when extraction emits per-source SNR |

### `per_trial_summary.h5`

The trial axis matches `trial_table.h5` (all analysis trials); trials without alignment or source-extraction data are left as NaN in the corresponding slices. Coordinate fields use the same **0-indexed** `[z_loc, y_loc, x_loc]` convention as `experiment_summary.h5`; `z_loc` is currently always `0`.

| Field | Size | Data type | Description |
| --- | --- | --- | --- |
| `row_major` | 1 x 1 | uint8 | Layout flag (see above) |
| `Path{n}` | — | group | One group per imaging path |
| `Path{n}/visualizations/per_trial_mean_im` | trials x channels x fastz x rows x cols | numeric | Trial-aligned mean registered image per channel / Z slice |
| `Path{n}/visualizations/per_trial_act_im` | trials x fastz x rows x cols | numeric | Trial-aligned activity / localization summary image |
| `Path{n}/visualizations/per_trial_act_im_peaks` | trials x max_peaks x 3 | numeric | Per-trial detected peak locations `[z_loc, y_loc, x_loc]`, **0-indexed**, NaN-padded when a trial has fewer than `max_peaks` |
| `Path{n}/visualizations/per_trial_num_peaks` | trials x 1 | integer | Number of valid peaks per trial |
| `Path{n}/sources/spatial/per_trial_profiles` | trials x sources x fastz x rows x cols | numeric | Spatial component / pixel weights per source per trial |
| `Path{n}/sources/spatial/per_trial_coords` | trials x sources x 3 | numeric | Source centers per trial: `[z_loc, y_loc, x_loc]`, **0-indexed** |
| `Path{n}/sources/temporal/per_trial_SNR` | trials x sources | numeric | Per-source SNR for each analysis trial |
