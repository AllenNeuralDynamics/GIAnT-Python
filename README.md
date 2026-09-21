# GIAnT-python

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Python](https://img.shields.io/badge/python->=3.10-blue?logo=python)

An incremental Python port of [GIAnT-MATLAB](https://github.com/AllenNeuralDynamics/GIAnT-MATLAB), currently focused on **SLAP2 BandSILo extraction and annotation from a pre-registered trial table**. This is not a full MATLAB pipeline replacement.

## Implemented scope

| Capability | Status |
| --- | --- |
| BandSILo extraction and ROI annotation | Implemented for pre-registered SLAP2 band recordings |
| Trial-table loading | Implemented; accepts an exact filename or a loaded `TrialTable` |
| Generic HDF5 decoding and parameter-group writing | Implemented shared helpers; not a promise of lossless support for every reference field |
| `ExperimentSummary.from_h5()` / `.to_h5()` | Implemented for the emitted band-summary subset, with eager arrays |
| Raw acquisition and alignment reading | Implemented SLAP2 reader access, saved alignment decoding and native TIFF reading |
| Trial-table writing and `AlignmentData` model codecs | Not available; placeholder APIs removed |
| Organization, registration, standard SILo, deconvolution | Not implemented; no placeholder stages or functions are retained |
| Config-file construction, checkpoint/resume | Not implemented; no `from_config()` or `run_all()` API |

The pre-registration inputs must already include the raw recordings and reference TIFFs, alignment files, and the band registration lookup table. Python consumes these externally generated assets; it does not perform registration. See [architecture](docs/source/architecture.rst) and [API and data conventions](docs/source/migration.rst).

### Placeholder removal — 2026-09-20

The earlier requirement to retain future placeholders is superseded: all unimplemented placeholder functions, files and their exports have been removed. This includes the generic numerical stubs, future pipeline stages, `AlignParams`, `AlignmentData`, unsupported IO wrappers and model writers. Working band science, peak fitting, the band-specific NMF solver, native readers, HDF5 writing/codecs and the `Slap2Reader` typing protocol remain. The removal does not alter scientific algorithms. The MATLAB file schemas below remain useful external-format references, not promises of supported Python APIs.

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

Verbose progress uses the declared `tqdm` dependency, imported only when needed. The existing ROI editor uses OpenCV and tkinter (the latter must be available in the Python installation), not napari. Acquisition and GUI imports remain lazy; ordinary model access does not open a GUI.

## Supported usage

Supply the **actual** trial-table filename; it need not have a conventional basename. The following function can also receive an already loaded `TrialTable`:

```python
from giant_python import (
   AnnotationOptions, BandSiloParams, ExecutionOptions,
   extract_band_sources,
)


def extract_registered_table(table_or_filename):
   return extract_band_sources(
      table_or_filename,
      BandSiloParams(),
      execution=ExecutionOptions(max_workers=6, verbose=True),
      annotations=AnnotationOptions(enabled=False, interactive=False),
   )
```

The returned `ExperimentSummary` is also written beneath the loaded table's `savedr`, in the `source_extraction` subdirectory. Moving or copying the table alone does **not** change `savedr` or `datadr`. For staging and explicit in-memory directory overrides, use [examples/extract_band.py](examples/extract_band.py).

For a chainable session, use the same services through `Pipeline`:

```python
from giant_python import (
   AnnotationOptions, BandSiloParams, ExecutionOptions, Pipeline,
)


def extract_session(table_or_filename):
   pipeline = Pipeline.from_trial_table(
      table_or_filename,
      params=BandSiloParams(),
      execution=ExecutionOptions(max_workers=6),
      annotations=AnnotationOptions(enabled=False, interactive=False),
   ).extract()
   return pipeline.summary
```

`Pipeline` retains the exact source filename when known and accepts a loaded model without serializing/reloading it or guessing a source path from `savedr`. Use `summary.paths` and each `PathSummary.path_index` (zero-based); disk groups are `Path{path_index + 1}`. Arrays are eager, and per-source/ROI entries can share array storage rather than owning independent copies.

The public workflow methods are `from_trial_table()`, `annotate()` and `extract()`; `extract_sources(...)` is the standalone generic service, not a second `Pipeline` method. `from_config()`, `organize()`, `register()` and `run_all()` are absent, as are the constructor options/attributes `save_dir` and `align_params`. Use the loaded table's `datadr` and `savedr` for explicit directory overrides. Generic `save_struct_h5()` was removed; use the working summary codec for supported band outputs rather than treating the parameter-group writer as a general struct serializer.

To use existing ROIs without drawing, pass `AnnotationOptions(enabled=True, interactive=False)`. Missing required annotations fail rather than silently omitting ROIs. The explicit `annotate_rois(...)` service or `Pipeline.annotate()` loads existing annotations or draws them when the interactivity policy allows it; annotation files are stored in the table's results directory under `annotations`. There is no implemented config-file loader or resume mode.

The `giant` CLI exposes only `annotate` and `extract`, each taking an exact trial-table filename. Its only microscope choice is `slap2` and its only scan-mode choice is `band`. Unsupported CLI choices are argument errors; unsupported routing through standalone services raises `ValueError`, not `NotImplementedError`. Pipeline operations delegate to those same services. For headless extraction using existing ROIs, select `--draw-user-rois --headless`.

### Canonical API and ownership

The Python API preceding this cleanup was unreleased and unused. Compatibility-only Python imports and adapters have been removed, not deprecated. MATLAB file-format compatibility is retained separately; there is no production Python API migration requirement.

- `extract_band_sources(input, params=None, *, execution=None, annotations=None)` accepts an exact filename or loaded `TrialTable`. Use `Pipeline(..., params=...)` or `Pipeline.from_trial_table(..., params=...).extract()` for a session.
- `resolve_band_options(params=None, execution=None, annotations=None)` returns fresh validated `(BandSiloParams, ExecutionOptions, AnnotationOptions)` objects. Each argument accepts **only its corresponding dataclass, a dictionary of fields belonging to that scope, or `None`**. Mixed dictionaries, cross-scope fields and unknown keys are rejected; they are not redistributed or converted. Caller inputs are not mutated.
- `params` holds science only; channel count is inferred from acquisition metadata and source activity extraction uses channel zero. `execution` holds `max_workers`, `verbose`, and `max_trials`; `annotations` holds `enabled` and `interactive`. Generic services and `Pipeline` take `microscope` / `scan_mode` as explicit routing options; only SLAP2 band extraction is implemented (`scan_mode=None` also selects band). There is no registration configuration class.
- All per-path results are accessed through `ExperimentSummary.paths`. A `PathSummary` owns `sources`, `user_rois`, `visualizations`, `frame_info`, `global_f`, and `z_depths`. There are no flat or aggregate summary properties. Access frame fields as attributes, such as `path.frame_info.offlineXshifts`, or call the shallow `FrameInfo.to_dict()`; `FrameInfo` is not a mapping.
- `giant_python.io.read_summary(path)` returns an `ExperimentSummary`; `giant_python.io.write_summary(summary, path)` accepts that model and returns `None`. Their owner is `giant_python.io.experiment_summary`; the model's `from_h5()` / `to_h5()` delegate there. No public assembled-dictionary writer or alternate summary IO alias is supported.
- `giant_python.extraction.band.result_assembly.assemble_path_summary()` is the band result assembly boundary and returns a `PathSummary`. Scientific backend code belongs in `giant_python.extraction.band`; shared kernels belong in `giant_python.numerics`, not the removed compatibility namespaces.
- `giant_python.gui.parameters_gui.run_parameter_gui(params=None, *, execution=None, annotations=None)` returns `(BandSiloParams, ExecutionOptions, AnnotationOptions)` or `None` on cancellation. It retains the split scopes and does not modify caller-owned options.
- `giant_python.io` exports the working `load_struct_h5`, `read_trial_table`, `read_summary`, `write_summary`, `read_annotations_h5`, `save_annotations_h5`, `load_alignment_data_h5`, `open_slap2_file`, `read_tiff` and `Slap2Reader`. `write_dict_to_h5group` remains a working helper in `giant_python.io.hdf5`, not a universal model serializer. `Slap2Reader` describes the vendor reader contract for typing; it is not an unimplemented workflow stage.

The shared `gaussian_peaks_integrated()` API in `giant_python.numerics.peaks` accepts only `(N, 5)` parameters `[amp, mu_y, mu_x, sigma_y, sigma_x]`; equal widths must be supplied in both columns. Peak fitting includes an isotropic warm start followed by anisotropic refinement, using canonical five-column parameters with tied widths internally during the first stage. Removing the alternate parameter representation does not retune this fitting policy or the scientific defaults.

## Epoch and Analysis Trial

The following describes the reference GIAnT organization concepts. Python currently consumes these structures from an existing trial table rather than implementing the organization stage.

Epochs are full experimental sessions that can be aligned with each other (i.e. the same field of view and regions of interest are being imaged). Analysis trials are generally contiguous subsets (in time) of an epoch. These analysis trials may not align exactly with experimental trials.

For SLAP2, analysis trials are the experimental trials if the data was collected using the multi-trial functions of the SLAP2 and each trial is saved off the microscope in a different file. If data was continuously collected on SLAP2, the experiment will be split up into analysis trials of length 200000 lines (~20 sec) to help parallelize processing.

For data not collected on SLAP2, the reference GIAnT-MATLAB pipeline sets Epochs to be 1 and each file that is selected to be processed is an analysis trial. These analysis trials must be able to be aligned to one another. This is reference behavior, not a supported Python acquisition route.

## Pipeline Outputs

The extensive trees and field tables below are **reference schemas**, not a list of everything Python emits. The working BandSILo writer emits a subset of the experiment summary: source `profiles`, `coords`, `dF_ls`, `F0`, `SNR`; visualizations; `Z_depths`; global `F`; frame bookkeeping and shifts; and, when enabled, ROI `labels`, `mask`, and `F`. Disabled annotations omit the ROI group; enabled-but-empty annotations are distinct.

**Future/reference-only output:** `dF_denoised`, `events`, ROI `Fsvd`, and the entire per-trial summary are **not emitted**. Fields retained on model classes do not imply that the band codec reads/writes them. Arbitrary reference files are not guaranteed to round-trip losslessly through `ExperimentSummary`.

Trial-table writing is unavailable, and `TrialTable.to_h5` and the `AlignmentData` model have been removed. The shared HDF5 decoder and raw acquisition adapters nevertheless read the inputs needed by BandSILo. Existing summary writes replace supplied path groups but retain unrelated path groups and existing root parameters/layout flags; this is not an atomic replacement or resume mechanism. Prefer a fresh output location for a new analysis.

### Reading and writing H5

The implemented codecs use [`h5py`](https://www.h5py.org/). New Python band summary and annotation files set `row_major` to `1`: the logical dimensions below then match the dataset's h5py `shape`. Reference-only fields retain their intended logical dimensions for future ports; they are not all written by Python. Updating an existing summary preserves its existing layout flag, so use a fresh destination rather than mixing producers in place.

Files produced by GIAnT-**MATLAB** instead set `row_major` to `0` (or omit it), and their dimension tuples follow MATLAB `size()`. The shared decoder inspects this flag and **reverses multidimensional axes as needed** to recover logical orientation. Schema-specific support is still required; layout conversion does not imply support for every MATLAB file. When in doubt:

| `row_major` value | Layout | Axis order of the tuples below |
| --- | --- | --- |
| `1` | row-major (h5py / NumPy C order) | matches h5py `shape` as written here |
| `0` or absent | column-major (MATLAB `size()`) | reverse of the tuples here |

**Emitted vector shapes are preserved on disk.** Band summary frame vectors retain rank two, rather than being squeezed to `(nFrames,)`. For reference-only fields, the listed vector shapes describe the reference schema, not a Python writer guarantee. Layout handling alone does not establish full file interchangeability.

### Index and coordinate conventions

These conventions describe the supported subset and the reference fields below:

- **Line/frame indices** (`first_line`, `last_line`, `DSframes`, `frame_line_idxs`) are **1-indexed** to retain the SLAP2 line-indexing convention.
- **Spatial coordinate fields** in the summaries (`sources/spatial/coords`, `act_im_peaks`, `per_trial_*_coords`, etc.) are **0-indexed** `[z_loc, y_loc, x_loc]`, matching image axis order (`fastz`, rows, cols). `z_loc` is the 0-based index into the `fastz` axis.
- **Annotation coordinate fields** (`position`, `center`) are **0-indexed** `[y_loc, x_loc]` when `coords_zero_indexed = 1`; legacy files without the flag use MATLAB `images.roi` convention (1-indexed `[x, y]`).
- **Images** are stored `rows x cols` (row = Y, col = X). Motion shifts: `motionDSr` / `motionR` are **row** shifts, `motionDSc` / `motionC` are **column** shifts.
- `params/activityChannel` is **1-indexed** into the recording's `numChannels` channels.

The last field is a reference parameter: do not assume every reference parameter is emitted by the band writer. Python source activity extraction always uses channel zero (zero-based), and source activity arrays for other channels retain NaNs. Channel count is inferred from acquisition metadata, not supplied in `BandSiloParams`.

**Motion-sign convention:** public `FrameInfo.offlineXshifts`, `offlineYshifts`, and `offlineZshifts` contain alignment displacement with the standard MATLAB SILo sign. Private band reference offsets have the opposite sign and are converted once during `assemble_path_summary()`, not by generic IO. If an experimental file from an unreleased Python checkout exists, establish its producer/alignment provenance before interpreting or changing its shifts; there is no known production Python migration to perform. Never negate MATLAB-produced or already-correct summaries again. Online shifts are unchanged. Matching this sign does not change the existing linear/clamped interpolation to MATLAB's PCHIP/extrapolation. Removing Python compatibility code leaves the HDF5 field spelling `Z_depths`, layout flags, coordinate conversions, scientific defaults, motion-bin ordering, and numerical call order unchanged; no new caching or execution policy is introduced.

Legend for the trees below (🗄️ file · 📁 group · 🔤 string · 🔢 integer · 📈 numeric · 🖼️ image · ☑️ bool).

### Trial Table

Reference input schema: a trial table summarizes file locations and analysis trial structures. Python loads an existing pre-registered table; it does not currently create/save this model or promise to append stage groups. `slap2_info` is SLAP2-specific; the reference pipeline populates `motion_correction` and `source_extraction` downstream. (Model: `giant_python.models.TrialTable` / `Slap2Info`.)

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

Reference input schema: the external motion correction stage saves an alignment HDF5 file for each trial. Python reads the band fields through `giant_python.io.slap2.load_alignment_data_h5`; there is no `AlignmentData` model or Python registration stage. The full external schema is retained below even where fields are not consumed by the band workflow.

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
      |  |  ├ 📈 dF_denoised (sources x channels x total frames; REFERENCE ONLY, not emitted)
      |  |  ├ 📈 events (sources x channels x total frames; REFERENCE ONLY, not emitted)
    |  |  ├ 📈 F0 (sources x channels x total frames)
    |  |  └ 📈 SNR (sources x 1)
    |  └ 📁 spatial
    |     ├ 🖼️ profiles (sources x fastz x rows x cols)
    |     └ 📈 coords (sources x 3 [z_loc, y_loc, x_loc])
    ├ 📁 user_rois
    |  ├ 🔤 labels (rois x 1)
    |  ├ 🖼️ mask (rois x fastz x rows x cols)
      |  ├ 📈 Fsvd (rois x channels x total frames; REFERENCE ONLY, not emitted)
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

**Reference only — not emitted by Python BandSILo.** The reference per-trial summary holds fields that vary across analysis trials. Its trial axis matches the reference trial table (all analysis trials); trials without alignment or source-extraction data are represented as NaN slices. The entire tree below is retained as external-format documentation, not as a Python implementation commitment.

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

External MATLAB `BandRegistration` builds this reference input under `motion_correction/bandRegLookupTable.h5`. Python BandSILo consumes it; Python registration and lookup-table generation are not implemented. XY search limits (`xPre`, `yPre`, etc.) are shared across paths; per-path superpixel and reference-stack data live under `Path{n}`.

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

These tables retain the full reference fields for orientation and future ports. They do not override the implemented-subset limits above. In particular, registration outputs are inputs to Python, and all per-trial-summary fields are future-only.

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
| `Path{n}/allSuperPixelIDs` | nSP x 1 | numeric | Unique superpixel keys for this path: `superPixIdx * 100 + zIdx` (band-scan pixels only when `bandOnly` is true) |
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
| `Path{n}/user_rois/Fsvd` | rois x channels x total frames | numeric | **Future/reference only; not emitted.** ROI signals after SVD / projection step |
| `Path{n}/user_rois/F` | rois x channels x total frames | numeric | Raw or baseline-corrected ROI fluorescence |
| `Path{n}/sources/spatial/profiles` | sources x fastz x rows x cols | numeric | Spatial component / pixel weights per source, averaged across trials with footprints |
| `Path{n}/sources/spatial/coords` | sources x 3 | numeric | Source centers per row: `[z_loc, y_loc, x_loc]`, **0-indexed** |
| `Path{n}/sources/temporal/dF_ls` | sources x channels x total frames | numeric | Least-squares ΔF (absolute or scaled) |
| `Path{n}/sources/temporal/dF_denoised` | sources x channels x total frames | numeric | **Future/reference only; not emitted.** Denoised ΔF |
| `Path{n}/sources/temporal/events` | sources x channels x total frames | numeric | **Future/reference only; not emitted.** Deconvolved source events |
| `Path{n}/sources/temporal/F0` | sources x channels x total frames | numeric | Baseline estimate used for normalization |
| `Path{n}/sources/temporal/SNR` | sources x 1 | numeric | (optional) Signal-to-noise ratio metric; only written when extraction emits per-source SNR |

### `per_trial_summary.h5`

**All fields in this table are future/reference only; this file is not emitted.** The reference trial axis matches the trial table (all analysis trials); trials without alignment or source-extraction data are left as NaN in the corresponding slices. Coordinate fields use **0-indexed** `[z_loc, y_loc, x_loc]` coordinates.

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
