# GIAnT-Python

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Python](https://img.shields.io/badge/python->=3.10-blue?logo=python)

GIAnT-Python is a Python package for processing glutamate imaging data, with advanced capabilities for SLAP2 recordings.

Currently implemented: **SLAP2 raster registration** (a port of GIAnT-MATLAB's MultiRoiRegistration, on the CPU or a CUDA GPU) and **SLAP2 Band Scan processing**: **BandSILo source extraction and Band Scan ROI annotation** from pre-registered recordings. Additional functionality is available in [GIAnT-MATLAB](https://github.com/AllenNeuralDynamics/GIAnT-MATLAB), and efforts are underway to port it into GIAnT-Python.

## Requirements and scope

Provide an existing trial table, raw recordings with sidecars, reference TIFFs, alignment files, and the band registration lookup table. Python does not organize recordings. Raster recordings can be registered in Python (below); band registration and its lookup table still come from GIAnT-MATLAB. Standard SILo, standalone deconvolution, general trial-table writing, config-file loading, and checkpoint/resume are not supported.

Analysis trials come from the supplied table; Python does not split or merge them.

## Installation

From the repository root, with Python 3.10 or later:

```bash
pip install -e .
```

Installation requires Git/network access for the `slap2_utils` dependency. Interactive ROI drawing uses OpenCV and tkinter; tkinter must be available in the Python installation. Headless extraction does not need a display.

For development setup and documentation builds, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Register raster recordings

`register_multiroi` registers every trial and DMD path of a MATLAB-written trial table, as GIAnT-MATLAB's `MultiRoiRegistration` does, and writes the same files: `E<epoch>T<trial>DMD<path>_REGISTERED_DOWNSAMPLED-80Hz.h5` (or `.tif`) and `..._ALIGNMENTDATA.h5` under `<savedr>/motion_correction`. It then rewrites the table's `motion_correction` group in MATLAB's layout, so GIAnT-MATLAB SILo can use the results unchanged.

```python
from giant_python import ExecutionOptions, register_multiroi

register_multiroi(
    "path/to/trial_table.h5",
    {"align_hz": 80},  # MultiRoiRegistrationParams fields
    execution=ExecutionOptions(device="auto", max_workers=6, verbose=True),
)
```

or `giant register path/to/trial_table.h5 --device cuda`. On the CPU, trials run in `max_workers` parallel processes; on a GPU, one at a time. The command lists trials flagged `registrationFailed` and exits with status 1 if the registration of any trial raised (those trials get no outputs and empty filenames).

- By default frames are registered sequentially, in MATLAB's order, so results match GIAnT-MATLAB within single-precision rounding. `schedule="grow"` instead matches frames in batches, growing outward from seed frames spread over the trial (see `giant_python.registration.growth`); it is experimental and gives slightly different motion.
- `compensate_online=True` (`--compensate-online`) undoes the online motion correction line by line, as MATLAB's `compensateOnline` does. The alignment file then records each frame's reference offset (`slap2/onlineCompensation`).
- `varFacDS` is stored deflate-compressed (`var_factor_deflate=1`, about 16x smaller, identical values; MATLAB's `h5read` decompresses transparently). Set it to 0 for MATLAB's uncompressed layout. Registered movies are deflated HDF5 by default (`movie_deflate=1`, about 7x smaller than TIFF), which MATLAB SILo also reads. `save_tif=True` (`--tif`) writes MATLAB's default BigTIFF instead.
- Not yet ported: `refStackTemplate`, `isReVolt` and integration ROIs. Setting any of them raises `NotImplementedError`.
- Only single-plane recordings with one or two channels are supported.

## Run BandSILo

Use the actual trial-table filename, or pass a loaded `TrialTable`:

```python
from giant_python import ExecutionOptions, extract_band_sources

summary = extract_band_sources(
    "path/to/trial_table.h5",
    execution=ExecutionOptions(max_workers=6, verbose=True),
)
```

Annotations are disabled by default. Use `BandSiloParams` for scientific settings, `ExecutionOptions` for workers/progress/trial limits, and `AnnotationOptions` for ROI inclusion and GUI policy. A chainable `Pipeline` is also available; see [API and data conventions](docs/source/migration.rst).

**Paths matter:** raw inputs come from the table's `datadr`; results are written under its `savedr`, in `source_extraction/experiment_summary.h5`. Copying the table does not change either directory. For explicit directory overrides and a fresh output location, use [the staging example](examples/extract_band.py).

### Manual ROIs

- Draw or load annotations with `annotate_rois(table_or_filename)` or `Pipeline.annotate()`.
- Extract using existing ROIs with `annotations=AnnotationOptions(enabled=True, interactive=False)`. Missing required annotations raise an error.
- The CLI provides `giant annotate` and `giant extract`, each taking a trial-table filename. For extraction with existing ROIs and no GUI, select `--draw-user-rois --headless`.

Annotations are stored beneath the table's `savedr` in `annotations/annotations.h5`.

## Results and cautions

The returned `ExperimentSummary` is also saved to disk. Access imaging paths through `summary.paths`; each path contains sources, ROIs, visualizations, frame information, depths, and global fluorescence. Arrays are eager and may share storage. `path_index` is zero-based; disk groups start at `Path1`.

- Extraction saves least-squares `dF_ls`, convolution-constrained `dF_denoised`, and fitted `events`. Source activity uses channel zero; other channels and missing observations are NaN. Events are not automatically neuronal spikes, and smoother traces do not establish improved SNR.
- ROI `Fsvd` is written as NaN unless supplied explicitly; BandSILo does not compute it. Disabled or empty annotations produce zero-ROI datasets.
- Acquisition line indices are one-based; spatial coordinates are zero-based `[z, y, x]`. Saved offline shifts follow the MATLAB SILo alignment-displacement sign.
- **Use a fresh output location for each analysis.** Updating an existing summary preserves unrelated paths and existing root parameters/layout flags; writes are not atomic replacement or resume.

## Reference

Documentation provenance.[^documentation]

- [API, configuration, and staging](docs/source/migration.rst)
- [Trace estimation, controls, and scientific caveats](docs/source/traces.rst)
- [File formats, coordinates, and MATLAB interoperability](docs/source/file_formats.rst)
- [Architecture and module ownership](docs/source/architecture.rst)

## Support

**Occasional updates:** no fixed update schedule. Community involvement through issues and pull requests is welcome.

[^documentation]: Current documentation has been generated by large language models (LLMs) from human-verified code.