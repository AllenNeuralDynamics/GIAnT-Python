# GIAnT-Python

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Python](https://img.shields.io/badge/python->=3.10-blue?logo=python)

GIAnT-Python is a Python package for processing glutamate imaging data, with advanced capabilities for SLAP2 recordings.

Currently implemented: **SLAP2 raster registration** (a port of GIAnT-MATLAB's MultiRoiRegistration, on the CPU or a CUDA GPU), **SILo source extraction for SLAP2 raster (multi-ROI) recordings** (a port of GIAnT-MATLAB's SILo, on one machine or as stages to scatter), and **SLAP2 Band Scan processing**: **BandSILo source extraction and Band Scan ROI annotation** from pre-registered recordings. Additional functionality is available in [GIAnT-MATLAB](https://github.com/AllenNeuralDynamics/GIAnT-MATLAB), and efforts are underway to port it into GIAnT-Python.

## Requirements and scope

Provide a trial table (SLAP2 trial tables can be built in Python, below), raw recordings with sidecars, reference TIFFs, alignment files, and the band registration lookup table. Python does not organize recordings. Raster recordings can be registered and source-extracted in Python (below); band registration and its lookup table still come from GIAnT-MATLAB. SILo for recordings other than SLAP2 raster, its affine and patchwise across-trial alignment and WMF seeding, standalone deconvolution, general trial-table writing and config-file loading are not supported.

Analysis trials come from the supplied table; Python does not split or merge them.

## Installation

From the repository root, with Python 3.10 or later:

```bash
pip install -e .
```

Installation requires Git/network access for the `slap2_utils` dependency. Interactive ROI drawing uses OpenCV and tkinter; tkinter must be available in the Python installation. Headless extraction does not need a display.

For development setup and documentation builds, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Build SLAP2 trial tables

`build_trial_table_slap2` writes a SLAP2 session's `trial_table.h5` as GIAnT-MATLAB's `buildTrialTableSLAP2(dr, savedr, true)` does: every `.dat` file of the folder in one epoch, triggered recordings lined up across the two paths (including the Feb 2024 trial-numbering bug), continuous (`-CYCLE-`) recordings cut into trials of at most 200000 lines, reference stacks found below the folder, and trial times from the FPGA line clock in the `.meta` file (else from file times). MATLAB and Python registration and SILo read it unchanged.

```python
from giant_python.pipeline import build_trial_table_slap2

build_trial_table_slap2("path/to/session", save_dir="path/to/output", max_duration_s=180)
```

or `giant trial-table path/to/session --save-dir path/to/output --max-duration-s 180`. `max_duration_s` keeps only the first seconds of the session, as the integration-test capsule's `trimTrialTableSLAP2` does; the whole table is then also written as `trial_table_untrimmed.h5`.

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

or `giant register path/to/trial_table.h5 --device cuda`. On the CPU, trials run in `max_workers` parallel processes; on a GPU, one at a time. The command lists trials flagged `registrationFailed` and exits with status 1 if the registration of any trial raised (those trials get no outputs and empty filenames). Cells of a ragged trial table without a recording are skipped: their filenames stay empty and they are flagged `registration_failed`.

- By default frames are registered sequentially, in MATLAB's order, so results match GIAnT-MATLAB within single-precision rounding. `schedule="grow"` instead matches frames in batches, growing outward from seed frames spread over the trial (see `giant_python.registration.growth`); it is experimental and gives slightly different motion.
- `compensate_online=True` (`--compensate-online`) undoes the online motion correction line by line, as MATLAB's `compensateOnline` does. The alignment file then records each frame's reference offset (`slap2/onlineCompensation`).
- `varFacDS` is stored deflate-compressed (`var_factor_deflate=1`, about 16x smaller, identical values; MATLAB's `h5read` decompresses transparently). Set it to 0 for MATLAB's uncompressed layout. Registered movies are deflated HDF5 by default (`movie_deflate=1`, about 7x smaller than TIFF), which MATLAB SILo also reads. `save_tif=True` (`--tif`) writes MATLAB's default BigTIFF instead.
- Each trial also gets `<stem>_REGSTATS-80Hz.h5`: per-pixel summary statistics for the F0 network (slow and fast variance, fast skewness, shared modes, share of zeros), each with its photon-noise expectation from the trial's raw samples (`--no-stats` skips them). `giant reg-stats path/to/trial_table.h5` computes them for trials registered earlier, GIAnT-MATLAB's included. See [Registration summary statistics](docs/source/reg_stats.rst).
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

## Run SILo on raster recordings

`giant silo` extracts sources from registered SLAP2 multi-ROI (raster) recordings as GIAnT-MATLAB's `SILo.m` does, and writes the same files in MATLAB's schema: `source_extraction/experiment_summary.h5`, `source_extraction/per_trial_summary.h5` and the trial table's `source_extraction` fields. MATLAB's algorithm is the default (moving-median localization, fitted baseline, the kinetics test), except that the trials are aligned with patchwise displacement fields instead of MATLAB's default rigid shifts (`--alignment rigid`), the fits run to convergence instead of stopping at MATLAB's iteration caps, by default the footprint fit holds its variance at its start (the solver's default, `footprint_variance='fixed'`; `--set footprint_variance=model` gives MATLAB's loss), by default the seeding is faster and does not seed one source twice (below), and footprints may spread over 5 x 5 pixels around their source instead of MATLAB's 3 x 3 (`--valid-kernel 3` is MATLAB's).

```bash
giant register path/to/trial_table.h5
giant silo path/to/trial_table.h5 --verbose
```

or `run_silo(table, SiloParams(), SiloExecution(...))` from `giant_python.pipeline.silo` (in a script, under `if __name__ == "__main__":`: the worker processes are spawned and import it). Per path, trials are localized in parallel (`--workers`), the path is planned, and each trial is loaded at the selected pixels while the previous trial's subproblems are solved (`--solver-workers` processes of `--solver-threads` threads). `SiloParams` (in `giant_python.models.params`) is the one parameter set of all stages, with MATLAB's SLAP2 defaults; set fields with `--set name=value` or `--params file.json`. Existing annotations (`annotations/annotations.h5`: soma masks and user ROIs) are read; ROIs are not drawn.

The stages also run on their own, for any subset of their units, with file contracts between them, so that a pipeline can scatter them over machines: `giant silo-localize`, `silo-plan`, `silo-prepare` (one container per trial-path), `silo-dispatch --machines N` (job files of about equal work), `silo-solve JOB` (needs only the job file and its containers) and `silo-collect`. Every output records a settings hash; units whose outputs exist with the same settings are skipped. Results do not depend on how the work is split. See [SILo for SLAP2 raster recordings](docs/source/silo.rst).

With the F0 network's baselines (slap2-F0-network) the run localizes and extracts with them instead of the moving median and the fitted baseline: `giant silo TABLE --baseline-dir DIR` (baselines made beforehand), or `--f0-ckpt PATH` to predict the missing ones with `slap2-f0 predict-registered` (on the GPU; GIAnT-Python calls its command line and does not import it). This sets `baseline_source='network'`: no surround, the bootstrap null and threshold 8 in localization, and the network's baseline times one gain per pixel in the solver (`baseline_mode='gain'`); each can still be set. The summary records the baseline files and the checkpoint. See [SILo for SLAP2 raster recordings](docs/source/silo.rst).

To inspect runs by eye, `giant view RUN [RUN ...]` serves a local page (http://127.0.0.1:8766): the activity images with the runs' sources overlaid (two runs compared source by source), and, trial by trial, the aligned 80 Hz movie, the fitted activity rendered into it and the residual movie - activity (with `--baseline-dir`, also the network's baseline). See "Viewing a run" in [SILo for SLAP2 raster recordings](docs/source/silo.rst).

Two seeding settings are not MATLAB's and are on by default: `seed_fit='local'` runs the Gaussian seed fits locally and to convergence, which takes seconds per path instead of minutes; `seed_merge_px=1.5` rejects a residual peak that ends within 1.5 pixels of another one, so a source that falls between pixels is not seeded two or three times. `--set seed_fit=matlab --set seed_merge_px=0` gives MATLAB's seeding.

## Raster SILo localization (multi-ROI)

The per-trial localization of SILo (`loadAndProcessTrialAsync`, `localizeSources_vIM` and `getActImPeaks`) can also be used on its own. It reads GIAnT-MATLAB's registered 80 Hz movies and returns each trial's activity image, the summary image that is seeded, and the seeds:

```python
from giant_python.extraction.raster.localization import RasterLocalizationParams
from giant_python.extraction.raster.trial import localize_trial

loc, trial = localize_trial(
    "path/to/motion_correction",
    "E1T1DMD1_REGISTERED_DOWNSAMPLED-80Hz.tif",
    num_channels=2,
    baseline_dir="path/to/baselines",  # optional: slap2-f0 predict-registered
)
loc.peaks  # [amplitude, row, col, sigma], 0-based
```

With the defaults the activity image agrees with MATLAB to float32 rounding, and seeds agree at the default threshold. Three detection options, all off by default, are meant for an unbiased baseline such as the F0 network's (`baseline_dir`):
- `surround=False`: no difference-of-Gaussians surround in the spatial filter.
- `null="bootstrap"`: each trial's activity image is calibrated per pixel brightness against noise movies with the trial's baseline, gain, noise correlations and skew. `variance_floor=False` drops the `Vb` floor of the Z-score with it.
- `score="soft_threshold"`: at each local maximum, `max(0, u - soft_th)^2` replaces the squared value.

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
- [SILo for SLAP2 raster recordings: commands, stages and file contracts](docs/source/silo.rst)

## Support

**Occasional updates:** no fixed update schedule. Community involvement through issues and pull requests is welcome.

[^documentation]: Current documentation has been generated by large language models (LLMs) from human-verified code.