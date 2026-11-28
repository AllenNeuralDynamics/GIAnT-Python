# GIAnT-Python

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Python](https://img.shields.io/badge/python->=3.10-blue?logo=python)

GIAnT-Python is a Python package for processing glutamate imaging data, with advanced capabilities for SLAP2 recordings.

Currently implemented: **SLAP2 raster registration** (a port of GIAnT-MATLAB's MultiRoiRegistration, on the CPU or a CUDA GPU), **SILo source extraction for SLAP2 raster (multi-ROI) recordings** (a port of GIAnT-MATLAB's SILo, on one machine or as stages to scatter), and **SLAP2 Band Scan processing**: **BandSILo source extraction and Band Scan ROI annotation** from pre-registered recordings. Additional functionality is available in [GIAnT-MATLAB](https://github.com/AllenNeuralDynamics/GIAnT-MATLAB), and efforts are underway to port it into GIAnT-Python.

New in this release: SILo source extraction for SLAP2 raster recordings, with an optional baseline from the F0 network. For what changed relative to GIAnT-MATLAB, how the results compare and what is still open, see the [release notes](#release-notes-slap2-source-extraction-in-python-november-2026).

## Requirements and scope

Provide a trial table (SLAP2 trial tables can be built in Python, below), raw recordings with sidecars, reference TIFFs, alignment files, and the band registration lookup table. Python does not organize recordings. Raster recordings can be registered and source-extracted in Python (below); band registration and its lookup table still come from GIAnT-MATLAB. SILo for recordings other than SLAP2 raster, WMF seeding, standalone deconvolution, general trial-table writing and config-file loading are not supported.

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

With the F0 network's baselines (slap2-F0-network) the run localizes and extracts with them instead of the moving median and the fitted baseline: `giant silo TABLE --baseline-dir DIR` (baselines made beforehand), or `--f0-ckpt PATH` to predict the missing ones with `slap2-f0 predict-registered` (on the GPU; GIAnT-Python calls its command line and does not import it). This sets `baseline_source='network'`: no surround and the bootstrap null in localization, the network's baseline times one gain per pixel in the solver (`baseline_mode='gain'`), and removal of sources with fewer than 2 events per minute in the kinetics test (`--min-source-event-rate`); the seed threshold is 3 as with the moving median. Each can still be set, and `--peak-th 8 --min-source-event-rate 0` gives the defaults before 2026-11-28. The summary records the baseline files and the checkpoint. See [SILo for SLAP2 raster recordings](docs/source/silo.rst).

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

## Release notes: SLAP2 source extraction in Python (November 2026)

GIAnT-Python now processes SLAP2 raster (multi-ROI) recordings from raw data to activity traces without MATLAB. It builds the trial table, corrects motion, finds the sources and extracts the activity of each one. A *source* is a small spot that brightens when glutamate is released there, in most recordings a single synapse. This section describes what changed relative to GIAnT-MATLAB, how the results compare, how long processing takes, and what we know to be imperfect. The commands are in the sections above and in [SILo for SLAP2 raster recordings](docs/source/silo.rst).

### What the pipeline does

1. **Registration** corrects brain motion within each trial and writes one movie per trial and imaging path.
2. **Baseline** (optional) predicts, for every pixel and frame, the fluorescence expected if no glutamate were being released.
3. **Source finding** builds an image of where activity occurred in each trial, aligns the trials to each other, and places a candidate source on each spot of activity.
4. **Extraction** fits each source's shape (its *footprint*) and its activity over time, separating the activity of neighbours. A test of each trace's time course then rejects candidates whose transients decay faster than the indicator can; candidates with too few transients to judge are kept.
5. **Inspection**: `giant view` shows the sources on the activity images and, for each trial, the movie, the fitted activity and what the fit left unexplained.

The output files have GIAnT-MATLAB's layout, so code that reads `experiment_summary.h5` works unchanged. Without the baseline of step 2, GIAnT-Python follows GIAnT-MATLAB's method. It differs in seeding, footprints and alignment, which can each be set to GIAnT-MATLAB's behaviour, and in running its fits until they have settled where GIAnT-MATLAB stops after a fixed number of steps. Matched settings therefore give closely agreeing results, not identical ones.

### What changed relative to GIAnT-MATLAB

**A learned baseline.** Every value this pipeline reports is a change relative to a baseline (F0): the fluorescence a pixel would show without glutamate release. The baseline drifts with bleaching, with residual motion and with slow changes in the tissue. GIAnT-MATLAB estimates it twice: with a running median of each pixel when it looks for sources, and with a slowly varying curve fitted together with the activity when it extracts traces. Both estimates are influenced by the activity they are meant to exclude. Where a synapse is active often, part of its signal is taken into the baseline, and a fitted baseline can also be lowered so that noise is explained as small events. On a control recording of GFP, which shows no glutamate signals, the fitted baseline came out about 13% too low where we had added artificial events.

GIAnT-Python can instead take the baseline from a neural network, the [F0 network](https://github.com/AllenNeuralDynamics/slap2-F0-network), which reads the registered movie and predicts the baseline of every pixel and frame. On the same control recording its baseline was 2 to 3% too low. The network is used when baselines are supplied (`giant silo TABLE --baseline-dir DIR`) or a model file is given (`--f0-ckpt PATH`).

**A simpler fit.** With the network's baseline, extraction no longer estimates a time-varying baseline of its own. The fit keeps the time course of the network's baseline and adjusts a single scale factor per pixel, which stays close to 1 in practice. Fewer quantities are estimated from the same data, and the baseline can no longer change its time course to absorb activity; its level at each pixel is still fitted together with the sources. Extraction also runs faster.

**Source finding against measured chance.** With the network's baseline, the image of activity is computed relative to that baseline directly; GIAnT-MATLAB subtracts each pixel's surroundings at this step to remove slow background changes, which also removes part of the signal. Each trial's image is then calibrated against simulated movies that have the trial's baseline and noise but no activity, so that its values express how far a spot exceeds what noise alone produces. The trials' images are combined, and candidates are accepted above a fixed threshold on the combined image.

**Seeding.** Each spot in the activity image is fitted with a small bell-shaped profile to place a candidate source. GIAnT-MATLAB fits all spots of an imaging path together and stops after a fixed number of steps, which takes minutes per path and often seeds a source lying between pixels two or three times. GIAnT-Python fits each spot on its own until the fit has settled, which takes seconds. Further candidates are then placed where this first fit left activity unexplained, and such a candidate is discarded if it ends within 1.5 pixels of one already placed. In tests with artificial sources at known positions this removed nearly all duplicates, while pairs 2 pixels or more apart were still separated.

**Wider footprints and local alignment.** A footprint may now extend over 5 x 5 pixels instead of 3 x 3; on real recordings a third to two thirds of the footprints were wider than the smaller window could represent. Trials are aligned to each other with a smooth map of local displacements instead of one shift for the whole field, which matters when the tissue deforms between trials.

**Running at scale.** The many small fitting problems of a recording can be solved on separate machines or one after another on a single machine. The results are identical as long as the numerical settings (threads per process, precision, CPU or GPU) are the same. In the cloud, registration, the network's prediction and extraction run as separate steps, so a later step can be repeated without redoing the earlier ones.

### How the results compare with GIAnT-MATLAB

With settings matched to GIAnT-MATLAB's the two implementations agree closely: on sparse recordings the traces of matched sources correlate at 1.00 and amplitudes agree within 1%. The defaults differ, and the number of sources differs with them. The table counts the sources kept in the first three minutes of three recordings, both imaging paths together.

| recording | GIAnT-MATLAB | GIAnT-Python, fitted baseline | GIAnT-Python, network baseline |
|---|---|---|---|
| sparse (878030) | 73 | 73 | 43 |
| intermediate (885233) | 143 | 135 | 72 |
| dense (780125) | 440 | 355 | 269 |

- **Fitted baseline.** GIAnT-Python finds as many sources as GIAnT-MATLAB on the sparse recording and about 20% fewer on the dense one. About three quarters of the sources missing there lie within 2 pixels of a source that GIAnT-Python kept. Without the 1.5 pixel rule a single source is often seeded twice, so some of these pairs were duplicates; others are probably real neighbours, and we could not determine the proportions.
- **Network baseline.** GIAnT-Python keeps 40 to 50% fewer sources than GIAnT-MATLAB. More than eight in ten of them are also found with the fitted baseline; the difference is candidates that drop out under the stricter threshold applied with the network baseline. On the dense recording most of these are candidates that the time-course test rejects in any case: with the fitted baseline it rejects a quarter to more than half of the candidates there, with the network baseline almost none. On the sparser recordings the sources that drop out had passed the test, and we cannot tell from real recordings how many of them are real.
- **Control recording.** On GFP, where every source found is false, the fitted baseline gave three false sources per path and the network baseline none. Of artificial sources added to that recording, 99% were found with the fitted baseline and 97% with the network baseline.

Values read from the outputs also shift. With the network baseline, the baseline of matched sources is 10 to 20% higher and their events are 10 to 15% smaller than with the fitted baseline, so dF/F is about two thirds of the earlier value. The wider footprints lower dF/F by a further 5 to 15%, because the baseline is now taken over a larger area around each source while the events at its centre are unchanged. Comparisons of dF/F across pipeline versions therefore need the same settings on both sides.

### Processing time

Times are minutes of processing per minute of recording with two imaging paths, measured on three-minute excerpts.

| step | machine | minutes per recorded minute |
|---|---|---|
| registration | one cloud GPU (48 GB) | 2 to 7 |
| network baseline | the same GPU | 7 to 10 |
| source finding and extraction | 32 processor cores | 4 to 5 |
| all steps | | 15 to 20 |

Source finding and extraction take about as long as in GIAnT-MATLAB on sparse recordings (5 minutes per recorded minute there) and about a fifth as long on the dense one (25 minutes there). On a workstation the dense recording took 7 minutes per recorded minute against 17 with GIAnT-MATLAB. The network's prediction is now the slowest step, and it takes several times longer on a desktop GPU with 8 GB of memory, so baselines are better computed in the cloud.

### Known limitations and biases

- **The baseline is slightly low**, by 2 to 3% on the control recording. The network was trained on targets derived from GIAnT-MATLAB's fits, which carry that method's errors, and the way those targets were made left 5 to 7% of the peak of each fast event in the baseline the network was asked to predict.
- **Small events are overestimated.** Events well above the noise are found in more than nine cases in ten and sized within a few percent. Events near the noise level are found about half the time, and those that are found appear about a quarter too large, because noise helps them over the threshold.
- **Event sizes with the two baselines differ.** On real recordings events come out 10 to 15% smaller with the network baseline than with the fitted one, more than the 3 to 6% seen for artificial events. The cause is not established, and we do not yet know which is closer to the truth for real events.
- **Two effects of the settings**, described above: dF/F depends on the footprint size, and a weaker source within about 1.5 pixels of a stronger one is usually not seeded separately.
- **Source positions vary more between trials** with the network baseline (about 0.2 pixels against 0.12 to 0.19), for reasons we have not identified.
- **Trial edges.** Baselines are least certain in the first and last two seconds of a trial, where less of the surrounding recording is available.
- **Scope.** Recordings other than SLAP2 raster and Band Scan, registration of Band Scan recordings, drawing of regions of interest on raster recordings, and a few raster registration options (`refStackTemplate`, `isReVolt`, integration ROIs) still require GIAnT-MATLAB.

### Change of 28 November 2026: more sources with the network baseline

With the network's baseline, sources are now seeded at threshold 3, as with the fitted baseline, instead of 8. A source seeded less than 2 units above the threshold is kept only if the time-course test finds at least 2 events per minute in its trace; stronger seeds are kept as before (GIAnT-Python#51). Of artificial sources added to the GFP control recording, those at about a quarter of the standard event size are now found in all cases instead of 62%, and dim sources of that size 2 to 4 pixels from a brighter one in a third of cases instead of none; the control recording gives 3.5 false sources per imaging path instead of none. On the three recordings above the network baseline now keeps 128, 180 and 564 sources instead of 43, 72 and 269, and source finding and extraction take 1.6 to 3 times as much processor time. `--peak-th 8 --min-source-event-rate 0` gives the earlier behaviour.

### Open directions

1. **Retrain the F0 network on cleaner data.** We will extract the training recordings again with the network's own baseline and a corrected rendering of events, retrain, and re-examine the biases listed above. Registration already saves the per-pixel statistics that the next version of the network will take as input.
2. **Spine size.** The baseline fluorescence at a synapse scales with the amount of indicator, and so with the amount of membrane, in its footprint. We are building a readout of spine size, and of its change over a session, from the network's baseline.
3. **Event sizes and dF/F.** Measure the accuracy of event sizes in dense activity against ground truth, and settle on a definition of dF/F that does not depend on the footprint size.
4. **Dim sources.** Establish, on recordings with ground truth, how many real sources the stricter detection loses.
5. **Speed and memory.** Shorten the network's prediction and the loading of trials, which now set the pace, and reduce the memory needed per worker.
6. **Coverage.** Port the remaining recording types and registration options, and provide a single cloud pipeline that runs all steps for a session.

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