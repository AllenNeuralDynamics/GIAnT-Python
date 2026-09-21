"""Public band orchestration without vendor data or an expensive NMF fit.

The real driver, per-path composition, low-res assembly, motion geometry,
high-res trace solver, result assembly and HDF5 writer run together. Only
external input reads, background/noise estimation and localization are mocked.
"""

import unittest
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from giant_python.extraction.band import traces
from giant_python.extraction.band import workflow as wf
from giant_python.extraction.band.types import TrialTraceResult
from giant_python.io.experiment_summary import read_summary
from giant_python.models.params import (
    AnnotationOptions,
    BandSiloParams,
    ExecutionOptions,
)
from giant_python.models.trial_table import Slap2Info, TrialTable
from giant_python.pipeline.extract import extract_band_sources


class TestBandWorkflow(unittest.TestCase):
    """Exercise the public API through the driver to saved results."""

    def _run_mock_session(self, *, loaded=False, rois=False):
        """Check schema, policy and motion values in a two-path session."""
        n_frames, side, n_sp = 120, 15, 49
        positions = [(r, c) for r in range(4, 11) for c in range(4, 11)]
        smi = np.array(
            [
                [r * side + c, index + 1]
                for index, (r, c) in enumerate(positions)
            ],
            dtype=np.int32,
        )
        frames = np.arange(n_frames, dtype=float) + 11
        psf = np.ones((3, 3), dtype=np.float32)
        options = ExecutionOptions(max_workers=3, max_trials=2)
        annotation = AnnotationOptions(
            enabled=rois, interactive=False, operator="workflow test"
        )
        science_args = dict(
            analyze_hz=10, denoise_window_s=0.3, baseline_window_s=0.5
        )
        params = BandSiloParams(**science_args)
        before = (asdict(params), asdict(options), asdict(annotation))

        with TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            # Deliberately not savedr/trial_table.h5, and not even on disk.
            source = root / "selection" / "only-selected-trials.h5"
            table = TrialTable(
                source_path=source,
                savedr=root / "results",
                datadr=root / "raw",
                filename=np.array(
                    [
                        [
                            f"p{d}-skip.dat",
                            f"p{d}-selected.dat",
                            f"p{d}-beyond-limit.dat",
                        ]
                        for d in range(2)
                    ],
                    dtype=object,
                ),
                slap2_info=Slap2Info(
                    first_line=np.full((2, 3), 11),
                    last_line=np.full((2, 3), 130),
                    ref_stack={"selection": "unchanged"},
                ),
                motion_correction={
                    "fn_adata": np.array(
                        [
                            [
                                f"p{d}-skip.h5",
                                f"p{d}-selected.h5",
                                f"p{d}-beyond-limit.h5",
                            ]
                            for d in range(2)
                        ],
                        dtype=object,
                    )
                },
            )
            filenames_before = table.filename.copy()
            align_before = table.motion_correction["fn_adata"].copy()
            lookup = {
                "fastZ2RefZ": {
                    f"DMD{d + 1}": np.array([[10 + d]]) for d in range(2)
                },
                "allSuperPixelIDs": {
                    f"DMD{d + 1}": np.arange(n_sp).reshape(-1, 1)
                    for d in range(2)
                },
                "sparseMaskInds": {f"DMD{d + 1}": smi for d in range(2)},
            }
            read_table = stack.enter_context(
                patch.object(TrialTable, "from_h5", return_value=table)
            )
            prepare = stack.enter_context(
                patch.object(wf, "load_trial_table", wraps=wf.load_trial_table)
            )
            stack.enter_context(
                patch.object(
                    wf.inputs, "load_lookup_table", return_value=lookup
                )
            )
            stack.enter_context(
                patch.object(
                    wf.inputs,
                    "load_psf",
                    return_value={"DMD1": psf, "DMD2": psf},
                )
            )
            stack.enter_context(
                patch.object(
                    wf.inputs,
                    "load_reference_stack",
                    return_value=(np.zeros((1, 1, side, side)), None, None),
                )
            )
            stack.enter_context(
                patch.object(
                    wf.geo, "build_subsample_matrix_inds", return_value=smi
                )
            )
            keep = stack.enter_context(
                patch.object(
                    wf,
                    "compute_keep_trials",
                    return_value=np.array(
                        [[False, True, True], [False, True, True]]
                    ),
                )
            )
            stack.enter_context(
                patch.object(
                    wf,
                    "read_align_info",
                    return_value=({"DMD1": 10.0, "DMD2": 10.0}, 2),
                )
            )
            geometry = stack.enter_context(
                patch.object(
                    wf, "build_user_roi_geometry", return_value=({}, {})
                )
            )
            roi_selection = {
                "user_roi_masks": {
                    f"DMD{d + 1}": [np.ones((1, side, side), dtype=bool)]
                    for d in range(2)
                },
                "user_roi_labels": {
                    f"DMD{d + 1}": [f"ROI {d}"] for d in range(2)
                },
                "user_roi_superpixels": {
                    f"DMD{d + 1}": [np.array([0, 1])] for d in range(2)
                },
            }
            resolve_rois = stack.enter_context(
                patch.object(
                    wf, "resolve_user_rois", return_value=roi_selection
                )
            )

            def read_trial(
                trial_ix,
                keep_trial,
                dmd_ix,
                samp_freq,
                sp_ids,
                datadr,
                runtime_table,
                num_channels,
                all_channels=True,
            ):
                """Supply kept data and verify resolved read inputs."""
                self.assertIs(runtime_table["trial_table"], table)
                self.assertEqual(runtime_table["source_path"], source)
                self.assertEqual(num_channels, 2)
                self.assertEqual(samp_freq, 10)
                self.assertEqual(datadr, str(table.datadr))
                self.assertIn(trial_ix, (0, 1))
                if not keep_trial:
                    return None
                self.assertEqual(trial_ix, 1)
                self.assertEqual(
                    runtime_table["filename"][dmd_ix, trial_ix],
                    f"p{dmd_ix}-selected.dat",
                )
                # Internal reference offsets, not saved displacement.
                alignment = {
                    "DSframes": frames,
                    "motionDSr": np.full(n_frames, 2.0),
                    "motionDSc": np.full(n_frames, -1.0),
                    "motionDSz": np.zeros(n_frames),
                    "onlineYshift": np.full(n_frames, 4.0),
                    "onlineXshift": np.full(n_frames, 5.0),
                    "onlineZshift": np.full(n_frames, 6.0),
                }
                data = np.full(
                    (n_sp, n_frames), 20 + 10 * dmd_ix, dtype=np.float32
                )
                return {
                    "data": data,
                    "data_count": np.full_like(data, 2),
                    "data2": data * 2,
                    "data_count2": np.full_like(data, 2),
                    "alignment": alignment,
                    "ds_frames": frames,
                }

            low_read = stack.enter_context(
                patch.object(
                    wf.td, "read_band_trial_data", side_effect=read_trial
                )
            )
            high_read = stack.enter_context(
                patch.object(
                    traces, "read_band_trial_data", side_effect=read_trial
                )
            )
            mapper = stack.enter_context(
                patch.object(
                    wf,
                    "map_trials",
                    side_effect=lambda fn, items, max_workers, **kw: [
                        fn(i) for i in items
                    ],
                )
            )
            stack.enter_context(
                patch.object(
                    wf,
                    "_estimate_background",
                    side_effect=lambda data, *args, **kw: data * 0.75,
                )
            )
            stack.enter_context(
                patch.object(
                    wf.bg,
                    "fit_noise_variance_model",
                    side_effect=lambda data, *args: (np.ones_like(data), 1, 0),
                )
            )
            stack.enter_context(
                patch.object(
                    wf,
                    "_localize",
                    return_value=(
                        np.ones((1, side, side), dtype=np.float32),
                        np.array([[0, 7, 7]]),
                        torch.ones((side * side, 1)),
                        torch.tensor(
                            [[0, 7, 7, 1, 1, 1]], dtype=torch.float32
                        ),
                        np.array([2], dtype=np.float32),
                    ),
                )
            )
            assemble = stack.enter_context(
                patch.object(
                    wf, "assemble_path_summary", wraps=wf.assemble_path_summary
                )
            )
            writer = stack.enter_context(
                patch.object(
                    wf,
                    "write_summary",
                    wraps=wf.write_summary,
                )
            )

            input_value = table if loaded else source
            summary = extract_band_sources(
                input=input_value,
                params=params,
                execution=options,
                annotations=annotation,
            )
            self.assertIs(prepare.call_args.args[0], input_value)
            prepare.assert_called_once()
            if loaded:
                read_table.assert_not_called()
            else:
                read_table.assert_called_once_with(source)
            np.testing.assert_array_equal(table.filename, filenames_before)
            np.testing.assert_array_equal(
                table.motion_correction["fn_adata"], align_before
            )
            self.assertEqual(table.source_path, source)
            self.assertEqual(
                keep.call_args.args[0][0, 1],
                str(table.savedr / "motion_correction" / "p0-selected.h5"),
            )
            self.assertEqual(low_read.call_count, 4)
            self.assertEqual(high_read.call_count, 2)
            self.assertEqual(mapper.call_count, 4)
            for call in mapper.call_args_list:
                self.assertEqual(call.args[2], options.max_workers)
                self.assertEqual(list(call.args[1]), [0, 1])
            if rois:
                geometry.assert_called_once()
                self.assertFalse(resolve_rois.call_args.kwargs["interactive"])
            else:
                geometry.assert_not_called()
                resolve_rois.assert_not_called()

            writer.assert_called_once()
            output = (
                table.savedr / "source_extraction" / "experiment_summary.h5"
            )
            self.assertIs(writer.call_args.args[0], summary)
            self.assertEqual(writer.call_args.args[1], output)
            restored = read_summary(output)
            self.assertEqual(len(summary.paths), 2)
            self.assertEqual(len(restored.paths), 2)
            self.assertEqual(summary.params["numChannels"], 2)
            self.assertEqual(summary.params["operator"], "workflow test")
            self.assertEqual(summary.params["draw_user_rois"], rois)
            for dmd_ix, (path, saved) in enumerate(
                zip(summary.paths, restored.paths)
            ):
                runtime_result = assemble.call_args_list[dmd_ix].args[0]
                self.assertEqual(len(runtime_result.trace_results), 1)
                self.assertIsInstance(
                    runtime_result.trace_results[0], TrialTraceResult
                )
                self.assertEqual(path.path_index, dmd_ix)
                np.testing.assert_array_equal(path.z_depths, [[10 + dmd_ix]])
                np.testing.assert_array_equal(
                    path.frame_info.trial_num_frames, [[n_frames]]
                )
                np.testing.assert_array_equal(
                    path.frame_info.frame_line_idxs.ravel(), frames
                )
                np.testing.assert_array_equal(
                    path.frame_info.offlineYshifts, -2
                )
                np.testing.assert_array_equal(
                    path.frame_info.offlineXshifts, 1
                )
                np.testing.assert_array_equal(path.frame_info.onlineYshifts, 4)
                np.testing.assert_array_equal(path.frame_info.onlineXshifts, 5)
                np.testing.assert_allclose(
                    path.global_f[0], n_sp * (10 + 5 * dmd_ix)
                )
                np.testing.assert_array_equal(path.global_f, saved.global_f)
                np.testing.assert_array_equal(
                    path.frame_info.offlineXshifts,
                    saved.frame_info.offlineXshifts,
                )
                np.testing.assert_allclose(
                    path.sources[0].df_ls,
                    saved.sources[0].df_ls,
                    equal_nan=True,
                )
                self.assertEqual(len(path.user_rois), int(rois))
                self.assertEqual(saved.annotation_enabled, rois)
                if rois:
                    self.assertEqual(path.user_rois[0].label, f"ROI {dmd_ix}")
                    self.assertEqual(path.user_rois[0].f.shape, (2, n_frames))
            self.assertNotEqual(
                summary.paths[0].global_f[0, 0],
                summary.paths[1].global_f[0, 0],
            )
        self.assertEqual(
            before, (asdict(params), asdict(options), asdict(annotation))
        )
        self.assertIsNone(params.num_channels)

    def test_exact_nonstandard_selected_trial_table_path(self):
        """Preserve the chosen filename through extraction and persistence."""
        self._run_mock_session()

    def test_loaded_table_is_not_reopened(self):
        """Prepare a loaded model directly without a second file read."""
        self._run_mock_session(loaded=True)

    def test_split_policy_preserves_rois_and_caller_options(self):
        """Split policy leaves scientific options unchanged."""
        self._run_mock_session(loaded=True, rois=True)

    def test_missing_channel_inference_fails_before_processing(self):
        """Missing channel metadata raises before ROI resolution or writing."""
        params = BandSiloParams()
        with TemporaryDirectory() as directory, ExitStack() as stack:
            table = dict(
                savedr=directory,
                n_dmds=1,
                fn_adata=[],
                filename=[],
                datadr=directory,
            )
            stack.enter_context(
                patch.object(wf, "load_trial_table", return_value=table)
            )
            stack.enter_context(patch.object(wf.inputs, "load_lookup_table"))
            stack.enter_context(patch.object(wf.inputs, "load_psf"))
            stack.enter_context(patch.object(wf, "compute_keep_trials"))
            stack.enter_context(
                patch.object(wf, "read_align_info", return_value=({}, None))
            )
            rois = stack.enter_context(
                patch.object(wf, "_resolve_session_user_rois")
            )
            process = stack.enter_context(patch.object(wf, "_process_dmd"))
            writer = stack.enter_context(patch.object(wf, "write_summary"))
            with self.assertRaisesRegex(
                ValueError, "num_channels.*determined"
            ):
                wf.extract_band_sources("selected.h5", params)
            rois.assert_not_called()
            process.assert_not_called()
            writer.assert_not_called()
            self.assertIsNone(params.num_channels)

    def test_enabled_empty_annotations_keep_resolution_context(self):
        """An enabled empty selection retains geometry and headless policy."""
        table, lookup = {"n_dmds": 2}, {"lookup": "unchanged"}
        geometry, references, selection = {}, {"DMD1": "reference.tif"}, {}
        with (
            patch.object(
                wf,
                "build_user_roi_geometry",
                return_value=(geometry, references),
            ) as build,
            patch.object(
                wf, "resolve_user_rois", return_value=selection
            ) as resolve,
        ):
            result = wf._resolve_session_user_rois(
                "results",
                table,
                lookup,
                AnnotationOptions(enabled=True, interactive=False),
            )
        build.assert_called_once_with(table, lookup)
        resolve.assert_called_once_with(
            Path("results") / "annotations",
            2,
            geometry,
            interactive=False,
            ref_files=references,
        )
        self.assertIs(result, selection)


if __name__ == "__main__":
    unittest.main()
