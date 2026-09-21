"""Canonical band output assembly, shared writing and localization wiring.

Tests target the implementation owners rather than compatibility re-exports.
Public driver composition is covered separately in test_band_workflow.
"""

import os
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import call, patch

import h5py
import numpy as np
import torch

from giant_python.extraction.band import result_assembly as pl
from giant_python.extraction.band import workflow as wf
from giant_python.extraction.band.types import TrialTraceResult
from giant_python.io.experiment_summary import write_summary
from giant_python.models.experiment_summary import (
    FRAME_INFO_KEYS,
    ExperimentSummary,
)
from giant_python.models.params import BandSiloParams, ExecutionOptions


def _trial(n_frames, n_sources, n_channels, n_soma):
    """One synthetic named trial record, matching the trace producer."""
    rng = np.random.default_rng(n_frames)
    return TrialTraceResult(
        rng.random((n_frames, n_sources)).astype(np.float32),  # phi (dF_ls)
        (rng.random((n_frames, n_sources)) + 5).astype(np.float32),  # F0
        np.arange(n_frames, dtype=float),  # frames
        np.arange(3),  # sel_pix_idxs
        rng.random((n_frames, n_channels)).astype(np.float32),  # global_f
        (rng.random(n_frames), rng.random(n_frames), rng.random(n_frames)),
        (
            np.zeros(n_frames, np.int16),
            np.zeros(n_frames, np.int16),
            np.zeros(n_frames, np.int16),
        ),
        rng.random((n_frames, n_soma, n_channels)).astype(np.float32),
    )


def _path_result(
    n_sources=2,
    n_channels=1,
    nz=1,
    npc=4,
    npr=5,
    draw_user_rois=False,
    soma_masks=None,
    soma_labels=None,
    n_soma=0,
):
    """Build a synthetic PathResult for one DMD path."""
    n_pixels = nz * npc * npr
    results = [
        _trial(6, n_sources, n_channels, n_soma),
        _trial(7, n_sources, n_channels, n_soma),
    ]
    return pl.PathResult(
        a=torch.rand((n_pixels, n_sources)),
        source_params=torch.rand((n_sources, 6)),
        source_snr=np.arange(1, n_sources + 1, dtype=np.float32),
        act_im=np.random.rand(nz, npc, npr).astype(np.float32),
        mean_im=np.random.rand(n_channels, nz, npc, npr).astype(np.float32),
        act_im_peaks=np.zeros((n_sources, 3)),
        trace_results=results,
        z_depths=np.zeros((nz, 1), dtype=np.int32),
        num_fast_zs=nz,
        dmd_pixels_per_column=npc,
        dmd_pixels_per_row=npr,
        num_channels=n_channels,
        denoise_window=3,
        baseline_window=5,
        draw_user_rois=draw_user_rois,
        soma_masks=soma_masks,
        soma_labels=soma_labels,
        yx_shape=(npc, npr),
    )


class TestLocalizeMedianMotion(unittest.TestCase):
    """Check median geometry without running the expensive rho/NMF stages."""

    def test_dog_settings_and_all_source_columns_reach_output(self):
        """Keep DoG widths, reference centers and source matrices unchanged."""
        ref_d = np.array([0, 0])
        ref_r, ref_c = np.array([1, 2]), np.array([2, 2])
        selected = np.array([0, 1, 2])
        coords = np.zeros((3, 3))
        umyx = np.array([[2, -1]])
        mot_inds = np.zeros(4, dtype=int)
        residual = np.ones((2, 4))
        rho = np.ones((3, 4))
        seeds = np.array([[0, 1, 2], [0, 2, 2]])
        fitted = {
            "A": torch.ones((12, 2)),
            "source_params": torch.ones((2, 6)),
            "source_snr": np.ones(2),
        }
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    wf.operators, "build_motion_h_matrices", return_value=["H"]
                )
            )
            kernels = stack.enter_context(
                patch.object(
                    wf.operators,
                    "gaussian_kernel_2d",
                    side_effect=[("center", (0, 0)), ("surround", (1, 1))],
                )
            )
            stack.enter_context(
                patch.object(
                    wf.geo,
                    "selected_pixels_2d_for_plane",
                    return_value=(selected, coords),
                )
            )
            convolution = stack.enter_context(
                patch.object(
                    wf.operators,
                    "build_convolution_matrix",
                    side_effect=["D", "D_exp"],
                )
            )
            compute_rho = stack.enter_context(
                patch.object(wf.activity, "compute_rho", return_value=rho)
            )
            stack.enter_context(
                patch.object(
                    wf.activity, "mask_high_nan_rho", return_value=np.zeros(3)
                )
            )
            stack.enter_context(patch.object(wf.activity, "decay_kernel_1d"))
            stack.enter_context(
                patch.object(wf.activity, "smooth_rho", return_value=rho)
            )
            stack.enter_context(
                patch.object(wf.si, "accumulate_activity_image")
            )
            stack.enter_context(patch.object(wf.si, "finalize_activity_image"))
            stack.enter_context(
                patch.object(wf, "get_act_im_peaks", return_value=seeds)
            )
            fit = stack.enter_context(
                patch.object(wf, "fit_sources", return_value=fitted)
            )
            result = wf._localize(
                residual,
                umyx,
                mot_inds,
                selected,
                coords,
                12,
                ref_d,
                ref_r,
                ref_c,
                1,
                3,
                4,
                np.ones((3, 3)),
                2,
                None,
                None,
                10,
                BandSiloParams(),
                execution=ExecutionOptions(verbose=False),
            )
        self.assertEqual(kernels.call_args_list, [call(0.9), call(4.5)])
        self.assertEqual(
            convolution.call_args_list[0].args[1:], ("center", (0, 0))
        )
        self.assertEqual(
            convolution.call_args_list[1].args[1:], ("surround", (1, 1))
        )
        self.assertIs(compute_rho.call_args.args[7], ref_d)
        self.assertIs(compute_rho.call_args.args[8], ref_r)
        self.assertIs(compute_rho.call_args.args[9], ref_c)
        self.assertIs(fit.call_args.args[0], seeds)
        self.assertIs(fit.call_args.args[1], residual)
        self.assertIs(fit.call_args.args[3], umyx)
        self.assertIs(fit.call_args.args[4], mot_inds)
        self.assertIs(fit.call_args.args[5], selected)
        self.assertIs(fit.call_args.args[6], coords)
        self.assertIs(result[2], fitted["A"])
        self.assertIs(result[3], fitted["source_params"])
        self.assertIs(result[4], fitted["source_snr"])
        np.testing.assert_array_equal(result[1], seeds)
        self.assertIsNot(result[1], seeds)

    def test_dominant_retained_motion_shifts_only_median_geometry(self):
        """Dropped frames do not vote; default run policy preserves centers."""
        ref_r = np.array([10, 19])
        ref_c = np.array([8, 8])
        # Dropped frames outnumber either retained bin and must be ignored.
        mot_inds = np.array([-1, -1, -1, -1, 0, 1, 1, 1])
        for displacement in ((2, -5), (-3, 4), (0, 0)):
            with self.subTest(displacement=displacement), ExitStack() as stack:
                umyx = np.array([[7, 9], displacement], dtype=float)
                for owner, name in (
                    (wf.operators, "build_motion_h_matrices"),
                    (wf.operators, "gaussian_kernel_2d"),
                    (wf.geo, "selected_pixels_2d_for_plane"),
                    (wf.operators, "build_convolution_matrix"),
                    (wf.activity, "compute_rho"),
                    (wf.activity, "mask_high_nan_rho"),
                    (wf.activity, "decay_kernel_1d"),
                    (wf.activity, "smooth_rho"),
                ):
                    stack.enter_context(
                        patch.object(owner, name, return_value=(None, None))
                    )
                stack.enter_context(
                    patch.object(wf.si, "accumulate_activity_image")
                )
                # Stop at finalization; peak detection and NMF are unrelated.
                finalize = stack.enter_context(
                    patch.object(
                        wf.si,
                        "finalize_activity_image",
                        side_effect=RuntimeError("median reached"),
                    )
                )
                with self.assertRaisesRegex(RuntimeError, "median reached"):
                    wf._localize(
                        residual=np.zeros((2, mot_inds.size)),
                        umyx=umyx,
                        mot_inds_yx=mot_inds,
                        sel_pix_idxs=np.array([0]),
                        pixel_coords=None,
                        n_pixels=1600,
                        ref_d=np.array([0, 0]),
                        ref_r=ref_r,
                        ref_c=ref_c,
                        num_fast_zs=1,
                        dmd_pixels_per_column=40,
                        dmd_pixels_per_row=40,
                        psf2d=np.ones((3, 3)),
                        num_super_pixels=2,
                        sparse_h_inds=None,
                        sparse_h_vals=None,
                        align_hz=10,
                        params=BandSiloParams(),
                    )
                finalize.assert_called_once()
                np.testing.assert_array_equal(
                    finalize.call_args.kwargs["ref_d"], [0, 0]
                )
                np.testing.assert_array_equal(
                    finalize.call_args.kwargs["ref_r"], ref_r + displacement[0]
                )
                np.testing.assert_array_equal(
                    finalize.call_args.kwargs["ref_c"], ref_c + displacement[1]
                )
                np.testing.assert_array_equal(ref_r, [10, 19])
                np.testing.assert_array_equal(ref_c, [8, 8])


class TestStageComposition(unittest.TestCase):
    """Check background wiring and empty localization without sparse fits."""

    def test_default_execution_skips_empty_planes_and_preserves_grid(self):
        """Interpolate selected planes; baseline and assembly share data."""
        data = np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        selected = np.array([4, 5])
        ref_d, ref_r, ref_c = (
            np.array([1, 1]),
            np.array([0, 1]),
            np.array([0, 1]),
        )
        umyx, mot_inds = np.array([[0, 0]]), np.array([0, 0])
        plane_pixels = np.array([[0, 0], [0, 1]])
        interpolated = data + 10
        baseline, assembled = data + 20, data + 30
        params = BandSiloParams(baseline_window_s=0.5)
        with ExitStack() as stack:
            expand = stack.enter_context(
                patch.object(wf.operators, "expand_psf")
            )
            planes = stack.enter_context(
                patch.object(
                    wf.geo,
                    "selected_pixels_2d_for_plane",
                    side_effect=[
                        (np.array([], dtype=int), np.empty((0, 2))),
                        (np.array([0, 1]), plane_pixels),
                    ],
                )
            )
            interpolate = stack.enter_context(
                patch.object(
                    wf.bg,
                    "build_interp_data",
                    return_value=(interpolated, None, None),
                )
            )
            rolling = stack.enter_context(
                patch.object(
                    wf.bg, "compute_rolling_baseline", return_value=baseline
                )
            )
            assemble = stack.enter_context(
                patch.object(
                    wf.bg, "assemble_background", return_value=assembled
                )
            )
            psf = np.ones((3, 3))
            result = wf._estimate_background(
                data,
                umyx,
                mot_inds,
                selected,
                ref_d,
                ref_r,
                ref_c,
                2,
                2,
                2,
                psf,
                2,
                10,
                params,
            )
        expand.assert_called_once_with(psf)
        self.assertEqual(planes.call_count, 2)
        interpolate.assert_called_once()
        np.testing.assert_array_equal(interpolate.call_args.args[0], data)
        np.testing.assert_array_equal(interpolate.call_args.args[1], ref_r)
        np.testing.assert_array_equal(interpolate.call_args.args[2], ref_c)
        self.assertIs(interpolate.call_args.args[3], plane_pixels)
        self.assertEqual(
            interpolate.call_args.kwargs["method"],
            params.background_interpolation,
        )
        rolling.assert_called_once()
        np.testing.assert_array_equal(rolling.call_args.args[0], interpolated)
        self.assertEqual(rolling.call_args.args[0].dtype, np.float32)
        self.assertEqual(rolling.call_args.args[1], 5)
        assemble.assert_called_once()
        self.assertIs(assemble.call_args.args[0], baseline)
        self.assertIs(assemble.call_args.args[1], umyx)
        self.assertIs(assemble.call_args.args[2], mot_inds)
        self.assertEqual(assemble.call_args.args[7:], (2, 2, 2, 2))
        self.assertIs(result, assembled)

    def test_no_retained_motion_bins_do_not_invent_a_source_fit(self):
        """Characterize the existing empty argmax error without a fallback."""
        with ExitStack() as stack:
            for owner, name in (
                (wf.operators, "build_motion_h_matrices"),
                (wf.activity, "compute_rho"),
                (wf.activity, "mask_high_nan_rho"),
                (wf.activity, "smooth_rho"),
                (wf.si, "accumulate_activity_image"),
            ):
                stack.enter_context(patch.object(owner, name))
            finalize = stack.enter_context(
                patch.object(wf.si, "finalize_activity_image")
            )
            fit = stack.enter_context(patch.object(wf, "fit_sources"))
            with self.assertRaisesRegex(ValueError, "argmax.*empty"):
                wf._localize(
                    np.ones((2, 3)),
                    np.empty((0, 2)),
                    np.full(3, -1),
                    np.array([], dtype=int),
                    np.empty((0, 3)),
                    4,
                    np.array([0, 0]),
                    np.array([0, 1]),
                    np.array([0, 1]),
                    0,
                    2,
                    2,
                    np.ones((3, 3)),
                    2,
                    None,
                    None,
                    10,
                    BandSiloParams(),
                )
        finalize.assert_not_called()
        fit.assert_not_called()


class TestDmdUserRois(unittest.TestCase):
    """_dmd_user_rois slices a resolved session selection per DMD."""

    def test_none_is_neutral(self):
        """No session selection yields (None, None, [])."""
        self.assertEqual(pl._dmd_user_rois(None, "DMD1"), (None, None, []))

    def test_slices_key(self):
        """The named DMD's masks, labels, and superpixels are returned."""
        sps = [np.array([0, 1])]
        user_rois = {
            "user_roi_masks": {"DMD1": ["mask"]},
            "user_roi_labels": {"DMD1": ["roi"]},
            "user_roi_superpixels": {"DMD1": sps},
        }
        masks, labels, out_sps = pl._dmd_user_rois(user_rois, "DMD1")
        self.assertEqual(masks, ["mask"])
        self.assertEqual(labels, ["roi"])
        self.assertIs(out_sps, sps)

    def test_missing_key_defaults(self):
        """A DMD absent from the selection yields empty superpixels."""
        user_rois = {
            "user_roi_masks": {},
            "user_roi_labels": {},
            "user_roi_superpixels": {},
        }
        masks, labels, out_sps = pl._dmd_user_rois(user_rois, "DMD2")
        self.assertIsNone(masks)
        self.assertIsNone(labels)
        self.assertEqual(out_sps, [])


class TestWithNanChannel(unittest.TestCase):
    """_with_nan_channel reshapes and NaN-fills non-activity channels."""

    def test_shape_and_channels(self):
        """(frames, sources) -> (sources, channels, frames); ch>0 is NaN."""
        arr = np.arange(6, dtype=np.float32).reshape(3, 2)  # 3 frames, 2 src
        out = pl._with_nan_channel(arr, 2)
        self.assertEqual(out.shape, (2, 2, 3))
        np.testing.assert_array_equal(out[:, 0, :], arr.T)
        self.assertTrue(np.all(np.isnan(out[:, 1, :])))


class TestAssemblePathSummary(unittest.TestCase):
    """Typed path assembly preserves numerical shapes and content."""

    def test_shapes(self):
        """Every dataset has the schema shape."""
        pr = _path_result(n_sources=2, n_channels=1, nz=1, npc=4, npr=5)
        path = pl.assemble_path_summary(pr)
        self.assertEqual(len(path.sources), 2)
        for source in path.sources:
            self.assertEqual(source.profile.shape, (1, 4, 5))
            self.assertEqual(source.coords.shape, (3,))
            self.assertEqual(source.df_ls.shape, (1, 13))
            self.assertEqual(source.f0.shape, (1, 13))
            self.assertEqual(source.df_ls.dtype, np.float64)
            self.assertEqual(source.f0.dtype, np.float64)
        np.testing.assert_array_equal([s.snr for s in path.sources], [1, 2])
        self.assertEqual(path.global_f.shape, (1, 13))
        self.assertEqual(
            list(path.frame_info.trial_num_frames.ravel()), [6, 7]
        )
        self.assertEqual(path.frame_info.discard_frames.shape, (13, 1))
        self.assertEqual(path.user_rois, [])
        self.assertFalse(path.annotation_enabled)
        self.assertIs(path.z_depths, pr.z_depths)

    def test_two_channels_global_shape(self):
        """Two channels widen the global-F channel axis."""
        pr = _path_result(n_channels=2)
        path = pl.assemble_path_summary(pr)
        self.assertEqual(path.global_f.shape, (2, 13))
        self.assertEqual(path.sources[0].df_ls.shape, (2, 13))
        self.assertTrue(np.isnan(path.sources[0].df_ls[1]).all())

    def test_user_rois_assembled(self):
        """draw_user_rois with masks assembles the user-ROI datasets."""
        masks = [np.ones((1, 4, 5), dtype=bool), np.zeros((1, 4, 5), bool)]
        pr = _path_result(
            draw_user_rois=True,
            soma_masks=masks,
            soma_labels=["SOMA", "SOMA2"],
            n_soma=2,
        )
        path = pl.assemble_path_summary(pr)
        self.assertTrue(path.annotation_enabled)
        self.assertEqual(len(path.user_rois), 2)
        for index, roi in enumerate(path.user_rois):
            np.testing.assert_array_equal(roi.mask, masks[index])
            self.assertEqual(roi.f.shape, (1, 13))
            self.assertEqual(roi.f.dtype, np.float32)
            np.testing.assert_array_equal(
                roi.f,
                np.concatenate(
                    [r.user_roi_f[:, index] for r in pr.trace_results]
                ).T,
            )

    def test_user_rois_empty_masklist(self):
        """draw_user_rois with an empty mask list yields a 0-ROI mask array."""
        pr = _path_result(
            draw_user_rois=True, soma_masks=[], soma_labels=[], n_soma=0
        )
        path = pl.assemble_path_summary(pr)
        self.assertTrue(path.annotation_enabled)
        self.assertEqual(path.user_rois, [])
        self.assertEqual(path._empty_roi_arrays["mask"].shape, (0, 1, 4, 5))
        self.assertEqual(path._empty_roi_arrays["F"].shape, (0, 1, 13))

    def test_empty_sources_preserve_geometry_and_frame_data(self):
        """No sources still produces shaped empty tensors and real globals."""
        pr = _path_result(n_sources=0, n_channels=2)
        path = pl.assemble_path_summary(pr)
        self.assertEqual(path.sources, [])
        self.assertEqual(
            path._empty_source_arrays["profiles"].shape, (0, 1, 4, 5)
        )
        self.assertEqual(path._empty_source_arrays["dF_ls"].shape, (0, 2, 13))
        self.assertEqual(path._empty_source_arrays["SNR"].dtype, np.float32)
        self.assertEqual(path.global_f.shape, (2, 13))
        self.assertFalse(path.frame_info.discard_frames.any())


class TestWriteExperimentSummary(unittest.TestCase):
    """write_summary produces the MATLAB-compatible schema."""

    def _write_and_read(self, assembled_paths, params=None):
        """Write then reopen, returning the h5 file handle's data as dict."""
        params = params or {"operator": "test", "numChannels": 1}
        path = tempfile.mktemp(suffix=".h5")
        write_summary(
            ExperimentSummary(paths=assembled_paths, params=params), path
        )
        return path

    def test_schema_and_values(self):
        """Root flag/params + one Path group with the expected datasets."""
        asm = pl.assemble_path_summary(_path_result())
        path = self._write_and_read([asm])
        try:
            with h5py.File(path, "r") as f:
                self.assertEqual(int(f["row_major"][()]), 1)
                self.assertIn("params", f)
                grp = f["Path1"]
                self.assertEqual(
                    grp["sources/spatial/profiles"].shape, (2, 1, 4, 5)
                )
                self.assertEqual(
                    grp["sources/temporal/dF_ls"].shape, (2, 1, 13)
                )
                self.assertEqual(grp["global/F"].shape, (1, 13))
                np.testing.assert_array_equal(grp["Z_depths"], asm.z_depths)
                self.assertNotIn("z_depths", grp)
                for key in FRAME_INFO_KEYS:
                    self.assertIn(key, grp["frame_info"])
                self.assertIn("act_im", grp["visualizations"])
                self.assertNotIn("user_rois", grp)
        finally:
            os.remove(path)

    def test_user_rois_written(self):
        """The user_rois group is written when soma ROIs are present."""
        masks = [np.ones((1, 4, 5), dtype=bool)]
        pr = _path_result(
            draw_user_rois=True,
            soma_masks=masks,
            soma_labels=["SOMA"],
            n_soma=1,
        )
        path = self._write_and_read([pl.assemble_path_summary(pr)])
        try:
            with h5py.File(path, "r") as f:
                ur = f["Path1/user_rois"]
                self.assertEqual(ur["mask"].shape, (1, 1, 4, 5))
                self.assertEqual(ur["F"].shape, (1, 1, 13))
                self.assertEqual(ur["labels"].shape, (1, 1))
        finally:
            os.remove(path)

    def test_multi_path_and_overwrite(self):
        """Two paths write Path1/Path2; re-writing overwrites in place."""
        asm = [pl.assemble_path_summary(_path_result(), i) for i in range(2)]
        path = self._write_and_read(asm)
        try:
            # second write to the same file overwrites the Path groups
            write_summary(
                ExperimentSummary(paths=asm, params={"operator": "test"}), path
            )
            with h5py.File(path, "r") as f:
                self.assertIn("Path1", f)
                self.assertIn("Path2", f)
        finally:
            os.remove(path)


class TestExperimentSummary(unittest.TestCase):
    """Construct summaries directly from numerical path models."""

    def test_all_path_fields_are_retained(self):
        """Paths retain distinct images, frame metadata and globals."""
        assembled = [
            pl.assemble_path_summary(_path_result(), i) for i in range(2)
        ]
        assembled[0].z_depths = np.array([[10]])
        assembled[1].z_depths = np.array([[20]])
        assembled[0].global_f[:] = 3
        assembled[1].global_f[:] = 7
        summary = ExperimentSummary(paths=assembled, params={})
        self.assertEqual(len(summary.paths), 2)
        for index, path in enumerate(summary.paths):
            self.assertEqual(path.path_index, index)
            np.testing.assert_array_equal(path.z_depths, [[10 + 10 * index]])
            np.testing.assert_array_equal(path.global_f, 3 + 4 * index)
            self.assertIs(path, assembled[index])

    def test_sources_and_visualizations(self):
        """Per-path sources are populated with profile/coords/traces/SNR."""
        asm = pl.assemble_path_summary(_path_result(n_sources=2))
        summary = ExperimentSummary(paths=[asm], params={"operator": "t"})
        self.assertEqual(len(summary.paths), 1)
        self.assertEqual(len(summary.paths[0].sources), 2)
        src = summary.paths[0].sources[0]
        self.assertEqual(src.profile.shape, (1, 4, 5))
        self.assertEqual(src.snr, 1.0)
        self.assertIsNotNone(summary.paths[0].visualizations.act_im)
        self.assertEqual(summary.paths[0].user_rois, [])
        self.assertIsNotNone(summary.paths[0].frame_info.trial_num_frames)

    def test_user_rois_populated(self):
        """User ROIs appear in the summary when present."""
        masks = [np.ones((1, 4, 5), dtype=bool)]
        pr = _path_result(
            draw_user_rois=True,
            soma_masks=masks,
            soma_labels=["SOMA"],
            n_soma=1,
        )
        asm = pl.assemble_path_summary(pr)
        summary = ExperimentSummary(paths=[asm], params={"operator": "t"})
        self.assertEqual(len(summary.paths[0].user_rois), 1)
        self.assertEqual(summary.paths[0].user_rois[0].label, "SOMA")


if __name__ == "__main__":
    unittest.main()
