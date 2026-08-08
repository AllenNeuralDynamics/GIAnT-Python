"""Tests for giant_python.bandsilo.pipeline output assembly + writing.

Covers the pure/testable Phase-8 surface: per-path output assembly, the
byte-compatible ``experiment_summary.h5`` writer, and the returned
``ExperimentSummary`` builder. The full orchestration driver
(``extract_band_sources`` / ``_process_dmd``) reads SLAP2 files, runs
the torch NMF, and does disk IO end-to-end, so it is not unit-tested here (its
Phase 1-7 kernels are each covered/cross-checked in their own modules).
"""

import os
import tempfile
import unittest

import h5py
import numpy as np
import torch

from giant_python.bandsilo import pipeline as pl


def _trial(n_frames, n_sources, n_channels, n_soma):
    """One synthetic get_high_res_traces 8-tuple."""
    rng = np.random.default_rng(n_frames)
    return (
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


class TestAssemblePathOutputs(unittest.TestCase):
    """assemble_path_outputs shapes and content."""

    def test_shapes(self):
        """Every dataset has the schema shape."""
        pr = _path_result(n_sources=2, n_channels=1, nz=1, npc=4, npr=5)
        asm = pl.assemble_path_outputs(pr)
        self.assertEqual(asm["profiles"].shape, (2, 1, 4, 5))
        self.assertEqual(asm["coords"].shape, (2, 3))
        self.assertEqual(asm["dF_ls"].shape, (2, 1, 13))  # 6+7 frames
        self.assertEqual(asm["F0"].shape, (2, 1, 13))
        self.assertEqual(asm["SNR"].shape, (2, 1))
        self.assertEqual(asm["global_f"].shape, (1, 13))
        self.assertEqual(list(asm["trial_num_frames"].ravel()), [6, 7])
        self.assertEqual(asm["discard_frames"].shape, (13, 1))
        self.assertIsNone(asm["user_rois"])

    def test_two_channels_global_shape(self):
        """Two channels widen the global-F channel axis."""
        pr = _path_result(n_channels=2)
        asm = pl.assemble_path_outputs(pr)
        self.assertEqual(asm["global_f"].shape, (2, 13))
        self.assertEqual(asm["dF_ls"].shape[1], 2)

    def test_user_rois_assembled(self):
        """draw_user_rois with masks assembles the user-ROI datasets."""
        masks = [np.ones((1, 4, 5), dtype=bool), np.zeros((1, 4, 5), bool)]
        pr = _path_result(
            draw_user_rois=True,
            soma_masks=masks,
            soma_labels=["SOMA", "SOMA2"],
            n_soma=2,
        )
        asm = pl.assemble_path_outputs(pr)
        self.assertIsNotNone(asm["user_rois"])
        self.assertEqual(asm["user_rois"]["mask"].shape, (2, 1, 4, 5))
        # F: (rois, channels, total_frames)
        self.assertEqual(asm["user_rois"]["F"].shape, (2, 1, 13))

    def test_user_rois_empty_masklist(self):
        """draw_user_rois with an empty mask list yields a 0-ROI mask array."""
        pr = _path_result(
            draw_user_rois=True, soma_masks=[], soma_labels=[], n_soma=0
        )
        asm = pl.assemble_path_outputs(pr)
        self.assertEqual(asm["user_rois"]["mask"].shape, (0, 1, 4, 5))


class TestWriteExperimentSummary(unittest.TestCase):
    """write_experiment_summary produces the byte-compatible schema."""

    def _write_and_read(self, assembled_paths, params=None):
        """Write then reopen, returning the h5 file handle's data as dict."""
        params = params or {"operator": "test", "numChannels": 1}
        path = tempfile.mktemp(suffix=".h5")
        pl.write_experiment_summary(path, params, assembled_paths)
        return path

    def test_schema_and_values(self):
        """Root flag/params + one Path group with the expected datasets."""
        asm = pl.assemble_path_outputs(_path_result())
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
                for key in pl._FRAME_INFO_KEYS:
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
        path = self._write_and_read([pl.assemble_path_outputs(pr)])
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
        asm = [pl.assemble_path_outputs(_path_result()) for _ in range(2)]
        path = self._write_and_read(asm)
        try:
            # second write to the same file overwrites the Path groups
            pl.write_experiment_summary(path, {"operator": "test"}, asm)
            with h5py.File(path, "r") as f:
                self.assertIn("Path1", f)
                self.assertIn("Path2", f)
        finally:
            os.remove(path)


class TestBuildExperimentSummary(unittest.TestCase):
    """build_experiment_summary populates the returned dataclass."""

    def test_sources_and_visualizations(self):
        """Per-path sources are populated with profile/coords/traces/SNR."""
        asm = pl.assemble_path_outputs(_path_result(n_sources=2))
        summary = pl.build_experiment_summary({"operator": "t"}, [asm])
        self.assertEqual(len(summary.sources), 1)
        self.assertEqual(len(summary.sources[0]), 2)
        src = summary.sources[0][0]
        self.assertEqual(src.profile.shape, (1, 4, 5))
        self.assertEqual(src.snr, 1.0)
        self.assertIsNotNone(summary.visualizations.act_im)
        self.assertEqual(summary.user_rois, [[]])
        self.assertIn("trial_num_frames", summary.frame_info)

    def test_user_rois_populated(self):
        """User ROIs appear in the summary when present."""
        masks = [np.ones((1, 4, 5), dtype=bool)]
        pr = _path_result(
            draw_user_rois=True,
            soma_masks=masks,
            soma_labels=["SOMA"],
            n_soma=1,
        )
        asm = pl.assemble_path_outputs(pr)
        summary = pl.build_experiment_summary({"operator": "t"}, [asm])
        self.assertEqual(len(summary.user_rois[0]), 1)
        self.assertEqual(summary.user_rois[0][0].label, "SOMA")


if __name__ == "__main__":
    unittest.main()
