"""Tests for the models subpackage."""

import os
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np

from giant_python.models import (
    AnnotationOptions,
    BandSiloParams,
    ExperimentSummary,
    FrameInfo,
    PathSummary,
    Slap2Info,
    Source,
    TrialTable,
    UserRoi,
    Visualizations,
)


class TestDataclasses(unittest.TestCase):
    """Dataclasses should construct with sensible defaults."""

    def test_trial_table_defaults(self):
        """TrialTable constructs empty with no motion/source groups."""
        tt = TrialTable()
        self.assertIsNone(tt.motion_correction)
        self.assertIsNone(tt.source_extraction)

    def test_slap2_info_defaults(self):
        """Slap2Info constructs with empty collections."""
        self.assertEqual(Slap2Info().ref_stack, {})

    def test_experiment_summary_defaults(self):
        """ExperimentSummary owns per-path results, not aggregate aliases."""
        summary = ExperimentSummary()
        self.assertEqual(summary.paths, [])
        path = PathSummary()
        summary.paths.append(path)
        self.assertIsInstance(summary.paths[0].visualizations, Visualizations)
        self.assertIsInstance(summary.paths[0].frame_info, FrameInfo)
        self.assertEqual(summary.paths[0].sources, [])
        for name in (
            "sources",
            "user_rois",
            "visualizations",
            "frame_info",
            "global_f",
            "z_depths",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(summary, name))

    def test_source_and_user_roi(self):
        """Source and UserRoi construct with defaults."""
        self.assertIsNone(Source().snr)
        self.assertEqual(UserRoi().type, "polygon")

    def test_param_models(self):
        """Fixed interpolation is not exposed as a model parameter."""
        self.assertFalse(hasattr(BandSiloParams(), "background_interpolation"))

    def test_band_value_params_have_concrete_defaults(self):
        """Value params default to concrete (non-None) values."""
        p = BandSiloParams()
        self.assertEqual(p.analyze_hz, 100.0)
        # Seven is the current implementation default, not a new retuning.
        self.assertEqual(p.peakth, 7.0)
        # Only genuinely runtime-resolved fields stay None sentinels.
        self.assertIsNone(p.num_channels)
        self.assertIsNone(AnnotationOptions().interactive)


class TestH5RoundTrip(unittest.TestCase):
    """Implemented model codecs preserve data and delegate to shared IO."""

    def test_trial_table_from_h5(self):
        """from_h5 faithfully mirrors the trial_table.h5 group hierarchy."""
        str_dt = h5py.string_dtype(encoding="utf-8")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trial_table.h5")
            with h5py.File(path, "w") as f:
                f.create_dataset("row_major", data=1)
                f.create_dataset("datadr", data="/data", dtype=str_dt)
                f.create_dataset("savedr", data="/res", dtype=str_dt)
                f.create_dataset(
                    "filename",
                    data=np.array([["t1", "t2"]], dtype=object),
                    dtype=str_dt,
                )
                si = f.create_group("slap2_info")
                si.create_dataset("first_line", data=np.array([[1, 2]]))
                si.create_dataset("last_line", data=np.array([[10, 20]]))
                rs = si.create_group("ref_stack")
                rs.create_group("Path1").create_dataset(
                    "channels", data=np.array([1, 2])
                )
                mc = f.create_group("motion_correction")
                mc.create_dataset(
                    "fn_adata",
                    data=np.array([["a.h5", ""]], dtype=object),
                    dtype=str_dt,
                )
            tt = TrialTable.from_h5(path)
        self.assertEqual(str(tt.datadr), os.path.normpath("/data"))
        self.assertEqual(str(tt.savedr), os.path.normpath("/res"))
        self.assertIsInstance(tt.slap2_info, Slap2Info)
        np.testing.assert_array_equal(
            np.asarray(tt.slap2_info.first_line), np.array([[1, 2]])
        )
        self.assertIn("Path1", tt.slap2_info.ref_stack)
        self.assertEqual(tt.motion_correction["fn_adata"][0, 0], "a.h5")
        self.assertIsNone(tt.source_extraction)

    def test_trial_table_from_h5_non_slap2(self):
        """A table without slap2_info yields slap2_info=None."""
        str_dt = h5py.string_dtype(encoding="utf-8")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trial_table.h5")
            with h5py.File(path, "w") as f:
                f.create_dataset("row_major", data=1)
                f.create_dataset("datadr", data="/data", dtype=str_dt)
                f.create_dataset("savedr", data="/res", dtype=str_dt)
                f.create_dataset(
                    "filename",
                    data=np.array([["t1"]], dtype=object),
                    dtype=str_dt,
                )
            tt = TrialTable.from_h5(path)
        self.assertIsNone(tt.slap2_info)
        self.assertIsNone(tt.motion_correction)

    def test_experiment_summary_from_h5(self):
        """ExperimentSummary.from_h5 delegates to the shared band codec."""
        expected = ExperimentSummary()
        with mock.patch(
            "giant_python.io.experiment_summary.read_summary",
            return_value=expected,
        ) as reader:
            result = ExperimentSummary.from_h5("custom_summary.h5")
        self.assertIs(result, expected)
        reader.assert_called_once_with("custom_summary.h5")

    def test_experiment_summary_to_h5(self):
        """ExperimentSummary.to_h5 delegates without rewriting the model."""
        summary = ExperimentSummary()
        with mock.patch(
            "giant_python.io.experiment_summary.write_summary"
        ) as writer:
            self.assertIsNone(summary.to_h5("custom_summary.h5"))
        writer.assert_called_once_with(summary, "custom_summary.h5")


if __name__ == "__main__":
    unittest.main()
