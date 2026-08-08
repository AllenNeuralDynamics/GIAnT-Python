"""Tests for the models subpackage."""

import os
import tempfile
import unittest

import h5py
import numpy as np

from giant_python.models import (
    AlignmentData,
    AlignParams,
    ExperimentSummary,
    SiloParams,
    Slap2Info,
    Source,
    TrialTable,
    UserRoi,
    Visualizations,
    set_params,
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

    def test_alignment_data_defaults(self):
        """AlignmentData defaults to a non-failed registration."""
        self.assertFalse(AlignmentData().registration_failed)

    def test_experiment_summary_defaults(self):
        """ExperimentSummary nests a Visualizations instance."""
        summary = ExperimentSummary()
        self.assertIsInstance(summary.visualizations, Visualizations)
        self.assertEqual(summary.sources, [])

    def test_source_and_user_roi(self):
        """Source and UserRoi construct with defaults."""
        self.assertIsNone(Source().snr)
        self.assertEqual(UserRoi().type, "polygon")

    def test_param_models(self):
        """Param models expose typed defaults."""
        self.assertEqual(AlignParams().align_hz, 80.0)
        self.assertEqual(SiloParams().microscope, "slap2")
        self.assertEqual(SiloParams().scan_mode, "standard")

    def test_silo_value_params_have_concrete_defaults(self):
        """Value params default to concrete (non-None) values."""
        p = SiloParams()
        self.assertEqual(p.analyze_hz, 100.0)
        self.assertEqual(p.peakth, 8.0)
        # Only genuinely runtime-resolved fields stay None sentinels.
        self.assertIsNone(p.num_channels)
        self.assertIsNone(p.interactive)


class TestH5RoundTrip(unittest.TestCase):
    """from_h5/to_h5 are not implemented yet."""

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

    def test_trial_table_to_h5(self):
        """TrialTable.to_h5 raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            TrialTable().to_h5("trial_table.h5")

    def test_alignment_data_from_h5(self):
        """AlignmentData.from_h5 raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            AlignmentData.from_h5("a.h5")

    def test_alignment_data_to_h5(self):
        """AlignmentData.to_h5 raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            AlignmentData().to_h5("a.h5")

    def test_experiment_summary_from_h5(self):
        """ExperimentSummary.from_h5 raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            ExperimentSummary.from_h5("experiment_summary.h5")

    def test_experiment_summary_to_h5(self):
        """ExperimentSummary.to_h5 raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            ExperimentSummary().to_h5("experiment_summary.h5")


class TestSetParams(unittest.TestCase):
    """Tests for the set_params compatibility shim."""

    def test_not_implemented(self):
        """set_params raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            set_params("SILo")


if __name__ == "__main__":
    unittest.main()
