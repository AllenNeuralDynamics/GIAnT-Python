"""Tests for the models subpackage."""

import unittest

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


class TestH5RoundTrip(unittest.TestCase):
    """from_h5/to_h5 are not implemented yet."""

    def test_trial_table_from_h5(self):
        """TrialTable.from_h5 raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            TrialTable.from_h5("trial_table.h5")

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
