"""Tests for the implemented annotation and extraction pipeline facade."""

import unittest
from pathlib import Path
from unittest import mock

from giant_python.models import BandSiloParams, ExperimentSummary, TrialTable
from giant_python.pipeline import Pipeline


class TestPipelineFacade(unittest.TestCase):
    """Tests for the Pipeline facade."""

    def setUp(self):
        """Construct a pipeline for each test."""
        self.pipe = Pipeline(microscope="slap2")

    def test_holds_session_state(self):
        """The facade holds config and starts with no data."""
        self.assertEqual(self.pipe.microscope, "slap2")
        self.assertIsNone(self.pipe.tt)
        self.assertIsNone(self.pipe.summary)
        self.assertIsInstance(self.pipe.params, BandSiloParams)
        self.assertEqual(self.pipe.scan_mode, "band")
        self.assertFalse(hasattr(self.pipe, "silo_params"))

    def test_params_keyword(self):
        """Only the canonical scientific keyword is accepted."""
        params = BandSiloParams()
        self.assertIs(Pipeline(params=params).params, params)
        with self.assertRaises(TypeError):
            Pipeline(silo_params=params)

    def test_annotate_delegates(self):
        """Explicit standard annotation fails instead of pretending to work."""
        self.pipe.scan_mode = "standard"
        self.pipe.tt = TrialTable()
        with self.assertRaisesRegex(ValueError, "standard"):
            self.pipe.annotate()

    def test_annotate_band_delegates(self):
        """A band scan mode uses the BandSILo standalone annotation step."""
        self.pipe.tt = TrialTable(savedr=Path("/results"))
        with mock.patch(
            "giant_python.pipeline.pipeline.annotate_rois",
            return_value="annotations.h5",
        ) as annotate_band:
            result = self.pipe.annotate()
        annotate_band.assert_called_once()
        self.assertIs(annotate_band.call_args[0][0], self.pipe.tt)
        self.assertTrue(self.pipe.annotations.enabled)
        self.assertIs(result, self.pipe)

    def test_extract_delegates(self):
        """extract dispatches by scan mode and stores the summary."""
        self.pipe.tt = TrialTable()
        summary = ExperimentSummary()
        with mock.patch(
            "giant_python.pipeline.pipeline.extract_sources",
            return_value=summary,
        ) as extractor:
            result = self.pipe.extract()
        self.assertIs(result, self.pipe)
        self.assertIs(self.pipe.summary, summary)
        self.assertIs(extractor.call_args[0][0], self.pipe.tt)

    def test_from_trial_table(self):
        """from_trial_table loads once and retains the provided filename."""
        table = TrialTable(savedr=Path("/results"))
        path = Path("/data/nonstandard_name.h5")
        with mock.patch.object(
            TrialTable, "from_h5", return_value=table
        ) as load:
            pipeline = Pipeline.from_trial_table(path)
        load.assert_called_once_with(path)
        self.assertIs(pipeline.tt, table)
        self.assertEqual(pipeline.trial_table_path, path)
        self.assertEqual(table.source_path, path)


if __name__ == "__main__":
    unittest.main()
