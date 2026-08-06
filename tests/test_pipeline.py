"""Tests for the pipeline subpackage (stage functions, classes, facade)."""

import unittest
from unittest import mock

from giant_python.models import AlignParams, SiloParams
from giant_python.pipeline import (
    BandRegistration,
    BandSourceExtractor,
    MotionCorrector,
    MultiRoiRegistration,
    Pipeline,
    SourceExtractor,
    StandardSourceExtractor,
    StripRegistration,
    TrialTableBuilder,
    band_registration,
    build_trial_table,
    build_trial_table_slap2,
    multi_roi_registration,
    silo,
    strip_registration,
    verify_files,
)


class TestStageFunctions(unittest.TestCase):
    """The porting-faithful stage functions raise NotImplementedError."""

    def test_build_trial_table(self):
        """build_trial_table raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            build_trial_table()

    def test_build_trial_table_slap2(self):
        """build_trial_table_slap2 raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            build_trial_table_slap2()

    def test_verify_files(self):
        """verify_files raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            verify_files("trial_table.h5", "/some/dir", {})

    def test_multi_roi_registration(self):
        """multi_roi_registration raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            multi_roi_registration()

    def test_strip_registration(self):
        """strip_registration raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            strip_registration()

    def test_band_registration(self):
        """band_registration raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            band_registration()

    def test_silo(self):
        """silo raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            silo()


class TestStageClasses(unittest.TestCase):
    """Stage classes construct; their run methods raise NotImplementedError."""

    def test_trial_table_builder(self):
        """TrialTableBuilder.run raises NotImplementedError."""
        builder = TrialTableBuilder("slap2")
        with self.assertRaises(NotImplementedError):
            builder.run("/data", "/results")

    def test_motion_corrector_for_microscope(self):
        """MotionCorrector.for_microscope raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            MotionCorrector.for_microscope("slap2", AlignParams())

    def test_motion_corrector_run(self):
        """MotionCorrector.run raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            MotionCorrector().run(None)

    def test_registration_backends_run(self):
        """Each registration backend's run raises NotImplementedError."""
        for backend in (
            MultiRoiRegistration,
            StripRegistration,
            BandRegistration,
        ):
            with self.assertRaises(NotImplementedError):
                backend().run(None)

    def test_source_extractor(self):
        """SourceExtractor.run raises NotImplementedError."""
        extractor = SourceExtractor(SiloParams())
        with self.assertRaises(NotImplementedError):
            extractor.run(None)

    def test_source_extractor_for_scan_mode(self):
        """for_scan_mode dispatches by scan mode and rejects unknown modes."""
        self.assertIsInstance(
            SourceExtractor.for_scan_mode("band", SiloParams()),
            BandSourceExtractor,
        )
        self.assertIsInstance(
            SourceExtractor.for_scan_mode("standard", SiloParams()),
            StandardSourceExtractor,
        )
        with self.assertRaises(ValueError):
            SourceExtractor.for_scan_mode("bogus", SiloParams())

    def test_standard_backend_run_not_implemented(self):
        """The standard backend's run is still a stub."""
        with self.assertRaises(NotImplementedError):
            StandardSourceExtractor().run(None)

    def test_band_backend_run_delegates(self):
        """The band backend resolves the trial-table path and delegates.

        The heavy driver is patched so only the wiring is exercised.
        """
        import giant_python.bandsilo.pipeline as bpl
        from giant_python.models import TrialTable

        captured = {}

        def _fake(path, params_in=None):
            """Capture the delegated call and return a sentinel."""
            captured["path"] = path
            captured["params"] = params_in
            return "SUMMARY"

        orig = bpl.extract_band_sources
        bpl.extract_band_sources = _fake
        try:
            params = SiloParams(scan_mode="band")
            extractor = BandSourceExtractor(params)
            out = extractor.run(TrialTable(savedr="/results"))
        finally:
            bpl.extract_band_sources = orig

        self.assertEqual(out, "SUMMARY")
        self.assertEqual(captured["path"].name, "trial_table.h5")
        self.assertIs(captured["params"], params)


class TestPipelineFacade(unittest.TestCase):
    """Tests for the Pipeline facade."""

    def setUp(self):
        """Construct a pipeline for each test."""
        self.pipe = Pipeline(microscope="slap2", save_dir="/results")

    def test_holds_session_state(self):
        """The facade holds config and starts with no data."""
        self.assertEqual(self.pipe.microscope, "slap2")
        self.assertIsNone(self.pipe.tt)
        self.assertIsNone(self.pipe.summary)
        self.assertEqual(self.pipe.silo_params.microscope, "slap2")

    def test_organize_delegates(self):
        """organize delegates to the builder and stores its trial table."""
        with mock.patch(
            "giant_python.pipeline.pipeline.TrialTableBuilder"
        ) as builder:
            builder.return_value.run.return_value = "TT"
            result = self.pipe.organize("/data")
        self.assertIs(result, self.pipe)
        self.assertEqual(self.pipe.tt, "TT")

    def test_register_delegates(self):
        """register delegates to the motion corrector and updates the table."""
        with mock.patch(
            "giant_python.pipeline.pipeline.MotionCorrector"
        ) as corrector:
            corrector.for_microscope.return_value.run.return_value = "TT2"
            self.pipe.tt = "TT"
            result = self.pipe.register()
        self.assertIs(result, self.pipe)
        self.assertEqual(self.pipe.tt, "TT2")

    def test_annotate_delegates(self):
        """A non-band scan mode uses the generic annotation GUI."""
        self.pipe.silo_params.scan_mode = "standard"
        with mock.patch("giant_python.gui.annotate_rois") as annotate_rois:
            result = self.pipe.annotate()
        annotate_rois.assert_called_once()
        self.assertIs(result, self.pipe)

    def test_annotate_band_delegates(self):
        """A band scan mode uses the BandSILo standalone annotation step."""
        self.pipe.silo_params.scan_mode = "band"
        self.pipe.tt = mock.Mock(savedr="/results")
        with mock.patch(
            "giant_python.bandsilo.annotate.annotate_band_rois"
        ) as annotate_band:
            result = self.pipe.annotate()
        annotate_band.assert_called_once()
        called_path = annotate_band.call_args[0][0]
        self.assertEqual(called_path.name, "trial_table.h5")
        self.assertIs(result, self.pipe)

    def test_extract_delegates(self):
        """extract dispatches by scan mode and stores the summary."""
        self.pipe.silo_params.scan_mode = "band"
        with mock.patch(
            "giant_python.pipeline.pipeline.SourceExtractor"
        ) as extractor:
            extractor.for_scan_mode.return_value.run.return_value = "SUMMARY"
            result = self.pipe.extract()
        self.assertIs(result, self.pipe)
        self.assertEqual(self.pipe.summary, "SUMMARY")

    def test_run_all_sequences_stages(self):
        """run_all sequences organize -> register -> annotate -> extract."""
        with mock.patch.object(
            self.pipe, "organize", return_value=self.pipe
        ) as organize, mock.patch.object(
            self.pipe, "register", return_value=self.pipe
        ) as register, mock.patch.object(
            self.pipe, "annotate", return_value=self.pipe
        ) as annotate, mock.patch.object(
            self.pipe, "extract", return_value=self.pipe
        ) as extract:
            result = self.pipe.run_all("/data", annotate=True)
        organize.assert_called_once_with("/data")
        register.assert_called_once()
        annotate.assert_called_once()
        extract.assert_called_once()
        self.assertIs(result, self.pipe)

    def test_run_all_skips_annotation(self):
        """run_all skips annotation when annotate is False."""
        with mock.patch.object(
            self.pipe, "organize", return_value=self.pipe
        ), mock.patch.object(
            self.pipe, "register", return_value=self.pipe
        ), mock.patch.object(
            self.pipe, "annotate", return_value=self.pipe
        ) as annotate, mock.patch.object(
            self.pipe, "extract", return_value=self.pipe
        ):
            self.pipe.run_all("/data")
        annotate.assert_not_called()

    def test_from_trial_table(self):
        """from_trial_table raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            Pipeline.from_trial_table("trial_table.h5", "slap2")

    def test_from_config(self):
        """from_config raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            Pipeline.from_config("config.yaml")


if __name__ == "__main__":
    unittest.main()
