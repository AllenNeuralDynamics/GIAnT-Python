"""Tests for the pipeline subpackage (stage functions, classes, facade)."""

import unittest

from giant_python.models import AlignParams, SiloParams
from giant_python.pipeline import (
    IntegrationRegistration,
    IntegrationSourceExtractor,
    MotionCorrector,
    MultiRoiRegistration,
    Pipeline,
    SourceExtractor,
    StandardSourceExtractor,
    StripRegistration,
    TrialTableBuilder,
    build_trial_table,
    build_trial_table_slap2,
    extract_integration_sources,
    integration_registration,
    multi_roi_registration,
    sel_act,
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

    def test_integration_registration(self):
        """integration_registration raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            integration_registration()

    def test_sel_act(self):
        """sel_act raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            sel_act()

    def test_extract_integration_sources(self):
        """extract_integration_sources raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            extract_integration_sources()


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
            IntegrationRegistration,
        ):
            with self.assertRaises(NotImplementedError):
                backend().run(None)

    def test_source_extractor(self):
        """SourceExtractor.run raises NotImplementedError."""
        extractor = SourceExtractor(SiloParams())
        with self.assertRaises(NotImplementedError):
            extractor.run(None)

    def test_source_extractor_for_scan_mode(self):
        """SourceExtractor.for_scan_mode raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            SourceExtractor.for_scan_mode("integration", SiloParams())

    def test_extractor_backends_run(self):
        """Each extraction backend's run raises NotImplementedError."""
        for backend in (
            StandardSourceExtractor,
            IntegrationSourceExtractor,
        ):
            with self.assertRaises(NotImplementedError):
                backend().run(None)


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
        """organize delegates to the (unimplemented) builder."""
        with self.assertRaises(NotImplementedError):
            self.pipe.organize("/data")

    def test_register_delegates(self):
        """register delegates to the (unimplemented) corrector."""
        with self.assertRaises(NotImplementedError):
            self.pipe.register()

    def test_extract_delegates(self):
        """extract delegates to the (unimplemented) extractor."""
        with self.assertRaises(NotImplementedError):
            self.pipe.extract()

    def test_run_all(self):
        """run_all raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.pipe.run_all("/data")

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
