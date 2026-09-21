"""Mocked service contracts: path identity, dispatch and API convergence."""

import sys
import unittest
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from types import ModuleType
from unittest import mock

import giant_python
from giant_python import cli
from giant_python.models import (
    AnnotationOptions,
    BandSiloParams,
    ExecutionOptions,
    ExperimentSummary,
    TrialTable,
)
from giant_python.pipeline import (
    Pipeline,
    annotate_rois,
    extract_band_sources,
    extract_sources,
)


class TestPublicExports(unittest.TestCase):
    """Lazy exports resolve to canonical objects and advertise real APIs."""

    def test_lazy_exports_match_their_owner(self):
        """Every advertised model and workflow resolves and is cached."""
        for name in giant_python.__all__:
            with self.subTest(name=name):
                owner = (
                    "giant_python.models"
                    if name in giant_python._MODEL_EXPORTS
                    else "giant_python.pipeline"
                )
                value = giant_python.__getattr__(name)
                self.assertIs(value, getattr(import_module(owner), name))
                self.assertIs(giant_python.__dict__[name], value)
                self.assertIn(name, dir(giant_python))

    def test_unknown_export_has_standard_attribute_error(self):
        """Unknown names raise AttributeError without importing a backend."""
        with self.assertRaisesRegex(AttributeError, "no attribute 'missing'"):
            giant_python.__getattr__("missing")

    def test_legacy_apis_are_absent(self):
        """Removed carriers and adapter classes have no reexport shims."""
        names = {
            "giant_python.models.params": (
                "SiloParams",
                "set_params",
                "to_legacy_silo_params",
            ),
            "giant_python.models": (
                "SiloParams",
                "set_params",
                "to_legacy_silo_params",
            ),
            "giant_python.pipeline.extract": (
                "SourceExtractor",
                "BandSourceExtractor",
                "StandardSourceExtractor",
            ),
            "giant_python.pipeline": (
                "SourceExtractor",
                "BandSourceExtractor",
                "StandardSourceExtractor",
            ),
            "giant_python": (
                "SiloParams",
                "SourceExtractor",
                "BandSourceExtractor",
                "StandardSourceExtractor",
            ),
        }
        for module_name, removed in names.items():
            module = import_module(module_name)
            for name in removed:
                with self.subTest(module=module_name, name=name):
                    self.assertFalse(hasattr(module, name))


class TestSharedServices(unittest.TestCase):
    """All supported entry points converge without importing real backends."""

    def setUp(self):
        """Inject only the backend modules; keep public delegation real."""
        self.summary = ExperimentSummary()
        self.extract = mock.Mock(return_value=self.summary)
        self.annotate = mock.Mock(return_value="annotations.h5")
        workflow = ModuleType("giant_python.extraction.band.workflow")
        workflow.extract_band_sources = self.extract
        annotation = ModuleType("giant_python.extraction.band.annotation")
        annotation.annotate_band_rois = self.annotate
        patch = mock.patch.dict(
            sys.modules,
            {
                workflow.__name__: workflow,
                annotation.__name__: annotation,
            },
        )
        patch.start()
        self.addCleanup(patch.stop)

    def test_direct_function_forwards_original_inputs(self):
        """Scientific/run options are resolved once by the backend service."""
        table = TrialTable()
        params = BandSiloParams()
        execution = ExecutionOptions(max_workers=2)
        annotations = AnnotationOptions(interactive=False)
        result = extract_band_sources(
            table, params, execution=execution, annotations=annotations
        )
        self.assertIs(result, self.summary)
        args, kwargs = self.extract.call_args
        self.assertIs(args[0], table)
        self.assertIs(args[1], params)
        self.assertIs(kwargs["execution"], execution)
        self.assertIs(kwargs["annotations"], annotations)

    def test_only_canonical_params_keyword(self):
        """All public services accept params, not params_in."""
        for function in (extract_band_sources, extract_sources, annotate_rois):
            with self.subTest(function=function.__name__):
                function("custom.h5", params=BandSiloParams())
                with self.assertRaises(TypeError):
                    function("custom.h5", params_in={})

    def test_generic_dispatch(self):
        """No params, band params and explicit band all select band."""
        for params, mode in (
            (None, None),
            (BandSiloParams(), None),
            ({}, None),
            (BandSiloParams(), "band"),
        ):
            with self.subTest(params=params, mode=mode):
                self.assertIs(
                    extract_sources("custom.h5", params, scan_mode=mode),
                    self.summary,
                )
        self.assertEqual(self.extract.call_count, 4)

    def test_explicit_unsupported_modes_are_never_ignored(self):
        """Explicit routing is checked before backend invocation."""
        cases = (
            {"params": BandSiloParams(), "scan_mode": "standard"},
            {"scan_mode": "unknown"},
            {"microscope": "bergamo", "scan_mode": "band"},
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "supported"):
                    extract_sources("custom.h5", **kwargs)
                with self.assertRaisesRegex(ValueError, "supported"):
                    annotate_rois("custom.h5", **kwargs)
        self.extract.assert_not_called()
        self.annotate.assert_not_called()

    def test_missing_input_is_actionable(self):
        """An omitted table fails before a heavy backend is invoked."""
        for function in (extract_band_sources, extract_sources, annotate_rois):
            with self.subTest(function=function.__name__):
                with self.assertRaisesRegex(ValueError, "trial-table"):
                    function(None)
        self.extract.assert_not_called()
        self.annotate.assert_not_called()

    def test_annotation_forwards_options_without_mutation(self):
        """Annotation forwards split options directly to the backend."""
        table = TrialTable()
        params = BandSiloParams()
        options = AnnotationOptions(interactive=False, operator="Ada")
        before = deepcopy((params, options))
        result = annotate_rois(table, params, annotations=options)
        self.assertEqual(result, "annotations.h5")
        input, science = self.annotate.call_args.args
        self.assertIs(input, table)
        self.assertIs(science, params)
        self.assertIs(self.annotate.call_args.kwargs["annotations"], options)
        self.assertIsNone(self.annotate.call_args.kwargs["execution"])
        self.assertEqual((params, options), before)

    def test_pipeline_loads_nonstandard_filename_only_once(self):
        """A table outside savedr is preserved through annotate/extract."""
        table = TrialTable(savedr=Path("/different/results"))
        path = Path("/metadata/session_curated_v2.h5")
        with mock.patch.object(
            TrialTable, "from_h5", return_value=table
        ) as load:
            pipeline = Pipeline.from_trial_table(path)
            result = pipeline.annotate().extract()
        load.assert_called_once_with(path)
        self.assertIs(result, pipeline)
        self.assertIs(pipeline.summary, self.summary)
        self.assertEqual(pipeline.trial_table_path, path)
        self.assertEqual(table.source_path, path)
        self.assertIs(self.annotate.call_args.args[0], table)
        self.assertIs(self.extract.call_args.args[0], table)
        self.assertTrue(self.extract.call_args.kwargs["annotations"].enabled)

    def test_in_memory_pipeline_never_guesses_source_path(self):
        """savedr does not imply a trial_table.h5 exists in that directory."""
        for savedr in (None, Path("/results")):
            with self.subTest(savedr=savedr):
                table = TrialTable(savedr=savedr)
                with mock.patch.object(TrialTable, "from_h5") as load:
                    pipeline = Pipeline.from_trial_table(table)
                    pipeline.extract()
                load.assert_not_called()
                self.assertIs(pipeline.tt, table)
                self.assertIsNone(pipeline.trial_table_path)
                self.assertIsNone(table.source_path)
                self.assertIs(self.extract.call_args.args[0], table)

    def test_in_memory_known_source_path_is_retained(self):
        """Metadata already loaded by IO retains its actual source path."""
        path = Path("/metadata/curated.h5")
        table = TrialTable(source_path=path)
        pipeline = Pipeline.from_trial_table(table)
        self.assertIs(pipeline.tt, table)
        self.assertEqual(pipeline.trial_table_path, path)
        self.assertEqual(table.source_path, path)

    def test_annotation_chain_copies_policy(self):
        """Explicit annotation enables later ROI use, not caller options."""
        options = AnnotationOptions(interactive=False)
        pipeline = Pipeline.from_trial_table(TrialTable(), annotations=options)
        pipeline.annotate().extract()
        self.assertFalse(options.enabled)
        self.assertTrue(pipeline.annotations.enabled)
        self.assertIsNot(pipeline.annotations, options)

    def test_top_level_and_cli_share_backend_service(self):
        """Direct, top-level, Pipeline and CLI paths reach the same driver."""
        self.assertIs(giant_python.extract_sources, extract_sources)
        self.assertIs(giant_python.annotate_rois, annotate_rois)
        result = giant_python.extract_sources("curated.h5")
        self.assertIs(result, self.summary)
        self.assertEqual(
            cli.main(
                [
                    "extract",
                    "curated.h5",
                    "--headless",
                    "--max-workers",
                    "2",
                ]
            ),
            0,
        )
        args, kwargs = self.extract.call_args
        self.assertEqual(args[0], "curated.h5")
        self.assertIsInstance(args[1], BandSiloParams)
        self.assertEqual(kwargs["execution"].max_workers, 2)
        self.assertFalse(kwargs["annotations"].interactive)
        self.assertEqual(cli.main(["annotate", "curated.h5", "--headless"]), 0)
        self.assertEqual(self.annotate.call_args.args[0], "curated.h5")
        self.assertFalse(
            self.annotate.call_args.kwargs["annotations"].interactive
        )


class TestPipelineErrors(unittest.TestCase):
    """Unsupported or uninitialized pipeline use fails before side effects."""

    def test_mixed_science_rejected_before_loading(self):
        """Real services reject policy/routing fields rather than dispatch."""
        from giant_python.extraction.band import annotation, workflow

        with (
            mock.patch.object(workflow, "load_trial_table") as extract_load,
            mock.patch.object(annotation, "load_trial_table") as roi_load,
        ):
            for field, value in (
                ("max_workers", 2),
                ("operator", "Ada"),
                ("draw_user_rois", True),
                ("scan_mode", "band"),
                ("microscope", "slap2"),
            ):
                params = {"peakth": 7.0, field: value}
                for function in (
                    extract_band_sources,
                    extract_sources,
                    annotate_rois,
                ):
                    with self.subTest(field=field, api=function.__name__):
                        with self.assertRaisesRegex(TypeError, field):
                            function("curated.h5", params=params)
                pipeline = Pipeline.from_trial_table(
                    TrialTable(), params=params
                )
                for stage in (pipeline.extract, pipeline.annotate):
                    with self.subTest(field=field, stage=stage.__name__):
                        with self.assertRaisesRegex(TypeError, field):
                            stage()
        extract_load.assert_not_called()
        roi_load.assert_not_called()

    def test_missing_table(self):
        """No table is a clear error, not an AttributeError on savedr."""
        pipeline = Pipeline()
        for name in ("annotate", "extract"):
            with self.subTest(stage=name):
                with self.assertRaisesRegex(ValueError, "No trial table"):
                    getattr(pipeline, name)()

    def test_invalid_table_input(self):
        """Invalid input types cannot be mistaken for paths."""
        with self.assertRaisesRegex(TypeError, "TrialTable"):
            Pipeline.from_trial_table(None)


if __name__ == "__main__":
    unittest.main()
