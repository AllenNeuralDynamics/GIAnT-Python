"""Contracts for scientific configuration and independent run policies."""

import math
import unittest
from copy import deepcopy

from giant_python.models.params import (
    AnnotationOptions,
    BandSiloParams,
    ExecutionOptions,
    resolve_band_options,
)


class TestBandConfiguration(unittest.TestCase):
    """Resolution preserves numerical defaults and never mutates inputs."""

    def test_defaults_match_existing_science(self):
        """The split is architectural, not a scientific retuning."""
        science, execution, annotations = resolve_band_options()
        self.assertEqual(science.peakth, 7.0)
        self.assertEqual(science.denoise_window_s, 1.0)
        self.assertEqual(science.vif, 1.38)
        self.assertEqual(science.sparse_fac, math.exp(-3.0))
        self.assertEqual(execution, ExecutionOptions(6, False, None))
        self.assertEqual(annotations, AnnotationOptions(False, None))
        self.assertFalse(hasattr(annotations, "operator"))
        self.assertFalse(hasattr(science, "max_workers"))
        self.assertFalse(hasattr(science, "scan_mode"))
        self.assertFalse(hasattr(science, "interactive"))

    def test_science_rejects_mixed_and_routing_fields(self):
        """Science dictionaries never dispatch other scopes or old names."""
        for name, value in (
            ("microscope", "slap2"),
            ("scan_mode", "band"),
            ("max_workers", 2),
            ("max_trials", 3),
            ("verbose", True),
            ("operator", "Ada"),
            ("enabled", True),
            ("interactive", False),
            ("draw_user_rois", True),
            ("sigma_px", 1.5),
            ("lambda_", 0.1),
            ("phi", None),
            ("tau_s", None),
            ("photon_scale", None),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(TypeError, name):
                    resolve_band_options({"peakth": 7.0, name: value})

    def test_every_result_is_fresh(self):
        """Typed inputs and each default result are independent objects."""
        original = (BandSiloParams(), ExecutionOptions(), AnnotationOptions())
        resolved = resolve_band_options(*original)
        for actual, expected in zip(resolved, original):
            self.assertEqual(actual, expected)
            self.assertIsNot(actual, expected)
        self.assertIsNot(resolve_band_options()[0], resolved[0])

    def test_dictionary_inputs_and_overrides_are_not_modified(self):
        """Each dictionary affects only its scope and is copied."""
        params = {"analyze_hz": 80.0}
        execution = {"max_workers": 2, "verbose": True}
        annotations = {
            "enabled": True,
            "interactive": False,
        }
        original = deepcopy((params, execution, annotations))
        science, run, roi = resolve_band_options(
            params, execution, annotations
        )
        self.assertEqual(science.analyze_hz, 80.0)
        self.assertEqual(run, ExecutionOptions(2, True, None))
        self.assertEqual(roi, AnnotationOptions(True, False))
        self.assertEqual((params, execution, annotations), original)


class TestConfigurationValidation(unittest.TestCase):
    """Reject unsupported settings before they reach numerical kernels."""

    def test_positive_scientific_values(self):
        """Rates, windows and positive scale values must be finite."""
        for name in (
            "analyze_hz",
            "decay_tau_s",
            "baseline_window_s",
            "denoise_window_s",
            "vif",
            "sparse_fac",
            "peakth",
        ):
            for value in (0, -1, math.nan, math.inf, True, "1", None):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        BandSiloParams(**{name: value})

    def test_positive_counts(self):
        """Spatial counts require positive integers, not floats or bools."""
        for name in ("d_xy", "peak_buffer", "psf_dilation"):
            for value in (0, -1, 1.5, True):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        BandSiloParams(**{name: value})

    def test_background_interpolation_is_not_configurable(self):
        """Neither typed nor dictionary options accept interpolation modes."""
        for mode in ("linear", "cubic", "nearest"):
            with self.subTest(mode=mode):
                options = {"background_interpolation": mode}
                with self.assertRaisesRegex(
                    TypeError, "background_interpolation"
                ):
                    BandSiloParams(**options)
                with self.assertRaisesRegex(
                    TypeError, "background_interpolation"
                ):
                    resolve_band_options(options)

    def test_channel_parameters_are_not_configurable(self):
        """Removed channel options fail for both typed and dictionary inputs."""
        for name, value in (
            ("num_channels", None),
            ("num_channels", 2),
            ("activity_channel", 0),
            ("activity_channel", 1),
        ):
            with self.subTest(name=name, value=value):
                options = {name: value}
                with self.assertRaisesRegex(TypeError, name):
                    BandSiloParams(**options)
                with self.assertRaisesRegex(TypeError, name):
                    resolve_band_options(options)

    def test_operator_is_not_configurable(self):
        """Removed operator metadata is rejected rather than ignored."""
        for value in ("Ada", "SLAP2 User", None):
            with self.subTest(value=value):
                options = {"operator": value}
                with self.assertRaisesRegex(TypeError, "operator"):
                    AnnotationOptions(**options)
                with self.assertRaisesRegex(TypeError, "operator"):
                    resolve_band_options(annotations=options)

    def test_execution_and_annotation_policy(self):
        """Execution counts and tri-state GUI settings are validated."""
        for name in ("max_workers", "max_trials"):
            for value in (0, -1, 1.5, True):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        ExecutionOptions(**{name: value})
        with self.assertRaisesRegex(ValueError, "verbose"):
            ExecutionOptions(verbose=1)
        for name, value in (
            ("enabled", 1),
            ("interactive", "false"),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, name):
                    AnnotationOptions(**{name: value})

    def test_resolver_revalidates_mutated_carriers(self):
        """Dataclass mutability cannot bypass validation at resolution."""
        science = BandSiloParams()
        science.d_xy = 0
        with self.assertRaisesRegex(ValueError, "d_xy"):
            resolve_band_options(science)
        execution = ExecutionOptions()
        execution.max_workers = 0
        with self.assertRaisesRegex(ValueError, "max_workers"):
            resolve_band_options(execution=execution)
        annotations = AnnotationOptions()
        annotations.interactive = "false"
        with self.assertRaisesRegex(ValueError, "interactive"):
            resolve_band_options(annotations=annotations)

    def test_bad_input_and_unknown_keys(self):
        """Typos and wrong container types do not silently use defaults."""
        for params in (42, {"analyze_hzz": 80}):
            with self.assertRaises(TypeError):
                resolve_band_options(params)
        for keyword, value in (
            ("execution", False),
            ("annotations", "headless"),
            ("execution", {"max_worker": 2}),
            ("annotations", {"draw_user_rois": True}),
            ("execution", {"operator": "Ada"}),
            ("execution", {"peakth": 7}),
            ("annotations", {"max_workers": 2}),
            ("annotations", {"scan_mode": "band"}),
        ):
            with self.subTest(keyword=keyword, value=value):
                with self.assertRaises(TypeError):
                    resolve_band_options(**{keyword: value})
        with self.assertRaises(TypeError):
            resolve_band_options({"enabled": True, "draw_user_rois": False})


if __name__ == "__main__":
    unittest.main()
