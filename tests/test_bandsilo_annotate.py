"""Tests for band annotation's pure seams and split-policy orchestration."""

import os
import tempfile
import unittest
from copy import deepcopy
from unittest import mock

import numpy as np

from giant_python.extraction.band import annotation as an
from giant_python.io.annotations import save_annotations_h5
from giant_python.models import (
    AnnotationOptions,
    BandSiloParams,
    ExecutionOptions,
    TrialTable,
)


class TestFirstValidTrial(unittest.TestCase):
    """first_valid_trial returns the first kept trial or raises."""

    def test_first_kept(self):
        """Returns the index of the first True entry for the DMD."""
        keep = np.array([[False, True, True], [True, False, False]])
        self.assertEqual(an.first_valid_trial(keep, 0), 1)
        self.assertEqual(an.first_valid_trial(keep, 1), 0)

    def test_none_kept_raises(self):
        """A DMD with no kept trials raises ValueError."""
        keep = np.array([[False, False]])
        with self.assertRaises(ValueError):
            an.first_valid_trial(keep, 0)


class TestMotionMedian(unittest.TestCase):
    """motion_median_from_adata returns per-axis rounded medians."""

    def test_medians(self):
        """Each axis is rounded then reduced by the median."""
        a_data = {
            "motionDSr": np.array([1.2, 1.8, 3.1]),
            "motionDSc": np.array([0.0, 0.0, 4.0]),
            "motionDSz": np.array([5.0, 5.0, 5.0]),
        }
        self.assertEqual(an.motion_median_from_adata(a_data), (2, 0, 5))


class TestResolveInteractivity(unittest.TestCase):
    """resolve_interactivity honors override, env, then TTY fallback."""

    def test_override_wins(self):
        """An explicit override short-circuits detection."""
        self.assertTrue(an.resolve_interactivity(True))
        self.assertFalse(an.resolve_interactivity(False))

    def test_headless_env(self):
        """GIANT_HEADLESS forces headless when no override is given."""
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"GIANT_HEADLESS": "1"}):
            self.assertFalse(an.resolve_interactivity(None))

    def test_tty_fallback(self):
        """With no override/env, falls back to the stdin-TTY probe."""
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(an, "_stdin_is_tty", return_value=True):
                self.assertTrue(an.resolve_interactivity(None))
            with mock.patch.object(an, "_stdin_is_tty", return_value=False):
                self.assertFalse(an.resolve_interactivity(None))


class TestEmptyUserRois(unittest.TestCase):
    """_empty_user_rois builds a neutral per-DMD selection."""

    def test_shape(self):
        """Every per-DMD field is an empty list and annotated is False."""
        out = an._empty_user_rois(2)
        self.assertFalse(out["annotated"])
        for key in ("DMD1", "DMD2"):
            self.assertEqual(out["user_roi_masks"][key], [])
            self.assertEqual(out["user_roi_superpixels"][key], [])
            self.assertEqual(out["user_roi_labels"][key], [])
            self.assertEqual(out["roi_records"][key], [])


class TestResolveUserRois(unittest.TestCase):
    """resolve_user_rois loads an existing file or fails fast when headless."""

    def _geo(self):
        """Return a minimal single-DMD geometry dict."""
        return {
            "DMD1": {
                "num_fast_z": 2,
                "yx_shape": (4, 5),
                "sp_fastz": np.array([0]),
                "sp_rows": np.array([1]),
                "sp_cols": np.array([1]),
            }
        }

    def _write_annotations(self, dr):
        """Write a one-ROI annotations.h5 into ``dr`` and return its path."""
        mask = np.zeros((2, 4, 5), dtype=bool)
        mask[0, 1, 1] = True
        records = {
            "DMD1": [
                {
                    "type": "polygon",
                    "label": "user_roi_1",
                    "position": np.array(
                        [[1, 1], [1, 3], [3, 3], [3, 1]], dtype=np.float64
                    ),
                }
            ]
        }
        return save_annotations_h5(dr, records, {"DMD1": [mask]}, 1)

    def test_loads_existing(self):
        """An existing annotations.h5 is loaded even when non-interactive."""
        with tempfile.TemporaryDirectory() as d:
            self._write_annotations(d)
            out = an.resolve_user_rois(d, 1, self._geo(), interactive=False)
        self.assertTrue(out["annotated"])
        self.assertEqual(out["user_roi_labels"]["DMD1"], ["user_roi_1"])
        self.assertTrue(out["user_roi_masks"]["DMD1"][0][0, 1, 1])
        np.testing.assert_array_equal(
            out["user_roi_superpixels"]["DMD1"][0], np.array([0])
        )

    def test_missing_headless_raises(self):
        """A missing file + headless raises with actionable guidance."""
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError) as ctx:
                an.resolve_user_rois(d, 1, self._geo(), interactive=False)
        self.assertIn("giant annotate", str(ctx.exception))

    def test_missing_interactive_draws(self):
        """A missing file + interactive delegates to the drawing helper."""
        from unittest import mock

        sentinel = {"annotated": True}
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(
                an, "_draw_and_save_user_rois", return_value=sentinel
            ) as draw:
                out = an.resolve_user_rois(
                    d, 1, self._geo(), interactive=True, ref_files={}
                )
        draw.assert_called_once()
        self.assertIs(out, sentinel)


class TestAnnotationService(unittest.TestCase):
    """Annotation validates scoped options before IO without legacy bridges."""

    def test_split_options_and_in_memory_input(self):
        """The explicit step uses GUI/run policy and never mutates callers."""
        input = TrialTable()
        params = BandSiloParams()
        execution = ExecutionOptions(verbose=True)
        annotations = AnnotationOptions(interactive=False)
        before = deepcopy((params, execution, annotations))
        table = {
            "fn_adata": np.array([["a.h5"]]),
            "filename": np.array([["trial"]]),
            "datadr": "/data",
            "annotation_save_dr": "/annotations",
            "moco_save_dr": "/motion",
            "n_dmds": 1,
        }
        with (
            mock.patch.object(
                an, "load_trial_table", return_value=table
            ) as load,
            mock.patch.object(
                an, "compute_keep_trials", return_value=np.array([[True]])
            ),
            mock.patch.object(an.inputs, "load_lookup_table"),
            mock.patch.object(
                an, "build_user_roi_geometry", return_value=({}, {})
            ),
            mock.patch.object(an, "resolve_user_rois") as resolve,
            mock.patch.object(an, "log") as log,
        ):
            result = an.annotate_band_rois(
                input,
                params=params,
                execution=execution,
                annotations=annotations,
            )
        load.assert_called_once_with(input)
        resolve.assert_called_once_with(
            "/annotations", 1, {}, interactive=False, ref_files={}
        )
        self.assertEqual(
            result, os.path.join("/annotations", "annotations.h5")
        )
        self.assertTrue(all(call.args[1] for call in log.call_args_list))
        self.assertEqual((params, execution, annotations), before)
        self.assertFalse(annotations.enabled)

    def test_mixed_options_rejected_before_io(self):
        """Mis-scoped dictionaries and removed keywords cannot load data."""
        with mock.patch.object(an, "load_trial_table") as load:
            for kwargs in (
                {"params": {"max_workers": 2}},
                {"params": {"operator": "Ada"}},
                {"params": {"scan_mode": "band"}},
                {"execution": {"enabled": True}},
                {"annotations": {"draw_user_rois": True}},
                {"params_in": {}},
            ):
                with self.subTest(kwargs=kwargs):
                    with self.assertRaises(TypeError):
                        an.annotate_band_rois("trial.h5", **kwargs)
        load.assert_not_called()
        self.assertFalse(hasattr(an, "_resolve_params"))


if __name__ == "__main__":
    unittest.main()
