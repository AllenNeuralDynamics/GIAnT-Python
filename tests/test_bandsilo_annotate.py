"""Tests for giant_python.bandsilo.annotate (pure seams + load/fail-fast)."""

import tempfile
import unittest

import numpy as np

from giant_python.bandsilo import annotate as an
from giant_python.bandsilo.gui import save_annotations_h5


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


if __name__ == "__main__":
    unittest.main()
