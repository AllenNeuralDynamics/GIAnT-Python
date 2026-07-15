"""Tests for the gui subpackage."""

import unittest

from giant_python.gui import DrawROIs, annotate_rois


class TestAnnotateRois(unittest.TestCase):
    """Tests for the ROI annotation entry point."""

    def test_annotate_rois(self):
        """annotate_rois raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            annotate_rois(None)


class TestDrawROIs(unittest.TestCase):
    """Tests for the DrawROIs class."""

    def test_show(self):
        """DrawROIs.show raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            DrawROIs(None).show()

    def test_save_rois(self):
        """DrawROIs.save_rois raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            DrawROIs(None).save_rois("annotations.h5")


if __name__ == "__main__":
    unittest.main()
