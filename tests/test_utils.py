"""Tests for the utils subpackage."""

import unittest

import numpy as np

from giant_python.utils.interp_frame import interp_frame
from giant_python.utils.scanimagetiff_data_wrapper import (
    scanimagetiff_data_wrapper,
)
from giant_python.utils.scanimagetiff_wrapper import scanimagetiff_wrapper
from giant_python.utils.set_params import set_params
from giant_python.utils.slap2.get_online_motion import get_online_motion
from giant_python.utils.verify_files import verify_files
from giant_python.utils.xcorr2_nans_weighted import xcorr2_nans_weighted


class TestInterpFrame(unittest.TestCase):
    """Tests for interp_frame."""

    def test_not_implemented(self):
        """Test that interp_frame raises NotImplementedError."""
        img = np.zeros((4, 4))
        with self.assertRaises(NotImplementedError):
            interp_frame(img, img, img, img)


class TestScanImageTiffDataWrapper(unittest.TestCase):
    """Tests for scanimagetiff_data_wrapper."""

    def test_not_implemented(self):
        """Test that scanimagetiff_data_wrapper raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            scanimagetiff_data_wrapper(None, "dummy.tif")


class TestScanImageTiffWrapper(unittest.TestCase):
    """Tests for scanimagetiff_wrapper."""

    def test_not_implemented(self):
        """Test that scanimagetiff_wrapper raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            scanimagetiff_wrapper("dummy.tif")


class TestSetParams(unittest.TestCase):
    """Tests for set_params."""

    def test_not_implemented(self):
        """Test that set_params raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            set_params("SELAct")


class TestVerifyFiles(unittest.TestCase):
    """Tests for verify_files."""

    def test_not_implemented(self):
        """Test that verify_files raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            verify_files("trialTable.mat", "/some/dir", {})


class TestXcorr2NansWeighted(unittest.TestCase):
    """Tests for xcorr2_nans_weighted."""

    def test_not_implemented(self):
        """Test that xcorr2_nans_weighted raises NotImplementedError."""
        img = np.zeros((4, 4))
        with self.assertRaises(NotImplementedError):
            xcorr2_nans_weighted(img, img, img, np.zeros(2), 3.0)


class TestGetOnlineMotion(unittest.TestCase):
    """Tests for get_online_motion."""

    def test_not_implemented(self):
        """Test that get_online_motion raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            get_online_motion(None, np.arange(10))


if __name__ == "__main__":
    unittest.main()
