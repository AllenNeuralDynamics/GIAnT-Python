"""Tests for the motion_correction subpackage."""

import unittest

from giant_python.motion_correction.multi_roi_registration import (
    multi_roi_registration,
)
from giant_python.motion_correction.strip_registration import (
    strip_registration,
)


class TestMultiRoiRegistration(unittest.TestCase):
    """Tests for multi_roi_registration."""

    def test_not_implemented(self):
        """Test that multi_roi_registration raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            multi_roi_registration()


class TestStripRegistration(unittest.TestCase):
    """Tests for strip_registration."""

    def test_not_implemented(self):
        """Test that strip_registration raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            strip_registration()


if __name__ == "__main__":
    unittest.main()
