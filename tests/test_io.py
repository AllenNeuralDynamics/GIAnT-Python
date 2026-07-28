"""Tests for the io subpackage."""

import unittest

import numpy as np

from giant_python.io import (
    get_online_motion,
    load_struct_h5,
    read_band_trial_data,
    ref_pixs_to_drc,
    save_struct_h5,
    scanimagetiff_data_wrapper,
    scanimagetiff_wrapper,
)


class TestHdf5(unittest.TestCase):
    """Tests for the generic struct <-> HDF5 helpers."""

    def test_load_struct_h5(self):
        """load_struct_h5 raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            load_struct_h5("dummy.h5")

    def test_save_struct_h5(self):
        """save_struct_h5 raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            save_struct_h5({}, "dummy.h5")


class TestTiff(unittest.TestCase):
    """Tests for the ScanImage TIFF readers."""

    def test_scanimagetiff_wrapper(self):
        """scanimagetiff_wrapper raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            scanimagetiff_wrapper("dummy.tif")

    def test_scanimagetiff_data_wrapper(self):
        """scanimagetiff_data_wrapper raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            scanimagetiff_data_wrapper(None, "dummy.tif")


class TestSlap2(unittest.TestCase):
    """Tests for SLAP2 readers (online motion + band)."""

    def test_get_online_motion(self):
        """get_online_motion raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            get_online_motion(None, np.arange(10))

    def test_ref_pixs_to_drc(self):
        """ref_pixs_to_drc raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            ref_pixs_to_drc(np.arange(10), 4, 4)

    def test_read_band_trial_data(self):
        """read_band_trial_data raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            read_band_trial_data(None, np.arange(5), np.arange(3))


if __name__ == "__main__":
    unittest.main()
