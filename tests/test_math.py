"""Tests for the math subpackage."""

import unittest

import numpy as np

from giant_python.math import (
    dft_register_clipped,
    interp_frame,
    xcorr2_nans,
    xcorr2_nans3d,
    xcorr2_nans_weighted,
)
from giant_python.math import (
    compute_f0,
    detect_peaks_2d,
    gaussian_peaks_integrated,
    get_act_im_peaks,
)
from giant_python.math.deconv import deconvolve_trace
from giant_python.math.nmf import nmf_decompose
from giant_python.math.variance import activity_image


class TestRegistration(unittest.TestCase):
    """Tests for registration kernels."""

    def test_xcorr2_nans_weighted(self):
        """xcorr2_nans_weighted raises NotImplementedError."""
        img = np.zeros((4, 4))
        with self.assertRaises(NotImplementedError):
            xcorr2_nans_weighted(img, img, img, np.zeros(2), 3.0)

    def test_xcorr2_nans(self):
        """xcorr2_nans raises NotImplementedError."""
        img = np.zeros((4, 4))
        with self.assertRaises(NotImplementedError):
            xcorr2_nans(img, img)

    def test_xcorr2_nans3d(self):
        """xcorr2_nans3d raises NotImplementedError."""
        vol = np.zeros((2, 4, 4))
        with self.assertRaises(NotImplementedError):
            xcorr2_nans3d(vol, vol)

    def test_dft_register_clipped(self):
        """dft_register_clipped raises NotImplementedError."""
        img = np.zeros((4, 4))
        with self.assertRaises(NotImplementedError):
            dft_register_clipped(img, img)


class TestInterpolation(unittest.TestCase):
    """Tests for interp_frame."""

    def test_interp_frame(self):
        """interp_frame raises NotImplementedError."""
        img = np.zeros((4, 4))
        with self.assertRaises(NotImplementedError):
            interp_frame(img, img, img, img)


class TestExtractionKernels(unittest.TestCase):
    """Tests for nmf/deconv/variance kernels."""

    def test_nmf_decompose(self):
        """nmf_decompose raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            nmf_decompose(np.zeros((16, 10)), 2)

    def test_deconvolve_trace(self):
        """deconvolve_trace raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            deconvolve_trace(np.zeros(10), 0.5, 30.0)

    def test_activity_image(self):
        """activity_image raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            activity_image(np.zeros((10, 4, 4)), 1.0)


class TestPeaks(unittest.TestCase):
    """Tests for the shared peak-detection kernels."""

    def test_gaussian_peaks_integrated(self):
        """gaussian_peaks_integrated raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            gaussian_peaks_integrated(np.zeros((1, 4)), np.zeros((4, 2)))

    def test_detect_peaks_2d(self):
        """detect_peaks_2d raises NotImplementedError."""
        img = np.zeros((8, 8))
        with self.assertRaises(NotImplementedError):
            detect_peaks_2d(img, np.zeros((8, 8), bool), 0.0, 1.0, 3.0, 3.0)

    def test_get_act_im_peaks(self):
        """get_act_im_peaks raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            get_act_im_peaks(np.zeros((2, 8, 8)))


class TestBaseline(unittest.TestCase):
    """Tests for the shared baseline kernel."""

    def test_compute_f0(self):
        """compute_f0 raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            compute_f0(np.zeros((20, 3)), 5, 8)


if __name__ == "__main__":
    unittest.main()
