"""Tests for the io subpackage."""

import os
import tempfile
import unittest

import h5py
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
from giant_python.io.hdf5 import _decode_one, _decode_strings


class TestDecodeHelpers(unittest.TestCase):
    """_decode_one / _decode_strings handle bytes, str, and empty arrays."""

    def test_decode_one(self):
        """bytes decode via utf-8; non-bytes pass through str()."""
        self.assertEqual(_decode_one(b"ab"), "ab")
        self.assertEqual(_decode_one("cd"), "cd")

    def test_decode_empty_array(self):
        """An empty array is returned unchanged (no vectorize crash)."""
        out = _decode_strings(np.array([], dtype=object))
        self.assertEqual(out.size, 0)


class TestHdf5(unittest.TestCase):
    """Tests for the generic struct <-> HDF5 helpers."""

    def test_load_struct_h5_faithful(self):
        """load_struct_h5 mirrors groups, decodes strings, honors row_major."""
        str_dt = h5py.string_dtype(encoding="utf-8")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.h5")
            with h5py.File(path, "w") as f:
                # Column-major (row_major absent) -> 2-D axes transposed.
                f.create_dataset("name", data="hello", dtype=str_dt)
                f.create_dataset("mat", data=np.array([[1, 2, 3], [4, 5, 6]]))
                grp = f.create_group("inner")
                grp.create_dataset(
                    "labels",
                    data=np.array([["a", "b"]], dtype=object),
                    dtype=str_dt,
                )
            out = load_struct_h5(path)
        self.assertEqual(out["name"], "hello")
        # (2, 3) written column-major reads back transposed to (3, 2).
        np.testing.assert_array_equal(
            out["mat"], np.array([[1, 4], [2, 5], [3, 6]])
        )
        np.testing.assert_array_equal(
            out["inner"]["labels"], np.array([["a"], ["b"]], dtype=object)
        )

    def test_load_struct_h5_row_major(self):
        """A row_major file keeps its on-disk orientation."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.h5")
            with h5py.File(path, "w") as f:
                f.create_dataset("row_major", data=1)
                f.create_dataset("mat", data=np.array([[1, 2, 3]]))
            out = load_struct_h5(path)
        np.testing.assert_array_equal(out["mat"], np.array([[1, 2, 3]]))

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
