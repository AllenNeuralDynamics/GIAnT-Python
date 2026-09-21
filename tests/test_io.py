"""Tests for the io subpackage."""

import os
import tempfile
import unittest

import h5py
import numpy as np

from giant_python import io
from giant_python.extraction.band import geometry, trial_data
from giant_python.io import load_struct_h5, slap2
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
    """Tests for the generic HDF5 struct reader."""

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


class TestSlap2(unittest.TestCase):
    """Tests for SLAP2 reader and band reduction ownership."""

    def test_ref_pixs_to_drc(self):
        """Reference-pixel conversion belongs to band geometry, not IO."""
        self.assertTrue(callable(geometry.ref_pixs_to_drc))
        self.assertEqual(
            geometry.ref_pixs_to_drc.__module__, geometry.__name__
        )
        for module in (io, slap2):
            self.assertFalse(hasattr(module, "ref_pixs_to_drc"))
        self.assertNotIn("ref_pixs_to_drc", io.__all__)

    def test_read_band_trial_data(self):
        """Band reduction remains canonical without an IO compatibility API."""
        self.assertTrue(callable(trial_data.read_band_trial_data))
        self.assertEqual(
            trial_data.read_band_trial_data.__module__, trial_data.__name__
        )
        for module in (io, slap2):
            self.assertFalse(hasattr(module, "read_band_trial_data"))
        self.assertNotIn("read_band_trial_data", io.__all__)


if __name__ == "__main__":
    unittest.main()
