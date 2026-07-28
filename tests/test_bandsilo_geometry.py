"""Tests for giant_python.bandsilo.geometry."""

import os
import tempfile
import unittest

import h5py
import numpy as np
import tifffile

from giant_python.bandsilo.geometry import (
    _pad_to,
    build_combined_psf,
    build_sparse_h,
    build_subsample_matrix_inds,
    default_psf,
    find_reference_file,
    load_lookup_table,
    load_psf,
    load_reference_stack,
    ref_pixs_to_drc,
    threshold_and_crop_psf,
)


class TestRefPixsToDrc(unittest.TestCase):
    """ref_pixs_to_drc maps flat pixels to (depth, column, row)."""

    def test_known_mapping(self):
        """A hand-computed pixel decomposes to the expected d/c/r."""
        # C (per column) = 4, R (per row) = 5 -> plane = 20.
        # pix = d*plane + c*C + r = 1*20 + 2*4 + 2 = 30
        d, c, r = ref_pixs_to_drc(np.array([30]), 4, 5)
        self.assertEqual((int(d[0]), int(c[0]), int(r[0])), (1, 2, 2))


class TestSubsampleMatrixInds(unittest.TestCase):
    """build_subsample_matrix_inds picks median open pixels."""

    def test_median_reference_pixel(self):
        """Odd-length pixels use the median; even-length use the middle."""
        all_ids = np.array([[1], [2]], dtype=np.int32)
        mask = np.array(
            [[10, 1], [12, 1], [14, 1], [20, 2], [22, 2]], dtype=np.int32
        )
        out = build_subsample_matrix_inds(all_ids, mask)
        np.testing.assert_array_equal(
            out, np.array([[11, 1], [21, 2]], dtype=np.int32)
        )

    def test_odd_length_uses_median_not_middle_index(self):
        """Unsorted odd-length open pixels pick median, not middle by order."""
        all_ids = np.array([[1]], dtype=np.int32)
        # 0-based open pixels: [13, 9, 11] -> median 11, middle index -> 9
        mask = np.array([[14, 1], [10, 1], [12, 1]], dtype=np.int32)
        out = build_subsample_matrix_inds(all_ids, mask)
        np.testing.assert_array_equal(out, np.array([[11, 1]], dtype=np.int32))


class TestSparseH(unittest.TestCase):
    """build_sparse_h assembles the PSF-convolution COO matrix."""

    def test_known_values(self):
        """Indices/values match a hand-computed single-superpixel case."""
        sub = np.array([[30, 1]], dtype=np.int32)  # d=1, c=2, r=2 (C=4,R=5)
        psf = np.array(
            [[0.0, 1.0, 0.0], [2.0, 3.0, 4.0], [0.0, 5.0, 0.0]],
            dtype=np.float32,
        )
        inds, vals = build_sparse_h(sub, psf, 4, 5)
        np.testing.assert_array_equal(inds[0], np.zeros(5, dtype=np.int32))
        np.testing.assert_array_equal(
            inds[1], np.array([27, 31, 32, 33, 37], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            vals, np.array([1, 2, 3, 4, 5], dtype=np.float32)
        )


class TestLookupTable(unittest.TestCase):
    """load_lookup_table reshapes per-DMD lookup arrays."""

    def test_load(self):
        """Superpixel ids/mask/fastZ arrays load in expected shapes."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bandRegLookupTable.h5")
            with h5py.File(path, "w") as f:
                f.create_dataset("row_major", data=1)
                p1 = f.create_group("Path1")
                p1.create_dataset("allSuperPixelIDs", data=np.array([1, 2, 3]))
                p1.create_dataset(
                    "sparseMaskInds",
                    data=np.array([[1, 1], [2, 1], [3, 2]]),
                )
                p1.create_dataset("fastZ2RefZ", data=np.array([1, 2]))
            out = load_lookup_table(path, 1)
        self.assertEqual(out["allSuperPixelIDs"]["DMD1"].shape, (3, 1))
        self.assertEqual(out["sparseMaskInds"]["DMD1"].shape, (3, 2))
        self.assertEqual(out["fastZ2RefZ"]["DMD1"].shape, (2, 1))


class TestReferenceFiles(unittest.TestCase):
    """find_reference_file and load_reference_stack."""

    def test_find_first_pattern_and_missing(self):
        """CONFIG2 pattern is matched; an empty dir returns None."""
        with tempfile.TemporaryDirectory() as d:
            name = "expt_DMD1_CONFIG2-REFERENCE.tif"
            open(os.path.join(d, name), "wb").close()
            self.assertTrue(find_reference_file(d, 0).endswith(name))
        with tempfile.TemporaryDirectory() as empty:
            self.assertIsNone(find_reference_file(empty, 0))

    def test_load_reference_stack(self):
        """A synthetic REFERENCE tif reshapes to [channel, z, y, x]."""
        meta = {"Path1": {"channels": np.array([1, 2])}}
        with tempfile.TemporaryDirectory() as d:
            data = (np.arange(6 * 4 * 4).reshape(6, 4, 4)).astype(np.float32)
            tifffile.imwrite(
                os.path.join(d, "expt_DMD1-REFERENCE.tif"),
                data,
                photometric="minisblack",
            )
            stack, channels, ref_file = load_reference_stack(d, meta, 0)
        self.assertEqual(stack.shape, (2, 3, 4, 4))
        self.assertEqual(list(channels), [1, 2])
        self.assertTrue(ref_file.endswith("expt_DMD1-REFERENCE.tif"))

    def test_load_reference_stack_missing(self):
        """No REFERENCE file yields (None, channels, None)."""
        meta = {"Path1": {"channels": np.array([1, 2])}}
        with tempfile.TemporaryDirectory() as d:
            stack, channels, ref_file = load_reference_stack(d, meta, 0)
        self.assertIsNone(stack)
        self.assertIsNone(ref_file)
        self.assertEqual(list(channels), [1, 2])


class TestPsf(unittest.TestCase):
    """PSF loading, padding, thresholding, cropping."""

    def test_default_psf(self):
        """A bundled PSF template loads as a 2-D float32 array."""
        psf = default_psf(17)
        self.assertEqual(psf.ndim, 2)
        self.assertEqual(psf.dtype, np.float32)

    def test_pad_and_combine(self):
        """build_combined_psf center-pads to common dims."""
        p1 = np.ones((3, 3), dtype=np.float32)
        p2 = np.full((1, 1), 5.0, dtype=np.float32)
        combined = build_combined_psf([p1, p2])
        self.assertEqual(combined.shape, (2, 3, 3))
        padded = _pad_to(p2, 3, 3)
        self.assertEqual(padded.shape, (3, 3))

    def test_threshold_and_crop(self):
        """Sub-threshold pixels are zeroed and boundary zeros cropped."""
        arr = np.zeros((5, 5), dtype=np.float32)
        arr[2, 2] = 10.0
        arr[1, 2] = arr[3, 2] = arr[2, 1] = arr[2, 3] = 1.0
        arr[0, 0] = 0.1  # below 10*exp(-3) ~ 0.498 -> zeroed
        cropped = threshold_and_crop_psf(arr)
        self.assertEqual(cropped.shape, (3, 3))
        self.assertEqual(cropped[1, 1], 10.0)

    def test_load_psf_from_assets(self):
        """load_psf always builds per-DMD PSFs from bundled assets."""
        psf = load_psf(17, 2)
        self.assertEqual(set(psf), {"DMD1", "DMD2"})
        self.assertEqual(psf["DMD1"].ndim, 2)
        self.assertEqual(psf["DMD1"].dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
