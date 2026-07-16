"""Tests for giant_python.bandsilo.gui (pure + annotation-IO paths)."""

import os
import tempfile
import unittest

import h5py
import numpy as np

from giant_python.bandsilo.gui import (
    _h5_scalar_str,
    _read_roi_records,
    compute_user_roi_geometry,
    load_annotations_h5,
    save_annotations_h5,
    user_roi_superpixel_lists_from_masks,
)


def _str_dt():
    """Return the h5py variable-length utf-8 string dtype."""
    return h5py.string_dtype(encoding="utf-8")


class TestSuperpixelLists(unittest.TestCase):
    """user_roi_superpixel_lists_from_masks selects per-ROI superpixels."""

    def test_overlapping_rois(self):
        """A superpixel inside two ROIs appears in both lists."""
        sp_fastz = np.array([0, 0, 1])
        sp_rows = np.array([1, 2, 0])
        sp_cols = np.array([1, 2, 0])
        mask_a = np.zeros((2, 4, 5), dtype=bool)
        mask_a[0, 1, 1] = True
        mask_a[0, 2, 2] = True
        mask_b = np.zeros((2, 4, 5), dtype=bool)
        mask_b[0, 2, 2] = True
        out = user_roi_superpixel_lists_from_masks(
            [mask_a, mask_b], sp_fastz, sp_rows, sp_cols
        )
        np.testing.assert_array_equal(out[0], np.array([0, 1]))
        np.testing.assert_array_equal(out[1], np.array([1]))


class TestComputeUserRoiGeometry(unittest.TestCase):
    """compute_user_roi_geometry derives channel/z-map/superpixel indices."""

    def _ref(self):
        """Return a 2-channel ref stack with channel 1 brighter."""
        ref = np.zeros((2, 3, 4, 5), dtype=np.float32)
        ref[1] = 10.0
        return ref

    def test_no_motion(self):
        """Zero motion: z_map is fastZ-1; indices from pixel decomposition."""
        ref = self._ref()
        fastz_to_refz = np.array([[1], [2], [3]])
        sub = np.array([[22, 1]], dtype=np.int32)
        geo = compute_user_roi_geometry(ref, fastz_to_refz, sub, (0, 0, 0))
        self.assertEqual(geo["best_ch"], 1)
        self.assertEqual(geo["num_ref_z"], 3)
        self.assertEqual(geo["yx_shape"], (4, 5))
        self.assertEqual(geo["num_fast_z"], 3)
        np.testing.assert_array_equal(geo["z_map"], np.array([0, 1, 2]))
        np.testing.assert_array_equal(geo["sp_fastz"], np.array([1]))
        np.testing.assert_array_equal(geo["sp_cols"], np.array([0]))
        np.testing.assert_array_equal(geo["sp_rows"], np.array([2]))
        self.assertTrue(geo["sp_mask"][1, 2, 0])
        self.assertEqual(geo["sp_mask"].sum(), 1)

    def test_with_motion(self):
        """Nonzero motion shifts z_map and row/column indices."""
        ref = self._ref()
        fastz_to_refz = np.array([[1], [2], [3]])
        sub = np.array([[22, 1]], dtype=np.int32)
        geo = compute_user_roi_geometry(ref, fastz_to_refz, sub, (1, 1, 1))
        np.testing.assert_array_equal(geo["z_map"], np.array([1, 2, 3]))
        np.testing.assert_array_equal(geo["sp_rows"], np.array([3]))
        np.testing.assert_array_equal(geo["sp_cols"], np.array([1]))

    def test_all_invalid_leaves_mask_empty(self):
        """Out-of-bounds superpixels leave sp_mask all False."""
        ref = self._ref()
        fastz_to_refz = np.array([[1]])
        sub = np.array([[100, 1]], dtype=np.int32)
        geo = compute_user_roi_geometry(ref, fastz_to_refz, sub, (0, 0, 0))
        self.assertEqual(geo["sp_mask"].sum(), 0)

    def test_empty_channels(self):
        """A zero-channel ref falls back to best_ch = 0."""
        ref = np.zeros((0, 3, 4, 5), dtype=np.float32)
        geo = compute_user_roi_geometry(
            ref, np.array([[1]]), np.zeros((0, 2), dtype=np.int32), (0, 0, 0)
        )
        self.assertEqual(geo["best_ch"], 0)


class TestH5ScalarStr(unittest.TestCase):
    """_h5_scalar_str decodes scalar/array/bytes string datasets."""

    def test_variants(self):
        """Bytes and 1-element arrays decode to plain str."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.h5")
            with h5py.File(path, "w") as f:
                f.create_dataset("a", data="hi", dtype=_str_dt())
                f.create_dataset(
                    "b", data=np.array(["x"], dtype=object), dtype=_str_dt()
                )
                f.create_dataset("c", data=5)
            with h5py.File(path, "r") as f:
                self.assertEqual(_h5_scalar_str(f["a"]), "hi")
                self.assertEqual(_h5_scalar_str(f["b"]), "x")
                self.assertEqual(_h5_scalar_str(f["c"]), "5")


class TestAnnotationsRoundTrip(unittest.TestCase):
    """save_annotations_h5 / load_annotations_h5 round trip."""

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

    def _records_and_masks(self):
        """Return one ROI record + matching mask for DMD1."""
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
        return records, {"DMD1": [mask]}

    def test_round_trip(self):
        """A saved ROI reloads with matching label, mask, and superpixels."""
        records, masks = self._records_and_masks()
        with tempfile.TemporaryDirectory() as d:
            path = save_annotations_h5(
                d, records, masks, 1, ref_files={"DMD1": "/data/ref.tif"}
            )
            self.assertTrue(os.path.exists(path))
            skip, m, sps, labels, recs = load_annotations_h5(
                path, 1, self._geo()
            )
        self.assertTrue(skip)
        self.assertEqual(labels["DMD1"], ["user_roi_1"])
        self.assertTrue(m["DMD1"][0][0, 1, 1])
        np.testing.assert_array_equal(sps["DMD1"][0], np.array([0]))
        self.assertEqual(recs["DMD1"][0]["type"], "polygon")
        self.assertIn("position", recs["DMD1"][0])

    def test_missing_file(self):
        """A missing annotations file returns the empty result."""
        out = load_annotations_h5("/nope/annotations.h5", 1, self._geo())
        self.assertEqual(out, (False, {}, {}, {}, {}))

    def test_missing_path_group(self):
        """A file with no Path group yields no valid selection."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "annotations.h5")
            with h5py.File(path, "w") as f:
                f["row_major"] = 1
            out = load_annotations_h5(path, 1, self._geo())
        self.assertEqual(out, (False, {}, {}, {}, {}))

    def test_bad_mask_shape(self):
        """A mask shape mismatch treats the path as no selection."""
        records, masks = self._records_and_masks()
        with tempfile.TemporaryDirectory() as d:
            path = save_annotations_h5(d, records, masks, 1)
            geo = self._geo()
            geo["DMD1"]["num_fast_z"] = 3
            out = load_annotations_h5(path, 1, geo)
        self.assertEqual(out, (False, {}, {}, {}, {}))

    def test_empty_records_not_valid(self):
        """A path with zero ROIs is read but not marked valid."""
        with tempfile.TemporaryDirectory() as d:
            path = save_annotations_h5(d, {"DMD1": []}, {"DMD1": []}, 1)
            out = load_annotations_h5(path, 1, self._geo())
        self.assertEqual(out, (False, {}, {}, {}, {}))

    def test_unreadable_file(self):
        """A non-HDF5 file that exists returns the empty result."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "annotations.h5")
            with open(path, "w") as fh:
                fh.write("not an hdf5 file")
            out = load_annotations_h5(path, 1, self._geo())
        self.assertEqual(out, (False, {}, {}, {}, {}))


class TestReadRoiRecords(unittest.TestCase):
    """_read_roi_records handles missing masks, labels, and gaps."""

    def test_defaults_and_missing_mask(self):
        """Missing label/type default; missing mask -> zero mask."""
        expected = (2, 4, 5)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.h5")
            with h5py.File(path, "w") as f:
                grp = f.create_group("Path1")
                grp.create_dataset("n_rois", data=2)
                grp.create_group("roi_000")
            with h5py.File(path, "r") as f:
                masks, labels, recs, bad = _read_roi_records(
                    f["Path1"], expected
                )
        self.assertFalse(bad)
        self.assertEqual(labels, ["ROI1"])
        self.assertEqual(recs[0]["type"], "polygon")
        self.assertEqual(masks[0].shape, expected)
        self.assertFalse(masks[0].any())


if __name__ == "__main__":
    unittest.main()
