"""Tests for giant_python.bandsilo.hdf5."""

import os
import tempfile
import unittest

import h5py
import numpy as np

from giant_python.bandsilo.hdf5 import (
    _decode_h5_strings,
    _decode_one,
    _reshape_1d,
    _resolve_fn_adata,
    compute_keep_trials,
    load_alignment_data_h5,
    load_struct_from_h5,
    load_trial_table,
    read_align_info,
    to_serializable,
    write_dict_to_h5group,
)


def _str_dt():
    """Return the h5py variable-length utf-8 string dtype."""
    return h5py.string_dtype(encoding="utf-8")


class TestToSerializable(unittest.TestCase):
    """to_serializable converts numpy structures to plain Python."""

    def test_dict_list_tuple(self):
        """Nested dict/list/tuple are recursed into."""
        out = to_serializable({"a": [np.int64(1), (np.float64(2.0),)]})
        self.assertEqual(out, {"a": [1, (2.0,)]})
        self.assertIsInstance(out["a"][1], tuple)

    def test_ndarray_and_generic(self):
        """ndarray -> list and numpy scalar -> Python scalar."""
        self.assertEqual(to_serializable(np.arange(3)), [0, 1, 2])
        self.assertEqual(to_serializable(np.float32(1.5)), 1.5)

    def test_passthrough(self):
        """Plain Python values pass through unchanged."""
        self.assertEqual(to_serializable("x"), "x")


class TestDecodeHelpers(unittest.TestCase):
    """_decode_one / _decode_h5_strings handle bytes, str, arrays."""

    def test_decode_one(self):
        """bytes decode to str; str stays str."""
        self.assertEqual(_decode_one(b"ab"), "ab")
        self.assertEqual(_decode_one("cd"), "cd")

    def test_decode_scalar_and_array(self):
        """Scalar bytes -> str; object array -> decoded object array."""
        self.assertEqual(_decode_h5_strings(b"z"), "z")
        arr = _decode_h5_strings(np.array([b"a", b"b"], dtype=object))
        self.assertEqual(list(arr), ["a", "b"])

    def test_decode_empty_array(self):
        """An empty array is returned unchanged (no vectorize crash)."""
        out = _decode_h5_strings(np.array([], dtype=object))
        self.assertEqual(out.size, 0)


class TestReadDataset(unittest.TestCase):
    """load_struct_from_h5 read-side orientation and string handling."""

    def test_scalar_string(self):
        """A scalar string dataset reads back as plain str."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.h5")
            with h5py.File(path, "w") as f:
                f.create_dataset("datadr", data="hello", dtype=_str_dt())
            out = load_struct_from_h5(path)
            self.assertEqual(out["datadr"], "hello")

    def test_single_element_string_array(self):
        """A 1-element string array collapses to a scalar str."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.h5")
            with h5py.File(path, "w") as f:
                f.create_dataset(
                    "name",
                    data=np.array(["x"], dtype=object),
                    dtype=_str_dt(),
                )
            out = load_struct_from_h5(path)
            self.assertEqual(out["name"], "x")

    def test_string_grid_column_major_transposed(self):
        """A 2-D string grid is transposed when column-major (no flag)."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.h5")
            grid = np.array([["a", "b", "c"], ["d", "e", "f"]], dtype=object)
            with h5py.File(path, "w") as f:
                f.create_dataset("filename", data=grid, dtype=_str_dt())
            out = load_struct_from_h5(path)
            self.assertEqual(out["filename"].shape, (3, 2))

    def test_numeric_transpose_and_row_major(self):
        """Numeric 2-D arrays honor the row_major flag."""
        arr = np.arange(6).reshape(2, 3)
        with tempfile.TemporaryDirectory() as d:
            col = os.path.join(d, "c.h5")
            with h5py.File(col, "w") as f:
                f.create_dataset("m", data=arr)
            self.assertEqual(load_struct_from_h5(col)["m"].shape, (3, 2))

            rowm = os.path.join(d, "r.h5")
            with h5py.File(rowm, "w") as f:
                f.create_dataset("row_major", data=1)
                f.create_dataset("m", data=arr)
            got = load_struct_from_h5(rowm)["m"]
            self.assertEqual(got.shape, (2, 3))
            np.testing.assert_array_equal(got, arr)

    def test_nested_group(self):
        """Nested groups become nested dicts."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.h5")
            with h5py.File(path, "w") as f:
                f.create_dataset("row_major", data=1)
                g = f.create_group("inner")
                g.create_dataset("v", data=np.arange(3))
            out = load_struct_from_h5(path)
            np.testing.assert_array_equal(out["inner"]["v"], np.arange(3))


class TestWriteRoundTrip(unittest.TestCase):
    """write_dict_to_h5group covers every value type."""

    def test_round_trip(self):
        """All supported value kinds survive a write/read round trip."""
        payload = {
            "row_major": 1,
            "flag": True,
            "name": "expt",
            "count": 7,
            "scale": 2.5,
            "nums": [1, 2, 3],
            "words": ["a", "b"],
            "empty": [],
            "missing": None,
            "weird": 1 + 2j,
            "nested": {"x": 1},
        }
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "w.h5")
            with h5py.File(path, "w") as f:
                write_dict_to_h5group(f, payload)
            out = load_struct_from_h5(path)

        self.assertNotIn("empty", out)
        self.assertNotIn("missing", out)
        self.assertNotIn("row_major", out)
        self.assertEqual(out["name"], "expt")
        self.assertEqual(int(out["count"]), 7)
        self.assertAlmostEqual(float(out["scale"]), 2.5)
        self.assertEqual(int(out["flag"]), 1)
        np.testing.assert_array_equal(out["nums"], np.array([1, 2, 3]))
        self.assertEqual(list(out["words"]), ["a", "b"])
        self.assertEqual(out["weird"], "(1+2j)")
        self.assertEqual(int(out["nested"]["x"]), 1)


class TestReshapeAndAlignment(unittest.TestCase):
    """_reshape_1d and load_alignment_data_h5."""

    def test_reshape_1d(self):
        """Missing key -> None; present key -> flattened array."""
        self.assertIsNone(_reshape_1d({}, "k"))
        np.testing.assert_array_equal(
            _reshape_1d({"k": np.array([[1], [2]])}, "k"),
            np.array([1, 2]),
        )

    def test_load_alignment_data(self):
        """aData loads flat arrays; missing online fields become None."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.h5")
            with h5py.File(path, "w") as f:
                f.create_dataset("row_major", data=1)
                f.create_dataset("DSframes", data=np.arange(4))
                f.create_dataset("motionDSr", data=np.zeros(4))
                f.create_dataset("motionDSc", data=np.zeros(4))
                f.create_dataset("motionDSz", data=np.zeros(4))
                f.create_dataset("numChannels", data=2)
                f.create_dataset("alignHz", data=100.0)
            out = load_alignment_data_h5(path)
        self.assertEqual(out["numChannels"], 2)
        self.assertAlmostEqual(out["alignHz"], 100.0)
        self.assertEqual(out["DSframes"].shape, (4,))
        self.assertIsNone(out["onlineYshift"])


class TestTrialTable(unittest.TestCase):
    """load_trial_table resolve step + keep-trial / align-info helpers."""

    def test_resolve_fn_adata(self):
        """Empty entries stay empty; others become absolute paths."""
        arr = np.array([["", "a.h5"]], dtype=object)
        out = _resolve_fn_adata(arr, os.path.join("root", "moco"))
        self.assertEqual(out[0, 0], "")
        self.assertEqual(out[0, 1], os.path.join("root", "moco", "a.h5"))

    def _write_trial_table(self, path, savedr="/res", with_fn_adata=True):
        """Write a minimal synthetic trial_table.h5 (row-major)."""
        fnames = np.array([["t1.dat", "t2.dat"]], dtype=object)
        fn_adata = np.array([["t1_AD.h5", ""]], dtype=object)
        with h5py.File(path, "w") as f:
            f.create_dataset("row_major", data=1)
            f.create_dataset("datadr", data="/data", dtype=_str_dt())
            f.create_dataset("savedr", data=savedr, dtype=_str_dt())
            f.create_dataset("filename", data=fnames, dtype=_str_dt())
            si = f.create_group("slap2_info")
            si.create_dataset("first_line", data=np.array([[1, 2]]))
            si.create_dataset("last_line", data=np.array([[10, 20]]))
            rs = si.create_group("ref_stack")
            p1 = rs.create_group("Path1")
            p1.create_dataset("channels", data=np.array([1, 2]))
            mc = f.create_group("motion_correction")
            if with_fn_adata:
                mc.create_dataset("fn_adata", data=fn_adata, dtype=_str_dt())
            ap = mc.create_group("align_params")
            ap.create_dataset("align_hz", data=80.0)

    def test_load_trial_table(self):
        """Resolved dict is flat with hoisted arrays and absolute paths."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trial_table.h5")
            self._write_trial_table(path)
            out = load_trial_table(path)

        self.assertEqual(
            os.path.normpath(out["datadr"]), os.path.normpath("/data")
        )
        self.assertEqual(
            os.path.normpath(out["savedr"]), os.path.normpath("/res")
        )
        self.assertEqual(
            os.path.normpath(out["moco_save_dr"]),
            os.path.normpath(os.path.join("/res", "motion_correction")),
        )
        self.assertEqual(out["n_dmds"], 1)
        self.assertEqual(out["n_trials"], 2)
        self.assertAlmostEqual(out["align_params"]["align_hz"], 80.0)
        self.assertEqual(out["fn_adata"][0, 1], "")
        self.assertEqual(
            os.path.normpath(out["fn_adata"][0, 0]),
            os.path.normpath(
                os.path.join("/res", "motion_correction", "t1_AD.h5")
            ),
        )
        self.assertIn("Path1", out["ref_stack"])
        np.testing.assert_array_equal(out["first_line"], np.array([[1, 2]]))

    def test_load_trial_table_unregistered(self):
        """A table without motion_correction/fn_adata is rejected."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trial_table.h5")
            self._write_trial_table(path, with_fn_adata=False)
            with self.assertRaises(ValueError):
                load_trial_table(path)

    def test_compute_keep_trials(self):
        """Only trials whose aData and source files both exist are kept."""
        with tempfile.TemporaryDirectory() as d:
            adata = os.path.join(d, "t1_AD.h5")
            with open(adata, "w") as fh:
                fh.write("x")
            with open(os.path.join(d, "t1.dat"), "w") as fh:
                fh.write("x")
            fn_adata = np.array([[adata, ""]], dtype=object)
            filename = np.array([["t1.dat", "t2.dat"]], dtype=object)
            keep = compute_keep_trials(fn_adata, filename, d)
        np.testing.assert_array_equal(keep, np.array([[True, False]]))

    def test_read_align_info(self):
        """align_hz/num_channels come from the first kept trial's aData."""
        with tempfile.TemporaryDirectory() as d:
            adata = os.path.join(d, "t1_AD.h5")
            with h5py.File(adata, "w") as f:
                f.create_dataset("row_major", data=1)
                f.create_dataset("numChannels", data=2)
                f.create_dataset("alignHz", data=80.0)
            fn_adata = np.array([[adata, ""]], dtype=object)
            keep = np.array([[True, False]])
            align_hz, num_channels = read_align_info(fn_adata, keep, 1)
        self.assertAlmostEqual(align_hz["DMD1"], 80.0)
        self.assertEqual(num_channels, 2)

    def test_read_align_info_no_valid(self):
        """A DMD with no kept trials contributes no align_hz entry."""
        fn_adata = np.array([["", ""]], dtype=object)
        keep = np.array([[False, False]])
        align_hz, num_channels = read_align_info(fn_adata, keep, 1)
        self.assertEqual(align_hz, {})
        self.assertIsNone(num_channels)


if __name__ == "__main__":
    unittest.main()
