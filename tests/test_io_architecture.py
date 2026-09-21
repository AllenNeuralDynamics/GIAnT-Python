"""IO ownership, faithful decoding, and loaded-TrialTable regression tests."""

import ast
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import h5py
import numpy as np

from giant_python.extraction.band import inputs, result_assembly
from giant_python.io import experiment_summary, hdf5, slap2, trial_table
from giant_python.io.annotations import (
    read_annotations_h5,
    save_annotations_h5,
)
from giant_python.io.tiff import read_tiff
from giant_python.io.trial_table import read_trial_table
from giant_python.models.trial_table import Slap2Info, TrialTable


class TestSharedDecoder(unittest.TestCase):
    """There is one HDF5 decoder and it preserves the storage convention."""

    def test_removed_aliases_are_unavailable(self):
        """There is one public spelling for each raw/schema operation."""
        from giant_python import io

        for owner, names in (
            (
                hdf5,
                (
                    "load_struct_from_h5",
                    "_h5_is_string_dataset",
                    "_decode_h5_strings",
                    "_read_h5_dataset",
                    "_read_h5_group",
                ),
            ),
            (slap2, ("read_alignment_data_h5",)),
            (trial_table, ("load_trial_table",)),
            (
                experiment_summary,
                (
                    "write_experiment_summary",
                    "read_experiment_summary",
                    "path_summary_from_assembled",
                    "path_summary_to_assembled",
                ),
            ),
            (
                result_assembly,
                ("assemble_path_outputs", "build_experiment_summary"),
            ),
            (io, ("load_struct_from_h5", "write_experiment_summary")),
        ):
            for name in names:
                with self.subTest(owner=owner.__name__, name=name):
                    self.assertFalse(hasattr(owner, name))
        self.assertIs(inputs.load_struct_h5, hdf5.load_struct_h5)
        self.assertIs(
            inputs.read_alignment_data_h5, slap2.load_alignment_data_h5
        )

    def test_layout_flags_and_string_grids(self):
        """Missing/zero flags reverse 3D axes; rank-two vectors survive."""
        cube = np.arange(24).reshape(2, 3, 4)
        labels = np.array([["a", "b", "c"], ["d", "e", "f"]], dtype=object)
        with tempfile.TemporaryDirectory() as dr:
            for flag in (None, 0, 1):
                with self.subTest(flag=flag):
                    path = Path(dr) / "layout.h5"
                    with h5py.File(path, "w") as f:
                        if flag is not None:
                            f["row_major"] = flag
                        f["cube"] = cube
                        f["vector"] = np.arange(3).reshape(1, 3)
                        f.create_dataset(
                            "labels", data=labels, dtype=h5py.string_dtype()
                        )
                    decoded = hdf5.load_struct_h5(path)
                    np.testing.assert_array_equal(
                        decoded["cube"], cube if flag else cube.T
                    )
                    np.testing.assert_array_equal(
                        decoded["labels"], labels if flag else labels.T
                    )
                    self.assertEqual(
                        decoded["vector"].shape, (1, 3) if flag else (3, 1)
                    )

    def test_raw_alignment_sign_and_private_adapter(self):
        """Raw displacement stays unchanged; the backend negates offsets."""
        saved = np.array([1.25, -2.5, np.nan])
        with tempfile.TemporaryDirectory() as dr:
            path = Path(dr) / "alignment.h5"
            with h5py.File(path, "w") as f:
                f["row_major"] = 1
                f["DSframes"] = np.array([[1, 5, 9]])
                f["numChannels"] = 2
                f["alignHz"] = 10.0
                for key in ("motionDSr", "motionDSc", "motionDSz"):
                    f[key] = saved.reshape(-1, 1)
                f.create_group("slap2")["onlineMotionXshift"] = saved.reshape(
                    1, -1
                )
            raw = slap2.load_alignment_data_h5(path)
            with patch.object(
                inputs, "read_alignment_data_h5", return_value=raw
            ):
                normalized = inputs.load_alignment_data_h5(path)
            for key in ("motionDSr", "motionDSc", "motionDSz"):
                np.testing.assert_array_equal(raw[key], saved)
                np.testing.assert_array_equal(normalized[key], -saved)
                self.assertFalse(np.shares_memory(raw[key], normalized[key]))
            self.assertIs(normalized["onlineXshift"], raw["onlineXshift"])
            self.assertIsNone(normalized["onlineYshift"])


class TestTrialTableBoundary(unittest.TestCase):
    """Models retain raw metadata; backend views do path/file preparation."""

    def test_in_memory_table_is_never_reloaded(self):
        """A loaded model keeps identity without reconstructing a filename."""
        table = TrialTable(
            datadr=Path("raw"),
            savedr=Path("results"),
            filename=np.array([["trial.dat"]], dtype=object),
            slap2_info=Slap2Info(first_line=[[1]], last_line=[[9]]),
            motion_correction={"fn_adata": np.array([["trial.h5"]])},
            source_path=Path("unusual-name.h5"),
        )
        with patch.object(
            TrialTable, "from_h5", side_effect=AssertionError("reload")
        ):
            view = inputs.load_trial_table(table)
        self.assertIs(view["trial_table"], table)
        self.assertEqual(view["source_path"], Path("unusual-name.h5"))
        self.assertEqual(
            view["fn_adata"][0, 0],
            str(Path("results/motion_correction/trial.h5")),
        )
        self.assertNotIn("keep_trials", view)
        self.assertNotIn("align_hz", view)

    def test_model_codec_records_source_without_probing(self):
        """Noncanonical table filenames survive the IO-model delegation."""
        with tempfile.TemporaryDirectory() as dr:
            path = Path(dr) / "not-the-standard-name.h5"
            with h5py.File(path, "w") as f:
                f["row_major"] = 1
                f["datadr"] = "not-a-real-directory"
                f.create_dataset(
                    "filename", data=[["x.dat"]], dtype=h5py.string_dtype()
                )
                f["true_trial_ix"] = np.array([[4]])
                f.create_group("slap2_info")["first_line"] = np.array([[1]])
            table = TrialTable.from_h5(path)
            self.assertEqual(table.source_path, path.absolute())
            self.assertEqual(table.filename.shape, (1, 1))
            np.testing.assert_array_equal(table.true_trial_ix, [[4]])
            with h5py.File(path, "r") as f:
                self.assertNotIn("source_path", f)
            self.assertEqual(
                read_trial_table(path).source_path, table.source_path
            )


class TestRawAnnotations(unittest.TestCase):
    """Raw annotations are readable without any reference-stack geometry."""

    def test_provenance_overlap_and_empty_path(self):
        """Masks and provenance survive; empty Path groups are explicit."""
        mask = np.ones((2, 3, 4), dtype=bool)
        with tempfile.TemporaryDirectory() as dr:
            file = save_annotations_h5(
                dr,
                {"DMD1": [{"label": "A"}, {"label": "B"}]},
                {"DMD1": [mask, mask]},
                2,
                ref_files={"DMD1": str(Path(dr) / "reference.tif")},
            )
            raw = read_annotations_h5(file)
        self.assertEqual(raw["Path1"]["fn"], "reference.tif")
        self.assertEqual(int(raw["Path2"]["n_rois"]), 0)
        np.testing.assert_array_equal(raw["Path1"]["roi_000"]["mask"], mask)
        np.testing.assert_array_equal(raw["Path1"]["roi_001"]["mask"], mask)


class TestRawAcquisitionSeams(unittest.TestCase):
    """Raw reader construction and TIFF reads preserve native contracts."""

    def test_open_preserves_vendor_selection_and_caller_ownership(self):
        """Opening reads/closes nothing; CYCLE names use MultiDataFiles."""
        vendor = ModuleType("slap2_utils")
        single, multiple = Mock(), Mock()
        vendor.DataFile = Mock(return_value=single)
        vendor.MultiDataFiles = Mock(return_value=multiple)
        with patch.dict(sys.modules, {"slap2_utils": vendor}):
            with patch("importlib.reload", return_value=vendor) as reload:
                self.assertIs(slap2.open_slap2_file("trial.dat"), single)
                self.assertIs(
                    slap2.open_slap2_file("trial_CYCLE001.dat"), multiple
                )
                self.assertEqual(reload.call_count, 2)
        vendor.DataFile.assert_called_once_with("trial.dat")
        vendor.MultiDataFiles.assert_called_once_with("trial_CYCLE001.dat")
        single.getLineData.assert_not_called()
        multiple.getLineData.assert_not_called()
        single.close.assert_not_called()
        multiple.close.assert_not_called()

    def test_native_tiff_shape_dtype_and_intensity(self):
        """Shared TIFF decoding does not perform band reference scaling."""
        import tifffile

        array = np.arange(60, dtype=np.uint16).reshape(3, 4, 5)
        with tempfile.TemporaryDirectory() as dr:
            path = Path(dr) / "reference.tif"
            tifffile.imwrite(path, array, photometric="minisblack")
            loaded = read_tiff(path)
        self.assertEqual(loaded.dtype, array.dtype)
        np.testing.assert_array_equal(loaded, array)


class TestImportOwnership(unittest.TestCase):
    """Source-level import boundaries remain independent of heavy backends."""

    def test_io_and_models_do_not_import_band_or_toolkits(self):
        """Shared readers and models do not import band/numerics/GUI."""
        root = Path(__file__).resolve().parents[1] / "src/giant_python"
        paths = list((root / "io").glob("*.py"))
        paths += [
            root / "models/experiment_summary.py",
            root / "models/trial_table.py",
        ]
        for path in paths:
            with self.subTest(path=path.name):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        self.assertFalse(
                            any(
                                part in module.split(".")
                                for part in (
                                    "bandsilo",
                                    "extraction",
                                    "torch",
                                    "cv2",
                                    "tkinter",
                                )
                            )
                        )

    def test_numerical_assembly_constructs_models_without_io(self):
        """Results may depend on models/numerics, never persistence or GUI."""
        tree = ast.parse(
            Path(result_assembly.__file__).read_text(encoding="utf-8")
        )
        forbidden = {"io", "h5py", "workflow", "gui", "bandsilo"}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertFalse(
                    forbidden.intersection((node.module or "").split("."))
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertFalse(
                        forbidden.intersection(alias.name.split("."))
                    )
        public_functions = {
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and not node.name.startswith("_")
        }
        self.assertEqual(public_functions, {"assemble_path_summary"})
