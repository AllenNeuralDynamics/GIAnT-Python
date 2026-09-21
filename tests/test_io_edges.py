"""Regression coverage for schema validation and public IO/model seams."""

import importlib
import sys
import unittest
from dataclasses import fields, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import h5py
import numpy as np

from giant_python.extraction.band import annotation, inputs
from giant_python.io.experiment_summary import read_summary, write_summary
from giant_python.io.hdf5 import write_dict_to_h5group
from giant_python.models.experiment_summary import (
    FRAME_INFO_KEYS,
    ExperimentSummary,
    FrameInfo,
    PathSummary,
    Source,
    UserRoi,
    Visualizations,
)
from giant_python.models.params import (
    AnnotationOptions,
    BandSiloParams,
    ExecutionOptions,
)
from giant_python.models.trial_table import Slap2Info


def _empty_source_path(**kwargs):
    """Provide empty band tensors with explicit rank/dtype templates."""
    values = {
        "z_depths": np.array([[1], [2]], dtype=np.int32),
        "global_f": np.ones((2, 5), dtype=np.float32),
        "frame_info": FrameInfo(
            **{key: np.zeros((5, 1)) for key in FRAME_INFO_KEYS}
        ),
        "visualizations": Visualizations(
            mean_im=np.ones((2, 2, 3, 4), dtype=np.float32),
            act_im=np.ones((2, 3, 4), dtype=np.float32),
            act_im_peaks=np.empty((0, 3), dtype=np.int32),
        ),
        **kwargs,
    }
    return PathSummary(
        _empty_source_arrays={
            "profiles": np.empty((0, 2, 3, 4), dtype=np.float32),
            "coords": np.empty((0, 3), dtype=np.int32),
            "dF_ls": np.empty((0, 2, 5), dtype=np.float32),
            "F0": np.empty((0, 2, 5), dtype=np.float32),
            "SNR": np.empty((0, 1), dtype=np.float32),
        },
        **values,
    )


def _roundtrip(path):
    """Exercise public model persistence in a real temporary HDF5 file."""
    with TemporaryDirectory() as directory:
        destination = Path(directory) / "summary.h5"
        write_summary(ExperimentSummary(paths=[path]), destination)
        return read_summary(destination).paths[0]


class TestSummaryValidation(unittest.TestCase):
    """Invalid models fail explicitly rather than silently losing ROI data."""

    def test_missing_arrays_preserve_existing_file_groups(self):
        """Report missing arrays by field/index before replacing the path."""
        source = Source(
            profile=np.ones((1, 2, 3), dtype=np.float32),
            coords=np.array([0, 1, 2], dtype=np.int32),
            df_ls=np.ones((1, 4), dtype=np.float32),
            f0=np.ones((1, 4), dtype=np.float32),
            snr=2.0,
        )
        roi = UserRoi(
            label="retained μ",
            mask=np.ones((1, 2, 3), dtype=bool),
            f=np.ones((1, 4), dtype=np.float32),
        )
        shifts = np.array([[1.5, -2.0, np.nan]], dtype=np.float32)
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "summary.h5"
            with h5py.File(destination, "w") as file:
                file["row_major"] = 1
                file.create_group("params").attrs["owner"] = "original"
                for name in ("Path1", "Path2"):
                    group = file.create_group(name)
                    group.attrs["marker"] = name
                    group.create_group("frame_info").create_dataset(
                        "offlineXshifts", data=shifts
                    )
            for owner, field in (
                ("Source", "profile"),
                ("Source", "coords"),
                ("Source", "df_ls"),
                ("Source", "f0"),
                ("UserRoi", "mask"),
                ("UserRoi", "f"),
            ):
                with self.subTest(owner=owner, field=field):
                    path = PathSummary(
                        path_index=1,
                        sources=[source, replace(source)],
                        user_rois=[roi, replace(roi)],
                        annotation_enabled=True,
                    )
                    entry = (
                        path.sources[1]
                        if owner == "Source"
                        else path.user_rois[1]
                    )
                    setattr(entry, field, None)
                    message = rf"{owner}\.{field}\[1\] is required; got None"
                    with self.assertRaisesRegex(ValueError, message):
                        write_summary(
                            ExperimentSummary(paths=[path]), destination
                        )
                    with h5py.File(destination, "r") as file:
                        self.assertEqual(
                            set(file),
                            {"row_major", "params", "Path1", "Path2"},
                        )
                        self.assertEqual(file["row_major"][()], 1)
                        self.assertEqual(
                            file["params"].attrs["owner"], "original"
                        )
                        self.assertEqual(set(file["params"]), set())
                        for name in ("Path1", "Path2"):
                            group = file[name]
                            self.assertEqual(group.attrs["marker"], name)
                            self.assertEqual(set(group), {"frame_info"})
                            self.assertEqual(
                                set(group["frame_info"]), {"offlineXshifts"}
                            )
                            saved = group["frame_info/offlineXshifts"]
                            np.testing.assert_array_equal(saved[()], shifts)
                            self.assertEqual(saved.dtype, shifts.dtype)

    def test_sources_without_shape_templates_are_rejected(self):
        """Empty source lists need geometry that cannot be inferred."""
        with self.assertRaisesRegex(ValueError, "shape templates"):
            _roundtrip(PathSummary())

    def test_disabled_annotations_reject_supplied_rois(self):
        """Callers must explicitly enable serialization of supplied ROIs."""
        path = _empty_source_path(user_rois=[UserRoi(label="keep me")])
        with self.assertRaisesRegex(ValueError, "annotation_enabled=True"):
            _roundtrip(path)
        self.assertEqual(path.user_rois[0].label, "keep me")
        self.assertFalse(path.annotation_enabled)

    def test_direct_empty_rois_infer_layout_and_preserve_dtype(self):
        """Enabled empty ROIs infer mask/trace geometry without mutation."""
        mean = np.ones((2, 2, 3, 4), dtype=np.float32)
        global_f = np.ones((2, 5), dtype=np.float64)
        path = _empty_source_path(
            annotation_enabled=True,
            visualizations=Visualizations(
                mean_im=mean,
                act_im=np.ones((2, 3, 4)),
                act_im_peaks=np.empty((0, 3)),
            ),
            global_f=global_f,
        )
        out = _roundtrip(path)
        rois = out._empty_roi_arrays
        self.assertEqual(rois["mask"].shape, (0, 2, 3, 4))
        self.assertEqual(rois["mask"].dtype, np.dtype(bool))
        self.assertEqual(rois["F"].shape, (0, 2, 5))
        self.assertEqual(rois["F"].dtype, global_f.dtype)
        self.assertEqual(out.user_rois, [])
        self.assertIs(path.visualizations.mean_im, mean)
        self.assertIs(path.global_f, global_f)
        np.testing.assert_array_equal(out.visualizations.mean_im, mean)
        np.testing.assert_array_equal(out.global_f, global_f)
        self.assertEqual(path._empty_roi_arrays, {})

    def test_empty_rois_require_both_geometry_ranks(self):
        """Neither malformed image geometry nor trace geometry is guessed."""
        for mean_shape, trace_shape in (
            ((2, 3, 4), (2, 5)),
            ((2, 2, 3, 4), (5,)),
        ):
            with self.subTest(mean=mean_shape, traces=trace_shape):
                path = _empty_source_path(
                    annotation_enabled=True,
                    visualizations=Visualizations(
                        mean_im=np.zeros(mean_shape)
                    ),
                    global_f=np.zeros(trace_shape),
                )
                with self.assertRaisesRegex(ValueError, "geometry"):
                    _roundtrip(path)

    def test_duplicate_path_ids_fail_before_opening_destination(self):
        """Duplicate identities cannot replace data in a destination file."""
        summary = ExperimentSummary(
            paths=[PathSummary(path_index=3), PathSummary(path_index=3)]
        )
        with patch("giant_python.io.experiment_summary.h5py.File") as opener:
            with self.assertRaisesRegex(ValueError, "duplicate path_index"):
                write_summary(summary, "must-not-open.h5")
        opener.assert_not_called()

    def test_model_identity_validation_and_empty_fields(self):
        """Path IDs are nonnegative integers; empty summaries stay empty."""
        for value in (-1, 1.5, "1", None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "nonnegative"):
                    PathSummary(path_index=value)
        self.assertEqual(PathSummary(path_index=np.int64(2)).path_index, 2)
        summary = ExperimentSummary()
        self.assertEqual(summary.paths, [])
        self.assertIsNone(summary.params)
        self.assertEqual(
            {item.name for item in fields(summary)}, {"paths", "params"}
        )
        info = FrameInfo()
        self.assertEqual(tuple(info.to_dict()), FRAME_INFO_KEYS)
        self.assertTrue(
            all(value is None for value in info.to_dict().values())
        )

    def test_path_fields_share_original_arrays(self):
        """Explicit path access preserves array identity, dtype and rank."""
        path = PathSummary(
            global_f=np.ones((2, 5), dtype=np.float32),
            z_depths=np.array([[1], [3]], dtype=np.int16),
        )
        summary = ExperimentSummary(paths=[path])
        self.assertIs(summary.paths[0].global_f, path.global_f)
        self.assertIs(summary.paths[0].z_depths, path.z_depths)
        self.assertIs(summary.paths[0].visualizations, path.visualizations)

    def test_append_retains_layout_flags_and_unrelated_groups(self):
        """Replacement preserves root metadata, including MATLAB flags."""
        for layout in (0, 1):
            with (
                self.subTest(layout=layout),
                TemporaryDirectory() as directory,
            ):
                destination = Path(directory) / "summary.h5"
                with h5py.File(destination, "w") as file:
                    file["row_major"] = np.uint8(layout)
                    file["coords_zero_indexed"] = np.uint8(0)
                    file["params/operator"] = "original μ"
                    file["Path1/obsolete"] = 5
                    file["Path12/retained"] = np.array([[2, 7]], np.int16)
                    file["unrelated/labels"] = "retained μ"
                path = _empty_source_path()
                write_summary(
                    ExperimentSummary(
                        paths=[path], params={"operator": "replacement"}
                    ),
                    destination,
                )
                with h5py.File(destination, "r") as file:
                    self.assertEqual(file["row_major"][()], layout)
                    self.assertEqual(file["row_major"].dtype, np.uint8)
                    self.assertEqual(file["coords_zero_indexed"][()], 0)
                    self.assertEqual(
                        file["params/operator"].asstr()[()], "original μ"
                    )
                    self.assertEqual(
                        file["unrelated/labels"].asstr()[()], "retained μ"
                    )
                    self.assertNotIn("obsolete", file["Path1"])
                    np.testing.assert_array_equal(
                        file["Path1/Z_depths"], path.z_depths
                    )
                    np.testing.assert_array_equal(
                        file["Path12/retained"], [[2, 7]]
                    )
                    self.assertEqual(file["Path12/retained"].dtype, np.int16)


class TestBandInputValidation(unittest.TestCase):
    """Missing lookup/reference groups fail at the DMD input boundary."""

    def test_missing_lookup_groups_identify_requested_dmd(self):
        """Absent or None aliases fail after any preceding valid DMD group."""
        valid = {
            "allSuperPixelIDs": np.array([1]),
            "sparseMaskInds": np.array([[1, 2]]),
            "fastZ2RefZ": np.array([1]),
        }
        for missing in ({}, {"Path2": None}, {"DMD2": None}):
            with self.subTest(missing=missing):
                raw = {"Path1": valid, **missing}
                with patch.object(inputs, "load_struct_h5", return_value=raw):
                    with self.assertRaisesRegex(
                        ValueError, "Missing lookup-table group Path2 or DMD2"
                    ):
                        inputs.load_lookup_table("lookup.h5", 2)

    def test_missing_reference_groups_fail_before_file_discovery(self):
        """Distinguish missing metadata from a missing reference TIFF."""
        for missing in ({}, {"Path2": None}, {"DMD2": None}):
            with self.subTest(missing=missing):
                raw = {"Path1": {"channels": np.array([1])}, **missing}
                with (
                    patch.object(inputs, "find_reference_file") as find,
                    patch.object(inputs, "read_tiff") as read,
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        "Missing reference-stack group Path2 or DMD2",
                    ):
                        inputs.load_reference_stack("raw", raw, 1)
                find.assert_not_called()
                read.assert_not_called()


class TestParameterWriterEdges(unittest.TestCase):
    """Arrays and Python sequences retain their persisted string policies."""

    def test_workflow_parameters_keep_numeric_storage(self):
        """Canonical Python params retain scalar types and UTF-8 labels."""
        summary = ExperimentSummary(
            params={
                "numChannels": 2,
                "analyzeHz": 100.0,
                "draw_user_rois": True,
                "operator": "operator μ",
            }
        )
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "summary.h5"
            write_summary(summary, destination)
            with h5py.File(destination, "r") as file:
                expected = {
                    "numChannels": (2, np.int64),
                    "analyzeHz": (100.0, np.float64),
                    "draw_user_rois": (1, np.uint8),
                }
                for name, (value, dtype) in expected.items():
                    dataset = file[f"params/{name}"]
                    self.assertEqual(dataset.shape, ())
                    self.assertEqual(dataset.dtype, dtype)
                    self.assertEqual(dataset[()], value)
                self.assertEqual(
                    file["params/operator"].asstr()[()], "operator μ"
                )

    def test_preserved_strings_and_sequences(self):
        """UTF-8 grids retain shape; sequence bytes keep their str spelling."""
        values = {
            "unicode": np.array([["μ", ""], ["é", "last"]]),
            "bytes": np.array([[b"first", b"\xff"]]),
            "objects": np.array([["μ", b"tail"]], dtype=object),
            "scalar": np.str_("μ"),
            "empty": np.empty((0, 2), dtype="U1"),
            "nested": {"flag": np.uint8(1)},
        }
        with h5py.File(
            "parameter-edges.h5", "w", driver="core", backing_store=False
        ) as file:
            write_dict_to_h5group(file, values)
            np.testing.assert_array_equal(
                file["unicode"].asstr()[()], values["unicode"]
            )
            np.testing.assert_array_equal(
                file["bytes"].asstr()[()], [["first", "�"]]
            )
            np.testing.assert_array_equal(
                file["objects"].asstr()[()], [["μ", "tail"]]
            )
            self.assertEqual(file["scalar"].asstr()[()], "μ")
            self.assertEqual(file["empty"].shape, (0, 2))
            self.assertEqual(file["nested/flag"].dtype, np.dtype("uint8"))
            legacy = file.create_group("legacy")
            write_dict_to_h5group(
                legacy,
                {
                    "bytes": [b"first", b"last"],
                    "numbers": (1, 2),
                    "empty": [],
                    "none": None,
                },
            )
            np.testing.assert_array_equal(
                legacy["bytes"].asstr()[()], ["b'first'", "b'last'"]
            )
            np.testing.assert_array_equal(legacy["numbers"][()], [1, 2])
            self.assertNotIn("empty", legacy)
            self.assertNotIn("none", legacy)


class TestModelAndImportBoundaries(unittest.TestCase):
    """Canonical model readers and public imports keep their ownership."""

    def test_trial_table_metadata_adapter(self):
        """Raw metadata decoding filters unknown fields without copying."""
        from giant_python.io.trial_table import slap2_info_from_raw

        first_line = np.array([[1, 9]], dtype=np.int32)
        raw = {"first_line": first_line, "unknown": "ignored"}
        info = slap2_info_from_raw(raw)
        self.assertIsInstance(info, Slap2Info)
        self.assertIs(info.first_line, first_line)
        self.assertFalse(hasattr(info, "unknown"))
        self.assertIsNone(slap2_info_from_raw(None))
        self.assertIsNone(slap2_info_from_raw({}))

    def test_lazy_io_exports_and_unknown_attribute(self):
        """Schema exports resolve and cache canonical functions lazily."""
        from giant_python import io

        importlib.reload(io)
        for name, owner in io._SCHEMA_EXPORTS.items():
            with self.subTest(name=name):
                expected = getattr(
                    importlib.import_module(owner, io.__name__), name
                )
                self.assertIs(io.__getattr__(name), expected)
                self.assertIs(getattr(io, name), expected)
                self.assertIn(name, io.__all__)
        with self.assertRaisesRegex(AttributeError, "unknown_codec"):
            io.__getattr__("unknown_codec")

    def test_annotation_and_options_import_without_toolkits(self):
        """Headless orchestration and scoped option metadata need no GUI."""
        with patch.dict(sys.modules, {"tkinter": None, "cv2": None}):
            importlib.reload(annotation)
            self.assertTrue(callable(annotation.annotate_band_rois))
            science = {item.name for item in fields(BandSiloParams)}
            execution = {item.name for item in fields(ExecutionOptions)}
            annotations = {item.name for item in fields(AnnotationOptions)}
            self.assertFalse(science & execution)
            self.assertFalse(science & annotations)
            self.assertFalse(execution & annotations)


class TestAnnotationGeometryAssembly(unittest.TestCase):
    """Per-DMD geometry uses each path's first kept trial and provenance."""

    def test_geometry_uses_first_kept_trial_for_each_dmd(self):
        """Distinct motion offsets and references stay associated by DMD."""
        references = [np.ones((1, 2, 5, 6)), np.full((1, 2, 5, 6), 2.0)]
        table = {
            "datadr": Path("raw"),
            "ref_stack": {"metadata": "retained"},
            "n_dmds": 2,
            "keep_trials": np.array([[False, True], [True, False]]),
            "fn_adata": np.array([["skip-a", "a.h5"], ["b.h5", "skip-b"]]),
        }
        lookup = {
            "allSuperPixelIDs": {"DMD1": [11], "DMD2": [22]},
            "sparseMaskInds": {"DMD1": [33], "DMD2": [44]},
            "fastZ2RefZ": {"DMD1": np.array([1, 2]), "DMD2": np.array([1, 2])},
        }
        motion = [
            {"motionDSr": [1.0], "motionDSc": [-1.0], "motionDSz": [0.0]},
            {"motionDSr": [-1.0], "motionDSc": [1.0], "motionDSz": [0.0]},
        ]
        with (
            patch.object(
                annotation.inputs,
                "load_reference_stack",
                side_effect=[
                    (references[0], None, "ref-a.tif"),
                    (references[1], None, "ref-b.tif"),
                ],
            ) as load_reference,
            patch.object(
                annotation.geo,
                "build_subsample_matrix_inds",
                return_value=np.array([[6], [36]]),
            ) as subsample,
            patch.object(
                annotation, "load_alignment_data_h5", side_effect=motion
            ) as load_alignment,
        ):
            geometry, provenance = annotation.build_user_roi_geometry(
                table, lookup
            )
        self.assertEqual(
            provenance, {"DMD1": "ref-a.tif", "DMD2": "ref-b.tif"}
        )
        self.assertEqual(
            [call.args[0] for call in load_alignment.call_args_list],
            ["a.h5", "b.h5"],
        )
        for index, key in enumerate(("DMD1", "DMD2")):
            with self.subTest(path=key):
                self.assertIs(geometry[key]["ref"], references[index])
                np.testing.assert_array_equal(geometry[key]["z_map"], [0, 1])
                np.testing.assert_array_equal(
                    geometry[key]["sp_rows"], [2, 2] if index == 0 else [0, 0]
                )
                np.testing.assert_array_equal(
                    geometry[key]["sp_cols"], [0, 0] if index == 0 else [2, 2]
                )
                self.assertEqual(np.count_nonzero(geometry[key]["sp_mask"]), 2)
                self.assertEqual(
                    load_reference.call_args_list[index].args,
                    ("raw", table["ref_stack"], index),
                )
                self.assertEqual(
                    subsample.call_args_list[index].args,
                    (
                        lookup["allSuperPixelIDs"][key],
                        lookup["sparseMaskInds"][key],
                    ),
                )
        np.testing.assert_array_equal(
            table["keep_trials"], [[False, True], [True, False]]
        )

    def test_annotation_option_resolution_preserves_input(self):
        """Annotation uses scoped option conversion without mutation."""
        original = AnnotationOptions(operator="operator μ", interactive=False)
        _, _, resolved = annotation.resolve_band_options(annotations=original)
        self.assertEqual(resolved.operator, "operator μ")
        self.assertFalse(resolved.interactive)
        self.assertEqual(original.operator, "operator μ")
        self.assertFalse(original.interactive)
