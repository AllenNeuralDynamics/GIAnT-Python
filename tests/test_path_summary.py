"""Nonzero two-path assembly, model, and emitted-schema round-trip coverage."""

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from giant_python.extraction.band.inputs import load_alignment_data_h5
from giant_python.extraction.band.result_assembly import (
    PathResult,
    assemble_path_summary,
)
from giant_python.extraction.band.types import TrialTraceResult
from giant_python.io.experiment_summary import (
    read_summary,
    write_summary,
)
from giant_python.models.experiment_summary import (
    FRAME_INFO_KEYS,
    ExperimentSummary,
    FrameInfo,
)
from tests.test_bandsilo_pipeline import _path_result


class _ArrayTensor:
    """Minimal tensor interface for assembly without a torch import."""

    def __init__(self, array):
        """Keep the supplied array unchanged."""
        self.array = array

    def numpy(self):
        """Expose the backing array, like the CPU tensor producer."""
        return self.array

    def __getitem__(self, key):
        """Return a tensor-like slice for source-parameter indexing."""
        return _ArrayTensor(self.array[key])


def _write_alignment(path, index):
    """Write signed fractional displacement knots with a missing sample."""
    saved = {
        "motionDSr": np.array([1.25, -2.5, np.nan]) + index,
        "motionDSc": np.array([-3.5, 0.75, 2.25]) - index,
        "motionDSz": np.array([0.5, -1.25, 3.0]) + 2 * index,
    }
    with h5py.File(path, "w") as f:
        f["row_major"] = 1
        f["DSframes"] = np.array([[1, 5, 9]])
        f["numChannels"] = 2
        f["alignHz"] = 10.0
        for key, value in saved.items():
            f[key] = value.reshape(-1, 1)
        online = f.create_group("slap2")
        for axis in ("X", "Y", "Z"):
            online[f"onlineMotion{axis}shift"] = np.array([[2, -3, 4]])
    return saved


def _result(index, alignment, roi_state="present"):
    """Build two stitched trials with distinct path shapes and ROI policies."""
    n_sources, channels, nz, rows, cols = (
        (2, 2, 1, 3, 4) if index == 0 else (3, 1, 2, 2, 3)
    )
    frames = np.arange(1, 10, dtype=float)
    # Match the existing band linear/clamped interpolation, not MATLAB PCHIP.
    offline = tuple(
        np.interp(frames, alignment["DSframes"], alignment[key])
        for key in ("motionDSr", "motionDSc", "motionDSz")
    )
    online = tuple(np.arange(9, dtype=np.int16) + axis for axis in (3, -2, 1))
    n_rois = 1 if roi_state == "present" else 0
    traces = []
    for section in (slice(0, 4), slice(4, 9)):
        n_frames = frames[section].size
        signal = (
            np.arange(n_frames * n_sources, dtype=np.float32).reshape(
                n_frames, n_sources
            )
            / 10
        )
        traces.append(
            TrialTraceResult(
                signal,
                np.full_like(signal, 5.0),
                frames[section],
                np.arange(3),
                np.full((n_frames, channels), index + 2.0, dtype=np.float32),
                tuple(axis[section] for axis in offline),
                tuple(axis[section] for axis in online),
                np.full(
                    (n_frames, n_rois, channels), index + 7.0, dtype=np.float32
                ),
            )
        )
    masks = [np.ones((nz, rows, cols), dtype=bool)] if n_rois else []
    return PathResult(
        a=_ArrayTensor(
            np.arange(nz * rows * cols * n_sources, dtype=np.float32).reshape(
                -1, n_sources
            )
        ),
        source_params=_ArrayTensor(
            np.arange(n_sources * 6, dtype=np.float32).reshape(n_sources, 6)
        ),
        source_snr=np.arange(1, n_sources + 1, dtype=np.float32),
        act_im=np.full((nz, rows, cols), index + 0.25, dtype=np.float32),
        mean_im=np.full(
            (channels, nz, rows, cols), index + 4.5, dtype=np.float32
        ),
        act_im_peaks=np.zeros((n_sources, 3), dtype=np.int32),
        trace_results=traces,
        z_depths=np.arange(nz).reshape(-1, 1),
        num_fast_zs=nz,
        dmd_pixels_per_column=rows,
        dmd_pixels_per_row=cols,
        num_channels=channels,
        denoise_window=3,
        baseline_window=5,
        draw_user_rois=roi_state != "absent",
        soma_masks=masks if roi_state != "absent" else None,
        soma_labels=["ROI μ"] if n_rois else [],
        yx_shape=(rows, cols),
    )


def _datasets(path):
    """Return raw dataset values and dtypes for complete schema comparison."""
    values = {}
    with h5py.File(path, "r") as f:

        def collect(name, item):
            """Collect datasets, retaining stored shape/dtype and values."""
            if isinstance(item, h5py.Dataset):
                values[name] = (item[()], item.dtype)

        f.visititems(collect)
    return values


class TestPathSummaryRoundTrip(unittest.TestCase):
    """Per-path data and corrected public shifts survive HDF5 round trips."""

    def test_nonzero_two_path_model_and_schema_roundtrip(self):
        """Raw -> normalized -> interpolated -> assembled -> model -> disk."""
        with tempfile.TemporaryDirectory() as dr:
            results, paths, expected = [], [], []
            for index in range(2):
                alignment_path = Path(dr) / f"alignment{index}.h5"
                saved = _write_alignment(alignment_path, index)
                result = _result(
                    index,
                    load_alignment_data_h5(alignment_path),
                    "present" if index == 0 else "empty",
                )
                results.append(result)
                paths.append(assemble_path_summary(result, index))
                expected.append(
                    {
                        axis: np.interp(
                            np.arange(1, 10), [1, 5, 9], saved[key]
                        )
                        for axis, key in (
                            ("Y", "motionDSr"),
                            ("X", "motionDSc"),
                            ("Z", "motionDSz"),
                        )
                    }
                )
            summary = ExperimentSummary(
                params={
                    "operator": "μ",
                    "numChannels": 2,
                    "draw_user_rois": True,
                },
                paths=paths,
            )
            self.assertEqual([p.path_index for p in summary.paths], [0, 1])
            self.assertEqual([len(p.sources) for p in summary.paths], [2, 3])
            self.assertEqual([len(p.user_rois) for p in summary.paths], [1, 0])
            self.assertTrue(all(p.annotation_enabled for p in summary.paths))
            for index, path in enumerate(summary.paths):
                self.assertIsInstance(path.frame_info, FrameInfo)
                self.assertIs(
                    path.frame_info.to_dict()["offlineXshifts"],
                    path.frame_info.offlineXshifts,
                )
                self.assertTrue(
                    np.shares_memory(
                        path.sources[0].profile, results[index].a.numpy()
                    )
                )
                for axis in ("X", "Y", "Z"):
                    np.testing.assert_allclose(
                        getattr(
                            path.frame_info, f"offline{axis}shifts"
                        ).ravel(),
                        expected[index][axis],
                        equal_nan=True,
                    )
                for axis, position in (("X", 1), ("Y", 0), ("Z", 2)):
                    np.testing.assert_array_equal(
                        getattr(
                            path.frame_info, f"online{axis}shifts"
                        ).ravel(),
                        np.concatenate(
                            [
                                r.online_shifts[position]
                                for r in results[index].trace_results
                            ]
                        ),
                    )
                # Assembling output did not invert the private input arrays.
                np.testing.assert_allclose(
                    -np.concatenate(
                        [
                            r.reference_offsets[1]
                            for r in results[index].trace_results
                        ]
                    ),
                    expected[index]["X"],
                    equal_nan=True,
                )
            model_file = Path(dr) / "model.h5"
            roundtrip_file = Path(dr) / "roundtrip.h5"
            self.assertIsNone(write_summary(summary, model_file))
            reloaded = ExperimentSummary.from_h5(model_file)
            self.assertIsNone(reloaded.to_h5(roundtrip_file))
            original = _datasets(model_file)
            for filename in (roundtrip_file,):
                actual = _datasets(filename)
                self.assertEqual(set(actual), set(original))
                for key, (value, dtype) in original.items():
                    with self.subTest(file=filename.name, dataset=key):
                        self.assertEqual(actual[key][1], dtype)
                        np.testing.assert_array_equal(actual[key][0], value)
            for index, path in enumerate(reloaded.paths):
                self.assertEqual(path.path_index, index)
                np.testing.assert_array_equal(
                    path.visualizations.mean_im, results[index].mean_im
                )
                np.testing.assert_array_equal(
                    path.global_f, paths[index].global_f
                )
                np.testing.assert_array_equal(
                    path.z_depths, results[index].z_depths
                )
            with h5py.File(model_file, "r") as f:
                self.assertEqual(f["Path2/user_rois/mask"].shape, (0, 2, 2, 3))
                self.assertEqual(f["Path2/user_rois/F"].shape, (0, 1, 9))
                self.assertEqual(f["Path2/user_rois/labels"].shape, (0, 1))
                for key in FRAME_INFO_KEYS:
                    self.assertEqual(f[f"Path1/frame_info/{key}"].ndim, 2)

    def test_explicit_paths_and_typed_frame_info_only(self):
        """No aggregate, single-path convenience or mapping APIs remain."""
        data = {
            "DSframes": [1, 5, 9],
            **{
                k: np.zeros(3) for k in ("motionDSr", "motionDSc", "motionDSz")
            },
        }
        first = assemble_path_summary(_result(0, data), path_index=0)
        second = assemble_path_summary(_result(1, data), path_index=1)
        summary = ExperimentSummary(paths=[first, second])
        for model in (
            summary,
            ExperimentSummary(paths=[first]),
            ExperimentSummary(),
        ):
            for name in (
                "frame_info",
                "visualizations",
                "z_depths",
                "global_f",
                "sources",
                "user_rois",
                "_single_path",
            ):
                with self.subTest(paths=len(model.paths), name=name):
                    self.assertFalse(hasattr(model, name))
        self.assertIs(summary.paths[1].sources, second.sources)
        self.assertIs(summary.paths[0].user_rois, first.user_rois)
        for name in ("__getitem__", "__iter__", "__len__"):
            self.assertFalse(hasattr(first.frame_info, name))
        self.assertEqual(set(first.frame_info.to_dict()), set(FRAME_INFO_KEYS))

    def test_absent_rois_and_overwrite_semantics(self):
        """Absent differs from empty; rewrites retain old params."""
        data = {
            "DSframes": [1, 5, 9],
            **{
                k: np.zeros(3) for k in ("motionDSr", "motionDSc", "motionDSz")
            },
        }
        first = assemble_path_summary(_result(0, data, "empty"), 0)
        second = assemble_path_summary(_result(1, data, "present"), 1)
        with tempfile.TemporaryDirectory() as dr:
            path = Path(dr) / "summary.h5"
            write_summary(
                ExperimentSummary(
                    paths=[first, second], params={"operator": "original"}
                ),
                path,
            )
            replacement = assemble_path_summary(_result(0, data, "absent"), 0)
            write_summary(
                ExperimentSummary(
                    paths=[replacement], params={"operator": "new"}
                ),
                path,
            )
            loaded = read_summary(path)
            self.assertFalse(loaded.paths[0].annotation_enabled)
            self.assertTrue(loaded.paths[1].annotation_enabled)
            self.assertEqual(loaded.params["operator"], "original")
            with h5py.File(path, "r") as f:
                self.assertNotIn("user_rois", f["Path1"])
                self.assertIn("user_rois", f["Path2"])

    def test_empty_sources_retain_shape_and_dtype(self):
        """Numerical assembly and codecs retain empty tensor layout."""
        empty = assemble_path_summary(_path_result(n_sources=0), 2)
        summary = ExperimentSummary(paths=[empty])
        with tempfile.TemporaryDirectory() as dr:
            path = Path(dr) / "empty.h5"
            summary.to_h5(path)
            restored = read_summary(path)
            self.assertEqual(restored.paths[0].path_index, 2)
            self.assertEqual(restored.paths[0].sources, [])
            for key, arr in restored.paths[0]._empty_source_arrays.items():
                expected = empty._empty_source_arrays[key]
                self.assertEqual(arr.shape, expected.shape)
                self.assertEqual(arr.dtype, expected.dtype)

    def test_column_major_summary_decodes_without_sign_conversion(self):
        """The summary codec uses the same 3D/4D axis convention as raw IO."""
        data = {
            "DSframes": [1, 5, 9],
            "motionDSr": np.array([1.25, -3.5, 2.0]),
            "motionDSc": np.array([-2.5, 0.25, 4.0]),
            "motionDSz": np.array([0.5, 1.5, -0.75]),
        }
        summary = ExperimentSummary(
            paths=[
                assemble_path_summary(_result(0, data, "present"), 0),
                assemble_path_summary(_result(1, data, "empty"), 1),
            ]
        )
        with tempfile.TemporaryDirectory() as dr:
            original = Path(dr) / "row.h5"
            matlab = Path(dr) / "column.h5"
            restored = Path(dr) / "restored.h5"
            write_summary(summary, original)
            datasets = _datasets(original)
            with h5py.File(matlab, "w") as f:
                for name, (value, dtype) in datasets.items():
                    array = np.asarray(value)
                    if name == "row_major":
                        array = np.asarray(0)
                    elif array.ndim >= 2:
                        array = array.T
                    f.create_dataset(name, data=array, dtype=dtype)
            loaded = read_summary(matlab)
            write_summary(loaded, restored)
            rewritten = _datasets(restored)
            for name, (value, dtype) in datasets.items():
                with self.subTest(dataset=name):
                    self.assertEqual(rewritten[name][1], dtype)
                    np.testing.assert_array_equal(rewritten[name][0], value)
