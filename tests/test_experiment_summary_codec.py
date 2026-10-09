"""Focused full-schema summary codec and legacy compatibility coverage."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np

from giant_python.io.experiment_summary import read_summary, write_summary
from giant_python.models.experiment_summary import (
    FRAME_INFO_KEYS,
    ExperimentSummary,
    FrameInfo,
    PathSummary,
    Source,
    UserRoi,
    Visualizations,
)


def _path(n_sources=2, n_rois=2, annotation_enabled=True, frames=7):
    """Use distinct values and non-square geometry to expose axis changes."""
    trace = np.arange(2 * frames, dtype=np.float32).reshape(2, frames)
    profile = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    return PathSummary(
        sources=[
            Source(
                profile=profile + index,
                coords=np.array([index, index + 1, index + 2], np.int32),
                df_ls=trace + index,
                df_denoised=trace + 100 + index,
                events=trace + 200 + index,
                f0=trace + 300 + index,
                snr=np.float32(index + 0.5),
            )
            for index in range(n_sources)
        ],
        user_rois=[
            UserRoi(
                label=f"ROI μ {index}",
                mask=(profile % (index + 2)) == 0,
                f=trace + 400 + index,
                fsvd=trace + 500 + index,
            )
            for index in range(n_rois)
        ],
        visualizations=Visualizations(
            mean_im=np.stack([profile, profile + 10]),
            act_im=profile + 20,
            act_im_peaks=np.zeros((n_sources, 3), np.int32),
        ),
        frame_info=FrameInfo(**{
            key: (np.arange(frames, dtype=np.float32) - index).reshape(-1, 1)
            for index, key in enumerate(FRAME_INFO_KEYS)
        }),
        global_f=trace + 600,
        z_depths=np.array([[2], [4], [8]], dtype=np.int16),
        annotation_enabled=annotation_enabled,
    )


def _snapshot(destination):
    """Capture datasets, groups and attributes without relying on HDF5 bytes."""
    snapshot = {}
    with h5py.File(destination, "r") as file:
        def collect(name, item):
            """Keep values, dtype, shape and all metadata for comparison."""
            snapshot[(name, "kind")] = type(item).__name__
            for key, value in item.attrs.items():
                snapshot[(name, "attribute", key)] = value
            if isinstance(item, h5py.Dataset):
                snapshot[(name, "value")] = item[()]
                snapshot[(name, "dtype")] = item.dtype
                snapshot[(name, "shape")] = item.shape

        collect("", file)
        file.visititems(collect)
    return snapshot


class TestExperimentSummaryCodec(unittest.TestCase):
    """Optional data stay genuine, missing data stay explicitly unavailable."""

    def assert_full_schema(self, file, sources, rois, frames=7):
        """Verify the full row-major schema, including zero-length axes."""
        group = file["Path1"]
        self.assertEqual(set(group), {
            "Z_depths", "sources", "frame_info", "global",
            "visualizations", "user_rois",
        })
        self.assertEqual(group["Z_depths"].shape, (3, 1))
        self.assertEqual(set(group["sources"]), {"spatial", "temporal"})
        spatial = group["sources/spatial"]
        self.assertEqual(set(spatial), {"profiles", "coords"})
        self.assertEqual(spatial["profiles"].shape, (sources, 3, 4, 5))
        self.assertEqual(spatial["coords"].shape, (sources, 3))
        temporal = group["sources/temporal"]
        self.assertEqual(set(temporal), {
            "dF_ls", "dF_denoised", "events", "F0", "SNR",
        })
        for key in ("dF_ls", "dF_denoised", "events", "F0"):
            self.assertEqual(temporal[key].shape, (sources, 2, frames))
        self.assertEqual(temporal["SNR"].shape, (sources, 1))
        roi_group = group["user_rois"]
        self.assertEqual(set(roi_group), {"mask", "labels", "F", "Fsvd"})
        self.assertEqual(roi_group["mask"].shape, (rois, 3, 4, 5))
        self.assertEqual(roi_group["labels"].shape, (rois, 1))
        for key in ("F", "Fsvd"):
            self.assertEqual(roi_group[key].shape, (rois, 2, frames))
        self.assertEqual(set(group["frame_info"]), set(FRAME_INFO_KEYS))
        self.assertEqual(set(group["global"]), {"F"})
        self.assertEqual(group["global/F"].shape, (2, frames))
        self.assertEqual(set(group["visualizations"]), {
            "mean_im", "act_im", "act_im_peaks",
        })

    def test_supplied_arrays_roundtrip_without_scientific_changes(self):
        """All new and existing arrays preserve values, dtype and orientation."""
        original = _path()
        original.sources[0].df_denoised[0, 1] = np.nan
        original.sources[1].events = original.sources[1].events.astype(np.int16)
        original.user_rois[0].fsvd = original.user_rois[0].fsvd.astype(np.float64)
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "summary.h5"
            rewritten = Path(directory) / "rewritten.h5"
            write_summary(ExperimentSummary(
                paths=[original], params={"operator": "μ", "channels": 2}
            ), destination)
            with h5py.File(destination, "r") as file:
                self.assert_full_schema(file, 2, 2)
                self.assertEqual(file["row_major"][()], 1)
                self.assertTrue(
                    file["Path1/user_rois"].attrs["annotation_enabled"]
                )
                for dataset, field in (
                    ("dF_denoised", "df_denoised"), ("events", "events")
                ):
                    np.testing.assert_array_equal(
                        file[f"Path1/sources/temporal/{dataset}"][()],
                        np.stack([getattr(s, field) for s in original.sources]),
                    )
                np.testing.assert_array_equal(
                    file["Path1/user_rois/Fsvd"][()],
                    np.stack([roi.fsvd for roi in original.user_rois]),
                )
            summary = read_summary(destination)
            loaded = summary.paths[0]
            self.assertTrue(loaded.annotation_enabled)
            self.assertEqual(summary.params["operator"], "μ")
            for before, after in zip(original.sources, loaded.sources):
                for field in (
                    "profile", "coords", "df_ls", "df_denoised",
                    "events", "f0", "snr",
                ):
                    np.testing.assert_array_equal(
                        getattr(after, field), getattr(before, field)
                    )
            for before, after in zip(original.user_rois, loaded.user_rois):
                self.assertEqual(after.label, before.label)
                for field in ("mask", "f", "fsvd"):
                    np.testing.assert_array_equal(
                        getattr(after, field), getattr(before, field)
                    )
            for key in FRAME_INFO_KEYS:
                np.testing.assert_array_equal(
                    getattr(loaded.frame_info, key),
                    getattr(original.frame_info, key),
                )
            for field in ("mean_im", "act_im", "act_im_peaks"):
                np.testing.assert_array_equal(
                    getattr(loaded.visualizations, field),
                    getattr(original.visualizations, field),
                )
            np.testing.assert_array_equal(loaded.global_f, original.global_f)
            np.testing.assert_array_equal(loaded.z_depths, original.z_depths)
            write_summary(summary, rewritten)
            before, after = _snapshot(destination), _snapshot(rewritten)
            self.assertEqual(before.keys(), after.keys())
            for key in before:
                np.testing.assert_equal(after[key], before[key])

    def test_missing_optional_arrays_are_nan_not_copies_or_zeros(self):
        """Mixed missing/present traces are handled independently per entry."""
        for integer_reference in (False, True):
            with self.subTest(integer_reference=integer_reference):
                original = _path()
                original.sources[0].df_denoised = None
                original.sources[1].events = None
                original.sources[0].snr = None
                original.user_rois[0].fsvd = None
                if integer_reference:
                    for source in original.sources:
                        source.df_ls = source.df_ls.astype(np.int16)
                    for roi in original.user_rois:
                        roi.f = roi.f.astype(np.int16)
                with TemporaryDirectory() as directory:
                    destination = Path(directory) / "summary.h5"
                    write_summary(ExperimentSummary(paths=[original]), destination)
                    loaded = read_summary(destination).paths[0]
                    for missing, reference in (
                        (loaded.sources[0].df_denoised, loaded.sources[0].df_ls),
                        (loaded.sources[1].events, loaded.sources[1].df_ls),
                        (loaded.user_rois[0].fsvd, loaded.user_rois[0].f),
                    ):
                        self.assertEqual(missing.shape, reference.shape)
                        self.assertTrue(np.isnan(missing).all())
                    self.assertTrue(np.isnan(loaded.sources[0].snr))
                    np.testing.assert_array_equal(
                        loaded.sources[1].df_denoised,
                        original.sources[1].df_denoised,
                    )
                    np.testing.assert_array_equal(
                        loaded.sources[0].events, original.sources[0].events
                    )
                    np.testing.assert_array_equal(
                        loaded.user_rois[1].fsvd, original.user_rois[1].fsvd
                    )
                self.assertIsNone(original.sources[0].df_denoised)
                self.assertIsNone(original.sources[1].events)
                self.assertIsNone(original.user_rois[0].fsvd)

    def test_empty_sources_and_rois_always_have_full_rank(self):
        """Geometry inference and legacy templates work for either ROI state."""
        for enabled in (False, True):
            for templates in (False, True):
                for frames in (0, 7):
                    with self.subTest(
                        enabled=enabled, templates=templates, frames=frames
                    ), TemporaryDirectory() as directory:
                        original = _path(0, 0, enabled, frames)
                        if templates:
                            original._empty_source_arrays = {
                                "profiles": np.empty((0, 3, 4, 5), np.float64),
                                "coords": np.empty((0, 3), np.int16),
                                "dF_ls": np.empty((0, 2, frames), np.float32),
                                "F0": np.empty((0, 2, frames), np.float32),
                                "SNR": np.empty((0, 1), np.float32),
                            }
                            original._empty_roi_arrays = {
                                "mask": np.empty((0, 3, 4, 5), bool),
                                "F": np.empty((0, 2, frames), np.float64),
                            }
                        destination = Path(directory) / "summary.h5"
                        rewritten = Path(directory) / "rewritten.h5"
                        write_summary(
                            ExperimentSummary(paths=[original]), destination
                        )
                        loaded = read_summary(destination)
                        self.assertEqual(loaded.paths[0].sources, [])
                        self.assertEqual(loaded.paths[0].user_rois, [])
                        self.assertEqual(
                            loaded.paths[0].annotation_enabled, enabled
                        )
                        write_summary(loaded, rewritten)
                        for filename in (destination, rewritten):
                            with h5py.File(filename, "r") as file:
                                self.assert_full_schema(file, 0, 0, frames)
                                self.assertEqual(
                                    bool(file["Path1/user_rois"].attrs[
                                        "annotation_enabled"
                                    ]), enabled,
                                )
                                if templates:
                                    self.assertEqual(
                                        file["Path1/sources/spatial/profiles"].dtype,
                                        np.dtype(np.float64),
                                    )
                                    self.assertEqual(
                                        file["Path1/user_rois/F"].dtype,
                                        np.dtype(np.float64),
                                    )
                        self.assertNotIn(
                            "dF_denoised", original._empty_source_arrays
                        )
                        self.assertNotIn("Fsvd", original._empty_roi_arrays)

    def test_legacy_missing_datasets_and_annotation_metadata(self):
        """Old absent/present ROI groups preserve their original semantics."""
        for sources in (0, 2):
            for state in ("disabled", "empty", "populated"):
                with self.subTest(sources=sources, state=state):
                    rois = 2 if state == "populated" else 0
                    original = _path(sources, rois, state != "disabled")
                    with TemporaryDirectory() as directory:
                        destination = Path(directory) / "legacy.h5"
                        rewritten = Path(directory) / "upgraded.h5"
                        write_summary(
                            ExperimentSummary(paths=[original]), destination
                        )
                        with h5py.File(destination, "a") as file:
                            for key in ("dF_denoised", "events"):
                                del file[f"Path1/sources/temporal/{key}"]
                            if state == "disabled":
                                del file["Path1/user_rois"]
                            else:
                                del file["Path1/user_rois/Fsvd"]
                                del file["Path1/user_rois"].attrs[
                                    "annotation_enabled"
                                ]
                        summary = read_summary(destination)
                        loaded = summary.paths[0]
                        self.assertEqual(
                            loaded.annotation_enabled, state != "disabled"
                        )
                        for source in loaded.sources:
                            self.assertIsNone(source.df_denoised)
                            self.assertIsNone(source.events)
                        for roi in loaded.user_rois:
                            self.assertIsNone(roi.fsvd)
                        write_summary(summary, rewritten)
                        with h5py.File(rewritten, "r") as file:
                            self.assert_full_schema(file, sources, rois)
                            for key in ("dF_denoised", "events"):
                                self.assertTrue(np.isnan(
                                    file[f"Path1/sources/temporal/{key}"][()]
                                ).all())
                            self.assertTrue(np.isnan(
                                file["Path1/user_rois/Fsvd"][()]
                            ).all())
                        self.assertEqual(
                            read_summary(rewritten).paths[0].annotation_enabled,
                            state != "disabled",
                        )

    def test_optional_shape_errors_do_not_replace_existing_path(self):
        """No broadcasting, transposing or reshaping supplied optional data."""
        for owner, field in (
            ("Source", "df_denoised"), ("Source", "events"),
            ("UserRoi", "fsvd"),
        ):
            for shape in ((7,), (1, 7), (2, 6), (7, 2), (2, 7, 1)):
                with self.subTest(owner=owner, field=field, shape=shape):
                    original = _path()
                    unrelated = _path(0, 0, False)
                    unrelated.path_index = 4
                    with TemporaryDirectory() as directory:
                        destination = Path(directory) / "summary.h5"
                        write_summary(ExperimentSummary(
                            paths=[original, unrelated],
                            params={"operator": "original"},
                        ), destination)
                        with h5py.File(destination, "a") as file:
                            file["Path1"].attrs["marker"] = "retain"
                            file["params"].attrs["owner"] = "original"
                            file["unrelated/value"] = np.arange(3)
                        before = _snapshot(destination)
                        entries = (
                            original.sources if owner == "Source"
                            else original.user_rois
                        )
                        setattr(entries[1], field, np.ones(shape))
                        with self.assertRaisesRegex(
                            ValueError, rf"{owner}\.{field}\[1\].*shape"
                        ):
                            write_summary(ExperimentSummary(
                                paths=[original], params={"operator": "new"}
                            ), destination)
                        after = _snapshot(destination)
                        self.assertEqual(after.keys(), before.keys())
                        for key in before:
                            np.testing.assert_equal(after[key], before[key])

    def test_empty_optional_template_shapes_are_validated(self):
        """Zero-sized optional tensors must still have the right axes."""
        for key in ("dF_denoised", "events", "Fsvd"):
            with self.subTest(key=key), TemporaryDirectory() as directory:
                destination = Path(directory) / "summary.h5"
                write_summary(
                    ExperimentSummary(paths=[_path(0, 0)]), destination
                )
                original = read_summary(destination).paths[0]
                before = _snapshot(destination)
                templates = (
                    original._empty_roi_arrays if key == "Fsvd"
                    else original._empty_source_arrays
                )
                templates[key] = np.empty((0, 2, 8))
                with self.assertRaisesRegex(ValueError, rf"{key}.*shape"):
                    write_summary(
                        ExperimentSummary(paths=[original]), destination
                    )
                after = _snapshot(destination)
                self.assertEqual(after.keys(), before.keys())
                for name in before:
                    np.testing.assert_equal(after[name], before[name])

    def test_z_depths_are_always_column_vectors_without_mutation(self):
        """Accept vector inputs while persisting the required (fastz, 1) rank."""
        for shape in ((3,), (1, 3), (3, 1)):
            with self.subTest(shape=shape), TemporaryDirectory() as directory:
                original = _path()
                original.z_depths = original.z_depths.reshape(shape)
                destination = Path(directory) / "summary.h5"
                write_summary(ExperimentSummary(paths=[original]), destination)
                with h5py.File(destination, "r") as file:
                    np.testing.assert_array_equal(
                        file["Path1/Z_depths"][()], [[2], [4], [8]]
                    )
                    self.assertEqual(file["Path1/Z_depths"].dtype, np.int16)
                self.assertEqual(original.z_depths.shape, shape)


if __name__ == "__main__":
    unittest.main()