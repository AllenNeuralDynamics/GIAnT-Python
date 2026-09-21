"""Guard the band acquisition boundary without changing numerical policy."""

import ast
import inspect
import tempfile
import unittest
from importlib import resources
from pathlib import Path
from unittest.mock import call, patch

import h5py
import numpy as np
import tifffile

from giant_python.extraction.band import annotation, geometry, inputs, workflow
from giant_python.io import slap2
from giant_python.io.tiff import read_tiff


class TestAcquisitionOwnership(unittest.TestCase):
    """File loading cannot leak back into geometry or duplicate raw IO APIs."""

    def test_geometry_imports_only_numerical_support(self):
        """Inspect all imports, including deferred imports inside functions."""
        tree = ast.parse(inspect.getsource(geometry))
        allowed = {"__future__", "typing", "numpy", "scipy"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertIn(alias.name.split(".")[0], allowed)
            elif isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0)
                self.assertIn((node.module or "").split(".")[0], allowed)
            elif isinstance(node, ast.Call) and isinstance(
                node.func, ast.Name
            ):
                self.assertNotIn(node.func.id, {"open", "__import__"})

    def test_loaders_have_one_owner(self):
        """Workflows use input loaders, not numerical geometry or raw IO."""
        for name in (
            "load_lookup_table",
            "_ref_stack_group",
            "find_reference_file",
            "load_reference_stack",
            "default_psf",
            "load_psf",
        ):
            with self.subTest(name=name):
                function = getattr(inputs, name)
                self.assertEqual(function.__module__, inputs.__name__)
                self.assertFalse(hasattr(geometry, name))
                self.assertFalse(hasattr(slap2, name))
        self.assertIs(workflow.inputs, inputs)
        self.assertIs(annotation.inputs, inputs)
        self.assertIs(inputs.read_tiff, read_tiff)
        self.assertIs(inputs.build_combined_psf, geometry.build_combined_psf)
        self.assertIs(
            inputs.threshold_and_crop_psf, geometry.threshold_and_crop_psf
        )

    def test_reader_protocol_method_is_docstring_only(self):
        """The reader contract has no executable placeholder statement."""
        tree = ast.parse(inspect.getsource(slap2.Slap2Reader))
        method = next(
            node
            for node in tree.body[0].body
            if isinstance(node, ast.FunctionDef) and node.name == "getLineData"
        )
        self.assertEqual(len(method.body), 1)
        self.assertIsNotNone(ast.get_docstring(method))
        self.assertIsNone(method.returns)


class TestBandInputLoading(unittest.TestCase):
    """Preserve lookup layouts, reference page order and bundled PSF values."""

    def test_lookup_aliases_layouts_and_int32_columns(self):
        """Path groups take precedence; DMD groups and MATLAB layout work."""
        expected = {
            "allSuperPixelIDs": np.array([[7], [3], [11]], dtype=np.int32),
            "sparseMaskInds": np.array(
                [[19, 1], [5, 2], [27, 3]], dtype=np.int32
            ),
            "fastZ2RefZ": np.array([[4], [2]], dtype=np.int32),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lookup.h5"
            for layout in (None, 0, 1):
                with self.subTest(row_major=layout):
                    with h5py.File(path, "w") as file:
                        if layout is not None:
                            file.create_dataset("row_major", data=layout)
                        for group in ("Path1", "DMD2"):
                            for key, value in expected.items():
                                file.create_dataset(
                                    f"{group}/{key}",
                                    data=(value if layout else value.T).astype(
                                        np.float64
                                    ),
                                )
                        file.create_group("DMD1")
                    result = inputs.load_lookup_table(path, 2)
                    self.assertEqual(set(result), set(expected))
                    for key, value in expected.items():
                        self.assertEqual(set(result[key]), {"DMD1", "DMD2"})
                        for actual in result[key].values():
                            self.assertEqual(actual.dtype, np.dtype("int32"))
                            np.testing.assert_array_equal(actual, value)

    def test_reference_tiff_values_shapes_and_dtypes(self):
        """Decoding is raw; band scaling and channel reshape stay local."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample_DMD2-REFERENCE.tif"
            for dtype in (np.uint16, np.float32):
                for group in ("Path2", "DMD2"):
                    with self.subTest(dtype=dtype, group=group):
                        raw = np.arange(120).reshape(6, 4, 5).astype(dtype)
                        if dtype == np.float32:
                            raw[1, 2, 3] = np.nan
                        tifffile.imwrite(path, raw, photometric="minisblack")
                        channels = np.array([[2], [1]], dtype=np.int16)
                        meta = {group: {"channels": channels}}
                        with patch.object(
                            inputs, "read_tiff", wraps=read_tiff
                        ) as reader:
                            stack, actual_channels, filename = (
                                inputs.load_reference_stack(directory, meta, 1)
                            )
                        reader.assert_called_once_with(str(path))
                        decoded = read_tiff(path)
                        self.assertEqual(decoded.dtype, raw.dtype)
                        np.testing.assert_array_equal(decoded, raw)
                        expected = (raw / 100).reshape(3, 2, 4, 5)
                        expected = expected.transpose(1, 0, 2, 3)
                        self.assertEqual(stack.shape, (2, 3, 4, 5))
                        self.assertEqual(stack.dtype, expected.dtype)
                        np.testing.assert_array_equal(stack, expected)
                        np.testing.assert_array_equal(actual_channels, [2, 1])
                        self.assertEqual(actual_channels.dtype, channels.dtype)
                        np.testing.assert_array_equal(channels, [[2], [1]])
                        self.assertEqual(filename, str(path))

    def test_reference_discovery_precedence_and_missing(self):
        """Recursive CONFIG2 wins; alternate and missing cases still work."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fallback = root / "sample_DMD1-REFERENCE.tif"
            preferred = root / "nested" / "sample_DMD1_CONFIG2-REFERENCE.tif"
            preferred.parent.mkdir()
            fallback.touch()
            self.assertEqual(
                inputs.find_reference_file(root, 0), str(fallback)
            )
            preferred.touch()
            self.assertEqual(
                inputs.find_reference_file(root, 0), str(preferred)
            )
            channels = np.array([[3, 1]], dtype=np.uint8)
            with patch.object(inputs, "read_tiff") as reader:
                stack, actual, filename = inputs.load_reference_stack(
                    root, {"DMD2": {"channels": channels}}, 1
                )
            reader.assert_not_called()
            self.assertIsNone(stack)
            self.assertIsNone(filename)
            self.assertEqual(actual.dtype, channels.dtype)
            np.testing.assert_array_equal(actual, [3, 1])

    def test_reference_metadata_prefers_path_group(self):
        """Retain subgroup identity and reject missing reference metadata."""
        primary = {"channels": [2, 1]}
        fallback = {"channels": [3]}
        meta = {"Path1": primary, "DMD1": fallback}
        self.assertIs(inputs._ref_stack_group(meta, 0), primary)
        self.assertIs(inputs._ref_stack_group({"DMD1": fallback}, 0), fallback)
        with self.assertRaisesRegex(
            ValueError, "Missing reference-stack group Path2 or DMD2"
        ):
            inputs._ref_stack_group(meta, 1)

    def test_bundled_psf_preserves_asset_values_and_per_dmd_cropping(self):
        """Preserve the resource path, float32 conversion and crop."""
        resource = resources.files("giant_python").joinpath(
            "assets", "psfs", "dil-17.tif"
        )
        with resources.as_file(resource) as path:
            original = tifffile.imread(str(path)).astype(np.float32)
        with patch.object(inputs, "read_tiff", wraps=read_tiff) as reader:
            template = inputs.default_psf(17)
        self.assertEqual(Path(reader.call_args.args[0]).name, "dil-17.tif")
        reader.assert_called_once()
        self.assertEqual(template.dtype, np.dtype("float32"))
        np.testing.assert_array_equal(template, original)
        expected = original.copy()
        expected[expected < expected.max() * np.exp(-3)] = 0
        rows = np.flatnonzero(np.any(expected != 0, axis=1))
        cols = np.flatnonzero(np.any(expected != 0, axis=0))
        row_slice = slice(rows[0], rows[-1] + 1)
        col_slice = slice(cols[0], cols[-1] + 1)
        expected = expected[row_slice, col_slice]
        with patch.object(
            inputs, "default_psf", wraps=inputs.default_psf
        ) as loader:
            result = inputs.load_psf(17, 2)
        self.assertEqual(loader.call_args_list, [call(17), call(17)])
        self.assertEqual(set(result), {"DMD1", "DMD2"})
        for actual in result.values():
            self.assertEqual(actual.dtype, np.dtype("float32"))
            np.testing.assert_array_equal(actual, expected)
        self.assertFalse(np.shares_memory(result["DMD1"], result["DMD2"]))


if __name__ == "__main__":
    unittest.main()
