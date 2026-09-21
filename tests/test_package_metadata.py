"""Tests for package metadata declarations."""

import ast
import unittest
from importlib import import_module
from importlib.resources import files
from importlib.util import find_spec
from pathlib import Path

from setuptools import find_packages

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
PACKAGE_INIT = SOURCE_ROOT / "giant_python" / "__init__.py"


class PackageMetadataTest(unittest.TestCase):
    """Tests for package metadata declarations."""

    def test_canonical_packages_are_discoverable(self):
        """Source discovery includes canonical owners, not retired packages."""
        packages = set(find_packages(where=str(SOURCE_ROOT)))
        self.assertTrue(
            {
                "giant_python",
                "giant_python.extraction",
                "giant_python.extraction.band",
                "giant_python.numerics",
            }.issubset(packages)
        )
        for package in packages:
            with self.subTest(package=package):
                self.assertFalse(
                    package == "giant_python.bandsilo"
                    or package.startswith("giant_python.bandsilo.")
                    or package == "giant_python.math"
                    or package.startswith("giant_python.math.")
                )

    def test_summary_modules_have_explicit_names(self):
        """Summary owners keep public exports without old module shims."""
        import giant_python

        for owner, symbols in (
            ("models", ("ExperimentSummary", "PathSummary", "FrameInfo")),
            ("io", ("read_summary", "write_summary")),
        ):
            package = import_module(f"giant_python.{owner}")
            module = import_module(f"giant_python.{owner}.experiment_summary")
            for name in symbols:
                with self.subTest(owner=owner, symbol=name):
                    symbol = getattr(module, name)
                    self.assertEqual(symbol.__module__, module.__name__)
                    self.assertIs(getattr(package, name), symbol)
                    if owner == "models":
                        self.assertIs(getattr(giant_python, name), symbol)
            self.assertIsNone(find_spec(f"giant_python.{owner}.experiment"))
            old_path = SOURCE_ROOT / "giant_python" / owner / "experiment.py"
            self.assertFalse(old_path.exists())

    def test_band_and_gui_modules_have_explicit_names(self):
        """Renamed modules own their functions without forwarding shims."""
        renames = {
            "giant_python.extraction.band.results": (
                "result_assembly",
                "assemble_path_summary",
            ),
            "giant_python.extraction.band.motion": (
                "motion_binning",
                "bin_motion",
            ),
            "giant_python.gui.parameters": (
                "parameters_gui",
                "run_parameter_gui",
            ),
        }
        for old_name, (new_name, symbol) in renames.items():
            with self.subTest(module=new_name):
                parent = old_name.rsplit(".", 1)[0]
                module = import_module(f"{parent}.{new_name}")
                self.assertEqual(
                    getattr(module, symbol).__module__, module.__name__
                )
                self.assertIsNone(find_spec(old_name))
                old_path = SOURCE_ROOT.joinpath(
                    *old_name.split(".")
                ).with_suffix(".py")
                self.assertFalse(old_path.exists())

    def test_psf_assets_are_available_as_package_resources(self):
        """Every bundled dilation template is readable without a build."""
        assets = files("giant_python").joinpath("assets", "psfs")
        dilations = (*range(5, 32, 2), 41, 51)
        for dilation in dilations:
            name = f"dil-{dilation:02d}.tif"
            with self.subTest(asset=name):
                resource = assets.joinpath(name)
                self.assertTrue(resource.is_file())
                with resource.open("rb") as stream:
                    self.assertIn(
                        stream.read(4),
                        (
                            b"II\x2a\x00",
                            b"MM\x00\x2a",
                            b"II\x2b\x00",
                            b"MM\x00\x2b",
                        ),
                    )

    def test_init_module_is_valid_and_has_single_version_line(self):
        """Only one ``__version__`` line may exist so CI sed bumps stay safe.

        The shared AIND ``release-bump-version`` workflow rewrites every line
        that contains ``__version__`` with::

            sed 's|__version__.*|__version__ = "X.Y.Z"|'

        Listing ``"__version__"`` in ``__all__`` therefore produces invalid
        Python on the next bump. Keep a single assignment line instead.
        """
        source = PACKAGE_INIT.read_text(encoding="utf-8")
        module = ast.parse(source)

        version_lines = [
            line for line in source.splitlines() if "__version__" in line
        ]
        self.assertEqual(len(version_lines), 1)
        self.assertRegex(
            version_lines[0].strip(),
            r'^__version__ = "[0-9]+\.[0-9]+\.[0-9]+"$',
        )

        version_assignment = next(
            node
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__version__"
                for target in node.targets
            )
        )
        self.assertIsInstance(version_assignment.value, ast.Constant)
        self.assertIsInstance(version_assignment.value.value, str)
        self.assertRegex(
            version_assignment.value.value, r"^[0-9]+\.[0-9]+\.[0-9]+$"
        )


if __name__ == "__main__":
    unittest.main()
