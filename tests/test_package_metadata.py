"""Tests for package metadata declarations."""

import ast
import unittest
from pathlib import Path


PACKAGE_INIT = Path(__file__).resolve().parents[1] / (
    "src/giant_python/__init__.py"
)


class PackageMetadataTest(unittest.TestCase):
    """Tests for package metadata declarations."""

    def test_init_module_is_valid_and_exports_version(self):
        """The package __init__ module should parse and export __version__."""
        module = ast.parse(PACKAGE_INIT.read_text(encoding="utf-8"))

        all_assignment = next(
            node
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            )
        )
        self.assertIsInstance(all_assignment.value, (ast.List, ast.Tuple))
        exported_names = [
            elt.value
            for elt in all_assignment.value.elts
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
        ]

        self.assertIn("__version__", exported_names)


if __name__ == "__main__":
    unittest.main()
