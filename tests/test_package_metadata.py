"""Tests for package metadata declarations."""

import ast
import unittest
from pathlib import Path


PACKAGE_INIT = (
    Path(__file__).resolve().parents[1] / "src" / "giant_python" / "__init__.py"
)


class PackageMetadataTest(unittest.TestCase):
    """Tests for package metadata declarations."""

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
