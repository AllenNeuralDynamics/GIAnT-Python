"""Tests for the giant_python package."""

import unittest


class PackageTest(unittest.TestCase):
    """Tests for the giant_python package."""

    def test_import(self):
        """Test that the giant_python package can be imported."""
        import giant_python

        self.assertIsNotNone(giant_python.__version__)


if __name__ == "__main__":
    unittest.main()
