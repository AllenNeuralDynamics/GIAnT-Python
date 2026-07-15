"""Tests for the parallel helper and the CLI entry point."""

import unittest

from giant_python.cli import main
from giant_python.parallel import map_trials


class TestMapTrials(unittest.TestCase):
    """Tests for the per-trial parallelism helper."""

    def test_not_implemented(self):
        """map_trials raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            map_trials(lambda x: x, [1, 2, 3])


class TestCli(unittest.TestCase):
    """Tests for the CLI entry point."""

    def test_not_implemented(self):
        """main raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            main([])


if __name__ == "__main__":
    unittest.main()
