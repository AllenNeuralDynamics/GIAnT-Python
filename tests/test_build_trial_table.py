"""Tests for the build_trial_table module."""

import unittest

from giant_python.build_trial_table import build_trial_table


class TestBuildTrialTable(unittest.TestCase):
    """Tests for build_trial_table."""

    def test_not_implemented(self):
        """Test that build_trial_table raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            build_trial_table()


if __name__ == "__main__":
    unittest.main()
