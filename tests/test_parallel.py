"""Tests for the per-trial parallelism helper."""

import unittest

from giant_python.parallel import map_trials


def _double(x):
    """Module-level (picklable) worker for the parallel path."""
    return x * 2


class TestMapTrials(unittest.TestCase):
    """Tests for the per-trial parallelism helper."""

    def test_serial(self):
        """n_workers=1 runs serially, preserving order."""
        self.assertEqual(
            map_trials(_double, [1, 2, 3], n_workers=1), [2, 4, 6]
        )

    def test_single_trial_runs_serial(self):
        """A single trial short-circuits to the serial path."""
        self.assertEqual(map_trials(_double, [5], n_workers=4), [10])

    def test_parallel(self):
        """Multiple workers fan out and return results in input order."""
        self.assertEqual(
            map_trials(_double, [1, 2, 3, 4], n_workers=2), [2, 4, 6, 8]
        )


if __name__ == "__main__":
    unittest.main()
