"""Tests for giant_python.bandsilo.baseline (Phase 7 F0 / dF-F).

Covers the NaN-aware moving mean, the rolling-min-envelope ``compute_f0``
baseline estimator, and the dF/F assembly. ``compute_f0`` is additionally
cross-checked against a verbatim copy of the reference in the project's
development notes; here we assert structural properties on synthetic traces.
"""

import unittest

import numpy as np

from giant_python.bandsilo import baseline as bl


class TestMovmeanNan(unittest.TestCase):
    """_movmean_nan edge cases."""

    def test_window_one_is_identity(self):
        """A window of 1 returns a float copy of the input."""
        x = np.array([1.0, np.nan, 3.0])
        out = bl._movmean_nan(x, 1)
        np.testing.assert_array_equal(
            np.nan_to_num(out), np.nan_to_num(x.astype(float))
        )

    def test_ignores_nan(self):
        """NaNs are excluded from the mean, not treated as zero."""
        x = np.array([2.0, np.nan, 4.0])
        out = bl._movmean_nan(x, 3)
        # center window over index 1 sees {2, 4} -> mean 3
        self.assertAlmostEqual(out[1], 3.0)

    def test_all_nan_yields_nan(self):
        """A window with no valid samples yields NaN (den == 0 branch)."""
        x = np.full(5, np.nan)
        out = bl._movmean_nan(x, 3)
        self.assertTrue(np.all(np.isnan(out)))


class TestComputeF0(unittest.TestCase):
    """compute_f0 shape handling and baseline behavior."""

    def _drifting_trace(self, total=200, seed=0):
        """A drifting baseline plus positive transients and noise."""
        rng = np.random.default_rng(seed)
        t = np.arange(total)
        base = 5 + 2 * np.sin(t / 40)
        transients = np.clip(rng.standard_normal(total), 0, None) * 3
        return (base + transients + rng.standard_normal(total) * 0.2).astype(
            float
        )

    def test_tracks_below_signal(self):
        """F0 stays at or below the (transient-carrying) signal."""
        f = self._drifting_trace()
        f0 = bl.compute_f0(f, 9, 40)
        self.assertEqual(f0.shape, f.shape)
        # baseline should sit near the lower envelope, well below the mean
        self.assertLess(np.nanmean(f0), np.nanmean(f))

    def test_two_dimensional(self):
        """A 2-D input is handled column-wise and keeps its shape."""
        f = np.stack(
            [self._drifting_trace(seed=1), self._drifting_trace(seed=2)],
            axis=1,
        )
        f[10:15, 0] = np.nan
        f0 = bl.compute_f0(f, 9, 40)
        self.assertEqual(f0.shape, f.shape)

    def test_all_nan_column_skipped(self):
        """An all-NaN column is returned unchanged (all NaN)."""
        f = np.stack(
            [self._drifting_trace(seed=3), np.full(200, np.nan)], axis=1
        )
        f0 = bl.compute_f0(f, 9, 40)
        self.assertTrue(np.all(np.isnan(f0[:, 1])))
        self.assertFalse(np.all(np.isnan(f0[:, 0])))

    def test_short_input_returns_mean(self):
        """Fewer than 4 samples returns the broadcast column mean."""
        f = np.array([[2.0, 8.0], [4.0, 10.0], [6.0, 12.0]])
        f0 = bl.compute_f0(f, 9, 40)
        np.testing.assert_allclose(f0[:, 0], 4.0)
        np.testing.assert_allclose(f0[:, 1], 10.0)

    def test_sparse_hull_triggers_gap_interpolation(self):
        """A trace with NaN gaps exercises the residual-NaN interpolation."""
        f = self._drifting_trace(total=120, seed=4)
        f[30:70] = np.nan  # a wide gap forces envelope NaNs to be filled
        f0 = bl.compute_f0(f, 15, 80)
        self.assertEqual(f0.shape, f.shape)
        self.assertFalse(np.all(np.isnan(f0)))


class TestAssembleDff(unittest.TestCase):
    """assemble_dff combines the least-squares traces into dF/F."""

    def test_relationships(self):
        """F = dF_ls + F0_ls; dF = F - F0; dFF = dF / clip(F0)."""
        rng = np.random.default_rng(0)
        d_f_ls = np.clip(rng.standard_normal((150, 2)), 0, None) * 2
        f0_ls = np.full((150, 2), 5.0) + rng.standard_normal((150, 2)) * 0.1
        f, f0, d_f, d_ff = bl.assemble_dff(d_f_ls, f0_ls, 9, 40)
        np.testing.assert_allclose(f, d_f_ls + f0_ls)
        np.testing.assert_allclose(d_f, f - f0)
        np.testing.assert_allclose(d_ff, d_f / np.clip(f0, 1e-4, None))


if __name__ == "__main__":
    unittest.main()
