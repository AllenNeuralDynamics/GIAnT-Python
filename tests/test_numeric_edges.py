"""Value-based regressions for previously uncovered numerical edge paths."""

import unittest
from unittest.mock import patch

import numpy as np

from giant_python.extraction.band import background, operators, summary_images
from giant_python.numerics import peaks


class TestOperatorEdges(unittest.TestCase):
    """Protect normalization, centering, and nonnegative kernel weights."""

    def test_identity_shrink_preserves_normalized_psf(self):
        """Unit scales preserve a positive PSF up to sum normalization."""
        psf = np.arange(1, 26, dtype=np.float32).reshape(5, 5)
        original = psf.copy()
        actual = operators.shrink_psf(psf, scale_y=1, scale_x=1)
        np.testing.assert_allclose(actual, psf / psf.sum(), rtol=1e-6)
        np.testing.assert_array_equal(psf, original)
        self.assertEqual(actual.dtype, np.float32)

    def test_shrink_narrows_only_requested_axis(self):
        """An x-only shrink narrows the x moment without moving the center."""
        coords = np.arange(-6, 7, dtype=float)
        psf = np.exp(-(coords[:, None] ** 2 + coords[None, :] ** 2) / 8)
        actual = operators.shrink_psf(psf, scale_y=1, scale_x=0.5)
        original = psf / psf.sum()
        self.assertEqual(actual.shape, psf.shape)
        self.assertTrue(np.all(actual >= 0))
        self.assertAlmostEqual(float(actual.sum()), 1, places=6)
        np.testing.assert_allclose(actual, actual[::-1, ::-1], atol=1e-7)
        self.assertLess(
            np.sum(actual * coords[None, :] ** 2),
            np.sum(original * coords[None, :] ** 2),
        )
        np.testing.assert_allclose(
            actual.sum(axis=1), original.sum(axis=1), atol=1e-6
        )

    def test_nonpositive_psf_remains_zero_without_division(self):
        """Clip negative weights and avoid normalizing zero total mass."""
        for value in (0, -1):
            with self.subTest(value=value):
                actual = operators.shrink_psf(np.full((5, 5), value))
                np.testing.assert_array_equal(actual, np.zeros((5, 5)))
                self.assertEqual(actual.dtype, np.float32)

    def test_gaussian_kernel_matches_separable_weights(self):
        """Use ceil-truncated support and a centered unit-mass Gaussian."""
        sigma, truncate = 0.8, 2.0
        kernel, center = operators.gaussian_kernel_2d(sigma, truncate)
        self.assertEqual(center, (2, 2))
        actual = kernel.numpy()
        coords = np.arange(-2, 3)
        weights = np.exp(-(coords**2) / (2 * sigma**2))
        weights /= weights.sum()
        np.testing.assert_allclose(
            actual, np.outer(weights, weights), rtol=1e-6
        )
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(
            np.unravel_index(actual.argmax(), actual.shape), center
        )
        self.assertAlmostEqual(float(actual.sum()), 1, places=6)


class TestBackgroundEdges(unittest.TestCase):
    """Check fallback medians, minimal windows, and interpolation rejection."""

    def test_pandas_fallback_ignores_nan_and_preserves_dtype(self):
        """Fallback clips odd windows and preserves empty-window NaNs."""
        data = np.array(
            [[1, np.nan, 3, 100, 5], [np.nan] * 5], dtype=np.float32
        )
        original = data.copy()
        with patch.object(background, "bn", None):
            actual = background.compute_rolling_baseline(data, 3)
        expected = np.array([[1, 2, 51.5, 5, 52.5], [np.nan] * 5])
        np.testing.assert_allclose(actual, expected, equal_nan=True)
        np.testing.assert_array_equal(data, original)
        self.assertEqual(actual.dtype, data.dtype)

    def test_single_frame_window_preserves_samples(self):
        """Nonpositive and unit windows clamp to one without padding."""
        data = np.array([[1, np.nan, 7]], dtype=np.float32)
        for window in (-2, 0, 1):
            with self.subTest(window=window):
                actual = background.compute_rolling_baseline(data, window)
                np.testing.assert_array_equal(actual, data)
                self.assertEqual(actual.dtype, data.dtype)

    def test_internal_interpolation_rejects_unknown_method(self):
        """Reject invalid interpolation modes before mutating the output."""
        out = np.full((1, 1), np.nan, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "linear.*cubic"):
            background._interp_columns(
                out,
                {},
                {},
                {},
                np.array([0]),
                np.ones((1, 1)),
                np.array([0]),
                np.float32,
                method="nearest",
            )
        self.assertTrue(np.isnan(out).all())


class TestMedianGeometryEdges(unittest.TestCase):
    """Reject malformed geometry instead of producing plausible medians."""

    def test_phase_filter_rejects_invalid_geometry(self):
        """Validate image, coordinate, depth, and selected-index contracts."""
        defaults = dict(
            image=np.ones((1, 3, 3)),
            valid_pix_idxs=np.arange(9),
            ref_r=np.array([1]),
            ref_c=np.array([1]),
        )
        cases = [
            ({"image": np.ones((3, 3))}, "non-empty shape"),
            ({"image": np.ones((1, 0, 3))}, "non-empty shape"),
            ({"ref_r": []}, "non-empty and equally sized"),
            ({"ref_c": [0, 1]}, "non-empty and equally sized"),
            ({"ref_r": [np.nan]}, "reference rows must be finite"),
            ({"ref_d": [0, 0]}, "same size"),
            ({"ref_c": [0.5]}, "finite integers"),
            ({"ref_d": [np.inf]}, "finite integers"),
            ({"ref_d": [0.5]}, "finite integers"),
            ({"valid_pix_idxs": [[0]]}, "one-dimensional integer array"),
            ({"valid_pix_idxs": [0.5]}, "one-dimensional integer array"),
            ({"valid_pix_idxs": [-1]}, "out-of-bounds"),
            ({"valid_pix_idxs": [9]}, "out-of-bounds"),
        ]
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    summary_images.phase_matched_nanmedian(
                        **(defaults | overrides)
                    )

    def test_snake_filter_rejects_invalid_geometry(self):
        """Require odd windows and at least one matched in-bounds center."""
        defaults = dict(
            image=np.ones((1, 3, 3)),
            valid_pix_idxs=np.arange(9),
            ref_r=np.array([1]),
            ref_c=np.array([1]),
        )
        cases = [
            ({"height": 0}, "positive odd integers"),
            ({"width": 2}, "positive odd integers"),
            ({"ref_r": []}, "non-empty and equally sized"),
            ({"ref_c": [0, 1]}, "non-empty and equally sized"),
            ({"ref_c": [-1]}, "no reference centers"),
            ({"ref_c": [3]}, "no reference centers"),
        ]
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    summary_images.snake_aligned_nanmedian(
                        **(defaults | overrides)
                    )


class TestPeakBoundsEdges(unittest.TestCase):
    """Protect the shared-width initialization's feasible parameter domain."""

    def test_disjoint_width_bounds_rejected_before_fitting(self):
        """Disjoint y/x intervals cannot initialize an isotropic peak."""
        theta = np.array([[1, 0, 0, 0.5, 2.5]], dtype=float)
        original = theta.copy()
        lower = np.array([0, -1, -1, 0.25, 2])
        upper = np.array([2, 1, 1, 1, 3])
        with patch.object(peaks, "_lsq_curvefit") as fit:
            with self.assertRaisesRegex(ValueError, "overlapping y/x"):
                peaks._fit_isotropic_then_anisotropic(
                    theta, np.zeros((2, 1)), np.zeros(1), lower, upper
                )
            fit.assert_not_called()
        np.testing.assert_array_equal(theta, original)
