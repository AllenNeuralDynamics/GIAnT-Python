"""Phase-matched background medians, including overlapping band columns."""

import unittest

import numpy as np

from giant_python.bandsilo import summary_images as si


def brute_median(image, selected, ref_r, ref_c, ref_d, height, width,
                 tolerance, minimum):
    """Independent small-grid reference using explicit distance/window loops."""
    phases = np.full(image.shape, np.nan)
    med = np.full(image.shape, np.nan)
    for flat in selected:
        z, r, c = np.unravel_index(flat, image.shape)
        centers = np.unique(ref_r[(ref_d == z) & (ref_c == c)])
        if not centers.size:
            continue
        distances = np.abs(r - centers)
        phases[z, r, c] = r - centers[np.argmin(distances)]
    for flat in selected:
        z, r, c = np.unravel_index(flat, image.shape)
        if not np.isfinite(phases[z, r, c]) or not np.isfinite(image[z, r, c]):
            continue
        samples = []
        for sr in range(max(0, r - height // 2), min(image.shape[1], r + height // 2 + 1)):
            for sc in range(max(0, c - width // 2), min(image.shape[2], c + width // 2 + 1)):
                if abs(phases[z, sr, sc] - phases[z, r, c]) <= tolerance and np.isfinite(image[z, sr, sc]):
                    samples.append(image[z, sr, sc])
        if len(samples) >= minimum:
            med[z, r, c] = np.median(samples)
    first_med = med.copy()
    for flat in selected:
        z, r, c = np.unravel_index(flat, image.shape)
        if (not np.isfinite(phases[z, r, c]) or not np.isfinite(image[z, r, c])
                or np.isfinite(first_med[z, r, c])):
            continue
        window = first_med[z, max(0, r - height // 2):r + height // 2 + 1,
                           max(0, c - width // 2):c + width // 2 + 1]
        finite = window[np.isfinite(window)]
        if finite.size:
            med[z, r, c] = np.mean(finite)
    return med


class TestPhaseMatchedFilter(unittest.TestCase):
    def test_overlapping_curved_bands_match_brute_force(self):
        rng = np.random.default_rng(8)
        image = rng.normal(size=(2, 23, 17)).astype(np.float32)
        # Independent, nonparallel trajectories with overlapping column ranges.
        cols = np.arange(17)
        ref_c = np.tile(cols, 4)
        ref_r = np.r_[4 + abs(cols - 8) // 2, 16 - abs(cols - 8) // 2,
                      3 + cols // 3, 19 - cols // 3].astype(float)
        ref_d = np.repeat([0, 1], 34)
        keep = ref_c != 7  # Column gap is not interpolated.
        ref_r, ref_c, ref_d = ref_r[keep], ref_c[keep], ref_d[keep]
        selected = np.flatnonzero(rng.random(image.size) > 0.15)
        image.ravel()[selected[::29]] = np.nan
        image.ravel()[selected[::41]] = np.inf
        original = image.copy()
        expected = brute_median(image, selected, ref_r, ref_c, ref_d, 9, 7, 0, 3)
        for chunk_size in (1, 53, 4096):
            actual = si.phase_matched_nanmedian(
                image, selected[::-1], ref_r, ref_c, ref_d,
                height=9, width=7, min_samples=3,
                chunk_size=chunk_size,
            )
            np.testing.assert_allclose(actual, expected, atol=1e-6, equal_nan=True)
            self.assertTrue(np.isnan(actual[:, :, 7]).all())
        np.testing.assert_array_equal(image, original)

    def test_signed_phase_not_absolute_distance(self):
        image = np.zeros((1, 13, 11), dtype=np.float32)
        cols = np.arange(11)
        centers = np.full(11, 6)
        image[0, 5, :] = 10  # phase -1
        image[0, 7, :] = 100  # phase +1
        selected = np.arange(image.size)
        med = si.phase_matched_nanmedian(image, selected, centers, cols)
        self.assertEqual(med[0, 5, 5], 10)
        self.assertEqual(med[0, 7, 5], 100)

    def test_phase_tolerance_and_minimum_samples(self):
        image = np.arange(5 * 7, dtype=float).reshape(1, 5, 7)
        cols = np.arange(7)
        centers = 2 + 0.25 * (cols % 2)
        selected = np.arange(image.size)
        for tolerance in (0, 0.3, 1.0):
            for minimum in (1, 5, 100):
                truth = brute_median(image, selected, centers, cols, np.zeros(7),
                                     5, 7, tolerance, minimum)
                result = si.phase_matched_nanmedian(
                    image, selected, centers, cols, height=5, width=7,
                    phase_tolerance=tolerance, min_samples=minimum,
                )
                np.testing.assert_allclose(result, truth, equal_nan=True)

    def test_midpoint_rows_and_duplicate_centers_are_retained(self):
        image = np.ones((1, 16, 11))
        cols = np.tile(np.arange(11), 3)
        centers = np.repeat([4, 4, 12], 11)  # Deduplicate the top band.
        selected = np.arange(image.size)
        med = si.phase_matched_nanmedian(image, selected, centers, cols)
        np.testing.assert_array_equal(med[0, 8], 1)  # Equidistant bands.
        self.assertTrue(np.isfinite(med[0, 4]).all())
        centers = np.repeat([4, 4, 13], 11)
        med = si.phase_matched_nanmedian(image, selected, centers, cols)
        np.testing.assert_array_equal(med[0, 8:10], 1)  # Near ties also retained.

    def test_midpoint_tie_chooses_smaller_reference_row(self):
        image = np.array([[[10., 20., 30., 40., 50., 100., 200.]]])
        cols = np.arange(7)
        # First five pixels are ties with signed phase +4; last two are -4.
        lower = np.r_[np.full(5, -4), np.full(2, -14)]
        upper = np.full(7, 4)
        selected = np.arange(image.size)
        med = si.phase_matched_nanmedian(
            image, selected, np.r_[lower, upper], np.tile(cols, 2),
        )
        self.assertEqual(med[0, 0, 3], 30)

    def test_default_exact_phase_does_not_pool_nearby_offsets(self):
        image = np.full((1, 4, 11), np.nan)
        cols = np.arange(11)
        centers = np.where(cols % 2, 1.25, 1.0)
        image[0, 2, :] = np.where(cols % 2, 100, 10)
        selected = np.flatnonzero(np.isfinite(image))
        med = si.phase_matched_nanmedian(image, selected, centers, cols)
        self.assertEqual(med[0, 2, 5], 100)  # Five exact +0.75 matches.
        self.assertEqual(med[0, 2, 4], 10)  # Five exact +1 matches.
        tolerant = si.phase_matched_nanmedian(
            image, selected, centers, cols, phase_tolerance=0.5,
        )
        self.assertEqual(tolerant[0, 2, 5], 10)

    def test_irregular_spacing_keeps_pixel_offsets_not_fractional_phase(self):
        image = np.full((1, 9, 7), np.nan)
        cols = np.arange(7)
        gaps = np.array([8, 12, 8, 12, 8, 12, 8])
        image[0, 3, :] = np.arange(1, 8)  # +1 in every column.
        refs = np.r_[np.full(7, 2), 2 + gaps]
        med = si.phase_matched_nanmedian(
            image, np.flatnonzero(np.isfinite(image)), refs, np.tile(cols, 2),
        )
        self.assertEqual(med[0, 3, 3], 4)

    def test_fallback_uses_first_pass_medians_not_raw_or_filled_values(self):
        image = np.full((1, 3, 11), np.nan, dtype=np.float32)
        image[0, 0] = np.arange(11) ** 2
        image[0, 1:3, 5] = 9999
        selected = np.flatnonzero(np.isfinite(image))
        cols = np.arange(11)
        expected_med = []
        for col in cols:
            samples = image[0, 0, max(0, col - 5):col + 6]
            median = np.median(samples)
            expected_med.append(median)
        for chunk in (1, 4, 4096):
            med = si.phase_matched_nanmedian(
                image, selected[::-1], np.zeros(11), cols,
                height=3, chunk_size=chunk,
            )
            np.testing.assert_allclose(med[0, 0], expected_med)
            self.assertAlmostEqual(float(med[0, 1, 5]), np.mean(expected_med), places=5)
            # Filled row 1 cannot propagate into row 2, even in later chunks.
            self.assertTrue(np.isnan(med[0, 2, 5]))
            self.assertTrue(np.isnan(med[~np.isfinite(image)]).all())
            ordered = si.phase_matched_nanmedian(
                image, selected, np.zeros(11), cols, height=3, chunk_size=chunk,
            )
            np.testing.assert_allclose(ordered, med, equal_nan=True)
        out = si.finalize_activity_image(
            image.copy(), selected, np.zeros(selected.size),
            np.zeros(11), cols, phase_window_height=3,
        )
        expected = 9999 - np.mean(expected_med)
        np.testing.assert_allclose(out[0, 1, 5], expected, rtol=1e-6)
        self.assertEqual(out[0, 2, 5], 0)  # Vertical window hits image edge.

    def test_empty_fallback_is_nan_without_warnings(self):
        import warnings

        image = np.ones((1, 3, 3))
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            med = si.phase_matched_nanmedian(
                image, np.arange(image.size), np.ones(3), np.arange(3),
            )
        self.assertTrue(np.isnan(med).all())

    def test_missing_depth_geometry_and_empty_support(self):
        image = np.ones((2, 9, 11))
        cols, centers = np.arange(11), np.full(11, 4)
        selected = np.arange(image.size)
        with self.assertRaisesRegex(ValueError, "ref_d"):
            si.phase_matched_nanmedian(image, selected, centers, cols)
        med = si.phase_matched_nanmedian(
            image, selected, centers, cols, np.zeros(11)
        )
        self.assertTrue(np.isfinite(med[0]).all())
        self.assertTrue(np.isnan(med[1]).all())
        for valid in (np.array([], dtype=int), selected):
            med = si.phase_matched_nanmedian(
                image * np.nan, valid, centers, cols, np.zeros(11)
            )
            self.assertTrue(np.isnan(med).all())

    def test_finalization_masks_samples_and_preserves_motion_geometry(self):
        cols = np.arange(3, 14)
        centers = 5 + abs(cols - 8) // 2
        image = np.arange(20 * 20, dtype=float).reshape(1, 20, 20)
        selected = np.arange(image.size)
        nan_ct = np.zeros(image.size)
        nan_ct[::7] = 1
        dr, dc = 2, -1
        image_masked = image.copy()
        image_masked.ravel()[nan_ct > 0.5] = np.nan
        med = si.phase_matched_nanmedian(
            image_masked, selected[nan_ct <= 0.5], centers + dr, cols + dc,
        )
        expected = image_masked - med
        support = np.isfinite(image_masked)
        for z, r, c in zip(*np.where(support)):
            if (r < 3 or r + 3 >= image.shape[1]
                    or not support[z, r - 3:r + 4, c].all()):
                expected[z, r, c] = 0
        actual = si.finalize_activity_image(
            image.copy(), selected, nan_ct, centers + dr, cols + dc,
        )
        np.testing.assert_allclose(actual, expected, equal_nan=True)

    def test_invalid_parameters(self):
        image = np.ones((1, 5, 7))
        for options in (
            {"height": 4}, {"width": 0}, {"chunk_size": 0},
            {"min_samples": 0}, {"phase_tolerance": -1},
            {"phase_tolerance": np.inf}, {"phase_tolerance": np.nan},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                si.phase_matched_nanmedian(
                    image, np.arange(image.size), np.full(7, 2), np.arange(7),
                    **options,
                )


if __name__ == "__main__":
    unittest.main()