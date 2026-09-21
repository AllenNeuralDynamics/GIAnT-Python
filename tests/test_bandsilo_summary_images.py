"""Tests for giant_python.extraction.band.summary_images (Phase 4).

Exercises the mean image, the spatio-temporal local-maxima activity-image
accumulation, and the median-subtraction finalize on small synthetic inputs.
"""

import unittest
from unittest.mock import patch

import numpy as np

from giant_python.extraction.band import summary_images as si


class TestMeanImage(unittest.TestCase):
    """compute_mean_image single- and dual-channel."""

    def test_single_channel(self):
        """Channel-1 mean lands at the shifted reference pixels."""
        ref_d = np.array([0, 0])
        ref_r = np.array([1, 2])
        ref_c = np.array([1, 3])
        low = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
        unique_motion = np.array([[0, 0, 0]], dtype=float)
        mot_inds = np.array([0, 0])
        frames = np.array([True, True])
        mean_im = si.compute_mean_image(
            low,
            unique_motion,
            mot_inds,
            frames,
            ref_d,
            ref_r,
            ref_c,
            num_fast_zs=1,
            dmd_pixels_per_column=5,
            dmd_pixels_per_row=5,
            num_channels=1,
        )
        self.assertEqual(mean_im.shape, (1, 1, 5, 5))
        self.assertAlmostEqual(float(mean_im[0, 0, 1, 1]), 2.0, places=5)
        self.assertAlmostEqual(float(mean_im[0, 0, 2, 3]), 3.0, places=5)

    def test_dual_channel_fills_second(self):
        """Channel-2 mean is written when num_channels >= 2."""
        ref_d = np.array([0])
        ref_r = np.array([1])
        ref_c = np.array([1])
        low = np.array([[1.0, 3.0]], dtype=np.float32)
        low2 = np.array([[10.0, 30.0]], dtype=np.float32)
        unique_motion = np.array([[0, 0, 0]], dtype=float)
        mot_inds = np.array([0, 0])
        frames = np.array([True, True])
        mean_im = si.compute_mean_image(
            low,
            unique_motion,
            mot_inds,
            frames,
            ref_d,
            ref_r,
            ref_c,
            num_fast_zs=1,
            dmd_pixels_per_column=4,
            dmd_pixels_per_row=4,
            num_channels=2,
            low_res_data2_norm=low2,
        )
        self.assertEqual(mean_im.shape, (2, 1, 4, 4))
        self.assertAlmostEqual(float(mean_im[1, 0, 1, 1]), 20.0, places=5)


class TestAccumulateActivityImage(unittest.TestCase):
    """accumulate_activity_image local-maxima detection and accumulation."""

    def test_single_spike_is_detected(self):
        """One bright voxel exceeding its neighbors accumulates its square."""
        num_fast_zs, npc, npr = 1, 7, 7
        n_pix = npc * npr
        sel_pix_idxs = np.arange(n_pix)
        n_frames = 5
        rho = np.zeros((n_pix, n_frames), dtype=np.float32)
        # spike at center pixel (row 3, col 3), frame 2, value 4 -> 16
        center_flat = 3 * npr + 3
        rho[center_flat, 2] = 4.0
        act_im = si.accumulate_activity_image(
            rho, sel_pix_idxs, num_fast_zs, npc, npr
        )
        self.assertEqual(act_im.shape, (num_fast_zs, npc, npr))
        self.assertAlmostEqual(float(act_im[0, 3, 3]), 16.0, places=4)
        # neighbors did not accumulate
        self.assertAlmostEqual(float(act_im[0, 3, 4]), 0.0, places=5)

    def test_nan_neighbor_suppresses_peak(self):
        """A NaN adjacent to a candidate peak removes it via dilation."""
        num_fast_zs, npc, npr = 1, 7, 7
        n_pix = npc * npr
        sel_pix_idxs = np.arange(n_pix)
        n_frames = 5
        rho = np.zeros((n_pix, n_frames), dtype=np.float32)
        center_flat = 3 * npr + 3
        rho[center_flat, 2] = 4.0
        rho[3 * npr + 4, 2] = np.nan  # NaN right next to the peak
        act_im = si.accumulate_activity_image(
            rho, sel_pix_idxs, num_fast_zs, npc, npr
        )
        self.assertAlmostEqual(float(act_im[0, 3, 3]), 0.0, places=5)

    def test_small_batches_cover_multiple_iterations(self):
        """A tiny batch_size forces multiple batches, same output."""
        num_fast_zs, npc, npr = 1, 7, 7
        n_pix = npc * npr
        sel_pix_idxs = np.arange(n_pix)
        rho = np.zeros((n_pix, 8), dtype=np.float32)
        rho[3 * npr + 3, 4] = 3.0
        big = si.accumulate_activity_image(
            rho, sel_pix_idxs, num_fast_zs, npc, npr, batch_size=1000
        )
        small = si.accumulate_activity_image(
            rho, sel_pix_idxs, num_fast_zs, npc, npr, batch_size=2
        )
        np.testing.assert_allclose(big, small, rtol=1e-5)

    def test_too_few_frames_yields_zero_image(self):
        """A batch with <= 2 frames has no temporal interior and accumulates
        nothing."""
        num_fast_zs, npc, npr = 1, 7, 7
        n_pix = npc * npr
        sel_pix_idxs = np.arange(n_pix)
        rho = np.ones((n_pix, 1), dtype=np.float32)  # single frame
        act_im = si.accumulate_activity_image(
            rho, sel_pix_idxs, num_fast_zs, npc, npr
        )
        self.assertTrue(np.all(act_im == 0.0))


class TestFinalizeActivityImage(unittest.TestCase):
    """Activity-image masking and local median subtraction in raw units."""

    def test_vertical_window_requires_full_valid_support(self):
        """Suppress windows crossing edges or invalid vertical support."""
        image = np.ones((2, 19, 13), dtype=np.float32)
        selected_mask = np.ones(image.shape, dtype=bool)
        selected_mask[0, :4] = False
        selected_mask[0, 15:] = False
        selected = np.flatnonzero(selected_mask)
        nan_ct = np.zeros(selected.size)
        dropped = np.ravel_multi_index((0, 9, 6), image.shape)
        nan_ct[selected == dropped] = 1
        image[1, 9, 7] = np.nan
        image[1, 9, 8] = np.inf
        support = selected_mask & np.isfinite(image)
        support[0, 9, 6] = False
        for height in (1, 3, 7):
            geometry = dict(
                ref_r=np.array([9, 9]),
                ref_c=np.array([6, 6]),
                ref_d=np.array([0, 1]),
                phase_window_height=height,
            )
            # A fixed median isolates final edge suppression.
            with patch.object(
                si,
                "phase_matched_nanmedian",
                return_value=np.zeros_like(image),
            ):
                out = si.finalize_activity_image(
                    image.copy(),
                    selected,
                    nan_ct,
                    **geometry,
                )
            for z, r, c in zip(*np.where(support)):
                lo, hi = r - height // 2, r + height // 2 + 1
                fits = (
                    lo >= 0
                    and hi <= image.shape[1]
                    and support[z, lo:hi, c].all()
                )
                self.assertEqual(out[z, r, c], 1 if fits else 0)
            self.assertTrue(np.isnan(out[~selected_mask]).all())
            self.assertTrue(np.isnan(out[0, 9, 6]))
            self.assertTrue(np.isnan(out[1, 9, 7]))
            self.assertEqual(out[1, 9, 0], 1)  # Horizontal edges survive.

    def test_median_subtraction_preserves_mask(self):
        """Subtract the median without restoring excluded pixels."""
        image = np.ones((1, 13, 17), dtype=np.float32)
        image[0, 6, 8] = 7
        selected = np.arange(1, image.size)
        nan_ct = np.zeros(selected.size)
        nan_ct[-1] = 0.9
        for geometry in (
            {},
            {"ref_r": np.full(17, 6), "ref_c": np.arange(17)},
        ):
            with self.subTest(snake=bool(geometry)):
                out = si.finalize_activity_image(
                    image.copy(),
                    selected,
                    nan_ct,
                    **geometry,
                )
                self.assertEqual(out[0, 6, 8], 6)
                self.assertEqual(out[0, 6, 7], 0)
                self.assertTrue(np.isnan(out.ravel()[0]))
                self.assertTrue(np.isnan(out.ravel()[-1]))
                expected = np.zeros_like(image)
                expected[0, 6, 8] = 6
                expected.ravel()[[0, -1]] = np.nan
                np.testing.assert_allclose(out, expected, equal_nan=True)

    def test_constant_neighborhoods_are_zero_and_spike_is_preserved(self):
        """Subtract constant backgrounds without masking valid neighbors."""
        num_fast_zs, npc, npr = 1, 13, 13
        act_im = np.ones((num_fast_zs, npc, npr), dtype=np.float32)
        # all pixels selected and valid (nan_ct = 0)
        sel_pix_idxs = np.arange(npc * npr)
        nan_ct = np.zeros(sel_pix_idxs.size)
        for geometry in (
            {},
            {"ref_r": np.full(npr, 6), "ref_c": np.arange(npr)},
        ):
            for spike in (0, 10):
                with self.subTest(geometry=bool(geometry), spike=spike):
                    image = act_im.copy()
                    image[0, 6, 6] += spike
                    out = si.finalize_activity_image(
                        image, sel_pix_idxs, nan_ct, **geometry
                    )
                    radius = 3 if geometry else 2
                    self.assertTrue(np.all(out[:, :radius] == 0))
                    self.assertTrue(np.all(out[:, -radius:] == 0))
                    expected = np.zeros_like(act_im)
                    expected[0, 6, 6] = spike
                    np.testing.assert_array_equal(out, expected)

    def test_high_nan_pixels_masked(self):
        """Pixels with nan_ct > 0.5 are excluded and set to NaN."""
        npc, npr = 13, 13
        act_im = np.arange(npc * npr, dtype=np.float32).reshape(1, npc, npr)
        rows, cols = np.meshgrid(
            np.arange(2, 5), np.arange(2, 8), indexing="ij"
        )
        sel_pix_idxs = (rows * npr + cols).ravel()
        nan_ct = np.zeros(sel_pix_idxs.size)
        nan_ct[sel_pix_idxs == 3 * npr + 4] = 0.9
        out = si.finalize_activity_image(act_im, sel_pix_idxs, nan_ct)
        # the dropped pixel and all unselected pixels are NaN
        self.assertTrue(np.isnan(out[0, 3, 4]))
        self.assertTrue(np.isnan(out[0, 0, 0]))
        self.assertFalse(np.isnan(out[0, 3, 3]))

    def test_median_follows_curved_snake(self):
        """The 3x11 footprint follows reference-center row changes."""
        size = 17
        cols = np.arange(size)
        center_rows = np.abs(cols - size // 2) + 3
        act_im = np.zeros((1, size, size), dtype=np.float32)
        for col, row in zip(cols, center_rows):
            act_im[0, row - 1 : row + 2, col] = 10.0

        sel_pix_idxs = np.arange(size * size)
        med = si.snake_aligned_nanmedian(
            act_im,
            sel_pix_idxs,
            ref_r=center_rows,
            ref_c=cols,
            height=3,
        )

        center = size // 2
        self.assertAlmostEqual(float(med[0, 3, center]), 10.0, places=5)

    def test_median_follows_motion_shifted_bend(self):
        """Shifted geometry follows the bend in output-image coordinates."""
        cols = np.arange(8, 25)
        center_rows = np.abs(cols - 16) + 8
        for dr, dc in ((2, 5), (-2, -5), (0, 0)):
            with self.subTest(dr=dr, dc=dc):
                act_im = np.zeros((1, 40, 40), dtype=np.float32)
                for col, row in zip(cols + dc, center_rows + dr):
                    act_im[0, row - 1 : row + 2, col] = 10.0
                sel_pix_idxs = np.arange(act_im.size)
                med = si.snake_aligned_nanmedian(
                    act_im,
                    sel_pix_idxs,
                    ref_r=center_rows + dr,
                    ref_c=cols + dc,
                )
                self.assertAlmostEqual(float(med[0, 8 + dr, 16 + dc]), 10.0)
                if dc:
                    unshifted = si.snake_aligned_nanmedian(
                        act_im,
                        sel_pix_idxs,
                        ref_r=center_rows,
                        ref_c=cols,
                    )
                    self.assertLess(float(unshifted[0, 8 + dr, 16 + dc]), 10.0)

    def test_trajectory_does_not_jump_when_outer_snake_is_missing(self):
        """Changing snake count does not move the shared trajectory."""
        size = 40
        cols = np.arange(11)
        ref_c = np.repeat(cols, 5)
        ref_r = np.tile(np.arange(1, 38, 9), cols.size)
        keep = ~((ref_c >= 5) & (ref_r == 1))

        act_im = np.zeros((1, size, cols.size), dtype=np.float32)
        act_im[0, 18:21, :] = 10.0
        sel_pix_idxs = np.arange(act_im.size)
        med = si.snake_aligned_nanmedian(
            act_im,
            sel_pix_idxs,
            ref_r=ref_r[keep],
            ref_c=ref_c[keep],
        )

        self.assertAlmostEqual(float(med[0, 19, 4]), 10.0, places=5)

    def test_median_subtraction_preserves_raw_units_and_scaling(self):
        """Both paths subtract footprint medians and retain input scaling."""
        image = (
            np.broadcast_to(np.arange(17), (1, 17, 17))
            .astype(np.float32)
            .copy()
        )
        image[0, 8, 8] += 6
        selected = np.arange(image.size)
        for geometry, height in (
            ({}, 5),
            ({"ref_r": np.full(17, 8), "ref_c": np.arange(17)}, 1),
        ):
            with self.subTest(height=height):
                samples = image[0, 8 - height // 2 : 9 + height // 2, 3:14]
                median = np.nanmedian(samples)
                out = si.finalize_activity_image(
                    image.copy(), selected, np.zeros(selected.size), **geometry
                )
                self.assertAlmostEqual(
                    float(out[0, 8, 8]), float(image[0, 8, 8] - median)
                )
                rescaled = si.finalize_activity_image(
                    4 * image + 13,
                    selected,
                    np.zeros(selected.size),
                    **geometry,
                )
                np.testing.assert_allclose(4 * out, rescaled, equal_nan=True)

    def test_median_subtraction_follows_curved_footprint(self):
        """Subtract the phase-matched median at a curved band center."""
        cols = np.arange(17)
        ref_r = np.abs(cols - 8) + 5
        image = np.zeros((1, 25, 17), dtype=np.float32)
        for col, row in zip(cols, ref_r):
            image[0, row - 2 : row + 3, col] = (
                20 + col - 8 + 100 * np.arange(-2, 3)
            )
        image[0, 5, 8] = 25
        out = si.finalize_activity_image(
            image,
            np.arange(image.size),
            np.zeros(image.size),
            ref_r=ref_r,
            ref_c=cols,
        )
        # The default 7-row window reaches center-phase samples in columns
        # 5:12 at this bend. Replacing 20 with 25 gives median=21.
        self.assertAlmostEqual(float(out[0, 5, 8]), 4.0, places=6)


class TestSnakeMedian(unittest.TestCase):
    """Medians use the exact curved, clipped, section-limited footprint."""

    def test_matches_explicit_samples_across_chunks(self):
        """Match clipped trajectory samples independently of chunk size."""
        rng = np.random.default_rng(42)
        image = rng.normal(size=(2, 20, 17)).astype(np.float32)
        image[:, 4, ::3] = np.nan
        cols = np.r_[np.arange(7), np.arange(10, 17)]
        ref_r = np.abs(cols - 8) + 3
        points = [(0, 3, 0), (0, 4, 6), (1, 10, 10), (1, 19, 16)]
        selected = np.ravel_multi_index(np.array(points).T, image.shape)
        original = image.copy()
        for height in (3, 5):
            med = si.snake_aligned_nanmedian(
                image,
                selected,
                ref_r,
                cols,
                height=height,
                chunk_size=2,
            )
            for depth, row, col in points:
                samples = []
                for sample_col in range(max(0, col - 5), min(17, col + 6)):
                    # Nearest reference endpoint extends into the gap;
                    # ties select the left section (columns 0 through 8).
                    if (sample_col <= 8) != (col <= 8):
                        continue
                    nearest_col = cols[np.argmin(np.abs(cols - sample_col))]
                    delta = abs(nearest_col - 8) - abs(col - 8)
                    for offset in range(-(height // 2), height // 2 + 1):
                        sample_row = row + delta + offset
                        if 0 <= sample_row < image.shape[1]:
                            samples.append(
                                image[depth, sample_row, sample_col]
                            )
                expected_med = np.nanmedian(samples)
                self.assertAlmostEqual(
                    float(med[depth, row, col]), float(expected_med)
                )
            unselected = np.ones(image.size, dtype=bool)
            unselected[selected] = False
            self.assertTrue(np.all(np.isnan(med.ravel()[unselected])))
            for chunk_size in (1, 4096):
                other_chunk = si.snake_aligned_nanmedian(
                    image,
                    selected,
                    ref_r,
                    cols,
                    height=height,
                    chunk_size=chunk_size,
                )
                np.testing.assert_allclose(med, other_chunk, equal_nan=True)
        np.testing.assert_array_equal(image, original)

    def test_empty_and_all_nan_support(self):
        """Preserve NaNs for empty output selections and missing samples."""
        image = np.full((1, 5, 11), np.nan, dtype=np.float32)
        cols = np.arange(11)
        for selected in (np.array([], dtype=int), np.array([27])):
            with self.subTest(selected=selected.size):
                med = si.snake_aligned_nanmedian(
                    image, selected, np.full(11, 2), cols
                )
                self.assertTrue(np.all(np.isnan(med)))


if __name__ == "__main__":
    unittest.main()
