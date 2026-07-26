"""Tests for giant_python.bandsilo.summary_images (Phase 4).

Exercises the mean image, the spatio-temporal local-maxima activity-image
accumulation, and the median-subtraction finalize on small synthetic inputs.
"""

import unittest

import numpy as np

from giant_python.bandsilo import summary_images as si


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
    """finalize_activity_image masking and local median subtraction."""

    def test_masks_and_subtracts_median(self):
        """Never-valid pixels become NaN; valid ones median-subtracted."""
        num_fast_zs, npc, npr = 1, 13, 13
        act_im = np.ones((num_fast_zs, npc, npr), dtype=np.float32)
        # all pixels selected and valid (nan_ct = 0)
        sel_pix_idxs = np.arange(npc * npr)
        nan_ct = np.zeros(sel_pix_idxs.size)
        out = si.finalize_activity_image(act_im.copy(), sel_pix_idxs, nan_ct)
        # constant image minus its local median -> ~0 everywhere valid
        self.assertTrue(np.all(np.isfinite(out)))
        np.testing.assert_allclose(out, 0.0, atol=1e-5)

    def test_high_nan_pixels_masked(self):
        """Pixels with nan_ct > 0.5 are excluded and set to NaN."""
        num_fast_zs, npc, npr = 1, 13, 13
        act_im = np.ones((num_fast_zs, npc, npr), dtype=np.float32)
        sel_pix_idxs = np.array([3 * npr + 3, 3 * npr + 4])
        nan_ct = np.array([0.0, 0.9])  # second pixel dropped
        out = si.finalize_activity_image(act_im, sel_pix_idxs, nan_ct)
        # the dropped pixel and all unselected pixels are NaN
        self.assertTrue(np.isnan(out[0, 3, 4]))
        self.assertTrue(np.isnan(out[0, 0, 0]))
        self.assertFalse(np.isnan(out[0, 3, 3]))


if __name__ == "__main__":
    unittest.main()
