"""Tests for giant_python.bandsilo.peaks (Phase 5 peak detection).

Covers the integrated-Gaussian forward/Jacobian kernels, the bounded LM
solver, the per-plane detector, and the 3-D ``get_act_im_peaks`` driver with
synthetic activity images. The full detector pipeline is additionally
cross-checked against a verbatim copy of the reference in the project's
development notes; here we assert structural/numerical behavior on small cases.
"""

import unittest
from unittest.mock import patch

import numpy as np

from giant_python.bandsilo import peaks as pk


def _grid(height, width):
    """Return an ``(H*W, 2)`` array of all ``[y, x]`` pixel centers."""
    yy, xx = np.mgrid[0:height, 0:width]
    return np.column_stack(
        [yy.ravel().astype(float), xx.ravel().astype(float)]
    )


def _single_gaussian_plane(height, width, cy, cx, amp, sigma, sigma_x=0.8):
    """Render one integrated Gaussian over an ``(H, W)`` plane."""
    yx = _grid(height, width)
    theta = np.array([[amp, cy, cx, sigma, sigma_x]])
    return pk.gaussian_peaks_integrated(theta, yx).reshape(height, width)


class TestGaussianForward(unittest.TestCase):
    """gaussian_peaks_integrated analytic properties."""

    def test_anisotropic_total_integral(self):
        """Unequal widths integrate to amp*2*pi*sigma_y*sigma_x."""
        theta = np.array([[3.0, 50.3, 12.7, 8.0, 0.7]])
        total = pk.gaussian_peaks_integrated(theta, _grid(101, 27)).sum()
        self.assertAlmostEqual(total, 3.0 * 2 * np.pi * 8.0 * 0.7, places=6)

    def test_equal_widths_match_legacy_isotropic(self):
        yx = _grid(25, 25)
        isotropic = np.array([[2.0, 12.3, 10.7, 0.9]])
        anisotropic = np.column_stack([isotropic, isotropic[:, 3]])
        np.testing.assert_allclose(
            pk.gaussian_peaks_integrated(anisotropic, yx),
            pk.gaussian_peaks_integrated(isotropic, yx),
            rtol=1e-12,
        )

    def test_total_integral(self):
        """Integrated Gaussian summed over a wide grid gives amp*2*pi*s^2."""
        yx = _grid(60, 60)
        amp, s = 3.0, 2.0
        theta = np.array([[amp, 30.0, 30.0, s]])
        total = pk.gaussian_peaks_integrated(theta, yx).sum()
        self.assertAlmostEqual(total, amp * 2 * np.pi * s**2, places=3)

    def test_superposition(self):
        """Two Gaussians sum to the sum of their individual renders."""
        yx = _grid(30, 30)
        t1 = np.array([[2.0, 10.0, 12.0, 1.5]])
        t2 = np.array([[3.0, 20.0, 18.0, 1.0]])
        both = pk.gaussian_peaks_integrated(np.vstack([t1, t2]), yx)
        sep = pk.gaussian_peaks_integrated(
            t1, yx
        ) + pk.gaussian_peaks_integrated(t2, yx)
        np.testing.assert_allclose(both, sep, rtol=1e-10)


class TestValJac(unittest.TestCase):
    """_gaussian_peaks_integrated_val_jac value + analytical Jacobian."""

    def test_anisotropic_jacobian_matches_finite_difference(self):
        """Both width derivatives are independent and correctly ordered."""
        yx = _grid(35, 19)
        theta = np.array([
            [2.0, 16.3, 7.7, 8.0, 0.7],
            [1.2, 21.0, 12.0, 3.0, 1.0],
        ])
        val, jac = pk._gaussian_peaks_integrated_val_jac(theta, yx)
        np.testing.assert_allclose(
            val, pk.gaussian_peaks_integrated(theta, yx), rtol=1e-12
        )
        self.assertEqual(jac.shape, (len(yx), theta.size))
        eps = 1e-6
        for k in range(theta.size):
            hi = theta.ravel().copy()
            lo = theta.ravel().copy()
            hi[k] += eps
            lo[k] -= eps
            fd = (
                pk.gaussian_peaks_integrated(hi.reshape(-1, 5), yx)
                - pk.gaussian_peaks_integrated(lo.reshape(-1, 5), yx)
            ) / (2 * eps)
            np.testing.assert_allclose(jac[:, k], fd, rtol=1e-4, atol=1e-7)

    def test_value_matches_forward(self):
        """The val_jac value equals the forward-only evaluation."""
        yx = _grid(20, 20)
        theta = np.array([[2.0, 9.3, 11.7, 1.4], [1.5, 14.0, 6.0, 1.1]])
        val, _ = pk._gaussian_peaks_integrated_val_jac(theta, yx)
        fwd = pk.gaussian_peaks_integrated(theta, yx)
        np.testing.assert_allclose(val, fwd, rtol=1e-12)

    def test_jacobian_matches_finite_difference(self):
        """Analytical Jacobian matches a central finite-difference estimate."""
        yx = _grid(16, 16)
        theta = np.array([[2.0, 8.3, 7.7, 1.3], [1.2, 11.0, 5.0, 0.9]])
        _, jac = pk._gaussian_peaks_integrated_val_jac(theta, yx)
        flat = theta.ravel().copy()
        eps = 1e-6
        for k in range(flat.size):
            hi = flat.copy()
            hi[k] += eps
            lo = flat.copy()
            lo[k] -= eps
            v_hi = pk.gaussian_peaks_integrated(hi.reshape(-1, 4), yx)
            v_lo = pk.gaussian_peaks_integrated(lo.reshape(-1, 4), yx)
            fd = (v_hi - v_lo) / (2 * eps)
            np.testing.assert_allclose(jac[:, k], fd, rtol=1e-4, atol=1e-4)


class TestLsqCurvefit(unittest.TestCase):
    """_lsq_curvefit recovery and control-flow branches."""

    def test_recovers_known_gaussian(self):
        """Fitting noise-free data recovers the generating parameters."""
        yx = _grid(21, 21)
        true = np.array([[5.0, 10.4, 9.6, 6.0, 0.7]])
        ydata = pk.gaussian_peaks_integrated(true, yx)
        theta0 = np.array([[3.0, 10.0, 10.0, 0.5, 0.5]])
        lb, ub = pk._make_bounds(theta0[:, 1:3], 21, 21)
        fit = pk._lsq_curvefit(theta0, yx, ydata, lb, ub)
        np.testing.assert_allclose(fit, true, atol=1e-2)

    def test_width_limits(self):
        """A target wider than either limit cannot exceed the fitted bounds."""
        yx = _grid(41, 21)
        true = np.array([[5.0, 20.4, 9.6, 10.0, 1.5]])
        theta0 = np.array([[4.0, 20.0, 10.0, 6.0, 0.8]])
        lb, ub = pk._make_bounds(theta0[:, 1:3], 41, 21)
        fit = pk._lsq_curvefit(
            theta0, yx, pk.gaussian_peaks_integrated(true, yx), lb, ub
        )
        self.assertTrue(np.all(fit.ravel() >= lb))
        self.assertTrue(np.all(fit.ravel() <= ub))
        np.testing.assert_allclose(fit[0, 3:], [8.0, 1.0], atol=1e-2)

    def test_max_nfev_early_break(self):
        """max_nfev=1 returns the (clipped) initial parameters immediately."""
        yx = _grid(11, 11)
        ydata = pk.gaussian_peaks_integrated(
            np.array([[4.0, 5.0, 5.0, 1.0]]), yx
        )
        theta0 = np.array([[3.0, 5.0, 5.0, 1.0, 1.0]])
        lb, ub = pk._make_bounds(theta0[:, 1:3], 11, 11)
        out = pk._lsq_curvefit(theta0, yx, ydata, lb, ub, max_nfev=1)
        np.testing.assert_allclose(out, theta0)

    def test_non_improving_step_increases_damping(self):
        """A clipped, non-improving step exercises the damping-increase branch.

        Tight mean bounds around a wrong location force steps that cannot
        reduce cost, so the solver repeatedly raises the LM damping.
        """
        yx = _grid(21, 21)
        # data has its peak far from the seed; bounds pin the mean near seed
        ydata = pk.gaussian_peaks_integrated(
            np.array([[6.0, 15.0, 15.0, 1.2]]), yx
        )
        theta0 = np.array([[6.0, 4.0, 4.0, 1.2]])
        lb = np.array([5.9, 3.9, 3.9, 1.1])
        ub = np.array([6.1, 4.1, 4.1, 1.3])
        out = pk._lsq_curvefit(theta0, yx, ydata, lb, ub, max_nfev=50)
        # stays within the tight bounds (no crash, no bound violation)
        self.assertTrue(np.all(out.ravel() >= lb - 1e-9))
        self.assertTrue(np.all(out.ravel() <= ub + 1e-9))


class TestIsotropicWarmStart(unittest.TestCase):
    """Two-stage fitting preserves round peaks while permitting elongation."""

    def test_stages_and_bounds(self):
        """Relaxation starts from the fitted shared widths, not the seed."""
        theta0 = np.array([[3.0, 10.0, 10.0, 4.0, 0.8]])
        original = theta0.copy()
        lb, ub = pk._make_bounds(theta0[:, 1:3], 21, 21)
        iso_fit = np.array([[5.0, 10.3, 9.7, 0.9]])
        final_fit = np.array([[5.0, 10.3, 9.7, 4.0, 0.7]])
        with patch.object(
            pk, "_lsq_curvefit", side_effect=[iso_fit, final_fit]
        ) as solver:
            out = pk._fit_isotropic_then_anisotropic(
                theta0, _grid(21, 21), np.zeros(441), lb, ub,
                max_nfev=100,
            )
        self.assertEqual(solver.call_count, 2)
        first, second = solver.call_args_list
        self.assertEqual(first.args[0].shape, (1, 4))
        self.assertEqual(first.args[3][3], 0.35)
        self.assertEqual(first.args[4][3], 1.0)
        np.testing.assert_array_equal(
            second.args[0], [[5.0, 10.3, 9.7, 0.9, 0.9]]
        )
        np.testing.assert_array_equal(second.args[3], lb)
        np.testing.assert_array_equal(second.args[4], ub)
        self.assertEqual(first.kwargs["max_nfev"], 100)
        self.assertEqual(second.kwargs["max_nfev"], 100)
        np.testing.assert_array_equal(out, final_fit)
        np.testing.assert_array_equal(theta0, original)

    def test_recovers_round_and_elongated_peaks(self):
        """Joint fitting recovers round and tall peaks after relaxation."""
        yx = _grid(65, 31)
        true = np.array([
            [5.0, 15.3, 8.7, 0.8, 0.8],
            [8.0, 40.4, 22.2, 7.0, 0.7],
        ])
        theta0 = np.array([
            [3.0, 15.0, 9.0, 0.5, 0.5],
            [4.0, 40.0, 22.0, 0.5, 0.5],
        ])
        lb, ub = pk._make_bounds(theta0[:, 1:3], 65, 31)
        out = pk._fit_isotropic_then_anisotropic(
            theta0, yx, pk.gaussian_peaks_integrated(true, yx), lb, ub
        )
        np.testing.assert_allclose(out, true, atol=0.01)

    def test_exact_isotropic_minimum_stays_isotropic(self):
        """Relaxing an already exact round fit does not change it."""
        theta = np.array([[5.0, 10.3, 9.7, 0.8, 0.8]])
        yx = _grid(21, 21)
        lb, ub = pk._make_bounds(theta[:, 1:3], 21, 21)
        out = pk._fit_isotropic_then_anisotropic(
            theta, yx, pk.gaussian_peaks_integrated(theta, yx), lb, ub
        )
        np.testing.assert_array_equal(out, theta)

    def test_empty_parameters(self):
        """No peaks need no optimization."""
        theta = np.empty((0, 5))
        out = pk._fit_isotropic_then_anisotropic(
            theta, _grid(3, 3), np.zeros(9), np.array([]), np.array([])
        )
        self.assertEqual(out.shape, (0, 5))


class TestSmallHelpers(unittest.TestCase):
    """_make_bounds / _peak_mask / _buffer_mask."""

    def test_make_bounds(self):
        """Means clip to the image; sigma_y <= 8 and sigma_x <= 1."""
        plocs = np.array([[0.0, 9.0]])
        lb, ub = pk._make_bounds(plocs, 10, 10)
        # y lower clipped to 0; x within range
        self.assertAlmostEqual(lb[1], 0.0)
        self.assertAlmostEqual(ub[2], 9.0)  # min(W-1, 9+1.5) = 9
        np.testing.assert_array_equal(lb[3:], [0.35, 0.35])
        np.testing.assert_array_equal(ub[3:], [8.0, 1.0])

    def test_peak_mask_and_empty(self):
        """Peak centers mark the mask; an empty parameter set marks nothing."""
        tf = np.array([[1.0, 3.2, 4.8, 1.0]])
        mask = pk._peak_mask(tf, 8, 8)
        self.assertTrue(mask[3, 5])
        self.assertEqual(mask.sum(), 1)
        empty = pk._peak_mask(np.zeros((0, 4)), 8, 8)
        self.assertFalse(empty.any())

    def test_buffer_mask(self):
        """buffer_size>0 dilates; <=0 returns the mask unchanged."""
        pim = np.zeros((7, 7), dtype=bool)
        pim[3, 3] = True
        buffered = pk._buffer_mask(pim, 3)
        self.assertEqual(buffered.sum(), 9)  # 3x3 dilation
        same = pk._buffer_mask(pim, 0)
        np.testing.assert_array_equal(same, pim)


class TestProcessCcNewPeak(unittest.TestCase):
    """_process_cc_new_peak accept / reject (with and without other peaks)."""

    def test_rejects_integer_peak_with_others(self):
        """A peak that converges to its integer seed, beside an existing one,
        is rejected (and the others refit)."""
        h = w = 15
        # a blob centered exactly on the integer seed -> fit stays near-integer
        act = _single_gaussian_plane(h, w, 5.0, 5.0, 8.0, 1.1)
        labeled = np.ones((h, w), dtype=int)
        thetaf = np.array([[1.0, 10.0, 10.0, 1.0, 0.8]])  # existing peak in cc
        p_locs = np.array([[10.0, 10.0]])
        reject_mask = np.zeros((h, w), dtype=bool)
        new_thetaf, _, _, _ = pk._process_cc_new_peak(
            ((5, 5), 1), thetaf, p_locs, labeled, act, 0.0, reject_mask, h, w
        )
        self.assertEqual(new_thetaf.shape[0], 1)  # new peak removed
        self.assertTrue(reject_mask[5, 5])

    def test_rejects_flat_peak_alone(self):
        """A non-moving lone peak is rejected (no other peaks to refit)."""
        h = w = 15
        act = np.zeros((h, w))
        labeled = np.zeros((h, w), dtype=int)
        labeled[3:12, 3:12] = 1
        thetaf = np.zeros((0, 5))
        p_locs = np.zeros((0, 2))
        reject_mask = np.zeros((h, w), dtype=bool)
        new_thetaf, _, _, _ = pk._process_cc_new_peak(
            ((5, 5), 1), thetaf, p_locs, labeled, act, 0.0, reject_mask, h, w
        )
        self.assertEqual(new_thetaf.shape[0], 0)
        self.assertTrue(reject_mask[5, 5])

    def test_accepts_moving_peak(self):
        """A peak that moves off its integer seed is kept."""
        h = w = 15
        act = _single_gaussian_plane(h, w, 5.4, 5.6, 6.0, 1.1)
        labeled = np.ones((h, w), dtype=int)
        thetaf = np.zeros((0, 5))
        p_locs = np.zeros((0, 2))
        reject_mask = np.zeros((h, w), dtype=bool)
        new_thetaf, _, _, _ = pk._process_cc_new_peak(
            ((5, 5), 1), thetaf, p_locs, labeled, act, 0.0, reject_mask, h, w
        )
        self.assertEqual(new_thetaf.shape[0], 1)
        self.assertFalse(reject_mask.any())
        # the kept peak moved toward the true (off-integer) center
        self.assertGreater(
            abs(new_thetaf[0, 1] - round(new_thetaf[0, 1])), 1e-3
        )


class TestRefineResidualPeaks(unittest.TestCase):
    """_refine_residual_peaks early-exit branch."""

    def test_empty_support_returns_unchanged(self):
        """No support (labels max 0) exits immediately, returning thetaf."""
        h = w = 10
        thetaf = np.array([[1.0, 5.0, 5.0, 1.0, 0.8]])
        p_locs = np.array([[5.0, 5.0]])
        act = np.zeros((h, w))
        empty2d = np.zeros((h, w), dtype=bool)
        out = pk._refine_residual_peaks(
            thetaf,
            p_locs,
            act,
            empty2d,  # exclusion
            0.0,
            1.0,
            3.0,
            0,
            h,
            w,
            empty2d.copy(),  # buffer_mask
            np.zeros((h, w)),  # res_im
            np.zeros((h, w), dtype=bool),  # fit_support
            np.zeros((h, w), dtype=bool),  # act_sel_pix -> no labels
            np.zeros((h, w)),  # fit_im
        )
        np.testing.assert_array_equal(out, thetaf)


class TestDetectPeaks2d(unittest.TestCase):
    """detect_peaks_2d single-plane behavior."""

    def test_no_peaks_returns_empty(self):
        """A plane with nothing above threshold returns no peaks."""
        rng = np.random.default_rng(0)
        act = rng.standard_normal((30, 30)) * 0.2
        out = pk.detect_peaks_2d(
            act,
            np.zeros((30, 30), dtype=bool),
            mu_bg=0.0,
            sigma_bg=0.2,
            peak_thresh=100.0,  # unreachable
            peak_th=100.0,
        )
        self.assertEqual(out.shape, (0, 5))

    def test_elongated_blob_detected(self):
        """A tall, narrow blob is fit as one anisotropic Gaussian."""
        act = _single_gaussian_plane(65, 25, 32.3, 12.4, 10.0, 7.0, 0.7)
        out = pk.detect_peaks_2d(
            act, np.zeros_like(act, dtype=bool),
            mu_bg=0.0, sigma_bg=0.05, peak_thresh=1.0, peak_th=3.0,
        )
        self.assertEqual(out.shape, (1, 5))
        np.testing.assert_allclose(
            out[0], [10.0, 32.3, 12.4, 7.0, 0.7], atol=0.05
        )

    def test_single_blob_detected(self):
        """A single blob yields one peak near its true center."""
        act = _single_gaussian_plane(30, 30, 14.3, 16.7, 10.0, 1.3)
        out = pk.detect_peaks_2d(
            act,
            np.zeros((30, 30), dtype=bool),
            mu_bg=0.0,
            sigma_bg=0.05,
            peak_thresh=1.0,
            peak_th=3.0,
        )
        self.assertGreaterEqual(out.shape[0], 1)
        # the strongest peak should sit near the true center
        best = out[np.argmax(out[:, 0])]
        self.assertAlmostEqual(best[1], 14.3, delta=0.5)
        self.assertAlmostEqual(best[2], 16.7, delta=0.5)

    def test_close_pair_adds_residual_peak(self):
        """Two blobs too close to both be initial maxima: the second is
        recovered by the residual-peak refinement loop."""
        act = _single_gaussian_plane(
            34, 34, 16.4, 15.0, 10.0, 1.2
        ) + _single_gaussian_plane(34, 34, 16.4, 16.8, 9.0, 1.2)
        out = pk.detect_peaks_2d(
            act,
            np.zeros((34, 34), dtype=bool),
            mu_bg=0.0,
            sigma_bg=0.05,
            peak_thresh=1.0,
            peak_th=3.0,
        )
        # both blobs recovered though only one initial local max exists
        self.assertEqual(out.shape[0], 2)


class TestGetActImPeaks(unittest.TestCase):
    """get_act_im_peaks 3-D driver and its guard branches."""

    def test_elongated_blob_seed(self):
        """The 3-D driver retains its [z, y, x] output for unequal widths."""
        rng = np.random.default_rng(42)
        act = rng.normal(0.0, 0.01, (1, 65, 25))
        act[0] += _single_gaussian_plane(65, 25, 32.3, 12.4, 10.0, 7.0, 0.7)
        seeds = pk.get_act_im_peaks(act, peak_th=6.0, buffer_size=3)
        self.assertEqual(seeds.shape, (1, 3))
        np.testing.assert_allclose(seeds[0], [0.0, 32.3, 12.4], atol=0.1)

    def _act_im(self, seed=0):
        """Two-plane activity image with a blob per plane, low noise, NaNs."""
        rng = np.random.default_rng(seed)
        act = rng.standard_normal((2, 40, 40)) * 0.3
        act[0] += _single_gaussian_plane(40, 40, 12.4, 18.6, 9.0, 1.3)
        act[1] += _single_gaussian_plane(40, 40, 25.0, 15.0, 8.0, 1.2)
        act[:, :3, :3] = np.nan
        return act

    def test_detects_seeds_per_plane(self):
        """Both planes' blobs are found, tagged with their z index."""
        seeds = pk.get_act_im_peaks(self._act_im(), peak_th=4.0, buffer_size=3)
        self.assertEqual(seeds.shape[1], 3)
        self.assertTrue(np.any(seeds[:, 0] == 0))
        self.assertTrue(np.any(seeds[:, 0] == 1))

    def test_exclusion_mask_2d(self):
        """A 2-D exclusion mask suppresses seeds in the masked region."""
        excl = np.zeros((40, 40), dtype=bool)
        excl[8:18, 14:24] = True  # covers the plane-0 blob
        seeds = pk.get_act_im_peaks(
            self._act_im(), peak_th=4.0, exclusion_mask=excl, buffer_size=3
        )
        # no seed inside the excluded box
        for z, y, x in seeds:
            self.assertFalse(8 <= y < 18 and 14 <= x < 24)

    def test_exclusion_mask_3d(self):
        """A 3-D per-plane exclusion mask is applied plane-by-plane."""
        excl = np.zeros((2, 40, 40), dtype=bool)
        excl[1, 20:30, 10:20] = True  # covers the plane-1 blob only
        seeds = pk.get_act_im_peaks(
            self._act_im(), peak_th=4.0, exclusion_mask=excl, buffer_size=3
        )
        z1_in_box = any(
            int(z) == 1 and 20 <= y < 30 and 10 <= x < 20 for z, y, x in seeds
        )
        self.assertFalse(z1_in_box)

    def test_all_nan_returns_empty(self):
        """An all-NaN activity image yields no seeds."""
        act = np.full((2, 20, 20), np.nan)
        out = pk.get_act_im_peaks(act)
        self.assertEqual(out.shape, (0, 3))

    def test_zero_mad_returns_empty(self):
        """A constant image (MAD sigma 0) yields no seeds."""
        act = np.ones((2, 20, 20))
        out = pk.get_act_im_peaks(act)
        self.assertEqual(out.shape, (0, 3))

    def test_no_peaks_returns_empty(self):
        """Pure low noise with a high threshold yields no seeds."""
        rng = np.random.default_rng(1)
        act = rng.standard_normal((2, 30, 30)) * 0.3
        out = pk.get_act_im_peaks(act, peak_th=50.0)
        self.assertEqual(out.shape, (0, 3))


if __name__ == "__main__":
    unittest.main()
