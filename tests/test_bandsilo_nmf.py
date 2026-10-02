"""Tests for canonical band localization (Phase 6 source localization).

Covers the profile/init kernels, the least-squares/SNR helpers, and the full
``fit_sources`` driver (multiplicative NMF + Adam Gaussian fit + variance
sort + SNR pruning) on a small synthetic geometry. We assert structural
behavior and determinism rather than bit-for-bit agreement with the
reference's regularized normal-equation solve.
"""

import unittest

import numpy as np
import torch

from giant_python.extraction.band import activity, geometry
from giant_python.extraction.band import localization as nmf
from tests.test_bandsilo_background import _small_geometry


def _tiny_coords(size=5):
    """Return (pixel_coords, sel_pix_idxs, n_pixels) for one square plane."""
    rr, cc = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    coords = np.column_stack(
        [np.zeros(size * size), rr.ravel(), cc.ravel()]
    ).astype(np.int32)
    sel_pix_idxs = np.arange(size * size)
    return coords, sel_pix_idxs, size * size


class TestProfiles(unittest.TestCase):
    """sel_pix_gaussian_profile / sel_pix_patch_profile."""

    def test_gaussian_profile_normalized_and_confined(self):
        """Each Gaussian column sums to 1 and only its z-plane is populated."""
        coords, _, _ = _tiny_coords(5)
        pct = torch.tensor(coords, dtype=torch.float32)
        params = torch.tensor([[0.0, 2.0, 2.0, 1.0, 1.0, 0.0]])
        prof = nmf.sel_pix_gaussian_profile(params, pct)
        self.assertEqual(prof.shape, (25, 1))
        self.assertAlmostEqual(float(prof.sum()), 1.0, places=5)
        # peak at the center pixel (row 2, col 2 -> flat index 12)
        self.assertEqual(int(torch.argmax(prof[:, 0])), 12)

    def test_gaussian_profile_wrong_plane_empty(self):
        """A source on a different z-plane yields an all-nan column."""
        coords, _, _ = _tiny_coords(5)
        pct = torch.tensor(coords, dtype=torch.float32)
        params = torch.tensor(
            [[1.0, 2.0, 2.0, 1.0, 1.0, 0.0]]
        )  # z=1, no pixels
        prof = nmf.sel_pix_gaussian_profile(params, pct)
        # sum is 0 before normalization -> division yields nan
        self.assertTrue(torch.all(torch.isnan(prof)))

    def test_patch_profile_membership(self):
        """The patch is the strict box within the radii on the source plane."""
        coords, _, _ = _tiny_coords(5)
        pct = torch.tensor(coords, dtype=torch.float32)
        params = torch.tensor([[0.0, 2.0, 2.0, 2.0, 2.0]])  # radius 2
        patch = nmf.sel_pix_patch_profile(params, pct)
        self.assertEqual(patch.shape, (25, 1))
        # |y-2|<2 and |x-2|<2 -> rows/cols {1,2,3} -> 9 pixels
        self.assertEqual(int(patch.sum()), 9)


class TestInit(unittest.TestCase):
    """init_source_params / build_a_patches / project_spatial_profiles."""

    def test_init_source_params(self):
        """Params start as [z, y, x, 1, 1, 0]."""
        seeds = np.array([[0.0, 3.0, 4.0], [1.0, 5.0, 6.0]])
        params = nmf.init_source_params(seeds)
        self.assertEqual(tuple(params.shape), (2, 6))
        np.testing.assert_array_equal(params[:, 3:5].numpy(), np.ones((2, 2)))
        np.testing.assert_array_equal(params[:, 5].numpy(), np.zeros(2))

    def test_build_a_patches(self):
        """Patches are True only within each source's box on the pixel grid."""
        coords, sel, n_pix = _tiny_coords(5)
        pct = torch.tensor(coords, dtype=torch.float32)
        seeds = np.array([[0.0, 2.0, 2.0]])
        patches = nmf.build_a_patches(seeds, pct, sel, n_pix, d_xy=2)
        self.assertEqual(tuple(patches.shape), (25, 1))
        self.assertEqual(int(patches.sum()), 9)

    def test_project_spatial_profiles_normalized_in_patch(self):
        """A is normalized to unit column mass and zero outside the patch."""
        coords, sel, n_pix = _tiny_coords(5)
        pct = torch.tensor(coords, dtype=torch.float32)
        seeds = np.array([[0.0, 2.0, 2.0]])
        params = nmf.init_source_params(seeds)
        patches = nmf.build_a_patches(seeds, pct, sel, n_pix, d_xy=2)
        a = nmf.project_spatial_profiles(params, pct, sel, n_pix, patches)
        self.assertAlmostEqual(float(a.sum()), 1.0, places=5)
        # every nonzero pixel lies inside the patch
        self.assertTrue(torch.all(patches[a > 0]))


class TestVarianceSortAndReorder(unittest.TestCase):
    """variance_sortorder / reorder_sources."""

    def test_variance_sortorder_descending(self):
        """Sources are ordered by descending temporal variance."""
        phi = torch.zeros((10, 3))
        phi[:, 0] = torch.linspace(0, 1, 10)  # small variance
        phi[:, 1] = torch.linspace(0, 10, 10)  # large variance
        phi[:, 2] = torch.linspace(0, 5, 10)  # medium
        order = nmf.variance_sortorder(phi, 3)
        self.assertEqual(list(order), [1, 2, 0])

    def test_reorder_sources(self):
        """Reordering reindexes every per-source array consistently."""
        order = np.array([2, 0, 1])
        source_params = torch.arange(18).reshape(3, 6).float()
        source_seeds = np.arange(9).reshape(3, 3).astype(float)
        a = torch.arange(12).reshape(4, 3).float()
        a_patches = torch.ones((4, 3), dtype=torch.bool)
        x_support = [torch.arange(6).reshape(2, 3)]
        phi = torch.arange(15).reshape(5, 3).float()
        sp, ss, na, nap, xs, nphi = nmf.reorder_sources(
            order, source_params, source_seeds, a, a_patches, x_support, phi
        )
        np.testing.assert_array_equal(ss[:, 0], [6.0, 0.0, 3.0])
        np.testing.assert_array_equal(na[0].numpy(), [2.0, 0.0, 1.0])
        np.testing.assert_array_equal(xs[0][0].numpy(), [2, 0, 1])


class TestSolvePhiMotion(unittest.TestCase):
    """Unregularized, rank-revealing QR temporal solve."""

    def test_overdetermined_multiple_frames(self):
        """QR matches an independent SVD least-squares reference."""
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                x = torch.tensor(
                    [[1, 2], [3, 1], [2, 4], [0, 1]], dtype=dtype
                )
                data = torch.tensor(
                    [[2, 1], [0, 3], [4, 2], [1, 5]], dtype=dtype
                )
                phi = nmf.solve_phi_motion(x, data)
                expected = np.linalg.lstsq(
                    x.numpy(), data.numpy(), rcond=None
                )[0].T
                self.assertEqual(phi.shape, (2, 2))
                self.assertEqual(phi.dtype, dtype)
                self.assertEqual(phi.device, x.device)
                np.testing.assert_allclose(
                    phi.numpy(), expected, rtol=1e-5, atol=1e-6
                )

    def test_small_profiles_are_not_regularized(self):
        """Tiny profile magnitudes must not shrink the temporal weights."""
        x = 1e-6 * torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
        expected = torch.tensor([[2., 3.], [4., 5.]])
        phi = nmf.solve_phi_motion(x, x @ expected.T)
        torch.testing.assert_close(phi, expected)

    def test_nearly_collinear_profiles(self):
        """QR resolves columns whose float32 normal equations lose rank."""
        x = torch.tensor([[1., 1.], [1., 1.0001], [1., 0.9999]])
        expected = torch.tensor([[2., -1.], [-3., 4.]])
        phi = nmf.solve_phi_motion(x, x @ expected.T)
        torch.testing.assert_close(phi, expected, rtol=5e-3, atol=5e-3)

    def test_rank_deficient_basic_solution(self):
        """Dependent coefficients are zero, not shared by minimum norm."""
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                x = torch.tensor([[1, 2], [2, 4], [3, 6]], dtype=dtype)
                # Include an inconsistent RHS to test least squares too.
                data = torch.tensor([[3, 1], [6, 0], [9, 0]], dtype=dtype)
                phi = nmf.solve_phi_motion(x, data)
                expected = torch.tensor(
                    [[0, 1.5], [0, 1 / 28]], dtype=dtype
                )
                torch.testing.assert_close(phi, expected)
                self.assertTrue(torch.all(phi[:, 0] == 0))
                # Residual is orthogonal to the column space.
                torch.testing.assert_close(
                    x.T @ (x @ phi.T - data),
                    torch.zeros((2, 2), dtype=dtype),
                    atol=2e-5, rtol=0,
                )

    def test_underdetermined_basic_solution(self):
        """Solve only the independent pivot columns and undo permutation."""
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                # Distinct pivot norms avoid ambiguous tie-breaking.
                x = torch.tensor([[1, 0, 2], [0, 3, 0]], dtype=dtype)
                data = torch.tensor([[4, -2], [9, 6]], dtype=dtype)
                phi = nmf.solve_phi_motion(x, data)
                expected = torch.tensor([[0, 3, 2], [0, 2, -1]], dtype=dtype)
                torch.testing.assert_close(phi, expected)
                self.assertTrue(torch.all(phi[:, 0] == 0))
                torch.testing.assert_close(x @ phi.T, data)

    def test_rank_deficient_underdetermined(self):
        """A wide rank-one system uses exactly one pivot coefficient."""
        x = torch.tensor([[0., 1., 2.], [0., 2., 4.]])
        data = torch.tensor([[6., -2.], [12., -4.]])
        phi = nmf.solve_phi_motion(x, data)
        torch.testing.assert_close(
            phi, torch.tensor([[0., 0., 3.], [0., 0., -1.]])
        )
        self.assertTrue(torch.all(phi[:, :2] == 0))

    def test_zero_rank(self):
        """A zero design matrix returns the all-zero basic solution."""
        x = torch.zeros((3, 2))
        phi = nmf.solve_phi_motion(x, torch.ones((3, 4)))
        torch.testing.assert_close(phi, torch.zeros((4, 2)))

    def test_nonfinite_frames_do_not_abort_finite_fits(self):
        """Missing observations invalidate a frame, not the whole batch."""
        for dtype in (torch.float32, torch.float64):
            for design in (
                [[1, 0], [0, 1], [1, 1]],
                [[1, 2], [2, 4], [3, 6]],
                [[1, 0, 2], [0, 3, 0]],
                [[0, 0], [0, 0], [0, 0]],
            ):
                with self.subTest(dtype=dtype, design=design):
                    x = torch.tensor(design, dtype=dtype)
                    data = torch.arange(
                        x.shape[0] * 6, dtype=dtype
                    ).reshape(x.shape[0], 6)
                    data[0, 1] = float("nan")
                    data[:, 2] = float("nan")
                    data[-1, 3] = float("inf")
                    data[0, 4] = -float("inf")
                    original = data.clone()
                    expected = nmf.solve_phi_motion(x, data[:, [0, 5]])
                    actual = nmf.solve_phi_motion(x, data)
                    self.assertEqual(actual.shape, (6, x.shape[1]))
                    self.assertEqual(actual.dtype, dtype)
                    torch.testing.assert_close(actual[[0, 5]], expected)
                    self.assertTrue(torch.isnan(actual[1:5]).all())
                    torch.testing.assert_close(data, original, equal_nan=True)

    def test_all_frames_missing(self):
        """An entirely missing bin returns NaNs, including dependent sources."""
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                x = torch.tensor([[1, 2], [2, 4], [3, 6]], dtype=dtype)
                data = torch.full((3, 4), float("nan"), dtype=dtype)
                actual = nmf.solve_phi_motion(x, data)
                self.assertEqual(actual.shape, (4, 2))
                self.assertEqual(actual.dtype, dtype)
                self.assertTrue(torch.isnan(actual).all())

    def test_nonfinite_profiles_still_raise(self):
        """Invalid spatial models must not be silently accepted by QR."""
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                x = torch.tensor([[1., 0.], [0., value], [1., 1.]])
                with self.assertRaisesRegex(ValueError, "infs or NaNs"):
                    nmf.solve_phi_motion(x, torch.ones((3, 2)))

    def test_numerically_dependent_column(self):
        """The dtype-relative cutoff drops a tiny trailing QR diagonal."""
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                x = torch.tensor(
                    [[1, 0], [0, torch.finfo(dtype).eps / 10], [0, 0]],
                    dtype=dtype,
                )
                data = x @ torch.ones((2, 1), dtype=dtype)
                phi = nmf.solve_phi_motion(x, data)
                torch.testing.assert_close(
                    phi, torch.tensor([[1, 0]], dtype=dtype)
                )

    def test_empty_sources_and_frames(self):
        """Empty bins and fully pruned sources preserve output dimensions."""
        for n_sources, n_frames in ((0, 3), (2, 0), (0, 0)):
            with self.subTest(n_sources=n_sources, n_frames=n_frames):
                x = torch.eye(4, n_sources)
                data = torch.zeros((4, n_frames))
                phi = nmf.solve_phi_motion(x, data)
                self.assertEqual(phi.shape, (n_frames, n_sources))


class TestPhiAndSnr(unittest.TestCase):
    """fit_phi_all_motions / build_x_support_mots / compute_source_snr."""

    def _setup(self, seed=0):
        """Build A / data / geometry for a couple of sources."""
        g = _small_geometry(seed=seed)
        npc, npr, nz = g["npc"], g["npr"], g["num_fast_zs"]
        n_pix = nz * npc * npr
        sel = g["sel_pix_idxs"]
        pct = torch.tensor(
            geometry.pixel_coords_from_idxs(sel, npc, npr), dtype=torch.float32
        )
        seeds = np.array([[0.0, 7.0, 7.0], [1.0, 7.0, 8.0]])
        params = nmf.init_source_params(seeds)
        patches = nmf.build_a_patches(seeds, pct, sel, n_pix, d_xy=5)
        a = nmf.project_spatial_profiles(params, pct, sel, n_pix, patches)
        return g, a, sel, seeds

    def test_build_x_support_and_fit_phi(self):
        """phi solves the per-motion least squares for structured data."""
        g, a, sel, seeds = self._setup()
        n_motions = g["umyx"].shape[0]
        supports = nmf.build_x_support_mots(g["h_mots"], a, sel, n_motions)
        self.assertEqual(len(supports), n_motions)

        n_frames = 8
        mot_yx = np.array([0, 1] * (n_frames // 2), dtype=np.int32)
        # structured data: exactly reconstructable from A with unit phi
        data = torch.zeros((g["num_super_pixels"], n_frames))
        for i in range(n_motions):
            frames = np.flatnonzero(mot_yx == i)
            x = torch.sparse.mm(g["h_mots"][i], a[sel, :])
            data[:, frames] = x @ torch.ones((x.shape[1], len(frames)))
        phi = torch.full((n_frames, seeds.shape[0]), float("nan"))
        phi = nmf.fit_phi_all_motions(
            a, g["h_mots"], sel, data, mot_yx, n_motions, phi
        )
        # recovered phi should be ~1 for the fit frames
        fit_frames = mot_yx >= 0
        np.testing.assert_allclose(
            phi.numpy()[fit_frames], 1.0, rtol=1e-3, atol=1e-3
        )

    def test_compute_source_snr_shape_and_high_for_signal(self):
        """SNR is per-source and high when data is explained by the source."""
        g, a, sel, seeds = self._setup()
        n_motions = g["umyx"].shape[0]
        n_frames = 8
        mot_yx = np.array([0, 1] * (n_frames // 2), dtype=np.int32)
        data = torch.zeros((g["num_super_pixels"], n_frames))
        phi = torch.full((n_frames, seeds.shape[0]), float("nan"))
        for i in range(n_motions):
            frames = np.flatnonzero(mot_yx == i)
            x = torch.sparse.mm(g["h_mots"][i], a[sel, :])
            data[:, frames] = x @ torch.ones((x.shape[1], len(frames)))
        phi = nmf.fit_phi_all_motions(
            a, g["h_mots"], sel, data, mot_yx, n_motions, phi
        )
        snr = nmf.compute_source_snr(
            a, g["h_mots"], sel, data, phi, mot_yx, n_motions, seeds.shape[0]
        )
        self.assertEqual(tuple(snr.shape), (2,))
        # data is fully explained -> residual ~0 -> very large SNR
        self.assertTrue(torch.all(snr > 100))


class TestLocalizeSources(unittest.TestCase):
    """fit_sources end-to-end driver."""

    def _inputs(self, seed=0):
        """Small geometry + seeds + random residual (survives pruning)."""
        g = _small_geometry(seed=seed)
        npc, npr, nz = g["npc"], g["npr"], g["num_fast_zs"]
        n_pix = nz * npc * npr
        sel = g["sel_pix_idxs"]
        coords = geometry.pixel_coords_from_idxs(sel, npc, npr)
        seeds = np.array([[0.0, 7.0, 7.0], [0.0, 7.0, 9.0], [1.0, 7.0, 8.0]])
        rng = np.random.default_rng(seed + 100)
        n_frames = 16
        residual = rng.standard_normal(
            (g["num_super_pixels"], n_frames)
        ).astype(np.float32)
        mot_yx = np.array([0, 1] * (n_frames // 2), dtype=np.int32)
        return g, seeds, residual, mot_yx, coords, n_pix

    def test_full_run_with_pruning(self):
        """Three outer iterations exercise sparsify, sort, and prune."""
        g, seeds, residual, mot_yx, coords, n_pix = self._inputs()
        torch.manual_seed(7)
        out = nmf.fit_sources(
            seeds.copy(),
            residual,
            g["h_mots"],
            g["umyx"],
            mot_yx,
            g["sel_pix_idxs"],
            coords,
            n_pix,
            d_xy=5,
            sparse_fac=float(np.exp(-3.0)),
            outer_loop_iters=3,
            mult_nmf_max_iters=4,
        )
        self.assertEqual(out["A"].shape[0], n_pix)
        self.assertEqual(out["A"].shape[1], out["n_sources"])
        self.assertEqual(out["source_params"].shape[1], 6)
        self.assertEqual(out["phi_low_res"].shape[0], residual.shape[1])
        self.assertEqual(out["source_snr"].shape[0], out["n_sources"])
        self.assertGreaterEqual(out["n_sources"], 1)

    def test_deterministic_under_seed(self):
        """Same torch seed -> identical spatial profiles."""
        g, seeds, residual, mot_yx, coords, n_pix = self._inputs(seed=1)
        kwargs = dict(
            h_mots=g["h_mots"],
            unique_motion_to_keep_yx=g["umyx"],
            mot_inds_yx=mot_yx,
            sel_pix_idxs=g["sel_pix_idxs"],
            pixel_coords=coords,
            n_pixels=n_pix,
            d_xy=5,
            sparse_fac=float(np.exp(-3.0)),
            outer_loop_iters=2,
            mult_nmf_max_iters=3,
        )
        torch.manual_seed(3)
        a1 = nmf.fit_sources(seeds.copy(), residual, **kwargs)["A"]
        torch.manual_seed(3)
        a2 = nmf.fit_sources(seeds.copy(), residual, **kwargs)["A"]
        np.testing.assert_array_equal(a1.numpy(), a2.numpy())

    def test_temporal_filter_matches_activity_filter(self):
        """Reuse activity smoothing and omit its low-coverage edge frames."""
        g, seeds, residual, mot_yx, coords, n_pix = self._inputs(seed=1)
        original = residual.copy()
        kernel = activity.decay_kernel_1d(0.1, 10).astype(np.float32)
        filtered = activity.smooth_rho(residual.copy(), kernel)
        filtered_motions = mot_yx.copy()
        filtered_motions[~np.all(np.isfinite(filtered), axis=0)] = -1
        self.assertTrue(np.any(filtered_motions < 0))
        self.assertFalse(np.allclose(filtered, residual))
        kwargs = dict(
            h_mots=g["h_mots"],
            unique_motion_to_keep_yx=g["umyx"],
            sel_pix_idxs=g["sel_pix_idxs"],
            pixel_coords=coords,
            n_pixels=n_pix,
            d_xy=5,
            sparse_fac=float(np.exp(-3.0)),
            outer_loop_iters=3,
            mult_nmf_max_iters=3,
        )
        torch.manual_seed(3)
        expected = nmf.fit_sources(
            seeds.copy(), filtered, mot_inds_yx=filtered_motions, **kwargs
        )
        torch.manual_seed(3)
        actual = nmf.fit_sources(
            seeds.copy(), residual, mot_inds_yx=mot_yx,
            temporal_kernel=kernel, **kwargs
        )
        np.testing.assert_array_equal(residual, original)
        self.assertEqual(actual["n_sources"], expected["n_sources"])
        for key in ("A", "phi_low_res", "source_params", "source_snr"):
            np.testing.assert_allclose(
                actual[key], expected[key], rtol=1e-5, atol=1e-6,
                err_msg=key,
            )
        np.testing.assert_array_equal(
            actual["source_seeds"], expected["source_seeds"]
        )

    def test_temporal_filter_dropped_frames_survive_pruning(self):
        """Missing samples must not poison otherwise usable source fits."""
        g, seeds, residual, mot_yx, coords, n_pix = self._inputs()
        residual = np.tile(residual, (1, 16))
        mot_yx = np.tile(mot_yx, 16)
        mot_yx[128] = -1
        original_motions = mot_yx.copy()
        kernel = activity.decay_kernel_1d(0.15, 100)
        for missing_value in (np.nan, 1e6):
            with self.subTest(missing_value=missing_value):
                residual[:, 128] = missing_value
                original = residual.copy()
                torch.manual_seed(7)
                out = nmf.fit_sources(
                    seeds.copy(), residual, g["h_mots"], g["umyx"],
                    mot_yx, g["sel_pix_idxs"], coords, n_pix,
                    d_xy=5, sparse_fac=float(np.exp(-3.0)),
                    outer_loop_iters=3, mult_nmf_max_iters=4,
                    temporal_kernel=kernel,
                )
                self.assertGreater(out["n_sources"], 0)
                self.assertTrue(np.all(np.isfinite(out["source_snr"])))
                self.assertTrue(torch.all(torch.isfinite(out["A"])))
                self.assertTrue(torch.all(torch.isnan(out["phi_low_res"][128])))
                self.assertTrue(torch.any(torch.isfinite(out["phi_low_res"])))
                np.testing.assert_array_equal(residual, original)
                np.testing.assert_array_equal(mot_yx, original_motions)

    def test_temporal_filter_excludes_empty_motion_bins(self):
        """Low coverage can remove an entire bin without poisoning Adam."""
        g, seeds, residual, mot_yx, coords, n_pix = self._inputs()
        # Only the first frame uses bin 0; a uniform kernel rejects both
        # edges. An isolated row-wise NaN also rejects nearby frames.
        mot_yx[:] = 1
        mot_yx[0] = 0
        residual[0, 8] = np.nan
        kernel = np.ones(3) / 3
        filtered = activity.smooth_rho(residual.copy(), kernel)
        expected_motions = np.full_like(mot_yx, -1)
        expected_motions[np.all(np.isfinite(filtered), axis=0)] = 0
        kwargs = dict(
            sel_pix_idxs=g["sel_pix_idxs"], pixel_coords=coords,
            n_pixels=n_pix, d_xy=5, sparse_fac=0.05,
            outer_loop_iters=1, mult_nmf_max_iters=3,
        )
        torch.manual_seed(7)
        expected = nmf.fit_sources(
            seeds.copy(), filtered, [g["h_mots"][1]], g["umyx"][1:],
            expected_motions, **kwargs,
        )
        torch.manual_seed(7)
        actual = nmf.fit_sources(
            seeds.copy(), residual, g["h_mots"], g["umyx"], mot_yx,
            temporal_kernel=kernel, **kwargs,
        )
        self.assertTrue(torch.all(torch.isfinite(actual["A"])))
        for key in ("A", "phi_low_res", "source_params", "source_snr"):
            np.testing.assert_allclose(actual[key], expected[key])
        excluded = expected_motions < 0
        self.assertTrue(
            torch.all(torch.isnan(actual["phi_low_res"][excluded]))
        )

    def test_temporal_filter_no_usable_frames(self):
        """Report missing data explicitly instead of silently pruning all seeds."""
        g, seeds, residual, mot_yx, coords, n_pix = self._inputs()
        residual[:] = np.nan
        with self.assertRaisesRegex(ValueError, "No usable frames remain"):
            nmf.fit_sources(
                seeds, residual, g["h_mots"], g["umyx"], mot_yx,
                g["sel_pix_idxs"], coords, n_pix, d_xy=5,
                sparse_fac=0.05, temporal_kernel=np.ones(1),
            )

    def test_adam_early_convergence_break(self):
        """A huge gd_tol triggers the Adam convergence break."""
        g, seeds, residual, mot_yx, coords, n_pix = self._inputs(seed=2)
        torch.manual_seed(5)
        out = nmf.fit_sources(
            seeds.copy(),
            residual,
            g["h_mots"],
            g["umyx"],
            mot_yx,
            g["sel_pix_idxs"],
            coords,
            n_pix,
            d_xy=5,
            sparse_fac=float(np.exp(-3.0)),
            outer_loop_iters=1,
            mult_nmf_max_iters=2,
            gd_tol=1e9,
        )
        self.assertEqual(out["n_sources"], 3)


if __name__ == "__main__":
    unittest.main()
