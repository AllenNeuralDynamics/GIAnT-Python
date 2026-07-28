"""Tests for giant_python.bandsilo.nmf (Phase 6 source localization).

Covers the profile/init kernels, the least-squares/SNR helpers, and the full
``fit_sources`` driver (multiplicative NMF + Adam Gaussian fit + variance
sort + SNR pruning) on a small synthetic geometry. The driver is additionally
cross-checked bit-for-bit against a verbatim copy of the reference (under a
shared torch seed) in the project's development notes; here we assert
structural behavior and determinism.
"""

import unittest

import numpy as np
import torch

from giant_python.bandsilo import background as bg
from giant_python.bandsilo import nmf
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


class TestPhiAndSnr(unittest.TestCase):
    """fit_phi_all_motions / build_x_support_mots / compute_source_snr."""

    def _setup(self, seed=0):
        """Build A / data / geometry for a couple of sources."""
        g = _small_geometry(seed=seed)
        npc, npr, nz = g["npc"], g["npr"], g["num_fast_zs"]
        n_pix = nz * npc * npr
        sel = g["sel_pix_idxs"]
        pct = torch.tensor(
            bg.pixel_coords_from_idxs(sel, npc, npr), dtype=torch.float32
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
        coords = bg.pixel_coords_from_idxs(sel, npc, npr)
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
