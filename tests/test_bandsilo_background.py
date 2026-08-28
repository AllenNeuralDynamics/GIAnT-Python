"""Tests for giant_python.bandsilo.background (Phase 4 kernels).

Covers the motion-binning, PSF/convolution, interpolated-background,
noise-model, and rho kernels with synthetic fixtures. ``build_interp_data`` is
additionally cross-checked against a verbatim copy of the reference in the
project's development notes; here we assert its structural behavior. The rho
path is exercised end-to-end on a small synthetic geometry built from the real
:mod:`giant_python.bandsilo.geometry` helpers, and checked for linearity in the
residual (rho is a linear projection of the residual).
"""

import unittest

import numpy as np
import torch

from giant_python.bandsilo import background as bg
from giant_python.bandsilo import geometry as geo


def _small_geometry(seed=0, positions_per_plane=None):
    """Build a small consistent DMD geometry for rho/convolution tests.

    Returns a dict of the geometry arrays, PSF tensors, selected pixels, per
    plane convolution matrices, and per-motion H matrices.

    ``positions_per_plane`` is a list (one entry per z-plane) of ``(row, col)``
    reference-pixel positions. It defaults to a dense 7x7 block in every plane
    so the wide horizontal erosion in compute_rho leaves a non-empty valid
    region.
    """
    rng = np.random.default_rng(seed)
    npc = 15  # dmd_pixels_per_column
    npr = 15  # dmd_pixels_per_row
    plane_size = npc * npr

    if positions_per_plane is None:
        block = [(r, c) for r in range(4, 11) for c in range(4, 11)]
        positions_per_plane = [block, block]
    num_fast_zs = len(positions_per_plane)

    rows = []
    for z, positions in enumerate(positions_per_plane):
        for r, c in positions:
            rows.append([z * plane_size + r * npr + c, len(rows) + 1])
    subsample_matrix_inds = np.array(rows, dtype=np.int32)
    num_super_pixels = subsample_matrix_inds.shape[0]

    # A small normalized 5x5 PSF (odd so it has a well-defined center).
    yy, xx = np.mgrid[-2:3, -2:3]
    psf2d = np.exp(-(yy**2 + xx**2) / 2.0).astype(np.float32)

    ref_d, ref_c, ref_r = geo.ref_pixs_to_drc(
        subsample_matrix_inds[:, 0], npc, npr
    )
    sparse_h_inds, sparse_h_vals = geo.build_sparse_h(
        subsample_matrix_inds, psf2d, npc, npr
    )

    umyx = np.array([[0, 0], [1, 0]], dtype=np.float64)
    sel_pix_mask, sel_pix_idxs = bg.build_selected_pixel_mask(
        umyx, ref_d, ref_r, ref_c, num_fast_zs, npc, npr, psf2d
    )

    psf_tensor, psf_center, psf_tensor_exp, psf_center_exp = bg.expand_psf(
        psf2d
    )

    d_mats = []
    d_mats_exp = []
    for z in range(num_fast_zs):
        _, sel_2d = bg.selected_pixels_2d_for_plane(sel_pix_idxs, z, npc, npr)
        d_mats.append(
            bg.build_convolution_matrix(sel_2d, psf_tensor, psf_center)
        )
        d_mats_exp.append(
            bg.build_convolution_matrix(sel_2d, psf_tensor_exp, psf_center_exp)
        )

    h_mots = bg.build_motion_h_matrices(
        sparse_h_inds, sparse_h_vals, umyx, sel_pix_idxs, num_super_pixels, npr
    )

    return dict(
        rng=rng,
        npc=npc,
        npr=npr,
        num_fast_zs=num_fast_zs,
        num_super_pixels=num_super_pixels,
        psf2d=psf2d,
        ref_d=ref_d,
        ref_r=ref_r,
        ref_c=ref_c,
        umyx=umyx,
        sel_pix_idxs=sel_pix_idxs,
        d_mats=d_mats,
        d_mats_exp=d_mats_exp,
        h_mots=h_mots,
    )


class TestMotionBinning(unittest.TestCase):
    """bin_motion / select_motion_bins / bin_motion_yx."""

    def test_bin_motion_groups_identical_rounded_vectors(self):
        """Frames with equal rounded motion share a bin index."""
        mr = np.array([0.1, 0.2, 5.0])
        mc = np.array([0.0, 0.0, 0.0])
        mz = np.array([0.0, 0.0, 0.0])
        unique_motion, mot_inds = bg.bin_motion(mr, mc, mz)
        self.assertEqual(unique_motion.shape[1], 3)
        self.assertEqual(mot_inds[0], mot_inds[1])
        self.assertNotEqual(mot_inds[0], mot_inds[2])
        self.assertEqual(mot_inds.ndim, 1)

    def test_select_motion_bins_filters_z_and_count(self):
        """Bins far in z or with too few frames are dropped."""
        # bin 0: z=0, many frames; bin 1: z=0, few frames; bin 2: z=10, many
        unique_motion = np.array(
            [[0, 0, 0], [1, 0, 0], [2, 0, 10]], dtype=float
        )
        mot_inds = np.array([0] * 5 + [1] * 2 + [2] * 5)
        motion_z = np.zeros(12)
        keep, frames = bg.select_motion_bins(
            unique_motion, mot_inds, motion_z, z_thresh=1.5, min_frames=3
        )
        self.assertEqual(list(keep), [0])
        self.assertEqual(int(frames.sum()), 5)

    def test_bin_motion_yx_marks_dropped_frames(self):
        """Frames outside frames_to_keep get index -1."""
        mr = np.array([0.0, 0.0, 1.0, 1.0])
        mc = np.array([0.0, 0.0, 0.0, 0.0])
        frames_to_keep = np.array([True, True, True, False])
        umyx, mot_inds_yx = bg.bin_motion_yx(mr, mc, frames_to_keep)
        self.assertEqual(mot_inds_yx[3], -1)
        self.assertEqual(mot_inds_yx[0], mot_inds_yx[1])
        self.assertEqual(umyx.shape[1], 2)


class TestSelectedPixels(unittest.TestCase):
    """build_selected_pixel_mask / pixel_coords / selected_pixels_2d."""

    def test_mask_and_coords_round_trip(self):
        """Flat selected indices decode back to their z/row/col."""
        ref_d = np.array([0])
        ref_r = np.array([5])
        ref_c = np.array([5])
        umyx = np.array([[0, 0]], dtype=float)
        psf2d = np.ones((3, 3), dtype=np.float32)
        mask, idxs = bg.build_selected_pixel_mask(
            umyx, ref_d, ref_r, ref_c, 1, 12, 12, psf2d
        )
        self.assertEqual(int(mask.sum()), idxs.size)
        coords = bg.pixel_coords_from_idxs(idxs, 12, 12)
        # every decoded coordinate should be marked in the mask
        for z, r, c in coords:
            self.assertTrue(mask[z, r, c])

    def test_selected_pixels_2d_for_plane(self):
        """Plane selection returns the in-plane row/col for that z only."""
        idxs = np.array([0 * 25 + 2 * 5 + 3, 1 * 25 + 4 * 5 + 1])
        z0, sel0 = bg.selected_pixels_2d_for_plane(idxs, 0, 5, 5)
        self.assertEqual(sel0.tolist(), [[2, 3]])
        z1, sel1 = bg.selected_pixels_2d_for_plane(idxs, 1, 5, 5)
        self.assertEqual(sel1.tolist(), [[4, 1]])


class TestPsfConvolution(unittest.TestCase):
    """expand_psf and build_convolution_matrix."""

    def test_expand_psf_normalized_and_larger(self):
        """Both PSFs sum to 1 and the expanded one is ex_fac larger."""
        yy, xx = np.mgrid[-2:3, -2:3]
        psf2d = np.exp(-(yy**2 + xx**2) / 2.0).astype(np.float32)
        pt, pc, pte, pce = bg.expand_psf(psf2d, ex_fac=2)
        self.assertAlmostEqual(float(pt.sum()), 1.0, places=5)
        self.assertAlmostEqual(float(pte.sum()), 1.0, places=5)
        self.assertEqual(tuple(pte.shape), (10, 10))
        self.assertEqual(pc, (2, 2))
        self.assertEqual(pce, (5, 5))

    def test_convolution_matrix_values(self):
        """D[t, s] equals the PSF weight at the target-source offset."""
        yy, xx = np.mgrid[-2:3, -2:3]
        psf2d = np.exp(-(yy**2 + xx**2) / 2.0).astype(np.float32)
        pt, pc, _, _ = bg.expand_psf(psf2d)
        sel_2d = np.array([[5, 5], [6, 5], [5, 8]])
        d = bg.build_convolution_matrix(sel_2d, pt, pc)
        # self entry == center weight
        self.assertAlmostEqual(float(d[0, 0]), float(pt[pc[0], pc[1]]), 5)
        # offset (1, 0): target row+1 -> psf[center+1, center]
        self.assertAlmostEqual(
            float(d[1, 0]), float(pt[pc[0] + 1, pc[1]]), places=5
        )
        # offset (0, 3): rel_col = center + 3 = 5 is outside the 5x5 PSF -> 0
        self.assertEqual(float(d[2, 0]), 0.0)

    def test_convolution_matrix_empty(self):
        """An empty plane yields a 0x0 matrix."""
        pt = torch.ones((3, 3))
        d = bg.build_convolution_matrix(
            np.zeros((0, 2), dtype=int), pt, (1, 1)
        )
        self.assertEqual(tuple(d.shape), (0, 0))


class TestInterpData(unittest.TestCase):
    """build_interp_data structural behavior and edge branches."""

    def test_basic_fill_and_bounds(self):
        """Interpolation fills selected pixels within the motion window."""
        ref_r = np.array([4, 6, 8], dtype=np.int32)
        ref_c = np.array([5, 5, 5], dtype=np.int32)
        data = np.array([[1.0], [3.0], [5.0]], dtype=np.float32)  # one frame
        sel_2d = np.array([[4, 5], [5, 5], [6, 5], [7, 5]])
        umyx = np.array([[0, 0]], dtype=float)
        mot_yx = np.array([0])
        out, rb, cb = bg.build_interp_data(
            data, ref_r, ref_c, sel_2d, umyx, mot_yx
        )
        # exact rows (4, 6) match data; row 5 interpolates to 2.0
        self.assertAlmostEqual(float(out[0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(out[1, 0]), 2.0, places=5)
        self.assertAlmostEqual(float(out[2, 0]), 3.0, places=5)
        self.assertEqual(cb, (5, 5))

    def test_empty_motion_and_missing_column(self):
        """Empty bins and columns absent from the selection are skipped."""
        ref_r = np.array([4, 6], dtype=np.int32)
        ref_c = np.array([3, 3], dtype=np.int32)  # column 3 not selected
        data = np.array([[1.0], [2.0]], dtype=np.float32)
        sel_2d = np.array([[4, 5], [6, 5]])  # only column 5 selected
        umyx = np.array([[0, 0], [1, 0]], dtype=float)
        mot_yx = np.array([0])  # bin 1 has no frames
        out, _, _ = bg.build_interp_data(
            data, ref_r, ref_c, sel_2d, umyx, mot_yx
        )
        # column 3 not in selection -> nothing filled
        self.assertTrue(np.all(np.isnan(out)))

    def test_single_point_column(self):
        """A single-reference-pixel column is copied, not interpolated."""
        ref_r = np.array([5], dtype=np.int32)
        ref_c = np.array([5], dtype=np.int32)
        data = np.array([[7.0]], dtype=np.float32)
        sel_2d = np.array([[5, 5]])
        umyx = np.array([[0, 0]], dtype=float)
        mot_yx = np.array([0])
        out, _, _ = bg.build_interp_data(
            data, ref_r, ref_c, sel_2d, umyx, mot_yx
        )
        self.assertAlmostEqual(float(out[0, 0]), 7.0, places=5)

    def test_all_exact_column_skips_interpolation(self):
        """A multi-point column whose targets all match exactly interpolates
        nothing (the interp branch is empty)."""
        ref_r = np.array([4, 6], dtype=np.int32)
        ref_c = np.array([5, 5], dtype=np.int32)
        data = np.array([[1.0], [3.0]], dtype=np.float32)
        # selected rows are exactly the reference rows -> all exact matches
        sel_2d = np.array([[4, 5], [6, 5]])
        umyx = np.array([[0, 0]], dtype=float)
        mot_yx = np.array([0])
        out, _, _ = bg.build_interp_data(
            data, ref_r, ref_c, sel_2d, umyx, mot_yx
        )
        self.assertAlmostEqual(float(out[0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(out[1, 0]), 3.0, places=5)

    def test_out_of_window_motion(self):
        """A motion shift pushing all refs outside the window fills nothing."""
        ref_r = np.array([4, 6], dtype=np.int32)
        ref_c = np.array([5, 5], dtype=np.int32)
        data = np.array([[1.0], [2.0]], dtype=np.float32)
        sel_2d = np.array([[4, 5], [6, 5]])
        # single bin, but with a huge column shift beyond width via tiny W
        umyx = np.array([[0, 0]], dtype=float)
        mot_yx = np.array([0])
        out, _, _ = bg.build_interp_data(
            data, ref_r, ref_c, sel_2d, umyx, mot_yx, height=1, width=1
        )
        self.assertTrue(np.all(np.isnan(out)))


class TestRollingBaseline(unittest.TestCase):
    """baseline_window_frames + compute_rolling_baseline."""

    def test_window_frames(self):
        """Window length is int(hz * seconds)."""
        self.assertEqual(bg.baseline_window_frames(80.0, 4.0), 320)

    def test_no_nan_matches_brute_median(self):
        """With no NaNs, the baseline is a centered moving median."""

        def brute_rolling_median(data, window):
            n = data.shape[1]
            out = np.full(data.shape, np.nan, dtype=float)
            for r in range(data.shape[0]):
                for i in range(n):
                    lo = max(0, i - (window - 1) // 2)
                    hi = min(n, i + window // 2 + 1)
                    vals = data[r, lo:hi]
                    vals = vals[~np.isnan(vals)]
                    if vals.size:
                        out[r, i] = np.median(vals)
            return out

        data = np.arange(20, dtype=np.float32).reshape(2, 10)
        for window in (3, 5):
            expected = brute_rolling_median(data, window)
            out = bg.compute_rolling_baseline(data.copy(), window)
            np.testing.assert_allclose(out, expected, rtol=1e-5)

    def test_ignores_nan_in_window(self):
        """NaNs are dropped from each window's median."""
        data = np.array([[1.0, np.nan, 3.0, 100.0, 5.0]], dtype=np.float32)
        out = bg.compute_rolling_baseline(data.copy(), 3)
        # window at idx 1 sees {1, 3} -> median 2; idx 2 sees {3, 100} -> 51.5
        np.testing.assert_allclose(out[0, 0], 1.0)
        np.testing.assert_allclose(out[0, 1], 2.0)
        np.testing.assert_allclose(out[0, 2], 51.5)

    def test_all_nan_row_stays_nan(self):
        """A fully-NaN row yields a fully-NaN baseline."""
        data = np.full((1, 6), np.nan, dtype=np.float32)
        out = bg.compute_rolling_baseline(data, 3)
        self.assertTrue(np.all(np.isnan(out)))


class TestBackgroundAssembly(unittest.TestCase):
    """assemble_background maps selected-pixel background to superpixels."""

    def test_maps_reference_pixels(self):
        """Each superpixel receives its shifted reference-pixel background."""
        npc = npr = 10
        plane = npc * npr
        ref_d = np.array([0, 0])
        ref_r = np.array([2, 4])
        ref_c = np.array([3, 5])
        # sel pixels are exactly the (unshifted) reference pixels
        sel_pix_idxs = np.sort(
            np.array([0 * plane + 2 * npr + 3, 0 * plane + 4 * npr + 5])
        )
        interp_bg = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        umyx = np.array([[0, 0]], dtype=float)
        mot_yx = np.array([0, 0])
        out = bg.assemble_background(
            interp_bg,
            umyx,
            mot_yx,
            sel_pix_idxs,
            ref_d,
            ref_r,
            ref_c,
            n_super_pixels=2,
            n_frames=2,
            dmd_pixels_per_column=npc,
            dmd_pixels_per_row=npr,
        )
        self.assertFalse(np.any(np.isnan(out)))
        self.assertEqual(out.shape, (2, 2))


class TestNoiseModel(unittest.TestCase):
    """fit_noise_variance_model + compute_residual."""

    def test_data_std_positive_and_residual_zscored(self):
        """data_std is positive; residual is (data-bg)/std."""
        rng = np.random.default_rng(1)
        # Per-pixel mean/count spread so var_pred varies (else the bright
        # selection is empty and the fit degenerates).
        means = np.linspace(1.0, 10.0, 30).astype(np.float32)
        background = np.tile(means[:, None], (1, 200))
        low = background + rng.standard_normal((30, 200)).astype(np.float32)
        v_im = np.tile(
            (0.05 + 0.01 * np.arange(30)).astype(np.float32)[:, None],
            (1, 200),
        )
        data_std, v_k, v_b = bg.fit_noise_variance_model(
            low, background, v_im, vif=1.38
        )
        self.assertTrue(np.all(data_std > 0))
        self.assertTrue(np.isfinite(v_b))
        residual = bg.compute_residual(low, background, data_std)
        np.testing.assert_allclose(
            residual, (low - background) / data_std, rtol=1e-6
        )


class TestRho(unittest.TestCase):
    """build_motion_h_matrices + compute_rho + smoothing."""

    def test_h_matrices_shape(self):
        """One sparse H per motion bin, sized (superpixels, sel pixels)."""
        g = _small_geometry()
        self.assertEqual(len(g["h_mots"]), len(g["umyx"]))
        h0 = g["h_mots"][0]
        self.assertEqual(
            tuple(h0.size()),
            (g["num_super_pixels"], g["sel_pix_idxs"].shape[0]),
        )

    def test_compute_rho_shape_and_linearity(self):
        """rho has the selected-pixel shape and is linear in the residual."""
        g = _small_geometry()
        n_frames = 12
        n_sp = g["num_super_pixels"]
        residual = (
            g["rng"].standard_normal((n_sp, n_frames)).astype(np.float32)
        )
        # assign frames to the two motion bins
        mot_yx = np.array([0, 1] * (n_frames // 2), dtype=np.int32)

        kwargs = dict(
            unique_motion_to_keep_yx=g["umyx"],
            h_mots=g["h_mots"],
            d_mats=g["d_mats"],
            d_mats_expanded=g["d_mats_exp"],
            sel_pix_idxs=g["sel_pix_idxs"],
            ref_d=g["ref_d"],
            ref_r=g["ref_r"],
            ref_c=g["ref_c"],
            num_fast_zs=g["num_fast_zs"],
            dmd_pixels_per_column=g["npc"],
            dmd_pixels_per_row=g["npr"],
            psf2d=g["psf2d"],
        )
        rho = bg.compute_rho(residual, mot_yx, **kwargs)
        self.assertEqual(rho.shape, (g["sel_pix_idxs"].shape[0], n_frames))
        self.assertTrue(np.any(np.isfinite(rho)))

        rho2 = bg.compute_rho(2.0 * residual, mot_yx, **kwargs)
        finite = np.isfinite(rho) & np.isfinite(rho2)
        np.testing.assert_allclose(
            rho2[finite], 2.0 * rho[finite], rtol=1e-4, atol=1e-5
        )

    def test_compute_rho_handles_empty_motion_and_empty_plane(self):
        """A frameless motion bin and an empty z-plane are skipped."""
        g = _small_geometry()
        n_frames = 6
        residual = (
            g["rng"]
            .standard_normal((g["num_super_pixels"], n_frames))
            .astype(np.float32)
        )
        # All frames in bin 0; bin 1 gets none -> motion_frames empty branch
        mot_yx = np.zeros(n_frames, dtype=np.int32)
        # Add a third, always-empty z-plane -> new_ncols == 0 branch
        rho = bg.compute_rho(
            residual,
            mot_yx,
            unique_motion_to_keep_yx=g["umyx"],
            h_mots=g["h_mots"],
            d_mats=g["d_mats"] + [torch.zeros((0, 0))],
            d_mats_expanded=g["d_mats_exp"] + [torch.zeros((0, 0))],
            sel_pix_idxs=g["sel_pix_idxs"],
            ref_d=g["ref_d"],
            ref_r=g["ref_r"],
            ref_c=g["ref_c"],
            num_fast_zs=g["num_fast_zs"] + 1,
            dmd_pixels_per_column=g["npc"],
            dmd_pixels_per_row=g["npr"],
            psf2d=g["psf2d"],
        )
        self.assertEqual(rho.shape[1], n_frames)

    def test_compute_rho_skips_plane_with_no_valid_columns(self):
        """A plane whose eroded valid region is empty is skipped (its rho
        rows stay NaN) while the dense plane is still filled."""
        block = [(r, c) for r in range(4, 11) for c in range(4, 11)]
        # plane 1 has a single reference pixel -> erosion wipes its valid
        # region even though it has selected pixels and H columns.
        g = _small_geometry(positions_per_plane=[block, [(7, 7)]])
        n_frames = 6
        residual = (
            g["rng"]
            .standard_normal((g["num_super_pixels"], n_frames))
            .astype(np.float32)
        )
        mot_yx = np.array([0, 1] * (n_frames // 2), dtype=np.int32)
        rho = bg.compute_rho(
            residual,
            mot_yx,
            unique_motion_to_keep_yx=g["umyx"],
            h_mots=g["h_mots"],
            d_mats=g["d_mats"],
            d_mats_expanded=g["d_mats_exp"],
            sel_pix_idxs=g["sel_pix_idxs"],
            ref_d=g["ref_d"],
            ref_r=g["ref_r"],
            ref_c=g["ref_c"],
            num_fast_zs=g["num_fast_zs"],
            dmd_pixels_per_column=g["npc"],
            dmd_pixels_per_row=g["npr"],
            psf2d=g["psf2d"],
        )
        plane_size = g["npc"] * g["npr"]
        z_of_sel = g["sel_pix_idxs"] // plane_size
        # plane 0 produced some finite rho; plane 1 (single pixel) did not
        self.assertTrue(np.any(np.isfinite(rho[z_of_sel == 0])))
        self.assertTrue(np.all(np.isnan(rho[z_of_sel == 1])))

    def test_mask_high_nan_rho(self):
        """Rows that are mostly NaN are fully NaN'd; fraction is returned."""
        rho = np.array(
            [[1.0, 2.0, 3.0, 4.0], [np.nan, np.nan, np.nan, 1.0]],
            dtype=np.float32,
        )
        nan_ct = bg.mask_high_nan_rho(rho, thresh=0.5)
        self.assertAlmostEqual(nan_ct[0], 0.0)
        self.assertAlmostEqual(nan_ct[1], 0.75)
        self.assertTrue(np.all(np.isnan(rho[1])))
        self.assertFalse(np.any(np.isnan(rho[0])))

    def test_decay_kernel_normalized(self):
        """The decay kernel sums to 1 and rises to its last sample."""
        k = bg.decay_kernel_1d(0.15, 80.0)
        self.assertAlmostEqual(float(k.sum()), 1.0, places=6)
        self.assertEqual(np.argmax(k), len(k) - 1)

    def test_smooth_rho_preserves_constant(self):
        """A finite constant row is preserved; an all-NaN row stays NaN."""
        rho = np.vstack(
            [
                np.full((1, 50), 2.0, dtype=np.float32),
                np.full((1, 50), np.nan, dtype=np.float32),
            ]
        )
        k = bg.decay_kernel_1d(0.05, 80.0)
        out = bg.smooth_rho(rho, k)
        # interior samples of the constant row remain ~2.0
        np.testing.assert_allclose(out[0, 10:40], 2.0, rtol=1e-4)
        self.assertTrue(np.all(np.isnan(out[1])))

    def test_smooth_rho_all_nan_chunk(self):
        """A chunk with no finite data is passed through unchanged."""
        rho = np.full((3, 20), np.nan, dtype=np.float32)
        k = bg.decay_kernel_1d(0.05, 80.0)
        out = bg.smooth_rho(rho, k)
        self.assertTrue(np.all(np.isnan(out)))


if __name__ == "__main__":
    unittest.main()
