"""Tests for canonical band traces (Phase 7 high-res traces).

Covers the per-trial compute (motion interp/binning, per-motion least-squares
``phi``/``F0``, global and per-ROI fluorescence) and the IO wrapper's
skip/monkeypatched-read paths. ``compute_high_res_traces`` is additionally
cross-checked against a verbatim copy of the reference in the project's
development notes; here we assert structural behavior on synthetic data.
"""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from giant_python.extraction.band import operators
from giant_python.extraction.band import traces as tr
from giant_python.extraction.band.types import TrialTraceResult
from giant_python.extraction.band.transient import TransientOptions, TransientResult


def _geometry(npc=15, npr=15, num_fast_zs=2):
    """Build a small consistent geometry (subsample map + sparse H + PSF)."""
    plane = npc * npr
    positions = [(r, c) for r in range(4, 11) for c in range(4, 11)]
    rows = []
    for z in range(num_fast_zs):
        for r, c in positions:
            rows.append([z * plane + r * npr + c, len(rows) + 1])
    smi = np.array(rows, dtype=np.int32)
    yy, xx = np.mgrid[-2:3, -2:3]
    psf2d = np.exp(-(yy**2 + xx**2) / 2.0).astype(np.float32)
    sh_inds, sh_vals = operators.build_sparse_h(smi, psf2d, npc, npr)
    return dict(
        npc=npc,
        npr=npr,
        nz=num_fast_zs,
        nsp=smi.shape[0],
        smi=smi,
        psf2d=psf2d,
        sh_inds=sh_inds,
        sh_vals=sh_vals,
    )


def _trial_inputs(num_channels=1, unique_motion_ds=None, seed=2):
    """Synthetic loaded-trial arrays + geometry for compute_high_res_traces."""
    g = _geometry()
    rng = np.random.default_rng(seed)
    nsp = g["nsp"]
    n_ds = 60
    n_frames = 40
    ds_frames = np.arange(n_ds, dtype=float)
    frames = np.linspace(1, n_ds - 2, n_frames)
    a_data = dict(
        DSframes=ds_frames,
        motionDSr=(np.arange(n_ds) % 2).astype(float),  # bins 0 and 1
        motionDSc=np.zeros(n_ds),
        motionDSz=np.zeros(n_ds),
        onlineYshift=rng.standard_normal(n_ds),
        onlineXshift=rng.standard_normal(n_ds),
        onlineZshift=rng.standard_normal(n_ds),
    )
    data = rng.standard_normal((nsp, n_frames)).astype(np.float32) + 2
    data2 = (
        rng.standard_normal((nsp, n_frames)).astype(np.float32) + 1
        if num_channels >= 2
        else None
    )
    background_ds = rng.standard_normal((nsp, n_ds)).astype(np.float32) + 1
    if unique_motion_ds is None:
        unique_motion_ds = np.array([[0, 0], [1, 0]], dtype=float)
    mot_keep_ds = np.arange(unique_motion_ds.shape[0])
    n_pixels = g["nz"] * g["npc"] * g["npr"]
    a_final = torch.rand((n_pixels, 3), dtype=torch.float32)
    soma_sps = [np.array([0, 1, 2]), np.array([5, 6, 7, 8])]
    return dict(
        g=g,
        data=data,
        data2=data2,
        background_ds=background_ds,
        frames=frames,
        a_data=a_data,
        a_final=a_final,
        unique_motion_ds=unique_motion_ds,
        mot_keep_ds=mot_keep_ds,
        soma_sps=soma_sps,
        num_channels=num_channels,
        n_frames=n_frames,
    )


def _compute(inp, **kwargs):
    """Call compute_high_res_traces with the assembled inputs dict."""
    g = inp["g"]
    return tr.compute_high_res_traces(
        inp["data"],
        inp["data2"],
        inp["background_ds"],
        inp["frames"],
        inp["a_data"],
        g["smi"],
        g["sh_inds"],
        g["sh_vals"],
        inp["a_final"],
        inp["unique_motion_ds"],
        inp["mot_keep_ds"],
        0.0,
        g["psf2d"],
        g["nsp"],
        g["nz"],
        g["npc"],
        g["npr"],
        inp["num_channels"],
        inp["soma_sps"],
        **kwargs,
    )


class TestTrialTraceResult(unittest.TestCase):
    """The original eight positional fields retain their identity and order."""

    def test_legacy_constructor_defaults_and_extended_fields(self):
        original = _compute(_trial_inputs())
        legacy = TrialTraceResult(*original[:8])
        self.assertEqual(len(legacy), 10)
        self.assertEqual(legacy._fields, (
            "d_f", "f0_ls", "frame_line_idxs", "selected_pixels", "global_f",
            "reference_offsets", "online_shifts", "user_roi_f",
            "d_f_denoised", "events",
        ))
        self.assertIsNone(legacy.d_f_denoised)
        self.assertIsNone(legacy.events)
        denoised = np.full_like(legacy.d_f, 13)
        events = np.full_like(legacy.d_f, 2)
        partial = TrialTraceResult(*legacy[:8], denoised)
        self.assertIs(partial.d_f_denoised, denoised)
        self.assertIsNone(partial.events)
        extended = TrialTraceResult(*legacy[:8], denoised, events)
        for index, value in enumerate(original[:8]):
            self.assertIs(legacy[index], value)
            self.assertIs(extended[index], value)
        self.assertIs(extended.d_f_denoised, denoised)
        self.assertIs(extended.events, events)
        self.assertIs(extended[8], denoised)
        self.assertIs(extended[9], events)


class TestWeightedTraceAverage(unittest.TestCase):
    """Simple traces use profile intensities, not a regression coefficient."""

    def test_normalized_intensities_and_scale_invariance(self):
        """Overlapping and duplicate profiles are averaged independently."""
        profiles = torch.tensor([[1., 0., 7.], [3., 2., 21.], [0., 2., 0.]])
        data = torch.tensor([[2., -2.], [6., 4.], [10., 8.]])
        expected = np.array([[5., 8., 5.], [2.5, 6., 2.5]])
        actual = tr._weighted_trace_average(profiles, data)
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(
            tr._weighted_trace_average(profiles * 0.01, data),
            expected, rtol=1e-6,
        )

    def test_uniform_zero_and_empty_profiles(self):
        """Uniform weights give the mean; no support is NaN, not zero."""
        data = torch.tensor([[2., 4.], [6., 8.]])
        profiles = torch.tensor([[3., 0.], [3., 0.]])
        result = tr._weighted_trace_average(profiles, data)
        np.testing.assert_allclose(result[:, 0], [4., 6.])
        self.assertTrue(torch.isnan(result[:, 1]).all())
        self.assertEqual(
            tr._weighted_trace_average(torch.empty(2, 0), data).shape,
            (2, 0),
        )

    def test_motion_projection_background_and_dropped_frames(self):
        """Normalize after motion projection and use identical dF/F0 weights."""
        profiles = torch.tensor([[1., 2.], [3., 0.], [0., 4.]])
        operators_by_motion = [
            torch.eye(3).to_sparse(),
            torch.tensor([[0., 2., 0.], [0., 0., 1.], [0., 0., 0.]]).to_sparse(),
        ]
        data = np.array([[2., 4., 8., 1.], [6., 8., 2., 2.],
                         [10., 12., 4., 3.]], dtype=np.float32)
        background = np.array([[1., 1., 2., 0.], [2., 2., 1., 0.],
                               [3., 3., 3., 0.]], dtype=np.float32)
        bins = np.array([0, 0, 1, -1])
        args = (
            data, background, np.array([[0, 0], [1, 0]]), bins,
            np.array([0, 1]), np.empty((0, 2)), np.empty(0), np.arange(3),
            profiles, 3, 3,
        )
        with patch.object(
            tr, "build_motion_h_matrices", return_value=operators_by_motion
        ), patch.object(
            tr, "solve_phi_motion", side_effect=AssertionError("LS called")
        ):
            phi, f0 = tr._solve_trial_phi_f0(
                *args, simple_trace_extraction=True
            )
        for motion, h in enumerate(operators_by_motion):
            selected = np.flatnonzero(bins == motion)
            projected = (h.to_dense() @ profiles).numpy()
            for source in range(profiles.shape[1]):
                weights = projected[:, source]
                np.testing.assert_allclose(
                    phi[selected, source],
                    np.average((data - background)[:, selected], axis=0,
                               weights=weights), rtol=1e-6,
                )
                np.testing.assert_allclose(
                    f0[selected, source],
                    np.average(background[:, selected], axis=0,
                               weights=weights), rtol=1e-6,
                )
                np.testing.assert_allclose(
                    (phi + f0)[selected, source],
                    np.average(data[:, selected], axis=0, weights=weights),
                    rtol=1e-6,
                )
        self.assertTrue(torch.isnan(phi[-1]).all())
        self.assertTrue(torch.isnan(f0[-1]).all())


class TestLeastSquaresTraceExtraction(unittest.TestCase):
    """Missing high-resolution observations must not abort a trial."""

    def test_transient_outputs_never_replace_qr_and_default_return_stays_pair(self):
        """Distinct fit payloads stay separate, including their missing masks."""
        profiles = torch.eye(2)
        h_mots = [torch.eye(2).to_sparse(), (2 * torch.eye(2)).to_sparse()]
        data = np.array([
            [-4., 2., 10., 4., 5., np.inf, 14., 8.],
            [3., np.nan, 6., 8., 1., 3., 10., 4.],
        ], dtype=np.float32)
        background = np.ones_like(data)
        background[0, 3] = np.nan
        bins = np.array([0, 0, 1, 1, -1, 0, 1, 0])
        valid = np.array([True, False, True, False, False, False, True, True])
        denoised = np.arange(16, dtype=np.float32).reshape(8, 2) + 100
        denoised[~valid] = np.nan
        latent_events = np.arange(16, dtype=np.float32).reshape(8, 2) + 20
        latent_events[0] = 0  # Initial state is not a new event.
        expected_events = latent_events.copy()
        expected_events[~valid] = np.nan
        args = (
            data, background, np.array([[0, 0], [1, 0]]), bins,
            np.array([0, 1]), np.empty((0, 2)), np.empty(0), np.arange(2),
            profiles, 2, 2,
        )
        options = TransientOptions()

        def fitted(*args, **kwargs):
            """Return fresh arrays so masking cannot affect later calls."""
            return TransientResult(
                d_f=denoised.copy(), events=latent_events.copy(),
                initial_state=np.array([100., 101.]),
                model_trace=np.full((8, 2), 999., dtype=np.float32),
                metadata={"converged": True},
            )

        with patch.object(tr, "build_motion_h_matrices", return_value=h_mots):
            qr = tr._solve_trial_phi_f0(*args)
            no_fit = tr._solve_trial_phi_f0(*args, return_transient=True)
            with patch.object(tr, "fit_transient_traces", side_effect=fitted) as fit:
                pair = tr._solve_trial_phi_f0(
                    *args, transient_options=options, analyze_hz=50,
                )
                extended = tr._solve_trial_phi_f0(
                    *args, transient_options=options, analyze_hz=50,
                    return_transient=True,
                )
        self.assertEqual(len(qr), 2)
        self.assertEqual(len(pair), 2)
        self.assertEqual(len(extended), 4)
        self.assertEqual(len(no_fit), 4)
        self.assertEqual(fit.call_count, 2)
        for result in (pair, extended, no_fit):
            for actual, expected in zip(result[:2], qr):
                torch.testing.assert_close(actual, expected, equal_nan=True)
        np.testing.assert_array_equal(qr[0][0], [-5., 2.])
        np.testing.assert_array_equal(qr[0][2], [4.5, 2.5])
        np.testing.assert_array_equal(qr[1][[1, 5]], np.ones((2, 2)))
        np.testing.assert_array_equal(extended[2], denoised)
        np.testing.assert_array_equal(extended[3], expected_events)
        self.assertTrue(torch.isnan(no_fit[2]).all())
        self.assertTrue(torch.isnan(no_fit[3]).all())
        for call in fit.call_args_list:
            np.testing.assert_array_equal(call.args[0], data - background)
            np.testing.assert_array_equal(call.args[1][0], np.eye(2))
            np.testing.assert_array_equal(call.args[1][1], 2 * np.eye(2))
            np.testing.assert_array_equal(call.args[2], bins)
            np.testing.assert_array_equal(call.args[3], valid)
            self.assertEqual(call.args[4], 50)
            self.assertIs(call.kwargs["options"], options)

    def test_missing_data_and_background_in_motion_bins(self):
        """dF and F0 retain their own missing-frame masks across motion bins."""
        profiles = torch.tensor([[1., 2.], [3., 0.], [0., 4.]])
        h_mots = [
            torch.eye(3).to_sparse(),
            torch.diag(torch.tensor([2., 1., 3.])).to_sparse(),
        ]
        data = np.arange(24, dtype=np.float32).reshape(3, 8) + 5
        background = np.ones_like(data)
        bins = np.array([0, 0, 0, 1, 1, 1, 1, -1])
        args = (
            data, background, np.array([[0, 0], [1, 0]]), bins,
            np.array([0, 1]), np.empty((0, 2)), np.empty(0), np.arange(3),
            profiles, 3, 3,
        )
        with patch.object(tr, "build_motion_h_matrices", return_value=h_mots):
            expected_phi, expected_f0 = tr._solve_trial_phi_f0(*args)
            data[0, 1] = np.nan
            data[:, 2] = np.nan
            background[1, 3] = np.nan
            data[2, 4] = np.inf
            phi, f0 = tr._solve_trial_phi_f0(*args)

        torch.testing.assert_close(phi[[0, 5, 6]], expected_phi[[0, 5, 6]])
        self.assertTrue(torch.isnan(phi[[1, 2, 3, 4, 7]]).all())
        valid_f0 = [0, 1, 2, 4, 5, 6]
        torch.testing.assert_close(f0[valid_f0], expected_f0[valid_f0])
        self.assertTrue(torch.isnan(f0[[3, 7]]).all())


class TestBinTrialMotion(unittest.TestCase):
    """The configured z tolerance controls trial-frame retention."""

    def test_z_tolerance_and_inclusive_boundaries(self):
        motion_z = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=float)
        args = (
            np.zeros(7), np.zeros(7), motion_z, 0.5,
            np.array([[0, 0]]), np.array([0]),
        )
        default = tr._bin_trial_motion(*args)
        np.testing.assert_array_equal(
            default[3], np.abs(motion_z - 0.5) <= 1.5
        )
        for tolerance in (0.0, 0.5, 2.5, 4.0):
            with self.subTest(z_tol=tolerance):
                _, mot_inds, _, kept = tr._bin_trial_motion(
                    *args, z_tol=tolerance
                )
                expected = np.abs(motion_z - 0.5) <= tolerance
                np.testing.assert_array_equal(kept, expected)
                np.testing.assert_array_equal(mot_inds[~expected], -1)


class TestComputeHighResTraces(unittest.TestCase):
    """compute_high_res_traces per-trial numerics."""

    def test_transient_preserves_qr_masks_geometry_and_auxiliary_outputs(self):
        """A real fit adds denoised/events without replacing any legacy field."""
        inp = _trial_inputs(num_channels=2)
        inp["a_data"]["motionDSz"][20:30] = 5
        inp["data"][0, 4] = np.nan
        inp["data"][1, 5] = np.inf
        inp["background_ds"][0, 0:4] = np.nan
        inp["frames"] = np.ceil(inp["frames"])
        baseline = _compute(inp)
        spatial_before = inp["a_final"].clone()
        options = TransientOptions(max_iter=1500)
        fitted = []
        original_fit = tr.fit_transient_traces

        def fit_trial(*args, **kwargs):
            """Capture the real fit before the wrapper masks exposed events."""
            result = original_fit(*args, **kwargs)
            fitted.append(result)
            return result

        with patch.object(tr, "fit_transient_traces", side_effect=fit_trial) as solve:
            out = _compute(inp, transient_options=options, analyze_hz=50)
        solve.assert_called_once()
        self.assertEqual(solve.call_args.args[4], 50)
        self.assertIs(solve.call_args.kwargs["options"], options)
        self.assertEqual(len(out), 10)
        for actual, expected in zip(out[:8], baseline[:8]):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(
            out.d_f_denoised, fitted[0].d_f.astype(np.float32)
        )
        expected_events = fitted[0].events.astype(np.float32)
        expected_events[~solve.call_args.args[3]] = np.nan
        np.testing.assert_array_equal(out.events, expected_events)
        missing = np.isnan(baseline.d_f)
        self.assertTrue(missing.any())
        self.assertTrue((~missing).any())
        for values in (out.d_f_denoised, out.events):
            self.assertEqual(values.shape, baseline.d_f.shape)
            self.assertEqual(values.dtype, np.float32)
            np.testing.assert_array_equal(np.isnan(values), missing)
            self.assertTrue((values[~missing] >= 0).all())
            self.assertFalse(np.shares_memory(values, out.d_f))
        self.assertFalse(np.shares_memory(out.d_f_denoised, out.events))
        self.assertFalse(np.allclose(out.d_f, out.d_f_denoised, equal_nan=True))
        torch.testing.assert_close(inp["a_final"], spatial_before)

    def test_transient_all_missing_no_sources_and_failed_fit(self):
        """Empty trials/sources are safe; failed fits never become normal output."""
        inp = _trial_inputs()
        options = TransientOptions()
        inp["data"][:] = np.nan
        with patch.object(tr, "fit_transient_traces") as solve:
            out = _compute(inp, transient_options=options, analyze_hz=50)
        solve.assert_not_called()
        self.assertTrue(np.isnan(out.d_f).all())
        self.assertTrue(np.isnan(out.d_f_denoised).all())
        self.assertTrue(np.isnan(out.events).all())
        inp["a_final"] = inp["a_final"][:, :0]
        out = _compute(inp, transient_options=options, analyze_hz=50)
        self.assertEqual(out.d_f.shape, (40, 0))
        self.assertEqual(out.d_f_denoised.shape, (40, 0))
        self.assertEqual(out.events.shape, (40, 0))
        inp = _trial_inputs()
        with self.assertWarns(RuntimeWarning), self.assertRaisesRegex(RuntimeError, "did not converge"):
            _compute(inp, transient_options=TransientOptions(max_iter=1), analyze_hz=50)

    def test_transient_requires_explicit_rate_and_ordered_frames(self):
        """Line positions must not be interpreted as time in seconds."""
        inp = _trial_inputs()
        options = TransientOptions()
        with self.assertRaisesRegex(ValueError, "analyze_hz"):
            _compute(inp, transient_options=options)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            _compute(inp, transient_options=options, analyze_hz=50,
                     simple_trace_extraction=True)
        inp["frames"] = inp["frames"][::-1]
        with self.assertRaisesRegex(ValueError, "chronological"):
            _compute(inp, transient_options=options, analyze_hz=50)

    def test_default_remains_least_squares(self):
        """Isolated low-level extraction still permits QR without a fit."""
        inp = _trial_inputs()
        with (
            patch.object(tr, "solve_phi_motion", wraps=tr.solve_phi_motion) as ls,
            patch.object(tr, "fit_transient_traces") as fit,
        ):
            default = _compute(inp)
        fit.assert_not_called()
        self.assertGreater(ls.call_count, 0)
        explicit = _compute(inp, simple_trace_extraction=False)
        np.testing.assert_array_equal(default.d_f, explicit.d_f)
        np.testing.assert_array_equal(default.f0_ls, explicit.f0_ls)
        self.assertTrue(np.isnan(default.d_f_denoised).all())
        self.assertTrue(np.isnan(default.events).all())

    def test_simple_mode_preserves_mask_and_auxiliary_outputs(self):
        """Only dF/F0 extraction changes; z rejection and other channels stay."""
        inp = _trial_inputs(num_channels=2)
        inp["a_data"]["motionDSz"][20:40] = 5
        least_squares = _compute(inp)
        with patch.object(
            tr, "solve_phi_motion", side_effect=AssertionError("LS called")
        ):
            simple = _compute(inp, simple_trace_extraction=True)
        self.assertTrue(np.isnan(simple.d_f).any())
        self.assertTrue(np.isfinite(simple.d_f).any())
        for a, b in ((simple.d_f, least_squares.d_f),
                     (simple.f0_ls, least_squares.f0_ls)):
            self.assertEqual(a.shape, b.shape)
            np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
        for field in ("global_f", "user_roi_f", "frame_line_idxs",
                      "selected_pixels", "reference_offsets", "online_shifts"):
            np.testing.assert_array_equal(
                getattr(simple, field), getattr(least_squares, field)
            )

    def test_single_channel_shapes_and_fit(self):
        """Returns ten fields; the original eight retain their positions."""
        inp = _trial_inputs(num_channels=1)
        out = _compute(inp)
        self.assertIsInstance(out, TrialTraceResult)
        self.assertIs(out.d_f, out[0])
        self.assertIs(out.f0_ls, out[1])
        self.assertIs(out.frame_line_idxs, out[2])
        self.assertIs(out.selected_pixels, out[3])
        self.assertIs(out.global_f, out[4])
        self.assertIs(out.reference_offsets, out[5])
        self.assertIs(out.online_shifts, out[6])
        self.assertIs(out.user_roi_f, out[7])
        self.assertIs(out.d_f_denoised, out[8])
        self.assertIs(out.events, out[9])
        (
            phi, f0, frames, sel, global_f, motion, online, f_soma,
            denoised, events,
        ) = out
        n_sources = inp["a_final"].shape[1]
        self.assertEqual(phi.shape, (inp["n_frames"], n_sources))
        self.assertEqual(f0.shape, (inp["n_frames"], n_sources))
        for values in (denoised, events):
            self.assertEqual(values.shape, phi.shape)
            self.assertTrue(np.isnan(values).all())
        self.assertEqual(frames.shape, (inp["n_frames"],))
        self.assertEqual(global_f.shape, (inp["n_frames"], 1))
        self.assertEqual(
            f_soma.shape, (inp["n_frames"], len(inp["soma_sps"]), 1)
        )
        self.assertEqual(len(motion), 3)
        self.assertEqual(len(online), 3)
        self.assertTrue(np.any(np.isfinite(phi)))

    def test_two_channels(self):
        """A second channel adds columns to global and soma fluorescence."""
        inp = _trial_inputs(num_channels=2)
        out = _compute(inp)
        global_f, f_soma = out.global_f, out.user_roi_f
        self.assertEqual(global_f.shape, (inp["n_frames"], 2))
        self.assertEqual(
            f_soma.shape, (inp["n_frames"], len(inp["soma_sps"]), 2)
        )
        # both channels populated where frames are kept
        self.assertTrue(np.any(np.isfinite(f_soma[:, :, 1])))

    def test_unmatched_ds_bin_skipped(self):
        """A low-res motion bin absent from the trial is skipped."""
        umd = np.array([[0, 0], [1, 0], [5, 5]], dtype=float)  # [5,5] absent
        inp = _trial_inputs(num_channels=1, unique_motion_ds=umd)
        out = _compute(inp)
        # still produces finite fits for the two present bins
        self.assertTrue(np.any(np.isfinite(out[0])))


class TestGetHighResTraces(unittest.TestCase):
    """get_high_res_traces IO wrapper (skip + monkeypatched read)."""

    def test_skipped_trial_returns_empty(self):
        """keep_trial=False returns all ten empty fields without reading."""
        loader = patch.object(
            tr, "_load_high_res_trial_data", side_effect=AssertionError("read skipped trial")
        )
        fit = patch.object(
            tr, "fit_transient_traces", side_effect=AssertionError("fit skipped trial")
        )
        loader.start()
        fit.start()
        self.addCleanup(loader.stop)
        self.addCleanup(fit.stop)
        inp = _trial_inputs(num_channels=2)
        g = inp["g"]
        out = tr.get_high_res_traces(
            (0, False, inp["background_ds"]),
            0,
            100.0,
            np.zeros((g["nsp"], 1), dtype=np.int32),
            "unused",
            {},
            g["smi"],
            g["sh_inds"],
            g["sh_vals"],
            inp["a_final"].numpy(),
            inp["unique_motion_ds"],
            inp["mot_keep_ds"],
            0.0,
            g["psf2d"],
            g["nsp"],
            g["nz"],
            g["npc"],
            g["npr"],
            2,
            inp["soma_sps"],
            transient_options=TransientOptions(),
        )
        self.assertIsInstance(out, TrialTraceResult)
        self.assertEqual(len(out), 10)
        self.assertEqual(out[0].shape, (0, inp["a_final"].shape[1]))
        self.assertEqual(out.d_f_denoised.shape, out.d_f.shape)
        self.assertEqual(out.events.shape, out.d_f.shape)
        self.assertEqual(out[4].shape, (0, 2))  # globalF (0, channels)
        self.assertEqual(out[2].shape, (0,))  # frames

    def test_kept_trial_reads_and_computes(self):
        """A kept trial reads (monkeypatched) then computes the traces."""
        for simple in (False, True):
            with self.subTest(simple_trace_extraction=simple):
                self._check_kept_trial(simple)
        self._check_kept_trial(False, TransientOptions(max_iter=1500))

    def _check_kept_trial(self, simple, transient_options=None):
        """Verify IO wrapper forwards the selected extraction mode."""
        inp = _trial_inputs(num_channels=1)
        g = inp["g"]

        def fake_load(*args, **kwargs):
            """Return the synthetic loaded-trial arrays."""
            return (
                inp["data"],
                inp["data2"],
                inp["a_data"],
                inp["frames"],
            )

        orig = tr._load_high_res_trial_data
        tr._load_high_res_trial_data = fake_load
        try:
            out = tr.get_high_res_traces(
                (0, True, inp["background_ds"]),
                0,
                100.0,
                np.zeros((g["nsp"], 1), dtype=np.int32),
                "unused",
                {},
                g["smi"],
                g["sh_inds"],
                g["sh_vals"],
                inp["a_final"].numpy(),  # numpy -> exercises tensor conversion
                inp["unique_motion_ds"],
                inp["mot_keep_ds"],
                0.0,
                g["psf2d"],
                g["nsp"],
                g["nz"],
                g["npc"],
                g["npr"],
                1,
                inp["soma_sps"],
                simple_trace_extraction=simple,
                transient_options=transient_options,
            )
        finally:
            tr._load_high_res_trial_data = orig

        # matches a direct compute with the same inputs
        expected = _compute(
            inp, simple_trace_extraction=simple,
            transient_options=transient_options, analyze_hz=100,
        )
        for field in ("d_f", "f0_ls", "d_f_denoised", "events"):
            np.testing.assert_allclose(
                getattr(out, field), getattr(expected, field),
                rtol=1e-6, equal_nan=True,
            )
        self.assertEqual(out[0].shape, (inp["n_frames"], 3))


if __name__ == "__main__":
    unittest.main()
