"""Fixed-profile temporal fitting: motion, gaps, boundaries and gradients."""

import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import OptimizeResult

from giant_python.extraction.band import transient as tt


def synthetic_problem():
    """Two overlapping sources, alternating motion and an observation gap."""
    designs = {
        0: np.array([[1, .2], [.1, 1], [.5, .3]], dtype=np.float32),
        1: np.array([[.2, 1], [1, .1], [.4, .8]], dtype=np.float32),
    }
    events = np.zeros((65, 2))
    events[0] = [.4, .2]
    events[15, 0], events[38, 1] = 2, 3
    expected = tt.TransientOperator(50, .15).forward(events)
    motion = np.arange(65) % 2
    residual = np.stack([
        designs[int(m)] @ expected[t] for t, m in enumerate(motion)
    ], axis=1).astype(np.float32)
    valid = np.ones(65, dtype=bool)
    valid[9:12] = False
    residual[:, 9:12] = np.nan
    return residual, designs, motion, valid, expected, events


class TestTransientSolver(unittest.TestCase):
    """Numerical contracts independent of acquisition files and localization."""

    def test_adjoint_and_gradient(self):
        """Analytic derivatives agree with directional finite differences."""
        rng = np.random.default_rng(11)
        s, z = rng.normal(size=(40, 2)), rng.normal(size=(40, 2))
        groups = {0: np.arange(0, 40, 2), 1: np.arange(1, 40, 2)}
        normal = {}
        for m, ix in groups.items():
            x = rng.normal(size=(3, 2))
            normal[m] = x.T @ x, rng.normal(size=(2, len(ix)))
        for kernel in (None, [1, .8, .2]):
            op = tt.TransientOperator(50, .15, kernel)
            self.assertAlmostEqual(
                np.sum(op.forward(s) * z), np.sum(s * op.adjoint(z)), places=10
            )
            _, grad = tt._temporal_quadratic(normal, groups, op.forward(s))
            direction = rng.normal(size=s.shape)
            eps = 1e-5
            hi = tt._temporal_quadratic(normal, groups, op.forward(s + eps * direction))[0]
            lo = tt._temporal_quadratic(normal, groups, op.forward(s - eps * direction))[0]
            self.assertAlmostEqual(
                (hi - lo) / (2 * eps), np.sum(direction * op.adjoint(grad)), places=5
            )

    def test_recovery_masks_initial_state_and_independent_trials(self):
        """Motion changes and missing frames never reset the decay state."""
        r, designs, motion, valid, expected, events = synthetic_problem()
        options = tt.TransientOptions(beta=0, max_iter=1500, tolerance=1e-12)
        result = tt.fit_transient_traces(r, designs, motion, valid, 50, options=options)
        self.assertTrue(result.metadata["converged"])
        np.testing.assert_allclose(result.d_f[valid], expected[valid], atol=5e-4)
        np.testing.assert_allclose(result.initial_state, events[0], atol=5e-4)
        self.assertTrue(np.isnan(result.d_f[~valid]).all())
        self.assertTrue(np.isfinite(result.model_trace).all())
        self.assertTrue((result.events >= 0).all())
        self.assertTrue((result.events[0] == 0).all())
        self.assertEqual(np.argmax(result.events[:, 0]), 15)
        self.assertEqual(np.argmax(result.events[:, 1]), 38)
        reconstructed = result.events.copy()
        reconstructed[0] = result.initial_state
        np.testing.assert_array_equal(
            result.model_trace, tt.TransientOperator(50, .15).forward(reconstructed)
        )
        second = tt.fit_transient_traces(
            np.zeros_like(r), designs, motion, np.ones_like(valid), 50, options=options
        )
        np.testing.assert_array_equal(second.model_trace, 0)
        # Neither caller masks nor input data are modified.
        np.testing.assert_array_equal(valid, synthetic_problem()[3])
        np.testing.assert_array_equal(r, synthetic_problem()[0])

    def test_debias_and_event_penalty(self):
        """Optional support refitting restores amplitudes without new events."""
        r, designs, motion, valid, expected, events = synthetic_problem()
        settings = dict(beta=.2, max_iter=2000, tolerance=1e-12)
        selected = tt.fit_transient_traces(
            r, designs, motion, valid, 50, options=tt.TransientOptions(**settings)
        )
        result = tt.fit_transient_traces(
            r, designs, motion, valid, 50,
            options=tt.TransientOptions(**settings, debias=True, event_threshold=.01),
        )
        self.assertTrue(result.metadata["converged"])
        self.assertTrue(result.metadata["debias"]["performed"])
        self.assertEqual(result.metadata["final_event_penalty"], 0)
        np.testing.assert_allclose(result.d_f[valid], expected[valid], atol=5e-4)
        np.testing.assert_allclose(result.events[1:], events[1:], atol=5e-4)
        self.assertTrue((result.events[selected.events <= .01] == 0).all())

    def test_nonconvergence_and_failed_debias(self):
        """Both optimization stages expose failures, never silently succeed."""
        r, designs, motion, valid, *_ = synthetic_problem()
        options = tt.TransientOptions(max_iter=1, debias=True)
        with self.assertWarns(RuntimeWarning):
            result = tt.fit_transient_traces(r, designs, motion, valid, 50, options=options)
        self.assertFalse(result.metadata["converged"])
        self.assertFalse(result.metadata["debias"]["performed"])
        success = OptimizeResult(x=np.ones(130), fun=0., success=True, nit=2, message="ok")
        failed = OptimizeResult(x=np.ones(130), fun=0., success=False, nit=1, message="limit")
        with patch.object(tt, "minimize", side_effect=[success, failed]):
            with self.assertWarns(RuntimeWarning):
                result = tt.fit_transient_traces(
                    r, designs, motion, valid, 50,
                    options=tt.TransientOptions(debias=True, event_threshold=0),
                )
        self.assertFalse(result.metadata["converged"])
        self.assertTrue(result.metadata["debias"]["performed"])

    def test_invalid_settings_and_inputs(self):
        """Reject malformed arrays, rates and optimization options."""
        for key, values in {
            "tau_s": [0, -1, np.nan, True],
            "beta_fraction": [-1, np.inf, True],
            "beta": [-1, np.nan, True],
            "max_iter": [0, 1.5, True],
            "tolerance": [0, np.inf],
            "debias": [1, "true"],
            "event_threshold": [-1, np.nan],
        }.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    tt.TransientOptions(**{key: value})
        r, designs, motion, valid, *_ = synthetic_problem()
        for rate in (None, 0, -1, np.nan, True):
            with self.assertRaises(ValueError):
                tt.fit_transient_traces(r, designs, motion, valid, rate)
        for kernel in ([0, 1], [], [-1, 2], [1, np.nan]):
            with self.assertRaises(ValueError):
                tt.TransientOperator(50, .15, kernel)
        for rr, xx, mm, vv in (
            (r.ravel(), designs, motion, valid),
            (r, designs, motion[:-1], valid),
            (r, {}, motion, valid),
            (r, {0: np.ones(3)}, motion, valid),
            (r, {0: np.full((3, 2), np.nan)}, motion, valid),
            (r, {0: np.zeros((3, 2))}, motion, valid),
            (r, designs, motion, np.zeros_like(valid)),
        ):
            with self.assertRaises(ValueError):
                tt.fit_transient_traces(rr, xx, mm, vv, 50)

    def test_rank_deficiency_and_nonfinite_complete_frames(self):
        """No inverse is required, and individual bad samples mask whole frames."""
        r, designs, motion, valid, *_ = synthetic_problem()
        for x in designs.values():
            x[:, 1] = x[:, 0]
        r[0, 1] = np.inf
        motion[5] = -1
        result = tt.fit_transient_traces(r, designs, motion, valid, 50)
        self.assertTrue(result.metadata["converged"])
        self.assertTrue(np.isnan(result.d_f[[1, 5, 9, 10, 11]]).all())
        self.assertTrue(np.isfinite(result.d_f[[0, 2, 3, 4]]).all())


if __name__ == "__main__":
    unittest.main()