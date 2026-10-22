"""Fixed-profile, output-grid sparse transient fitting.

Port of the standalone trace benchmark, not a localization/NMF solver.
Fits 0.5 * ||r - X K s||^2 + beta * sum(s[1:]), s >= 0, using
float64 sufficient statistics and bounded L-BFGS-B. The first coefficient
is an unpenalized initial state. Trials must be solved independently.
Reader averaging is not modeled exactly; events are not spike times.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve, lfilter

from ...models.params import _nonnegative, _positive


@dataclass(frozen=True)
class TransientOptions:
    """Settings for trace-only fitting with fixed spatial profiles.

    The workflow supplies tau_s from the shared BandSiloParams.decay_tau_s.
    ``beta`` overrides the trial-relative ``beta_fraction`` when supplied.
    ``debias`` optionally refits selected events without the L1 penalty.
    ``event_threshold`` is in absolute coefficient units, not scaled units.
    """

    tau_s: float = 0.15
    beta_fraction: float = 0.01
    beta: float | None = None
    max_iter: int = 500
    tolerance: float = 1e-7
    debias: bool = False
    event_threshold: float = 1e-6

    def __post_init__(self):
        """Reject invalid solver settings before fitting or reading data."""
        _positive("tau_s", self.tau_s)
        _positive("max_iter", self.max_iter, integer=True)
        _positive("tolerance", self.tolerance)
        _nonnegative("beta_fraction", self.beta_fraction)
        _nonnegative("event_threshold", self.event_threshold)
        if self.beta is not None:
            _nonnegative("beta", self.beta)
        if not isinstance(self.debias, bool):
            raise ValueError("debias must be a bool")


@dataclass
class TransientResult:
    """Trace, latent predictions/events, and convergence diagnostics."""

    d_f: np.ndarray
    metadata: dict
    events: np.ndarray
    initial_state: np.ndarray
    model_trace: np.ndarray


class TransientOperator:
    """Causal convolution and exact adjoint without a dense temporal matrix."""

    def __init__(self, analyze_hz, tau_s, kernel=None):
        """Use a unit-peak exponential or a nonnegative causal kernel."""
        _positive("analyze_hz", analyze_hz)
        _positive("tau_s", tau_s)
        self.gamma = float(np.exp(-1 / (analyze_hz * tau_s)))
        self.kernel = (
            None if kernel is None else np.asarray(kernel, dtype=float)
        )
        if self.kernel is not None and (
            self.kernel.ndim != 1
            or not self.kernel.size
            or not np.all(np.isfinite(self.kernel))
            or np.any(self.kernel < 0)
            or self.kernel[0] <= 0
        ):
            raise ValueError(
                "kernel must be finite, nonnegative, with kernel[0]>0"
            )

    def forward(self, events):
        """Convolve all sources in chronological order, including gaps."""
        if self.kernel is None:
            return lfilter([1.0], [1.0, -self.gamma], events, axis=0)
        return fftconvolve(events, self.kernel[:, None], mode="full", axes=0)[
            : len(events)
        ]

    def adjoint(self, traces):
        """Apply the exact transpose, including finite-trial boundaries."""
        return self.forward(traces[::-1])[::-1]


def _temporal_quadratic(normal, groups, trace):
    """Half squared error minus its data constant, and the trace gradient."""
    gradient = np.zeros_like(trace)
    value = 0.0
    for motion, (gram, rhs) in normal.items():
        ix = groups[motion]
        fit = trace[ix]
        gt = fit @ gram
        value += np.sum(fit * (0.5 * gt - rhs.T))
        gradient[ix] = gt - rhs.T
    return float(value), gradient


def _prepare_transient_system(residual, designs, motion_indices, valid):
    """Validate arrays and build motion-specific sufficient statistics."""
    residual = np.asarray(residual)
    motion_indices = np.asarray(motion_indices)
    valid = np.asarray(valid, dtype=bool).copy()
    if residual.ndim != 2:
        raise ValueError("residual must have shape (superpixels, frames)")
    n_sp, n_frames = residual.shape
    if valid.shape != (n_frames,) or motion_indices.shape != (n_frames,):
        raise ValueError(
            "motion_indices and valid must have one entry per frame"
        )
    if not designs:
        raise ValueError("Transient fitting requires projected profiles")
    arrays = {m: np.asarray(x) for m, x in designs.items()}
    first = next(iter(arrays.values()))
    if first.ndim != 2:
        raise ValueError("designs must have shape (superpixels, sources)")
    n_sources = first.shape[1]
    for x in arrays.values():
        if x.shape != (n_sp, n_sources) or not np.all(np.isfinite(x)):
            raise ValueError(
                "Projected profiles must be finite with matching shapes"
            )
    valid &= np.all(np.isfinite(residual), axis=0)
    valid &= np.isin(motion_indices, list(arrays))
    if not np.any(valid) or n_sources == 0:
        raise ValueError(
            "Transient fitting requires observed frames and sources"
        )
    groups, normal = {}, {}
    for motion, x in arrays.items():
        ix = np.flatnonzero(valid & (motion_indices == motion))
        if ix.size:
            xd = x.astype(np.float64)
            groups[motion] = ix
            normal[motion] = (
                xd.T @ xd,
                xd.T @ residual[:, ix].astype(np.float64),
            )
    return (n_frames, n_sources), valid, groups, normal


def fit_transient_traces(
    residual,
    designs,
    motion_indices,
    valid,
    analyze_hz,
    *,
    options=None,
    kernel=None,
):
    """Fit one trial using fixed motion-specific (superpixel, source) designs.

    Residual is (superpixels, frames). The complete nominal output grid must
    be retained, in chronological order: never compress out gaps or group
    disjoint motion frames before convolution. Nonfinite observations reject
    the entire frame, as in QR. ``valid`` also excludes unmatched motion/z.
    Public dF is NaN there, while model_trace contains latent gap predictions.

    Low-level nonconvergence returns diagnostics and warns. Production callers
    must check ``metadata['converged']`` before saving a result.
    """
    options = TransientOptions() if options is None else options
    if not isinstance(options, TransientOptions):
        raise TypeError("options must be TransientOptions or None")
    op = TransientOperator(analyze_hz, options.tau_s, kernel)
    shape, valid, groups, normal = _prepare_transient_system(
        residual, designs, motion_indices, valid
    )
    rhs_trace = np.zeros(shape)
    for motion, (_, rhs) in normal.items():
        rhs_trace[groups[motion]] = rhs.T
    event_rhs = op.adjoint(rhs_trace)
    beta_scale = float(max(0.0, np.max(event_rhs[1:], initial=0.0)))
    penalty = float(
        options.beta
        if options.beta is not None
        else options.beta_fraction * beta_scale
    )
    diagonals = np.concatenate([np.diag(g) for g, _ in normal.values()])
    positive = diagonals[diagonals > 0]
    if not positive.size:
        raise ValueError("No observable profile support")
    gram_scale = float(np.max(positive))
    amplitude_scale = max(float(np.max(np.abs(rhs_trace))) / gram_scale, 1e-12)
    objective_scale = gram_scale * amplitude_scale**2

    def objective(flat, event_penalty=penalty):
        """Evaluate the scaled objective and its analytic event gradient."""
        coefficients = flat.reshape(shape) * amplitude_scale
        trace = op.forward(coefficients)
        value, trace_gradient = _temporal_quadratic(normal, groups, trace)
        gradient = op.adjoint(trace_gradient).copy()
        value += event_penalty * coefficients[1:].sum()
        gradient[1:] += event_penalty
        return (
            value / objective_scale,
            (gradient * amplitude_scale / objective_scale).ravel(),
        )

    optimizer_options = {
        "maxiter": options.max_iter,
        "ftol": options.tolerance,
        "gtol": options.tolerance,
        "maxls": 40,
        "maxcor": 10,
    }
    result = minimize(
        objective,
        np.zeros(np.prod(shape)),
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, None)] * int(np.prod(shape)),
        options=optimizer_options,
    )
    selection = {
        "converged": bool(result.success),
        "iterations": int(result.nit),
        "message": str(result.message),
        "objective_without_data_constant": float(result.fun * objective_scale),
    }
    debias_info = {
        "requested": options.debias,
        "performed": False,
        "event_threshold": float(options.event_threshold),
    }
    if options.debias and result.success:
        active = (
            result.x.reshape(shape) * amplitude_scale > options.event_threshold
        )
        active[0] = True
        free = np.flatnonzero(active.ravel())
        debias_info["retained_events"] = int(active[1:].sum())

        def restricted_objective(reduced):
            """Refit selected events only; the initial state remains free."""
            full = np.zeros(np.prod(shape))
            full[free] = reduced
            value, gradient = objective(full, event_penalty=0.0)
            return value, gradient[free]

        refit = minimize(
            restricted_objective,
            result.x[free],
            method="L-BFGS-B",
            jac=True,
            bounds=[(0.0, None)] * len(free),
            options=optimizer_options,
        )
        full = np.zeros(np.prod(shape))
        full[free] = refit.x
        refit.x = full
        result = refit
        debias_info.update(
            performed=True,
            converged=bool(refit.success),
            iterations=int(refit.nit),
            message=str(refit.message),
        )
    elif options.debias:
        debias_info["skip_reason"] = "Support-selection fit did not converge"
    coefficients = result.x.reshape(shape) * amplitude_scale
    model_trace = op.forward(coefficients)
    phi = model_trace.copy()
    phi[~valid] = np.nan
    events = coefficients.copy()
    initial = events[0].copy()
    events[0] = 0
    converged = bool(
        result.success
        and np.isfinite(result.fun)
        and np.all(np.isfinite(model_trace))
    )
    if not converged:
        warnings.warn(
            f"Transient solver did not converge: {result.message}",
            RuntimeWarning,
            stacklevel=2,
        )
    return TransientResult(
        phi,
        {
            "method": "transient",
            "tau_s": options.tau_s,
            "beta": penalty,
            "beta_fraction": (
                options.beta_fraction if options.beta is None else None
            ),
            "beta_scale": beta_scale,
            "converged": converged,
            "iterations": int(result.nit),
            "message": str(result.message),
            "kernel": (
                "exponential"
                if kernel is None
                else np.asarray(kernel).tolist()
            ),
            "sampling_model": (
                "nominal output grid; acquisition averaging not modeled "
                "exactly"
            ),
            "selection": selection,
            "debias": debias_info,
            "final_event_penalty": (
                0.0 if debias_info["performed"] else penalty
            ),
            "objective_without_data_constant": float(
                result.fun * objective_scale
            ),
        },
        events,
        initial,
        model_trace,
    )
