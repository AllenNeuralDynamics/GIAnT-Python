Trace estimation
================

Production BandSILo extraction computes and saves least-squares
``sources/temporal/dF_ls``, convolution-constrained
``sources/temporal/dF_denoised``, and fitted ``sources/temporal/events``.
Each array has shape ``(sources, channels, total frames)``. Source activity
extraction uses **channel zero**; other channels and missing observations are
NaN. Both trace estimates are mandatory production outputs, not selectable
trace modes. Low-level loaded-array functions also support weighted averages
for diagnostics. See :doc:`file_formats` for the saved layout.

Trial-wise objective
--------------------

The trace-only solver holds spatial profiles fixed and minimizes

.. math::

   \min_{s \geq 0}\; \frac{1}{2}
   \left\|(d-b)-H_{\mathrm{motion}} A K s\right\|_2^2
   + \beta \sum_j \sum_{t=1}^{T-1} s_{j,t}.

Here ``d`` is the observed data, ``b`` is the background, ``A`` contains the
fixed source profiles, ``H_motion`` is the motion-dependent observation
operator, and ``K`` applies a causal unit-peak exponential. For each source,
:math:`s_{j,0}` is an unpenalized nonnegative initial state; the remaining entries
are nonnegative event amplitudes. The solver operates chronologically over
each complete trial, including motion changes and missing frames. No state or
event inference crosses trial boundaries. Localization, NMF, and profile
fitting are separate from this trace-only optimization.

Returned fluorescence changes preserve the QR complete-frame validity mask.
The standard least-squares trace and its background retain baseline
re-estimation for ``dF_ls`` and ``F0``; ``F0`` comes from QR, not the transient
model. ``dF_denoised`` is the fitted convolution trace itself, with no second
baseline subtraction. ``events`` excludes the initial state at each trial's
first frame and stores NaN for discarded frames.

Controls and convergence
------------------------

``BandSiloParams.decay_tau_s`` controls both the trace kernel and localization
filtering. Its default is 0.15 seconds, and it is saved as ``decayTau_s``.
The GUI's Decay Tau field and CLI ``--decay-tau-s`` flag change this shared
setting; there is no separate trace decay parameter.

.. list-table:: Trace-only controls
   :header-rows: 1
   :widths: 28 12 60

   * - Parameter
     - Default
     - Meaning
   * - ``trace_beta_fraction``
     - ``0.01``
     - Fraction of each trial's positive adjoint scale; zero disables L1.
   * - ``trace_max_iter``
     - ``500``
     - Maximum L-BFGS-B iterations per optimization stage.
   * - ``trace_tolerance``
     - ``1e-7``
     - Objective/gradient tolerance.
   * - ``trace_debias``
     - ``False``
     - Refit selected events without L1 shrinkage.
   * - ``trace_event_threshold``
     - ``1e-6``
     - Absolute event-amplitude threshold for debias support.

These controls are recorded in summary parameters and are available as
hyphenated CLI flags (for example, ``--trace-beta-fraction``).
A nonconverged production fit raises an error rather than saving an unfinished
estimate. Increase the iteration limit or inspect a low-level fit when this
occurs. Per-trial optimizer diagnostics and latent in-gap predictions are not
saved in the summary.

Motion selection and reproducibility
------------------------------------

``BandSiloParams(z_tol=1.5)`` sets the low-resolution motion-bin selection
tolerance in microns. Bins must lie within this absolute distance of the
median frame z and satisfy the minimum frame count. The value must be positive
and finite. The CLI and staging example expose ``--z-tol``; the parameter GUI
labels it **Z Tolerance (µm)**.

``BandSiloParams(random_seed=0)`` seeds the random motion-bin order used by
source localization (NMF and Adam). It must be a nonnegative integer. Each
path and outer iteration uses its own generator derived from this seed,
independently of the global torch RNG. The seed is saved in summary parameters
and exposed as ``--random-seed``. It fixes random draws, not floating-point
results across hardware, thread counts, or package versions.

Loaded-array API
----------------

For loaded arrays, pass ``transient_options=TransientOptions(...)`` and
``analyze_hz=...`` to
``giant_python.extraction.band.traces.compute_high_res_traces``.
``TransientOptions`` and ``fit_transient_traces`` are defined in
``giant_python.extraction.band.transient``. The latter accepts an absolute
``beta`` or an explicit causal kernel and exposes events, the initial state,
in-gap model predictions, actual beta, and convergence diagnostics.
Retain the complete chronological output grid, including gaps, when supplying
arrays. Low-level nonconvergence warns and returns diagnostics; callers must
check ``metadata['converged']`` before saving a result.
``TrialTraceResult`` retains its original eight fields in order and appends
``d_f_denoised`` and ``events``.

Scientific interpretation
-------------------------

This is an output-grid estimator: overlapping acquisition-line averaging is
not modeled exactly. In-gap predictions are not observations, and glutamate
events are not automatically neuronal spikes. A smoother trace alone does not
establish improved SNR. Use a fresh output location when comparing estimates
or settings; summary updates are not an atomic replacement or a resume
mechanism.