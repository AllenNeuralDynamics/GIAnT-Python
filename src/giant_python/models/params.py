"""Scientific configuration and independent execution and annotation policy.

``BandSiloParams`` contains only band science; ``SiloParams`` is the science
of raster SILo (SLAP2 multi-ROI) and ``SiloExecution`` how it runs.
``ExecutionOptions`` controls scheduling/debug limits; ``AnnotationOptions``
controls ROI use and GUI policy. Resolution copies inputs without retuning
the existing algorithm.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, fields
from numbers import Integral, Real
from typing import Optional, Union

import numpy as np


def _positive(name: str, value: object, *, integer: bool = False) -> None:
    """Reject nonfinite, nonpositive, boolean, or nonnumeric values."""
    kind = Integral if integer else Real
    if (
        isinstance(value, bool)
        or not isinstance(value, kind)
        or not math.isfinite(value)
        or value <= 0
    ):
        label = "positive integer" if integer else "positive finite number"
        raise ValueError(f"{name} must be a {label}")


def _nonnegative(name: str, value: object) -> None:
    """Reject nonfinite, negative, boolean, or nonnumeric values."""
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite nonnegative number")


@dataclass
class BandSiloParams:
    """Scientific settings for the implemented SLAP2 band algorithm.

    Attributes
    ----------
    analyze_hz : float
        Temporal analysis rate in Hz.
    decay_tau_s : float
        Decay time constant in seconds, shared by localization filtering and
        convolution-constrained trace extraction. Saved as decayTau_s.
    baseline_window_s, denoise_window_s : float
        Baseline and temporal denoising windows in seconds.
    vif : float
        Noise-model variance inflation factor.
    d_xy : int
        Spatial downsampling / superpixel spacing in pixels.
    sparse_fac : float
        NMF sparsity factor, not its logarithm.
    peakth : float
        Peak threshold. Seven preserves the current implementation; this
        configuration split does not retune the algorithm.
    peak_buffer : int
        Peak exclusion buffer diameter in pixels.
    psf_dilation : int
        Positive dilation size selecting the bundled PSF template. Template
        availability is checked by the backend when loading the PSF.
    z_tol : float
        Positive z tolerance in microns: maximum absolute motion-bin z
        deviation from the median frame z for low-resolution frame selection.
    trace_beta_fraction : float
        Nonnegative event penalty relative to each trial's adjoint scale.
    trace_max_iter : int
        Maximum L-BFGS-B iterations per temporal optimization stage.
    trace_tolerance : float
        Positive optimizer objective/gradient tolerance.
    trace_debias : bool
        Refit selected events without L1 shrinkage (default False).
    trace_event_threshold : float
        Absolute event amplitude threshold for optional debias support.
    random_seed : int
        Nonnegative seed for the random motion-bin order used by source
        localization (NMF and Adam). Each DMD path and outer iteration
        draws from its own generator derived from this seed.

    Notes
    -----
    Scheduling, trial limits, logging and annotation policy are deliberately
    separate from science. Numerical defaults preserve the band algorithm.
    Sparse-superpixel background interpolation always uses cubic splines.
    Channel count is inferred from acquisition metadata; source activity
    extraction always uses channel zero. Production extraction always saves
    least-squares dF_ls alongside convolution dF_denoised and events.
    """

    analyze_hz: float = 100.0
    decay_tau_s: float = 0.15
    baseline_window_s: float = 4.0
    denoise_window_s: float = 1.0
    vif: float = 1.38
    d_xy: int = 5
    sparse_fac: float = math.exp(-3.0)
    peakth: float = 7.0
    peak_buffer: int = 3
    psf_dilation: int = 17
    z_tol: float = 1.5
    trace_beta_fraction: float = 0.01
    trace_max_iter: int = 500
    trace_tolerance: float = 1e-7
    trace_debias: bool = False
    trace_event_threshold: float = 1e-6
    random_seed: int = 0

    def __post_init__(self) -> None:
        """Validate values supported by the existing numerical path."""
        for name in (
            "analyze_hz",
            "decay_tau_s",
            "baseline_window_s",
            "denoise_window_s",
            "vif",
            "sparse_fac",
            "peakth",
            "z_tol",
            "trace_tolerance",
        ):
            _positive(name, getattr(self, name))
        for name in ("d_xy", "peak_buffer", "psf_dilation", "trace_max_iter"):
            _positive(name, getattr(self, name), integer=True)
        if not isinstance(self.trace_debias, bool):
            raise ValueError("trace_debias must be a bool")
        for name in ("trace_beta_fraction", "trace_event_threshold"):
            _nonnegative(name, getattr(self, name))
        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, Integral)
            or self.random_seed < 0
        ):
            raise ValueError("random_seed must be a nonnegative integer")


@dataclass
class MultiRoiRegistrationParams:
    """Scientific settings for SLAP2 raster registration.

    Field meanings and defaults follow GIAnT-MATLAB
    ``setParams('MultiRoiRegistration')``; they are saved under their MATLAB
    names in the trial table's ``motion_correction/align_params``.

    Attributes
    ----------
    align_hz : float
        Rate of the registered frames (alignHz).
    maxshift : int
        Largest motion in pixels; the output grid is padded by it.
    clip_shift : int
        Smallest search radius per frame around the previous frame's motion
        (clipShift).
    alpha : float
        Unused template decay (kept for parity with MATLAB's parameters).
    overwrite_existing : bool
        Re-register trials whose outputs exist. Otherwise existing outputs
        are kept if they were made with the same settings (all fields but
        this one), and registered again if not.
    save_tif : bool
        Save the registered movie as TIFF, as GIAnT-MATLAB does by default.
        Otherwise (the default) it is a deflated HDF5 ``/data``, about 7x
        smaller, which GIAnT-MATLAB SILo also reads.
    ref_stack_template, is_revolt, include_integration_rois : bool
        MATLAB options not yet ported; must be False.
    compensate_online : bool
        Undo the online motion correction line by line, so that frames
        spanning a change of the DMD offset are not blurred
        (compensateOnline).
    init_frames : int
        Frames matched pairwise for the initial template (MATLAB: 40).
    init_search : int
        Largest shift of the pairwise matching (MATLAB: 3).
    min_samples : int
        Contributions a template pixel needs (MATLAB minSamps: 15).
    schedule : str
        ``'sequential'`` (MATLAB's frame loop, the default) or ``'grow'``
        (batched growth from seed frames, see
        :mod:`giant_python.registration.growth`).
    frames_per_front : int
        ``grow``: frames taken from each island edge per sub-round.
    min_fronts : int
        ``grow``: add seeds when fewer fronts remain.
    seed_accept_quantile : float
        ``grow``: a seed's peak correlation and margin must reach this
        quantile of the aligned frames'.
    junction_jump_px : float or None
        ``grow``: motion jump at a junction that triggers realignment
        (``None``: max(1, 99.9th percentile of per-frame steps)).
    polish : bool
        ``grow``: rematch every frame to the final template without itself.
    polish_radius : int
        ``grow``: search radius of the polish pass.
    var_factor_deflate : int
        Deflate level (0-9) of ``varFacDS`` in the alignment file; 0 stores
        it uncompressed, as MATLAB does. MATLAB's ``h5read`` decompresses
        transparently.
    movie_deflate : int
        Deflate level (0-9) of HDF5 movies (``save_tif=False``, the
        default); MATLAB writes 4.
    """

    align_hz: float = 80.0
    maxshift: int = 40
    clip_shift: int = 5
    alpha: float = 0.005
    overwrite_existing: bool = False
    save_tif: bool = False
    ref_stack_template: bool = False
    is_revolt: bool = False
    include_integration_rois: bool = False
    compensate_online: bool = False
    init_frames: int = 40
    init_search: int = 3
    min_samples: int = 15
    schedule: str = "sequential"
    frames_per_front: int = 1
    min_fronts: int = 16
    seed_accept_quantile: float = 0.1
    junction_jump_px: Optional[float] = None
    polish: bool = False
    polish_radius: int = 2
    var_factor_deflate: int = 1
    movie_deflate: int = 1

    def __post_init__(self) -> None:
        """Validate values and reject options that are not ported."""
        _positive("align_hz", self.align_hz)
        for name in (
            "maxshift",
            "clip_shift",
            "init_frames",
            "init_search",
            "min_samples",
            "frames_per_front",
            "min_fronts",
            "polish_radius",
        ):
            _positive(name, getattr(self, name), integer=True)
        _nonnegative("alpha", self.alpha)
        self._validate_growth_and_output()
        for name in (
            "overwrite_existing",
            "save_tif",
            "polish",
            "ref_stack_template",
            "is_revolt",
            "include_integration_rois",
            "compensate_online",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")
        for name in (
            "ref_stack_template",
            "is_revolt",
            "include_integration_rois",
        ):
            if getattr(self, name):
                raise NotImplementedError(
                    f"{name}=True is not supported yet in GIAnT-Python"
                )

    def _validate_growth_and_output(self) -> None:
        """Validate the growth-schedule and output-compression settings."""
        if self.schedule not in ("sequential", "grow"):
            raise ValueError("schedule must be 'sequential' or 'grow'")
        if not 0 <= self.seed_accept_quantile < 1:
            raise ValueError("seed_accept_quantile must be in [0, 1)")
        if self.junction_jump_px is not None:
            _positive("junction_jump_px", self.junction_jump_px)
        for name in ("var_factor_deflate", "movie_deflate"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or not 0 <= value <= 9
            ):
                raise ValueError(f"{name} must be an integer from 0 to 9")

    def to_matlab(self) -> dict:
        """The parameters under GIAnT-MATLAB's ``align_params`` names."""
        return {
            "alignHz": float(self.align_hz),
            "maxshift": float(self.maxshift),
            "clipShift": float(self.clip_shift),
            "alpha": float(self.alpha),
            "overwriteExisting": self.overwrite_existing,
            "saveTif": self.save_tif,
            "refStackTemplate": self.ref_stack_template,
            "isReVolt": self.is_revolt,
            "includeIntegrationROIs": self.include_integration_rois,
            "compensateOnline": self.compensate_online,
            "initFrames": float(self.init_frames),
            "initSearch": float(self.init_search),
            "minSamps": float(self.min_samples),
            "schedule": self.schedule,
            "framesPerFront": float(self.frames_per_front),
            "minFronts": float(self.min_fronts),
            "seedAcceptQuantile": float(self.seed_accept_quantile),
            "junctionJumpPx": (
                float("nan")
                if self.junction_jump_px is None
                else float(self.junction_jump_px)
            ),
            "polish": self.polish,
            "polishRadius": float(self.polish_radius),
            "varFactorDeflate": float(self.var_factor_deflate),
            "movieDeflate": float(self.movie_deflate),
        }


@dataclass
class ExecutionOptions:
    """Run policy, separate from scientific configuration.

    Attributes
    ----------
    max_workers : int
        Positive number of worker processes; preserves the default of six.
    verbose : bool
        Enable stage messages and progress bars.
    max_trials : int or None
        Positive debug limit per DMD. ``None`` processes all trials.
    device : str
        Torch device for registration: ``'auto'`` (CUDA when available),
        ``'cpu'``, ``'cuda'`` or ``'cuda:N'``. Band extraction ignores it.
    """

    max_workers: int = 6
    verbose: bool = False
    max_trials: Optional[int] = None
    device: str = "auto"

    def __post_init__(self) -> None:
        """Validate scheduling and debug limits."""
        _positive("max_workers", self.max_workers, integer=True)
        if self.max_trials is not None:
            _positive("max_trials", self.max_trials, integer=True)
        if not isinstance(self.verbose, bool):
            raise ValueError("verbose must be a bool")
        if not isinstance(self.device, str) or not (
            self.device in ("auto", "cpu") or self.device.startswith("cuda")
        ):
            raise ValueError("device must be 'auto', 'cpu' or 'cuda[:N]'")


@dataclass
class AnnotationOptions:
    """ROI inclusion and GUI policy, separate from scientific settings.

    Attributes
    ----------
    enabled : bool
        Include user ROIs during extraction.
    interactive : bool or None
        Allow or forbid drawing; ``None`` retains backend auto-detection.
    """

    enabled: bool = False
    interactive: Optional[bool] = None

    def __post_init__(self) -> None:
        """Validate the explicit and automatic annotation policies."""
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool")
        if self.interactive is not None and not isinstance(
            self.interactive, bool
        ):
            raise ValueError("interactive must be a bool or None")


BandParamsInput = Optional[Union[BandSiloParams, dict]]
ExecutionInput = Optional[Union[ExecutionOptions, dict]]
AnnotationInput = Optional[Union[AnnotationOptions, dict]]
RegistrationParamsInput = Optional[Union[MultiRoiRegistrationParams, dict]]


def _option_values(
    option: object,
    model: type[
        Union[
            BandSiloParams,
            ExecutionOptions,
            AnnotationOptions,
            MultiRoiRegistrationParams,
            SiloParams,
            SiloExecution,
        ]
    ],
) -> dict:
    """Copy a typed option or validate a strictly scoped dictionary."""
    if option is None:
        return {}
    if isinstance(option, model):
        return asdict(option)
    if isinstance(option, dict):
        allowed = {item.name for item in fields(model)}
        unknown = option.keys() - allowed
        if unknown:
            raise TypeError(f"Unknown {model.__name__} fields: {unknown}")
        return dict(option)
    raise TypeError(f"Expected {model.__name__}, dict, or None")


def resolve_band_options(
    params: BandParamsInput = None,
    execution: ExecutionInput = None,
    annotations: AnnotationInput = None,
) -> tuple[BandSiloParams, ExecutionOptions, AnnotationOptions]:
    """Resolve independent band science, execution and annotation options.

    Parameters
    ----------
    params : BandSiloParams, dict or None
        Scientific settings only. Routing, execution, annotation and unknown
        fields raise ``TypeError`` rather than being dispatched elsewhere.
    execution : ExecutionOptions, dict or None
        Run options or a dictionary containing only execution fields.
    annotations : AnnotationOptions, dict or None
        Annotation policy or a dictionary containing only annotation fields.

    Returns
    -------
    tuple of BandSiloParams, ExecutionOptions, AnnotationOptions
        Fresh validated objects. Caller inputs are never mutated, including
        when channel count or GUI interactivity is later resolved.
    """
    return (
        BandSiloParams(**_option_values(params, BandSiloParams)),
        ExecutionOptions(**_option_values(execution, ExecutionOptions)),
        AnnotationOptions(**_option_values(annotations, AnnotationOptions)),
    )


def resolve_registration_options(
    params: RegistrationParamsInput = None,
    execution: ExecutionInput = None,
) -> tuple[MultiRoiRegistrationParams, ExecutionOptions]:
    """Resolve registration science and execution options.

    Parameters
    ----------
    params : MultiRoiRegistrationParams, dict or None
        Scientific settings only; unknown fields raise ``TypeError``.
    execution : ExecutionOptions, dict or None
        Run options.

    Returns
    -------
    tuple of MultiRoiRegistrationParams, ExecutionOptions
        Fresh validated objects.
    """
    return (
        MultiRoiRegistrationParams(
            **_option_values(params, MultiRoiRegistrationParams)
        ),
        ExecutionOptions(**_option_values(execution, ExecutionOptions)),
    )


_SILO_STAGES = ("localize", "plan", "prepare", "solve", "collect")


def _odd(name: str, value: int) -> None:
    """Reject an even integer."""
    if value % 2 == 0:
        raise ValueError(f"{name} must be odd, not {value}")


def _choice(name: str, value: object, allowed: tuple) -> None:
    """Reject a value that is not one of ``allowed``."""
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, not {value!r}")


@dataclass(frozen=True)
class SiloParams:
    """Scientific settings of SILo for SLAP2 multi-ROI raster recordings.

    One parameter set for every stage of source extraction (localization,
    plan, trial loading, subproblem solving, collection). The defaults are
    GIAnT-MATLAB main's ``setParams('SILo')`` for SLAP2 (``isSLAP2``), with
    its modality defaults resolved (``peakth`` 3, ``minKineticScore`` 0.2,
    ``photonScale`` 25); the footprint fit's variance term is the solver's
    default (``footprint_variance='model'`` gives MATLAB's loss), and the
    footprints' support is 5 x 5 (``valid_kernel=3`` is MATLAB's). MATLAB's
    names are given in brackets; :meth:`to_matlab` returns the ``params``
    struct that SILo saves. The stages' own parameter types are derived
    with :meth:`plan_params`, :meth:`load_params`, :meth:`solver_params` and
    :meth:`localization_params`.

    ``baseline_source`` chooses the baseline of the whole run, and with it
    the defaults of the settings that go with it (each can still be set):

    ============================  ===============  =====================
    setting                       ``'median'``     ``'network'``
    ============================  ===============  =====================
    ``localization_baseline``     ``'median'``     ``'network'``
    ``localization_surround``     True             False
    ``localization_null``         ``'none'``       ``'bootstrap'``
    ``peak_th``                   3                3
    ``min_source_event_rate``     0                2
    ``baseline_mode``             ``'fit'`` (R0)   ``'gain'`` (N1)
    ============================  ===============  =====================

    ``'median'`` is MATLAB's SILo. ``'network'`` uses the F0 network's
    baselines (``slap2-f0 predict-registered``): localization with the
    settings recommended in slap2-F0-network#6 (no surround, the bootstrap
    null, the variance floor kept), seeding at ``peak_th`` 3 with the
    evidence test ``min_source_event_rate`` 2 events per minute
    (GIAnT-Python#51: more of the dim sources; until 2026-11-28 ``peak_th``
    was 8 with no evidence test, ``peak_th=8, min_source_event_rate=0``),
    and
    extraction with the network's baseline times one gain per pixel (N1),
    the variant that the baseline test of the SILo port (WP6) found within
    tolerance of refitting the baseline from the network's.

    Attributes
    ----------
    baseline_source : {'median', 'network'}
        Baseline of the run: MATLAB's (moving median for localization,
        the solver's own fit for extraction) or the F0 network's, whose
        files are given at run time. Sets the defaults above. Not a MATLAB
        option.
    activity_channel : int
        One-based channel of the activity signal (``activityChannel``).
    analyze_hz : float
        Frame rate of extraction (``analyzeHz``).
    sigma_px : float
        Gaussian sigma of the PSF, pixels (``sigma_px``).
    d_xy : float
        How large sources can be, pixels (``dXY``); the selected disk around
        a source has radius ``ceil(2 d_xy)`` (``selRadius``).
    valid_kernel : int
        Side of the square around each source where the footprint fit may
        put intensity (odd; default 5; MATLAB's ``validKernel``, which
        MATLAB hard-codes as 3 x 3). One setting for the whole run: it
        sets the
        plan's grouping zones (radius ``ceil(1.5 sigma_px +
        (valid_kernel - 1) / 2)``, :class:`PlanParams`) and the solver's
        footprints (``SolverParams.valid_kernel``) together. The footprint
        is ``hfilter`` blurring that support, so its width lies between
        0.9 ``sigma_px`` and ``sqrt((0.9 sigma_px)**2 + r**2)``,
        r = (valid_kernel - 1) / 2: 1.20 to 1.56 px with 3 at ``sigma_px``
        1.33, up to 2.33 px with 5. 3 is the setting to compare with
        MATLAB.
    tau_s, tau2_s : float
        Decay time constants of the activity channel and of channel 2
        (``tau_s``, ``tau2_s``).
    baseline_window_s : float
        Timescale of the baseline (``baselineWindow_Glu_s``).
    baseline_window_ca_s : float
        ``baselineWindow_Ca_s``; recorded only, as in MATLAB's SILo.
    denoise_window_s : float
        Timescale over which signals may be smoothed (``denoiseWindow_s``).
    vif : float
        Variance inflation factor of localization (``VIF``).
    motion_thresh : float
        Registered frames whose motion error exceeds this are discarded
        (``motionThresh``).
    discard_initial_s : float
        Seconds discarded at the start of each trial (``discardInitial_s``).
    nan_thresh : float
        Largest share of missing samples of an included pixel
        (``nanThresh``).
    peak_th : float or None
        Seed threshold, robust SDs of the activity image (``peakth``; 3 for
        SLAP2). Used by the plan's seeding and, unless
        ``localization_peak_th`` is set, by the per-trial peaks. None takes
        the default of ``baseline_source`` (3 with both; it was 8 with the
        network until 2026-11-28).
    min_peak_distance : int
        Minimum Chebyshev distance of seeds (``minPeakDistance``).
    localization_baseline : {'median', 'network'} or None
        Baseline of localization: MATLAB's moving median, or the F0
        network's ``*_BASELINE-80Hz.h5`` files (``slap2-f0
        predict-registered``), whose folder is given at run time. Not a
        MATLAB option. None takes ``baseline_source``.
    localization_surround : bool or None
        Subtract the surround in the localization score (MATLAB's). None
        takes the default of ``baseline_source`` (True, or False with the
        network).
    localization_null : {'none', 'bootstrap'} or None
        Null calibration of the localization score. None takes the default
        of ``baseline_source`` ('none', or 'bootstrap' with the network).
    localization_null_draws, localization_null_seed : int
        Noise movies per trial of the bootstrap null, and the first seed.
    localization_null_scale : {'tail', 'sd'}
        Calibration of the bootstrap null.
    localization_variance_floor : bool
        Add the variance floor ``Vb`` to the score's variance (MATLAB's).
    localization_score : {'squared_maxima', 'soft_threshold'}
        Localization score (MATLAB's: ``'squared_maxima'``).
    localization_soft_th : float
        Threshold of the soft-threshold score, robust SDs.
    localization_peak_th : float or None
        Threshold of the per-trial peaks; None uses ``peak_th``.
    localization_keep_null : bool or None
        Keep each trial's calibrated null images (the bootstrap null's
        activity images) in its localization file, for the null seed
        thresholds. Each trial-path's null movies are then drawn with its
        own seed (``localization_null_seed`` plus a trial-path offset,
        :func:`~giant_python.pipeline.silo.trial_null_seed`), so that the
        trials' nulls are independent; the calibration differs from a run
        without kept nulls only by that random draw. None: kept exactly when
        ``seed_acceptance`` is not 'image'. Not a MATLAB option.
    trial_peaks : bool
        Fit each trial's peaks (Gaussians at the per-trial activity image's
        maxima). They feed only the per-trial summary's
        ``per_trial_act_im_peaks`` and ``per_trial_num_peaks``; False
        leaves those out of the file.
    inter_trial_alignment : {'rigid', 'affine', 'patchwise'}
        Across-trial alignment (``interTrialAlignment``): a shift per
        trial, refined for the valid trials to an affine map or to a
        patchwise displacement field (see
        :mod:`~giant_python.extraction.raster.alignment`). With
        ``'patchwise'`` (the default; MATLAB's default is ``'rigid'``) the
        localize stage also stores each trial's half images.
    align_patch_px : int
        Imaged pixels per patch of the patchwise field (``alignPatchPx``).
    peak_method : {'getactimpeaks'}
        Seeding (``peakMethod``); WMF is not ported yet.
    seed_fit : {'local', 'matlab'}, optional
        Gaussian fits of the seeding and of the per-trial peaks: local fits
        run to convergence (``'local'``, the default: seconds per path
        instead of minutes, sources nearer MATLAB's own), or MATLAB's
        capped joint fits (``'matlab'``). See
        :func:`~giant_python.extraction.raster.peaks.get_act_im_peaks`.
        None takes the plan stage's default
        (:class:`~giant_python.extraction.raster.plan.PlanParams`).
    seed_merge_px : float, optional
        Reject a residual peak that ends closer than this many pixels to
        another peak (default 1.5, which keeps a source between pixels
        from being seeded two or three times; 0 is MATLAB's rule only).
        None takes the plan stage's default.
    seed_acceptance : {'image', 'null', 'null_sd'}
        How the plan's seed threshold is set: ``peak_th`` robust SDs of the
        combined activity image ('image', MATLAB's); from the null of the
        combined image so that ``seed_false_per_field`` false seeds are
        expected per path ('null'; ``peak_th`` then sets only the per-trial
        peaks); or ``peak_th`` robust SDs of that null ('null_sd'). Both
        null modes need ``localization_null='bootstrap'``. Not a MATLAB
        option.
    seed_false_per_field : float
        Expected false seeds per path with ``seed_acceptance='null'``.
    seed_amplitude_rule : {'matlab', 'pixel'}
        The plan's final test of each fitted seed (see
        :func:`~giant_python.extraction.raster.peaks.get_act_im_peaks`):
        MATLAB's (its 2 x 2 pixel mean, which drops sharp peaks) or its
        value on its own pixel ('pixel', not in MATLAB).
    wmf_peak_th, wmf_dip, wmf_deblend_th : float
        ``wmfPeakth``, ``wmfDip``, ``wmfDeblendth`` (WMF only; recorded).
    photon_scale : float
        Raw counts per unit of the data (``photonScale``; 25 for SLAP2).
    lam : float
        L1 penalty of the activity fit (``lambda``).
    phi : float
        Share of ``lam`` kept in the debias refit (``phi``).
    nmf_iter : int
        Outer iterations of the solver (``nmfIter``).
    min_baseline : float
        Lower bound of the baseline, normalized units (``minBaseline``).
    baseline_mode : {'fit', 'fixed', 'gain', 'gain_shared', 'fit_from'} or None
        How the solver finds the activity channel's baseline
        (``SolverParams.baseline_mode``): 'fit' is MATLAB's own fit (R0);
        the others start from the network's baseline (N0-N3 of the baseline
        test). None takes the default of ``baseline_source`` ('fit', or
        'gain' with the network). Not a MATLAB option.
    debias_variance : {'fixed', 'model'}
        Variance term of the debias refit (``debiasVariance``).
    footprint_variance : {'fixed', 'model'} or None
        Variance term of the footprint fit: held at the fit's start
        ('fixed') or following the model as in MATLAB ('model'). Not a
        MATLAB option. None (the default) takes the solver's default,
        ``SolverParams.footprint_variance`` ('fixed' since
        GIAnT-Python#23); the resolved value is what the object holds.
    baseline_knot_factor, baseline_penalty : float
        The solver's baseline spline (MATLAB's fixed values).
    min_kinetic_score : float
        Sources whose transients score below this in the kinetics test are
        removed after extraction (``minKineticScore``; 0 keeps all).
    min_source_snr : float
        The evidence test (not in MATLAB; 0, the default, turns it off):
        sources whose SNR (the solver's, averaged over the extracted
        trials) is below this are removed after extraction with those the
        kinetics test rejects. Meant for permissive seeding (a lower
        ``peak_th`` or a larger ``seed_false_per_field``).
    min_source_event_rate : float or None
        The evidence test's other criterion (0: off; not in MATLAB):
        sources whose trace has fewer kinetics-test events per minute than
        this (events over the minutes of the trace's frames that were not
        discarded) are removed. A rate, so that it means the same in short
        and long recordings. None takes the default of ``baseline_source``
        (0, or 2 with the network, where it goes with the seed threshold).
    evidence_margin : float or None
        The evidence test judges only the sources seeded below ``peak_th``
        + ``evidence_margin`` robust SDs (their seed height,
        :func:`~giant_python.extraction.raster.seeding.seed_heights`):
        stronger seeds are kept whatever their evidence. Default 2; None
        judges every source, as do the null seed thresholds
        (``seed_acceptance`` 'null' or 'null_sd', which do not count in
        the combined image's robust SDs above ``peak_th``). Matters only
        when the evidence test is on. Not a MATLAB option.
    min_frames : int
        Trials with fewer frames are loaded but not source-extracted
        (MATLAB's ``minFrames``).
    user_rois : bool
        Read the session's ``annotations/annotations.h5`` (soma masks and
        user ROIs; ``drawUserRois``). ROIs are never drawn here: without a
        file the session has no user ROIs, as MATLAB's SILo without the
        GUI.
    """

    activity_channel: int = 1
    analyze_hz: float = 200.0
    sigma_px: float = 1.33
    d_xy: float = 3.0
    valid_kernel: int = 5
    tau_s: float = 0.024
    tau2_s: float = 0.15
    baseline_window_s: float = 4.0
    baseline_window_ca_s: float = 4.0
    denoise_window_s: float = 0.2
    vif: float = 1.67
    motion_thresh: float = 2.0
    discard_initial_s: float = 0.0
    nan_thresh: float = 0.33
    peak_th: Optional[float] = None
    min_peak_distance: int = 1
    baseline_source: str = "median"
    localization_baseline: Optional[str] = None
    localization_surround: Optional[bool] = None
    localization_null: Optional[str] = None
    localization_null_draws: int = 1
    localization_null_seed: int = 0
    localization_null_scale: str = "tail"
    localization_variance_floor: bool = True
    localization_score: str = "squared_maxima"
    localization_soft_th: float = 2.0
    localization_peak_th: Optional[float] = None
    localization_keep_null: Optional[bool] = None
    trial_peaks: bool = True
    inter_trial_alignment: str = "patchwise"
    align_patch_px: int = 1500
    peak_method: str = "getactimpeaks"
    seed_fit: Optional[str] = None
    seed_merge_px: Optional[float] = None
    seed_acceptance: str = "image"
    seed_false_per_field: float = 1.0
    seed_amplitude_rule: str = "matlab"
    wmf_peak_th: float = 3.0
    wmf_dip: float = 3.0
    wmf_deblend_th: float = 8.0
    photon_scale: float = 25.0
    lam: float = 0.9
    phi: float = 0.0
    nmf_iter: int = 2
    min_baseline: float = 0.01
    baseline_mode: Optional[str] = None
    debias_variance: str = "fixed"
    footprint_variance: Optional[str] = None
    baseline_knot_factor: float = 0.5
    baseline_penalty: float = 30.0
    min_kinetic_score: float = 0.2
    min_source_snr: float = 0.0
    min_source_event_rate: Optional[float] = None
    evidence_margin: Optional[float] = 2.0
    min_frames: int = 100
    user_rois: bool = True

    def __post_init__(self) -> None:
        """Validate values; reject modes that are not ported."""
        self._baseline_defaults()
        for name in (
            "analyze_hz",
            "sigma_px",
            "d_xy",
            "tau_s",
            "tau2_s",
            "baseline_window_s",
            "baseline_window_ca_s",
            "denoise_window_s",
            "vif",
            "motion_thresh",
            "nan_thresh",
            "peak_th",
            "photon_scale",
            "min_baseline",
            "baseline_knot_factor",
        ):
            _positive(name, getattr(self, name))
        for name in (
            "activity_channel",
            "min_peak_distance",
            "nmf_iter",
            "align_patch_px",
            "localization_null_draws",
            "min_frames",
            "valid_kernel",
        ):
            _positive(name, getattr(self, name), integer=True)
        # the footprint's support is centred on the source's pixel
        _odd("valid_kernel", self.valid_kernel)
        for name in (
            "discard_initial_s",
            "lam",
            "phi",
            "baseline_penalty",
            "min_kinetic_score",
            "min_source_snr",
            "min_source_event_rate",
            "wmf_peak_th",
            "wmf_dip",
            "wmf_deblend_th",
            "localization_soft_th",
        ):
            _nonnegative(name, getattr(self, name))
        if self.localization_peak_th is not None:
            _positive("localization_peak_th", self.localization_peak_th)
        if self.evidence_margin is not None:
            _nonnegative("evidence_margin", self.evidence_margin)
        for name in (
            "localization_surround",
            "localization_variance_floor",
            "trial_peaks",
            "user_rois",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")
        _choice("debias_variance", self.debias_variance, ("fixed", "model"))
        self._seed_defaults()
        self._solver_defaults()
        _choice(
            "inter_trial_alignment",
            self.inter_trial_alignment,
            ("rigid", "affine", "patchwise"),
        )
        if self.peak_method != "getactimpeaks":
            raise NotImplementedError(
                f"peakMethod {self.peak_method!r} is not ported yet; only "
                "'getactimpeaks'"
            )
        if not self.nan_thresh <= 1:
            # a share of missing samples: above 1 pixels without any sample
            # would pass
            raise ValueError("nan_thresh must not be above 1")
        self._check_localization()
        # the stages' own types check the rest
        self.localization_params(80.0)
        self.solver_params()

    def _baseline_defaults(self) -> None:
        """Fill the settings left at None from ``baseline_source``."""
        _choice("baseline_source", self.baseline_source, ("median", "network"))
        for name, value in BASELINE_SOURCE_DEFAULTS[
            self.baseline_source
        ].items():
            if getattr(self, name) is None:
                object.__setattr__(self, name, value)

    def _seed_defaults(self) -> None:
        """Take the seed fit's defaults from the plan stage (one default)."""
        from ..extraction.raster.plan import PlanParams

        for name in ("seed_fit", "seed_merge_px"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, getattr(PlanParams, name))
        _choice("seed_fit", self.seed_fit, ("matlab", "local"))
        _nonnegative("seed_merge_px", self.seed_merge_px)
        _choice(
            "seed_acceptance",
            self.seed_acceptance,
            ("image", "null", "null_sd"),
        )
        _positive("seed_false_per_field", self.seed_false_per_field)
        _choice(
            "seed_amplitude_rule",
            self.seed_amplitude_rule,
            ("matlab", "pixel"),
        )
        if self.localization_keep_null is None:
            object.__setattr__(
                self,
                "localization_keep_null",
                self.seed_acceptance != "image",
            )
        if not isinstance(self.localization_keep_null, bool):
            raise ValueError("localization_keep_null must be a bool")
        null_seeding = self.seed_acceptance != "image"
        if (
            self.localization_keep_null or null_seeding
        ) and self.localization_null != "bootstrap":
            raise ValueError(
                "seed_acceptance 'null' or 'null_sd' and "
                "localization_keep_null need localization_null='bootstrap'"
            )
        if null_seeding and not self.localization_keep_null:
            raise ValueError(
                f"seed_acceptance={self.seed_acceptance!r} needs "
                "localization_keep_null"
            )

    def _solver_defaults(self) -> None:
        """Take the footprint fit's default from the solver (one default)."""
        if self.footprint_variance is None:
            from ..extraction.raster.solver_setup import SolverParams

            object.__setattr__(
                self, "footprint_variance", SolverParams.footprint_variance
            )
        _choice(
            "footprint_variance", self.footprint_variance, ("model", "fixed")
        )

    def _check_localization(self) -> None:
        """Checks of the localization options that their type cannot do."""
        _choice(
            "localization_baseline",
            self.localization_baseline,
            ("median", "network"),
        )
        if (
            not self.localization_variance_floor
            and self.localization_baseline != "network"
        ):
            # localization needs an external baseline without the floor
            raise ValueError(
                "localization_variance_floor=False needs "
                "localization_baseline='network'"
            )
        seed = self.localization_null_seed
        if (
            isinstance(seed, bool)
            or not isinstance(seed, Integral)
            or seed < 0
        ):
            raise ValueError(
                "localization_null_seed must be a nonnegative integer"
            )

    @property
    def evidence_test(self) -> bool:
        """Whether the evidence test runs (``min_source_snr`` or
        ``min_source_event_rate`` above 0)."""
        return self.min_source_snr > 0 or self.min_source_event_rate > 0

    @property
    def evidence_judge_below(self) -> Optional[float]:
        """Seed height below which the evidence test judges a source
        (``peak_th + evidence_margin``); None judges every source: without
        a margin, and with the null seed thresholds (their threshold and
        scale are not the combined image's ``peak_th`` and robust SD)."""
        if self.evidence_margin is None or self.seed_acceptance != "image":
            return None
        return float(self.peak_th) + float(self.evidence_margin)

    @property
    def sel_radius(self) -> int:
        """Radius of a source's selected pixels (``selRadius``)."""
        return math.ceil(2 * self.d_xy)

    def plan_params(self):
        """The plan stage's parameters
        (:class:`~giant_python.extraction.raster.plan.PlanParams`)."""
        from ..extraction.raster.plan import PlanParams

        return PlanParams(
            nan_thresh=self.nan_thresh,
            inter_trial_alignment=self.inter_trial_alignment,
            align_patch_px=float(self.align_patch_px),
            peak_method=self.peak_method,
            seed_fit=self.seed_fit,
            seed_merge_px=self.seed_merge_px,
            peak_th=self.peak_th,
            seed_acceptance=self.seed_acceptance,
            seed_false_per_field=self.seed_false_per_field,
            seed_amplitude_rule=self.seed_amplitude_rule,
            min_peak_distance=self.min_peak_distance,
            d_xy=self.d_xy,
            sigma_px=self.sigma_px,
            valid_kernel=self.valid_kernel,
        )

    def load_params(self):
        """The trial loader's parameters
        (:class:`~giant_python.extraction.raster.trial_data.TrialLoadParams`).
        """
        from ..extraction.raster.trial_data import TrialLoadParams

        return TrialLoadParams(
            analyze_hz=self.analyze_hz,
            activity_channel=self.activity_channel,
            photon_scale=self.photon_scale,
            baseline_window_s=self.baseline_window_s,
            nan_thresh=self.nan_thresh,
        )

    def solver_params(self):
        """The subproblem solver's parameters
        (:class:`~giant_python.extraction.raster.solver_setup.SolverParams`).
        """
        from ..extraction.raster.solver_setup import SolverParams

        return SolverParams(
            analyze_hz=self.analyze_hz,
            tau_s=self.tau_s,
            tau2_s=self.tau2_s,
            sigma_px=self.sigma_px,
            sel_radius=self.sel_radius,
            baseline_window_s=self.baseline_window_s,
            denoise_window_s=self.denoise_window_s,
            min_baseline=self.min_baseline,
            lam=self.lam,
            phi=self.phi,
            nmf_iter=self.nmf_iter,
            photon_scale=self.photon_scale,
            debias_variance=self.debias_variance,
            footprint_variance=self.footprint_variance,
            baseline_knot_factor=self.baseline_knot_factor,
            baseline_penalty=self.baseline_penalty,
            baseline_mode=self.baseline_mode,
            valid_kernel=self.valid_kernel,
        )

    @property
    def needs_network_baseline(self) -> bool:
        """Whether a stage of the run reads the network's baselines
        (localization, or the solver's baseline mode)."""
        return (
            self.localization_baseline == "network"
            or self.baseline_mode != "fit"
        )

    def localization_params(self, align_hz: float):
        """The per-trial localization's parameters (a
        :class:`~giant_python.extraction.raster.localization.RasterLocalizationParams`).

        Parameters
        ----------
        align_hz : float
            Frame rate of the registered movies (from the alignment files).
        """
        from ..extraction.raster.localization import RasterLocalizationParams

        th = self.localization_peak_th
        return RasterLocalizationParams(
            align_hz=float(align_hz),
            tau_s=self.tau_s,
            sigma_px=self.sigma_px,
            baseline_window_s=self.baseline_window_s,
            denoise_window_s=self.denoise_window_s,
            vif=self.vif,
            peak_th=self.peak_th if th is None else th,
            min_peak_distance=self.min_peak_distance,
            nan_thresh=self.nan_thresh,
            surround=self.localization_surround,
            null=self.localization_null,
            null_draws=self.localization_null_draws,
            null_seed=self.localization_null_seed,
            null_scale=self.localization_null_scale,
            variance_floor=self.localization_variance_floor,
            score=self.localization_score,
            soft_th=self.localization_soft_th,
        )

    def stage_settings(self, stage: str) -> dict:
        """The settings one stage's outputs depend on.

        Each stage records a hash of these (with its inputs' identities),
        so that a rerun keeps outputs made with the same settings.

        Parameters
        ----------
        stage : {'localize', 'plan', 'prepare', 'solve', 'collect'}

        Returns
        -------
        dict
            Field names and values (JSON types).
        """
        _choice("stage", stage, _SILO_STAGES)
        out = {n: getattr(self, n) for n in _STAGE_FIELDS[stage]}
        # fields added later count only when they differ from the value
        # that reproduces what the stage did before they existed, so
        # outputs made then stay current exactly when they still apply
        for n, before in _STAGE_FIELDS_ADDED.get(stage, {}).items():
            if getattr(self, n) != before:
                out[n] = getattr(self, n)
        if stage == "collect" and self.evidence_test:
            # which sources the evidence test judges (an option added
            # with it; without the test it changes nothing)
            out["evidence_margin"] = self.evidence_margin
        return out

    def to_matlab(
        self,
        num_channels: int,
        align_hz: float,
        n_workers: int = 1,
        start_time: str = "",
        end_time: str = "",
        save_subproblems_dir: str = "",
    ) -> dict:
        """The ``params`` struct that GIAnT-MATLAB's SILo saves.

        Names, values and classes as in ``experiment_summary.h5``'s
        ``/params`` (and the trial table's
        ``source_extraction/analysis_params``), including the fields that
        SILo and ``setParamsExtractTrial`` derive (filters, kernels,
        windows in frames). Numbers are doubles, logicals ``np.bool_``
        (saved as int8), strings ``str``; arrays are in MATLAB's
        orientation. A setting without a MATLAB name is added (in
        camelCase) when it differs from MATLAB's algorithm, so that a run
        with MATLAB's algorithm has MATLAB's fields exactly. With the
        defaults that is ``footprintVariance`` alone while the solver's
        default is ``'fixed'``. ``validKernel`` is MATLAB's own field and
        has the run's size (5 x 5 by default; MATLAB's is 3 x 3).

        Parameters
        ----------
        num_channels : int
            Channels of the recording (``numChannels``).
        align_hz : float
            Frame rate of the registered movies (``alignHz``).
        n_workers : int
            ``nWorkers`` (recorded).
        start_time, end_time : str
            ``startTime`` and ``endTime`` (ISO 8601 with milliseconds and
            the UTC offset, as MATLAB writes them).
        save_subproblems_dir : str
            ``saveSubproblemsDir``.

        Returns
        -------
        dict
        """
        sp = self.solver_params()
        out = {
            "isSLAP2": np.bool_(True),
            "includeIntegrationROIs": np.bool_(False),
            "sigma_px": float(self.sigma_px),
            "nmfIter": float(self.nmf_iter),
            "dXY": float(self.d_xy),
            "photonScale": float(self.photon_scale),
            "lambda": float(self.lam),
            "phi": float(self.phi),
            "debiasVariance": self.debias_variance,
            "denoiseWindow_s": float(self.denoise_window_s),
            "baselineWindow_Glu_s": float(self.baseline_window_s),
            "baselineWindow_Ca_s": float(self.baseline_window_ca_s),
            "activityChannel": float(self.activity_channel),
            "tau_s": float(self.tau_s),
            "tau2_s": float(self.tau2_s),
            "VIF": float(self.vif),
            "peakMethod": self.peak_method,
            "peakth": float(self.peak_th),
            "minPeakDistance": float(self.min_peak_distance),
            "interTrialAlignment": self.inter_trial_alignment,
            "alignPatchPx": float(self.align_patch_px),
            "wmfPeakth": float(self.wmf_peak_th),
            "wmfDip": float(self.wmf_dip),
            "wmfDeblendth": float(self.wmf_deblend_th),
            "minKineticScore": float(self.min_kinetic_score),
            "nWorkers": float(n_workers),
            "drawUserRois": np.bool_(self.user_rois),
            "allowAnnotationGUI": np.bool_(False),
            "motionThresh": float(self.motion_thresh),
            "analyzeHz": float(self.analyze_hz),
            "nanThresh": float(self.nan_thresh),
            "discardInitial_s": float(self.discard_initial_s),
            "saveSubproblemsDir": save_subproblems_dir,
            "minBaseline": float(self.min_baseline),
            "startTime": start_time,
            "endTime": end_time,
            "numChannels": float(num_channels),
            "alignHz": float(align_hz),
            "selRadius": float(self.sel_radius),
            "tau_full": float(sp.tau_frames),
            "tau2_samps": float(sp.tau2_frames),
            "baselineWindow_samps": float(sp.baseline_window),
            "denoiseWindow_samps": float(
                self.denoise_window_s * self.analyze_hz
            ),
            "k": np.asarray(sp.k, dtype=np.float64),
            "k2": np.asarray(sp.k2, dtype=np.float64),
            "Hfilter": np.asarray(sp.hfilter, dtype=np.float64),
            # MATLAB's Bfilter is size x size x 3
            "Bfilter": np.ascontiguousarray(sp.bfilter.transpose(1, 2, 0)),
            # MATLAB's name; MATLAB itself always has 3 x 3
            "validKernel": np.ones(
                (self.valid_kernel, self.valid_kernel), dtype=np.bool_
            ),
        }
        if self.evidence_test:
            margin = self.evidence_margin
            out["evidenceMargin"] = float(
                np.inf if margin is None else margin
            )
        for name, (key, matlab) in _PYTHON_ONLY.items():
            value = getattr(self, name)
            if value == matlab:
                continue
            if isinstance(value, bool):
                out[key] = np.bool_(value)
            elif isinstance(value, str):
                out[key] = value
            else:
                out[key] = float(value)
        return out


_STAGE_FIELDS = {
    "localize": (
        "activity_channel",
        "tau_s",
        "sigma_px",
        "baseline_window_s",
        "denoise_window_s",
        "vif",
        "motion_thresh",
        "discard_initial_s",
        "nan_thresh",
        "peak_th",
        "min_peak_distance",
        "localization_baseline",
        "localization_surround",
        "localization_null",
        "localization_null_draws",
        "localization_null_seed",
        "localization_null_scale",
        "localization_variance_floor",
        "localization_score",
        "localization_soft_th",
        "localization_peak_th",
        "trial_peaks",
    ),
    "plan": (
        "nan_thresh",
        "inter_trial_alignment",
        "peak_method",
        "peak_th",
        "min_peak_distance",
        "d_xy",
        "sigma_px",
        "user_rois",
    ),
    "prepare": (
        "analyze_hz",
        "activity_channel",
        "photon_scale",
        "baseline_window_s",
        "nan_thresh",
    ),
    "solve": (
        "analyze_hz",
        "tau_s",
        "tau2_s",
        "sigma_px",
        "d_xy",
        "baseline_window_s",
        "denoise_window_s",
        "min_baseline",
        "lam",
        "phi",
        "nmf_iter",
        "photon_scale",
        "debias_variance",
        "footprint_variance",
        "baseline_knot_factor",
        "baseline_penalty",
        "min_frames",
    ),
    "collect": (
        "analyze_hz",
        "activity_channel",
        "tau_s",
        "tau2_s",
        "min_kinetic_score",
    ),
}

# settings without a MATLAB name: their names in the saved params, and the
# value that is MATLAB's algorithm
# stage fields added after the stage files' first version, and the value
# that reproduces what the stage did before (see SiloParams.stage_settings)
_STAGE_FIELDS_ADDED = {
    "localize": {
        "seed_fit": "matlab",
        "seed_merge_px": 0.0,
        "localization_keep_null": False,
    },
    "plan": {
        "seed_fit": "matlab",
        "seed_merge_px": 0.0,
        "seed_acceptance": "image",
        "seed_false_per_field": 1.0,
        "seed_amplitude_rule": "matlab",
        "align_patch_px": 1500,
        # each stage after localization checks the source of the stage
        # before it, so changing only the source redoes them
        "baseline_source": "median",
        # the zones that group sources
        "valid_kernel": 3,
    },
    "prepare": {"baseline_source": "median"},
    "solve": {"baseline_mode": "fit", "valid_kernel": 3},
    "collect": {"min_source_snr": 0.0, "min_source_event_rate": 0.0},
}

# the settings that SiloParams.baseline_source sets, per source (see the
# class docstring)
BASELINE_SOURCE_DEFAULTS = {
    "median": {
        "localization_baseline": "median",
        "localization_surround": True,
        "localization_null": "none",
        "peak_th": 3.0,
        "min_source_event_rate": 0.0,
        "baseline_mode": "fit",
    },
    "network": {
        "localization_baseline": "network",
        "localization_surround": False,
        "localization_null": "bootstrap",
        "peak_th": 3.0,
        "min_source_event_rate": 2.0,
        "baseline_mode": "gain",
    },
}


_PYTHON_ONLY = {
    "footprint_variance": ("footprintVariance", "model"),
    "localization_baseline": ("localizationBaseline", "median"),
    "localization_surround": ("localizationSurround", True),
    "localization_null": ("localizationNull", "none"),
    "localization_null_draws": ("localizationNullDraws", 1),
    "localization_null_seed": ("localizationNullSeed", 0),
    "localization_null_scale": ("localizationNullScale", "tail"),
    "localization_variance_floor": ("localizationVarianceFloor", True),
    "localization_score": ("localizationScore", "squared_maxima"),
    "localization_soft_th": ("localizationSoftTh", 2.0),
    "localization_peak_th": ("localizationPeakth", None),
    "trial_peaks": ("trialPeaks", True),
    "baseline_knot_factor": ("baselineKnotFactor", 0.5),
    "baseline_penalty": ("baselinePenalty", 30.0),
    "min_frames": ("minFrames", 100),
    "seed_fit": ("seedFit", "matlab"),
    "seed_merge_px": ("seedMergePx", 0.0),
    "localization_keep_null": ("localizationKeepNull", False),
    "min_source_snr": ("minSourceSNR", 0.0),
    "min_source_event_rate": ("minSourceEventRate", 0.0),
    "seed_acceptance": ("seedAcceptance", "image"),
    "seed_amplitude_rule": ("seedAmplitudeRule", "matlab"),
    "seed_false_per_field": ("seedFalsePerField", 1.0),
    "baseline_source": ("baselineSource", "median"),
    "baseline_mode": ("baselineMode", "fit"),
}


_DEVICE = re.compile(r"cpu|cuda(:\d+)?")


@dataclass
class SiloExecution:
    """How SILo runs on one machine; never changes the results.

    Attributes
    ----------
    workers : int
        Processes that localize trials at once (each holds one registered
        trial in memory: a few GB for a 30-s SLAP2 trial).
    solver_workers : int
        Processes that solve subproblems at once.
    solver_threads : int
        Threads of each solver process (``OMP_NUM_THREADS`` and torch's);
        with several processes, 4 to 8 each.
    device : str
        Torch device of the solver ('cpu', 'cuda', 'cuda:N').
    dtype : {'float64', 'float32'}
        Precision of the solver's fits (float64: MATLAB's).
    batch_bytes : float or None
        Solve subproblems in padded batches of this size (pays off on a
        GPU only); None solves them one at a time.
    loader_device : str
        Torch device of the trial loader.
    overwrite : bool
        Redo units whose outputs exist and were made with the same
        settings.
    verbose : bool
        Print progress messages.
    """

    workers: int = 6
    solver_workers: int = 3
    solver_threads: int = 4
    device: str = "cpu"
    dtype: str = "float64"
    batch_bytes: Optional[float] = None
    loader_device: str = "cpu"
    overwrite: bool = False
    verbose: bool = False

    def __post_init__(self) -> None:
        """Validate the options."""
        for name in ("workers", "solver_workers", "solver_threads"):
            _positive(name, getattr(self, name), integer=True)
        for name in ("device", "loader_device"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _DEVICE.fullmatch(value):
                raise ValueError(
                    f"{name} must be 'cpu', 'cuda' or 'cuda:N', not {value!r}"
                )
        _choice("dtype", self.dtype, ("float64", "float32"))
        if self.batch_bytes is not None:
            _positive("batch_bytes", self.batch_bytes)
        for name in ("overwrite", "verbose"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")

    def solver_options(self):
        """The solver's options
        (:class:`~giant_python.extraction.raster.solver.SolverOptions`)."""
        from ..extraction.raster.solver import SolverOptions

        return SolverOptions(
            device=self.device, dtype=self.dtype, batch_bytes=self.batch_bytes
        )


SiloParamsInput = Optional[Union[SiloParams, dict]]
SiloExecutionInput = Optional[Union[SiloExecution, dict]]


def resolve_silo_options(
    params: SiloParamsInput = None,
    execution: SiloExecutionInput = None,
) -> tuple[SiloParams, SiloExecution]:
    """Resolve SILo's science and execution options.

    Parameters
    ----------
    params : SiloParams, dict or None
        Scientific settings; unknown fields raise ``TypeError``.
    execution : SiloExecution, dict or None
        Run options; unknown fields raise ``TypeError``.

    Returns
    -------
    tuple of SiloParams, SiloExecution
        Fresh validated objects.
    """
    return (
        SiloParams(**_option_values(params, SiloParams)),
        SiloExecution(**_option_values(execution, SiloExecution)),
    )
