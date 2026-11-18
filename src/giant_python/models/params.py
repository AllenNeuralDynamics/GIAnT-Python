"""Scientific configuration and independent execution and annotation policy.

``BandSiloParams`` contains only band science.
``ExecutionOptions`` controls scheduling/debug limits; ``AnnotationOptions``
controls ROI use and GUI policy. Resolution copies inputs without retuning
the existing algorithm.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from numbers import Integral, Real
from typing import Optional, Union


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
        Re-register trials whose outputs exist.
    save_tif : bool
        Save the registered movie as TIFF (else HDF5 ``/data``).
    ref_stack_template, is_revolt, include_integration_rois : bool
        MATLAB options not yet ported; must be False.
    compensate_online : bool
        Undo the online motion correction line by line; not yet ported,
        must be False.
    init_frames : int
        Frames matched pairwise for the initial template (MATLAB: 40).
    init_search : int
        Largest shift of the pairwise matching (MATLAB: 3).
    min_samples : int
        Contributions a template pixel needs (MATLAB minSamps: 15).
    """

    align_hz: float = 80.0
    maxshift: int = 40
    clip_shift: int = 5
    alpha: float = 0.005
    overwrite_existing: bool = False
    save_tif: bool = True
    ref_stack_template: bool = False
    is_revolt: bool = False
    include_integration_rois: bool = False
    compensate_online: bool = False
    init_frames: int = 40
    init_search: int = 3
    min_samples: int = 15

    def __post_init__(self) -> None:
        """Validate values and reject options that are not ported."""
        _positive("align_hz", self.align_hz)
        for name in (
            "maxshift",
            "clip_shift",
            "init_frames",
            "init_search",
            "min_samples",
        ):
            _positive(name, getattr(self, name), integer=True)
        _nonnegative("alpha", self.alpha)
        for name in (
            "overwrite_existing",
            "save_tif",
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
            "compensate_online",
        ):
            if getattr(self, name):
                raise NotImplementedError(
                    f"{name}=True is not supported yet in GIAnT-Python"
                )

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
