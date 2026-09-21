"""Scientific configuration and independent execution and annotation policy.

``BandSiloParams`` contains only band science and channel overrides.
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


@dataclass
class BandSiloParams:
    """Scientific settings for the implemented SLAP2 band algorithm.

    Attributes
    ----------
    analyze_hz : float
        Temporal analysis rate in Hz.
    decay_tau_s : float
        Calcium-decay time constant in seconds.
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
    num_channels : int or None
        Positive channel-count override; otherwise infer from acquisition
        metadata without modifying this object.
    activity_channel : int
        Zero-based activity channel. Only channel zero is implemented.

    Notes
    -----
    Scheduling, trial limits, logging and annotation policy are deliberately
    separate from science. Numerical defaults preserve the band algorithm.
    Sparse-superpixel background interpolation always uses cubic splines.
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
    num_channels: Optional[int] = None
    activity_channel: int = 0

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
        ):
            _positive(name, getattr(self, name))
        for name in ("d_xy", "peak_buffer", "psf_dilation"):
            _positive(name, getattr(self, name), integer=True)
        if self.num_channels is not None:
            _positive("num_channels", self.num_channels, integer=True)
        if (
            isinstance(self.activity_channel, bool)
            or not isinstance(self.activity_channel, Integral)
            or self.activity_channel != 0
        ):
            raise ValueError("Only activity_channel=0 is implemented")


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
    """

    max_workers: int = 6
    verbose: bool = False
    max_trials: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate scheduling and debug limits."""
        _positive("max_workers", self.max_workers, integer=True)
        if self.max_trials is not None:
            _positive("max_trials", self.max_trials, integer=True)
        if not isinstance(self.verbose, bool):
            raise ValueError("verbose must be a bool")


@dataclass
class AnnotationOptions:
    """ROI inclusion and GUI policy, separate from scientific settings.

    Attributes
    ----------
    enabled : bool
        Include user ROIs during extraction.
    interactive : bool or None
        Allow or forbid drawing; ``None`` retains backend auto-detection.
    operator : str
        Operator name recorded in metadata.
    """

    enabled: bool = False
    interactive: Optional[bool] = None
    operator: str = "SLAP2 User"

    def __post_init__(self) -> None:
        """Validate the explicit and automatic annotation policies."""
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool")
        if self.interactive is not None and not isinstance(
            self.interactive, bool
        ):
            raise ValueError("interactive must be a bool or None")
        if not isinstance(self.operator, str):
            raise ValueError("operator must be a string")


BandParamsInput = Optional[Union[BandSiloParams, dict]]
ExecutionInput = Optional[Union[ExecutionOptions, dict]]
AnnotationInput = Optional[Union[AnnotationOptions, dict]]


def _option_values(
    option: object,
    model: type[Union[BandSiloParams, ExecutionOptions, AnnotationOptions]],
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
