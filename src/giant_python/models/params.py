"""Typed parameter models and the legacy ``set_params`` helper.

The dataclasses here replace the untyped parameter dicts used in
GIAnT-MATLAB with validated, autocompleting parameter objects. The
:func:`set_params` function is retained as a dict-based compatibility shim
(a direct port of setParams.m) for code paths that still pass raw dicts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class AlignParams:
    """Motion-correction parameters (mirrors the registration params struct).

    Attributes
    ----------
    align_hz : float
        Downsampled alignment rate, in Hz.
    maxshift : float
        Maximum allowed shift, in pixels.
    clip_shift : float or None
        Clip threshold for per-frame shift changes.
    alpha : float or None
        Regularization weight.
    n_workers : int
        Number of parallel workers for per-trial processing.
    overwrite_existing : bool
        Recompute trials whose outputs already exist.
    is_revolt : bool
        Bergamo ReVolt acquisition mode.
    """

    align_hz: float = 80.0
    maxshift: float = 20.0
    clip_shift: Optional[float] = None
    alpha: Optional[float] = None
    n_workers: int = 1
    overwrite_existing: bool = False
    is_revolt: bool = False


@dataclass
class SiloParams:
    """Source-extraction (SILo) parameters (mirrors the SILo params struct).

    Attributes
    ----------
    microscope : str
        ``"slap2"`` or ``"bergamo"``.
    scan_mode : str
        SLAP2 scan mode selecting the extraction backend:
        ``"standard"`` (pixel-movie SILo) or ``"integration"`` (the
        band-scan / superpixel variant, paired with BandRegistration).
    sigma_px : float
        Spatial smoothing sigma, in pixels.
    lambda_ : float
        NMF sparsity weight (``lambda`` in MATLAB; renamed to avoid the
        Python keyword).
    phi : float or None
        NMF spatial-coherence weight.
    tau_s : float or None
        Deconvolution time constant, in seconds.
    photon_scale : float or None
        Photons per digital unit (variance modeling).
    peakth : float or None
        Activity-image peak detection threshold.
    activity_channel : int
        Channel index used for activity detection.
    draw_user_rois : bool
        Whether to prompt for user ROIs before extraction.
    analyze_hz : float or None
        Temporal analysis rate, in Hz. When ``None``, the BandSILo backend
        falls back to ``100``.

    Notes
    -----
    The remaining attributes below are specific to the ``"integration"``
    (BandSILo) backend and mirror the GUI defaults in
    ``extractSLAP2IntegrationSources.py``. They are ignored by the standard
    (pixel-movie) SILo backend.

    decay_tau_s : float
        Calcium-decay time constant, in seconds.
    baseline_window_s : float
        Rolling-baseline window for dF/F, in seconds.
    denoise_window_s : float
        Temporal denoising window, in seconds.
    vif : float
        Variance inflation factor for the noise model.
    d_xy : int
        Spatial downsampling / superpixel spacing, in pixels.
    sparse_fac : float
        NMF sparsity factor (the GUI exposes its natural-log; the default is
        ``exp(-3)``).
    peak_buffer : int
        Peak exclusion buffer diameter, in pixels.
    max_workers : int
        Number of worker processes for per-trial extraction.
    num_channels : int or None
        Number of acquisition channels; when ``None``, read from the aData
        alignment file at runtime.
    psf_dilation : int
        Dilation size selecting the bundled ``dil-NN.tif`` PSF template.
    operator : str
        Operator name recorded in the output metadata.
    """

    microscope: str = "slap2"
    scan_mode: str = "standard"
    sigma_px: float = 1.5
    lambda_: float = 0.1
    phi: Optional[float] = None
    tau_s: Optional[float] = None
    photon_scale: Optional[float] = None
    peakth: Optional[float] = None
    activity_channel: int = 0
    draw_user_rois: bool = False
    analyze_hz: Optional[float] = None
    decay_tau_s: float = 0.15
    baseline_window_s: float = 4.0
    denoise_window_s: float = 1.0
    vif: float = 1.38
    d_xy: int = 5
    sparse_fac: float = math.exp(-3.0)
    peak_buffer: int = 3
    max_workers: int = 6
    num_channels: Optional[int] = None
    psf_dilation: int = 17
    operator: str = "SLAP2 User"


def set_params(
    fn_name: str,
    params_in: Optional[dict] = None,
) -> dict:
    """Return a parameter dictionary for the named GIAnT function.

    Populates default parameters for the specified function and merges any
    caller-supplied overrides. Corresponds to setParams.m in GIAnT-MATLAB.
    Prefer the typed :class:`AlignParams` / :class:`SiloParams` models for new
    code; this shim exists for dict-based compatibility.

    Parameters
    ----------
    fn_name : str
        Name of the GIAnT function whose parameters should be configured.
        Supported values: ``"SILo"``, ``"MultiRoiRegistration"``,
        ``"StripRegistration"``, ``"BandRegistration"``.
    params_in : dict, optional
        Caller-supplied parameter overrides. Any key present in *params_in*
        takes precedence over the corresponding default value.

    Returns
    -------
    dict
        Parameter dictionary with defaults merged with *params_in*.

    Raises
    ------
    ValueError
        If *fn_name* is not a recognised GIAnT function name.
    """
    raise NotImplementedError
