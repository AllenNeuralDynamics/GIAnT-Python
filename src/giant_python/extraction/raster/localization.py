"""Multi-ROI (raster) SiLo localization of one trial.

Port of GIAnT-MATLAB's ``localizeSources_vIM.m`` (main at 9695d27), with an
optional external baseline (GIAnT-MATLAB branch
``localization-network-baseline``: the F0 network's ``*_BASELINE-80Hz.h5``).
One trial's movie, NaN where a pixel was not imaged or a frame was discarded,
becomes an activity image:

1. a baseline (the moving median of the 8-frame-smoothed movie, or the external
   one), and a Z-score against a variance model ``Vk b v + Vb`` fitted to the
   first 500 frames;
2. the anticausal exponential filter of the sensor's decay in time;
3. a Gaussian of ``sigma_px`` minus a Gaussian 5x as wide (the DoG) in space;
4. the sum, over the frames in which a pixel is a local maximum in space (3x3)
   and time, of the squared filtered value, divided by the number of
   observations plus 300.

The summary image (the activity image scaled, minus its 5x5 NaN-median) is what
:func:`~giant_python.extraction.raster.peaks.get_act_im_peaks` seeds sources
in.

Options, all off by default (:class:`RasterLocalizationParams`):

* ``surround=False`` keeps the Gaussian alone (no DoG surround), meant for an
  unbiased baseline such as the F0 network's.
* ``null="bootstrap"`` calibrates the activity image per pixel brightness with
  noise movies of the trial's baseline, gain, noise correlations and skew put
  through the same filters (:mod:`giant_python.extraction.raster.null`), so
  that one threshold means one false-positive rate on dim and bright pixels.
* ``variance_floor=False`` drops ``Vb`` from the Z-score's variance (meant with
  the bootstrap null, which handles the dim pixels). It needs an external
  baseline: the moving median is biased low at low counts.
* ``score="soft_threshold"`` sums ``max(0, u - soft_th)^2`` at the local maxima
  instead of their squares, with ``u`` the filtered value in robust SD units,
  so noise maxima add nothing.

Arrays are ``(frames, rows, cols)``: MATLAB's ``rows x cols x frames`` with the
frame axis first. Rows and columns are those of the registered TIFF.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import scipy.ndimage as ndimage
from numpy.lib.stride_tricks import sliding_window_view

from .peaks import get_act_im_peaks

_NULLS = ("none", "bootstrap")
_SCORES = ("squared_maxima", "soft_threshold")


@dataclass(frozen=True)
class RasterLocalizationParams:
    """Parameters of SiLo's multi-ROI localization (GIAnT-MATLAB names).

    Attributes
    ----------
    align_hz : float
        Frame rate of the registered movie (``alignHz``).
    tau_s : float
        Decay time constant of the sensor (``tau_s``).
    sigma_px : float
        Gaussian sigma of the PSF in pixels (``sigma_px``).
    baseline_window_s : float
        Moving-median window of the baseline (``baselineWindow_Glu_s``).
    denoise_window_s : float
        Twice the moving-mean window of the variance model
        (``denoiseWindow_s``).
    vif : float
        Variance inflation factor of the floor ``Vb`` (``VIF``).
    peak_th : float
        Seed threshold in robust SDs of the summary image (``peakth``).
    min_peak_distance : int
        Minimum Chebyshev distance of residual seeds (``minPeakDistance``).
    nan_thresh : float
        Largest fraction of NaN frames of an included pixel (``nanThresh``).
    surround : bool
        Subtract the 5-sigma surround (DoG); False keeps the Gaussian alone.
    null : str
        ``"none"`` or ``"bootstrap"`` (per-brightness null calibration).
    null_draws : int
        Noise movies per trial for the bootstrap null.
    null_seed : int
        Seed of the first noise movie (draw ``k`` uses ``null_seed + k``).
    null_scale : str
        Calibration per brightness: ``"tail"`` (median and 99.5th percentile)
        or ``"sd"`` (mean and SD).
    variance_floor : bool
        Add ``Vb`` to the Z-score's variance model.
    score : str
        ``"squared_maxima"`` or ``"soft_threshold"``.
    soft_th : float
        Threshold of the soft-threshold score, in robust SD units.
    """

    align_hz: float = 80.0
    tau_s: float = 0.024
    sigma_px: float = 1.33
    baseline_window_s: float = 4.0
    denoise_window_s: float = 0.2
    vif: float = 1.67
    peak_th: float = 8.0
    min_peak_distance: int = 1
    nan_thresh: float = 0.33
    surround: bool = True
    null: str = "none"
    null_draws: int = 1
    null_seed: int = 0
    null_scale: str = "tail"
    variance_floor: bool = True
    score: str = "squared_maxima"
    soft_th: float = 2.0

    def __post_init__(self) -> None:
        """Validate the option values."""
        for name in (
            "align_hz",
            "tau_s",
            "sigma_px",
            "baseline_window_s",
            "denoise_window_s",
            "vif",
        ):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive")
        if self.null not in _NULLS:
            raise ValueError(
                f"null must be one of {_NULLS}, not {self.null!r}"
            )
        if self.null_scale not in ("tail", "sd"):
            raise ValueError("null_scale must be 'tail' or 'sd'")
        if self.score not in _SCORES:
            raise ValueError(
                f"score must be one of {_SCORES}, not {self.score!r}"
            )
        if (
            isinstance(self.null_draws, bool)
            or not isinstance(self.null_draws, (int, np.integer))
            or self.null_draws < 1
        ):  # a float such as 2.0 would fail later in range()
            raise ValueError("null_draws must be a positive integer")
        if not (np.isfinite(self.soft_th) and self.soft_th >= 0):
            raise ValueError("soft_th must be a finite number >= 0")

    @property
    def tau_frames(self) -> float:
        """Decay time constant in frames."""
        return self.tau_s * self.align_hz


@dataclass
class RasterLocalization:
    """Result of :func:`localize_sources` for one trial.

    Attributes
    ----------
    act_im : ndarray of shape (rows, cols)
        The activity image (MATLAB's ``skIm``), NaN outside the valid pixels;
        calibrated (null mean 0, SD 1 per brightness) with the bootstrap null.
    summary : ndarray of shape (rows, cols)
        The scaled activity image minus its 5x5 NaN-median (what is seeded).
    peaks : ndarray of shape (N, 4)
        ``[amplitude, row, col, sigma]`` of the seeds at ``peak_th``
        (0-based), or None when peaks were not requested.
    valid : ndarray of bool
        Pixels imaged in at least ``1 - nan_thresh`` of the frames.
    info : dict
        ``Vk``, ``Vb``, the soft-threshold scale ``sd_u``, and with the
        bootstrap null the noise model and calibration.
    """

    act_im: np.ndarray
    summary: np.ndarray
    peaks: np.ndarray | None
    valid: np.ndarray
    info: dict = field(default_factory=dict)


# pixels missing more than this share of frames do not enter the variance
# model (Vb, Vk), as in MATLAB
_VAR_NAN_FRAC = 0.4


def movmean_nan(x: np.ndarray, window: int) -> np.ndarray:
    """MATLAB ``smoothdata(x, 1, 'movmean', window, 'omitnan')`` (axis 0).

    Even windows reach ``window/2`` frames back and ``window/2 - 1`` ahead, as
    MATLAB's; the window shrinks at the ends; NaN where it holds no value.
    """
    back = window // 2 if window % 2 == 0 else (window - 1) // 2
    ahead = window - 1 - back
    ok = ~np.isnan(x)
    zero = np.zeros((1,) + x.shape[1:])
    # float64 prefix sums: float32 ones drift over a long recording
    s = np.concatenate(
        [zero, np.cumsum(np.where(ok, x, 0.0), axis=0, dtype=np.float64)]
    )
    c = np.concatenate([zero, np.cumsum(ok, axis=0, dtype=np.float64)])
    n = x.shape[0]
    hi = np.minimum(np.arange(n) + ahead + 1, n)
    lo = np.maximum(np.arange(n) - back, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (s[hi] - s[lo]) / (c[hi] - c[lo])
    return out.astype(x.dtype, copy=False)


def movmedian_nan(x: np.ndarray, window: int) -> np.ndarray:
    """MATLAB ``smoothdata(x, 1, 'movmedian', window, 'omitnan')`` (axis 0).

    The median of the non-NaN values in each window (the mean of the two middle
    ones for an even count), window placement as in :func:`movmean_nan`; NaN
    where the window holds no value.
    """
    import bottleneck as bn

    back = window // 2 if window % 2 == 0 else (window - 1) // 2
    ahead = window - 1 - back
    pad_shape = lambda k: (k,) + x.shape[1:]  # noqa: E731
    padded = np.concatenate(
        [
            np.full(pad_shape(back), np.nan, x.dtype),
            x,
            np.full(pad_shape(ahead), np.nan, x.dtype),
        ]
    )
    # a trailing window ending at padded frame t + window - 1 is centred on t
    med = bn.move_median(padded, window, min_count=1, axis=0)
    return med[window - 1 : window - 1 + x.shape[0]].astype(x.dtype)


def matlab_prctile(x: np.ndarray, p: float) -> float:
    """MATLAB ``prctile(x, p, 'all')``: NaN ignored, midpoint interpolation."""
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, p, method="hazen"))


def _nanvar1(x: np.ndarray) -> np.ndarray:
    """MATLAB ``var(x, 0, 1, 'omitmissing')`` (N-1, or N for one value)."""
    ok = ~np.isnan(x)
    n = ok.sum(0)
    x64 = np.where(ok, x, 0.0).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = x64.sum(0) / n
        ss = (np.where(ok, x64 - m, 0.0) ** 2).sum(0)
        v = ss / np.maximum(n - 1, 1)
    v[n == 0] = np.nan
    return v


def gaussian_kernel(sigma: float) -> np.ndarray:
    """MATLAB ``imgaussfilt``'s 1-D kernel: radius ceil(2 sigma), sum 1."""
    r = math.ceil(2 * sigma)
    x = np.arange(-r, r + 1, dtype=np.float64)
    h = np.exp(-(x**2) / (2 * sigma**2))
    return h / h.sum()


def gauss_filter(stack: np.ndarray, sigma: float) -> np.ndarray:
    """MATLAB ``imgaussfilt(stack, [sigma sigma])`` of each frame.

    Separable, replicate padding; frames are axis 0.
    """
    h = gaussian_kernel(sigma)
    out = ndimage.correlate1d(stack, h, axis=1, mode="nearest")
    return ndimage.correlate1d(out, h, axis=2, mode="nearest")


def nanmedfilt2(a: np.ndarray, size: int = 5) -> np.ndarray:
    """GIAnT's ``nanmedfilt2``: the median of each window's non-NaN values.

    NaN padding at the border; NaN where the window has no value.
    """
    m = (size - 1) // 2
    padded = np.pad(np.asarray(a, np.float64), m, constant_values=np.nan)
    win = sliding_window_view(padded, (size, size)).reshape(
        a.shape + (size * size,)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(win, axis=-1)


def temporal_filter(imf: np.ndarray, tau: float) -> np.ndarray:
    """SiLo's anticausal exponential filter, in place (frames are axis 0).

    A NaN frame value carries the filter's memory forward, as in MATLAB.
    """
    gamma = np.float32(np.exp(-1.0 / tau))
    one_minus = np.float32(1.0 - np.exp(-1.0 / tau))
    mem = np.fmax(np.float32(0), gamma * imf[-1])
    for t in range(imf.shape[0] - 1, -1, -1):
        frame = imf[t]
        nan_t = np.isnan(frame)
        frame[nan_t] = mem[nan_t]
        imf[t] = gamma * mem + one_minus * frame
        mem = imf[t]
    return imf


def _robust_scale(imf: np.ndarray, last: int) -> float:
    """1.4826 x MAD of the scored frames' filtered values (subsampled)."""
    x = imf[1:last].reshape(-1)
    if x.size > 60_000_000:  # sample before copying (most voxels are NaN)
        x = x[:: int(np.ceil(x.size / 60_000_000))]
    x = x[~np.isnan(x)].astype(np.float64)
    if x.size > 20_000_000:
        x = x[:: int(np.ceil(x.size / 20_000_000))]
    if x.size == 0:
        return 1.0
    s = 1.4826 * float(np.median(np.abs(x - np.median(x))))
    return s if s > 0 else 1.0


def activity_score(
    imf: np.ndarray,
    nans: np.ndarray,
    params: RasterLocalizationParams,
    sd_u: float | None = None,
) -> tuple[np.ndarray, float | None]:
    """Activity image of a Z-scored, high-passed movie (filters and score).

    ``imf`` (frames, rows, cols; NaN at ``nans``) is overwritten. Returns the
    activity image (MATLAB's ``skIm`` before masking) and the scale ``sd_u`` of
    the soft-threshold score (None for squared maxima). Passing ``sd_u`` reuses
    a trial's scale, so that its null movies go through the same function.
    """
    tau = params.tau_frames
    temporal_filter(imf, tau)
    imf[nans] = np.nan
    imf[nans] = 0
    g = gauss_filter(imf, params.sigma_px)
    if params.surround:
        imf = g - gauss_filter(g, 5 * params.sigma_px)
    else:
        imf = g
    del g
    imf[nans] = np.nan

    n = imf.shape[0]
    last = n - math.ceil(1.5 * tau)  # frames 1 .. last-1 are scored (0-based)
    soft = params.score == "soft_threshold"
    if soft and sd_u is None:
        sd_u = _robust_scale(imf, last)
    elif not soft:
        sd_u = None
    sk = np.zeros(imf.shape[1:], dtype=np.float64)
    block = 128
    with np.errstate(invalid="ignore"):
        for a in range(1, last, block):
            b = min(last, a + block)  # frames a .. b-1, with neighbours
            cur = imf[a:b]
            is_max = (
                (cur == _max3x3(cur))
                & (cur > imf[a - 1 : b - 1])
                & (cur >= imf[a + 1 : b + 1])
            )
            v = np.where(is_max, cur, 0).astype(np.float64)
            if soft:
                v = np.where(
                    is_max, np.maximum(0.0, v / sd_u - params.soft_th), 0.0
                )
            sk += (v**2).sum(0)
    n_obs = (~nans[1:last]).sum(0)
    return sk / (300.0 + n_obs), sd_u


def _max3x3(stack: np.ndarray) -> np.ndarray:
    """``ordfilt2(frame, 9, ones(3))`` of each frame (axis 0).

    The 3x3 maximum with zero padding, NaN where the neighbourhood holds a NaN.
    """
    nan = np.isnan(stack)
    out = ndimage.maximum_filter(
        np.where(nan, -np.inf, stack), size=(1, 3, 3), mode="constant", cval=0
    )
    nan_win = ndimage.maximum_filter(
        nan, size=(1, 3, 3), mode="constant", cval=False
    )
    out[nan_win] = np.nan
    return out


def check_inputs(
    shape: tuple,
    params: RasterLocalizationParams,
    baseline=None,
    var_factor=None,
) -> None:
    """Reject inputs :func:`localize_sources` cannot use (before any retry).

    Raises
    ------
    ValueError
        If ``variance_floor=False`` has no external baseline, or the baseline
        or the variance factor does not have the movie's shape (a transposed
        one of the same size would otherwise scramble the pixels).
    """
    if var_factor is not None and np.shape(var_factor) != tuple(shape):
        raise ValueError(
            f"var_factor is {np.shape(var_factor)}, the movie {shape}"
        )
    if not params.variance_floor and baseline is None:
        # the moving median is biased low at low counts (often 0), so without
        # the floor dim pixels' Z-scores blow up (4x lower AP on simulations)
        raise ValueError("variance_floor=False needs an external baseline")
    if baseline is not None and np.shape(baseline) != tuple(shape):
        raise ValueError(
            f"external baseline is {np.shape(baseline)}, the movie {shape}"
        )


def summary_image(act_im: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """The image SiLo seeds in: scaled, minus its 5x5 NaN-median."""
    s = np.array(act_im, dtype=np.float64)
    top = np.nanmax(s) if np.isfinite(s).any() else np.nan
    if top > 0:
        s = s / 10.0 ** (math.floor(math.log10(top)) - 1)
    s[~valid] = np.nan
    s = s - nanmedfilt2(s, 5)
    s[~valid] = np.nan
    return s


def localize_sources(  # noqa: C901
    movie: np.ndarray,
    var_factor: np.ndarray | None = None,
    params: RasterLocalizationParams | None = None,
    baseline: np.ndarray | None = None,
    *,
    find_peaks: bool = True,
) -> RasterLocalization:
    """Localize sources in one trial (port of ``localizeSources_vIM``).

    Parameters
    ----------
    movie : ndarray of shape (frames, rows, cols)
        The trial's activity channel, NaN where not imaged or discarded.
    var_factor : ndarray of the same shape, optional
        SLAP2's variance factor (``varFacDS``); ones for raster imaging.
    params : RasterLocalizationParams, optional
        Parameters and options (defaults: GIAnT-MATLAB's SLAP2 SiLo).
    baseline : ndarray of the same shape, optional
        External baseline in the movie's units (e.g. the F0 network's); it
        replaces the moving median in the high-pass and the variance model.
    find_peaks : bool
        Seed sources at ``params.peak_th`` (MATLAB's ``P``).

    Returns
    -------
    RasterLocalization
    """
    p = params or RasterLocalizationParams()
    movie = np.asarray(movie)
    check_inputs(movie.shape, p, baseline, var_factor)
    n_t = movie.shape[0]
    baseline_window = math.ceil(p.baseline_window_s * p.align_hz)
    denoise_window = math.ceil(p.denoise_window_s * p.align_hz)
    nans = np.isnan(movie)
    valid = nans.mean(0) < p.nan_thresh
    if not valid.any():
        warnings.warn("Recording had no valid pixels; likely too much motion")
        empty = np.full(movie.shape[1:], np.nan)
        peaks = np.zeros((0, 4)) if find_peaks else None
        return RasterLocalization(empty, empty.copy(), peaks, valid)

    imf = np.array(
        movie, dtype=np.float32, order="C"
    )  # reshape must be a view
    imf[:, ~valid] = np.nan
    nans = np.isnan(imf)
    # per-pixel steps on frames x valid pixels (pixels outside are all NaN)
    vidx = np.flatnonzero(valid.ravel())
    shape = imf.shape
    flat = imf.reshape(n_t, -1)
    y = flat[:, vidx]
    vf = (
        np.ones(y.shape, np.float32)
        if var_factor is None
        else np.asarray(var_factor, np.float32).reshape(n_t, -1)[:, vidx]
    )
    vf = np.array(vf, dtype=np.float32)
    nan_v = np.isnan(y)
    vf[nan_v] = np.nan

    # the movie smoothed at a timescale where fluctuations look Gaussian, for
    # the variance model
    w = math.ceil(denoise_window / 2)
    ims = movmean_nan(y / vf, w)
    # the bootstrap's noise follows the data's per-frame variance factor; the
    # Z-score uses the smoothed one, as MATLAB
    vf_raw = vf if p.null == "bootstrap" else None
    vf = (1.0 / movmean_nan(1.0 / vf, w)).astype(np.float32)
    ims *= vf

    if baseline is not None:
        baseline = np.asarray(baseline)
        imb = np.array(baseline.reshape(n_t, -1)[:, vidx], dtype=np.float32)
        imb[nan_v] = np.nan
    else:
        imb = movmedian_nan(ims, baseline_window)

    # Vb (floor) and Vk (slope) of the variance-brightness model; MATLAB's
    # percentiles over all pixels see NaN outside the valid ones
    first = np.flatnonzero((~nans).any(axis=(1, 2)))[:500]
    var_im = _nanvar1(ims[first])
    var_im[nan_v.mean(0) > _VAR_NAN_FRAC] = np.nan  # MATLAB: nanFrac > 0.4
    vb = p.vif * matlab_prctile(var_im, 10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        var_pred = np.nanmean(imb[first].astype(np.float64), 0) * np.nanmean(
            vf[first].astype(np.float64), 0
        )
    with np.errstate(invalid="ignore", divide="ignore"):
        top = matlab_prctile(var_pred, 90)
        sel_bright = var_pred > top
        if not sel_bright.any():  # uniform brightness: MATLAB's Vk is NaN
            sel_bright = var_pred >= top
        vk = matlab_prctile(
            (var_im[sel_bright] - vb / p.vif) / var_pred[sel_bright], 10
        )
        if not p.variance_floor and not vk > 0:
            # without the floor nothing else keeps the variance positive
            vk = float(np.nanmedian(var_im / var_pred))
    del ims
    vb_used = vb if p.variance_floor else 0.0
    info: dict = {"Vk": vk, "Vb": vb_used}

    # high-pass in time (before the spatial filter, to avoid edge artifacts)
    y -= imb
    if p.null == "bootstrap":
        from .null import noise_model

        noise_frames = np.flatnonzero((~nans).any(axis=(1, 2)))[:1000]
        noise = noise_model(y, imb, vf_raw, vidx, shape[1:], noise_frames)

    with np.errstate(invalid="ignore"):
        std = np.sqrt(
            np.fmax(np.float32(0), np.float32(vk) * imb * vf)
            + np.float32(vb_used)
        )
    if not p.variance_floor:
        guard = 0.01 * np.nanmedian(std[~nan_v])
        if not guard > 0:
            raise ValueError("no positive noise variance without the floor")
        std = np.maximum(std, np.float32(guard))
    y /= std
    flat[:, vidx] = y
    del y
    if p.null != "bootstrap":
        del imb, vf, std

    sk, sd_u = activity_score(imf, nans, p)
    del imf, flat
    info["sd_u"] = sd_u
    if p.null == "bootstrap":
        from .null import null_calibration

        sk, info["null"] = null_calibration(
            sk, imb, vf_raw, std, nans, valid, vidx, noise, p, sd_u
        )
        del imb, vf, vf_raw, std

    summary = summary_image(sk, valid)
    sk[~valid] = np.nan
    peaks = (
        get_act_im_peaks(summary, p.peak_th, None, p.min_peak_distance)
        if find_peaks
        else None
    )
    return RasterLocalization(sk, summary, peaks, valid, info)
