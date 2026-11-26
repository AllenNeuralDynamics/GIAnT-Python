"""Parameters, filters and inputs of SiLo's subproblem solver.

Port of GIAnT-MATLAB's ``setParamsExtractTrial.m`` (the footprint filter,
the three surround filters, the temporal kernels and the windows) and of
the inputs that ``extractTrial.m`` hands to its subproblem solver.

A subproblem is one connected group of sources in one trial of one imaging
path. Its pixels are listed in MATLAB's column-major order of the 2-D
selected-pixel map (``find(selPix)``), so pixel ``k`` here is row ``k`` of
MATLAB's arrays. MATLAB v7.3 files read with h5py have their dimensions
reversed.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse

from .localization import gauss_filter


def gaussian_kernel_1d(sigma: float, size: int) -> np.ndarray:
    """MATLAB ``imgaussfilt``'s 1-D kernel of an odd ``size``, sum 1.

    Values below ``eps`` times the largest are set to 0 before the
    normalization, as ``images.internal.createGaussianKernel`` does.
    """
    r = (size - 1) / 2
    x = np.arange(-r, r + 1, dtype=np.float64)
    h = np.exp(-(x * x) / (sigma * sigma) / 2)
    h[h < np.finfo(np.float64).eps * h.max()] = 0
    return h / h.sum()


def _impulse_response(sigma: float, size: int) -> np.ndarray:
    """``imgaussfilt`` of a centred impulse with ``FilterSize`` = its size."""
    g = gaussian_kernel_1d(sigma, size)
    return g[:, None] * g[None, :]


def temporal_kernel(tau: float) -> np.ndarray:
    """Causal exponential kernel of ``setParamsExtractTrial``, sum 1.

    ``ceil(6 tau)`` zeros, then ``exp(-(0:ceil(6 tau)) / tau)``: convolving
    events with it ('same') gives the activity, and an event at frame t
    peaks at frame t.
    """
    n = math.ceil(6 * tau)
    k = np.concatenate([np.zeros(n), np.exp(-np.arange(n + 1) / tau)])
    return k / k.sum()


# the solver's ways of finding the baseline (SolverParams.baseline_mode)
BASELINE_MODES = ("fit", "fixed", "gain", "gain_shared", "fit_from")

_POSITIVE = (
    "analyze_hz",
    "tau_s",
    "tau2_s",
    "sigma_px",
    "sel_radius",
    "baseline_window_s",
    "denoise_window_s",
    "min_baseline",
    "baseline_knot_factor",
)
_POSITIVE_IF_GIVEN = ("tau_full", "photon_scale")
_NOT_NEGATIVE = ("lam", "phi", "baseline_penalty")


def _check_numbers(params) -> None:
    """Reject numbers that are not finite or have the wrong sign."""
    given = [n for n in _POSITIVE_IF_GIVEN if getattr(params, n) is not None]
    for name in list(_POSITIVE) + given:
        v = getattr(params, name)
        if not (v > 0 and math.isfinite(v)):
            raise ValueError(f"{name} must be positive and finite")
    for name in _NOT_NEGATIVE:
        v = getattr(params, name)
        if not (v >= 0 and math.isfinite(v)):
            raise ValueError(f"{name} must not be negative")


@dataclass(frozen=True)
class SolverParams:
    """Parameters of the subproblem solver.

    The defaults are SILo's for SLAP2, and ``footprint_variance``, which
    MATLAB does not have, is 'fixed'.

    Attributes
    ----------
    analyze_hz : float
        Frame rate of the extraction (``params.analyzeHz``).
    tau_s, tau2_s : float
        Decay time constants of the activity channel and of channel 2.
    sigma_px : float
        Gaussian sigma of the point-spread function, pixels.
    sel_radius : int
        Radius of the selected region around a source, pixels.
    baseline_window_s : float
        Timescale of the baseline (``baselineWindow_Glu_s``).
    denoise_window_s : float
        Timescale over which signals may be smoothed.
    min_baseline : float
        Lower bound of the baseline, in normalized units.
    lam : float
        L1 penalty on the events (``params.lambda``).
    phi : float
        Share of ``lam`` kept in the debias refit.
    nmf_iter : int
        Outer iterations (activity, baseline, footprints).
    photon_scale : float or None
        Counts per photon that the data were divided by. None for a
        recording whose scale ``extractTrial`` estimates.
    debias_variance : {'fixed', 'model'}
        Variance term of the debias refit: held at its value at the
        refit's start (a weighted least-squares refit) or following the
        model.
    footprint_variance : {'fixed', 'model'}
        Variance term of the footprint fit. 'fixed' (the default) holds it
        at its value at the fit's start: a weighted least-squares fit, as
        the debias refit's. 'model' lets it follow the model, which is
        MATLAB's loss; a wider footprint then lowers every sample's loss,
        and the converged footprints come out too spread. On the
        ground-truth simulations 'fixed' gives footprints closer to the
        true ones and denoised traces that correlate better with the
        truth than MATLAB's. Its footprints are more compact than
        MATLAB's, so F0 and the amplitudes come out lower (by about 16%
        and 4% there). Not a MATLAB option.
    baseline_knot_factor : float
        Knot spacing of the baseline spline, in baseline windows.
    baseline_penalty : float
        Curvature penalty of the baseline spline, in knot intervals' worth
        of data weight.
    tau_full : float or None
        ``tau_s`` in frames; None for ``tau_s * analyze_hz``.
    baseline_mode : str
        How the activity channel's baseline is found (the variants of the
        baseline test; see :mod:`~giant_python.extraction.raster.solver`):
        'fit' (MATLAB's: ``splitFreq`` start, then ``fitB``; R0), 'fixed'
        (an external baseline, held; N0), 'gain' (the external baseline
        times one gain per pixel; N1), 'gain_shared' (N1 times a time
        course shared by the subproblem's pixels; N2) or 'fit_from'
        (``fitB`` of the correction to the external baseline; N3). All but
        'fit' need ``Subproblem.baseline``. Not a MATLAB option.
    outside_footprint : float
        'gain_shared': the shared time course is estimated from the pixels
        where every source's footprint is below this share of its largest
        value.
    valid_kernel : int
        Side of the square around each source where the footprint fit may
        put intensity (MATLAB's ``validKernel``, 3 x 3; odd). The footprint
        is ``hfilter`` blurring that support, so its width (RMS per axis)
        lies between ``hfilter``'s sigma (0.9 ``sigma_px``) and
        ``sqrt(sigma**2 + r**2)``, r = (valid_kernel - 1) / 2 (all weight
        on the corners): 1.20 to 1.56 px for 3 and ``sigma_px`` 1.33, up
        to 2.33 px for 5. A Gaussian footprint of sigma 1.8 px is matched
        best at 1.56 px with 3 and at 1.74 px with 5.
        ``PlanParams.valid_kernel`` is the same quantity for the plan's
        zones.
    """

    analyze_hz: float = 200.0
    tau_s: float = 0.024
    tau2_s: float = 0.15
    sigma_px: float = 1.33
    sel_radius: int = 6
    baseline_window_s: float = 4.0
    denoise_window_s: float = 0.2
    min_baseline: float = 0.01
    lam: float = 0.9
    phi: float = 0.0
    nmf_iter: int = 2
    photon_scale: float | None = 25.0
    debias_variance: str = "fixed"
    footprint_variance: str = "fixed"
    baseline_knot_factor: float = 0.5
    baseline_penalty: float = 30.0
    tau_full: float | None = None
    baseline_mode: str = "fit"
    outside_footprint: float = 0.1
    valid_kernel: int = 3

    def __post_init__(self) -> None:
        """Reject values the solver cannot run with."""
        for name in ("debias_variance", "footprint_variance"):
            if getattr(self, name) not in ("fixed", "model"):
                raise ValueError(
                    f"{name} must be 'fixed' or 'model', not "
                    f"{getattr(self, name)!r}"
                )
        if self.baseline_mode not in BASELINE_MODES:
            raise ValueError(
                f"baseline_mode must be one of {BASELINE_MODES}, not "
                f"{self.baseline_mode!r}"
            )
        if not 0 < self.outside_footprint < 1:
            raise ValueError("outside_footprint must lie in (0, 1)")
        vk = self.valid_kernel
        if isinstance(vk, bool) or not isinstance(vk, (int, np.integer)):
            raise ValueError(f"valid_kernel must be an integer, not {vk!r}")
        if vk < 1 or vk % 2 == 0:
            raise ValueError(f"valid_kernel must be odd and positive, not {vk}")
        if self.nmf_iter < 1:
            raise ValueError(f"nmf_iter must be >= 1, not {self.nmf_iter}")
        _check_numbers(self)
        if self.lp_factor < 2:
            # splitFreq takes a low order statistic over this many pages
            raise ValueError(
                "baseline_window_s must be longer than denoise_window_s "
                "(in whole frames): the baseline is tracked over "
                f"{self.lp_factor} denoise window(s), and needs at least 2"
            )

    @property
    def tau_frames(self) -> float:
        """Decay time constant of the activity channel, frames."""
        if self.tau_full is not None:
            return float(self.tau_full)
        return self.tau_s * self.analyze_hz

    @property
    def tau2_frames(self) -> float:
        """Decay time constant of channel 2, frames."""
        return self.tau2_s * self.analyze_hz

    @property
    def baseline_window(self) -> float:
        """``baselineWindow_samps`` (not rounded, as in MATLAB)."""
        return self.baseline_window_s * self.analyze_hz

    @property
    def denoise_window(self) -> int:
        """``ceil(denoiseWindow_samps)``, as the solver uses it."""
        return math.ceil(self.denoise_window_s * self.analyze_hz)

    @property
    def lp_factor(self) -> int:
        """Pages per baseline window in :func:`split_freq`."""
        return math.ceil(self.baseline_window / self.denoise_window)

    @cached_property
    def k(self) -> np.ndarray:
        """Temporal kernel of the activity channel."""
        return temporal_kernel(self.tau_frames)

    @cached_property
    def k2(self) -> np.ndarray:
        """Temporal kernel of channel 2."""
        return temporal_kernel(self.tau2_frames)

    @cached_property
    def hfilter(self) -> np.ndarray:
        """Filter that turns super-resolved footprints Hs into H."""
        size = 4 * math.ceil(self.sigma_px) + 1
        return _impulse_response(max(0.5, 0.9 * self.sigma_px), size)

    @cached_property
    def bfilter(self) -> np.ndarray:
        """The three surround filters, ``(3, size, size)``.

        A ring (wide minus narrow Gaussian, negative values cut), and the
        ring times a left-right and an up-down gradient.
        """
        r = int(self.sel_radius)
        size = 4 * r + 1
        ring = np.maximum(
            0, _impulse_response(r, size) - _impulse_response(r / 2, size)
        )
        mask = np.zeros((size, size))
        c = math.ceil(size / 2) - 1
        mask[:, c + 1 :] = 1
        mask[:, c] = 0.5
        mask = mask - mask.mean()
        return np.stack([ring, ring * mask, ring * mask.T])


@dataclass
class Subproblem:
    """Inputs of one subproblem, in the convention shared by the port.

    Attributes
    ----------
    y : ndarray of shape (channels, pixels, frames)
        Data in counts / photonScale; channel 0 is the activity channel.
        float32 in containers; MATLAB's files load as float64.
    finv : ndarray of shape (pixels, frames)
        The variance multiplier (1 / freshness).
    sel : ndarray of bool, shape (rows, cols)
        Crop of the consensus map of selected pixels; its true entries, in
        column-major order, are the pixels.
    sources_rc : ndarray of int, shape (n, 2)
        Zero-based source positions (row, col) within the crop.
    pixels : ndarray of int or None
        Indices of the pixels in the trial's selected-pixel list.
    sources : ndarray of int or None
        Indices of the sources in the path's source list.
    row0, col0 : int
        Origin of the crop in the consensus frame (zero-based).
    baseline : ndarray of shape (pixels, frames) or None
        An external baseline of the activity channel in the data's units
        (the F0 network's, mapped by
        :func:`~giant_python.extraction.raster.baseline_map.map_baseline`),
        for ``SolverParams.baseline_mode`` other than 'fit'.
    """

    y: np.ndarray
    finv: np.ndarray
    sel: np.ndarray
    sources_rc: np.ndarray
    pixels: np.ndarray | None = None
    sources: np.ndarray | None = None
    row0: int = 0
    col0: int = 0
    baseline: np.ndarray | None = None
    _coords: tuple | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Check that the arrays fit together."""
        self.sel = np.asarray(self.sel, dtype=bool)
        self.sources_rc = np.asarray(self.sources_rc, dtype=np.int64)
        self.sources_rc = self.sources_rc.reshape(-1, 2)
        if self.y.ndim != 3:
            raise ValueError("y must be (channels, pixels, frames)")
        n_px = int(self.sel.sum())
        if self.y.shape[1] != n_px or self.finv.shape != self.y.shape[1:]:
            raise ValueError(
                f"sel has {n_px} pixels; y is {self.y.shape} and finv "
                f"{self.finv.shape}"
            )
        for name in ("pixels", "sources"):
            index = getattr(self, name)
            if index is not None and np.asarray(index).min(initial=0) < 0:
                raise ValueError(f"{name} holds a negative index")
        r, c = self.sources_rc[:, 0], self.sources_rc[:, 1]
        inside = (
            (r >= 0)
            & (r < self.sel.shape[0])
            & (c >= 0)
            & (c < self.sel.shape[1])
        )
        if not inside.all():
            raise ValueError("sources_rc lies outside the crop")
        if np.unique(self.sources_rc, axis=0).shape[0] != len(r):
            # two footprints would be the same: no unique activity
            raise ValueError("two sources share a position")
        if self.baseline is not None:
            if np.shape(self.baseline) != self.y.shape[1:]:
                raise ValueError(
                    f"the baseline is {np.shape(self.baseline)}; the data "
                    f"have {self.y.shape[1:]} pixels and frames"
                )
            if not np.isfinite(self.baseline).all():
                raise ValueError("the baseline holds NaN or infinite values")

    @property
    def n_pixels(self) -> int:
        """Number of selected pixels."""
        return self.y.shape[1]

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return self.y.shape[2]

    @property
    def n_sources(self) -> int:
        """Number of sources."""
        return self.sources_rc.shape[0]

    @property
    def coords(self) -> tuple[np.ndarray, np.ndarray]:
        """Rows and columns of the pixels within the crop."""
        if self._coords is None:
            self._coords = sel_coords(self.sel)
        return self._coords


def sel_coords(sel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rows and columns of ``find(sel)``: column-major order."""
    cols, rows = np.nonzero(np.asarray(sel, dtype=bool).T)
    return rows, cols


def conv_operator(sel: np.ndarray, filt: np.ndarray) -> sparse.csr_matrix:
    """``convn(map, filt, 'same')`` on the selected pixels, as a matrix.

    The map is zero outside ``sel``; the result is read at the selected
    pixels. Row p of the (pixels x pixels) matrix holds the filter values
    that pixel p receives from each selected pixel (MATLAB's
    ``getSurround``).
    """
    sel = np.asarray(sel, dtype=bool)
    rows, cols = sel_coords(sel)
    n = rows.size
    order = np.full(sel.shape, -1, dtype=np.int64)
    order[rows, cols] = np.arange(n)
    fr, fc = np.nonzero(filt)
    fv = filt[fr, fc]
    src_r = rows[:, None] + (filt.shape[0] // 2 - fr)[None, :]
    src_c = cols[:, None] + (filt.shape[1] // 2 - fc)[None, :]
    inside = (
        (src_r >= 0)
        & (src_r < sel.shape[0])
        & (src_c >= 0)
        & (src_c < sel.shape[1])
    )
    src = np.full(src_r.shape, -1, dtype=np.int64)
    src[inside] = order[src_r[inside], src_c[inside]]
    use = src >= 0
    dst = np.broadcast_to(np.arange(n)[:, None], src.shape)
    wts = np.broadcast_to(fv[None, :], src.shape)
    return sparse.csr_matrix(
        (wts[use], (dst[use], src[use])), shape=(n, n), dtype=np.float64
    )


def initial_footprints(
    sel: np.ndarray,
    sources_rc: np.ndarray,
    sigma_px: float,
    valid_kernel: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Starting footprints of ``extractSources``.

    Returns
    -------
    hs : ndarray of shape (pixels, sources)
        Super-resolved footprints: a plus-shaped 5 pixels at each source,
        sum 1 (over the crop).
    h : ndarray of shape (pixels, sources)
        ``hs`` blurred with a Gaussian of ``sigma_px`` (replicate padding).
    valid : ndarray of bool, shape (pixels, sources)
        The ``valid_kernel`` x ``valid_kernel`` pixels around each source
        (MATLAB: 3 x 3), where the footprint fit may put intensity.
    """
    r_v = (int(valid_kernel) - 1) // 2
    sel = np.asarray(sel, dtype=bool)
    rows, cols = sel_coords(sel)
    n_src = len(sources_rc)
    shape = (rows.size, n_src)
    hs, h = np.zeros(shape), np.zeros(shape)
    valid = np.zeros(shape, dtype=bool)
    nr, nc = sel.shape
    for i, (r, c) in enumerate(np.asarray(sources_rc, dtype=np.int64)):
        tmp = np.zeros(sel.shape)
        for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            if 0 <= r + dr < nr and 0 <= c + dc < nc:
                tmp[r + dr, c + dc] = 1
        tmp /= tmp.sum()
        ok = np.zeros(sel.shape, dtype=bool)
        ok[max(r - r_v, 0) : r + r_v + 1, max(c - r_v, 0) : c + r_v + 1] = True
        hs[:, i] = tmp[rows, cols]
        h[:, i] = gauss_filter(tmp[None], sigma_px)[0][rows, cols]
        valid[:, i] = ok[rows, cols]
    return hs, h, valid


def _is_empty(ds: h5py.Dataset) -> bool:
    """Whether a v7.3 dataset holds an empty MATLAB array."""
    return "MATLAB_empty" in ds.attrs and bool(ds.attrs["MATLAB_empty"])


def _mat_scalar(g: h5py.Group, name: str, default=None):
    """A numeric scalar field of a MATLAB struct, or ``default``."""
    if name not in g or _is_empty(g[name]):
        return default
    return float(np.asarray(g[name]).ravel()[0])


def _mat_str(g: h5py.Group, name: str, default: str) -> str:
    """A char field of a MATLAB struct, or ``default``."""
    if name not in g or _is_empty(g[name]):
        return default
    codes = np.asarray(g[name]).ravel()
    return "".join(chr(int(c)) for c in codes)


def _mat_valid_kernel(g: h5py.Group, default: int) -> int:
    """The side of a saved ``validKernel``: a square of trues, odd side."""
    if "validKernel" not in g or _is_empty(g["validKernel"]):
        return default
    k = np.asarray(g["validKernel"])
    ones = k == 1  # not astype(bool): 2, -1 or NaN are not ones
    if k.ndim != 2 or k.shape[0] != k.shape[1] or not ones.all():
        raise ValueError(
            f"validKernel must be a square of ones; it is {k.shape} with "
            f"{int(ones.sum())} ones"
        )
    return int(k.shape[0])


def params_from_matlab(g: h5py.Group) -> SolverParams:
    """The solver's parameters from a saved MATLAB ``params`` struct.

    The filters and kernels are derived again from the scalar parameters;
    a warning is given if they differ from the saved ones.
    ``footprint_variance`` is not a MATLAB parameter: it is read from a
    ``footprintVariance`` field if the struct has one, and is
    :class:`SolverParams`'s default ('fixed') otherwise. For MATLAB's loss
    use ``dataclasses.replace(params, footprint_variance='model')``.
    """
    d = SolverParams()

    def num(name, default):
        """A scalar field, or the default."""
        return _mat_scalar(g, name, default)

    p = SolverParams(
        analyze_hz=num("analyzeHz", d.analyze_hz),
        tau_s=num("tau_s", d.tau_s),
        tau2_s=num("tau2_s", d.tau2_s),
        sigma_px=num("sigma_px", d.sigma_px),
        sel_radius=int(num("selRadius", d.sel_radius)),
        baseline_window_s=num("baselineWindow_Glu_s", d.baseline_window_s),
        denoise_window_s=num("denoiseWindow_s", d.denoise_window_s),
        min_baseline=num("minBaseline", d.min_baseline),
        lam=num("lambda", d.lam),
        phi=num("phi", d.phi),
        nmf_iter=int(num("nmfIter", d.nmf_iter)),
        photon_scale=num("photonScale", None),
        debias_variance=_mat_str(g, "debiasVariance", "fixed").lower(),
        footprint_variance=_mat_str(
            g, "footprintVariance", d.footprint_variance
        ).lower(),
        baseline_knot_factor=num("baselineKnotFactor", 0.5),
        baseline_penalty=num("baselinePenalty", 30.0),
        tau_full=num("tau_full", None),
        valid_kernel=_mat_valid_kernel(g, d.valid_kernel),
    )
    for name, ours in (
        ("k", p.k),
        ("k2", p.k2),
        ("Hfilter", p.hfilter),
        ("Bfilter", p.bfilter),
    ):
        if name not in g or _is_empty(g[name]):
            continue
        saved = np.asarray(g[name], dtype=np.float64)
        saved = saved.transpose(*range(saved.ndim)[::-1])
        if name == "Bfilter":
            saved = saved.transpose(2, 0, 1)
        saved = saved.reshape(ours.shape) if saved.size == ours.size else saved
        if saved.shape != ours.shape or not np.allclose(
            saved, ours, rtol=1e-9, atol=1e-12
        ):
            warnings.warn(
                f"saved params.{name} differs from the one derived from "
                "the scalar parameters; the derived one is used",
                stacklevel=2,
            )
    return p


def _trial_pixel_index(trial_file: Path, row0, col0, sel) -> np.ndarray:
    """Indices of a crop's pixels in the trial's selected-pixel list."""
    with h5py.File(trial_file, "r") as f:
        full = np.asarray(f["selPix"], dtype=bool).T
    index = np.full(full.shape, -1, dtype=np.int64)
    rows, cols = sel_coords(full)
    index[rows, cols] = np.arange(rows.size)
    r, c = sel_coords(sel)
    inside = (r + row0 < full.shape[0]) & (c + col0 < full.shape[1])
    if not inside.all() or (index[r + row0, c + col0] < 0).any():
        raise ValueError(
            f"{trial_file} does not select every pixel of the subproblem: "
            "the two files are not from the same trial"
        )
    return index[r + row0, c + col0]


def load_matlab_subproblem(
    path, dtype=np.float64
) -> tuple[Subproblem, SolverParams]:
    """One of MATLAB's saved subproblems (``sub_###.mat``) and its params.

    ``extractTrial`` saves them with ``params.saveSubproblemsDir`` set:
    ``Y_p`` (already divided by ``photonScale``), ``F_inv_p``,
    ``sources_p``, ``selPix_p``, ``params``, and the crop's place in the
    trial's map. The pixels' indices in the trial's selected-pixel list
    come from ``trial.mat`` beside the file, when it is there.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        y = np.asarray(f["Y_p"], dtype=dtype)
        if y.ndim == 2:
            y = y[None]
        y = np.ascontiguousarray(y.transpose(0, 2, 1))
        finv = np.ascontiguousarray(np.asarray(f["F_inv_p"], dtype=dtype).T)
        sel = np.asarray(f["selPix_p"], dtype=bool).T
        rc = np.stack(
            [
                np.asarray(f["sources_p"]["R"]).ravel(),
                np.asarray(f["sources_p"]["C"]).ravel(),
            ],
            axis=1,
        )
        rc = np.round(rc).astype(np.int64) - 1
        params = params_from_matlab(f["params"])
        row0 = col0 = 0
        sources = None
        if "rowSupport" in f:
            row0 = int(np.asarray(f["rowSupport"]).ravel()[0]) - 1
            col0 = int(np.asarray(f["colSupport"]).ravel()[0]) - 1
        if "srcIdx" in f:
            sources = np.asarray(f["srcIdx"]).ravel().astype(np.int64) - 1
    pixels = None
    trial_file = path.with_name("trial.mat")
    if trial_file.exists():
        pixels = _trial_pixel_index(trial_file, row0, col0, sel)
    sub = Subproblem(
        y=y,
        finv=finv,
        sel=sel,
        sources_rc=rc,
        pixels=pixels,
        sources=sources,
        row0=row0,
        col0=col0,
    )
    return sub, params
