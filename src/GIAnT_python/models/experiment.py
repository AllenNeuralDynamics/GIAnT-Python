"""Experiment summary model (``exptSummary`` / experiment_summary.h5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np


@dataclass
class Source:
    """A single extracted source (bouton/spine) with spatial + temporal data.

    Mirrors one entry of the ``sources`` struct array in GIAnT-MATLAB.

    Attributes
    ----------
    profile : ndarray or None
        Spatial profile, shaped ``(fastz, rows, cols)``.
    coords : ndarray or None
        0-indexed ``[z, y, x]`` location.
    df_ls, df_denoised, events : ndarray or None
        Least-squares dF, denoised dF, and deconvolved events,
        shaped ``(channels, frames)``.
    f0 : ndarray or None
        Per-channel baseline.
    snr : float or None
        Signal-to-noise ratio.
    """

    profile: Optional[np.ndarray] = None
    coords: Optional[np.ndarray] = None
    df_ls: Optional[np.ndarray] = None
    df_denoised: Optional[np.ndarray] = None
    events: Optional[np.ndarray] = None
    f0: Optional[np.ndarray] = None
    snr: Optional[float] = None


@dataclass
class UserRoi:
    """A manually drawn ROI (annotations.h5) and its extracted traces.

    Mirrors one entry of the ``Path{n}/roi_###`` group in ``annotations.h5``
    (geometry) together with the corresponding ``user_rois`` entry in
    ``experiment_summary.h5`` (traces).

    Attributes
    ----------
    type : str
        ``"polygon"``, ``"circle"``, or ``"ellipse"``.
    label : str
        User label (e.g. ``"SOMA"``).
    mask : ndarray or None
        Binary mask. 2-D ``(rows, cols)`` in annotations.h5; ``(fastz, rows,
        cols)`` in experiment_summary.h5.
    position : ndarray or None
        Polygon vertices ``[y, x]`` (polygon ROIs); 0-indexed when the source
        file sets ``coords_zero_indexed``.
    center : ndarray or None
        Center ``[y, x]`` (circle/ellipse ROIs).
    radius : float or None
        Radius (circle ROIs).
    semi_axes : ndarray or None
        Semi-axes (ellipse ROIs).
    rotation_angle : float or None
        Rotation angle (ellipse ROIs).
    fsvd : ndarray or None
        ROI signal after SVD / projection, shaped ``(channels, total frames)``
        (``experiment_summary.h5`` ``user_rois/Fsvd``).
    f : ndarray or None
        Raw or baseline-corrected ROI fluorescence, shaped
        ``(channels, total frames)`` (``experiment_summary.h5`` ``user_rois/F``).
    """

    type: str = "polygon"
    label: str = ""
    mask: Optional[np.ndarray] = None
    position: Optional[np.ndarray] = None
    center: Optional[np.ndarray] = None
    radius: Optional[float] = None
    semi_axes: Optional[np.ndarray] = None
    rotation_angle: Optional[float] = None
    fsvd: Optional[np.ndarray] = None
    f: Optional[np.ndarray] = None


@dataclass
class Visualizations:
    """Summary images for QC/display (``exptSummary.visualizations``).

    Attributes
    ----------
    mean_im : ndarray or None
        Mean registered image, shaped ``(channels, fastz, rows, cols)``.
    act_im : ndarray or None
        Activity / localization summary image, shaped ``(fastz, rows, cols)``.
    act_im_peaks : ndarray or None
        Activity-image peak locations, shaped ``(sources, 3)`` as 0-indexed
        ``[z_loc, y_loc, x_loc]``.
    """

    mean_im: Optional[np.ndarray] = None
    act_im: Optional[np.ndarray] = None
    act_im_peaks: Optional[np.ndarray] = None


@dataclass
class ExperimentSummary:
    """Source-extraction output (``exptSummary`` / experiment_summary.h5).

    The headline result object returned by the extraction stage. Note that
    heavy arrays are intended to be disk-backed / lazily loaded; the
    :class:`~giant_python.pipeline.pipeline.Pipeline` facade holds a reference
    to this object but does not pin its pixel data in memory.

    Attributes
    ----------
    sources : list of list of Source
        Extracted sources, indexed ``[path][source]`` (one inner list per
        imaging path / ``Path{n}`` group).
    user_rois : list of list of UserRoi
        Manually drawn ROIs and their traces, indexed ``[path][roi]``.
    visualizations : Visualizations
        Summary images (single-path convenience view; see also
        per-path structure in ``experiment_summary.h5``).
    z_depths : ndarray or None
        Z depths per imaging plane (``Path{n}/Z_depths``; SLAP2 only),
        shaped ``(fastz,)``.
    global_f : ndarray or None
        Whole-field fluorescence per channel (``Path{n}/global/F``), shaped
        ``(channels, total frames)``.
    frame_info : dict or None
        Trial/frame bookkeeping for the stitched time series
        (``Path{n}/frame_info``): ``offline{X,Y,Z}shifts``,
        ``online{X,Y,Z}shifts``, ``trial_num_frames``, ``frame_line_idxs``
        (1-indexed), and ``discard_frames``.
    params : dict or None
        The SILo parameters used to produce this summary. ``activityChannel``
        is 1-indexed into the recording's channels.
    """

    sources: list = field(default_factory=list)
    user_rois: list = field(default_factory=list)
    visualizations: Visualizations = field(default_factory=Visualizations)
    z_depths: Optional[np.ndarray] = None
    global_f: Optional[np.ndarray] = None
    frame_info: Optional[dict] = None
    params: Optional[dict] = None

    @classmethod
    def from_h5(cls, path: Union[str, Path]) -> "ExperimentSummary":
        """Load an experiment summary from experiment_summary.h5.

        Parameters
        ----------
        path : str or Path
            Path to the experiment_summary.h5 file.

        Returns
        -------
        ExperimentSummary
            The deserialized experiment summary.
        """
        raise NotImplementedError

    def to_h5(self, path: Union[str, Path]) -> None:
        """Write this experiment summary to experiment_summary.h5.

        Parameters
        ----------
        path : str or Path
            Destination path for the experiment_summary.h5 file.
        """
        raise NotImplementedError
