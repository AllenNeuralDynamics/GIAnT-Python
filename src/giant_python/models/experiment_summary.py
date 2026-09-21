"""Experiment summary model (``exptSummary`` / experiment_summary.h5)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
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
    snr: Optional[Union[float, np.floating]] = None


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
        ``(channels, total frames)``
        (``experiment_summary.h5`` ``user_rois/F``).
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
class FrameInfo:
    """Eager, typed frame bookkeeping.

    Offline shifts carry saved MATLAB displacement, NOT private band
    reference offsets. No conversion occurs in this model or its codec.
    Frame line indexes remain 1-based; emitted vectors retain rank two.
    ``to_dict`` returns a shallow mapping, sharing the original arrays.
    """

    trial_num_frames: Optional[np.ndarray] = None
    discard_frames: Optional[np.ndarray] = None
    frame_line_idxs: Optional[np.ndarray] = None
    offlineXshifts: Optional[np.ndarray] = None
    offlineYshifts: Optional[np.ndarray] = None
    offlineZshifts: Optional[np.ndarray] = None
    onlineXshifts: Optional[np.ndarray] = None
    onlineYshifts: Optional[np.ndarray] = None
    onlineZshifts: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        """Return all frame fields without copying or changing their arrays."""
        return {item.name: getattr(self, item.name) for item in fields(self)}


FRAME_INFO_KEYS = tuple(item.name for item in fields(FrameInfo))


@dataclass
class PathSummary:
    """All eager output for one imaging path, identified by a 0-based index.

    Arrays remain NumPy-backed in memory; Source/UserRoi entries may be views
    of assembled arrays. ``annotation_enabled`` distinguishes an absent ROI
    group from an enabled-but-empty group. Private empty-array templates
    retain dimensions and dtypes that cannot be inferred from empty lists;
    they never duplicate populated source or ROI tensors.
    """

    path_index: int = 0
    sources: list[Source] = field(default_factory=list)
    user_rois: list[UserRoi] = field(default_factory=list)
    visualizations: Visualizations = field(default_factory=Visualizations)
    frame_info: FrameInfo = field(default_factory=FrameInfo)
    global_f: Optional[np.ndarray] = None
    z_depths: Optional[np.ndarray] = None
    annotation_enabled: bool = False
    _empty_source_arrays: dict = field(default_factory=dict, repr=False)
    _empty_roi_arrays: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Validate path identity without copying any numerical arrays."""
        if (
            not isinstance(self.path_index, (int, np.integer))
            or self.path_index < 0
        ):
            raise ValueError(
                "path_index must be a nonnegative, 0-based integer"
            )


@dataclass
class ExperimentSummary:
    """Eager, in-memory extraction output with explicit per-path ownership.

    This is not a lazy or disk-backed model. All imaging data belongs to
    entries in ``paths``; only extraction parameters are shared.
    """

    paths: list[PathSummary] = field(default_factory=list)
    params: Optional[dict] = None

    @classmethod
    def from_h5(cls, path: Union[str, Path]) -> "ExperimentSummary":
        """Read the currently emitted band-summary schema eagerly."""
        from ..io.experiment_summary import read_summary

        return read_summary(path)

    def to_h5(self, path: Union[str, Path]) -> None:
        """Write band-schema output using the shared IO codec; return None."""
        from ..io.experiment_summary import write_summary

        write_summary(self, path)
