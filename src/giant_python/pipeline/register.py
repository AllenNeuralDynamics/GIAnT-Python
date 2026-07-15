"""Stage 2 - motion correction (ports of *Registration.m).

Exposes the porting-faithful registration functions plus a
:class:`MotionCorrector` ABC with one backend per MATLAB registration variant
(MultiRoi / Strip / Band). Band-mode registration is exposed as
:func:`integration_registration` / :class:`IntegrationRegistration` (the
Python API name) and corresponds to ``BandRegistration.m``, writing
``bandRegLookupTable.h5``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from ..models.params import AlignParams
from .base import Stage


def multi_roi_registration(
    full_path_to_trial_table: Optional[Union[str, Path]] = None,
    params_in: Optional[dict] = None,
) -> None:
    """Perform motion correction for SLAP2 multi-ROI recordings.

    Registers each trial to a common template computed from initial frames,
    producing downsampled registered TIFF movies and alignment metadata for
    each DMD. Corresponds to MultiRoiRegistration.m in GIAnT-MATLAB.

    Parameters
    ----------
    full_path_to_trial_table : str or Path, optional
        Full path to the trial table file.
    params_in : dict, optional
        Parameter overrides. Keys and default values are defined in
        :func:`giant_python.models.params.set_params` under
        ``"MultiRoiRegistration"``.
    """
    raise NotImplementedError


def strip_registration(
    dr: Optional[Union[str, Path]] = None,
    fns: Optional[Union[str, list]] = None,
    params_in: Optional[dict] = None,
) -> None:
    """Perform strip-based motion correction for Bergamo recordings.

    Registers each trial using DFT-based rigid registration, producing
    downsampled and full-resolution registered TIFF or HDF5 movies together
    with per-trial alignment metadata. Corresponds to StripRegistration.m in
    GIAnT-MATLAB.

    Parameters
    ----------
    dr : str or Path, optional
        Directory containing TIFF or HDF5 files, or the path to a trial
        table file.
    fns : str or list, optional
        Filename(s) to register, or path to a trial table file.
    params_in : dict, optional
        Parameter overrides. Keys and default values are defined in
        :func:`giant_python.models.params.set_params` under
        ``"StripRegistration"``.
    """
    raise NotImplementedError


def integration_registration(
    full_path_to_trial_table: Optional[Union[str, Path]] = None,
    params_in: Optional[dict] = None,
) -> None:
    """Perform band / integration-mode motion correction for SLAP2 recordings.

    Builds the cached ``motion_correction/bandRegLookupTable.h5`` lookup table
    (on first run) and writes per-trial ``*_ALIGNMENTDATA.h5`` files including
    BandRegistration-only fields ``brightnessDS`` and ``logLikelihoodDS``.
    Corresponds to ``BandRegistration.m`` in GIAnT-MATLAB.

    Parameters
    ----------
    full_path_to_trial_table : str or Path, optional
        Full path to the trial table file.
    params_in : dict, optional
        Parameter overrides. Keys and default values are defined in
        :func:`giant_python.models.params.set_params` under
        ``"BandRegistration"``.
    """
    raise NotImplementedError


class MotionCorrector(Stage):
    """Stage 2 base class: register all trials in a trial table.

    Use :meth:`for_microscope` to obtain the appropriate backend. Concrete
    subclasses wrap the corresponding registration function.

    Parameters
    ----------
    params : AlignParams, optional
        Alignment parameters; defaults to :class:`AlignParams` defaults.
    """

    def __init__(self, params: Optional[AlignParams] = None) -> None:
        self.params = params or AlignParams()

    @staticmethod
    def for_microscope(
        microscope: str,
        params: Optional[AlignParams] = None,
    ) -> "MotionCorrector":
        """Return the registration backend for the given microscope.

        Parameters
        ----------
        microscope : str
            ``"slap2"`` (-> :class:`MultiRoiRegistration`) or ``"bergamo"``
            (-> :class:`StripRegistration`).
        params : AlignParams, optional
            Alignment parameters.

        Returns
        -------
        MotionCorrector
            The appropriate backend instance.
        """
        raise NotImplementedError

    def run(self, trial_table: "object") -> "object":
        """Register every trial and update the trial table in place.

        Parameters
        ----------
        trial_table : TrialTable
            The organized trial table.

        Returns
        -------
        TrialTable
            The trial table with its ``motion_correction`` group populated.
        """
        raise NotImplementedError


class MultiRoiRegistration(MotionCorrector):
    """SLAP2 multi-ROI backend (wraps ``multi_roi_registration``)."""

    def run(self, trial_table: "object") -> "object":
        """Register SLAP2 multi-ROI trials. See :meth:`MotionCorrector.run`."""
        raise NotImplementedError


class StripRegistration(MotionCorrector):
    """Bergamo strip backend (wraps ``strip_registration``)."""

    def run(self, trial_table: "object") -> "object":
        """Register Bergamo strip trials. See :meth:`MotionCorrector.run`."""
        raise NotImplementedError


class IntegrationRegistration(MotionCorrector):
    """SLAP2 BandRegistration backend (wraps ``integration_registration``).

    Corresponds to ``BandRegistration.m``; writes ``bandRegLookupTable.h5``.
    """

    def run(self, trial_table: "object") -> "object":
        """Register SLAP2 band/integration trials.

        See ``MotionCorrector.run``.
        """
        raise NotImplementedError
