"""Alignment data model (mirrors the ``aData`` struct / *_ALIGNMENTDATA.h5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np


@dataclass
class AlignmentData:
    """Per-trial motion-correction output written to ``*_ALIGNMENTDATA.h5``.

    Mirrors the ``aData`` struct in GIAnT-MATLAB. Field presence depends on
    which registration backend wrote the file (StripRegistration,
    MultiRoiRegistration, or BandRegistration); see the GIAnT-python README
    Pipeline Outputs section.

    Attributes
    ----------
    num_channels : int or None
        Number of channels in the recording (``numChannels``).
    frametime : float or None
        Seconds per downsampled frame.
    align_hz : float or None
        Frame rate (Hz) at which alignment was performed (``alignHz``).
    motion_ds_c, motion_ds_r : ndarray or None
        Per-downsampled-frame column / row shifts (``motionDSc`` /
        ``motionDSr``).
    motion_ds_z : ndarray or None
        Per-downsampled-frame Z shift (``motionDSz``); always written by
        BandRegistration; by MultiRoiRegistration only when
        ``refStackTemplate`` is enabled; never by StripRegistration.
    mean_im : ndarray or None
        Per-channel mean registered image, shaped ``(channels, rows, cols)``
        (``meanIM``); not written by BandRegistration.
    rec_neg_err : ndarray or None
        Reconstruction error per frame (``recNegErr``); not written by
        BandRegistration.
    motion_c, motion_r : ndarray or None
        Column / row shifts upsampled to raw frame rate (``motionC`` /
        ``motionR``); StripRegistration / Bergamo only.
    motion_z : ndarray or None
        Reserved (``motionZ``); not written by any current backend.
    brightness_ds : ndarray or None
        Per-channel brightness/scaling at the selected motion shift, shaped
        ``(nDSframes, channels)`` (``brightnessDS``); BandRegistration only.
    log_likelihood_ds : ndarray or None
        Peak log-likelihood of the motion match per downsampled frame
        (``logLikelihoodDS``); BandRegistration only.
    ds_frames : ndarray or None
        1-indexed line indices of the downsampled frames (``DSframes``);
        SLAP2 only (MultiRoiRegistration and BandRegistration).
    registration_failed : bool
        Whether registration failed for this trial (``registrationFailed``);
        SLAP2 only.
    slap2 : dict or None
        SLAP2-only sub-struct. Always may include
        ``onlineMotion{X,Y,Z}shift``; MultiRoiRegistration also writes
        ``varFacDS``, ``Z_depths``, ``cropRow``/``cropCol``,
        ``viewC``/``viewR``, ``trimRows``/``trimCols``.
    """

    num_channels: Optional[int] = None
    frametime: Optional[float] = None
    align_hz: Optional[float] = None
    motion_ds_c: Optional[np.ndarray] = None
    motion_ds_r: Optional[np.ndarray] = None
    motion_ds_z: Optional[np.ndarray] = None
    mean_im: Optional[np.ndarray] = None
    rec_neg_err: Optional[np.ndarray] = None
    motion_c: Optional[np.ndarray] = None
    motion_r: Optional[np.ndarray] = None
    motion_z: Optional[np.ndarray] = None
    brightness_ds: Optional[np.ndarray] = None
    log_likelihood_ds: Optional[np.ndarray] = None
    ds_frames: Optional[np.ndarray] = None
    registration_failed: bool = False
    slap2: Optional[dict] = field(default=None)

    @classmethod
    def from_h5(cls, path: Union[str, Path]) -> "AlignmentData":
        """Load alignment data from a ``*_ALIGNMENTDATA.h5`` file.

        Parameters
        ----------
        path : str or Path
            Path to the alignment data file.

        Returns
        -------
        AlignmentData
            The deserialized alignment data.
        """
        raise NotImplementedError

    def to_h5(self, path: Union[str, Path]) -> None:
        """Write alignment data to a ``*_ALIGNMENTDATA.h5`` file.

        Parameters
        ----------
        path : str or Path
            Destination path for the alignment data file.
        """
        raise NotImplementedError
