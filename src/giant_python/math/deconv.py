"""Temporal deconvolution of fluorescence traces into events (SILo)."""

import numpy as np


def deconvolve_trace(
    trace: np.ndarray,
    tau_s: float,
    frame_rate_hz: float,
) -> np.ndarray:
    """Deconvolve a fluorescence trace into an inferred event train.

    Removes the indicator decay kinetics (single-exponential with time
    constant *tau_s*) to recover an event/spike train. Corresponds to the
    deconvolution step in GIAnT-MATLAB source extraction.

    Parameters
    ----------
    trace : ndarray of shape (n_frames,)
        Denoised dF/F trace.
    tau_s : float
        Indicator decay time constant, in seconds.
    frame_rate_hz : float
        Acquisition/analysis frame rate, in Hz.

    Returns
    -------
    ndarray of shape (n_frames,)
        Inferred non-negative event train.
    """
    raise NotImplementedError
