"""Poisson variance modeling and activity-image construction (SILo)."""

import numpy as np


def activity_image(
    movie: np.ndarray,
    photon_scale: float,
) -> np.ndarray:
    """Build a heteroscedastic-noise-normalized activity image from a movie.

    Uses a Poisson-based variance model (variance proportional to mean photon
    count, via *photon_scale*) to weight temporal fluctuations, yielding an
    image that highlights active sources. Corresponds to the activity-image
    construction in GIAnT-MATLAB source extraction.

    Parameters
    ----------
    movie : ndarray of shape (n_frames, H, W)
        Motion-corrected movie.
    photon_scale : float
        Photons per digital unit, used to scale the variance model.

    Returns
    -------
    ndarray of shape (H, W)
        Activity image.
    """
    raise NotImplementedError
