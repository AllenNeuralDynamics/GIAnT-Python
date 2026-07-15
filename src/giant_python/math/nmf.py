"""Non-negative matrix factorization for source decomposition (SILo)."""

from typing import Tuple

import numpy as np


def nmf_decompose(
    activity: np.ndarray,
    n_sources: int,
    lambda_: float = 0.1,
    phi: float = 0.0,
    n_iter: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose pixel activity into spatial profiles and temporal traces.

    Sparse, spatially regularized non-negative matrix factorization used by
    the SILo source-extraction stage. Corresponds to the NMF refinement step
    in GIAnT-MATLAB source extraction.

    Parameters
    ----------
    activity : ndarray of shape (n_pixels, n_frames)
        Per-pixel activity matrix.
    n_sources : int
        Number of sources to extract.
    lambda_ : float
        Sparsity weight.
    phi : float
        Spatial-coherence weight.
    n_iter : int
        Number of update iterations.

    Returns
    -------
    spatial : ndarray of shape (n_pixels, n_sources)
        Spatial profiles.
    temporal : ndarray of shape (n_sources, n_frames)
        Temporal traces.
    """
    raise NotImplementedError
