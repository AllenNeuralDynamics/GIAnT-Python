"""Tests of the getActImPeaks port (raster SiLo seeds)."""

import numpy as np

from giant_python.extraction.raster.peaks import (
    _IntegratedGaussians,
    get_act_im_peaks,
    ordfilt3x3,
)


def test_ordfilt_matches_matlab():
    """MATLAB ordfilt2(A, 9 or 8, ones(3)): zero padding, NaN windows."""
    a = np.array([[np.nan, 1, -2], [2, 3, -1], [-5, np.nan, 4]])
    nan = np.nan
    np.testing.assert_array_equal(
        ordfilt3x3(a, 9), [[nan, nan, 3], [nan, nan, nan], [nan, nan, nan]]
    )
    np.testing.assert_array_equal(
        ordfilt3x3(a, 8), [[nan, nan, 1], [nan, nan, nan], [nan, nan, nan]]
    )
    np.testing.assert_array_equal(
        ordfilt3x3(np.array([[-1.0, -2], [-3, -4]]), 9), np.zeros((2, 2))
    )


def test_jacobian_matches_finite_differences():
    """The analytical Jacobian of the integrated Gaussians is right."""
    yy, xx = np.mgrid[1:12, 1:14]
    model = _IntegratedGaussians(yy.ravel() + 0.0, xx.ravel() + 0.0)
    theta = np.array([3.0, 1.5, 5.2, 7.7, 4.1, 9.3, 1.1, 0.8])
    jac = model.jacobian(theta)
    eps = 1e-6
    num = np.column_stack(
        [
            (model.value(theta + eps * e) - model.value(theta - eps * e))
            / (2 * eps)
            for e in np.eye(theta.size)
        ]
    )
    np.testing.assert_allclose(jac, num, rtol=1e-5, atol=1e-8)


def test_finds_separated_gaussians():
    """Two blobs on noise are found at their centres (0-based)."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:40, :50]
    img = 0.05 * rng.standard_normal((40, 50))
    for r, c in ((10.3, 12.6), (28.0, 35.4)):
        img += 4 * np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2 * 1.1**2))
    img[:, :2] = np.nan
    pk = get_act_im_peaks(img, 8)
    assert pk.shape[0] == 2
    order = np.argsort(pk[:, 1])
    np.testing.assert_allclose(
        pk[order, 1:3], [[10.3, 12.6], [28.0, 35.4]], atol=0.15
    )


def test_exclusion_mask_and_empty():
    """No seeds inside the exclusion mask; none in pure noise."""
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[:30, :30]
    img = 0.05 * rng.standard_normal((30, 30))
    img += 4 * np.exp(-((yy - 15) ** 2 + (xx - 15) ** 2) / 2)
    mask = np.zeros((30, 30), bool)
    mask[12:19, 12:19] = True
    assert get_act_im_peaks(img, 8, mask).shape[0] == 0
    assert get_act_im_peaks(rng.standard_normal((30, 30)), 8).shape == (0, 4)
