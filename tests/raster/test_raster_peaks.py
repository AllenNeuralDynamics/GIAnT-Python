"""Tests of the getActImPeaks port (raster SiLo seeds)."""

import numpy as np

from giant_python.extraction.raster.peaks import (
    _eval_budget,
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
    # the value-only path (no derivatives) is the Jacobian path's value
    a, (vy, _, _), (vx, _, _) = model.parts(theta)
    np.testing.assert_array_equal(model.value(theta), (vy * vx) @ a)


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


def test_fit_falls_back_when_svd_fails(monkeypatch):
    """A LAPACK SVD failure in the default solver retries with lsmr."""
    import giant_python.extraction.raster.peaks as peaks

    calls = []
    real = peaks.least_squares

    def flaky(*args, **kwargs):
        """least_squares whose default (SVD) solver always fails."""
        calls.append(kwargs.get("tr_solver"))
        if kwargs.get("tr_solver") is None:
            raise np.linalg.LinAlgError("SVD did not converge")
        return real(*args, **kwargs)

    monkeypatch.setattr(peaks, "least_squares", flaky)
    yy, xx = np.mgrid[1:16, 1:16]
    model = _IntegratedGaussians(yy.ravel() + 0.0, xx.ravel() + 0.0)
    data = model.value(np.array([3.0, 7.4, 8.2, 1.2]))
    theta = peaks._fit(
        np.array([[2.0, 7.0, 8.0, 1.0]]),
        yy.ravel() + 0.0,
        xx.ravel() + 0.0,
        data,
        np.array([[0.0, 5.5, 6.5, 0.2]]),
        np.array([[np.inf, 8.5, 9.5, 5.0]]),
    )
    assert calls == [None, "lsmr"]
    np.testing.assert_allclose(theta[0], [3.0, 7.4, 8.2, 1.2], atol=1e-3)


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


def test_no_fitted_centre_inside_the_exclusion_mask():
    """A source whose centre is just inside the mask seeds on its unmasked
    flank, and its fit moves back into the mask: it is dropped. A source
    clear of the mask is kept."""
    rng = np.random.default_rng(2)
    yy, xx = np.mgrid[:30, :40]
    img = 0.05 * rng.standard_normal((30, 40))
    for r, c in ((15.0, 15.2), (15.0, 30.0)):
        img += 4 * np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2 * 1.1**2))
    mask = np.zeros((30, 40), bool)
    mask[:, 15:20] = True  # the first source's centre and right flank
    pk = get_act_im_peaks(img, 8, mask)
    assert pk.shape[0] == 1
    np.testing.assert_allclose(pk[0, 1:3], [15.0, 30.0], atol=0.15)
    unmasked = get_act_im_peaks(img, 8)
    assert unmasked.shape[0] == 2


def test_eval_budget_matches_matlab_iterations():
    """MATLAB's 5000 evaluations with finite differences allow
    (5000 - 1) // (n + 1) iterations; the budget here allows as many steps
    (one evaluation each, plus the start), down to one step at n = 2500."""
    for n in (4, 40, 400, 2500, 4998):
        assert _eval_budget(n) - 1 == (5000 - 1) // (n + 1)
    assert _eval_budget(2500) == 2  # one step (was 1: none)
    assert _eval_budget(5000) == 1  # MATLAB cannot take a step either
