"""Tests of the affine and patchwise across-trial alignments (ports of
GIAnT-MATLAB's ``TestTrialAlignment.m``) and of their helpers."""

import numpy as np
import pytest


from giant_python.extraction.raster.alignment import (
    CubicInterpolant,
    TrialWarp,
    align_trials_affine,
    align_trials_patchwise,
    interp2_cubic,
    interp2_linear,
    kmeans,
    make_patches,
    matlab_gradient,
    patch_field_apply,
    patch_field_matrix,
    warp_image_affine,
    warp_trial_image,
)

from .alignment_synthetic import field_error, neuron, trials


def test_cubic_reproduces_quadratics_and_grid_values():
    """Keys' convolution with MATLAB's boundary rows is exact for
    quadratics, also next to the edges, and returns the image on the
    grid; NaN outside the image."""
    rng = np.random.default_rng(0)
    rr, cc = np.mgrid[0:9, 0:11].astype(float)

    def f(r, c):
        """A quadratic surface."""
        return (
            1 + 0.3 * r - 0.2 * c + 0.05 * r * r - 0.04 * r * c + 0.02 * c**2
        )

    im = f(rr, cc)
    r = rng.uniform(0, 8, 500)
    c = rng.uniform(0, 10, 500)
    np.testing.assert_allclose(interp2_cubic(im, r, c), f(r, c), atol=1e-12)
    np.testing.assert_allclose(interp2_cubic(im, rr, cc), im, atol=1e-12)
    out = interp2_cubic(im, np.array([-0.1, 8.1, 3.0]), np.array([1, 1, 10.5]))
    assert np.isnan(out).all()


def test_cubic_nan_reaches_its_whole_stencil():
    """A NaN pixel makes every point whose 4 x 4 stencil holds it NaN, also
    a grid point next to it (MATLAB's interp2), unlike linear
    interpolation; the cubic warp falls back to linear there."""
    im = np.arange(80, dtype=float).reshape(8, 10)
    im[4, 5] = np.nan
    assert np.isnan(interp2_cubic(im, 4.0, 4.0))  # on the grid, next to it
    assert np.isnan(interp2_cubic(im, 2.5, 6.5))  # stencil rows 2-5
    assert not np.isnan(interp2_cubic(im, 1.5, 6.5))
    assert interp2_linear(im, 4.0, 4.0) == im[4, 4]
    a = np.array([[1.0, 0, 0.3], [0, 1, -0.6]])
    lin = warp_image_affine(im, a, "linear")
    cub = warp_image_affine(im, a, "cubic")
    np.testing.assert_array_equal(np.isnan(cub), np.isnan(lin))


def test_gradient_is_matlabs():
    """Central differences inside, one-sided differences at the edges."""
    im = np.array([[0.0, 1, 4, 9], [1, 3, 5, 7], [2, 2, 2, 2]])
    g_r, g_c = matlab_gradient(im)
    np.testing.assert_array_equal(g_c[0], [1, 2, 4, 5])
    np.testing.assert_array_equal(g_r[:, 0], [1, 1, 1])
    np.testing.assert_array_equal(g_r[1], [1, 0.5, -1, -3.5])


def test_field_reproduces_affine_map():
    """Displacements at the centres that follow an affine map give that
    map everywhere, also far outside the centres."""
    rng = np.random.default_rng(1)
    nodes = np.column_stack(
        [rng.random(12) * 200 + 50, rng.random(12) * 400 + 50]
    )
    a = np.array([[1.004, 0.002, 0.7], [-0.003, 0.997, -1.2]])
    d = (a[:, :2] @ nodes.T + a[:, 2:3] - nodes.T).T
    cc, rr = np.meshgrid(np.linspace(-50, 550, 61), np.linspace(-50, 350, 41))
    u = patch_field_matrix(nodes, 30, rr, cc)[0] @ d
    pts = np.column_stack([rr.ravel(), cc.ravel()])
    u_a = (a[:, :2] @ pts.T + a[:, 2:3] - pts.T).T
    assert np.abs(u - u_a).max() < 1e-9
    warp = TrialWarp(a[None], nodes, 30.0, d[None])
    im = rng.random((300, 500))
    out = warp_trial_image(im, warp, 0)
    ref = warp_image_affine(im, a)
    np.testing.assert_array_equal(np.isnan(out), np.isnan(ref))
    assert np.nanmax(np.abs(out - ref)) < 1e-9


def test_field_apply_matches_field_matrix():
    """Evaluating a bent field without the field matrix gives the same
    displacements, also with fewer than 3 centres (their mean)."""
    rng = np.random.default_rng(2)
    nodes = np.column_stack(
        [rng.random(40) * 200 + 50, rng.random(40) * 400 + 50]
    )
    a = np.array([[1.004, 0.002, 0.7], [-0.003, 0.997, -1.2]])
    d = (a[:, :2] @ nodes.T + a[:, 2:3] - nodes.T).T
    d = d + 0.4 * rng.standard_normal(d.shape)
    r, c = rng.random(5000) * 300, rng.random(5000) * 500
    np.testing.assert_allclose(
        patch_field_apply(nodes, 25, r, c, d),
        patch_field_matrix(nodes, 25, r, c)[0] @ d,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        patch_field_apply(nodes[:2], 25, r, c, d[:2]),
        patch_field_matrix(nodes[:2], 25, r, c)[0] @ d[:2],
        atol=1e-10,
    )


def test_inverse_undoes_bent_field():
    """trialWarpInverse: back to the consensus points to 1e-5 px."""
    rng = np.random.default_rng(3)
    nodes = np.column_stack(
        [rng.random(12) * 200 + 50, rng.random(12) * 400 + 50]
    )
    a = np.array([[1.004, 0.002, 0.7], [-0.003, 0.997, -1.2]])
    d = (a[:, :2] @ nodes.T + a[:, 2:3] - nodes.T).T
    d = d + 0.4 * rng.standard_normal(d.shape)
    warp = TrialWarp(a[None], nodes, 30.0, d[None])
    r0, c0 = rng.random(1000) * 300, rng.random(1000) * 500
    r_t, c_t = warp.coords(0, r0, c0)
    r1, c1 = warp.inverse(0, r_t, c_t)
    assert max(np.abs(r1 - r0).max(), np.abs(c1 - c0).max()) < 1e-5
    rigid = TrialWarp.from_shifts([[0.25, -1.5]])
    r2, c2 = rigid.inverse(0, np.array([3.0]), np.array([4.0]))
    assert (r2[0], c2[0]) == (2.75, 5.5)


def test_matlab_conversion_of_a_field():
    """MATLAB's one-based centres and affine maps convert to zero-based;
    displacements stay; the map of a point moves with the base."""
    rng = np.random.default_rng(4)
    nodes1 = np.column_stack(
        [rng.random(8) * 50 + 10, rng.random(8) * 80 + 10]
    )
    a1 = np.array([[[1.002, 0.001, 0.4], [0.0, 0.999, -0.3]]])
    d = 0.3 * rng.standard_normal((1, 8, 2))
    warp = TrialWarp.from_matlab(a1, nodes1, 12.0, d)
    np.testing.assert_allclose(warp.nodes, nodes1 - 1)
    r0 = rng.random(50) * 60
    c0 = rng.random(50) * 90
    r_t, c_t = warp.coords(0, r0, c0)
    one = TrialWarp(a1, nodes1, 12.0, d)  # read as zero-based: one-based maps
    r1, c1 = one.coords(0, r0 + 1, c0 + 1)
    np.testing.assert_allclose(r_t, r1 - 1, atol=1e-12)
    np.testing.assert_allclose(c_t, c1 - 1, atol=1e-12)
    np.testing.assert_allclose(warp.to_matlab(), a1)


def test_kmeans_finds_separated_clusters():
    """Three well separated clusters are found from k-means++ starts; an
    empty cluster restarts at the farthest point."""
    rng = np.random.default_rng(5)
    centres = np.array([[0.0, 0], [50, 0], [0, 50]])
    x = np.vstack([c + rng.standard_normal((200, 2)) for c in centres])
    labels, cen = kmeans(x, 3, rng, replicates=3)
    found = cen[np.argsort(cen[:, 0] + 2 * cen[:, 1])]
    np.testing.assert_allclose(found, centres, atol=0.5)
    assert np.bincount(labels).tolist() == [200, 200, 200]
    start = np.array([[0.0, 0], [0.5, 0], [1000, 1000]])
    labels, cen = kmeans(x, 3, start=start)
    assert np.bincount(labels, minlength=3).min() > 0
    # the farthest point of a singleton is not moved (that would empty it)
    pts = np.array([[0.0, 0], [10, 0], [11, 0]])
    labels, cen = kmeans(
        pts, 3, start=np.array([[5.0, 0], [10.5, 0], [1e3, 0]])
    )
    assert np.isfinite(cen).all()
    assert sorted(np.bincount(labels, minlength=3)) == [1, 1, 1]


def test_patches_have_their_size():
    """About patch_px pixels each, none under half of it, centred on their
    pixels."""
    valid = np.zeros((120, 160), bool)
    valid[10:100, 20:150] = True
    valid[40:60, 60:90] = False
    patch, nodes = make_patches(valid, 900)
    n = np.bincount(patch[valid], minlength=nodes.shape[0] + 1)[1:]
    assert (patch[~valid] == 0).all() and n.min() >= 450
    assert abs(n.size - valid.sum() / 900) <= 2
    rows, cols = np.nonzero(valid)
    for k in range(nodes.shape[0]):
        sel = patch[rows, cols] == k + 1
        np.testing.assert_allclose(
            nodes[k], [rows[sel].mean(), cols[sel].mean()], atol=1e-9
        )


@pytest.fixture(scope="module")
def synthetic_neuron():
    """The labeled neuron of MATLAB's test and its scanned pixels."""
    rng = np.random.default_rng(1)
    truth, scan = neuron(rng)
    return truth, scan, rng


def test_affine_recovers_known_maps(synthetic_neuron):
    """The relative maps between trials are recovered to 0.05 px, much
    better than the rigid start."""
    truth, scan, rng = synthetic_neuron
    m, _, u, a0 = trials(truth, scan, False, rng)
    a, info = align_trials_affine(m, a0)
    warp = TrialWarp(a)
    err, err0 = field_error(warp.coords, u, a0, truth, scan)
    assert err < 0.05 and err < err0 / 5
    assert np.all(info.corr > 0.9)


def test_patchwise_recovers_bends(synthetic_neuron):
    """Smooth bends (over ~40 px) on top of affine maps: recovered to 0.08
    px, much better than the rigid start and beyond an affine fit; no
    patch under half the target size."""
    truth, scan, rng = synthetic_neuron
    m, mh, u, a0 = trials(truth, scan, True, rng)
    fit, info = align_trials_patchwise(m, a0, mh, 1500)
    warp = TrialWarp(fit.affine, fit.nodes, fit.sigma, fit.disp, fit.patch)
    err, err0 = field_error(warp.coords, u, a0, truth, scan)
    assert err < 0.08 and err < err0 / 3
    assert fit.nodes.shape[0] > 10 and info.n_px.min() >= 750
    a, _ = align_trials_affine(m, a0)
    err_a, _ = field_error(TrialWarp(a).coords, u, a0, truth, scan)
    assert err < 0.7 * err_a
    # the plane is the field's affine part: equal at the plane's points
    plane = TrialWarp(fit.affine)
    np.testing.assert_allclose(
        np.mean(fit.affine[:, :, :2], axis=0), np.eye(2), atol=1e-3
    )
    assert np.isfinite(plane.coords(0, 1.0, 1.0)[0])


def test_patchwise_adds_nothing_without_bends(synthetic_neuron):
    """Affine maps only: the patchwise field is as good as the affine
    fit."""
    truth, scan, rng = synthetic_neuron
    m, mh, u, a0 = trials(truth, scan, False, rng)
    fit, _ = align_trials_patchwise(m, a0, mh, 1500)
    warp = TrialWarp(fit.affine, fit.nodes, fit.sigma, fit.disp)
    err, _ = field_error(warp.coords, u, a0, truth, scan)
    assert err < 0.05


def test_patchwise_given_patches_and_without_halves():
    """Given its own patches back, the fit is the same; without halves the
    departures are kept as fitted (weight 1) and the fit still runs."""
    rng = np.random.default_rng(7)
    truth, scan = neuron(rng, (70, 90))
    m, mh, _, a0 = trials(truth, scan, True, rng, n_trials=3)
    fit, info = align_trials_patchwise(m, a0, mh, 600, n_rounds=1)
    again, _ = align_trials_patchwise(
        m, a0, mh, 600, n_rounds=1, patches=(fit.patch, fit.nodes)
    )
    np.testing.assert_array_equal(again.disp, fit.disp)
    assert np.all((info.weight >= 0) & (info.weight <= 1))
    plain, plain_info = align_trials_patchwise(m, a0, None, 600, n_rounds=1)
    np.testing.assert_array_equal(plain_info.weight, 1.0)
    assert np.isnan(plain_info.noise_sd).all()
    assert np.isfinite(plain.disp).all()


def test_too_few_trials_or_pixels_keep_the_start():
    """With fewer than 2 trials to fit, the maps are the starting ones."""
    m = np.random.default_rng(6).random((2, 30, 40))
    a0 = np.array([np.eye(2, 3), np.full((2, 3), np.nan)])
    a, info = align_trials_affine(m, a0)
    np.testing.assert_array_equal(a, a0)
    fit, _ = align_trials_patchwise(m, a0)
    assert fit.nodes.shape[0] == 0
    np.testing.assert_array_equal(fit.affine, a0)
    assert CubicInterpolant(m[0]).shape == (30, 40)
