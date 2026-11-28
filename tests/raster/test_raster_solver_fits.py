"""Tests of the iterative fits of the subproblem solver: the loss and its
derivatives, the kernel, the bounded Newton solver, and the fits against
GIAnT-MATLAB's from the same inputs."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from scipy import optimize

from giant_python.extraction.raster import solver_baseline as sb
from giant_python.extraction.raster import solver_fits as sf
from giant_python.extraction.raster.solver_setup import (
    SolverParams,
    conv_operator,
    temporal_kernel,
)

SUBPROBLEM = Path(__file__).parent / "data" / "matlab_solver_subproblem.h5"
PARAMS = SolverParams()
DT = torch.float64


def _t(a):
    """A float64 tensor with a leading batch dimension of one."""
    return torch.as_tensor(np.ascontiguousarray(a), dtype=DT)[None]


def _conv_same(x, k):
    """MATLAB convn(x, k, 'same') along the last axis."""
    c = len(k) // 2
    flat = x.reshape(-1, x.shape[-1])
    out = np.stack([np.convolve(r, k)[c : c + x.shape[-1]] for r in flat])
    return out.reshape(x.shape)


def _small_problem(seed=0, n_px=14, n_src=2, n_t=60, batch=1, tau=3.0):
    """Random data from overlapping footprints and sparse events."""
    rng = np.random.default_rng(seed)
    k = temporal_kernel(tau)
    h = rng.random((batch, n_px, n_src)) ** 3
    h /= h.sum(1, keepdims=True)
    s = (rng.random((batch, n_src, n_t)) < 0.1) * rng.uniform(
        20, 80, (batch, n_src, n_t)
    )
    b = 1.0 + rng.random((batch, n_px, 1)) + 0.2 * rng.random((1, 1, n_t))
    finv = 0.5 + rng.random((batch, n_px, n_t))
    mean = b + h @ _conv_same(s, k)
    z = finv * rng.poisson(mean / finv)
    tens = [torch.as_tensor(a, dtype=DT) for a in (z, h, b, 1 / finv, s)]
    return tens + [torch.as_tensor(k, dtype=DT)]


@pytest.mark.parametrize("fixed", [False, True])
def test_loss_derivatives(fixed):
    """First and second derivative of the loss in the model, by autograd."""
    rng = np.random.default_rng(0)
    z = torch.as_tensor(rng.random((2, 3, 5)) * 4)
    w = torch.as_tensor(0.5 + rng.random((2, 3, 5)))
    s0 = torch.as_tensor(1 + rng.random((2, 3, 5))) if fixed else None
    t = torch.as_tensor(rng.random((2, 3, 5)) * 3, dtype=DT).requires_grad_()
    f, g, c = sf.variance_loss(z, t, w, s0)
    var = s0 if fixed else t + 1
    expected = (w * (z - t) ** 2 / var).sum((-2, -1))
    assert torch.allclose(f, expected, rtol=1e-13)
    (g_auto,) = torch.autograd.grad(expected.sum(), t, create_graph=True)
    assert torch.allclose(g, g_auto, rtol=1e-12)
    (c_auto,) = torch.autograd.grad(g_auto.sum(), t)
    assert torch.allclose(c, c_auto, rtol=1e-12)
    assert sf.variance_loss(z, t, w, s0, 0)[1:] == (None, None)
    assert sf.variance_loss(z, t, w, s0, 1)[2] is None


@pytest.mark.parametrize("tau", [0.7, 4.8, 30.0])
@pytest.mark.parametrize("n_t", [5, 40, 400])
def test_time_kernel_matches_convolution(tau, n_t):
    """The exponential kernels by shifted additions equal a 'same'
    convolution, for trials shorter and longer than the kernel."""
    rng = np.random.default_rng(1)
    k = temporal_kernel(tau)
    kern = sf.TimeKernel(torch.as_tensor(k))
    assert kern.decay == pytest.approx(np.exp(-1 / tau), rel=1e-9)
    x = rng.standard_normal((2, 3, n_t))
    y = kern.conv(torch.as_tensor(x)).numpy()
    np.testing.assert_allclose(y, _conv_same(x, k), atol=1e-13)
    ya = kern.conv_adj(torch.as_tensor(x)).numpy()
    np.testing.assert_allclose(ya, _conv_same(x, k[::-1]), atol=1e-13)


def test_time_kernel_general():
    """Kernels that are not causal exponentials go through conv1d."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal((1, 2, 30))
    for k in (
        np.concatenate([np.zeros(4), [0.5, 0.3, 0.1, 0.07, 0.03]]),
        np.array([0.2, 0.5, 0.3]),
        np.array([0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.25]),
    ):
        kern = sf.TimeKernel(torch.as_tensor(k))
        assert kern.decay is None
        y = kern.conv(torch.as_tensor(x)).numpy()
        np.testing.assert_allclose(y, _conv_same(x, k), atol=1e-14)
        ya = kern.conv_adj(torch.as_tensor(x)).numpy()
        np.testing.assert_allclose(ya, _conv_same(x, k[::-1]), atol=1e-14)
    with pytest.raises(ValueError, match="odd"):
        sf.TimeKernel(torch.ones(4))


def _check_derivatives(problem, x):
    """Gradient, Hessian products and diagonal against autograd."""
    x = x.clone().requires_grad_()
    f = problem.value(x)
    (g_auto,) = torch.autograd.grad(f.sum(), x, create_graph=True)
    f2, g = problem.value_grad(x.detach())
    assert torch.allclose(f2, f.detach(), rtol=1e-13)
    assert torch.allclose(g, g_auto.detach(), rtol=1e-9, atol=1e-10)
    rng = np.random.default_rng(5)
    v = torch.as_tensor(rng.standard_normal(x.shape))
    (hv_auto,) = torch.autograd.grad((g_auto * v).sum(), x, retain_graph=True)
    hv = problem.hess_mult(v)
    assert torch.allclose(hv, hv_auto, rtol=1e-9, atol=1e-9)
    diag = torch.zeros_like(x)
    flat = diag.view(-1)
    for i in range(flat.numel()):
        e = torch.zeros(flat.numel(), dtype=DT)
        e[i] = 1
        flat[i] = problem.hess_mult(e.view(x.shape)).view(-1)[i]
    assert torch.allclose(problem.hess_diag(), diag, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("fixed", [False, True])
def test_activity_problem_derivatives(fixed):
    """The activity fit's gradient and curvature, with either variance."""
    z, h, b, w, s, k = _small_problem(n_px=6, n_t=14)
    s0 = (h @ sf.TimeKernel(k).conv(s) + b + 1) if fixed else None
    problem = sf.ActivityProblem(z, h, b, k, w, 0.9, s0)
    _check_derivatives(problem, s * 0.5 + 1)


def test_footprint_problem_derivatives():
    """The footprint fit's gradient and curvature."""
    z, h, b, w, s, k = _small_problem(n_px=6, n_t=14)
    rng = np.random.default_rng(6)
    g = torch.as_tensor(rng.random((1, 6, 6)) * (rng.random((1, 6, 6)) < 0.6))
    x = sf.TimeKernel(k).conv(s)
    support = torch.ones((1, 6, 2), dtype=torch.bool)
    problem = sf.FootprintProblem(z, x, b, g, w, support)
    _check_derivatives(problem, h.clone())


def test_preconditioner_inverts_one_source():
    """For one source the preconditioner is the inverse of the Hessian on
    the free events, up to the kernel's truncation."""
    z, h, b, w, s, k = _small_problem(n_px=8, n_src=1, n_t=200)
    problem = sf.ActivityProblem(z, h, b, k, w, 0.9)
    problem.value_grad(s)
    rng = np.random.default_rng(7)
    free = torch.as_tensor(rng.random(s.shape) < 0.4)
    free[..., 50:80] = True  # a run of consecutive free events
    fm = free.to(DT)
    v = torch.as_tensor(rng.standard_normal(s.shape)) * fm
    hv = fm * problem.hess_mult(v)
    back = problem.preconditioner(free, problem.hess_diag())(hv)
    assert float((back - v).norm() / v.norm()) < 0.02
    # the Hessian's diagonal alone is far from it
    jacobi = fm / problem.hess_diag() * hv
    assert float((jacobi - v).norm() / v.norm()) > 0.3


def test_preconditioner_falls_back_to_diagonal():
    """Kernels that are not exponential get the Hessian's diagonal."""
    z, h, b, w, s, _ = _small_problem(n_px=6, n_t=20)
    k = torch.as_tensor([0.0, 0.0, 0.6, 0.1, 0.3], dtype=DT)
    problem = sf.ActivityProblem(z, h, b, k, w, 0.9)
    problem.value_grad(s)
    free = torch.ones_like(s, dtype=torch.bool)
    r = torch.ones_like(s)
    diag = problem.hess_diag()
    assert torch.equal(problem.preconditioner(free, diag)(r), r / diag)


def test_activity_fit_reaches_scipy_optimum():
    """The fit ends at the optimum that L-BFGS-B approaches, with many
    events on their bound."""
    z, h, b, w, s_true, k = _small_problem(seed=3, n_px=20, n_t=120)
    start = torch.full_like(s_true, 5.0)
    s, info = sf.fit_activity(z, h, b, k, w, 0.9, start, history=True)
    assert bool(info.converged[0]) and int(info.iterations[0]) < 30
    assert float(info.optimality[0]) < 1e-5
    assert float((s == 0).double().mean()) > 0.4 and float(s.min()) == 0
    losses = [float(f[0]) for f in info.history]
    assert all(b2 <= a2 for a2, b2 in zip(losses, losses[1:]))

    problem = sf.ActivityProblem(z, h, b, k, w, 0.9)

    def fun(x):
        """Loss and gradient for scipy."""
        f, g = problem.value_grad(torch.as_tensor(x).view(s.shape))
        return float(f[0]), g.numpy().ravel()

    ref = optimize.minimize(
        fun,
        start.numpy().ravel(),
        jac=True,
        method="L-BFGS-B",
        bounds=[(0, None)] * s.numel(),
        options=dict(maxiter=20000, maxfun=50000, ftol=1e-15, gtol=1e-9),
    )
    assert float(info.loss[0]) <= ref.fun + 1e-9 * abs(ref.fun)
    np.testing.assert_allclose(s.numpy().ravel(), ref.x, atol=2e-2)
    # the sparse events are recovered
    found = _conv_same(s.numpy(), k.numpy())
    truth = _conv_same(s_true.numpy(), k.numpy())
    assert np.corrcoef(found.ravel(), truth.ravel())[0, 1] > 0.7


def test_fit_does_not_depend_on_its_start():
    """A convex fit has one optimum: two starts end at the same events."""
    z, h, b, w, s_true, k = _small_problem(seed=4, n_px=16, n_t=80)
    a, _ = sf.fit_activity(z, h, b, k, w, 0.9, torch.zeros_like(s_true))
    c, _ = sf.fit_activity(z, h, b, k, w, 0.9, s_true * 3 + 10)
    assert torch.allclose(a, c, atol=1e-4)


def test_batched_fit_equals_separate_fits():
    """Members of a batch, padded with weightless pixels and a source held
    at zero, get the fits they get alone."""
    z, h, b, w, s_true, k = _small_problem(seed=5, n_px=12, n_t=70, batch=2)
    # member 1: 9 pixels and one source; the rest is padding
    w[1, 9:] = 0
    z[1, 9:] = 0
    b[1, 9:] = 1
    h[1, :, 1] = 0
    h[1, 9:] = 0
    free = torch.ones_like(s_true, dtype=torch.bool)
    free[1, 1] = False
    start = torch.full_like(s_true, 2.0)
    s, info = sf.fit_activity(z, h, b, k, w, 0.9, start, free=free)
    assert bool(info.converged.all())
    assert torch.all(s[1, 1] == 0)
    one, i0 = sf.fit_activity(
        z[:1], h[:1], b[:1], k, w[:1], 0.9, start[:1], free=free[:1]
    )
    two, i1 = sf.fit_activity(
        z[1:, :9], h[1:, :9, :1], b[1:, :9], k, w[1:, :9], 0.9, start[1:, :1]
    )
    assert torch.allclose(s[0], one[0], atol=1e-5)
    assert torch.allclose(s[1, :1], two[0], atol=1e-5)
    assert float(info.loss[0]) == pytest.approx(float(i0.loss[0]), rel=1e-10)
    assert float(info.loss[1]) == pytest.approx(float(i1.loss[0]), rel=1e-10)


def test_iteration_cap_and_free_entries():
    """A capped fit reports that it did not converge and still lowers the
    loss; entries that are not free stay at zero."""
    z, h, b, w, s_true, k = _small_problem(seed=6, n_px=16, n_t=80)
    start = torch.full_like(s_true, 5.0)
    problem = sf.ActivityProblem(z, h, b, k, w, 0.9)
    s1, info = sf.fit_activity(z, h, b, k, w, 0.9, start, max_iter=1)
    assert not bool(info.converged[0]) and int(info.iterations[0]) == 1
    assert float(info.loss[0]) < float(problem.value(start)[0])
    full, done = sf.fit_activity(z, h, b, k, w, 0.9, start)
    assert float(done.loss[0]) < float(info.loss[0])
    free = s_true > 0
    s2, info2 = sf.fit_activity(z, h, b, k, w, 0.0, start, free=free)
    assert bool(info2.converged[0])
    assert torch.all(s2[~free] == 0) and float(s2[free].min()) >= 0


def test_fixed_variance_fit_is_weighted_least_squares():
    """With the variance held, the fit of the free events is the bounded
    weighted least-squares solution."""
    z, h, b, w, s_true, k = _small_problem(seed=7, n_px=10, n_src=1, n_t=40)
    kern = sf.TimeKernel(k)
    s0 = h @ kern.conv(s_true) + b + 1
    free = torch.zeros_like(s_true, dtype=torch.bool)
    free[..., ::4] = True
    s, info = sf.fit_activity(
        z, h, b, k, w, 0.0, s_true, s0=s0, free=free, ftol=1e-15
    )
    assert bool(info.converged[0])
    idx = np.nonzero(free.numpy().ravel())[0]
    cols = []
    for i in idx:
        e = torch.zeros_like(s_true)
        e.view(-1)[i] = 1
        cols.append((h @ kern.conv(e)).numpy().ravel())
    sw = np.sqrt((w / s0).numpy().ravel())
    design = np.stack(cols, 1) * sw[:, None]
    target = (z - b).numpy().ravel() * sw
    ref = optimize.lsq_linear(design, target, bounds=(0, np.inf)).x
    np.testing.assert_allclose(s.numpy().ravel()[idx], ref, atol=1e-5)


def test_footprint_fit_stays_on_its_support():
    """Footprints are fitted within their support, not below zero, and
    reach a lower loss than the start."""
    z, h, b, w, s, k = _small_problem(seed=8, n_px=12, n_t=90)
    x = sf.TimeKernel(k).conv(s)
    g = torch.eye(12, dtype=DT)[None] * 0.6 + 0.4 / 12
    valid = torch.zeros((1, 12, 2), dtype=torch.bool)
    valid[0, :6, 0] = True
    valid[0, 5:, 1] = True
    start = valid.to(DT) / 6
    hs, info = sf.fit_footprints(z, x, b, g, w, start, valid)
    assert bool(info.converged[0])
    assert torch.all(hs[~valid] == 0) and float(hs.min()) >= 0
    problem = sf.FootprintProblem(z, x, b, g, w, valid)
    assert float(info.loss[0]) < float(problem.value(start)[0])
    assert torch.allclose(problem.footprints(hs), g @ hs)


@pytest.fixture(scope="module")
def matlab():
    """MATLAB's fits of a real two-source subproblem, and its inputs."""
    with h5py.File(SUBPROBLEM, "r") as f:
        d = {
            g: {k: v[()] for k, v in f[g].items()}
            for g in ("in", "init", "it1", "it2", "debias", "ch2", "out")
        }
    y = d["in"]["y"].astype(np.float64)
    finv = d["in"]["finv"].astype(np.float64)
    sel = d["in"]["sel"].astype(bool)
    b0 = sb.initial_baseline(y[0], PARAMS)
    b1 = sb.fit_baseline(
        b0, y[0], d["it1"]["X_thr"], d["init"]["H"], finv, sel, PARAMS
    )
    d.update(
        z=_t(y[0]),
        z2=_t(y[1]),
        w=_t(1 / finv),
        b0=_t(b0),
        b1=_t(b1),
        b2=_t(sb.initial_baseline(y[1], PARAMS)),
        g=_t(conv_operator(sel, PARAMS.hfilter).toarray()),
        k=torch.as_tensor(PARAMS.k),
        k2=torch.as_tensor(PARAMS.k2),
    )
    return d


def test_loss_is_matlabs(matlab):
    """MATLAB's reported losses are this loss at MATLAB's solutions."""
    m = matlab
    lam = PARAMS.lam
    p1 = sf.ActivityProblem(
        m["z"], _t(m["init"]["H"]), m["b0"], m["k"], m["w"], lam
    )
    assert float(p1.value(_t(m["it1"]["S_out"]))[0]) == pytest.approx(
        m["it1"]["loss_S"], rel=1e-10
    )
    valid = torch.as_tensor(m["init"]["Hvalid"].astype(bool))[None]
    ph = sf.FootprintProblem(
        m["z"], _t(m["it1"]["X_thr"]), m["b1"], m["g"], m["w"], valid
    )
    # the baseline differs from MATLAB's by quadprog's tolerance
    assert float(ph.value(_t(m["it1"]["Hs_out"]))[0]) == pytest.approx(
        m["it1"]["loss_H"], rel=1e-6
    )
    p2 = sf.ActivityProblem(
        m["z"], _t(m["it1"]["H"]), m["b1"], m["k"], m["w"], lam
    )
    assert float(p2.value(_t(m["it2"]["S_out"]))[0]) == pytest.approx(
        m["it2"]["loss_S"], rel=1e-6
    )
    pc = sf.ActivityProblem(
        m["z2"], _t(m["out"]["H"]), m["b2"], m["k2"], m["w"], lam
    )
    assert float(pc.value(_t(m["ch2"]["S_l1"]))[0]) == pytest.approx(
        m["ch2"]["loss"], rel=1e-10
    )


def test_fits_reach_matlabs_loss_or_lower(matlab):
    """Each fit, started from MATLAB's inputs, ends at a loss no higher
    than MATLAB's (whose fits stop at iteration caps or tolerances)."""
    m = matlab
    lam = PARAMS.lam
    s, info = sf.fit_activity(
        m["z"],
        _t(m["init"]["H"]),
        m["b0"],
        m["k"],
        m["w"],
        lam,
        _t(m["init"]["S"]),
    )
    assert bool(info.converged[0])
    assert float(info.loss[0]) < m["it1"]["loss_S"]

    valid = torch.as_tensor(m["init"]["Hvalid"].astype(bool))[None]
    x_thr = _t(m["it1"]["X_thr"])
    hs, info = sf.fit_footprints(
        m["z"], x_thr, m["b1"], m["g"], m["w"], _t(m["init"]["Hs"]), valid
    )
    assert bool(info.converged[0])
    assert float(info.loss[0]) < m["it1"]["loss_H"] * (1 - 1e-5)

    h1 = _t(m["it1"]["H"])
    s, info = sf.fit_activity(
        m["z"], h1, m["b1"], m["k"], m["w"], lam, _t(m["it1"]["S"])
    )
    assert bool(info.converged[0])
    assert float(info.loss[0]) < m["it2"]["loss_S"] * (1 + 1e-6)

    # debias: MATLAB's refit of its own events, variance held at the start
    free = torch.as_tensor(m["debias"]["free"].astype(bool))[None]
    start = _t(m["debias"]["S_in"])
    s0 = h1 @ sf.TimeKernel(m["k"]).conv(start) + m["b1"] + 1
    problem = sf.ActivityProblem(m["z"], h1, m["b1"], m["k"], m["w"], 0.0, s0)
    f_matlab = float(problem.value(_t(m["debias"]["S_out"]))[0])
    s, info = sf.fit_activity(
        m["z"], h1, m["b1"], m["k"], m["w"], 0.0, start, s0=s0, free=free
    )
    assert bool(info.converged[0])
    assert float(info.loss[0]) <= f_matlab * (1 + 1e-9)
    # MATLAB's refit stops near this optimum: the activity agrees
    kern = sf.TimeKernel(m["k"])
    x, x_matlab = kern.conv(s), kern.conv(_t(m["debias"]["S_out"]))
    assert float((x - x_matlab).norm() / x_matlab.norm()) < 0.02

    s2, info = sf.fit_activity(
        m["z2"],
        _t(m["out"]["H"]),
        m["b2"],
        m["k2"],
        m["w"],
        lam,
        _t(m["ch2"]["LS"]),
    )
    assert bool(info.converged[0])
    assert float(info.loss[0]) < m["ch2"]["loss"]
