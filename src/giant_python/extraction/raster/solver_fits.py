"""The iterative fits of SiLo's subproblem solver, on one loss, in torch.

GIAnT-MATLAB's ``extractSources`` fits the activity (``fitS``), the
footprints (``objfun_Hs``) and the debias refit on one loss,

    sum((Z - T)^2 / (F (T + 1))),    T = H (S * k) + B,

the squared residual over a variance that follows the model (or, in the
debias refit, a variance held at ``F s0``). :func:`variance_loss` is that
loss with its first and second derivatives in T; the activity and the
footprint problems put their own linear maps around it, and
:func:`projected_newton` minimizes either within bounds.

Both problems are convex, so they have one optimum, which the solver
reaches (MATLAB's fmincon runs stop at iteration caps before it). All
tensors carry a leading batch dimension, so that subproblems padded to a
common size are fitted together on the CPU or a GPU: padded pixels have
weight 0, padded sources are held at 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as tnf


def variance_loss(z, t, w, s0=None, order: int = 2):
    """The solver's loss and its derivatives in the model T.

    Parameters
    ----------
    z, t : Tensor of shape (batch, pixels, frames)
        Data and model.
    w : Tensor of the same shape
        Weights 1 / F (the freshness); 0 for padding.
    s0 : Tensor of the same shape, optional
        If given, the variance term ``T + 1`` is held at ``s0``.
    order : {0, 1, 2}
        Highest derivative to return.

    Returns
    -------
    f : Tensor of shape (batch,)
        ``sum(w (z - t)^2 / (t + 1))``, summed in float64.
    g, c : Tensor or None
        First and second derivative of each sample's term in ``t``.
    """
    e = z - t
    if s0 is None:
        s = t + 1
        r = e / s
        f = (w * e * r).sum((-2, -1), dtype=torch.float64)
        if order == 0:
            return f, None, None
        g = -w * r * (2 + r)
        if order == 1:
            return f, g, None
        return f, g, 2 * w * (r + 1) ** 2 / s
    ws = w / s0
    f = (ws * e * e).sum((-2, -1), dtype=torch.float64)
    if order == 0:
        return f, None, None
    g = -2 * ws * e
    return f, g, (2 * ws if order == 2 else None)


def _geometric_sum(x: torch.Tensor, a: float, taps: int, back: bool):
    """``sum_d a^d x[t - d]`` over d < taps (``x[t + d]`` if not back).

    By doubling: the sum over 2m taps is the sum over m plus a^m times
    itself shifted by m, so a kernel of n taps costs about 2 log2(n)
    shifted additions instead of n.
    """

    def shifted_add(y, z, m, c):
        """y + c * (z shifted by m frames, zero-filled)."""
        if m >= y.shape[-1]:
            return y
        if y.requires_grad or z.requires_grad:
            # out= below is not differentiable; this is, and slower
            pad = (m, 0) if back else (0, m)
            cut = z[..., :-m] if back else z[..., m:]
            return y + c * tnf.pad(cut, pad)
        out = torch.empty_like(y)
        if back:
            out[..., :m] = y[..., :m]
            torch.add(y[..., m:], z[..., :-m], alpha=c, out=out[..., m:])
        else:
            out[..., -m:] = y[..., -m:]
            torch.add(y[..., :-m], z[..., m:], alpha=c, out=out[..., :-m])
        return out

    y, m = x, 1
    for bit in bin(taps)[3:]:
        y = shifted_add(y, y, m, a**m)
        m *= 2
        if bit == "1":
            y = shifted_add(x, y, 1, a)
            m += 1
    return y


class TimeKernel:
    """MATLAB ``convn(x, k, 'same')`` along the last axis, and its adjoint.

    ``k`` has an odd length, as the solver's kernels do. Only its nonzero
    stretch is applied (the kernels are causal: their first half is zero).
    A causal exponential (``gain * decay^d`` from the centre on, as
    ``setParamsExtractTrial`` builds them) is applied by shifted
    additions (:func:`_geometric_sum`), other kernels by ``conv1d``.
    """

    def __init__(self, k: torch.Tensor) -> None:
        """Find the kernel's nonzero stretch and whether it is exponential."""
        k = k.flatten()
        if k.numel() % 2 != 1:
            raise ValueError("the kernel must have an odd length")
        c = k.numel() // 2
        nz = torch.nonzero(k).flatten()
        first, last = int(nz[0]), int(nz[-1])
        m = last - first + 1
        if m - 1 - c + first < 0 or c - first < 0:
            first, last, m = 0, k.numel() - 1, k.numel()
        ks = k[first : last + 1]
        self.fwd = ks.flip(0).reshape(1, 1, -1).contiguous()
        self.adj = ks.reshape(1, 1, -1).contiguous()
        self.pad_fwd = (m - 1 - c + first, c - first)
        self.pad_adj = (c - first, m - 1 - c + first)
        # a causal exponential, gain * decay^(0..lag) from the centre on
        self.gain, self.decay, self.lag = float(ks[0]), None, m - 1
        if first == c and m > 1:
            ratio = ks[1:] / ks[:-1]
            a = float(ratio[0])
            if 0 < a < 1 and bool(((ratio - a).abs() <= 1e-6 * a).all()):
                self.decay = a

    def conv(self, x: torch.Tensor) -> torch.Tensor:
        """``convn(x, k, 'same')`` along the last axis."""
        if self.decay is not None:
            y = _geometric_sum(x, self.decay, self.lag + 1, True)
            return self.gain * y
        y = tnf.conv1d(
            tnf.pad(x.reshape(-1, 1, x.shape[-1]), self.pad_fwd), self.fwd
        )
        return y.reshape(x.shape)

    def conv_adj(self, x: torch.Tensor) -> torch.Tensor:
        """``convn(x, flip(k), 'same')``: the adjoint of :meth:`conv`."""
        if self.decay is not None:
            y = _geometric_sum(x, self.decay, self.lag + 1, False)
            return self.gain * y
        y = tnf.conv1d(
            tnf.pad(x.reshape(-1, 1, x.shape[-1]), self.pad_adj), self.adj
        )
        return y.reshape(x.shape)


def _overlap_pairs(o: torch.Tensor):
    """Index pairs (j <= l) of a batch of symmetric overlap matrices.

    Pairs whose overlap is above 1e-12 of the geometric mean of their own
    (in any batch member), and every (j, j).
    """
    d = torch.sqrt(torch.diagonal(o, dim1=-2, dim2=-1))
    keep = (o > 1e-12 * d[:, :, None] * d[:, None, :]).any(0)
    keep = keep | torch.eye(o.shape[-1], dtype=torch.bool, device=o.device)
    jj, ll = torch.nonzero(torch.triu(keep), as_tuple=True)
    return jj, ll


class _PairSum:
    """Sums over source pairs: y_j = sum_l M_jl x_l, M stored for j <= l."""

    def __init__(self, jj: torch.Tensor, ll: torch.Tensor) -> None:
        """Index lists for the pairs and for their mirror images."""
        self.jj, self.ll = jj, ll
        self.off = torch.nonzero(jj != ll).flatten()
        self.jo, self.lo = jj[self.off], ll[self.off]
        self.self_idx = torch.nonzero(jj == ll).flatten()

    def apply(self, m: torch.Tensor, x: torch.Tensor, dim: int = -1):
        """``m`` holds the pairs along ``dim``, ``x`` the sources."""
        y = torch.zeros_like(x)
        y.index_add_(dim, self.jj, m * x.index_select(dim, self.ll))
        if self.off.numel():
            y.index_add_(
                dim,
                self.lo,
                m.index_select(dim, self.off) * x.index_select(dim, self.jo),
            )
        return y


class ActivityProblem:
    """The activity fit: events S given footprints H and baseline B.

    Minimizes ``sum(w (Z - T)^2 / (T + 1)) + lam sum(S)`` with
    ``T = H (S * k) + B`` (MATLAB's ``fitS``); with ``s0`` the variance
    term is held at ``s0`` (the debias refit).

    Parameters
    ----------
    z, b, w : Tensor of shape (batch, pixels, frames)
        Data, baseline and weights 1 / F.
    h : Tensor of shape (batch, pixels, sources)
        Footprints.
    k : Tensor of shape (length,)
        Temporal kernel.
    lam : float
        L1 penalty on the events.
    s0 : Tensor of shape (batch, pixels, frames), optional
        Fixed variance term.
    """

    def __init__(self, z, h, b, k, w, lam: float, s0=None) -> None:
        """Keep the data and find the source pairs that overlap."""
        self.z, self.h, self.b, self.w = z, h, b, w
        self.ht = h.transpose(1, 2).contiguous()
        self.lam = float(lam)
        self.s0 = s0
        self.kernel = TimeKernel(k)
        self.kernel_sq = TimeKernel(k * k)
        a = h.abs()
        self.pairs = _PairSum(*_overlap_pairs(a.transpose(1, 2) @ a))
        hh = h[:, :, self.pairs.jj] * h[:, :, self.pairs.ll]
        self.hht = hh.transpose(1, 2).contiguous()
        self.q = None
        self._tail = None

    def model(self, s: torch.Tensor) -> torch.Tensor:
        """T = H (S * k) + B."""
        return self.h @ self.kernel.conv(s) + self.b

    def value(self, s: torch.Tensor) -> torch.Tensor:
        """The loss at S, per batch member."""
        f, _, _ = variance_loss(self.z, self.model(s), self.w, self.s0, 0)
        return f + self.lam * s.sum((-2, -1), dtype=torch.float64)

    def value_grad(self, s: torch.Tensor):
        """Loss and gradient at S; keeps the curvature for the products."""
        fixed_q = self.s0 is not None and self.q is not None
        f, g, c = variance_loss(
            self.z, self.model(s), self.w, self.s0, 1 if fixed_q else 2
        )
        if not fixed_q:
            # curvature summed over pixels: source pairs x frames
            self.q = self.hht @ c
            self._tail = None
        grad = self.kernel.conv_adj(self.ht @ g) + self.lam
        return f + self.lam * s.sum((-2, -1), dtype=torch.float64), grad

    def hess_mult(self, v: torch.Tensor) -> torch.Tensor:
        """Hessian product at the point of the last :meth:`value_grad`."""
        y = self.pairs.apply(self.q, self.kernel.conv(v), dim=1)
        return self.kernel.conv_adj(y)

    def hess_diag(self) -> torch.Tensor:
        """The Hessian's diagonal, in S's layout."""
        return self.kernel_sq.conv_adj(self.q[:, self.pairs.self_idx])

    def _tail_curvature(self):
        """Each source's curvature summed over the decay that follows.

        ``sum_v m[v] decay^(2 (v - u))`` over v >= u, for every frame u,
        with m the curvature of the source's own activity; from a
        cumulative sum in the log domain, which neither overflows nor
        cancels.
        """
        if self._tail is None:
            m = self.q[:, self.pairs.self_idx]
            floor = m.flatten(1).amax(1).clamp_min(1e-300) * 1e-12
            m = torch.maximum(m, _per_member(floor, m)).to(torch.float64)
            n_t = m.shape[-1]
            ramp = (
                2
                * math.log(self.kernel.decay)
                * torch.arange(n_t, dtype=torch.float64, device=m.device)
            )
            log_tail = torch.logcumsumexp((m.log() + ramp).flip(-1), -1)
            tail = torch.exp(log_tail.flip(-1) - ramp)
            self._tail = torch.cat([tail, torch.zeros_like(tail[..., :1])], -1)
        return self._tail

    def preconditioner(self, free: torch.Tensor, diag: torch.Tensor):
        """An approximate inverse of the Hessian on the free events.

        With an exponential kernel the activity follows a first-order
        recursion, x[t] = decay x[t-1] + gain s[t]. Between two free
        events it only decays, so one source's Hessian on its free events
        factors as D^-T M D^-1, with D the recursion between consecutive
        free events and M diagonal (the curvature summed over the decay
        up to the next free event). Its inverse, D M^-1 D^T, costs a few
        passes over the frames. It is exact for one source up to the
        kernel's truncation; the coupling between sources is left to the
        conjugate gradients. Other kernels get the Hessian's diagonal.
        """
        fm = free.to(diag.dtype)
        a = self.kernel.decay
        if a is None:
            return lambda r: fm / diag * r
        n_t = free.shape[-1]
        lag = self.kernel.lag
        idx = torch.arange(n_t, device=free.device).expand(free.shape)
        last = torch.where(free, idx, -1).cummax(-1).values
        prv = torch.cat(
            [torch.full_like(last[..., :1], -1), last[..., :-1]], -1
        )
        nxt = torch.where(free, idx, n_t).flip(-1).cummin(-1).values.flip(-1)
        nxt = torch.cat([nxt[..., 1:], torch.full_like(nxt[..., :1], n_t)], -1)
        gap_p, gap_n = idx - prv, nxt - idx
        dt = torch.float64
        cp = torch.where((prv >= 0) & (gap_p <= lag), a ** gap_p.to(dt), 0.0)
        cn = torch.where((nxt < n_t) & (gap_n <= lag), a ** gap_n.to(dt), 0.0)
        reach = gap_n.clamp(max=lag + 1)
        tail = self._tail_curvature()
        m = tail[..., :-1] - a ** (2 * reach.to(dt)) * tail.gather(
            -1, idx + reach
        )
        scale = fm.to(dt) / (self.kernel.gain**2 * m)
        prv, nxt = prv.clamp(min=0), nxt.clamp(max=n_t - 1)
        cp, cn = cp.to(diag.dtype), cn.to(diag.dtype)
        scale = scale.to(diag.dtype)

        def apply(r):
            """D M^-1 D^T on the free events."""
            u = (r - cn * r.gather(-1, nxt)) * scale
            return fm * (u - cp * u.gather(-1, prv))

        return apply


class FootprintProblem:
    """The footprint fit: super-resolved footprints Hs given the activity.

    Minimizes ``sum(w (Z - T)^2 / (T + 1))`` with ``T = (G Hs) X + B``,
    where G applies the footprint filter on the selected pixels (MATLAB's
    ``objfun_Hs`` and ``hessmult_Hs``); with ``s0`` the variance term is
    held at ``s0``.

    Parameters
    ----------
    z, b, w : Tensor of shape (batch, pixels, frames)
    x : Tensor of shape (batch, sources, frames)
        Activity.
    g : Tensor of shape (batch, pixels, pixels)
        The footprint filter as a matrix.
    support : Tensor of bool, shape (batch, pixels, sources)
        Where Hs may be nonzero; it decides which source pairs interact.
    s0 : Tensor of shape (batch, pixels, frames), optional
        Fixed variance term.
    """

    def __init__(self, z, x, b, g, w, support, s0=None) -> None:
        """Keep the data and find the source pairs that interact."""
        self.z, self.x, self.b, self.w, self.g = z, x, b, w, g
        self.s0 = s0
        self.xt = x.transpose(1, 2).contiguous()
        self.gt = g.transpose(1, 2).contiguous()
        reach = g.abs() @ support.to(g.dtype)
        self.pairs = _PairSum(*_overlap_pairs(reach.transpose(1, 2) @ reach))
        self.xx = (x[:, self.pairs.jj] * x[:, self.pairs.ll]).transpose(1, 2)
        self.g2t = (g * g).transpose(1, 2).contiguous()
        self.p = None

    def footprints(self, hs: torch.Tensor) -> torch.Tensor:
        """H = G Hs."""
        return self.g @ hs

    def value(self, hs: torch.Tensor) -> torch.Tensor:
        """The loss at Hs, per batch member."""
        t = (self.g @ hs) @ self.x + self.b
        return variance_loss(self.z, t, self.w, self.s0, 0)[0]

    def value_grad(self, hs: torch.Tensor):
        """Loss and gradient at Hs; keeps the curvature for the products."""
        t = (self.g @ hs) @ self.x + self.b
        f, g, c = variance_loss(self.z, t, self.w, self.s0, 2)
        # curvature summed over frames: pixels x source pairs
        self.p = c @ self.xx
        return f, self.gt @ (g @ self.xt)

    def hess_mult(self, v: torch.Tensor) -> torch.Tensor:
        """Hessian product at the point of the last :meth:`value_grad`."""
        return self.gt @ self.pairs.apply(self.p, self.g @ v)

    def hess_diag(self) -> torch.Tensor:
        """The Hessian's diagonal, in Hs's layout."""
        return self.g2t @ self.p[:, :, self.pairs.self_idx]

    def preconditioner(self, free: torch.Tensor, diag: torch.Tensor):
        """The inverse of the Hessian's diagonal on the free entries."""
        minv = free.to(diag.dtype) / diag
        return lambda r: minv * r


@dataclass
class FitInfo:
    """How a fit ended, per batch member.

    Attributes
    ----------
    loss : Tensor of shape (batch,)
        Final loss.
    optimality : Tensor of shape (batch,)
        Largest projected-gradient component at the end.
    predicted : Tensor of shape (batch,)
        Decrease of the loss that the last quadratic model predicted.
    iterations : Tensor of int, shape (batch,)
        Newton iterations used.
    converged : Tensor of bool, shape (batch,)
        Ended on the tolerance (False: iteration cap, or a line search
        that failed above it).
    evaluations : int
        Loss evaluations, for the batch.
    hess_products : int
        Hessian products, for the batch.
    history : list of Tensor
        Loss after each iteration, if requested.
    """

    loss: torch.Tensor
    optimality: torch.Tensor
    predicted: torch.Tensor
    iterations: torch.Tensor
    converged: torch.Tensor
    evaluations: int = 0
    hess_products: int = 0
    history: list = field(default_factory=list)


def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Inner product of each batch member, in float64."""
    return (a * b).flatten(1).sum(1, dtype=torch.float64)


def _per_member(v: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """A per-member vector shaped to broadcast against ``like``."""
    return v.reshape(-1, *([1] * (like.dim() - 1)))


def _scale(s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Each batch member of ``a`` times its own scalar."""
    return _per_member(s.to(a.dtype), a) * a


def _pcg(amul, rhs, precond, tol, max_iter: int):
    """Preconditioned conjugate gradients for a batch of systems."""
    x = torch.zeros_like(rhs)
    r = rhs.clone()
    zv = precond(r)
    p = zv.clone()
    rz = _dot(r, zv)
    bnorm = torch.sqrt(_dot(rhs, rhs))
    active = bnorm > 0
    n = 0
    if not active.any():
        return x, n
    for n in range(1, max_iter + 1):
        ap = amul(p)
        pap = _dot(p, ap)
        ok = active & (pap > 0)
        alpha = torch.where(ok, rz / pap.clamp_min(1e-300), 0.0)
        x += _scale(alpha, p)
        r -= _scale(alpha, ap)
        active = ok & (torch.sqrt(_dot(r, r)) > tol * bnorm)
        if not active.any():
            break
        zv = precond(r)
        rz_new = _dot(r, zv)
        beta = torch.where(active, rz_new / rz.clamp_min(1e-300), 0.0)
        p = zv + _scale(beta, p)
        rz = rz_new
    return x, n


def _arc_search(problem, g, d, hd, q, delta, t, lo, hi, search):
    """Backtracking search on the model along a projected step.

    From the step ``d`` (model value ``q``, ``hd`` = H d), tries
    ``clamp(d + t delta)`` for the members in ``search``, halving ``t``
    until the model decreases enough. Returns the new ``d``, ``hd``,
    ``q``, the unprojected step taken, the step lengths and the number
    of Hessian products used.
    """
    grad_q = g + hd
    d_old, raw = d, d
    n_prod = 0
    for _ in range(40):
        raw_try = d_old + _scale(t, delta)
        d_try = torch.maximum(torch.minimum(raw_try, hi), lo)
        hd_try = problem.hess_mult(d_try)
        n_prod += 1
        q_try = _dot(g, d_try) + 0.5 * _dot(d_try, hd_try)
        ok = search & (q_try <= q + 1e-4 * _dot(grad_q, d_try - d_old))
        okb = _per_member(ok, d)
        d = torch.where(okb, d_try, d)
        raw = torch.where(okb, raw_try, raw)
        hd = torch.where(okb, hd_try, hd)
        q = torch.where(ok, q_try, q)
        search = search & ~ok
        if not search.any():
            break
        t = torch.where(search, t / 2, t)
    return d, hd, q, raw, t, n_prod


def _model_step(problem, g, diag, lo, hi, alive, tol, cg_iter, rounds):
    """A step that lowers the quadratic model within the bounds.

    Minimizes ``q(d) = g.d + d.Hd / 2`` over ``lo <= d <= hi``
    approximately, as trust-region Newton methods for bound constraints
    do: a projected, diagonally scaled gradient step picks the variables
    that no bound holds; then conjugate gradients on those, a backtracking
    search along the projected step, and again with the variables that
    reached a bound held there, until none does. Each search starts from
    twice the step length that the round before accepted.

    Returns the step, the model's value there (per member) and the
    number of Hessian products.
    """
    zero = torch.zeros_like(g)
    q = torch.zeros(g.shape[0], dtype=torch.float64, device=g.device)
    # projected gradient step, from the exact step length along it
    delta = torch.maximum(torch.minimum(-g / diag, hi), lo)
    h_delta = problem.hess_mult(delta)
    curv = _dot(delta, h_delta)
    t = torch.where(curv > 0, -_dot(g, delta) / curv.clamp_min(1e-300), 1.0)
    d, hd, q, _, _, n_prod = _arc_search(
        problem, g, zero, zero, q, delta, t.clamp(max=1.0), lo, hi, alive
    )
    n_prod += 1
    free = (d > lo) & (d < hi)
    todo = alive
    t = torch.ones_like(q)
    for _ in range(rounds):
        fm = free.to(g.dtype)
        delta, n = _pcg(
            lambda v: fm * problem.hess_mult(fm * v),
            -(g + hd) * fm,
            problem.preconditioner(free, diag),
            tol,
            cg_iter,
        )
        n_prod += n
        search = todo & (_dot(delta, delta) > 0)
        if not search.any():
            break
        d, hd, q, raw, t, n = _arc_search(
            problem, g, d, hd, q, delta, t, lo, hi, search
        )
        n_prod += n
        t = (2 * t).clamp(max=1.0)
        # the variables that the bounds cut are held there from now on;
        # a member that none cut is done (whatever the others need)
        clipped = free & ((raw < lo) | (raw > hi)) & _per_member(search, d)
        todo = clipped.flatten(1).any(1)
        if not todo.any():
            break
        free = free & ~clipped
    return d, q, n_prod


def projected_newton(  # noqa: C901
    problem,
    x0: torch.Tensor,
    lb,
    ub,
    *,
    max_iter: int = 100,
    ftol: float | None = None,
    cg_iter: int = 100,
    cg_tol: float = 0.1,
    rounds: int = 10,
    history: bool = False,
) -> tuple[torch.Tensor, FitInfo]:
    """Minimize a convex function within bounds, for a batch of problems.

    A projected Newton method. Each iteration builds the quadratic model
    of the loss at the current point and lowers it within the bounds
    (:func:`_model_step`: a projected gradient step picks the variables
    that no bound holds; preconditioned conjugate gradients on the
    Hessian products solve for those; the variables that reach a bound
    are held there and the rest solved again). A backtracking search
    along that step enforces a sufficient decrease of the loss.

    Parameters
    ----------
    problem : object
        With ``value(x)``, ``value_grad(x)`` (which keeps the curvature),
        ``hess_mult(v)``, ``hess_diag()`` and ``preconditioner(free,
        diag)``; values are per batch member.
    x0 : Tensor of shape (batch, ...)
        Start; it is projected into the bounds.
    lb, ub : Tensor or float
        Bounds, broadcastable to ``x0``. Variables with ``lb == ub`` stay
        there.
    max_iter : int
        Cap on the Newton iterations.
    ftol : float, optional
        A member has converged when its model predicts a decrease of the
        loss below ``ftol * (1 + |f|)``. Default 1e-12 in float64 and 1e-7
        in float32 (about the rounding of the loss).
    cg_iter, cg_tol : int, float
        Cap and largest relative residual of the conjugate gradients (the
        tolerance tightens as the fit converges).
    rounds : int
        Cap on the rounds of :func:`_model_step`.
    history : bool
        Keep the loss after each iteration.

    Returns
    -------
    x : Tensor
    info : FitInfo
    """
    dt = x0.dtype
    if ftol is None:
        ftol = 1e-12 if dt == torch.float64 else 1e-7
    lb = torch.as_tensor(lb, dtype=dt, device=x0.device)
    ub = torch.as_tensor(ub, dtype=dt, device=x0.device)
    x = torch.maximum(torch.minimum(x0, ub), lb)
    batch = x.shape[0]
    f, g = problem.value_grad(x)
    n_eval, n_prod = 1, 0
    alive = torch.ones(batch, dtype=torch.bool, device=x.device)
    converged = torch.zeros_like(alive)
    iters = torch.zeros(batch, dtype=torch.long, device=x.device)
    pred = torch.full_like(f, float("inf"))
    hist = [f.clone()] if history else []
    opt0 = None
    for _ in range(max_iter):
        lo, hi = lb - x, ub - x
        opt = torch.maximum(torch.minimum(-g, hi), lo).abs()
        opt = opt.flatten(1).amax(1).to(torch.float64)
        if opt0 is None:
            opt0 = opt.clamp_min(1e-300)
        diag = problem.hess_diag()
        floor = diag.flatten(1).amax(1).clamp_min(1e-300) * 1e-12
        diag = torch.maximum(diag, _per_member(floor, diag))
        tol = torch.sqrt(opt / opt0).clamp(1e-8, cg_tol)
        d, q, n = _model_step(
            problem, g, diag, lo, hi, alive, tol, cg_iter, rounds
        )
        n_prod += n
        pred = torch.where(alive, -q, pred)
        done = alive & (-q <= ftol * (1 + f.abs()))
        converged |= done
        alive &= ~done
        if not alive.any():
            break
        d = d * _per_member(alive, d)
        slope = _dot(g, d)
        # backtracking search on the loss along the step
        t = torch.ones_like(f)
        x_new, f_new, g_new = x, f.clone(), None
        search = alive.clone()
        for trial_no in range(40):
            trial = torch.maximum(torch.minimum(x + _scale(t, d), ub), lb)
            if trial_no == 0:
                ft, g_trial = problem.value_grad(trial)
            else:
                ft = problem.value(trial)
            n_eval += 1
            ok = search & (ft <= f + 1e-4 * t * slope)
            x_new = torch.where(_per_member(ok, x), trial, x_new)
            f_new = torch.where(ok, ft, f_new)
            search &= ~ok
            if trial_no == 0 and not search.any():
                g_new = g_trial
            if not search.any():
                break
            t = torch.where(search, t / 2, t)
        # a member whose search failed stays where it is; below the
        # rounding of the loss that counts as converged
        converged |= search & (pred <= 1e3 * ftol * (1 + f.abs()))
        alive &= ~search
        iters += alive.long()
        x = x_new
        if g_new is None:
            f, g = problem.value_grad(x)
            n_eval += 1
        else:
            f, g = f_new, g_new
        if history:
            hist.append(f.clone())
        if not alive.any():
            break
    lo, hi = lb - x, ub - x
    opt = torch.maximum(torch.minimum(-g, hi), lo).abs().flatten(1).amax(1)
    info = FitInfo(
        loss=f,
        optimality=opt,
        predicted=pred,
        iterations=iters,
        converged=converged,
        evaluations=n_eval,
        hess_products=n_prod,
        history=hist,
    )
    return x, info


def fit_activity(
    z, h, b, k, w, lam: float, s_start, *, s0=None, free=None, **options
) -> tuple[torch.Tensor, FitInfo]:
    """Fit the events S >= 0 (MATLAB's ``fitS``), to convergence.

    Parameters
    ----------
    z, b, w : Tensor of shape (batch, pixels, frames)
        Data, baseline and weights 1 / F.
    h : Tensor of shape (batch, pixels, sources)
    k : Tensor
        Temporal kernel.
    lam : float
        L1 penalty.
    s_start : Tensor of shape (batch, sources, frames)
        Start (projected to >= 0).
    s0 : Tensor, optional
        Fixed variance term (the debias refit; see
        :class:`ActivityProblem`).
    free : Tensor of bool, optional
        Entries of S to fit; the others are held at 0.
    **options
        Passed to :func:`projected_newton`.
    """
    problem = ActivityProblem(z, h, b, k, w, lam, s0)
    ub = torch.full_like(s_start, float("inf"))
    if free is not None:
        ub = torch.where(free, ub, torch.zeros_like(ub))
        s_start = torch.where(free, s_start, torch.zeros_like(s_start))
    return projected_newton(problem, s_start, 0.0, ub, **options)


def fit_footprints(
    z, x, b, g, w, hs_start, valid, *, s0=None, **options
) -> tuple[torch.Tensor, FitInfo]:
    """Fit the super-resolved footprints Hs >= 0 on their support.

    MATLAB's footprint fit (``objfun_Hs``), to convergence.

    Parameters
    ----------
    z, b, w : Tensor of shape (batch, pixels, frames)
    x : Tensor of shape (batch, sources, frames)
        Activity.
    g : Tensor of shape (batch, pixels, pixels)
        The footprint filter as a matrix.
    hs_start : Tensor of shape (batch, pixels, sources)
        Start.
    valid : Tensor of bool, shape (batch, pixels, sources)
        Where Hs may be nonzero.
    s0 : Tensor, optional
        Fixed variance term (see :class:`FootprintProblem`).
    **options
        Passed to :func:`projected_newton`.
    """
    problem = FootprintProblem(z, x, b, g, w, valid, s0)
    ub = torch.where(
        valid,
        torch.full_like(hs_start, float("inf")),
        torch.zeros_like(hs_start),
    )
    return projected_newton(problem, hs_start, 0.0, ub, **options)
