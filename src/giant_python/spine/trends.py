"""Trend tests of normalized spine sizes: ports of GIAnT-MATLAB's
``spineTrends.m`` and ``spineTrendsResid.m``.

The input is a source's normalized size across the trials of a session
(:func:`~giant_python.spine.readout.norm_log`). Two statistics look for a
change, both directions:

``step``
    For each split ``k`` (the first ``k`` trials against the rest), the t
    statistic of mean(after) - mean(before) with the pooled SD; the source's
    statistic is the largest ``|t|`` over splits, signed
    (:func:`step_stat`).
``brief``
    A transient change that comes back: each run of 1 to ``max_len``
    consecutive trials with trials on both sides (within a block of
    consecutive acquisition numbers) is compared with the up to 2 trials
    before it and the up to 2 after it. The noise SD is the median
    trial-to-trial difference away from the run, divided by
    ``0.6745 sqrt(2)`` (insensitive to slow drift, and not inflated by the
    change itself). The run's statistic is the smaller of the two t's when
    both have the same sign (else 0); the source's is the largest ``|.|``
    over runs, signed (:func:`brief_stat`).

``p`` comes from random trial orders of the same source:
``(1 + #{|stat(perm)| >= |stat|}) / (n_perm + 1)``, and the q-values
from Benjamini-Hochberg across the tested sources (:func:`bh_adjust`).
A permuted statistic within a relative ``1e-9`` of the observed one counts
as a tie (``>=``): the step at the first or last split depends only on
which trial stands alone, so every order that puts the same trial there
gives the same statistic up to rounding.

A trend shared by the sources (bleaching beyond the per-trial median,
focus) is the first right singular vector ``v`` of the tested sources'
normalized sizes, signed so that its last three trials sum to a positive
number (:func:`shared_mode`). Each source's part of it, ``(L . v) v``, is
removed and the step test is run again on what is left (``resid``), as in
``spineTrendsResid.m``.

Differences from the MATLAB scripts: the permutations come from numpy's
generator seeded per source (``(seed, source)``), so p agrees with MATLAB's
within Monte Carlo error; all tests of one source use the same trial
orders; ties count as above (in MATLAB, rounding decided whether a tie
counted, so its p at the first or last split could be far too small);
and ``spineTrends.m``'s correlation with the shared mode takes the
sign of ``spineTrendsResid.m``'s ``v`` (MATLAB's own SVD sign there is
arbitrary).
"""

from __future__ import annotations

import math

import numpy as np

# MATLAB's constant for a normal SD from a median absolute deviation
_MAD_SD = 0.6745
# permuted statistics this close (relative) to the observed one are ties
_TIE_RTOL = 1e-9
# within-group sums of squares below this share of a series' total are
# rounding error of the cumulative sums: they are taken as zero
_SS_RTOL = 1e-10


def trial_blocks(acquisition, unknown_consecutive: bool = True) -> np.ndarray:
    """Blocks of consecutive acquisition numbers, as ``spineTrends.m``:
    a new block starts where the number jumps by more than 1.

    A trial without a number (negative, as
    :func:`~giant_python.spine.pipeline.table_trials` reports it) is a
    block of its own, so that it is neither in a brief run nor one of its
    neighbours; when no trial has a number, the trials are taken as
    consecutive (one block) unless ``unknown_consecutive`` is False (for a
    subset of trials whose path has numbers elsewhere).

    Parameters
    ----------
    acquisition : array_like of int, shape (trials,)
    unknown_consecutive : bool
        Take the trials as one block when none has a number.

    Returns
    -------
    ndarray of int, shape (trials,)
        Block number of each trial, starting at 0.
    """
    a = np.asarray(acquisition, dtype=np.int64).ravel()
    if a.size == 0:
        return np.zeros(0, np.int64)
    known = a >= 0
    if not known.any() and unknown_consecutive:
        return np.zeros(a.size, np.int64)
    new = np.zeros(a.size, bool)
    new[1:] = (np.diff(a) > 1) | ~known[1:] | ~known[:-1]
    return np.cumsum(new).astype(np.int64)


def check_trend_settings(n_perm, fdr_q, max_brief, seed) -> None:
    """Refuse settings the tests cannot use (ValueError): ``n_perm`` must
    be at least 0, ``max_brief`` at least 1, ``fdr_q`` in (0, 1] and
    ``seed`` at least 0 (numpy's seed sequences take no negative
    numbers)."""
    if int(n_perm) != n_perm or n_perm < 0:
        raise ValueError(f"n_perm must be a whole number >= 0, not {n_perm}")
    if int(max_brief) != max_brief or max_brief < 1:
        raise ValueError(
            f"max_brief must be a whole number >= 1, not {max_brief}"
        )
    if not 0 < fdr_q <= 1:
        raise ValueError(f"fdr_q must be in (0, 1], not {fdr_q}")
    if int(seed) != seed or seed < 0:
        raise ValueError(f"seed must be a whole number >= 0, not {seed}")


def step_stat(x: np.ndarray):
    """The largest ``|t|`` of a step over all splits, per row.

    Parameters
    ----------
    x : ndarray of shape (series, trials)

    Returns
    -------
    t : ndarray of float64, shape (series,)
        Signed t of mean(after) - mean(before) at the best split (0 when no
        split gives a non-zero t; infinite when the groups of a split have
        no spread, within ``1e-10`` of the series' sum of squares).
    k : ndarray of int, shape (series,)
        Trials before the step at that split, 1 to ``trials - 1``; 0 when
        no split gives a non-zero t (MATLAB reports 1 there).
    d : ndarray of float64, shape (series,)
        mean(after) - mean(before) at that split (0 without one).
    """
    x = np.asarray(x, dtype=np.float64)
    n, T = x.shape
    t_best = np.zeros(n)
    k_best = np.zeros(n, np.int64)
    d_best = np.zeros(n)
    if T < 3:
        return t_best, k_best, d_best
    # centred, so that the sums of squares lose nothing to cancellation
    xc = x - x.mean(axis=1, keepdims=True)
    cs = np.cumsum(xc, axis=1)
    cs2 = np.cumsum(xc * xc, axis=1)
    tot, tot2 = cs[:, -1], cs2[:, -1]
    with np.errstate(invalid="ignore", divide="ignore"):
        for k in range(1, T):
            sa = cs[:, k - 1]
            sb = tot - sa
            ma, mb = sa / k, sb / (T - k)
            ss = (cs2[:, k - 1] - sa * ma) + ((tot2 - cs2[:, k - 1]) - sb * mb)
            # no spread within the groups: the statistic is infinite, the
            # same for a series and its mirror image (ties)
            ss = np.where(ss <= _SS_RTOL * tot2, 0.0, ss)
            s = np.sqrt(ss / (T - 2))
            d = mb - ma
            t = d / (s * math.sqrt(1.0 / k + 1.0 / (T - k)))
            upd = np.abs(t) > np.abs(t_best)
            t_best[upd] = t[upd]
            k_best[upd] = k
            d_best[upd] = d[upd]
    return t_best, k_best, d_best


def step_at(x: np.ndarray, k):
    """The step statistic at given splits (``spineTrends.m``'s ``stepAt``).

    Parameters
    ----------
    x : ndarray of shape (series, trials)
    k : array_like of int, shape (series,)
        Trials before the step, 1 to ``trials - 1``.

    Returns
    -------
    t, d : ndarray of float64, shape (series,)
        As :func:`step_stat` at those splits (with its rule for groups
        without spread); NaN where ``k`` is out of range.
    """
    x = np.asarray(x, dtype=np.float64)
    k = np.asarray(k, dtype=np.int64).ravel()
    n, T = x.shape
    t = np.full(n, np.nan)
    d = np.full(n, np.nan)
    for i in range(n):
        if not 1 <= k[i] <= T - 1 or T < 3:
            continue
        a, b = x[i, : k[i]], x[i, k[i] :]
        d[i] = b.mean() - a.mean()
        ss = ((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()
        if ss <= _SS_RTOL * ((x[i] - x[i].mean()) ** 2).sum():
            ss = 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            t[i] = d[i] / (
                math.sqrt(ss / (T - 2)) * math.sqrt(1 / k[i] + 1 / (T - k[i]))
            )
    return t, d


def _kth_without(srt, removed, j: int) -> np.ndarray:
    """Per row, the ``j``-th smallest (zero-based) of the sorted values
    ``srt`` once the elements of ranks ``removed`` are taken out.

    The answer is the sorted element at the smallest index ``i`` with
    ``i - #{removed <= i} = j``, which the iteration
    ``i <- j + #{removed <= i}`` reaches from ``i = j`` in at most
    ``removed.shape[1] + 1`` steps (it stops early once ``i`` is fixed).
    """
    i = np.full(srt.shape[0], j, np.int64)
    for _ in range(removed.shape[1] + 1):
        nxt = j + (removed <= i[:, None]).sum(axis=1)
        if np.array_equal(nxt, i):
            break
        i = nxt
    return np.take_along_axis(srt, i[:, None], axis=1)[:, 0]


def _median_without(srt, removed, m: int) -> np.ndarray:
    """Per row, the median of the ``m`` sorted values left once the elements
    of ranks ``removed`` are taken out (as ``median`` of what is left)."""
    if m <= 0:
        return np.full(srt.shape[0], np.nan)
    if m % 2:
        return _kth_without(srt, removed, (m - 1) // 2)
    return (
        _kth_without(srt, removed, m // 2 - 1)
        + _kth_without(srt, removed, m // 2)
    ) / 2


def brief_stat(x: np.ndarray, blocks, max_len: int = 3):
    """The strongest transient change of 1 to ``max_len`` trials, per row.

    Parameters
    ----------
    x : ndarray of shape (series, trials)
    blocks : array_like of int, shape (trials,)
        Block of each trial (:func:`trial_blocks`); a run and its
        neighbours lie in one block.
    max_len : int
        Longest run tested.

    Returns
    -------
    t : ndarray of float64, shape (series,)
        Signed statistic of the strongest run (0 without one).
    start : ndarray of int, shape (series,)
        The run's first trial (zero-based, at least 1); -1 without a run
        with a non-zero statistic (MATLAB reports the first trial there).
    length : ndarray of int, shape (series,)
        The run's number of trials; 0 without a run.
    d : ndarray of float64, shape (series,)
        The run's mean minus the mean of the before and after means (0
        without a run).
    """
    x = np.asarray(x, dtype=np.float64)
    blocks = np.asarray(blocks).ravel()
    n, T = x.shape
    t_best = np.zeros(n)
    s_best = np.full(n, -1, np.int64)
    w_best = np.zeros(n, np.int64)
    d_best = np.zeros(n)
    if T < 3:
        return t_best, s_best, w_best, d_best
    dd = np.abs(np.diff(x, axis=1))
    cols = np.flatnonzero(blocks[1:] == blocks[:-1])  # within-block steps
    m_all = cols.size
    sub = dd[:, cols]
    order = np.argsort(sub, axis=1, kind="stable")
    srt = np.take_along_axis(sub, order, axis=1)
    rank = np.empty_like(order)
    np.put_along_axis(
        rank, order, np.broadcast_to(np.arange(m_all), order.shape), axis=1
    )
    col_of = np.full(T - 1, -1, np.int64)
    col_of[cols] = np.arange(m_all)
    for w in range(1, max_len + 1):
        for a in range(1, T - w):
            e = a + w - 1
            b = blocks[a]
            if blocks[a - 1] != b or blocks[e + 1] != b or blocks[e] != b:
                continue
            # the steps that touch the run (a-1 .. e) are all within the block
            removed = rank[:, col_of[a - 1 : e + 1]]
            sg = _median_without(srt, removed, m_all - (w + 1)) / (
                _MAD_SD * math.sqrt(2)
            )
            left = [i for i in range(max(a - 2, 0), a) if blocks[i] == b]
            right = [
                i
                for i in range(e + 1, min(e + 2, T - 1) + 1)
                if blocks[i] == b
            ]
            m = x[:, a : e + 1].mean(axis=1)
            ml = x[:, left].mean(axis=1)
            mr = x[:, right].mean(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                tl = (m - ml) / (sg * math.sqrt(1 / w + 1 / len(left)))
                tr = (m - mr) / (sg * math.sqrt(1 / w + 1 / len(right)))
            t = np.sign(tl) * np.minimum(np.abs(tl), np.abs(tr))
            t[np.sign(tl) != np.sign(tr)] = 0
            d = m - (ml + mr) / 2
            upd = np.abs(t) > np.abs(t_best)
            t_best[upd] = t[upd]
            s_best[upd] = a
            w_best[upd] = w
            d_best[upd] = d[upd]
    return t_best, s_best, w_best, d_best


def bh_adjust(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values).

    A test passes FDR ``q`` exactly when its q-value is at most ``q``, the
    same decision as ``spineTrends.m``'s step-up rule. NaN entries are left
    out of the count and stay NaN.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    out = np.full(p.shape, np.nan)
    idx = np.flatnonzero(np.isfinite(p))
    m = idx.size
    if m == 0:
        return out
    order = idx[np.argsort(p[idx], kind="stable")]
    scaled = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(scaled[::-1])[::-1]
    out[order] = np.minimum(q, 1.0)
    return out


def shared_mode(x: np.ndarray):
    """The trend shared by the rows: the first right singular vector.

    Parameters
    ----------
    x : ndarray of shape (series, trials)
        Normalized sizes of the sources taken into account.

    Returns
    -------
    v : ndarray of float64, shape (trials,)
        Unit vector, signed so that its last three entries sum to a
        positive number (``spineTrendsResid.m``); NaN without rows.
    var_share : float
        Share of the variance (sum of squares) it carries.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] == 0 or x.shape[1] == 0:
        return np.full(x.shape[1], np.nan), float("nan")
    _, sv, vt = np.linalg.svd(x, full_matrices=False)
    v = vt[0]
    sgn = np.sign(v[-3:].sum())
    v = v * (sgn if sgn != 0 else 1.0)
    tot = float((sv**2).sum())
    return v, float(sv[0] ** 2 / tot) if tot > 0 else float("nan")


def _corr(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Pearson correlation of each row of ``x`` with ``v``."""
    xc = x - x.mean(axis=1, keepdims=True)
    vc = v - v.mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        return (xc @ vc) / (np.linalg.norm(xc, axis=1) * np.linalg.norm(vc))


def trial_orders(seed: int, source: int, n_perm: int, n_trials: int):
    """The random trial orders of one source: ``(n_perm, n_trials)`` indices
    from numpy's generator seeded with ``(seed, source)``, so that a
    source's p-values do not depend on which other sources are tested."""
    rng = np.random.default_rng([int(seed), int(source)])
    base = np.broadcast_to(np.arange(n_trials), (n_perm, n_trials))
    return rng.permuted(base, axis=1)


TREND_FIELDS = (
    "step_t",
    "step_k",
    "step_d",
    "step_p",
    "step_q",
    "brief_t",
    "brief_start",
    "brief_len",
    "brief_d",
    "brief_p",
    "brief_q",
    "resid_t",
    "resid_k",
    "resid_d",
    "resid_p",
    "resid_q",
    "shared_loading",
    "shared_r",
)


def _perm_p(x, resid, t_obs, seed, source, n_perm, blocks, max_brief):
    """One source's permutation p-values of the step, brief and resid
    tests (``t_obs``: the observed statistics, in that order)."""
    orders = trial_orders(seed, source, n_perm, x.size)
    xp, rp = x[orders], resid[orders]
    stats = (
        step_stat(xp)[0],
        brief_stat(xp, blocks, max_brief)[0],
        step_stat(rp)[0],
    )
    out = []
    for stat, t in zip(stats, t_obs):
        hits = np.count_nonzero(np.abs(stat) >= abs(t) * (1 - _TIE_RTOL))
        out.append((1 + hits) / (n_perm + 1))
    return out


def _permute(
    out, idx, x, resid, seed, n_perm, blocks, max_brief, workers, progress
) -> None:
    """Fill ``out``'s step, brief and resid p-values of the sources ``idx``
    (rows of ``x`` and ``resid``), in threads (numpy releases the GIL)."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    if workers is None:
        workers = min(8, os.cpu_count() or 1)
    names = ("step", "brief", "resid")

    def one(j):
        """p-values of the j-th tested source."""
        s = idx[j]
        t_obs = [out[f"{nm}_t"][s] for nm in names]
        return _perm_p(
            x[j], resid[j], t_obs, seed, s, n_perm, blocks, max_brief
        )

    with ThreadPoolExecutor(max(1, int(workers))) as pool:
        for j, ps in enumerate(pool.map(one, range(idx.size))):
            for nm, pv in zip(names, ps):
                out[f"{nm}_p"][idx[j]] = pv
            if progress is not None:
                progress(j + 1, idx.size)


def trend_tests(
    norm: np.ndarray,
    ok,
    acquisition,
    n_perm: int = 20000,
    fdr_q: float = 0.1,
    max_brief: int = 3,
    seed: int = 1,
    workers: int | None = None,
    progress=None,
    unknown_consecutive: bool = True,
) -> dict:
    """The step, brief and beyond-trend tests of every source.

    Parameters
    ----------
    norm : ndarray of shape (trials, sources)
        Normalized sizes over the trials tested (in acquisition order).
    ok : array_like of bool, shape (sources,)
        Sources to test; a source with a non-finite size in a trial is left
        out too.
    acquisition : array_like of int, shape (trials,)
        The trials' acquisition numbers (:func:`trial_blocks`; negative
        where unknown).
    n_perm : int
        Random trial orders per source for the p-values (0: statistics
        only, p and q NaN).
    fdr_q : float
        FDR level, in (0, 1]; kept for the counts in the result.
    max_brief : int
        Longest run of the brief test.
    seed : int
        Seed of the trial orders (:func:`trial_orders`).
    workers : int, optional
        Threads for the permutations (default: up to 8, one per CPU); the
        result does not depend on it.
    progress : callable, optional
        Called with ``(done, total)`` as sources are tested.
    unknown_consecutive : bool
        Passed to :func:`trial_blocks`.

    Returns
    -------
    dict
        Per source (arrays of length ``sources``; NaN, or -1 for the
        indices, where untested):

        * ``tested`` (bool);
        * ``step_t``, ``step_k`` (trials before the step, 1 to
          ``trials - 1``; 0 for a tested source without a step),
          ``step_d``, ``step_p``, ``step_q``;
        * ``brief_t``, ``brief_start`` (zero-based first trial of the
          run; -1 for a tested source without a run), ``brief_len`` (0
          without a run), ``brief_d``, ``brief_p``, ``brief_q``;
        * ``resid_*``: the step test beyond the shared trend;
        * ``shared_loading`` (``L . v``) and ``shared_r`` (correlation of
          the source's sizes with ``v``);

        and ``shared_v`` (trials,), ``var_share``, ``blocks`` (trials,) and
        ``counts``: per test the sources with p < 0.05 and with
        ``q <= fdr_q``, up and down.
    """
    check_trend_settings(n_perm, fdr_q, max_brief, seed)
    L = np.asarray(norm, dtype=np.float64).T  # (sources, trials)
    n_src, T = L.shape
    tested = np.asarray(ok, bool).ravel() & np.isfinite(L).all(axis=1)
    if T < 3:
        tested[:] = False
    blocks = trial_blocks(acquisition, unknown_consecutive)
    out: dict = {f: np.full(n_src, np.nan) for f in TREND_FIELDS}
    for f in ("step_k", "brief_start", "brief_len", "resid_k"):
        out[f] = np.full(n_src, -1, np.int64)
    out["tested"] = tested
    out["blocks"] = blocks
    idx = np.flatnonzero(tested)
    v, var_share = shared_mode(L[idx]) if T else (np.zeros(0), np.nan)
    out["shared_v"] = v
    out["var_share"] = var_share
    if idx.size:
        x = L[idx]
        loading = x @ v
        resid = x - loading[:, None] * v[None, :]
        out["shared_loading"][idx] = loading
        out["shared_r"][idx] = _corr(x, v)
        for name, arrs in (
            ("step", step_stat(x)),
            ("resid", step_stat(resid)),
        ):
            out[f"{name}_t"][idx], out[f"{name}_k"][idx] = arrs[0], arrs[1]
            out[f"{name}_d"][idx] = arrs[2]
        bt, bs, bw, bd = brief_stat(x, blocks, max_brief)
        out["brief_t"][idx], out["brief_start"][idx] = bt, bs
        out["brief_len"][idx], out["brief_d"][idx] = bw, bd
        if n_perm > 0:
            _permute(
                out,
                idx,
                x,
                resid,
                seed,
                n_perm,
                blocks,
                max_brief,
                workers,
                progress,
            )
    counts = {}
    for name in ("step", "brief", "resid"):
        out[f"{name}_q"] = bh_adjust(out[f"{name}_p"])
        t, p, q = out[f"{name}_t"], out[f"{name}_p"], out[f"{name}_q"]
        with np.errstate(invalid="ignore"):
            counts[name] = {
                "p05_up": int(np.count_nonzero((p < 0.05) & (t > 0))),
                "p05_down": int(np.count_nonzero((p < 0.05) & (t < 0))),
                "fdr_up": int(np.count_nonzero((q <= fdr_q) & (t > 0))),
                "fdr_down": int(np.count_nonzero((q <= fdr_q) & (t < 0))),
            }
    out["counts"] = counts
    return out
