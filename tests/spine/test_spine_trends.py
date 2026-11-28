"""Tests of the trend tests (:mod:`giant_python.spine.trends`), against
direct transcriptions of GIAnT-MATLAB's ``spineTrends.m`` and
``spineTrendsResid.m``."""

import math

import numpy as np
import pytest

from giant_python.spine.trends import (
    _median_without,
    bh_adjust,
    brief_stat,
    shared_mode,
    step_at,
    step_stat,
    trend_tests,
    trial_blocks,
    trial_orders,
)


def _step_loop(x):
    """``stepStat2`` as written in MATLAB (rows = series)."""
    n, T = x.shape
    tb, kb, db = np.zeros(n), np.ones(n, int), np.zeros(n)
    for k in range(1, T):
        a, b = x[:, :k], x[:, k:]
        d = b.mean(1) - a.mean(1)
        ss = ((a - a.mean(1, keepdims=True)) ** 2).sum(1) + (
            (b - b.mean(1, keepdims=True)) ** 2
        ).sum(1)
        t = d / (np.sqrt(ss / (T - 2)) * math.sqrt(1 / k + 1 / (T - k)))
        upd = np.abs(t) > np.abs(tb)
        tb[upd], kb[upd], db[upd] = t[upd], k, d[upd]
    return tb, kb, db


def _brief_loop(x, blk, max_w):
    """``briefStat`` as written in MATLAB, zero-based, with np.median."""
    n, T = x.shape
    tb, db = np.zeros(n), np.zeros(n)
    sb, wb = np.zeros(n, int), np.ones(n, int)
    dd = np.abs(np.diff(x, axis=1))
    in_blk = blk[1:] == blk[:-1]
    for w in range(1, max_w + 1):
        for a in range(1, T - w):
            e = a + w - 1
            b = blk[a]
            if blk[a - 1] != b or blk[e + 1] != b or blk[e] != b:
                continue
            away = in_blk.copy()
            away[a - 1 : e + 1] = False
            sg = np.median(dd[:, away], axis=1) / (0.6745 * math.sqrt(2))
            left = [i for i in range(max(a - 2, 0), a) if blk[i] == blk[a]]
            right = [
                i
                for i in range(e + 1, min(e + 2, T - 1) + 1)
                if blk[i] == blk[a]
            ]
            m = x[:, a : e + 1].mean(1)
            ml, mr = x[:, left].mean(1), x[:, right].mean(1)
            tl = (m - ml) / (sg * math.sqrt(1 / w + 1 / len(left)))
            tr = (m - mr) / (sg * math.sqrt(1 / w + 1 / len(right)))
            t = np.sign(tl) * np.minimum(abs(tl), abs(tr))
            t[np.sign(tl) != np.sign(tr)] = 0
            d = m - (ml + mr) / 2
            upd = np.abs(t) > np.abs(tb)
            tb[upd], sb[upd], wb[upd], db[upd] = t[upd], a, w, d[upd]
    return tb, sb, wb, db


def test_trial_blocks_split_at_gaps():
    """A new block starts where the acquisition number jumps by more than 1
    (a repeat or a decrease does not start one, as in MATLAB)."""
    np.testing.assert_array_equal(
        trial_blocks([2, 3, 4, 7, 8, 8, 6, 10]), [0, 0, 0, 1, 1, 1, 1, 2]
    )
    assert trial_blocks([]).size == 0


def test_trials_without_numbers_are_blocks_of_their_own():
    """A trial without an acquisition number is neither in a brief run nor
    a neighbour of one; with no numbers at all, trials are consecutive."""
    np.testing.assert_array_equal(
        trial_blocks([10, 11, -1, -1, 14, 15]), [0, 0, 1, 2, 3, 3]
    )
    np.testing.assert_array_equal(trial_blocks([-1, -1, -1]), [0, 0, 0])
    np.testing.assert_array_equal(
        trial_blocks([-1, -1, -1], unknown_consecutive=False), [0, 1, 2]
    )
    x = np.zeros((1, 7))
    x[0, 3] = 1.0  # a transient at the trial without a number
    x += np.random.default_rng(0).normal(size=x.shape) * 1e-3
    t, start, _, _ = brief_stat(x, trial_blocks([1, 2, 3, -1, 5, 6, 7]), 1)
    assert start[0] != 3
    t, start, _, _ = brief_stat(x, trial_blocks([1, 2, 3, 4, 5, 6, 7]), 1)
    assert start[0] == 3


@pytest.mark.parametrize(
    "kw, match",
    [
        (dict(n_perm=-1), "n_perm"),
        (dict(max_brief=0), "max_brief"),
        (dict(fdr_q=2.0), "fdr_q"),
        (dict(fdr_q=0.0), "fdr_q"),
        (dict(seed=-3), "seed"),
    ],
)
def test_trend_settings_are_checked(kw, match):
    """Settings the tests cannot use are refused."""
    with pytest.raises(ValueError, match=match):
        trend_tests(np.zeros((5, 2)), np.ones(2, bool), np.arange(5), **kw)


def test_step_stat_is_the_matlab_loop():
    """Cumulative sums give MATLAB's statistic, split and size."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(200, 17)) * 0.1
    x[:20, 9:] += 0.3  # some real steps
    got = step_stat(x)
    ref = _step_loop(x)
    np.testing.assert_allclose(got[0], ref[0], rtol=1e-10)
    np.testing.assert_array_equal(got[1], ref[1])
    np.testing.assert_allclose(got[2], ref[2], rtol=1e-10, atol=1e-15)
    t, d = step_at(x, got[1])
    np.testing.assert_allclose(t, got[0], rtol=1e-10)
    np.testing.assert_allclose(d, got[2], rtol=1e-10, atol=1e-15)


def test_no_step_or_run_keeps_a_missing_position():
    """A series without a non-zero statistic reports no split and no run
    (not the first trial), and so does the summary through the pipeline's
    mapping."""
    x = np.zeros((2, 8))
    x[1] = np.arange(8) * 0.01
    t, k, d = step_stat(x)
    assert t[0] == 0 and k[0] == 0 and d[0] == 0
    assert k[1] > 0
    # a strictly linear series: every trial is the mean of its neighbours,
    # so no run differs from both sides
    t, start, length, d = brief_stat(x, np.zeros(8, int), 3)
    np.testing.assert_array_equal(t, 0)
    np.testing.assert_array_equal(start, -1)
    np.testing.assert_array_equal(length, 0)
    # blocks of one or two trials leave no run with neighbours
    rng = np.random.default_rng(1)
    y = rng.normal(size=(3, 6))
    t, start, length, _ = brief_stat(y, trial_blocks([1, 2, 5, 6, 9, 10]), 3)
    assert (t == 0).all() and (start == -1).all() and (length == 0).all()
    r = trend_tests(x.T, np.ones(2, bool), np.arange(8), n_perm=50)
    assert r["tested"].all()
    assert r["step_k"][0] == 0 and r["step_p"][0] == 1
    assert r["brief_start"][0] == -1 and r["brief_len"][0] == 0


def test_step_at_out_of_range_is_nan():
    """A split outside 1..T-1 gives NaN."""
    t, d = step_at(np.zeros((2, 5)) + [[0, 1, 2, 3, 4]], [0, 2])
    assert np.isnan(t[0]) and np.isnan(d[0]) and d[1] == 2.5


def test_median_without_matches_numpy():
    """The median once some ranks are taken out equals np.median of what
    is left, ties included."""
    rng = np.random.default_rng(5)
    for m_all, k in ((9, 2), (10, 3), (12, 4), (5, 4)):
        vals = rng.integers(0, 4, size=(300, m_all)).astype(float)
        order = np.argsort(vals, axis=1, kind="stable")
        srt = np.take_along_axis(vals, order, 1)
        rank = np.empty_like(order)
        np.put_along_axis(
            rank, order, np.broadcast_to(np.arange(m_all), order.shape), 1
        )
        cols = np.sort(rng.choice(m_all, size=k, replace=False))
        keep = np.setdiff1d(np.arange(m_all), cols)
        got = _median_without(srt, rank[:, cols], m_all - k)
        if m_all - k == 0:
            assert np.isnan(got).all()
        else:
            np.testing.assert_array_equal(got, np.median(vals[:, keep], 1))


def test_brief_stat_is_the_matlab_loop():
    """The brief statistic, run and size equal MATLAB's, with blocks."""
    rng = np.random.default_rng(4)
    T = 15
    x = rng.normal(size=(300, T)) * 0.1
    x[:30, 6:8] += 0.4  # transient changes
    blk = trial_blocks([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16])
    for max_w in (1, 3):
        got = brief_stat(x, blk, max_w)
        ref = _brief_loop(x, blk, max_w)
        np.testing.assert_allclose(got[0], ref[0], rtol=1e-10)
        np.testing.assert_array_equal(got[1], ref[1])
        np.testing.assert_array_equal(got[2], ref[2])
        np.testing.assert_allclose(got[3], ref[3], rtol=1e-10, atol=1e-15)
    # the planted runs are found
    t, start, length, _ = brief_stat(x[:30], blk, 3)
    assert (t > 0).mean() > 0.9
    assert np.median(start) == 6 and np.median(length) == 2


def test_brief_stat_keeps_runs_inside_blocks():
    """No run (or its neighbours) crosses a block boundary."""
    x = np.zeros((1, 8))
    x[0, 3] = 1.0  # trial 3 starts block 1: no neighbours before it there
    x += np.random.default_rng(0).normal(size=x.shape) * 1e-3
    blk = trial_blocks([1, 2, 3, 10, 11, 12, 13, 14])
    t, start, _, _ = brief_stat(x, blk, 1)
    assert start[0] != 3


def test_bh_adjust_is_the_step_up_rule():
    """q <= level exactly where the step-up rule rejects; NaN stays NaN."""
    rng = np.random.default_rng(6)
    p = np.concatenate([rng.uniform(size=80), rng.uniform(0, 0.002, 20)])
    p[5] = np.nan
    q = bh_adjust(p)
    assert np.isnan(q[5])
    fin = np.flatnonzero(np.isfinite(p))
    ps = np.sort(p[fin])
    m = fin.size
    below = np.flatnonzero(ps <= np.arange(1, m + 1) / m * 0.1)
    expect = np.zeros(p.size, bool)
    if below.size:
        expect[fin] = p[fin] <= ps[below[-1]]
    np.testing.assert_array_equal(q <= 0.1, expect)
    np.testing.assert_allclose(
        bh_adjust([0.01, 0.04, 0.03]), [0.03, 0.04, 0.04]
    )


def test_shared_mode_sign_and_share():
    """A rank-one pattern is recovered with all the variance, signed
    positive at the end."""
    v0 = np.linspace(1, -1, 10)
    v0 /= np.linalg.norm(v0)
    x = np.outer([1.0, -2.0, 0.5], v0)
    v, share = shared_mode(x)
    np.testing.assert_allclose(v, -v0, atol=1e-12)  # v0 ends negative
    assert share == pytest.approx(1.0)
    v, share = shared_mode(np.zeros((0, 4)))
    assert np.isnan(v).all() and np.isnan(share)


def _session(rng, n_src=40, T=20, step=0.25):
    """Normalized-like sizes (trials, sources): noise, a shared drift (most
    of the variance, as in real sessions) and one source with a step after
    trial 9."""
    drift = np.linspace(-0.2, 0.2, T)
    x = rng.normal(size=(T, n_src)) * 0.05 + drift[:, None] * rng.normal(
        size=n_src
    )
    x[10:, 0] += step
    return x - x.mean(0)


def test_trend_tests_find_a_planted_step():
    """A planted step has small p and q; beyond the shared drift, the other
    sources do not pile up at small p."""
    rng = np.random.default_rng(7)
    x = _session(rng)
    ok = np.ones(x.shape[1], bool)
    ok[3] = False
    r = trend_tests(x, ok, np.arange(2, 22), n_perm=2000, workers=2)
    assert r["tested"].sum() == x.shape[1] - 1 and not r["tested"][3]
    assert np.isnan(r["step_p"][3]) and r["step_k"][3] == -1
    assert r["step_k"][0] == 10 and r["step_t"][0] > 0
    assert r["step_p"][0] < 0.002 and r["step_q"][0] <= 0.1
    # the shared trend is the drift, not the planted step; the resid test
    # is the step test of what is left once each source's part is removed
    assert abs(np.corrcoef(r["shared_v"], np.arange(20))[0, 1]) > 0.95
    tested = r["tested"]
    lx = x.T[tested]
    v = shared_mode(lx)[0]
    np.testing.assert_allclose(r["shared_v"], v)
    np.testing.assert_allclose(r["shared_loading"][tested], lx @ v)
    rt, rk, rd = step_stat(lx - np.outer(lx @ v, v))
    np.testing.assert_allclose(r["resid_t"][tested], rt)
    np.testing.assert_array_equal(r["resid_k"][tested], rk)
    # every source follows the drift (a real step for the step test);
    # beyond it, the others are null
    assert (r["step_p"][4:] < 0.05).mean() > 0.5
    assert (r["resid_p"][4:] < 0.05).mean() < 0.2
    assert r["counts"]["step"]["fdr_up"] >= 1
    assert r["shared_v"].shape == (20,) and 0 < r["var_share"] < 1


def test_trend_tests_p_does_not_depend_on_the_others_or_threads():
    """Each source has its own trial orders: p is the same whichever other
    sources are tested and however many threads run."""
    rng = np.random.default_rng(8)
    x = _session(rng, n_src=6, T=12)
    acq = np.arange(12)
    a = trend_tests(x, np.ones(6, bool), acq, n_perm=500, workers=1)
    b = trend_tests(x, np.ones(6, bool), acq, n_perm=500, workers=4)
    np.testing.assert_array_equal(a["brief_p"], b["brief_p"])
    np.testing.assert_array_equal(a["step_p"], b["step_p"])
    ok = np.zeros(6, bool)
    ok[[0, 2]] = True
    c = trend_tests(x, ok, acq, n_perm=500)
    np.testing.assert_array_equal(c["step_p"][[0, 2]], a["step_p"][[0, 2]])
    np.testing.assert_array_equal(c["brief_p"][[0, 2]], a["brief_p"][[0, 2]])


def test_ties_at_the_edge_split_count():
    """One outlying first trial: every order that puts it first or last
    gives the same statistic, so p is about 2/T, not tiny."""
    rng = np.random.default_rng(9)
    T = 20
    x = rng.normal(size=(T, 1)) * 0.01
    x[0, 0] = 1.0
    r = trend_tests(x - x.mean(), np.ones(1, bool), np.arange(T), n_perm=4000)
    assert r["step_k"][0] == 1
    assert r["step_p"][0] == pytest.approx(2 / T, abs=0.02)


def test_noise_free_edge_outlier_ties_with_its_mirror():
    """A series without spread apart from one end trial has an infinite
    step statistic, equal for the series and its mirror image, so every
    order that puts the trial at either end ties (p about 2/T)."""
    x = np.zeros((2, 10))
    x[0, -1] = 1.0
    x[1, 0] = 1.0
    t, k, _ = step_stat(x)
    assert np.isinf(t).all() and t[0] > 0 > t[1]
    np.testing.assert_array_equal(k, [9, 1])
    r = trend_tests(x[:1].T, np.ones(1, bool), np.arange(10), n_perm=4000)
    assert r["step_p"][0] == pytest.approx(0.2, abs=0.03)
    # step_at agrees at the chosen split, also for a nearly noise-free step
    y = np.array([[0.0, 1e-6, 1.0, 1.000001]])
    t, k, d = step_stat(y)
    ta, da = step_at(y, k)
    assert np.isinf(t[0]) and k[0] == 2
    np.testing.assert_array_equal(ta, t)
    np.testing.assert_allclose(da, d)
    np.testing.assert_array_equal(step_at(x, [9, 1])[0], step_stat(x)[0])


def test_trend_tests_without_permutations():
    """n_perm 0 gives the statistics without p or q."""
    rng = np.random.default_rng(10)
    x = _session(rng, n_src=5, T=10)
    r = trend_tests(x, np.ones(5, bool), np.arange(10), n_perm=0)
    assert np.isfinite(r["step_t"]).all() and np.isnan(r["step_p"]).all()
    assert np.isnan(r["step_q"]).all()
    assert r["counts"]["step"]["fdr_up"] == 0


def test_trend_tests_need_three_trials():
    """Fewer than three trials: nothing is tested."""
    r = trend_tests(np.zeros((2, 3)), np.ones(3, bool), [1, 2], n_perm=10)
    assert not r["tested"].any() and np.isnan(r["step_t"]).all()


def test_trial_orders_are_seeded_permutations():
    """Each row is a permutation; the same seed and source repeat."""
    o = trial_orders(1, 7, 50, 9)
    assert (np.sort(o, axis=1) == np.arange(9)).all()
    np.testing.assert_array_equal(o, trial_orders(1, 7, 50, 9))
    assert not np.array_equal(o, trial_orders(1, 8, 50, 9))
