"""Rendering of SILo's activity in a trial's movie (giant view)."""

import numpy as np
import pytest

from giant_python.viewer.render import (
    TrialMap,
    frame_weights,
    kernel_dt,
    place_footprints,
)
from giant_python.viewer.run import SiloRun, footprint_width
from giant_python.viewer.trial import Session, load_trial, render_dt

from .synthetic import (
    CENTRES,
    H,
    N_TRIALS,
    W,
    gaussian,
    make_warp,
    write_session,
)


def _exporter_weights(lines200, disc, ds, dt, half):
    """The exporter's loop (exportF0TrainingData.m), as a dense matrix."""
    wk = np.zeros((lines200.size, ds.size))
    share = np.zeros(ds.size)
    for j in range(ds.size):
        d = lines200 - ds[j]
        sel = np.flatnonzero(np.abs(d) <= half)
        w = np.exp(-np.abs(d[sel]) / dt)
        share[j] = w[disc[sel]].sum() / max(w.sum(), np.finfo(float).eps)
        keep = ~disc[sel]
        if keep.any():
            wk[sel[keep], j] = w[keep] / w[keep].sum()
    return wk, share


def test_frame_weights_match_the_exporter():
    """Weights and discard shares equal the exporter's loop."""
    rng = np.random.default_rng(0)
    lines200 = np.ceil(1 + 53.4 * np.arange(400))
    disc = rng.random(400) < 0.1
    ds = 1.0 + 134 * np.arange(150) + 7
    w, share = frame_weights(lines200, disc, ds, 134, 402)
    ref, ref_share = _exporter_weights(lines200, disc, ds, 134, 402)
    np.testing.assert_allclose(w.toarray(), ref, atol=1e-12)
    np.testing.assert_allclose(share, ref_share, atol=1e-12)
    full = np.asarray(w.sum(0)).ravel()
    assert np.allclose(full[full > 0], 1.0)


def test_kernel_dt_and_matched_kernel():
    """dt from DSframes; the matched kernel removes the 200 Hz part."""
    assert kernel_dt(1 + 134 * np.arange(10), None, 2.5) == 134
    assert kernel_dt(np.array([5.0]), np.arange(0, 540, 54.0), 2.5) == 135
    lines = np.ceil(1 + 53.4 * np.arange(100))
    assert render_dt("exporter", 134, lines) == 134
    assert render_dt("matched", 134, lines) == pytest.approx(
        np.sqrt(134**2 - 54**2)
    )
    with pytest.raises(ValueError):
        render_dt("other", 134, lines)


@pytest.mark.parametrize("mode", ["rigid", "patchwise"])
def test_trial_map_inverse(mode):
    """The map's inverse undoes it (and agrees with TrialWarp's)."""
    warp = make_warp(mode)
    rng = np.random.default_rng(3)
    r, c = rng.uniform(3, H - 3, 200), rng.uniform(3, W - 3, 200)
    for t in range(N_TRIALS):
        field = None
        if mode == "patchwise":
            rt, ct = warp.grid_coords(t, (H, W))
            rr, cc = np.mgrid[:H, :W]
            field = np.stack([rt - rr, ct - cc])
        m = TrialMap(warp.affine[t], field)
        fr, fc = m.forward(r, c)
        er, ec = warp.coords(t, r, c)
        tol = 1e-9 if mode == "rigid" else 2e-3  # bilinear field
        np.testing.assert_allclose(fr, er, atol=tol)
        np.testing.assert_allclose(fc, ec, atol=tol)
        ir, ic = m.inverse(er, ec)
        wr, wc = warp.inverse(t, er, ec)
        np.testing.assert_allclose(ir, wr, atol=tol)
        np.testing.assert_allclose(ic, wc, atol=tol)


def test_affine_map_inverse_is_exact():
    """An affine map's inverse is exact."""
    a = np.array([[1.01, 0.02, 0.4], [-0.015, 0.99, -0.3]])
    m = TrialMap(a)
    r, c = np.array([3.0, 10.5]), np.array([7.0, 2.25])
    fr, fc = m.forward(r, c)
    ir, ic = m.inverse(fr, fc)
    np.testing.assert_allclose(ir, r, atol=1e-12)
    np.testing.assert_allclose(ic, c, atol=1e-12)


def _fp_tuple(img, s):
    """A canvas footprint image as ``(rows, cols, sources, values)``."""
    r, c = np.nonzero(img)
    return r, c, np.full(r.size, s), img[r, c]


@pytest.mark.parametrize("mode", ["rigid", "patchwise"])
def test_place_footprints_moves_them_through_the_map(mode):
    """A placed footprint lands at the mapped centroid; its sum changes
    only by the map's local area factor (a few percent for this bend)."""
    warp = make_warp(mode)
    parts = [_fp_tuple(gaussian((H, W), c), s) for s, c in enumerate(CENTRES)]
    fp = tuple(np.concatenate(x) for x in zip(*parts))
    rr, cc = np.mgrid[:H, :W]
    for t in range(N_TRIALS):
        field = None
        if mode == "patchwise":
            rt, ct = warp.grid_coords(t, (H, W))
            field = np.stack([rt - rr, ct - cc])
        m = TrialMap(warp.affine[t], field)
        h = place_footprints(fp, CENTRES.shape[0], m, (H, W)).toarray()
        for s in range(CENTRES.shape[0]):
            img = h[:, s].reshape(H, W)
            tol = 0.01 if mode == "rigid" else 0.05
            assert img.sum() == pytest.approx(1.0, abs=tol)
            cen = [(img * rr).sum() / img.sum(), (img * cc).sum() / img.sum()]
            want = warp.coords(t, *CENTRES[s])
            np.testing.assert_allclose(cen, np.ravel(want), atol=0.02)


def test_place_footprints_blur_keeps_the_sum():
    """A blurred footprint still sums to (about) 1."""
    m = TrialMap(np.array([[1.0, 0, 0.2], [0, 1.0, -0.1]]))
    fp = _fp_tuple(gaussian((H, W), CENTRES[1]), 0)
    h = place_footprints(fp, 1, m, (H, W), sigma=0.75).toarray()
    assert h.sum() == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize("mode", ["rigid", "patchwise"])
def test_rendered_activity_reproduces_the_movie(tmp_path, mode):
    """On a synthetic run whose movie is footprints x 80 Hz traces plus a
    baseline, the rendered activity has the movie's flux and centroids at
    every source (rigid and patchwise maps); pixel by pixel it differs by
    the bilinear interpolation of a 1.33 px footprint (up to ~12% at
    a half-pixel shift)."""
    syn = write_session(tmp_path, mode)
    run = SiloRun(syn["run"], "net")
    assert run.path(1).alignment == mode
    session = Session(syn["run"] / "trial_table.h5", syn["moco"], syn["base"])
    for t in range(N_TRIALS):
        tr = load_trial([run], session, 1, t)
        assert tr.layers == [
            "movie",
            "activity:net",
            "residual:net",
            "baseline",
            "residual-baseline:net",
        ]
        lay = tr.runs[0]
        act = np.asarray((lay.footprints @ lay.x80["exporter"]).T)
        signal = tr.movie - 10.0
        assert np.nanmax(np.abs(signal - act)) < 0.15 * np.nanmax(signal)
        f = tr.busiest_frame()
        for s in range(CENTRES.shape[0]):
            want = np.ravel(syn["warp"].coords(t, *CENTRES[s]))
            near = np.hypot(tr.rows - want[0], tr.cols - want[1]) < 4
            a, m = act[:, near].sum(1), signal[:, near].sum(1)
            np.testing.assert_allclose(a, m, atol=0.02 * m.max())
            for img in (act[f, near], signal[f, near]):
                cen = (
                    np.array(
                        [
                            (img * tr.rows[near]).sum(),
                            (img * tr.cols[near]).sum(),
                        ]
                    )
                    / img.sum()
                )
                np.testing.assert_allclose(cen, want, atol=0.03)
        fr = tr.frame(f, 4)
        assert fr.shape == (5, tr.N)
        np.testing.assert_allclose(fr[2], fr[0] - fr[1], atol=1e-4)
        tc, n = tr.trace(int(lay.kept_rc[0, 0]), int(lay.kept_rc[0, 1]), 1)
        assert n == 9 and tc.shape == (5, tr.T)
        np.testing.assert_allclose(tc[3], 10.0, atol=0.01)
        # the matched kernel is narrower: same totals, sharper peaks
        xm, xe = lay.x80["matched"], lay.x80["exporter"]
        np.testing.assert_allclose(xm.sum(1), xe.sum(1), rtol=0.02)
        assert (xm.max(1) > xe.max(1)).all()
        fm = tr.frame(f, 1, "matched")
        assert fm.shape == (5, tr.N) and not np.array_equal(fm[1], fr[1])


def test_run_reader(tmp_path):
    """Sources, images, traces and per-trial footprints of a run."""
    syn = write_session(tmp_path, "rigid")
    run = SiloRun(syn["run"], "r")
    p = run.path(1)
    assert p.shape == (H, W) and p.sources.n == 3
    np.testing.assert_allclose(p.sources.rc, CENTRES)
    assert p.sources.rejected_rc.tolist() == [[5.0, 5.0]]
    assert run.describe()["validKernel"] == 5
    tr = run.traces(1)
    assert tr["dF_denoised"].shape == (2 * syn["lines"].size, 3)
    fps = run.footprints(1)
    assert len(fps) == N_TRIALS and fps[0][3].sum() == pytest.approx(3.0)
    patch, r0, c0 = run.profile_patch(1, 0)
    assert patch.shape == (15, 15)
    w = footprint_width(gaussian((H, W), CENTRES[0]), *CENTRES[0])
    assert 1.2 < w < 1.4  # a truncated Gaussian of sigma 1.33


def test_movie_found_by_stem(tmp_path):
    """A table naming a TIFF finds the HDF5 movie, as SILo does."""
    import h5py

    syn = write_session(tmp_path, "rigid")
    table = syn["run"] / "trial_table.h5"
    with h5py.File(table, "r+") as f:
        names = f["motion_correction/fn_reg_ds"][()].astype(str)
        del f["motion_correction/fn_reg_ds"]
        f["motion_correction/fn_reg_ds"] = (
            np.char.replace(names, ".h5", ".tif")
            .astype(object)
            .astype(h5py.string_dtype())
        )
    session = Session(table, syn["moco"], syn["base"])
    assert session.has_movie(0, 0)
    assert session.movie_file(0, 0).suffix == ".h5"


def test_kernels_follow_each_runs_frame_lines(tmp_path):
    """Each run's matched kernel comes from its own 200 Hz frame lines."""
    syn = write_session(tmp_path, "rigid")
    session = Session(syn["run"] / "trial_table.h5", syn["moco"], syn["base"])
    a, b = SiloRun(syn["run"], "a"), SiloRun(syn["run"], "b")
    b.path(1).lines = (
        b.path(1).lines * 0 + np.arange(b.path(1).lines.size) * 30.0 + 1
    )  # another analysis rate: dt200 30 lines
    tr = load_trial([a, b], session, 1, 0)
    ka, kb = tr.runs[0].kernels["matched"][0], tr.runs[1].kernels["matched"][0]
    assert ka == pytest.approx(np.sqrt(134**2 - 54**2))
    assert kb == pytest.approx(np.sqrt(134**2 - 30**2))
    assert set(tr.info["render_dt_lines"]) == {"a", "b"}
