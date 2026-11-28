"""Solving on each trial's own pixel grid (``solve_grid='trial'``): the
layout of a trial's subproblems, footprints mapped back to the consensus
frame, ROIs moved into the trial, and the stages in this mode."""

from dataclasses import replace

import h5py
import numpy as np
import pytest

from giant_python.extraction.raster import trial_data as trial_data_mod
from giant_python.extraction.raster.alignment import (
    TrialWarp,
    mask_to_consensus,
)
from giant_python.extraction.raster.container import read_container
from giant_python.extraction.raster.plan import PlanParams, plan_path
from giant_python.extraction.raster.trial_grid import (
    check_solve_grid,
    footprints_to_consensus,
    masks_to_trial,
    trial_grid,
)
from giant_python.models.params import SiloParams
from giant_python.pipeline import silo, silo_collect, silo_solve
from giant_python.pipeline.silo_session import SiloSession
from giant_python.registration.outputs import matlab_round

from .test_raster_plan import _synthetic_trials
from .test_raster_silo_alignment import PATCHWISE
from .test_raster_silo_stages import (  # noqa: F401
    EX,
    _all_results,
    _equal,
    _summary,
    fake,
    fake_load,
    make_session,
)

TRIAL_GRID = replace(PATCHWISE, solve_grid="trial")


def _bent(n_trials, shape, seed=2, scale=0.8):
    """Per trial an affine map with a fractional shift and a bent field on
    top, as a patchwise map."""
    rng = np.random.default_rng(seed)
    nodes = rng.uniform([4, 4], [shape[0] - 4, shape[1] - 4], (8, 2))
    affine = np.zeros((n_trials, 2, 3))
    disp = np.zeros((n_trials, nodes.shape[0], 2))
    for t in range(n_trials):
        a = np.array(
            [[1.002, 0.003, 0.4 + 0.7 * t], [-0.002, 0.999, -1.3 + 0.2 * t]]
        )
        affine[t] = a
        disp[t] = (a[:, :2] @ nodes.T + a[:, 2:3] - nodes.T).T
        disp[t] += scale * rng.standard_normal(disp[t].shape)
    return TrialWarp(affine, nodes, 12.0, disp)


@pytest.fixture(scope="module")
def bent_plan():
    """The synthetic path's plan with a bent map per trial."""
    plan = plan_path(
        _synthetic_trials(),
        params=PlanParams(peak_th=12.0, inter_trial_alignment="rigid"),
    )
    assert len(plan.groups) > 2
    return replace(plan, warp=_bent(plan.n_trials, plan.shape))


def test_check_solve_grid():
    """The two grids, and nothing else."""
    assert check_solve_grid("trial") == "trial"
    with pytest.raises(ValueError, match="solve_grid"):
        check_solve_grid("trials")
    with pytest.raises(ValueError, match="solve_grid"):
        SiloParams(solve_grid="warped")


def test_layout_moves_each_subproblem_by_one_shift(bent_plan):
    """Each subproblem moves by the rounded mean displacement over its
    pixels; its crop, sources and their places in the crop stay; the grid's
    pixels are the union of the moved subproblems', in column-major order."""
    plan = bent_plan
    rows, cols = plan.selected_pixels()
    for t in range(plan.n_trials):
        g = trial_grid(plan, t)
        r_t, c_t = plan.warp.coords(t, rows.astype(float), cols.astype(float))
        np.testing.assert_allclose(g.mapped_rc, np.column_stack([r_t, c_t]))
        order = np.lexsort((g.points[:, 0], g.points[:, 1]))
        np.testing.assert_array_equal(order, np.arange(g.n_pixels))
        assert np.unique(g.points, axis=0).shape[0] == g.n_pixels
        src_r = plan.source_rows.astype(float)
        src_c = plan.source_cols.astype(float)
        sr, sc = plan.warp.coords(t, src_r, src_c)
        for k, (cons, moved) in enumerate(zip(plan.groups, g.groups)):
            pix, src = cons.pixels, cons.sources
            mean = [(sr - src_r)[src].mean(), (sc - src_c)[src].mean()]
            np.testing.assert_array_equal(g.shifts[k], matlab_round(mean))
            off = np.hypot(
                sr[src] - src_r[src] - g.shifts[k, 0],
                sc[src] - src_c[src] - g.shifts[k, 1],
            )
            assert g.residual[k] == pytest.approx(off.max())
            want = np.column_stack([rows[pix], cols[pix]]) + g.shifts[k]
            np.testing.assert_array_equal(g.points[moved.pixels], want)
            assert moved.sel is cons.sel
            np.testing.assert_array_equal(moved.sources, cons.sources)
            np.testing.assert_array_equal(moved.sources_rc, cons.sources_rc)
            assert (moved.row0, moved.col0) == (
                cons.row0 + g.shifts[k, 0],
                cons.col0 + g.shifts[k, 1],
            )
            src = np.column_stack(
                [
                    plan.source_rows[cons.sources],
                    plan.source_cols[cons.sources],
                ]
            )
            np.testing.assert_array_equal(
                g.sources_rc[cons.sources], src + g.shifts[k]
            )
        assert (g.residual < 1.0).all()
    # the plan's map differs between trials: so do the shifts
    shifts = [trial_grid(plan, t).shifts for t in range(plan.n_trials)]
    assert any(not np.array_equal(shifts[0], s) for s in shifts[1:])


def test_layout_refuses_a_map_that_stretches_a_subproblem(bent_plan):
    """One shift cannot serve a subproblem whose sources the map moves
    apart by more than about two pixels: the trial grid refuses it."""
    stretch = np.zeros((bent_plan.n_trials, 2, 3))
    stretch[:] = [[1.0, 0.0, 0.3], [0.0, 2.0, 0.0]]  # columns twice as far
    plan = replace(bent_plan, warp=TrialWarp(stretch))
    assert any(len(grp.sources) > 1 for grp in plan.groups)
    with pytest.raises(ValueError, match="consensus grid"):
        trial_grid(plan, 0)


def _trial_footprints(plan, grid, offset=(0.0, 0.0), sigma=1.3):
    """A Gaussian per source on its subproblem's pixels of the trial's
    grid, centred ``offset`` from the source's pixel within its 5 x 5
    support (each sums to 1), NaN outside the subproblem."""
    fp = np.full((grid.n_pixels, plan.n_sources), np.nan)
    for moved in grid.groups:
        pr, pc = grid.points[moved.pixels].T
        for s in moved.sources:
            c = grid.sources_rc[s] + np.asarray(offset)
            v = np.exp(-((pr - c[0]) ** 2 + (pc - c[1]) ** 2) / (2 * sigma**2))
            near = (np.abs(pr - grid.sources_rc[s, 0]) <= 2) & (
                np.abs(pc - grid.sources_rc[s, 1]) <= 2
            )
            v[~near] = 0.0
            fp[moved.pixels, s] = v / v.sum()
    return fp


def test_footprints_back_through_a_whole_pixel_shift(bent_plan):
    """With a whole-pixel shift the grid is the consensus grid moved: each
    footprint comes back unchanged."""
    shift = np.array([[2.0, -1.0]] * bent_plan.n_trials)
    plan = replace(bent_plan, warp=TrialWarp.from_shifts(shift))
    g = trial_grid(plan, 1)
    assert (g.shifts == [2, -1]).all() and np.allclose(g.residual, 0)
    fp = _trial_footprints(plan, g)
    back = footprints_to_consensus(fp, g, plan)
    assert back.dtype == np.float32
    for cons, moved in zip(plan.groups, g.groups):
        np.testing.assert_allclose(
            back[np.ix_(cons.pixels, cons.sources)],
            fp[np.ix_(moved.pixels, moved.sources)],
            rtol=1e-6,
        )
    outside = np.isnan(fp)  # pixels outside a source's subproblem stay NaN
    assert outside.sum() == np.isnan(back).sum()


def test_footprints_back_move_by_the_rest_of_the_map(bent_plan):
    """A footprint fitted on the trial's grid comes back where the map puts
    it: its centroid moves by the map's displacement minus the subproblem's
    shift, and its sum is kept."""
    plan = bent_plan
    rows, cols = plan.selected_pixels()
    for t in (0, 3):
        g = trial_grid(plan, t)
        fp = _trial_footprints(plan, g, offset=(0.2, -0.3))
        back = footprints_to_consensus(fp, g, plan)
        for cons, moved in zip(plan.groups, g.groups):
            for s in cons.sources:
                h_t = fp[moved.pixels, s]
                pr, pc = g.points[moved.pixels].T
                trial_c = np.array([h_t @ pr, h_t @ pc])
                h_c = back[cons.pixels, s].astype(float)
                assert h_c.sum() == pytest.approx(1.0, abs=1e-5)
                cons_c = np.array(
                    [h_c @ rows[cons.pixels], h_c @ cols[cons.pixels]]
                )
                # the map's inverse at the trial centroid
                want = np.array(plan.warp.inverse(t, *trial_c[:, None]))
                np.testing.assert_allclose(cons_c, want.ravel(), atol=0.03)


def test_footprints_back_keep_missing_sources(bent_plan):
    """A source without a footprint stays without one; the footprints'
    rows must be the grid's."""
    g = trial_grid(bent_plan, 2)
    fp = _trial_footprints(bent_plan, g)
    s = bent_plan.groups[0].sources[0]
    fp[:, s] = np.nan
    back = footprints_to_consensus(fp, g, bent_plan)
    assert np.isnan(back[:, s]).all()
    with pytest.raises(ValueError, match="pixels"):
        footprints_to_consensus(fp[:-1], g, bent_plan)


def test_masks_into_the_trial(bent_plan):
    """Each trial pixel takes the mask's value at the nearest pixel of its
    consensus position; a shift moves a mask by the rounded shift, the
    inverse of mask_to_consensus."""
    plan = bent_plan
    mask = np.zeros(plan.shape, bool)
    mask[20:31, 30:37] = True
    t = 1
    moved = masks_to_trial(
        [mask, np.zeros_like(mask)], plan.warp, t, plan.shape
    )
    assert not moved[1].any()
    rr, cc = np.meshgrid(
        np.arange(plan.shape[0]), np.arange(plan.shape[1]), indexing="ij"
    )
    x_r, x_c = plan.warp.inverse(t, rr.astype(float), cc.astype(float))
    x_r = np.floor(x_r + 0.5).astype(int)
    x_c = np.floor(x_c + 0.5).astype(int)
    ok = (
        (x_r >= 0) & (x_r < plan.shape[0]) & (x_c >= 0) & (x_c < plan.shape[1])
    )
    want = np.zeros(plan.shape, bool)
    want[ok] = mask[x_r[ok], x_c[ok]]
    np.testing.assert_array_equal(moved[0], want)
    shift = TrialWarp.from_shifts(np.array([[0.0, 0.0], [2.4, -1.6]]))
    trial_mask = masks_to_trial([mask], shift, 1, plan.shape)[0]
    np.testing.assert_array_equal(
        np.argwhere(trial_mask), np.argwhere(mask) + [2, -2]
    )
    np.testing.assert_array_equal(
        mask_to_consensus(trial_mask, shift, 1, plan.shape), mask
    )


def test_stage_settings_record_the_grid():
    """The consensus grid leaves every stage's settings as they were (files
    made before the option stay current); the trial grid enters the
    prepare and solve settings and the saved params."""
    base, trial = SiloParams(), SiloParams(solve_grid="trial")
    for stage in ("localize", "plan", "prepare", "solve", "collect"):
        assert "solve_grid" not in base.stage_settings(stage)
    assert trial.stage_settings("plan") == base.stage_settings("plan")
    for stage in ("prepare", "solve"):
        assert trial.stage_settings(stage)["solve_grid"] == "trial"
    assert "solveGrid" not in base.to_matlab(2, 80.0)
    assert trial.to_matlab(2, 80.0)["solveGrid"] == "trial"


def fake_load_points(*args, points=None, **kw):
    """:func:`fake_load` that also reads trial-frame points (in the
    column-major order the trial grid gives them)."""
    if points is None:
        return fake_load(*args, **kw)
    points = np.asarray(points)
    assert (points >= 0).all()
    sel = np.zeros(points.max(0) + 1, bool)
    sel[points[:, 0], points[:, 1]] = True
    args = args[:4] + (sel,) + args[5:]
    out = fake_load(*args, **kw)
    np.testing.assert_array_equal(out.pixel_rc, points)
    return out


@pytest.fixture
def grid_fake(fake, monkeypatch):  # noqa: F811
    """The fake session, with a loader that reads trial-frame points."""
    monkeypatch.setattr(trial_data_mod, "load_trial", fake_load_points)


def test_trial_grid_streamed_and_staged_runs_agree(tmp_path, grid_fake):
    """On the trial's grid the streaming driver and the stages give the
    same results and summaries; containers and results record the grid; the
    summary's footprints are in the consensus frame."""
    table = make_session(tmp_path / "streamed")
    silo.run_silo(table, TRIAL_GRID, EX)
    s1 = SiloSession.open(table)
    streamed = _all_results(s1)
    s2 = SiloSession.open(make_session(tmp_path / "staged"))
    silo.localize(s2, TRIAL_GRID, EX)
    silo.plan(s2, TRIAL_GRID, EX)
    silo.prepare(s2, TRIAL_GRID, EX)
    for job in silo_solve.dispatch(s2, TRIAL_GRID, 2, EX):
        silo_solve.solve_job(job, execution=EX)
    silo_collect.collect(s2, TRIAL_GRID, EX)
    staged = _all_results(s2)
    assert staged.keys() == streamed.keys() and staged
    for key in staged:
        for name, value in staged[key].items():
            np.testing.assert_array_equal(value, streamed[key][name])
    _equal(
        _summary(s2.out_dir / "experiment_summary.h5"),
        _summary(s1.out_dir / "experiment_summary.h5"),
    )
    c = read_container(s2.container_file(0, 0), data=False)
    assert c.params["solve_grid"] == "trial"
    files = sorted(s2.results_dir.glob("*.h5"))
    with h5py.File(files[0], "r") as f:
        assert f.attrs["solve_grid"] == "trial"
    with h5py.File(s2.out_dir / "experiment_summary.h5", "r") as f:
        prof = f["Path1/sources/spatial/profiles"]
        assert prof.shape[:2] == (30, 24)  # MATLAB order: (cols, rows, ...)
        assert np.isfinite(prof[()]).any()


def test_trial_grid_stages_refuse_the_other_grid(tmp_path, grid_fake):
    """Containers prepared on one grid are not solved on the other, and
    collect does not take results of the other grid."""
    s = SiloSession.open(make_session(tmp_path))
    silo.localize(s, TRIAL_GRID, EX)
    silo.plan(s, TRIAL_GRID, EX)
    silo.prepare(s, TRIAL_GRID, EX)
    consensus = replace(TRIAL_GRID, solve_grid="consensus")
    with pytest.raises(ValueError, match="solve_grid"):
        silo_solve.solve_containers(s, consensus, EX)
    for job in silo_solve.dispatch(s, TRIAL_GRID, 1, EX):
        silo_solve.solve_job(job, execution=EX)
    with pytest.raises(FileNotFoundError):
        silo_collect.collect(s, consensus, EX)
    silo_collect.collect(s, TRIAL_GRID, EX)


def test_skip_markers_belong_to_their_grid(tmp_path):
    """A no-data marker holds for its plan and grid only: the trial grid
    reads other pixels than the consensus grid. Markers written before the
    option are of the consensus grid."""
    from giant_python.pipeline.silo_session import skip_applies, write_skip

    f = tmp_path / "path1_trial0001.skip.json"
    assert not skip_applies(f, "p", "consensus")  # no marker
    write_skip(f, "no data", "h", "p")
    assert skip_applies(f, "p", "consensus")
    assert not skip_applies(f, "p", "trial")
    assert not skip_applies(f, "other plan", "consensus")
    write_skip(f, "no data", "h", "p", "trial")
    assert skip_applies(f, "p", "trial")
    assert not skip_applies(f, "p", "consensus")
    assert not skip_applies(f, "", "trial")
    f.write_text("not json")
    assert not skip_applies(f, "p", "trial")
