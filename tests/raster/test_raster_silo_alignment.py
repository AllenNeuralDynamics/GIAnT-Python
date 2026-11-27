"""The affine and patchwise across-trial alignments in giant silo's stages:
half images in the localization files, staged and streamed runs, the
plan's maps in the summaries, ROIs and per-trial peaks."""

from dataclasses import replace

import h5py
import numpy as np
import pytest

from giant_python.extraction.raster import trial as trial_mod
from giant_python.extraction.raster.alignment import (
    TrialWarp,
    mask_to_consensus,
    warp_trial_image,
)
from giant_python.extraction.raster.annotations import (
    PathAnnotations,
    UserRoi,
    rois_to_consensus,
)
from giant_python.extraction.raster.plan import (
    PlanParams,
    plan_path,
    read_plan,
    write_plan,
)
from giant_python.extraction.raster.stage_files import (
    read_localization,
    write_localization,
)
from giant_python.extraction.raster.summary import _peaks_to_consensus
from giant_python.pipeline import silo, silo_collect, silo_solve
from giant_python.pipeline.silo_session import SiloSession

from .test_raster_plan import _synthetic_trials
from .test_raster_silo_stages import (  # noqa: F401
    EX,
    N_PATHS,
    N_TRIALS,
    PARAMS,
    _all_results,
    _equal,
    _summary,
    fake,
    fake_localize,
    make_session,
)

PATCHWISE = replace(
    PARAMS, inter_trial_alignment="patchwise", align_patch_px=200
)


def _localize_with_halves(moco_dir, fn, num_channels, params=None, **kw):
    """fake_localize, with two noisy halves of the mean image when asked."""
    loc, reg = fake_localize(moco_dir, fn, num_channels, params, **kw)
    if kw.get("half_ims"):
        rng = np.random.default_rng(len(fn))
        noise = 0.05 * rng.standard_normal((2,) + reg.mean_im.shape)
        reg.half_ims = (reg.mean_im[None] + noise).astype(np.float32)
    return loc, reg


@pytest.fixture
def halves(fake, monkeypatch):  # noqa: F811
    """The fake session's localization with half images."""
    monkeypatch.setattr(trial_mod, "localize_trial", _localize_with_halves)


def test_patchwise_streamed_and_staged_runs_agree(tmp_path, halves):
    """Under patchwise alignment the streaming driver and the stages give
    the same results and summaries; the localization files hold the half
    images, the plans a patchwise field, the summary its displacements."""
    table = make_session(tmp_path / "streamed")
    silo.run_silo(table, PATCHWISE, EX)
    s1 = SiloSession.open(table)
    rec = read_localization(s1.localize_file(0, 0))
    assert rec.half_ims is not None and rec.attrs["half_ims"] == 1
    plan = read_plan(s1.plan_file(0))
    assert plan.warp.is_patchwise and plan.warp.patch is not None
    with h5py.File(s1.out_dir / "experiment_summary.h5", "r") as f:
        disp = f["Path1/frame_info/trial_alignment_displacement"][()]
        assert np.isfinite(disp).any()
    streamed = _all_results(s1)
    s2 = SiloSession.open(make_session(tmp_path / "staged"))
    silo.localize(s2, PATCHWISE, EX)
    silo.plan(s2, PATCHWISE, EX)
    silo.prepare(s2, PATCHWISE, EX)
    for job in silo_solve.dispatch(s2, PATCHWISE, 2, EX):
        silo_solve.solve_job(job, execution=EX)
    silo_collect.collect(s2, PATCHWISE, EX)
    staged = _all_results(s2)
    assert staged.keys() == streamed.keys()
    for key in staged:
        for name, value in staged[key].items():
            np.testing.assert_array_equal(value, streamed[key][name])
    _equal(
        _summary(s2.out_dir / "experiment_summary.h5"),
        _summary(s1.out_dir / "experiment_summary.h5"),
    )


def test_localize_redoes_files_without_halves(tmp_path, halves):
    """Files localized for a rigid plan lack the half images: a patchwise
    plan refuses them and the patchwise localize stage redoes them. Rigid
    and affine runs accept files with and without half images and redo
    neither, so existing localizations stay usable."""
    s = SiloSession.open(make_session(tmp_path))
    rigid = replace(PATCHWISE, inter_trial_alignment="rigid")
    affine = replace(PATCHWISE, inter_trial_alignment="affine")
    assert len(silo.localize(s, rigid, EX)) == N_PATHS * N_TRIALS
    assert read_localization(s.localize_file(0, 0)).half_ims is None
    for p in (rigid, affine):
        assert silo.localize_jobs(s, p) == []
        assert silo.plan_one(s, p, 0, overwrite=True).exists()
    with pytest.raises(ValueError, match="half images"):
        silo.plan_one(s, PATCHWISE, 0)
    jobs = silo.localize_jobs(s, PATCHWISE)
    assert len(jobs) == N_PATHS * N_TRIALS
    silo.localize(s, PATCHWISE, EX)
    assert silo.localize_jobs(s, PATCHWISE) == []
    for p in (rigid, affine):
        assert silo.localize_jobs(s, p) == []
        assert silo.plan_one(s, p, 0, overwrite=True).exists()
    assert silo.plan_one(s, PATCHWISE, 0).exists()


@pytest.mark.parametrize("mode", ["affine", "patchwise"])
def test_plan_with_a_refined_alignment(tmp_path, mode):
    """The valid trials get the refined maps (the same valid trials as
    rigid); the aligned images are warped through them; the shifts are the
    displacements at the canvas centre; the plan file keeps the maps."""
    trials = _synthetic_trials()
    for t in trials:
        t.half_ims = np.stack([t.mean_im, t.mean_im * 1.01])
    rigid = plan_path(trials, params=PlanParams(peak_th=12.0))
    details = {}
    plan = plan_path(
        trials,
        params=PlanParams(
            peak_th=12.0, inter_trial_alignment=mode, align_patch_px=600
        ),
        details=details,
    )
    np.testing.assert_array_equal(plan.valid_trials, rigid.valid_trials)
    assert np.isfinite(details["alignment_info"].corr[plan.valid_trials]).all()
    assert plan.warp.is_patchwise == (mode == "patchwise")
    h, w = plan.shape
    x0 = np.array([(h - 1) / 2, (w - 1) / 2])
    for t in plan.aligned_trials:
        r, c = plan.warp.coords(t, x0[:1], x0[1:])
        np.testing.assert_allclose(
            plan.shifts[t], [r[0] - x0[0], c[0] - x0[1]], atol=1e-12
        )
        np.testing.assert_array_equal(
            plan.mean_aligned[t],
            warp_trial_image(plan.mean_ims[t], plan.warp, t),
        )
    # the refinement moves the maps by well under a pixel here (pure
    # shifts), and the sources stay where they were
    assert np.abs(plan.shifts - rigid.shifts).max() < 0.2
    np.testing.assert_array_equal(plan.source_rows, rigid.source_rows)
    write_plan(tmp_path / "plan.h5", plan)
    back = read_plan(tmp_path / "plan.h5")
    np.testing.assert_array_equal(back.warp.affine, plan.warp.affine)
    np.testing.assert_array_equal(back.warp.disp, plan.warp.disp)
    assert back.warp.sigma == plan.warp.sigma or mode == "affine"
    if mode == "patchwise":
        np.testing.assert_array_equal(back.warp.patch, plan.warp.patch)
    assert back.params.align_patch_px == 600


def test_localization_file_keeps_half_images(tmp_path):
    """The half images round-trip as float32; files without them read as
    None."""
    rng = np.random.default_rng(0)
    mean = rng.random((5, 6, 2))
    halves = rng.random((2, 5, 6, 2))
    f = write_localization(
        tmp_path / "a.h5",
        mean,
        mean[..., 0],
        mean[..., 0],
        [0, 1],
        None,
        {"settings_hash": "x"},
        halves,
    )
    rec = read_localization(f)
    np.testing.assert_array_equal(rec.half_ims, halves.astype(np.float32))
    assert rec.trial_images().half_ims is rec.half_ims
    g = write_localization(
        tmp_path / "b.h5", mean, mean[..., 0], mean[..., 0], [0, 1], None, {}
    )
    assert read_localization(g).half_ims is None
    assert read_localization(g).attrs["half_ims"] == 0


def _bent_warp(n_trials=2, shape=(40, 50)):
    """An affine map with a bent field on top, per trial."""
    rng = np.random.default_rng(1)
    nodes = rng.uniform([5, 5], [shape[0] - 5, shape[1] - 5], (6, 2))
    affine = np.zeros((n_trials, 2, 3))
    disp = np.zeros((n_trials, 6, 2))
    for t in range(n_trials):
        a = np.array([[1.003, 0.002, 0.6 + t], [-0.002, 0.998, -1.1]])
        affine[t] = a
        disp[t] = (a[:, :2] @ nodes.T + a[:, 2:3] - nodes.T).T
        disp[t] += 0.3 * rng.standard_normal((6, 2))
    return TrialWarp(affine, nodes, 8.0, disp)


def test_rois_through_a_full_map():
    """Masks take the trial mask's value at each consensus pixel's mapped
    position; geometry goes through the inverse map (one-based [x, y])."""
    warp = _bent_warp()
    mask = np.zeros((40, 50), bool)
    mask[18:23, 20:26] = True
    center = np.array([23.0, 21.0])  # x (col), y (row), one-based
    ann = PathAnnotations(
        path=1,
        dr="",
        fn="E1T2DMD1",
        rois=[UserRoi("SOMA a", "circle", mask, center=center)],
    )
    rois = rois_to_consensus(
        ann, ["E1T1DMD1.tif", "E1T2DMD1.tif"], warp, (40, 50)
    )
    np.testing.assert_array_equal(
        rois[0].mask, mask_to_consensus(mask, warp, 1, (40, 50))
    )
    r, c = warp.coords(1, rois[0].center[1:] - 1, rois[0].center[:1] - 1)
    np.testing.assert_allclose([c[0] + 1, r[0] + 1], center, atol=1e-6)


def test_per_trial_peaks_into_the_consensus_frame():
    """Through the inverse of a non-shift map, minus the shift of a rigid
    one."""

    class Plan:
        """What _peaks_to_consensus reads of a plan."""

        def __init__(self, warp, shifts):
            """A warp and its shifts."""
            self.warp, self.shifts = warp, shifts

    peaks = np.array([[1.0, 10.0, 12.0, 0.5], [2.0, 30.5, 40.25, 0.5]])
    warp = _bent_warp()
    out = _peaks_to_consensus(Plan(warp, np.zeros((2, 2))), 1, peaks)
    r, c = warp.coords(1, out[:, 0], out[:, 1])
    np.testing.assert_allclose(
        np.column_stack([r, c]), peaks[:, 1:3], atol=1e-6
    )
    rigid = TrialWarp.from_shifts([[0.5, -1.0], [1.0, 2.0]])
    shifts = np.array([[0.5, -1], [1, 2]])
    out = _peaks_to_consensus(Plan(rigid, shifts), 1, peaks)
    np.testing.assert_array_equal(out, peaks[:, 1:3] - [1.0, 2.0])
