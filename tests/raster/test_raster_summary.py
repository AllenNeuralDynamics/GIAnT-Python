"""Tests of SILo's summaries in GIAnT-MATLAB's schema
(saveExperimentSummaryH5, savePerTrialSummaryH5)."""

import h5py
import numpy as np
import pytest

from giant_python.extraction.raster.alignment import TrialWarp
from giant_python.extraction.raster.kinetics import KineticsTest
from giant_python.extraction.raster.plan import (
    ExtractionPlan,
    PlanParams,
    group_sources,
)
from giant_python.extraction.raster.results import TrialOutcome, TrialRecord
from giant_python.extraction.raster.solver import TrialActivity
from giant_python.extraction.raster.summary import (
    AlignmentRecord,
    PathResult,
    _nearest,
    kinetics_test,
    write_summaries,
)
from giant_python.io.hdf5 import load_struct_h5
from giant_python.models.params import SiloParams

H, W, C = 14, 18, 2
N_TRIALS = 3
FRAMES = {1: 30, 2: 40}  # trial 0 is not valid
SHIFTS = np.array([[np.nan, np.nan], [0.0, 0.0], [1.0, -2.0]])


def make_plan():
    """A plan of four sources on a 14 x 18 frame, three trials."""
    rng = np.random.default_rng(0)
    rows, cols = np.array([3, 4, 9, 10]), np.array([3, 5, 12, 15])
    sel = np.zeros((H, W), bool)
    for r, c in zip(rows, cols):
        sel[max(r - 2, 0) : r + 3, max(c - 2, 0) : c + 3] = True
    groups = group_sources(rows, cols, sel, 3, 6)
    mean_aligned = rng.random((N_TRIALS, H, W, C)) + 1
    act_aligned = rng.random((N_TRIALS, H, W))
    act_aligned[0] = np.nan
    mean_aligned[0] = np.nan
    sel_cols, sel_rows = np.nonzero(sel.T)
    lists = []
    for r, c in zip(rows, cols):
        near = (abs(sel_rows - r) <= 2) & (abs(sel_cols - c) <= 2)
        lists.append(np.flatnonzero(near))
    return ExtractionPlan(
        params=PlanParams(),
        keep_trials=np.ones(N_TRIALS, bool),
        template=np.zeros((H + 10, W + 10)),
        template_trials=np.arange(N_TRIALS),
        shifts=SHIFTS,
        corr=np.array([np.nan, 0.9, 0.95]),
        corr_thresh=0.5,
        act_valid_pix=np.ones(N_TRIALS),
        valid_trials=np.array([1, 2]),
        warp=TrialWarp.from_shifts(SHIFTS),
        mean_ims=mean_aligned.copy(),
        act_ims=act_aligned.copy(),
        mean_aligned=mean_aligned,
        act_aligned=act_aligned,
        mean_im=np.nanmean(mean_aligned[1:], axis=0),
        act_im=rng.random((H, W)),
        soma_mask=np.zeros((H, W), bool),
        px_always_valid=np.ones((H, W), bool),
        seeds=np.zeros((4, 4)),
        source_rows=rows,
        source_cols=cols,
        source_values=np.ones(4),
        sel=sel,
        source_pixels=lists,
        groups=groups,
        trial_peaks=[None, None, None],
        trial_discard=[None, None, None],
    )


def outcome(t, n_px, extracted=True):
    """A trial's outcome with distinct values per source and channel."""
    n = FRAMES[t]
    rng = np.random.default_rng(t)
    rec = TrialRecord(
        frame_lines=100 * t + np.arange(n) * 8 + 1,
        discard=np.arange(n) % 7 == 0,
        global_f=rng.random((C, n)),
        roi_f=rng.random((2, C, n)),
        roi_fsvd=rng.random((2, C, n)),
    )
    if not extracted:
        return TrialOutcome(rec)
    fp = rng.random((n_px, 4)).astype(np.float32)
    fp[0, 1] = np.nan  # a pixel outside source 1's subproblem
    base = np.arange(4)[None, :, None] * 10 + np.arange(C)[:, None, None]
    trace = base + rng.random((C, 4, n))
    act = TrialActivity(
        footprints=fp,
        events=trace + 0.1,
        denoised=trace + 0.2,
        ls=trace,
        f0=trace + 0.3,
        snr=np.array([1.0, 2.0, np.nan, 4.0]) * t,
    )
    return TrialOutcome(rec, act)


def alignment(t):
    """A registration record with 5 registered frames over the trial."""
    ds = np.linspace(100 * t + 1, 100 * t + 1 + FRAMES[t] * 8, 5)
    return AlignmentRecord(
        ds_frames=ds,
        motion_r=np.linspace(0, 1, 5),
        motion_c=np.linspace(1, 0, 5),
        online_x=np.arange(5.0),
        online_y=-np.arange(5.0),
        online_z=np.zeros(5),
        z_depth=-18.75,
    )


@pytest.fixture
def written(tmp_path):
    """Both summaries of one path (activity channel 2)."""
    plan = make_plan()
    n_px = plan.n_pixels
    result = PathResult(
        plan=plan,
        trials=[None, outcome(1, n_px), lambda: outcome(2, n_px)],
        alignment=[None, alignment(1), alignment(2)],
        roi_labels=["spine01", "soma"],
        roi_masks=[np.eye(H, W, dtype=bool), np.ones((H, W), bool)],
        z_depth=-18.75,
        trial_peaks=[
            np.array([[1.0, 2.0, 3.0, 0.5]]),
            None,
            np.array([[1.0, 5.0, 6.0, 0.5], [2.0, 7.0, 8.0, 0.4]]),
        ],
    )
    result.kinetics = KineticsTest(
        keep=np.array([True, False, True, True]),
        score=np.array([0.9, 0.1, np.nan, 0.5]),
        n_events=np.array([10, 7, 2, 6]),
    )
    params = SiloParams(
        activity_channel=2,
        footprint_variance="model",
        seed_fit="matlab",
        seed_merge_px=0.0,
    ).to_matlab(C, 80.0, 4, "a", "b")
    exp, per = tmp_path / "experiment_summary.h5", tmp_path / "per.h5"
    write_summaries(exp, per, [result, None], params, N_TRIALS, C, 2)
    return exp, per, result


def sizes(path):
    """Every dataset's MATLAB size and class."""
    out = {}
    with h5py.File(path, "r") as f:

        def visit(name, obj):
            """Record a dataset."""
            if isinstance(obj, h5py.Dataset):
                out[name] = (
                    obj.shape[::-1],
                    obj.dtype.kind,
                    obj.dtype.itemsize,
                )

        f.visititems(visit)
    return out


def test_experiment_summary_schema(written):
    """Names, MATLAB sizes and classes of saveExperimentSummaryH5."""
    exp, _, _ = written
    s = sizes(exp)
    total = sum(FRAMES.values())
    want = {
        "Path1/Z_depths": ((1, 1), "f", 8),
        "Path1/frame_info/trial_num_frames": ((N_TRIALS, 1), "i", 4),
        "Path1/frame_info/offlineXshifts": ((total, 1), "f", 8),
        "Path1/frame_info/offlineYshifts": ((total, 1), "f", 8),
        "Path1/frame_info/frame_line_idxs": ((total, 1), "i", 4),
        "Path1/frame_info/discard_frames": ((total, 1), "i", 1),
        "Path1/frame_info/onlineXshifts": ((total, 1), "f", 8),
        "Path1/frame_info/onlineYshifts": ((total, 1), "f", 8),
        "Path1/frame_info/onlineZshifts": ((total, 1), "f", 8),
        "Path1/frame_info/trial_alignment_affine": ((N_TRIALS, 2, 3), "f", 8),
        "Path1/visualizations/mean_im": ((C, 1, H, W), "f", 8),
        "Path1/visualizations/act_im": ((1, H, W), "f", 8),
        "Path1/visualizations/act_im_peaks": ((3, 3), "f", 8),
        "Path1/global/F": ((C, total), "f", 8),
        "Path1/user_rois/labels": ((2, 1), "O", 8),
        "Path1/user_rois/mask": ((2, 1, H, W), "u", 1),
        "Path1/user_rois/F": ((2, C, total), "f", 8),
        "Path1/user_rois/Fsvd": ((2, C, total), "f", 8),
        "Path1/sources/rejected/seed_coords": ((1, 3), "f", 8),
        "Path1/sources/rejected/kinetic_score": ((1, 1), "f", 8),
        "Path1/sources/rejected/kinetic_n_events": ((1, 1), "i", 4),
        "Path1/sources/spatial/profiles": ((3, 1, H, W), "f", 4),
        "Path1/sources/spatial/coords": ((3, 3), "f", 8),
        "Path1/sources/temporal/dF_ls": ((3, C, total), "f", 8),
        "Path1/sources/temporal/dF_denoised": ((3, C, total), "f", 8),
        "Path1/sources/temporal/events": ((3, C, total), "f", 8),
        "Path1/sources/temporal/F0": ((3, C, total), "f", 8),
        "Path1/sources/temporal/SNR": ((3, 1), "f", 8),
        "Path1/sources/temporal/kinetic_score": ((3, 1), "f", 8),
        "Path1/sources/temporal/kinetic_n_events": ((3, 1), "i", 4),
        "row_major": ((1, 1), "u", 1),
    }
    params = {k for k in s if k.startswith("params/")}
    assert len(params) == 48
    assert {k: v for k, v in s.items() if k not in params} == want


def test_experiment_summary_values(written):
    """Traces at their trials' frames in file channel order; kept sources
    only; frame records; ROI traces reordered; profiles averaged."""
    exp, _, result = written
    e = load_struct_h5(exp)["Path1"]
    o1, o2 = result.trials[1], result.trials[2]()
    kept = [0, 2, 3]
    ls = e["sources"]["temporal"]["dF_ls"]
    # activity channel 2 is analysis channel 0
    np.testing.assert_array_equal(ls[:, 1, :30], o1.activity.ls[0][kept])
    np.testing.assert_array_equal(ls[:, 0, 30:], o2.activity.ls[1][kept])
    np.testing.assert_array_equal(
        e["sources"]["temporal"]["F0"][:, 1, 30:], o2.activity.f0[0][kept]
    )
    np.testing.assert_array_equal(
        e["sources"]["temporal"]["kinetic_score"].ravel(), [0.9, np.nan, 0.5]
    )
    np.testing.assert_array_equal(
        e["sources"]["temporal"]["SNR"].ravel(), [1.5, np.nan, 6.0]
    )
    np.testing.assert_array_equal(
        e["sources"]["rejected"]["seed_coords"], [[0, 4, 5]]
    )
    np.testing.assert_array_equal(
        e["visualizations"]["act_im_peaks"],
        [[0, 3, 3], [0, 9, 12], [0, 10, 15]],
    )
    fi = e["frame_info"]
    np.testing.assert_array_equal(fi["trial_num_frames"].ravel(), [0, 30, 40])
    np.testing.assert_array_equal(
        fi["frame_line_idxs"].ravel(),
        np.r_[o1.record.frame_lines, o2.record.frame_lines],
    )
    np.testing.assert_array_equal(
        fi["discard_frames"].ravel().astype(bool),
        np.r_[o1.record.discard, o2.record.discard],
    )
    # trial 2's shift is added to its offline motion
    a = alignment(2)
    first = np.interp(o2.record.frame_lines[0], a.ds_frames, a.motion_c)
    assert fi["offlineXshifts"].ravel()[30] == pytest.approx(first - 2.0)
    assert "offlineZshifts" not in fi
    np.testing.assert_array_equal(
        fi["trial_alignment_affine"][2], [[1, 0, 1], [0, 1, -2]]
    )
    np.testing.assert_array_equal(e["global"]["F"][:, :30], o1.record.global_f)
    roi_f = e["user_rois"]["F"]
    np.testing.assert_array_equal(roi_f[:, 1, :30], o1.record.roi_f[:, 0])
    np.testing.assert_array_equal(roi_f[:, 0, :30], o1.record.roi_f[:, 1])
    assert list(e["user_rois"]["labels"].ravel()) == ["spine01", "soma"]
    # profiles: mean of the trials' footprints, NaN as 0
    rows, cols = result.plan.selected_pixels()
    fp = (
        np.nan_to_num(o1.activity.footprints[:, kept])
        + np.nan_to_num(o2.activity.footprints[:, kept])
    ) / 2
    prof = e["sources"]["spatial"]["profiles"]
    np.testing.assert_allclose(prof[:, 0, rows, cols].T, fp, rtol=1e-6)
    np.testing.assert_allclose(
        e["sources"]["spatial"]["coords"][:, 1],
        (fp * rows[:, None]).sum(0) / fp.sum(0),
        rtol=1e-5,
    )
    np.testing.assert_array_equal(
        e["visualizations"]["mean_im"][:, 0],
        result.plan.mean_im.transpose(2, 0, 1),
    )


def test_per_trial_summary(written):
    """Per-trial images, peaks in the consensus frame, footprints."""
    _, per, result = written
    s = sizes(per)
    assert s["Path1/visualizations/per_trial_mean_im"][0] == (
        N_TRIALS,
        C,
        1,
        H,
        W,
    )
    assert s["Path1/visualizations/per_trial_act_im"][0] == (N_TRIALS, 1, H, W)
    assert s["Path1/visualizations/per_trial_act_im_peaks"][0] == (
        N_TRIALS,
        2,
        3,
    )
    assert s["Path1/sources/spatial/per_trial_profiles"][0] == (
        N_TRIALS,
        3,
        1,
        H,
        W,
    )
    assert s["Path1/sources/temporal/per_trial_SNR"][0] == (N_TRIALS, 3)
    p = load_struct_h5(per)["Path1"]
    v = p["visualizations"]
    assert np.isnan(v["per_trial_mean_im"][0]).all()
    np.testing.assert_array_equal(
        v["per_trial_mean_im"][2, :, 0],
        result.plan.mean_aligned[2].transpose(2, 0, 1),
    )
    np.testing.assert_array_equal(v["per_trial_num_peaks"].ravel(), [1, 0, 2])
    peaks = v["per_trial_act_im_peaks"]
    # trial 0 is not aligned (NaN shift); trial 2 moves by minus its shift
    assert np.isnan(peaks[0, 0, 1:]).all() and peaks[0, 0, 0] == 0
    np.testing.assert_array_equal(peaks[2], [[0, 4, 8], [0, 6, 10]])
    assert np.isnan(peaks[1]).all()
    prof = p["sources"]["spatial"]["per_trial_profiles"]
    assert np.isnan(prof[0]).all()
    rows, cols = result.plan.selected_pixels()
    fp = np.nan_to_num(result.trials[1].activity.footprints[:, [0, 2, 3]])
    np.testing.assert_array_equal(prof[1, :, 0, rows, cols], fp)
    snr = p["sources"]["temporal"]["per_trial_SNR"]
    np.testing.assert_array_equal(snr[1], [1, np.nan, 4])
    coords = p["sources"]["spatial"]["per_trial_coords"]
    assert (coords[:, :, 0] == 0).all() and np.isnan(coords[0, :, 1:]).all()


def test_short_trial_and_no_peaks(tmp_path):
    """A trial loaded but not extracted adds frames, no sources; without
    per-trial peaks the peak fields are left out."""
    plan = make_plan()
    result = PathResult(
        plan=plan,
        trials=[None, outcome(1, plan.n_pixels, extracted=False), None],
        alignment=[None, alignment(1), None],
    )
    kinetics_test(result, 0.024, 200.0, 0.2)
    exp, per = tmp_path / "e.h5", tmp_path / "p.h5"
    write_summaries(exp, per, [result], {"x": 1.0}, N_TRIALS, C, 1)
    e = load_struct_h5(exp)
    assert "sources" not in e["Path1"]
    assert e["Path1"]["frame_info"]["trial_num_frames"].ravel().tolist() == [
        0,
        30,
        0,
    ]
    assert e["Path1"]["global"]["F"].shape == (C, 30)
    assert "user_rois" not in e["Path1"]
    assert "Z_depths" not in e["Path1"]
    p = load_struct_h5(per)["Path1"]
    assert "per_trial_num_peaks" not in p["visualizations"]
    assert "sources" not in p


def test_trial_without_sources_is_nan(tmp_path):
    """Frames of a trial loaded but not extracted are NaN in the sources'
    traces when other trials have sources."""
    plan = make_plan()
    result = PathResult(
        plan=plan,
        trials=[
            None,
            outcome(1, plan.n_pixels, extracted=False),
            outcome(2, plan.n_pixels),
        ],
        alignment=[None, alignment(1), alignment(2)],
    )
    kinetics_test(result, 0.024, 200.0, 0.0)
    exp, per = tmp_path / "e.h5", tmp_path / "p.h5"
    write_summaries(exp, per, [result], {"x": 1.0}, N_TRIALS, C, 1)
    temporal = load_struct_h5(exp)["Path1"]["sources"]["temporal"]
    for key in ("dF_ls", "dF_denoised", "events", "F0"):
        assert np.isnan(temporal[key][:, :, :30]).all()
        assert not np.isnan(temporal[key][:, 0, 30:]).all()
    snr = load_struct_h5(per)["Path1"]["sources"]["temporal"]["per_trial_SNR"]
    assert np.isnan(snr[1]).all()


def test_nearest_interpolation():
    """MATLAB's interp1 'nearest' with 'extrap': ties go right."""
    ds = np.array([1.0, 5.0, 9.0])
    v = np.array([10.0, 20.0, 30.0])
    np.testing.assert_array_equal(
        _nearest(ds, v, [0, 2, 3, 4, 7, 12]), [10, 10, 20, 20, 30, 30]
    )
    np.testing.assert_array_equal(_nearest(ds[:1], v[:1], [0, 4]), [10, 10])


def test_evidence_test_removes_weak_sources(tmp_path):
    """With min_snr the sources whose SNR averaged over trials is below it
    are removed with the kinetics test's, and marked in the summary; the
    SNR of a source without one (NaN) does not judge it."""
    plan = make_plan()
    result = PathResult(
        plan=plan,
        trials=[None, outcome(1, plan.n_pixels), outcome(2, plan.n_pixels)],
        alignment=[None, alignment(1), alignment(2)],
    )
    kin = kinetics_test(result, 0.024, 200.0, 0.0, min_snr=3.5)
    # SNR per trial [1, 2, nan, 4] * t: means 1.5, 3, nan, 6
    np.testing.assert_allclose(kin.snr, [1.5, 3.0, np.nan, 6.0])
    np.testing.assert_array_equal(kin.weak, [True, True, False, False])
    np.testing.assert_array_equal(kin.keep, [False, False, True, True])
    exp, per = tmp_path / "e.h5", tmp_path / "p.h5"
    write_summaries(exp, per, [result], {"x": 1.0}, N_TRIALS, C, 1)
    rej = load_struct_h5(exp)["Path1"]["sources"]["rejected"]
    np.testing.assert_array_equal(rej["weak"].ravel(), [1, 1])
    np.testing.assert_allclose(rej["SNR"].ravel(), [1.5, 3.0])
    off = kinetics_test(result, 0.024, 200.0, 0.0)
    assert off.weak is None and off.keep.all()


def test_evidence_events_skipped_without_counts():
    """The event criterion judges nothing when the kinetics test cannot
    count events (decay under 3 frames); the SNR criterion still does."""
    plan = make_plan()
    result = PathResult(
        plan=plan,
        trials=[None, outcome(1, plan.n_pixels), outcome(2, plan.n_pixels)],
        alignment=[None, alignment(1), alignment(2)],
    )
    kin = kinetics_test(result, 0.005, 200.0, 0.0, min_events=3)
    assert not kin.weak.any() and kin.keep.all()
    kin = kinetics_test(result, 0.005, 200.0, 0.0, 3.5, 3)
    np.testing.assert_array_equal(kin.weak, [True, True, False, False])
