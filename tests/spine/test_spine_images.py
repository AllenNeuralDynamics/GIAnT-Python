"""Tests of the per-trial F0 images (:mod:`giant_python.spine.images`)."""

import warnings

import h5py
import numpy as np
import pytest

from giant_python.spine.images import (
    baseline_trial_mean,
    kept_frames,
    network_f0_stack,
    stack_key,
    to_consensus,
)
from giant_python.spine.summary_io import TrialMap

from . import synthetic as syn


def test_kept_frames_trims_both_ends():
    """The first and last round(trim_s * align_hz) frames are left out."""
    keep = kept_frames(np.zeros(100), align_hz=10.0, trim_s=0.5)
    assert not keep[:5].any() and not keep[-5:].any()
    assert keep[5:95].all()


def test_kept_frames_falls_back_when_too_short():
    """A trial too short to trim keeps its frames that are not discarded."""
    keep = kept_frames(np.zeros(10), align_hz=10.0, trim_s=1.0)
    assert keep.all()


def test_kept_frames_drops_motion_discards():
    """Frames SiLo discards for motion do not enter."""
    rne = np.zeros(400)
    rne[200:204] = 1.0  # a sharp rise of the registration error
    keep = kept_frames(rne, align_hz=80.0, trim_s=0.0)
    assert not keep[200:204].any()
    assert keep[:150].all()


def _baseline_file(path, frames):
    """Write ``frames`` ``(frame, rows, cols)`` as a baseline file."""
    with h5py.File(path, "w") as f:
        f["baseline"] = np.transpose(frames, (0, 2, 1)).astype(np.float32)


def test_baseline_trial_mean_matches_numpy(tmp_path):
    """The blockwise mean over kept frames equals numpy's, NaN kept out."""
    rng = np.random.default_rng(1)
    frames = rng.random((70, 6, 9))
    frames[:, 0, 0] = np.nan  # an invalid pixel
    path = tmp_path / "b.h5"
    _baseline_file(path, frames)
    keep = np.ones(70, bool)
    keep[::3] = False
    mean, n = baseline_trial_mean(path, keep, chunk=8)
    assert n == keep.sum()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # the NaN pixel
        expect = np.nanmean(frames[keep], axis=0)
    np.testing.assert_allclose(mean[1:, 1:], expect[1:, 1:], rtol=1e-6)
    assert np.isnan(mean[0, 0])


def test_baseline_trial_mean_checks_frame_count(tmp_path):
    """A frame mask of the wrong length is refused."""
    path = tmp_path / "b.h5"
    _baseline_file(path, np.ones((10, 3, 3)))
    with pytest.raises(ValueError, match="frame flags"):
        baseline_trial_mean(path, np.ones(9, bool))


def test_to_consensus_undoes_a_shift():
    """A shifted trial image lands back on the consensus scene."""
    tmap = TrialMap(
        np.array([[[1, 0, 0], [0, 1, 0]], [[1, 0, 2], [0, 1, -1]]], float)
    )
    scene = syn.scene(1)
    trial = np.full_like(scene, np.nan)
    trial[2:, :-1] = scene[:-2, 1:]  # trial pixel = consensus pixel + (2, -1)
    out = to_consensus(trial, tmap, 1, scene.shape)
    ok = np.isfinite(out)
    assert ok.sum() > 0.8 * ok.size
    np.testing.assert_allclose(out[ok], scene[ok])


def test_to_consensus_places_small_trials_top_left():
    """A trial smaller than the frame is padded at the bottom right."""
    tmap = TrialMap(np.array([[[1, 0, 0], [0, 1, 0]]], float))
    img = np.arange(12.0).reshape(3, 4)
    out = to_consensus(img, tmap, 0, (5, 6))
    np.testing.assert_array_equal(out[:3, :4], img)
    assert np.isnan(out[3:]).all() and np.isnan(out[:, 4:]).all()


def test_network_f0_stack_recovers_the_scene(tmp_path):
    """Per-trial F0 images equal the scaled scene where the trial covers
    it, and the trimmed frames do not enter."""
    s = syn.make_session(tmp_path)
    tmap = TrialMap(
        np.stack([[[1, 0, dr], [0, 1, dc]] for dr, dc in syn.SHIFTS]).astype(
            float
        )
    )
    stems = [f"E1T{t + 1}DMD1" for t in range(syn.N_TRIALS)]
    st = network_f0_stack(
        s["baselines"], s["moco"], stems, tmap, (syn.H, syn.W), trim_s=0.1
    )
    assert st.images.shape == (syn.N_TRIALS, syn.H, syn.W)
    assert (st.n_frames == syn.N_FRAMES - 16).all()
    for t in range(syn.N_TRIALS):
        expect = syn.consensus_trial(1, t)
        ok = np.isfinite(st.images[t]) & np.isfinite(expect)
        assert ok.sum() > 0.7 * ok.size
        # float16 storage of the baseline
        np.testing.assert_allclose(st.images[t][ok], expect[ok], rtol=1e-3)
    assert st.attrs["checkpoint"] == "synthetic.pt"
    assert st.inputs[0]["baseline"]["name"] == "E1T1DMD1_BASELINE-80Hz.h5"


def test_network_f0_stack_skips_missing_trials(tmp_path):
    """Trials without a stem, a baseline file or a map stay NaN."""
    s = syn.make_session(tmp_path)
    affine = np.zeros((syn.N_TRIALS, 2, 3))
    affine[:, :, :2] = np.eye(2)
    affine[2] = np.nan  # trial 3 not aligned
    stems = ["E1T1DMD1", None, "E1T3DMD1", "missing"]
    st = network_f0_stack(
        s["baselines"], s["moco"], stems, TrialMap(affine), (syn.H, syn.W)
    )
    assert np.isfinite(st.images[0]).any()
    assert all(np.isnan(st.images[t]).all() for t in (1, 2, 3))
    assert list(st.n_frames[1:]) == [0, 0, 0]


def test_stack_key_changes_with_inputs():
    """The cache key depends on every part."""
    a = stack_key("x", {"trim_s": 1.0}, np.zeros(3))
    assert a == stack_key("x", {"trim_s": 1.0}, np.zeros(3))
    assert a != stack_key("x", {"trim_s": 0.5}, np.zeros(3))
    assert a != stack_key("x", {"trim_s": 1.0}, np.ones(3))
