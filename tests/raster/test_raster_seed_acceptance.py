"""Tests of the seed threshold set from the combined image's null
(``seed_acceptance='null'``, GIAnT-Python#51) and the null images that
carry it from the localize stage to the plan stage."""

import numpy as np
import pytest
from scipy import ndimage

from giant_python.extraction.raster.peaks import get_act_im_peaks, ordfilt3x3
from giant_python.extraction.raster.plan import (
    PlanParams,
    TrialImages,
    combined_activity_image,
    combined_null_image,
    plan_path,
)
from giant_python.extraction.raster.seeding import (
    candidate_statistic,
    null_background,
    null_threshold,
    seed_sources,
)
from giant_python.extraction.raster.stage_files import (
    read_localization,
    write_localization,
)
from giant_python.models.params import SiloParams


def _noise_field(rng, shape=(120, 140), sigma=1.3):
    """Smoothed Gaussian noise: a stand-in for a null activity image."""
    return ndimage.gaussian_filter(rng.standard_normal(shape), sigma)


def test_candidate_statistic_is_the_seeding_rule():
    """Maxima whose second-largest neighbour exceeds a threshold are exactly
    get_act_im_peaks' candidates."""
    rng = np.random.default_rng(0)
    im = _noise_field(rng)
    im[:3] = np.nan
    excl = np.zeros(im.shape, bool)
    excl[50:60, 50:60] = True
    rc, value = candidate_statistic(im, excl)
    th = np.quantile(value, 0.9)
    explored = im * ~excl
    with np.errstate(invalid="ignore"):
        cand = (ordfilt3x3(explored, 8) > th) & (
            explored == ordfilt3x3(explored, 9)
        )
    mine = np.zeros(im.shape, bool)
    mine[tuple(rc[value > th].T)] = True
    np.testing.assert_array_equal(mine, cand)
    assert not excl[tuple(rc.T)].any()


def test_null_threshold_gives_the_expected_false_seeds():
    """A threshold set on one null field for 2 false candidates per field
    is crossed about twice per field on independent fields of the same
    noise; a lax target is read off the null, a strict one extrapolated."""
    rng = np.random.default_rng(1)
    fields = [_noise_field(rng) for _ in range(41)]
    (mu, sd), th, info = null_threshold(np.stack(fields[:4]), 2.0)
    assert info["extrapolated"] and info["draws"] == 4
    counts = [
        np.sum((candidate_statistic(f)[1] - mu) / sd > th) for f in fields[4:]
    ]
    assert 1.0 < np.mean(counts) < 4.0
    _, th_lax, info_lax = null_threshold(fields[0], 200.0)
    assert not info_lax["extrapolated"]
    assert (
        np.sum((candidate_statistic(fields[0])[1] - mu) / sd > th_lax) <= 210
    )
    _, th_strict, _ = null_threshold(fields[0], 0.1)
    assert th_strict > th_lax
    with pytest.raises(ValueError):
        null_threshold(fields[0], 0.0)
    with pytest.raises(ValueError):
        null_threshold(np.zeros((20, 20)), 1.0)


def test_background_override():
    """The image's own median and robust SD as the background reproduce
    MATLAB's threshold; a doubled scale with half the threshold too."""
    rng = np.random.default_rng(2)
    im = _noise_field(rng, (60, 60)) * 0.2
    for r, c in ((15, 15), (30, 44), (45, 20)):
        im[r - 2 : r + 3, c - 2 : c + 3] += 3 * np.exp(
            -np.add.outer(np.arange(-2, 3) ** 2, np.arange(-2, 3) ** 2) / 2
        )
    v = im[np.isfinite(im)]
    mu = np.median(v)
    sd = np.median(np.abs(v - mu)) / 0.6741891400433162
    ref = get_act_im_peaks(im, 5.0, fit="local")
    same = get_act_im_peaks(im, 5.0, fit="local", background=(mu, sd))
    half = get_act_im_peaks(im, 2.5, fit="local", background=(mu, 2 * sd))
    np.testing.assert_allclose(same, ref)
    np.testing.assert_allclose(half, ref)
    assert ref.shape[0] == 3
    with pytest.raises(ValueError):
        get_act_im_peaks(im, 5.0, background=(0.0, 0.0))


def test_seed_sources_with_the_null():
    """acceptance='null' seeds at the null's threshold and reports it; it
    needs the null image."""
    rng = np.random.default_rng(3)
    im = _noise_field(rng, (80, 80))
    g = np.exp(-np.add.outer(np.arange(-3, 4) ** 2, np.arange(-3, 4) ** 2) / 4)
    im[37:44, 37:44] += 3.0 * g  # a source well above the noise (SD ~0.2)
    null = np.stack([_noise_field(rng, (80, 80)) for _ in range(2)])
    info = {}
    seeds = seed_sources(
        im,
        99.0,
        fit="local",
        acceptance="null",
        null_im=null,
        false_per_field=1.0,
        info=info,
    )
    bg, th, _ = null_threshold(null, 1.0)
    np.testing.assert_allclose(
        seeds, get_act_im_peaks(im, th, fit="local", background=bg)
    )
    assert info["peak_th"] == pytest.approx(th) and info["draws"] == 2
    assert np.hypot(*(seeds[:, 1:3] - 40).T).min() < 1.0
    with pytest.raises(ValueError):
        seed_sources(im, 3.0, acceptance="null")
    with pytest.raises(ValueError):
        seed_sources(im, 3.0, acceptance="robust")
    info = {}
    seed_sources(im, 3.0, fit="local", info=info)
    assert info == {"peak_th": 3.0, "background": None}


def test_combined_null_is_combined_like_the_image():
    """Null images equal to the activity images combine to the combined
    activity image (same scale, mask and median subtraction)."""
    rng = np.random.default_rng(4)
    act = np.stack(
        [np.abs(_noise_field(rng, (30, 40))) * 50 for _ in range(4)]
    )
    act[1, :2] = np.nan
    im, _, _ = combined_activity_image(act, 0.33)
    null = combined_null_image(act[:, None], act, 0.33)
    assert null.shape == (1, 30, 40)
    np.testing.assert_allclose(null[0], im, equal_nan=True)


def _trials(rng, n=3, shape=(48, 52), null=True):
    """Trials of one structured mean image with noise activity images."""
    base = ndimage.gaussian_filter(rng.random(shape), 2) * 100 + 10
    out = []
    for _ in range(n):
        act = np.abs(_noise_field(rng, shape))
        act[20, 20] += 5
        nul = np.abs(_noise_field(rng, shape))[None] if null else None
        out.append(
            TrialImages(
                base[..., None] + rng.random(shape + (1,)), act, null_im=nul
            )
        )
    return out


def test_plan_with_null_acceptance():
    """The plan seeds with the trials' null images and reports the
    threshold; without null images it refuses."""
    rng = np.random.default_rng(5)
    trials = _trials(rng)
    det = {}
    p = PlanParams(
        seed_acceptance="null",
        seed_false_per_field=2.0,
        inter_trial_alignment="rigid",
    )
    plan = plan_path(trials, params=p, details=det)
    assert det["null_im"].shape == (1,) + plan.shape
    assert det["seed_info"]["draws"] == 1 and det["seed_info"]["background"]
    det0 = {}
    plan_path(
        trials, params=PlanParams(inter_trial_alignment="rigid"), details=det0
    )
    assert det0["seed_info"]["background"] is None and "null_im" not in det0
    with pytest.raises(ValueError, match="null images"):
        plan_path(_trials(rng, null=False), params=p)
    with pytest.raises(ValueError):
        PlanParams(seed_acceptance="robust")
    with pytest.raises(ValueError):
        PlanParams(seed_false_per_field=0)


def test_null_images_round_trip(tmp_path):
    """Localization files keep the null images when given."""
    rng = np.random.default_rng(6)
    act = rng.random((6, 7))
    null = rng.random((2, 6, 7)).astype(np.float32)
    path = tmp_path / "loc.h5"
    write_localization(
        path,
        rng.random((6, 7, 1)),
        act,
        act,
        np.zeros(5, bool),
        None,
        {},
        None,
        null,
    )
    rec = read_localization(path)
    np.testing.assert_array_equal(rec.null_im, null)
    assert rec.attrs["null_im"] == 1
    np.testing.assert_array_equal(rec.trial_images().null_im, null)
    write_localization(
        path, rng.random((6, 7, 1)), act, act, np.zeros(5, bool), None, {}
    )
    assert read_localization(path).null_im is None


def test_silo_params_null_acceptance():
    """The localization keeps null images exactly when the seeding needs
    them (unless set); both need the bootstrap null; defaults leave the
    stage hashes and MATLAB's fields alone."""
    default = SiloParams()
    assert default.seed_acceptance == "image"
    assert default.localization_keep_null is False
    net = SiloParams(baseline_source="network", seed_acceptance="null")
    assert net.localization_keep_null is True
    assert net.plan_params().seed_acceptance == "null"
    assert "localization_keep_null" in net.stage_settings("localize")
    assert "seed_acceptance" in net.stage_settings("plan")
    assert "localization_keep_null" not in SiloParams(
        baseline_source="network"
    ).stage_settings("localize")
    m = net.to_matlab(2, 80.0)
    assert m["seedAcceptance"] == "null" and m["localizationKeepNull"]
    assert "seedAcceptance" not in default.to_matlab(2, 80.0)
    keep = SiloParams(baseline_source="network", localization_keep_null=True)
    assert keep.seed_acceptance == "image" and keep.localization_keep_null
    for bad in (
        dict(seed_acceptance="null"),  # no bootstrap null
        dict(
            baseline_source="network",
            seed_acceptance="null",
            localization_keep_null=False,
        ),
        dict(baseline_source="network", seed_false_per_field=-1.0),
        dict(baseline_source="network", seed_acceptance="robust"),
    ):
        with pytest.raises(ValueError):
            SiloParams(**bad)


def test_silo_params_evidence_test():
    """The evidence test is off by default with MATLAB's baseline (on with
    the network's: min_source_event_rate 2 per minute) and reaches only the
    collect stage's settings when on."""
    default = SiloParams()
    assert default.min_source_snr == 0 and default.min_source_event_rate == 0
    assert "min_source_snr" not in default.stage_settings("collect")
    assert "min_source_event_rate" not in default.stage_settings("collect")
    net = SiloParams(baseline_source="network")
    assert net.min_source_snr == 0 and net.min_source_event_rate == 2.0
    assert net.stage_settings("collect")["min_source_event_rate"] == 2.0
    assert "min_source_event_rate" not in net.stage_settings("plan")
    on = SiloParams(min_source_snr=2.0, min_source_event_rate=1.5)
    assert on.stage_settings("collect")["min_source_snr"] == 2.0
    assert on.to_matlab(2, 80.0)["minSourceSNR"] == 2.0
    assert on.to_matlab(2, 80.0)["minSourceEventRate"] == 1.5
    for bad in (dict(min_source_snr=-1.0), dict(min_source_event_rate=-1)):
        with pytest.raises(ValueError):
            SiloParams(**bad)


def test_null_sd_acceptance():
    """acceptance='null_sd' counts peak_th in robust SDs of the null; the
    plan and SiloParams accept it with the null images."""
    rng = np.random.default_rng(7)
    im = _noise_field(rng, (80, 80))
    g = np.exp(-np.add.outer(np.arange(-3, 4) ** 2, np.arange(-3, 4) ** 2) / 4)
    im[37:44, 37:44] += 3.0 * g
    null = 2.0 * _noise_field(rng, (80, 80))[None]
    info = {}
    seeds = seed_sources(
        im, 4.0, fit="local", acceptance="null_sd", null_im=null, info=info
    )
    bg = null_background(null)
    np.testing.assert_allclose(
        seeds, get_act_im_peaks(im, 4.0, fit="local", background=bg)
    )
    assert info["background"] == pytest.approx(bg)
    with pytest.raises(ValueError):
        seed_sources(im, 4.0, acceptance="null_sd")
    det = {}
    plan_path(
        _trials(np.random.default_rng(8)),
        params=PlanParams(
            seed_acceptance="null_sd", inter_trial_alignment="rigid"
        ),
        details=det,
    )
    assert det["null_im"].shape[0] == 1 and det["seed_info"]["background"]
    p = SiloParams(baseline_source="network", seed_acceptance="null_sd")
    assert p.localization_keep_null and p.plan_params().seed_acceptance == (
        "null_sd"
    )


def test_amplitude_rule_pixel_keeps_sharp_peaks():
    """MATLAB's final test (a 2 x 2 pixel mean of the Gaussian) drops a
    sharp peak whose own pixel is well above the threshold; 'pixel' keeps
    it, and drops a peak whose own pixel is below."""
    from giant_python.extraction.raster.peaks import _finish

    theta = np.array([[0.9, 10.0, 10.0, 0.3], [0.1, 20.0, 20.0, 0.3]])
    excl = np.zeros((30, 30), bool)
    assert _finish(theta, 0.22, excl, False).shape[0] == 0
    kept = _finish(theta, 0.22, excl, False, "pixel")
    np.testing.assert_allclose(kept[:, 1:3], [[9.0, 9.0]])
    with pytest.raises(ValueError):
        get_act_im_peaks(np.zeros((10, 10)), 3.0, amplitude_rule="area")
    with pytest.raises(ValueError):
        PlanParams(seed_amplitude_rule="area")
    p = SiloParams(seed_amplitude_rule="pixel")
    assert p.plan_params().seed_amplitude_rule == "pixel"
    assert "seed_amplitude_rule" in p.stage_settings("plan")
    assert "seed_amplitude_rule" not in SiloParams().stage_settings("plan")


def test_null_images_from_localization_and_empty_plans():
    """The in-memory adapter carries the localization's null images; a
    plan without valid trials has no seeds in the null modes too."""
    from types import SimpleNamespace

    rng = np.random.default_rng(9)
    null = rng.random((1, 6, 7))
    loc = SimpleNamespace(
        act_im=rng.random((6, 7)), peaks=None, info={"null": {"null_im": null}}
    )
    trial = SimpleNamespace(
        mean_im=rng.random((6, 7, 1)), discard=np.zeros(4, bool), half_ims=None
    )
    np.testing.assert_array_equal(
        TrialImages.from_localization(loc, trial).null_im, null
    )
    loc.info = {}
    assert TrialImages.from_localization(loc, trial).null_im is None
    blank = [
        TrialImages(
            t.mean_im, np.full(t.act_im.shape, np.nan), null_im=t.null_im
        )
        for t in _trials(np.random.default_rng(10))
    ]
    plan = plan_path(
        blank,
        params=PlanParams(
            seed_acceptance="null", inter_trial_alignment="rigid"
        ),
    )
    assert plan.valid_trials.size == 0 and plan.seeds.shape == (0, 4)


def test_trial_null_seeds_are_independent():
    """With kept null images every trial-path draws its own null movies
    (distinct seeds, apart by more than a trial's draws); the default
    calibration's seed is not used for them."""
    from giant_python.pipeline.silo import trial_null_seed

    seeds = {trial_null_seed(0, d, t) for d in range(3) for t in range(200)}
    assert len(seeds) == 600
    s = sorted(seeds)
    assert min(np.diff(s)) >= 1000 and min(s) > 0
    assert trial_null_seed(7, 0, 0) == trial_null_seed(0, 0, 0) + 7


def test_pixel_rule_uses_the_fits_pixel_edges():
    """Beside a gap in the imaged columns the fits' pixel is wider, and the
    'pixel' rule integrates the same wider pixel."""
    from giant_python.extraction.raster.peaks import _finish, _pixel_bounds

    finite = np.ones((20, 20), bool)
    finite[:, 11:14] = False  # a gap of 3 columns right of column 10
    (ylo, yhi), (xlo, xhi) = _pixel_bounds(finite)
    np.testing.assert_allclose([ylo[5], yhi[5]], [5.5, 6.5])
    assert xhi[10] - xlo[10] == pytest.approx(2.5)  # midpoint to column 15
    theta = np.array([[0.3, 10.0, 11.4, 0.6]])  # 1-based centre in col 11
    excl = np.zeros((20, 20), bool)
    unit = _finish(theta, 0.22, excl, False, "pixel")
    wide = _finish(
        theta, 0.22, excl, False, "pixel", 0.0, ((ylo, yhi), (xlo, xhi))
    )
    assert unit.shape[0] == 0 and wide.shape[0] == 1


def test_trial_null_seeds_leave_room_for_many_draws():
    """With more draws than the default block, blocks widen so that no two
    trial-paths share a seed."""
    from giant_python.pipeline.silo import trial_null_seed

    draws = 2500
    used = set()
    for d in range(2):
        for t in range(4):
            s0 = trial_null_seed(0, d, t, draws)
            block = set(range(s0, s0 + draws))
            assert not block & used
            used |= block
