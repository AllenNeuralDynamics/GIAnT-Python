"""The growth schedule: wide search, scheduler pieces, end to end."""

import numpy as np
import pytest
import torch

from giant_python.io.hdf5 import load_struct_h5
from giant_python.registration import multiroi
from giant_python.registration.growth import (
    GROWN,
    REGROWN,
    _runs,
    seeds_from_consensus,
)
from giant_python.registration.matching import (
    correlation_map,
    peak_margin,
    shift_tables,
    wide_correlation,
)
from giant_python.registration.template import (
    ConsensusResult,
    TemplateState,
)

from .synthetic import make_layout, write_recording


def test_wide_correlation_equals_matrix_form():
    """FFT correlation over the padded template equals the GEMM form."""
    rng = np.random.default_rng(0)
    h, w, m = 14, 19, 4
    tmpl = rng.random((h + 2 * m, w + 2 * m)) * 10
    tmpl[rng.random(tmpl.shape) < 0.15] = np.nan
    frame = tmpl[m + 1 : m + 1 + h, m - 2 : m - 2 + w] + rng.normal(
        0, 0.3, (h, w)
    )
    frame[rng.random(frame.shape) < 0.3] = np.nan
    fresh = np.where(np.isnan(frame), 0, rng.random((h, w)) + 0.5)
    wide = wide_correlation(
        torch.from_numpy(frame)[None],
        torch.from_numpy(fresh)[None],
        torch.from_numpy(tmpl),
    )[0].numpy()
    rows, cols = np.nonzero(~np.isnan(frame))
    tables = shift_tables(
        torch.from_numpy(tmpl),
        torch.from_numpy(rows + m),
        torch.from_numpy(cols + m),
        m,
        torch.float64,
    )
    gemm = correlation_map(
        torch.from_numpy(frame[rows, cols])[None],
        torch.from_numpy(fresh[rows, cols])[None],
        tables,
        m,
    )[0].numpy()
    np.testing.assert_allclose(wide, gemm, atol=1e-9)
    r, c = np.unravel_index(np.nanargmax(wide), wide.shape)
    assert (r - m, c - m) == (1, -2)


def test_peak_margin_and_runs():
    """Margin excludes the peak's neighbourhood; runs of True."""
    corr = torch.zeros(1, 9, 9, dtype=torch.float64)
    corr[0, 4, 4] = 1.0
    corr[0, 5, 5] = 0.9
    corr[0, 0, 8] = 0.6
    np.testing.assert_allclose(peak_margin(corr).numpy(), [0.4])
    mask = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=bool)
    assert _runs(mask) == [(1, 3), (5, 6), (7, 10)]


def test_template_update_adds_and_removes():
    """Batch add then remove restores the sums."""
    total = torch.ones(3, 3, dtype=torch.float64) * 4
    count = torch.ones(3, 3, dtype=torch.float64)
    state = TemplateState(total.clone(), count.clone(), None, 1)
    imgs = torch.rand(5, 3, 3) * 9
    imgs[0, 1, 1] = float("nan")
    state.update(imgs)
    assert state.count[1, 1] == 5 and state.count[0, 0] == 6
    state.update(imgs, sign=-1)
    torch.testing.assert_close(state.total, total)
    torch.testing.assert_close(state.count, count)


def test_seeds_from_consensus():
    """Seeds are the nearest DS frames, with MATLAB's motion sign."""
    init = np.array([105, 300, 501, 700])
    ds = np.arange(1, 800, 20)
    mot = np.zeros((2, 4, 4))
    mot[:, 0, 2] = [1.4, -2.6]
    cons = ConsensusResult(
        np.eye(4), mot, reference=2, members=np.array([0, 2])
    )
    seeds, cands, indep = seeds_from_consensus(cons, init, ds, reach=30)
    assert seeds == [(5, -1, 3), (25, 0, 0)]
    assert cands == [15, 35]
    assert not indep[5] and not indep[25] and indep[15]


@pytest.fixture(scope="module")
def plain_recording(tmp_path_factory):
    """Synthetic recording without online offsets."""
    path = tmp_path_factory.mktemp("rec") / "grow_DMD1-TRIAL000001.dat"
    layout = make_layout(path, n_cycles=1400, seed=5)
    truth = write_recording(layout, seed=6)
    return layout, truth


def test_grow_matches_sequential(plain_recording, tmp_path, monkeypatch):
    """Growth aligns every frame, like sequential, and saves its record."""
    layout, _ = plain_recording
    monkeypatch.setattr(multiroi, "raster_layout", lambda path: layout)
    out = {}
    for sched in ("sequential", "grow"):
        params = multiroi.MultiRoiRegistrationParams(
            maxshift=6, schedule=sched, polish=sched == "grow"
        )
        out[sched] = multiroi.register_trial(
            layout.files[0],
            1,
            9000,
            tmp_path / f"{sched}.tif",
            tmp_path / f"{sched}.h5",
            params,
        )
    g = out["grow"].growth
    assert np.all(np.isfinite(out["grow"].motion_r))
    assert g.seeds_accepted >= 1 and g.round.max() < 450
    diff = np.hypot(
        out["grow"].motion_r - out["sequential"].motion_r,
        out["grow"].motion_c - out["sequential"].motion_c,
    )
    assert np.sqrt(np.mean(diff**2)) < 0.05
    qc = load_struct_h5(tmp_path / "grow.h5")["growthQC"]
    assert qc["kind"].size == out["grow"].ds_frames.size
    assert qc["seedsAccepted"].item() == g.seeds_accepted


def test_junction_regrows_a_wrong_island(plain_recording, monkeypatch):
    """An island shifted by 2 px is realigned from its neighbour."""
    from giant_python.registration.growth import FrameCache, Grower
    from giant_python.registration.multiroi import build_template, open_trial
    from giant_python.registration.sources import make_source

    layout, _ = plain_recording
    monkeypatch.setattr(multiroi, "raster_layout", lambda path: layout)
    params = multiroi.MultiRoiRegistrationParams(maxshift=6, schedule="grow")
    lay, former, init, ds, dt = open_trial(
        layout.files[0], 1, 9000, params, torch.device("cpu")
    )
    source = make_source(lay, former, 1, 9000, init, ds, dt, False)
    state, cons = build_template(source, init, params, torch.float32)
    seeds, cands, indep = seeds_from_consensus(cons, init, ds)
    grower = Grower(
        FrameCache(source, ds), state, params, seeds, cands, independent=indep
    )
    motion, _ = grower.run()
    good = motion.copy()
    bad = np.arange(200, 240)
    grower._withdraw(bad)
    grower._accept(
        bad,
        good[bad] + [2.0, 0.0],
        grower.qc.peak[bad] - 0.2,
        grower.qc.margin[bad],
        np.full(bad.size, -1),
        [GROWN] * bad.size,
        [-7] * bad.size,
    )
    grower.qc.anchor[240] = 241  # frame 240 was reached from the right
    grower._check_junctions([240])
    assert grower.qc.junctions and grower.qc.junctions[-1][3] == "regrown"
    assert np.all(grower.qc.kind[bad] == REGROWN)
    err = np.hypot(*(grower.motion[bad] - good[bad]).T)
    assert err.max() < 0.2


def test_sub_round_ends_when_seeds_align_the_last_frame(monkeypatch):
    """Seeds that align the last unaligned frame end the sub-round."""
    from giant_python.registration.growth import Grower

    class Frames:
        """Stands in for a frame cache; only its length is used."""

        def __len__(self):
            """Five frames."""
            return 5

    params = multiroi.MultiRoiRegistrationParams(schedule="grow")
    grower = Grower(Frames(), None, params, [], [4])
    grower.aligned[:4] = True
    grower.motion[:4] = 0.0
    grower.root[:4] = 0

    def seed(ks):
        """Align the candidates, as an accepted wide search would."""
        grower.aligned[ks] = True
        grower.motion[ks] = 0.0

    monkeypatch.setattr(grower, "_calibrated", lambda: True)
    monkeypatch.setattr(grower, "try_seeds", seed)
    grower._sub_round()
    assert grower.aligned.all()


def test_growth_handles_a_frame_without_valid_samples(
    plain_recording, monkeypatch
):
    """Inside the grower, a frame whose correlation is undefined keeps its
    anchor's motion and does not stop a realignment chain.

    (``register_grow`` leaves frames without samples out beforehand; this
    covers frames whose correlation is undefined for other reasons.)
    """
    from giant_python.registration.growth import FrameCache, Grower
    from giant_python.registration.multiroi import build_template, open_trial
    from giant_python.registration.sources import make_source

    layout, _ = plain_recording
    monkeypatch.setattr(multiroi, "raster_layout", lambda path: layout)
    params = multiroi.MultiRoiRegistrationParams(maxshift=6, schedule="grow")
    lay, former, init, ds, dt = open_trial(
        layout.files[0], 1, 9000, params, torch.device("cpu")
    )
    source = make_source(lay, former, 1, 9000, init, ds, dt, False)
    state, cons = build_template(source, init, params, torch.float32)
    seeds, cands, indep = seeds_from_consensus(cons, init, ds)
    taken = {k for k, _, _ in seeds} | set(cands)
    k = next(i for i in range(200, ds.size) if i not in taken)
    cache = FrameCache(source, ds)
    cache.values[:, k] = float("nan")
    cache.freshness[k] = 0
    grower = Grower(cache, state, params, seeds, cands, independent=indep)
    motion, qc = grower.run()
    assert grower.aligned.all() and np.all(np.isfinite(motion))
    assert np.isnan(qc.peak[k]) and qc.anchor[k] >= 0
    np.testing.assert_array_equal(motion[k], motion[qc.anchor[k]])
    chain, peak, _ = grower._chain([k], motion[k - 1])
    np.testing.assert_array_equal(chain[0], motion[k - 1])
    assert np.isnan(peak[0])


def test_consensus_seeds_stay_within_maxshift(plain_recording, monkeypatch):
    """Seeds start inside the padded template even when the consensus
    motion (up to ``init_search``) exceeds ``maxshift``."""
    from giant_python.registration.growth import FrameCache, Grower
    from giant_python.registration.multiroi import build_template, open_trial
    from giant_python.registration.sources import make_source

    layout, _ = plain_recording
    monkeypatch.setattr(multiroi, "raster_layout", lambda path: layout)
    params = multiroi.MultiRoiRegistrationParams(
        maxshift=2, init_search=3, schedule="grow"
    )
    lay, former, init, ds, dt = open_trial(
        layout.files[0], 1, 3000, params, torch.device("cpu")
    )
    source = make_source(lay, former, 1, 3000, init, ds, dt, False)
    state, cons = build_template(source, init, params, torch.float32)
    grower = Grower(FrameCache(source, ds), state, params, [(10, 3, -3)], [])
    assert grower.seeds == [(10, 2, -2)]
    grower._seed_round()
    assert grower.aligned[10] and np.all(np.isfinite(grower.motion[10]))
