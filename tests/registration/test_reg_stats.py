"""Registration summary statistics (``registration.stats``)."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from giant_python.io.hdf5 import save_struct_h5
from giant_python.io.slap2_raster import read_raster_samples
from giant_python.registration import multiroi
from giant_python.registration.frames import FrameFormer
from giant_python.registration.interpolation import interp_frame
from giant_python.registration.multiroi import frame_times
from giant_python.registration.sources import make_source
from giant_python.registration.stats import (
    STATS_VERSION,
    measure_gain_curves,
    new_accumulator,
    read_stats,
    stats_path,
)
from giant_python.registration.stats.gain import curve_from_sums, slot_sums
from giant_python.registration.stats.geometry import (
    FrameGeometry,
    bilinear_corners,
)
from giant_python.registration.stats.session import combine_trials

from .synthetic import make_layout, write_recording

MOVIE = "E1T1DMD1_REGISTERED_DOWNSAMPLED-80Hz.h5"
ADATA = "E1T1DMD1_ALIGNMENTDATA.h5"


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    """Layout and data file of a 6-second recording (moving scene)."""
    path = tmp_path_factory.mktemp("rec") / "synthetic_DMD1-TRIAL000001.dat"
    layout = make_layout(path, n_cycles=1400)
    write_recording(layout, invalid=0.0)
    return layout


@pytest.fixture
def patched(recording, monkeypatch):
    """The recording's layout, read by registration."""
    monkeypatch.setattr(multiroi, "raster_layout", lambda path: recording)
    return recording


def _source(layout, samples, first=1.0, last=3000.0):
    """Frame source of samples over the layout (as registration's)."""
    former = FrameFormer(layout, samples, 0)
    dt = (1 / layout.line_period_s) / 80
    dtk = int(np.ceil(dt))
    init, ds = frame_times(first, last, dt, 40)
    return make_source(layout, former, first, last, init, ds, dtk, False), ds


def _motion(n, seed=0, jump=None):
    """Sub-pixel motion (rows, cols) with jitter."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    m = np.stack(
        [
            0.7 * np.sin(t / 7.0) + rng.normal(0, 0.15, n),
            0.9 * np.cos(t / 11.0) + rng.normal(0, 0.15, n),
        ],
        1,
    )
    if jump is not None:
        m[jump] += [1.3, -1.1]
    return m


def _registered(source, ds, motion, margin):
    """Registered frames and variance factors (``interp_frame``)."""
    vals, fresh = source.frames(ds)
    win = source.window
    for k in range(ds.size):
        fimg = win.image(fresh[k], 0.0)
        regs = [
            interp_frame(
                win.image(vals[ch, k], float("nan")), fimg, *motion[k], margin
            )
            for ch in range(vals.shape[0])
        ]
        yield [r for r, _ in regs], regs[0][1]


# --------------------------------------------------------------- geometry


def _coefficients(layout, source, ds, motion, margin, t, p):
    """Sample coefficients of registered frame ``t`` at output pixel ``p``
    by the definition (slow loops)."""
    win = source.window
    w_out = win.shape[1] + 2 * margin
    ids = source.former.pixels[source.pixels.numpy()]
    where = {
        (int(r), int(c)): i
        for i, (r, c) in enumerate(zip(win.rows.numpy(), win.cols.numpy()))
    }
    dt = int(source.dt)
    reach = max(3 * dt, layout.lines_per_cycle)
    r, c = divmod(int(p), w_out)
    off_r, off_c, coef, _ = bilinear_corners(*motion[t], margin)
    parts, den = [], 0.0
    for k, (a, b) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        i = where.get((r + off_r + a, c + off_c + b))
        if i is None:
            continue
        slots = np.flatnonzero(layout.slot_pixel == ids[i])
        for s in slots:
            for cyc in range(layout.num_cycles):
                line = cyc * layout.lines_per_cycle + layout.slot_line[s]
                if abs(ds[t] - line) <= reach:
                    wgt = coef[k] * np.exp(-abs(ds[t] - line) / dt)
                    parts.append(((s, cyc), wgt))
                    den += wgt
    out = {}
    for key, v in parts:
        out[key] = out.get(key, 0.0) + v / den
    return out


def test_geometry_matches_definition(recording):
    """Photon-noise sums equal sums over the samples' coefficients."""
    layout = recording
    samples = read_raster_samples(layout, (0, layout.num_cycles))
    source, ds = _source(layout, samples)
    m = 6
    motion = _motion(ds.size, jump=12)
    geo = FrameGeometry(source, ds, m)
    assert geo.n_lags == 6
    vfs = torch.stack([vf.reshape(-1) for _, vf in
                       _registered(source, ds, motion, m)])  # fmt: skip
    ok = torch.nonzero(torch.isfinite(vfs[:30]).all(0)).ravel()
    pix = ok[:: max(1, ok.numel() // 6)][:6]
    f0, f1 = 8, 16
    e = f1 + geo.n_lags
    res = geo.registered(
        f0, f1, e, pix, motion[f0:e], np.ones(e - f0, bool), vfs[f0:e][:, pix]
    )
    for j, p in enumerate(pix.tolist()):
        for t in range(f0, f1):
            at = _coefficients(layout, source, ds, motion, m, t, p)
            g0 = sum(v * v for v in at.values())
            r0 = sum(v**3 for v in at.values())
            assert res["G"][0, t - f0, j] == pytest.approx(g0, rel=1e-5)
            assert res["R0"][t - f0, j] == pytest.approx(r0, rel=1e-5)
            for lag in (1, 3, 6):
                au = _coefficients(layout, source, ds, motion, m, t + lag, p)
                g = sum(v * au.get(k, 0) for k, v in at.items())
                rr = sum(v * v * au.get(k, 0) for k, v in at.items())
                got = res["G"][lag, t - f0, j]
                assert abs(got - g) <= 1e-5 * g0
                assert abs(res["R"][lag - 1, t - f0, j] - rr) <= 1e-5 * r0


# ------------------------------------------------------------------- gains


def _compound_poisson(rate, rng, k=5.0, theta=22.0):
    """Samples: sums of Poisson(rate) integer gamma pulses (int16)."""
    n = rng.poisson(rate)
    out = np.zeros(rate.shape)
    for j in range(int(n.max())):
        use = n > j
        out[use] += np.round(rng.gamma(k, theta, int(use.sum())))
    return out.astype(np.int16)


def test_gain_curve_of_compound_poisson():
    """Gains equal E[A^2]/E[A] and E[A^3]/E[A] of the pulse heights."""
    rng = np.random.default_rng(3)
    rate = np.repeat(np.geomspace(0.03, 3, 1600)[:, None], 3000, 1)
    x = _compound_poisson(rate, rng)
    pulses = np.round(rng.gamma(5.0, 22.0, 2_000_000))
    g2 = (pulses**2).mean() / pulses.mean()
    g3 = (pulses**3).mean() / pulses.mean()
    curve = curve_from_sums(slot_sums(torch.from_numpy(x[None]), 0))
    assert np.allclose(curve.g2, g2, rtol=0.03)  # >= 9000 photons each
    assert np.allclose(curve.g3, g3, rtol=0.08)
    nu = curve.at(curve.mu)[2]
    zero = (x == 0).mean(1)
    mean = x.mean(1)
    want = np.interp(curve.mu, mean, -np.log(zero))
    assert np.allclose(nu, want, rtol=0.05)


# --------------------------------------------------------------- the maps


def _stats(layout, samples, motion_seed=0, last=9000.0):
    """Statistics of samples through registration's frame formation."""
    source, ds = _source(layout, samples, last=last)
    acc = new_accumulator(source, ds, 6, layout.num_channels)
    motion = _motion(ds.size, motion_seed)
    for k, (regs, vf) in enumerate(_registered(source, ds, motion, 6)):
        acc.add(k, regs, vf, tuple(motion[k]))
    return acc.finish()


def _interior(maps):
    """Pixels whose maps are defined in every channel."""
    return np.isfinite(maps["slow_rel"]).all(0) & np.isfinite(
        maps["fast_rel"]
    ).all(0)


@pytest.fixture(scope="module")
def uniform(tmp_path_factory):
    """A uniform scene of compound-Poisson samples (0.05 and 2 photons
    per sample in the two channels; no file needed)."""
    path = tmp_path_factory.mktemp("uni") / "uniform_DMD1-TRIAL000001.dat"
    layout = make_layout(path, n_cycles=1600)
    rng = np.random.default_rng(5)
    shape = (layout.slot_line.size, layout.num_cycles)
    rate = np.stack([np.full(shape, 0.05), np.full(shape, 2.0)])
    return layout, _compound_poisson(rate, rng), rate


def test_photon_noise_maps_read_zero(uniform):
    """Under photon noise alone the normalised maps are 0 at any
    brightness (means over pixels: a variance from few blocks is unbiased
    but its median is low, as for chi-square)."""
    layout, samples, _ = uniform
    res = _stats(layout, samples, last=11000.0)
    maps = res["maps"]
    use = _interior(maps)
    assert use.sum() > 300
    for ch in range(2):
        for name, tol in (
            ("var_rel", 0.02),
            ("fast_rel", 0.02),
            ("slow_rel", 0.1),
            ("fast_skew_rel", 0.06),
            ("zero_excess", 0.003),
        ):
            mean = np.mean(maps[name][ch][use])
            assert abs(mean) < tol, (ch, name, mean)
    gains = measure_gain_curves(torch.from_numpy(samples))
    assert np.allclose(gains[1].g2, 132.0, rtol=0.03)
    assert res["svd"]["var_share"].shape == (2, 3)


def test_events_raise_fast_skewness(uniform):
    """Positive transients raise the fast maps where they are."""
    layout, _, rate = uniform
    rate = rate.copy()
    rows = (layout.slot_pixel - 1) // layout.n_cols
    cols = (layout.slot_pixel - 1) % layout.n_cols
    src = (np.abs(rows - 17) <= 1) & (np.abs(cols - 24) <= 1)
    p = layout.lines_per_cycle
    t = (np.arange(layout.num_cycles) * p) * layout.line_period_s
    kern = np.zeros(t.size)
    for onset in np.arange(0.3, t[-1], 0.9):
        kern += np.where(t >= onset, np.exp(-(t - onset) / 0.025), 0.0)
    rate[1, src] += 4.0 * kern
    samples = _compound_poisson(rate, np.random.default_rng(6))
    maps = _stats(layout, samples, last=11000.0)["maps"]
    m = 6
    sk = maps["fast_skew_rel"][1]
    fr = maps["fast_rel"][1]
    at = sk[17 - 8 + m, 24 - 10 + m]
    away = np.nanmedian(sk[m : m + 5])  # DMD rows 8-12
    assert at > 1.0 and abs(away) < 0.2
    assert fr[17 - 8 + m, 24 - 10 + m] > 1.0


# ------------------------------------------------- registration and command


def test_registration_and_command_agree(patched, tmp_path):
    """``giant reg-stats`` on registration's outputs gives the same maps;
    the mean equals ``meanIM``; statistics do not change the movie."""
    from giant_python.pipeline.reg_stats import trial_stats

    layout = patched
    out = {}
    for stats in (True, False):
        d = tmp_path / str(stats)
        d.mkdir()
        params = multiroi.MultiRoiRegistrationParams(maxshift=6)
        multiroi.register_trial(
            layout.files[0], 1, 3000, d / MOVIE, d / ADATA, params, stats=stats
        )
        with h5py.File(d / MOVIE) as f:
            out[stats] = f["data"][()]
    np.testing.assert_array_equal(out[True], out[False])
    assert not stats_path(tmp_path / "False" / MOVIE).exists()
    d = tmp_path / "True"
    a = read_stats(stats_path(d / MOVIE))
    assert a["attrs"]["stats_version"] == STATS_VERSION
    with h5py.File(d / ADATA) as f:
        mean_im = np.swapaxes(np.moveaxis(f["meanIM"][()], -1, 0), 1, 2)
    np.testing.assert_array_equal(a["maps"]["mean"], mean_im)
    cmd = tmp_path / "cmd" / stats_path(d / MOVIE).name
    trial_stats(layout.files[0], 1, 3000, d / MOVIE, d / ADATA, cmd, "cpu",
                {"maxshift": 6})  # fmt: skip
    b = read_stats(cmd)
    for grp in ("maps", "svd", "gain"):
        assert a[grp].keys() == b[grp].keys()
        for k in a[grp]:
            np.testing.assert_array_equal(a[grp][k], b[grp][k], err_msg=k)


def test_rerun_without_stats_removes_old_file(patched, tmp_path):
    """A registration without statistics leaves none from an earlier one."""
    layout = patched
    params = multiroi.MultiRoiRegistrationParams(maxshift=6)
    args = (layout.files[0], 1, 3000, tmp_path / MOVIE, tmp_path / ADATA)
    multiroi.register_trial(*args, params)
    assert stats_path(tmp_path / MOVIE).exists()
    multiroi.register_trial(*args, params, stats=False)
    assert not stats_path(tmp_path / MOVIE).exists()


def test_reg_stats_command(patched, tmp_path):
    """The CLI writes the statistics of a table's trials once."""
    from giant_python.cli import main
    from giant_python.pipeline.reg_stats import reg_stats
    from giant_python.pipeline.register import register_multiroi

    layout = patched
    dat = Path(layout.files[0])
    table = tmp_path / "trial_table.h5"
    save_struct_h5(
        table,
        {
            "datadr": str(dat.parent),
            "savedr": str(tmp_path),
            "filename": np.array([[dat.name]], dtype=object),
            "epoch": np.array([[1.0]]),
            "true_trial_ix": np.array([[1.0]]),
            "slap2_info": {
                "first_line": np.array([[1.0]]),
                "last_line": np.array([[3000.0]]),
            },
        },
    )
    register_multiroi(
        table,
        {"maxshift": 5},
        execution={"device": "cpu", "max_workers": 1},
        stats=False,
    )
    out = tmp_path / "stats"
    assert main(["reg-stats", str(table), "--out", str(out),
                 "--device", "cpu"]) == 0  # fmt: skip
    written = out / "E1T1DMD1_REGSTATS-80Hz.h5"
    assert written.exists()
    assert reg_stats(table, out, device="cpu") == []  # up to date


def test_stats_path_names():
    """Statistics files sit next to the movie, named after it."""
    assert stats_path(Path("d") / MOVIE).name == "E1T1DMD1_REGSTATS-80Hz.h5"
    tif = Path("E2T3DMD2_REGISTERED_DOWNSAMPLED-80Hz.tif")
    assert stats_path(tif).name == "E2T3DMD2_REGSTATS-80Hz.h5"
    assert stats_path(Path("m.h5")).name == "m_REGSTATS.h5"


# ----------------------------------------------------------------- session


def test_combine_trials_weights_by_frames():
    """Session maps: shifted, frame-weighted means; normalised maps from
    the averages."""
    shape = (1, 20, 30)
    trials = []
    for t, (n, var) in enumerate(((100.0, 2.0), (300.0, 4.0))):
        maps = {}
        for k in ("n_valid", "n_blocks", "n_fast"):
            maps[k] = np.full(shape, n)
        for k in ("mean", "var", "var_q", "zero_share", "zero_ph",
                  "gain_g2", "gain_g3", "sigma8_ph", "slow_var", "slow_q",
                  "fast_var", "fast_q", "fast_m3", "fast_q3",
                  "fast_m3_ph"):  # fmt: skip
            maps[k] = np.full(shape, 1.0)
        maps["var"] = np.full(shape, var)
        for k in ("var_ph", "slow_var_ph", "fast_var_ph"):
            maps[k] = np.full(shape, 2.0)
        maps["mean"][0, :, 0] = np.nan
        trials.append({"maps": maps})
    out = combine_trials(trials, shifts=np.array([[0.0, 0.0], [0.0, 1.0]]))
    assert out["n_valid"][0, 5, 5] == pytest.approx(400.0)
    assert out["var"][0, 5, 5] == pytest.approx((100 * 2 + 300 * 4) / 400)
    assert out["var_rel"][0, 5, 5] == pytest.approx(3.5 / 2 - 1)
    assert out["n_trials"][0, 5, 5] == 2
    # trial 2's first column is missing; shifted by one column it lands
    # left of the frame, and its last column has no data
    assert out["n_trials"][0, 5, -1] == 1
