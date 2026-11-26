"""Event injection into raw SLAP2 samples, on small synthetic recordings."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from giant_python.io.hdf5 import save_struct_h5
from giant_python.io.slap2_raster import INVALID_SAMPLE, read_raster_samples
from giant_python.pipeline.register import register_multiroi
from giant_python.registration import multiroi
from giant_python.simulation import injection as inj
from giant_python.simulation.render import iter_registered_frames

from ..registration.synthetic import (
    HEADER_WORDS,
    make_layout,
    online_offsets,
    write_recording,
)

N_CYCLES = 1400


def sha(path) -> str:
    """SHA-256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------- pure functions


def test_time_course_peaks_at_one():
    """Instant rise: exp decay from 1; short rise: peak 1 at its delay."""
    t = np.array([-0.01, 0.0, 0.024, 0.048])
    np.testing.assert_allclose(
        inj.time_course(t, 0.024), [0, 1, np.exp(-1), np.exp(-2)]
    )
    tp = inj.peak_delay_s(0.024, 0.003)
    assert 0 < tp < 0.024
    grid = np.linspace(0, 0.1, 100001)
    k = inj.time_course(grid, 0.024, 0.003)
    assert k.max() == pytest.approx(1.0, abs=1e-8)
    assert grid[np.argmax(k)] == pytest.approx(tp, abs=2e-6)
    assert inj.time_course(-1e-9, 0.024, 0.003) == 0


def test_event_traces_sum_events():
    """Traces are sums of size x time course at the line times."""
    s = inj.InjectionSettings(tau_s=0.01)
    ev = inj.Events(
        np.array([0, 0, 1]),
        np.array([1, 1, 1]),
        np.array([10.5, 30.0, 12.0]),
        np.array([0, 1, 2]),
        np.array([2.0, 3.0, 1.0]),
    )
    tr = inj.event_traces(ev, 2, (5, 60), 0.001, s)
    lines = np.arange(5, 61)
    want0 = 2 * inj.time_course((lines - 10.5) * 1e-3, 0.01) + 3 * (
        inj.time_course((lines - 30.0) * 1e-3, 0.01)
    )
    np.testing.assert_allclose(tr[0], want0, atol=1e-12)
    assert tr[1, 12 - 5] == pytest.approx(1.0)
    assert np.all(tr[1, : 12 - 5] == 0)


def test_draw_events_respect_edges_dead_time_and_classes():
    """Onsets stay inside the edges, apart by the dead time; classes mix."""
    s = inj.InjectionSettings(rate_hz=4.0, min_interval_s=0.1, edge_s=0.5)
    period = 1e-4
    ev = inj.draw_events(
        200, 1.0, 100001.0, period, s, np.random.default_rng(1), trial=3
    )
    assert np.all(ev.trial == 3)
    assert ev.onset_line.min() >= 1 + 0.5 / period
    assert ev.onset_line.max() <= 100001 - 0.5 / period
    for j in range(200):
        t = ev.onset_line[ev.source == j]
        assert np.all(np.diff(t) >= 0.1 / period - 1e-6)
    dur = 10.0 - 1.0
    rate = len(ev) / 200 / dur
    assert rate == pytest.approx(1 / (0.1 + 1 / 4.0), rel=0.05)
    counts = np.bincount(ev.size_class, minlength=6)
    assert counts.min() > 0.8 * counts.mean()
    np.testing.assert_array_equal(
        ev.size, np.asarray(s.size_classes)[ev.size_class]
    )
    again = inj.draw_events(
        200, 1.0, 100001.0, period, s, np.random.default_rng(1), trial=3
    )
    np.testing.assert_array_equal(again.onset_line, ev.onset_line)
    none = inj.draw_events(
        5, 1.0, 100001.0, period, replace(s, rate_hz=0.0), None
    )
    assert len(none) == 0


def test_settings_reject_bad_values():
    """Invalid settings raise."""
    with pytest.raises(ValueError):
        inj.InjectionSettings(rise_s=0.03)
    with pytest.raises(ValueError):
        inj.InjectionSettings(class_weights=(1.0, 2.0))
    with pytest.raises(ValueError):
        inj.InjectionSettings(rate_hz=-1)


def test_footprints():
    """Gaussian peak 1 and cut at its radius; image footprints interpolate."""
    g = inj.GaussianFootprint(1.33, 4.0)
    assert g(0.0, 0.0) == 1.0
    assert g(1.33, 0.0) == pytest.approx(np.exp(-0.5))
    assert g(5.4, 0.0) == 0.0 and g.radius == pytest.approx(5.32)
    img = np.zeros((5, 5))
    img[2, 2], img[2, 3] = 4.0, 2.0
    f = inj.ImageFootprint(img)
    assert f(0.0, 0.0) == 1.0
    assert f(0.0, 0.5) == pytest.approx(0.75)
    assert f(0.0, 9.0) == 0.0
    patch, origin = inj.footprint_patch(g, (10.0, 20.4), 3)
    assert origin == [7, 17] and patch.shape == (7, 7)
    assert patch[3, 3] == pytest.approx(np.exp(-0.5 * (0.4 / 1.33) ** 2))


def test_place_sources_follow_the_rules():
    """Sources sit on labeled pixels, away from edges, others and each
    other."""
    rng = np.random.default_rng(0)
    img = np.full((60, 80), 1.0)
    img[20:24, 5:75] = 10.0  # a horizontal "dendrite"
    img[30:50, 40:44] = 10.0  # a vertical one
    img[:, :3] = np.nan
    exclude = np.array([[22.0, 40.0]])
    s = inj.PlacementSettings(min_separation_px=6, edge_px=5, exclude_px=8)
    pos, n_cand = inj.place_sources(img, 100, rng, s, exclude_rc=exclude)
    assert 0 < pos.shape[0] < 100 and n_cand > pos.shape[0]
    lab = inj.labeled_pixels(img)
    r, c = pos.T.astype(int)
    assert lab[r, c].all()
    assert c.min() >= 3 + 5 - 1  # distance from the NaN columns
    d = np.hypot(*(pos[:, None, :] - pos[None, :, :]).transpose(2, 0, 1))
    assert d[np.triu_indices(len(pos), 1)].min() >= 6
    assert np.hypot(*(pos - exclude).T).min() >= 8


def test_align_images_recovers_subpixel_shift():
    """A smooth random image moved by a known shift is found again."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(3)
    ref = gaussian_filter(rng.random((90, 120)), 2.0) * 100
    ref[:, :10] = np.nan
    s = np.array([1.3, -2.6])
    moved = inj.shift_image(ref, -s)  # moved(p + s) = ref(p)
    est = inj.align_images(ref, moved)
    np.testing.assert_allclose(est, s, atol=0.08)


# ------------------------------------------------------ synthetic recordings


def write_static(layout, gain: float, rate: float, seed: int = 0):
    """Samples ``gain * Poisson(rate * brightness)`` of a static scene."""
    rng = np.random.default_rng(seed)
    words = np.zeros((layout.num_cycles, layout.words_per_cycle), np.int16)
    bright = 1 + 9 * ((layout.slot_pixel * 7919) % 97) / 97.0
    for ch in range(layout.num_channels):
        li = layout.slot_line - 1
        w = layout.row_base[li] + ch * layout.n_per_channel[li]
        w = w + layout.slot_pos
        lam = rate * bright
        words[:, w] = (
            gain * rng.poisson(lam, (layout.num_cycles, lam.size))
        ).astype(np.int16)
    with open(layout.files[0], "wb") as f:
        f.write(np.zeros(HEADER_WORDS, np.int16).tobytes())
        f.write(words.tobytes())


def test_measure_raw_per_photon(tmp_path):
    """The variance-to-mean ratio of scaled Poisson samples is the gain."""
    layout = make_layout(tmp_path / "g_DMD1-TRIAL000001.dat", 400, seed=2)
    write_static(layout, gain=110.0, rate=0.05)
    g = inj.measure_raw_per_photon(layout, channel=0)
    assert g.raw_per_photon == pytest.approx(110.0, rel=0.05)
    assert g.mean_pulse == pytest.approx(110.0, rel=0.01)
    assert g.cycles == (0, 400) and g.n_slots > 0


@pytest.fixture(scope="module")
def online(tmp_path_factory):
    """Synthetic recording with known motion and online offsets."""
    path = tmp_path_factory.mktemp("rec") / "inj_DMD1-TRIAL000001.dat"
    layout = make_layout(path, n_cycles=N_CYCLES, seed=5)
    offs = online_offsets(layout.total_lines)
    truth = write_recording(layout, seed=6, online=offs)
    path.with_name("inj_DMD1.meta").write_bytes(b"meta" * 100)
    return layout, offs, truth


def truth_geometry(layout, offs, truth, lines):
    """Geometry in the scene frame: sample of DMD (r, c) on line l shows
    the scene at (r + y_dmd - m_r, c + x - m_c)."""
    x, _, y_dmd = offs
    ln = np.arange(lines[0], lines[1] + 1)
    shift = np.stack(
        [y_dmd[ln - 1] - truth[ln - 1, 0], x[ln - 1] - truth[ln - 1, 1]], 1
    )
    return inj.TrialGeometry(lines[0], shift.astype(np.float64), True)


def test_sample_model_follows_the_tissue(online):
    """Expected counts are photon_scale x footprint at the sample's tissue
    position x the trace at its line, for every chunk of cycles."""
    layout, offs, truth = online
    lines = (8, layout.total_lines - 3)
    geo = truth_geometry(layout, offs, truth, lines)
    pos = np.array([[17.3, 22.0], [12.0, 30.6]])
    fps = [inj.GaussianFootprint(1.33), inj.GaussianFootprint(2.0)]
    n_l = lines[1] - lines[0] + 1
    rng = np.random.default_rng(0)
    traces = rng.random((2, n_l))
    model = inj.SampleModel(layout, geo, pos, fps, traces, 25.0)
    cycles = np.arange(0, 30)
    got = model.expected(cycles)
    pix = layout.slot_pixel[model.slots] - 1
    r0, c0 = pix // layout.n_cols, pix % layout.n_cols
    ln = (
        cycles[:, None] * layout.lines_per_cycle
        + layout.slot_line[model.slots]
    )
    inside = (ln >= lines[0]) & (ln <= lines[1])
    li = np.clip(ln - lines[0], 0, n_l - 1)
    want = np.zeros(got.shape)
    for j in range(2):
        f = fps[j](
            r0 + geo.shift[li, 0] - pos[j, 0],
            c0 + geo.shift[li, 1] - pos[j, 1],
        )
        want += 25.0 * f * traces[j, li]
    want[~inside] = 0
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
    # every slot whose footprint value can be non-zero is reachable
    every = np.arange(layout.slot_line.size)
    allpix = layout.slot_pixel[every] - 1
    near = np.zeros(every.size, bool)
    for j in range(2):
        for s in geo.shift[:: layout.lines_per_cycle]:
            d = np.hypot(
                allpix // layout.n_cols + s[0] - pos[j, 0],
                allpix % layout.n_cols + s[1] - pos[j, 1],
            )
            near |= d <= fps[j].radius
    assert np.all(np.isin(every[near], model.slots))


def registered_table(layout, tmp_path, compensate: bool, trials):
    """Register line ranges of a recording through a trial table."""
    dat = Path(layout.files[0])
    n = len(trials)
    table = tmp_path / "trial_table.h5"
    save_struct_h5(
        table,
        {
            "datadr": str(dat.parent),
            "savedr": str(tmp_path),
            "filename": np.array([[dat.name] * n], dtype=object),
            "epoch": np.ones((1, n)),
            "true_trial_ix": np.arange(1.0, n + 1)[None, :],
            "slap2_info": {
                "first_line": np.array([[float(a) for a, _ in trials]]),
                "last_line": np.array([[float(b) for _, b in trials]]),
            },
        },
    )
    register_multiroi(
        table,
        {"maxshift": 6, "compensate_online": compensate},
        execution={"device": "cpu", "max_workers": 1},
    )
    return table


@pytest.fixture(scope="module")
def registered(online, tmp_path_factory):
    """The online recording registered as two trials, both ways."""
    layout, _, _ = online
    out = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(multiroi, "raster_layout", lambda path: layout)
        for comp in (True, False):
            d = tmp_path_factory.mktemp(f"reg{int(comp)}")
            out[comp] = registered_table(
                layout, d, comp, [(1, 4800), (4801, 9700)]
            )
    return out


@pytest.mark.parametrize("compensate", [True, False])
def test_geometry_from_registration_tracks_true_motion(
    online, registered, compensate
):
    """The registered position of each line's samples follows the tissue:
    shift(l) equals the true scene shift up to a constant."""
    layout, offs, truth = online
    table = Path(registered[compensate])
    al = inj.read_alignment(
        table.parent / "motion_correction" / "E1T1DMD1_ALIGNMENTDATA.h5"
    )
    assert (al.compensation is not None) == compensate
    lines = (1, 4800)
    geo = inj.trial_geometry(layout, al, lines)
    want = truth_geometry(layout, offs, truth, lines).shift
    if not compensate:
        # uncompensated frames blur the offsets' steps (every 23-50 ms
        # here) over their kernel: the motion can only follow that blur
        k = np.exp(-np.abs(np.arange(-60, 61)) / 20.0)
        want = np.stack(
            [
                np.convolve(np.pad(w, 60, "edge"), k / k.sum(), "valid")
                for w in want.T
            ],
            1,
        )
    err = geo.shift - want
    err -= np.median(err, axis=0)
    # within the registration's own error (as its tests require)
    assert np.sqrt(np.mean(err**2)) < 0.12


def run_injection(online, table, out, monkeypatch, **kw):
    """inject_session on the synthetic recording."""
    layout, _, _ = online
    monkeypatch.setattr(inj, "raster_layout", lambda path: layout)
    return inj.inject_session(table, out, paths=(1,), **kw)


def test_no_events_leave_the_copy_identical(
    online, registered, tmp_path, monkeypatch
):
    """Without events the copy equals the original byte for byte, and the
    .meta file is copied."""
    layout, _, _ = online
    s = inj.InjectionSettings(rate_hz=0.0, raw_per_photon=1.0)
    summary = run_injection(
        online, registered[True], tmp_path / "out", monkeypatch, settings=s
    )
    src = Path(layout.files[0])
    dst = tmp_path / "out" / "raw" / src.name
    assert sha(dst) == sha(src)
    meta = src.with_name("inj_DMD1.meta")
    assert sha(dst.with_name(meta.name)) == sha(meta)
    assert summary["paths"][1]["n_events"] == 0
    truth = inj.read_truth(tmp_path / "out" / inj.TRUTH_NAME)
    assert truth["paths"][1]["events"]["source"].size == 0


BRIGHT = inj.InjectionSettings(
    size_classes=(4.0, 8.0),
    rate_hz=3.0,
    min_interval_s=0.06,
    edge_s=0.1,
    tau_s=0.012,
    raw_per_photon=2.0,
    seed=11,
    device="cpu",
)
BRIGHT_AT = {1: np.array([[16.0, 20.0], [18.0, 30.0], [12.0, 27.0]])}


@pytest.fixture(scope="module")
def injected(online, registered, tmp_path_factory):
    """Bright events injected into both trials (compensated registration)."""
    out = tmp_path_factory.mktemp("inj")
    with pytest.MonkeyPatch.context() as mp:
        summary = run_injection(
            online,
            registered[True],
            out,
            mp,
            settings=BRIGHT,
            positions=BRIGHT_AT,
        )
    return out, summary, BRIGHT


def test_injection_changes_only_injected_samples(online, injected):
    """Only channel-0 raster samples change, never invalid ones, and the
    added counts are the recorded ones."""
    layout, _, _ = online
    out, summary, s = injected
    src = Path(layout.files[0])
    a = np.fromfile(src, np.int16)
    b = np.fromfile(out / "raw" / src.name, np.int16)
    changed = np.flatnonzero(a != b)
    assert changed.size > 0
    words = changed - HEADER_WORDS
    w = words % layout.words_per_cycle
    li = layout.slot_line - 1
    ch0 = layout.row_base[li] + layout.slot_pos
    assert np.all(np.isin(w, ch0))
    assert np.all(b[changed] > a[changed])
    total = sum(t["added_raw"] for t in summary["paths"][1]["trials"])
    assert int((b.astype(np.int64) - a).sum()) == total
    stats = summary["paths"][1]["trials"]
    drawn = sum(t["drawn_photons"] for t in stats)
    lam = sum(t["expected_photons"] for t in stats)
    assert abs(drawn - lam) < 5 * np.sqrt(lam)
    assert total == pytest.approx(drawn * s.raw_per_photon, rel=1e-3)


def test_truth_file_describes_the_injection(online, injected):
    """The truth has positions per trial, events and traces at frame
    lines."""
    layout, _, _ = online
    out, summary, s = injected
    truth = inj.read_truth(out / inj.TRUTH_NAME)
    assert truth["seed"] == 11
    assert truth["settings"]["size_classes"] == [4.0, 8.0]
    p = truth["paths"][1]
    assert p["raw_per_photon"] == 2.0
    np.testing.assert_array_equal(p["trials"], [1, 2])
    np.testing.assert_allclose(p["trial_shift_rc"][0], 0.0)
    ev = p["events"]
    assert ev["source"].size == summary["paths"][1]["n_events"] > 0
    for t, (lo, hi) in ((1, (1, 4800)), (2, (4801, 9700))):
        on = ev["onset_line"][ev["trial"] == t]
        assert on.min() >= lo + 0.1 * 1600 and on.max() <= hi - 0.1 * 1600
        d = p["trials_detail"][t]
        np.testing.assert_allclose(
            d["positions_rc"], p["positions_rc"] + p["trial_shift_rc"][t - 1]
        )
        fl = d["frame_lines"]
        assert fl[0] == lo and np.all(np.diff(fl) == 8)
        # the trace peaks at each event's size within one frame of onset
        k = np.searchsorted(fl, on[0])
        j = ev["source"][ev["trial"] == t][0]
        size = ev["size"][ev["trial"] == t][0]
        assert d["trace"][j, k] <= size * 1.0001 + d["trace"][j].max()
        assert d["trace"].shape == (3, fl.size)
    assert p["footprint"]["patch"].shape[0] == 3
    with h5py.File(out / "trial_table.h5") as f:
        assert "motion_correction" not in f
    tt = inj.load_struct_h5(out / "trial_table.h5")
    assert Path(tt["datadr"]) == out / "raw"
    assert np.asarray(tt["slap2_info"]["last_line"]).ravel().tolist() == [
        4800.0,
        9700.0,
    ]


def test_injected_signal_is_still_and_as_expected(
    online, registered, injected
):
    """In registered frames made with the original's motion, the copy minus
    the original matches the expected counts within Poisson noise, and a
    static source's centroid stays put (drift < 0.1 px) across the online
    offsets' switches."""
    layout, _, _ = online
    out, _, s = injected
    table = Path(registered[True])
    truth = inj.read_truth(out / inj.TRUTH_NAME)["paths"][1]
    al = inj.read_alignment(
        table.parent / "motion_correction" / "E1T1DMD1_ALIGNMENTDATA.h5"
    )
    copy = replace(
        layout, files=(str(out / "raw" / Path(layout.files[0]).name),)
    )
    orig = read_raster_samples(layout, (0, layout.num_cycles))[0]
    new = read_raster_samples(copy, (0, layout.num_cycles))[0]
    diff = np.where(orig == INVALID_SAMPLE, INVALID_SAMPLE, new - orig)
    lines = (1, 4800)
    geo = inj.trial_geometry(layout, al, lines)
    pos = truth["trials_detail"][1]["positions_rc"]
    fp = [inj.GaussianFootprint(1.33)] * 3
    ev = truth["events"]
    sel = ev["trial"] == 1
    events = inj.Events(
        ev["source"][sel],
        ev["trial"][sel],
        ev["onset_line"][sel],
        ev["size_class"][sel],
        ev["size"][sel] / truth["frame_peak"][ev["source"][sel]],
    )
    tr = inj.event_traces(events, 3, lines, layout.line_period_s, s)
    expect = inj.SampleModel(layout, geo, pos, fp, tr, 25.0)
    static = inj.SampleModel(layout, geo, pos, fp, np.ones_like(tr), 25.0)
    n_c = layout.num_cycles  # the frame kernels reach past the trial
    samples = np.zeros((3, layout.slot_line.size, n_c), np.float32)
    samples[0] = diff[:, :n_c]
    samples[1][expect.slots] = expect.expected(np.arange(n_c)).T
    samples[2][static.slots] = static.expected(np.arange(n_c)).T
    samples[:, orig[:, :n_c] == INVALID_SAMPLE] = INVALID_SAMPLE
    times = al.ds_frames.astype(np.int64)
    # frames whose kernel (60 lines) stays inside the trial
    times = times[(times > lines[0] + 60) & (times < lines[1] - 60)]
    acc_d = acc_e = 0.0
    drift = {j: [] for j in range(3)}
    for idx, frames, _ in iter_registered_frames(
        layout, samples, 0, al, times, 20, chunk=50
    ):
        f = frames.numpy()
        for j, (r, c) in enumerate(pos):
            ri, ci = int(round(r)), int(round(c))
            win = f[:, :, ri - 3 : ri + 4, ci - 3 : ci + 4]
            acc_d = acc_d + np.nansum(win[0])
            acc_e = acc_e + np.nansum(win[1])
            whole = ~np.isnan(win[2]).any((1, 2))  # not cut by an edge
            st = win[2][whole]
            rr, cc = np.mgrid[-3:4, -3:4]
            m = st.sum((1, 2))
            cr = (st * rr).sum((1, 2)) / m + ri - r
            cc_ = (st * cc).sum((1, 2)) / m + ci - c
            drift[j].append(np.stack([cr, cc_], 1))
    for j in range(3):
        d = np.concatenate(drift[j])
        assert d.shape[0] > 0.9 * times.size
        assert np.abs(d - np.median(d, axis=0)).max() < 0.1
        assert np.abs(np.median(d, axis=0)).max() < 0.1
    # the summed signal at the sources: Poisson error of photons x g
    se = np.sqrt(acc_e * s.raw_per_photon) * 3
    assert abs(acc_d - acc_e) < 5 * se


def test_registered_copy_keeps_the_motion(
    online, registered, tmp_path, monkeypatch
):
    """Registering the injected copy gives the original's motion up to the
    grid's anchoring; the truth then records both."""
    layout, _, _ = online
    out = tmp_path / "inj"
    run_injection(
        online,
        registered[True],
        out,
        monkeypatch,
        settings=replace(BRIGHT, size_classes=(1.0, 2.0)),
        positions=BRIGHT_AT,
    )
    copy = replace(
        layout, files=(str(out / "raw" / Path(layout.files[0]).name),)
    )
    monkeypatch.setattr(multiroi, "raster_layout", lambda path: copy)
    register_multiroi(
        out / "trial_table.h5",
        {"maxshift": 6, "compensate_online": True},
        execution={"device": "cpu", "max_workers": 1},
    )
    assert inj.main(["compare", str(out)]) == 0
    truth = inj.read_truth(out / inj.TRUTH_NAME)["paths"][1]
    for t in (1, 2):
        d = truth["trials_detail"][t]
        assert d["copy_motion_rms_px"] < 0.05
        np.testing.assert_allclose(
            d["positions_copy_rc"], d["positions_rc"] + d["copy_shift_rc"]
        )
        a = inj.read_alignment(d["alignment"])
        b = inj.read_alignment(d["copy_alignment"])
        off = inj.registration_offset(a, b)
        np.testing.assert_allclose(off["shift_rc"], d["copy_shift_rc"])


def test_saturation_invalid_samples_and_chunking(online, tmp_path):
    """Huge events saturate at 32767; invalid samples stay; results do not
    depend on the chunk size."""
    layout, offs, truth = online
    src = Path(layout.files[0])
    data = np.fromfile(src, np.int16)
    rng = np.random.default_rng(2)
    li = layout.slot_line - 1
    ch0 = layout.row_base[li] + layout.slot_pos
    rows = HEADER_WORDS + np.arange(N_CYCLES)[:, None] * layout.words_per_cycle
    w = (rows + ch0[None, :]).ravel()
    bad = w[rng.random(w.size) < 0.05]
    data[bad] = INVALID_SAMPLE
    outs = []
    lines = (1, layout.total_lines)
    geo = truth_geometry(layout, offs, truth, lines)
    tr = np.full((1, lines[1]), 2000.0)
    for chunk in (7, 64):
        dst = tmp_path / f"c{chunk}.dat"
        data.tofile(dst)
        lay = replace(layout, files=(str(dst),))
        model = inj.SampleModel(
            lay, geo, np.array([[18.0, 25.0]]), [inj.GaussianFootprint()], tr
        )
        st = inj.inject_samples(lay, model, 3.0, 0, (5, 1, 1), chunk)
        outs.append(np.fromfile(dst, np.int16))
        assert st.saturated > 0 and st.lost_raw > 0
        assert st.expected_photons_invalid > 0
    np.testing.assert_array_equal(outs[0], outs[1])
    assert np.all(outs[0][bad] == INVALID_SAMPLE)
    assert outs[0].max() == inj.SAMPLE_MAX


def test_command_line(online, registered, tmp_path, monkeypatch, capsys):
    """``python -m giant_python.simulation.injection`` runs a session."""
    layout, _, _ = online
    monkeypatch.setattr(inj, "raster_layout", lambda path: layout)
    pos = tmp_path / "pos.h5"
    with h5py.File(pos, "w") as f:
        f["path1"] = np.array([[16.0, 20.0]])
    code = inj.main(
        [
            "inject",
            str(registered[False]),
            str(tmp_path / "cli"),
            "--paths",
            "1",
            "--trials",
            "2",
            "--positions",
            str(pos),
            "--raw-per-photon",
            "4",
            "--sizes",
            "8",
            "--seed",
            "3",
            "--size-reference",
            "tissue",
            "--no-frame-traces",
        ]
    )
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["paths"]["1"]["n_sources"] == 1
    truth = inj.read_truth(tmp_path / "cli" / inj.TRUTH_NAME)
    np.testing.assert_array_equal(truth["paths"][1]["trials"], [2])
    p = truth["paths"][1]
    assert not p["trials_detail"][2]["compensated"]
    np.testing.assert_array_equal(p["frame_peak"], [1.0])
    assert "trace_frames" not in p["trials_detail"][2]


def test_cuda_render_matches_cpu(online, registered):
    """Rendering on CUDA equals the CPU."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    layout, _, _ = online
    table = Path(registered[True])
    al = inj.read_alignment(
        table.parent / "motion_correction" / "E1T2DMD1_ALIGNMENTDATA.h5"
    )
    s = read_raster_samples(layout, (0, layout.num_cycles))[:1]
    times = al.ds_frames[:40].astype(np.int64)
    cpu = next(iter_registered_frames(layout, s, 0, al, times, 20))[1]
    gpu = next(iter_registered_frames(layout, s, 0, al, times, 20, "cuda"))[1]
    np.testing.assert_allclose(
        gpu.cpu().numpy(), cpu.numpy(), rtol=1e-5, atol=1e-4, equal_nan=True
    )


def test_ragged_table_keeps_copy_columns(
    online, registered, tmp_path, monkeypatch
):
    """A trial without a recording is skipped; the others keep their
    column of the copy's trial table."""
    from giant_python.io.hdf5 import load_struct_h5

    table = load_struct_h5(registered[False])
    table["filename"] = np.array([["", table["filename"][0, 1]]], object)
    ragged = tmp_path / "trial_table.h5"
    save_struct_h5(ragged, table)
    s = inj.InjectionSettings(raw_per_photon=4.0, seed=1)
    run_injection(
        online,
        ragged,
        tmp_path / "out",
        monkeypatch,
        settings=s,
        positions={1: np.array([[16.0, 20.0]])},
        moco_dir=Path(registered[False]).parent / "motion_correction",
    )
    truth = inj.read_truth(tmp_path / "out" / inj.TRUTH_NAME)["paths"][1]
    assert list(truth["trials_detail"]) == [2]
    assert truth["trials_detail"][2]["copy_trial"] == 2
    np.testing.assert_array_equal(np.unique(truth["events"]["trial"]), [2])


def test_sizes_are_peaks_in_frames(online, injected):
    """With sizes in frame units, an isolated event's noiseless peak in the
    200 Hz frames at its source's peak pixel is its size, on average."""
    out, _, s = injected
    truth = inj.read_truth(out / inj.TRUTH_NAME)["paths"][1]
    fp = truth["frame_peak"]
    assert np.all((fp > 0.2) & (fp < 1.0))
    ev = truth["events"]
    reach = 24  # max(3 * 8, P) lines at 1600 Hz and 200 Hz frames
    tail = 6 * s.tau_s * 1600
    ratios = []
    for t in (1, 2):
        d = truth["trials_detail"][t]
        fl, tf = d["frame_lines"], d["trace_frames"]
        assert tf.shape == d["trace"].shape
        # the ideal trace is sampled at frame lines: within a frame of each
        # largest peak (overlapping events may add up)
        top = np.array(
            [
                ev["size"][(ev["trial"] == t) & (ev["source"] == j)].max()
                for j in range(3)
            ]
        )
        assert np.all(d["trace"].max(1) >= top * np.exp(-0.005 / s.tau_s))
        for j in range(3):
            on = ev["onset_line"][(ev["trial"] == t) & (ev["source"] == j)]
            size = ev["size"][(ev["trial"] == t) & (ev["source"] == j)]
            for k, (t0, a) in enumerate(zip(on, size)):
                alone = (k == 0 or t0 - on[k - 1] > 2 * tail) and (
                    k == on.size - 1 or on[k + 1] - t0 > tail + reach
                )
                sel = (fl >= t0 - reach) & (fl <= t0 + tail)
                if alone and sel.any():
                    ratios.append(np.nanmax(tf[j, sel]) / a)
    assert len(ratios) > 20
    assert np.mean(ratios) == pytest.approx(1.0, abs=0.06)


def test_refuses_to_write_into_the_original(online, registered, monkeypatch):
    """The copy's folder cannot be the original's data folder."""
    layout, _, _ = online
    src = Path(layout.files[0])
    out = src.parent.parent / "same"
    with pytest.raises(ValueError, match="original's data folder"):
        run_injection(
            online,
            registered[True],
            out,
            monkeypatch,
            data_dir=out / "raw",
        )


def test_registration_offset_includes_the_grid(registered):
    """Different padding, window and reference offset move the copy's
    positions; mixed compensation is refused."""
    from dataclasses import replace as dc_replace

    moco = Path(registered[True]).parent / "motion_correction"
    a = inj.read_alignment(moco / "E1T1DMD1_ALIGNMENTDATA.h5")
    comp = dict(a.compensation, refOffset=a.compensation["refOffset"] + 1)
    b = dc_replace(
        a,
        margin=a.margin + 3,
        trim_r0=a.trim_r0 + 1,
        motion_r=a.motion_r + 0.25,
        compensation=comp,
    )
    off = inj.registration_offset(a, b)
    # continuous motion unchanged in columns; rows: +0.25 recorded, -1 from
    # the reference (mapping swaps x and y)
    np.testing.assert_allclose(off["motion_offset_rc"], [0.25 - 1, -1.0])
    np.testing.assert_allclose(
        off["shift_rc"], [3 - 1 - 1 - (0.25 - 1), 3 - 1 + 1]
    )
    assert off["rms_px"] < 1e-9
    plain = inj.read_alignment(
        Path(registered[False]).parent
        / "motion_correction"
        / "E1T1DMD1_ALIGNMENTDATA.h5"
    )
    with pytest.raises(ValueError, match="online motion correction"):
        inj.registration_offset(a, plain)


def test_frame_motion_uses_each_frames_own_reference(registered):
    """A frame at a registered frame's time but with another reference
    offset is moved by its own reference, not the recorded motion."""
    from giant_python.simulation.render import frame_motion

    moco = Path(registered[True]).parent / "motion_correction"
    al = inj.read_alignment(moco / "E1T1DMD1_ALIGNMENTDATA.h5")
    t = al.ds_frames.astype(np.int64)
    c = al.compensation
    refs = np.stack([c["frameRefX"], c["frameRefY"]], 1)
    ok = np.isfinite(al.motion_r)
    same = frame_motion(al, t, refs)
    np.testing.assert_allclose(same[ok, 0], al.motion_r[ok], atol=1e-9)
    np.testing.assert_allclose(same[ok, 1], al.motion_c[ok], atol=1e-9)
    other = frame_motion(al, t, refs + np.array([1, 0]))  # x one more
    np.testing.assert_allclose(other[ok, 1], al.motion_c[ok] - 1, atol=1e-9)
    np.testing.assert_allclose(other[ok, 0], al.motion_r[ok], atol=1e-9)


def test_calibration_keeps_overlapping_sources_apart(online, registered):
    """Overlapping sources are calibrated in turns: a close neighbour does
    not change a source's frame peak."""
    layout, _, _ = online
    moco = Path(registered[True]).parent / "motion_correction"
    al = inj.read_alignment(moco / "E1T1DMD1_ALIGNMENTDATA.h5")
    geo = inj.trial_geometry(layout, al, (1, 4800))
    fp = inj.GaussianFootprint(1.33)
    s = inj.InjectionSettings(tau_s=0.012, calibration_s=2.5)
    pair = np.array([[16.0, 20.0], [16.0, 22.0], [10.0, 33.0]])
    np.testing.assert_array_equal(
        inj.overlap_classes(pair, [fp] * 3), [0, 1, 0]
    )
    both = inj.calibrate_frame_peaks(
        layout,
        al,
        geo,
        pair,
        (0.0, 0.0),
        [fp] * 3,
        s,
        np.random.default_rng(0),
    )
    alone = inj.calibrate_frame_peaks(
        layout,
        al,
        geo,
        pair[:1],
        (0.0, 0.0),
        [fp],
        s,
        np.random.default_rng(0),
    )
    assert both[0] == pytest.approx(alone[0], rel=0.05)


def test_inputs_that_would_corrupt_a_dataset_are_refused(
    online, registered, tmp_path, monkeypatch
):
    """Repeated trials or paths, and an output folder that already holds a
    dataset, are refused before anything is written."""
    s = inj.InjectionSettings(
        raw_per_photon=4.0, frame_traces=False, size_reference="tissue"
    )
    pos = {1: np.array([[16.0, 20.0]])}
    with pytest.raises(ValueError, match="must not repeat"):
        run_injection(
            online,
            registered[False],
            tmp_path / "a",
            monkeypatch,
            settings=s,
            positions=pos,
            trials=[1, 1],
        )
    with pytest.raises(ValueError, match="must not repeat"):
        inj.inject_session(
            registered[False],
            tmp_path / "b",
            paths=(1, 1),
            settings=s,
            positions=pos,
        )
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()
    out = tmp_path / "c"
    (out / "motion_correction").mkdir(parents=True)
    with pytest.raises(ValueError, match="already holds motion_correction"):
        run_injection(
            online,
            registered[False],
            out,
            monkeypatch,
            settings=s,
            positions=pos,
        )


def test_copy_table_paths_and_absolute_folders(tmp_path, monkeypatch):
    """The copy's table has absolute folders and no files for paths that
    were not copied."""
    from giant_python.io.hdf5 import load_struct_h5

    table = {
        "datadr": "x",
        "savedr": "y",
        "filename": np.array([["a1", "b1"], ["a2", "b2"]], dtype=object),
        "epoch": np.ones((2, 2)),
        "slap2_info": {"first_line": np.ones((2, 2))},
        "motion_correction": {"fn_adata": "z"},
    }
    monkeypatch.chdir(tmp_path)
    Path("rel").mkdir()
    p = inj.write_copy_table(
        table, np.array([1]), Path("rel/raw"), Path("rel"), paths=[1]
    )
    t = load_struct_h5(p)
    assert Path(t["datadr"]).is_absolute() and Path(t["savedr"]).is_absolute()
    assert Path(t["savedr"]) == tmp_path / "rel"
    np.testing.assert_array_equal(
        np.asarray(t["filename"], dtype=object).ravel(), ["b1", ""]
    )
    assert "motion_correction" not in t


def test_image_footprint_peak_and_calibration(online, registered):
    """An off-centre image footprint is calibrated at its maximum."""
    layout, _, _ = online
    img = np.zeros((5, 5))
    img[2, 4] = 1.0
    fp = inj.ImageFootprint(img)
    assert fp.peak_offset == (0.0, 2.0)
    pos = np.array([[16.0, 20.0]])
    np.testing.assert_array_equal(inj.peak_pixels(pos, [fp]), [[16, 22]])
    moco = Path(registered[True]).parent / "motion_correction"
    al = inj.read_alignment(moco / "E1T1DMD1_ALIGNMENTDATA.h5")
    geo = inj.trial_geometry(layout, al, (1, 4800))
    s = inj.InjectionSettings(tau_s=0.012, calibration_s=2.0)
    peak = inj.calibrate_frame_peaks(
        layout, al, geo, pos, (0.0, 0.0), [fp], s, np.random.default_rng(0)
    )
    assert 0.05 < peak[0] < 1.0


def test_expected_frames_follow_the_trial_grid(online, registered):
    """A path-frame pixel outside the trial grid's size is still rendered
    where its trial position has data; channel 1 renders with channel 0's
    freshness."""
    layout, _, _ = online
    moco = Path(registered[True]).parent / "motion_correction"
    al = inj.read_alignment(moco / "E1T1DMD1_ALIGNMENTDATA.h5")
    geo = inj.trial_geometry(layout, al, (1, 4800))
    fp = [inj.GaussianFootprint(1.33)]
    model = inj.SampleModel(
        layout, geo, np.array([[16.3, 20.4]]), fp, np.ones((1, 4800))
    )
    fl = np.arange(200, 1200, 8)
    here = inj.expected_frames(
        layout, al, model, fl, [[16, 20]], (0.3, 0.4), 200.0
    )
    far = inj.expected_frames(
        layout, al, model, fl, [[16 - 40, 20 + 60]], (40.3, -59.6), 200.0
    )
    assert al.shape[1] < 80 and np.isfinite(far).all()
    np.testing.assert_allclose(far, here, rtol=1e-6)
    ch1 = inj.expected_frames(
        layout, al, model, fl, [[16, 20]], (0.3, 0.4), 200.0, channel=1
    )
    np.testing.assert_allclose(ch1, here, rtol=1e-6)  # no invalid samples


def test_trace_frames_skip_kernels_reaching_a_neighbour_trial(injected):
    """Two trials of one file: frames whose kernel reads the other trial's
    injected samples are NaN in the truth's frame traces."""
    out, _, _ = injected
    truth = inj.read_truth(out / inj.TRUTH_NAME)["paths"][1]
    reach = 24
    t1, t2 = truth["trials_detail"][1], truth["trials_detail"][2]
    near1 = t1["frame_lines"] + reach >= 4801
    near2 = t2["frame_lines"] - reach <= 4800
    assert near1.any() and near2.any()
    assert np.isnan(t1["trace_frames"][:, near1]).all()
    assert np.isnan(t2["trace_frames"][:, near2]).all()
    assert np.isfinite(t1["trace_frames"][:, ~near1]).mean() > 0.9


def test_relative_registration_folder_is_stored_absolute(
    online, registered, tmp_path, monkeypatch
):
    """A relative registration folder is recorded as an absolute path."""
    import os

    reg = Path(registered[False]).parent
    monkeypatch.chdir(reg)
    s = inj.InjectionSettings(
        raw_per_photon=4.0, frame_traces=False, size_reference="tissue"
    )
    run_injection(
        online,
        registered[False],
        tmp_path / "out",
        monkeypatch,
        settings=s,
        positions={1: np.array([[16.0, 20.0]])},
        trials=[1],
        moco_dir="motion_correction",
    )
    truth = inj.read_truth(tmp_path / "out" / inj.TRUTH_NAME)
    al = Path(str(truth["paths"][1]["trials_detail"][1]["alignment"]))
    assert al.is_absolute() and al.exists()
    assert os.path.isabs(truth["origin"]["moco_dir"])


def test_more_inputs_are_checked(online, registered, tmp_path):
    """Non-finite gains are refused; sources without data cannot be
    calibrated; copies' filenames are their basenames; larger trial grids
    keep their data when shifted into the path frame."""
    from giant_python.io.hdf5 import load_struct_h5

    for bad in (float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite"):
            inj.InjectionSettings(raw_per_photon=bad)
    layout, _, _ = online
    moco = Path(registered[True]).parent / "motion_correction"
    al = inj.read_alignment(moco / "E1T1DMD1_ALIGNMENTDATA.h5")
    geo = inj.trial_geometry(layout, al, (1, 4800))
    fp = inj.GaussianFootprint(1.33)
    s = inj.InjectionSettings(tau_s=0.012, calibration_s=2.0)
    with pytest.raises(ValueError, match=r"sources \[1\] have no data"):
        inj.calibrate_frame_peaks(
            layout,
            al,
            geo,
            np.array([[16.0, 20.0], [16.0, 200.0]]),
            (0.0, 0.0),
            [fp, fp],
            s,
            np.random.default_rng(0),
        )
    table = {
        "filename": np.array([["sub/a_DMD1.dat"], [""]], dtype=object),
        "slap2_info": {"first_line": np.ones((2, 1))},
    }
    p = inj.write_copy_table(table, np.array([0]), tmp_path, tmp_path)
    names = np.asarray(load_struct_h5(p)["filename"], dtype=object).ravel()
    np.testing.assert_array_equal(names, ["a_DMD1.dat", ""])
    big = np.arange(10 * 112, dtype=float).reshape(10, 112)
    out = inj.shift_image(big, (0.0, 8.0), (10, 100))
    np.testing.assert_allclose(out, big[:, 8:108])


def test_bad_chunks_and_name_collisions_are_refused(
    online, registered, tmp_path, monkeypatch
):
    """A nonpositive chunk size, and two recordings that would be copied
    under one name, are refused before anything is written."""
    import shutil

    from giant_python.io.hdf5 import load_struct_h5

    layout, _, _ = online
    s = inj.InjectionSettings(
        raw_per_photon=4.0, frame_traces=False, size_reference="tissue"
    )
    with pytest.raises(ValueError, match="chunk_cycles"):
        run_injection(
            online,
            registered[False],
            tmp_path / "c",
            monkeypatch,
            settings=s,
            chunk_cycles=0,
        )
    assert not (tmp_path / "c").exists()
    data = tmp_path / "data"
    src = Path(layout.files[0])
    for sub in ("a", "b"):
        (data / sub).mkdir(parents=True)
        shutil.copyfile(src, data / sub / src.name)
    table = load_struct_h5(registered[False])
    table["filename"] = np.array(
        [[f"a/{src.name}", f"b/{src.name}"]], dtype=object
    )
    twins = tmp_path / "twins.h5"
    save_struct_h5(twins, table)
    monkeypatch.setattr(
        inj, "raster_layout", lambda path: replace(layout, files=(str(path),))
    )
    with pytest.raises(ValueError, match="would both be copied"):
        inj.inject_session(
            twins, tmp_path / "t", paths=(1,), settings=s, data_dir=data
        )
    assert not (tmp_path / "t").exists()


def test_neighbours_share_recording_files():
    """Trials are neighbours when their recordings share a file, whatever
    file each trial names."""
    ctx = {
        "names": ["c0.dat", "c1.dat", "other.dat"],
        "files": {
            0: ("c0.dat", "c1.dat"),
            1: ("c0.dat", "c1.dat"),
            2: ("other.dat",),
        },
        "first": [1.0, 101.0, 1.0],
        "last": [100.0, 200.0, 50.0],
    }
    near = inj._neighbours(ctx, [0, 1, 2])
    assert near(0) == [(101.0, 200.0)]
    assert near(1) == [(1.0, 100.0)]
    assert near(2) == []


def test_no_events_need_no_calibration(
    online, registered, tmp_path, monkeypatch
):
    """Without events nothing is calibrated, even in a trial too short to
    calibrate, and the copy stays identical."""
    layout, _, _ = online
    s = inj.InjectionSettings(rate_hz=0.0, edge_s=1.6, raw_per_photon=1.0)
    run_injection(
        online, registered[True], tmp_path / "o", monkeypatch, settings=s
    )
    src = Path(layout.files[0])
    assert sha(tmp_path / "o" / "raw" / src.name) == sha(src)
    truth = inj.read_truth(tmp_path / "o" / inj.TRUTH_NAME)["paths"][1]
    np.testing.assert_array_equal(truth["frame_peak"], 1.0)


def test_silo_frames_leave_out_the_kernels_edge(online, registered):
    """Frame traces and calibration use SILo's extraction frames, which
    leave out the samples exactly at the kernel's reach."""
    layout, _, _ = online
    moco = Path(registered[True]).parent / "motion_correction"
    al = inj.read_alignment(moco / "E1T1DMD1_ALIGNMENTDATA.h5")
    geo = inj.trial_geometry(layout, al, (1, 4800))
    model = inj.SampleModel(
        layout,
        geo,
        np.array([[16.0, 20.0]]),
        [inj.GaussianFootprint()],
        np.ones((1, 4800)),
    )
    fl = np.arange(300, 900, 8)
    got = inj.expected_frames(layout, al, model, fl, [[16, 21]], (0, 0), 200)
    n_c = layout.num_cycles
    smp = np.zeros((1, layout.slot_line.size, n_c), np.float32)
    smp[0][model.slots] = model.expected(np.arange(n_c)).T
    want = {}
    for exclusive in (True, False):
        out = []
        for _, f, _ in iter_registered_frames(
            layout, smp, 0, al, fl, 8, chunk=200, exclusive=exclusive
        ):
            out.append(f[0][:, 16, 21].numpy())
        want[exclusive] = np.concatenate(out)
    np.testing.assert_allclose(got[0], want[True], rtol=1e-6)
    assert np.max(np.abs(want[True] - want[False])) > 1e-6
