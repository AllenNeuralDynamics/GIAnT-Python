"""Online-motion compensation (compensateOnline) on synthetic data."""

import numpy as np
import pytest
import torch

from giant_python.registration import multiroi
from giant_python.registration.compensation import (
    dmd_y,
    frame_offsets,
    line_offsets,
)
from giant_python.registration.sources import make_source

from .synthetic import make_layout, online_offsets, write_recording


def test_dmd_y_leads_header_switches():
    """y switches 2 lines early, 6 with x, never before the last switch."""
    x = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 7, 7])
    sw = np.flatnonzero((np.diff(x) != 0) | (np.diff(y) != 0)) + 2
    got = dmd_y(x, y, sw)
    # y switch at line index 8 (x unchanged): lines 6-7 take it; the
    # switch at 14 (x unchanged): lines 12-13; none before index 10 (x)
    want = y.copy()
    want[6:8] = 5
    want[12:14] = 7
    np.testing.assert_array_equal(got, want)
    x2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 3, 3])
    y2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 4, 4])
    sw2 = np.array([9])
    np.testing.assert_array_equal(
        dmd_y(x2, y2, sw2), [0, 0, 4, 4, 4, 4, 4, 4, 4, 4]
    )


@pytest.fixture(scope="module")
def online_recording(tmp_path_factory):
    """Synthetic recording whose online offsets move the DMD pattern."""
    path = tmp_path_factory.mktemp("rec") / "online_DMD1-TRIAL000001.dat"
    layout = make_layout(path, n_cycles=1400, seed=5)
    online = online_offsets(layout.total_lines)
    truth = write_recording(layout, seed=6, online=online)
    return layout, online, truth


def test_line_offsets_reproduce_dmd_timing(online_recording):
    """Headers decode, y_eff equals the generator's DMD y."""
    layout, (x, y, y_dmd), _ = online_recording
    off = line_offsets(layout, (1, layout.total_lines))
    np.testing.assert_array_equal(off.x, x)
    np.testing.assert_array_equal(off.y, y)
    np.testing.assert_array_equal(off.y_eff, y_dmd)
    part = line_offsets(layout, (2000, 3000))
    lo, hi = part.lines_read
    assert lo <= 2000 - 7 and hi >= 3000 + 7
    # exact on the requested lines
    np.testing.assert_array_equal(part.y_eff[1999:3000], y_dmd[1999:3000])
    assert part.x[0] == x[lo - 1] and part.x[-1] == x[hi - 1]


def frames_reference(layout, former_samples, offsets, times, dt, box):
    """Compensated weighted means by the definition (slow loops)."""
    p, total = layout.lines_per_cycle, layout.total_lines
    r0, r1, c0, c1 = box
    h, w = r1 - r0 + 1, c1 - c0 + 1
    refs = frame_offsets(offsets, times, dt, p)
    dt_read = max(3 * dt, p)
    num = np.zeros((layout.num_channels, len(times), h, w))
    den = np.zeros_like(num)
    pr = (layout.slot_pixel - 1) // layout.n_cols
    pc = (layout.slot_pixel - 1) % layout.n_cols
    for f, t in enumerate(times):
        for ln in range(max(1, t - dt_read), min(total, t + dt_read) + 1):
            wt = np.exp(-abs(t - ln) / dt)
            sel = np.flatnonzero(layout.slot_line == (ln - 1) % p + 1)
            r = pr[sel] + offsets.y_eff[ln - 1] - refs[f, 1] - r0
            c = pc[sel] + offsets.x[ln - 1] - refs[f, 0] - c0
            ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
            for ch in range(layout.num_channels):
                xv = former_samples[ch, sel, (ln - 1) // p].astype(float)
                np.add.at(num[ch, f], (r[ok], c[ok]), wt * xv[ok])
                np.add.at(den[ch, f], (r[ok], c[ok]), wt)
    return num, den


def test_compensated_frames_match_definition(online_recording):
    """Each sample moves by its line's offset minus the frame's."""
    from giant_python.io.slap2_raster import read_raster_samples
    from giant_python.registration.frames import FrameFormer

    layout, _, _ = online_recording
    samples = read_raster_samples(layout, (0, layout.num_cycles))
    former = FrameFormer(layout, samples, 0)
    times = np.array([400, 1210, 3333, 5000])
    src = make_source(layout, former, 300, 6000, times, times, 20, True)
    vals, fresh = src.frames(times)
    box = (
        former.rows.min(),
        former.rows.max(),
        former.cols.min(),
        former.cols.max(),
    )
    num, den = frames_reference(
        layout, samples, src.compensation, times, 20, box
    )
    with np.errstate(invalid="ignore"):
        want = (num / den).astype(np.float32)
    got = src.window.image(vals, float("nan")).numpy()
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_allclose(got, want, rtol=1e-6)
    np.testing.assert_allclose(
        src.window.image(fresh, 0.0).numpy(), den[0], rtol=1e-6
    )


def total_motion_error(res, adata, truth, header):
    """RMS error of the total motion against the kernel-averaged truth.

    Compensated: motion plus each frame's reference offset (relative to
    ``refOffset``). Uncompensated: motion plus the header offsets at the
    frame's line (``header = (x, y)``), as GIAnT-MATLAB's transplant
    evaluation adds them.
    """
    dt = 20
    kernel = np.exp(-np.abs(np.arange(-3 * dt, 3 * dt + 1)) / dt)
    kernel /= kernel.sum()
    true = np.stack(
        [np.convolve(truth[:, i], kernel, "same") for i in (0, 1)]
    )[:, res.ds_frames - 1]
    est = np.stack([res.motion_r, res.motion_c])
    oc = adata["slap2"]["onlineCompensation"]
    if oc["enabled"].item():
        ref = oc["refOffset"].ravel()
        est = est + np.stack(
            [
                oc["frameRefY"].ravel() - ref[1],
                oc["frameRefX"].ravel() - ref[0],
            ]
        )
    else:
        x, y = header
        est = est + np.stack([y[res.ds_frames - 1], x[res.ds_frames - 1]])
    err = est - true
    err -= np.median(err, axis=1, keepdims=True)
    return float(np.sqrt(np.mean(err**2)))


def test_compensated_registration_recovers_motion(
    online_recording, tmp_path, monkeypatch
):
    """Total motion follows the truth with compensation, not without."""
    from giant_python.io.hdf5 import load_struct_h5

    layout, (x, y, _), truth = online_recording
    monkeypatch.setattr(multiroi, "raster_layout", lambda path: layout)
    errors = {}
    for comp in (True, False):
        params = multiroi.MultiRoiRegistrationParams(
            maxshift=6, compensate_online=comp
        )
        res = multiroi.register_trial(
            layout.files[0],
            1,
            9000,
            tmp_path / f"{comp}.tif",
            tmp_path / f"{comp}.h5",
            params,
        )
        adata = load_struct_h5(tmp_path / f"{comp}.h5")
        errors[comp] = total_motion_error(res, adata, truth, (x, y))
        oc = adata["slap2"]["onlineCompensation"]
        assert oc["enabled"].item() == int(comp)
    oc = load_struct_h5(tmp_path / "True.h5")["slap2"]["onlineCompensation"]
    assert np.ptp(oc["frameRefX"]) >= 1 and np.ptp(oc["frameRefY"]) >= 1
    assert errors[True] < 0.1
    assert errors[True] < errors[False] / 3


def test_compensation_on_cuda_matches_cpu(online_recording):
    """Compensated frames agree on CUDA and CPU (skipped without a GPU)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from giant_python.io.slap2_raster import read_raster_samples
    from giant_python.registration.frames import FrameFormer

    layout, _, _ = online_recording
    samples = read_raster_samples(layout, (0, layout.num_cycles))
    times = np.array([500, 2500, 4100])
    out = []
    for dev in ("cpu", "cuda"):
        former = FrameFormer(layout, samples, 0, device=dev)
        src = make_source(layout, former, 300, 6000, times, times, 20, True)
        out.append(src.frames(times)[0].cpu().numpy())
    np.testing.assert_allclose(out[0], out[1], rtol=1e-6, equal_nan=True)
