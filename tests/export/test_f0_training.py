"""F0 training-data export (giant export-f0) on a synthetic SILo run."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from giant_python.cli import main
from giant_python.export import f0_qc
from giant_python.export.f0_training import ExportOptions, export_run

from ..viewer.synthetic import BASELINE, CENTRES, N_TRIALS, write_session

N80 = 600


@pytest.fixture(scope="module", params=["rigid", "patchwise"])
def session(request, tmp_path_factory):
    """A synthetic session (rigid or patchwise maps) of 600 frames."""
    root = tmp_path_factory.mktemp(f"syn_{request.param}")
    return write_session(root, request.param, n80=N80)


def _export(syn, out, **kw):
    """Export the synthetic run with the given options."""
    return export_run(
        syn["run"],
        out,
        syn["run"] / "trial_table.h5",
        syn["moco"],
        syn["base"],
        options=ExportOptions(**kw),
        log=lambda line: None,
    )


def test_layout_matches_the_training_reader(session, tmp_path):
    """Datasets, shapes, dtypes and attributes as slap2_f0.io.TrialFile
    reads them (the reader is not a dependency: its contract is checked
    here)."""
    s = _export(session, tmp_path, kernel="exporter")
    assert [q["trial"] for q in s["trials"]] == [1, 2]
    for q in s["trials"]:
        assert "error" not in q
        with h5py.File(q["file"], "r") as f:
            n_ch, T, R, C = f["movie"].shape
            assert (n_ch, T) == (1, N80) and f["movie"].dtype == np.float32
            assert f["activity"].shape == f["movie"].shape
            for m in ("valid", "labeled", "footprint_support"):
                assert f[f"mask/{m}"].shape == (R, C)
                assert f[f"mask/{m}"].dtype == np.uint8
            assert f["mean_image"].shape == (1, R, C)
            assert f["varfac"].shape == (R, C)
            for k in ("discard", "valid_mean", "rec_neg_err"):
                assert f[f"frame/{k}"].shape == (T,)
            assert f["sources/peak_row"].shape == (CENTRES.shape[0],)
            for k in (
                "label",
                "path",
                "trial",
                "peakR",
                "nSources",
                "alignHz",
                "channels",
                "activityChannel",
                "activityGain",
                "kernel",
                "exporter",
            ):
                assert k in f.attrs
            assert int(f.attrs["trial"]) == q["trial"]
            assert f["movie"].chunks == (1, 256, min(64, R), min(64, C))
        name = Path(q["file"])
        assert (name.parent.name, name.name) == (
            "Path1",
            f"trial_{q['trial']:03d}.h5",
        )


def test_target_is_the_baseline(session, tmp_path):
    """With the movie's own kernel and cubic placement the noiseless
    target (movie - activity) is the baseline to 2% of the activity's
    peak; QC says the activity matches the movie."""
    s = _export(session, tmp_path, kernel="exporter", interp="cubic")
    for q in s["trials"]:
        with h5py.File(q["file"], "r") as f:
            m, a = f["movie"][0], f["activity"][0]
            sup = f["mask/footprint_support"][()].astype(bool)
        res = (m - a - BASELINE)[:, sup]
        assert np.nanmax(np.abs(res)) < 0.02 * a.max()
        assert q["peakR"] > 0.99 and q["diskR"] > 0.99
        assert q["diskSlope"] == pytest.approx(1.0, abs=0.01)
        assert q["peakSlope"] == pytest.approx(1.0, abs=0.02)
        assert abs(q["eventResidualFraction"]) < 0.02
        assert q["blurShapeRatio"] == pytest.approx(1.0, abs=0.02)
        assert q["targetOverBaselineSupport"] == pytest.approx(1.0, abs=1e-3)


def test_linear_placement_blurs_the_peaks(session, tmp_path):
    """Bilinear placement keeps the flux but lowers the peak pixels (the
    viewer's rendering); cubic keeps both."""
    lin = _export(session, tmp_path / "lin", kernel="exporter")
    cub = _export(session, tmp_path / "cub", kernel="exporter", interp="cubic")
    for a, b in zip(lin["trials"], cub["trials"]):
        assert a["diskSlope"] == pytest.approx(b["diskSlope"], abs=0.02)
        assert a["blurShapeRatio"] > b["blurShapeRatio"]
        assert a["eventResidualFraction"] > b["eventResidualFraction"]


def test_exact_kernel_is_sharper(session, tmp_path):
    """The exact kernel keeps the activity's total and sharpens its
    peaks against the MATLAB exporter's kernel."""
    e = _export(session, tmp_path / "e", kernel="exporter", interp="cubic")
    x = _export(session, tmp_path / "x", kernel="exact", interp="cubic")
    for qe, qx in zip(e["trials"], x["trials"]):
        assert qx["kernelR"] == pytest.approx((54 / 134) ** 2)
        assert np.isnan(qe["kernelR"]) or qe["kernelR"] == 0
        with (
            h5py.File(qe["file"], "r") as fe,
            h5py.File(qx["file"], "r") as fx,
        ):
            ae, ax = fe["activity"][0], fx["activity"][0]
        assert ax.sum() == pytest.approx(ae.sum(), rel=0.02)
        assert ax.max() > ae.max()


def test_summary_and_skipped_trial(tmp_path):
    """export_summary.json holds every trial (NaN as null); a trial
    without an across-trial map is reported, not written."""
    syn = write_session(tmp_path / "syn", "rigid", n80=200)
    out = tmp_path / "out"
    _export(syn, out)  # a first export writes both trials
    assert (out / "Path1" / "trial_002.h5").exists()
    es = syn["run"] / "source_extraction" / "experiment_summary.h5"
    with h5py.File(es, "r+") as f:
        a = f["Path1/frame_info/trial_alignment_affine"][()]
        a[..., 1] = np.nan  # MATLAB orientation: trials last
        f["Path1/frame_info/trial_alignment_affine"][...] = a
    s = _export(syn, out)
    by_trial = {q["trial"]: q for q in s["trials"]}
    assert "error" not in by_trial[1]
    assert "map" in by_trial[2]["error"]
    assert "removed" in by_trial[2]["error"]  # no stale file is left
    assert not (out / "Path1" / "trial_002.h5").exists()
    text = (out / "export_summary.json").read_text()
    assert "NaN" not in text
    j = json.loads(text)
    assert len(j["trials"]) == N_TRIALS and j["options"]["kernel"] == "exact"


def test_default_export_reconciles_trial_files(tmp_path):
    """A default export removes earlier trial files it does not write and
    lists them; an export with chosen trials leaves other files alone."""
    syn = write_session(tmp_path / "syn", "rigid", n80=200)
    out = tmp_path / "out"
    folder = out / "Path1"
    folder.mkdir(parents=True)
    for name in ("trial_009.h5", "trial_1000.h5", "notes.txt"):
        (folder / name).write_bytes(b"old")
    s = _export(syn, out, trials=(1,))
    assert "removed" not in s
    assert (folder / "trial_009.h5").exists()
    s = _export(syn, out)
    assert s["removed"] == {"Path1": ["trial_009.h5", "trial_1000.h5"]}
    assert sorted(f.name for f in folder.iterdir()) == [
        "notes.txt",
        "trial_001.h5",
        "trial_002.h5",
    ]
    j = json.loads((out / "export_summary.json").read_text())
    assert j["removed"] == s["removed"]


def test_repeated_selectors_export_once(tmp_path):
    """Repeated paths and trials give one job each (two workers never
    write the same file)."""
    syn = write_session(tmp_path / "syn", "rigid", n80=200)
    s = _export(
        syn, tmp_path / "out", paths=(1, 1), trials=(2, 2, 1), workers=2
    )
    assert [(q["path"], q["trial"]) for q in s["trials"]] == [(1, 1), (1, 2)]
    assert all("error" not in q for q in s["trials"])


@pytest.mark.parametrize(
    "kw",
    [
        {"trials": (0,)},
        {"trials": (-1,)},
        {"trials": (N_TRIALS + 1,)},
        {"paths": (2,)},
    ],
)
def test_selectors_outside_the_run_are_refused(tmp_path, kw):
    """Trials outside 1..the table's trials and paths the run does not
    have are refused before any job starts; the CLI reports a usage
    error."""
    syn = write_session(tmp_path / "syn", "rigid", n80=200)
    with pytest.raises(ValueError, match="trials|paths"):
        _export(syn, tmp_path / "out", **kw)
    assert not (tmp_path / "out" / "Path1").exists()
    flag, values = next(iter(kw.items()))
    with pytest.raises(SystemExit) as e:
        main(
            [
                "export-f0",
                str(syn["run"]),
                str(tmp_path / "o2"),
                "--table",
                str(syn["run"] / "trial_table.h5"),
                "--moco-dir",
                str(syn["moco"]),
                f"--{flag}",
                *[str(v) for v in values],
            ]
        )
    assert e.value.code == 2


def test_ignore_sources(tmp_path):
    """Without sources: zero activity, no sources group, no QC."""
    syn = write_session(tmp_path / "syn", "rigid", n80=200)
    s = _export(syn, tmp_path / "out", ignore_sources=True)
    for q in s["trials"]:
        assert q["nSources"] == 0 and np.isnan(q["peakR"])
        with h5py.File(q["file"], "r") as f:
            assert not f["activity"][()].any()
            assert "sources" not in f
            assert not f["mask/footprint_support"][()].any()


def test_cli(tmp_path):
    """giant export-f0 writes the trials and returns 0."""
    syn = write_session(tmp_path / "syn", "rigid", n80=200)
    out = tmp_path / "out"
    rc = main(
        [
            "export-f0",
            str(syn["run"]),
            str(out),
            "--table",
            str(syn["run"] / "trial_table.h5"),
            "--moco-dir",
            str(syn["moco"]),
            "--kernel",
            "matched",
            "--interp",
            "cubic",
            "--label",
            "syn",
        ]
    )
    assert rc == 0
    with h5py.File(out / "Path1" / "trial_001.h5", "r") as f:
        assert f.attrs["label"] == "syn" and f.attrs["kernel"] == "matched"


def test_movmedian_omitnan():
    """The running median ignores NaN and shrinks at the ends."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 300))
    x[0, 50:60] = np.nan
    x[2, :] = np.nan
    m = f0_qc.movmedian_omitnan(x, 81)
    for i in range(2):
        for t in (0, 10, 55, 150, 299):
            w = x[i, max(0, t - 40) : t + 41]
            assert m[i, t] == pytest.approx(np.median(w[~np.isnan(w)]))
    assert np.isnan(m[2]).all()


def test_event_residual_and_peaks():
    """The event-triggered residual is 0 when the activity is the movie's
    signal and 1 when no activity is removed."""
    from scipy import sparse

    T = 400
    x = np.zeros((1, T))
    for e in (50, 150, 250, 350):
        x[0, e : e + 20] += 10 * np.exp(-np.arange(20) / 4.0)
    H = sparse.csr_matrix(np.array([[0.2], [0.5], [0.3]]))
    Y = (np.asarray(H @ x).T + 5.0).astype(np.float64)
    pk = f0_qc.peak_pixels(H)
    assert pk.tolist() == [1]
    frac, n = f0_qc.event_residual(H, x, Y, pk)
    assert n > 0 and abs(frac) < 1e-9
    frac0, _ = f0_qc.event_residual(H, x, Y + np.asarray(H @ x).T, pk)
    assert frac0 == pytest.approx(0.5)  # half of the doubled movie left
    sl, r = f0_qc.peak_fit(H, x, Y, pk)
    assert sl[0] == pytest.approx(1.0) and r[0] == pytest.approx(1.0)


def _two_channels(syn) -> None:
    """Rewrite the synthetic movies with a second channel (2 x the first
    + 5), interleaved by frame, and set ``numChannels`` 2."""
    for mf in sorted(syn["moco"].glob("*_REGISTERED_DOWNSAMPLED-80Hz.h5")):
        with h5py.File(mf, "r+") as f:
            a = f["mov"][()]  # (pages, cols, rows)
            del f["mov"]
            st = np.empty((2 * a.shape[0],) + a.shape[1:], a.dtype)
            st[0::2], st[1::2] = a, 2 * a + 5
            f["mov"] = st
    es = syn["run"] / "source_extraction" / "experiment_summary.h5"
    with h5py.File(es, "r+") as f:
        f["params/numChannels"][...] = 2


def test_written_channels_and_mean_image(tmp_path):
    """The mean image follows the written channels, as movie and
    activity do (the reader indexes all three alike)."""
    syn = write_session(tmp_path / "syn", "rigid", n80=200)
    _two_channels(syn)
    s = _export(syn, tmp_path / "c2", channels=(2,))
    with h5py.File(s["trials"][0]["file"], "r") as f:
        m, mi = f["movie"][()], f["mean_image"][()]
        assert m.shape[0] == 1 and mi.shape[0] == 1
        np.testing.assert_allclose(mi[0], np.nanmean(m[0], 0), rtol=1e-4)
        assert int(f.attrs["activityChannel"]) == 1
    s = _export(syn, tmp_path / "c21", channels=(2, 1))
    with h5py.File(s["trials"][0]["file"], "r") as f:
        mi = f["mean_image"][()]
        np.testing.assert_allclose(mi[0], 2 * mi[1] + 5, rtol=1e-4)


@pytest.mark.parametrize("channels", [(0,), (3,), (1, 2)])
def test_channels_outside_the_movie_are_refused(tmp_path, channels):
    """Channels are one-based and within numChannels (1 here)."""
    syn = write_session(tmp_path / "syn", "rigid", n80=200)
    with pytest.raises(ValueError, match="channels"):
        _export(syn, tmp_path / "out", channels=channels)
    with pytest.raises(SystemExit) as e:  # the CLI reports it as a usage error
        main(
            [
                "export-f0",
                str(syn["run"]),
                str(tmp_path / "o2"),
                "--table",
                str(syn["run"] / "trial_table.h5"),
                "--moco-dir",
                str(syn["moco"]),
                "--channels",
                *[str(c) for c in channels],
            ]
        )
    assert e.value.code == 2
