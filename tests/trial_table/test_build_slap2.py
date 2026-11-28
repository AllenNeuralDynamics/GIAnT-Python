"""Tests of the SLAP2 trial-table builder (ports of GIAnT-MATLAB's tests).

The per-file reader is replaced by a table of line counts and first-line
times, so no SLAP2 data are needed; reference stacks are small TIFFs with
SLAP2's per-page JSON descriptions. A separate test builds a MATLAB v7.3
``datetime`` by hand to check the ``.meta`` decoder.
"""

import datetime
import json
import math
import os
import struct
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from giant_python.io.hdf5 import load_struct_h5
from giant_python.io.slap2 import first_line_timestamp, read_mat_datetime_ms
from giant_python.pipeline import trial_table as tt
from giant_python.pipeline.trial_table import (
    DatInfo,
    TrialTableWarning,
    acquisition_order,
    build_trial_table_slap2,
    matlab_linspace,
    trim_trial_table,
)

LINE_PERIOD_S = 1e-4
START = datetime.datetime(2024, 2, 1, 12, 0, 0)


def datenum(t: datetime.datetime) -> float:
    """MATLAB datenum of a naive local datetime."""
    midnight = t.replace(hour=0, minute=0, second=0, microsecond=0)
    return t.toordinal() + 366 + (t - midnight).total_seconds() / 86400


class FakeSession:
    """A data folder whose .dat files are empty, with a fake reader."""

    def __init__(self, root: Path, monkeypatch):
        """Create the data folder and put the fake reader in place."""
        self.dir = root / "data"
        self.dir.mkdir()
        self.info = {}
        monkeypatch.setattr(tt, "read_dat_info", self.read)

    def read(self, path):
        """The fake reader: a file's line count and first-line time."""
        lines, first = self.info[Path(path).name]
        return DatInfo(lines, first, LINE_PERIOD_S)

    def dat(self, name, lines, mtime, first_line=None):
        """Add an empty .dat file with its line count, file time and first-line
        time."""
        p = self.dir / name
        p.write_bytes(b"")
        stamp = mtime.timestamp()
        os.utime(p, (stamp, stamp))
        first = math.nan if first_line is None else datenum(first_line)
        self.info[name] = (lines, first)

    def triggered(self, lines1, lines2):
        """Add triggered recordings of both paths with the given line
        counts."""
        dmd1 = [
            f"acq_DMD1-TRIAL{k:06d}.dat" for k in range(1, len(lines1) + 1)
        ]
        dmd2 = [
            f"acq_DMD2-TRIAL{k:06d}.dat" for k in range(1, len(lines2) + 1)
        ]
        for k, (name, n) in enumerate(zip(dmd1, lines1), start=1):
            self.dat(name, n, START + datetime.timedelta(minutes=k))
        for k, (name, n) in enumerate(zip(dmd2, lines2), start=1):
            self.dat(name, n, START + datetime.timedelta(minutes=k))
        return dmd1, dmd2

    def references(self):
        """Write a small reference stack per path, described as SLAP2 does."""
        import tifffile

        ref = self.dir / "reference"
        ref.mkdir()
        rng = np.random.default_rng(0)
        for dmd in (1, 2):
            with tifffile.TiffWriter(
                ref / f"acq_DMD{dmd}_CONFIG1-REFERENCE.tif"
            ) as tw:
                for z in (0, 1, 2):
                    for ch in (1, 2):
                        desc = {
                            "channel": ch,
                            "z": z,
                            "dmdPixel2SampleTransform": np.eye(3).tolist(),
                        }
                        page = rng.random((8, 6)).astype(np.float32)
                        tw.write(
                            page, description=json.dumps(desc), metadata=None
                        )

    def mtimes(self, names):
        """The files' modification times as MATLAB datenums."""
        return np.array([tt._mtime_datenum(self.dir / n) for n in names])

    def build(self, save_dir, **kw):
        """Build the trial table of this folder."""
        return build_trial_table_slap2(self.dir, save_dir, **kw)


@pytest.fixture
def session(tmp_path, monkeypatch):
    """A fake session in a temporary folder."""
    return FakeSession(tmp_path, monkeypatch)


def test_triggered_trials(session, tmp_path):
    """Triggered recordings of both paths give one trial per file pair."""
    lines = [100000, 120000, 110000]
    dmd1, dmd2 = session.triggered(lines, lines)
    session.references()
    t = session.build(tmp_path / "out")
    assert t["filename"].tolist() == [dmd1, dmd2]
    np.testing.assert_array_equal(t["epoch"], np.ones((2, 3)))
    np.testing.assert_array_equal(
        t["true_trial_ix"], np.tile([1, 2, 3], (2, 1))
    )
    s2 = t["slap2_info"]
    np.testing.assert_array_equal(s2["first_line"], np.ones((2, 3)))
    np.testing.assert_array_equal(s2["last_line"], np.tile(lines, (2, 1)))
    ends = session.mtimes(dmd1)
    np.testing.assert_allclose(
        s2["trial_end_time_from_pc"].ravel(), ends, rtol=0, atol=1e-9
    )
    np.testing.assert_allclose(
        s2["trial_start_time_inferred"].ravel(),
        ends - np.array(lines) * LINE_PERIOD_S / 86400,
        rtol=0,
        atol=1e-9,
    )


def test_triggered_trial_line_mismatch(session, tmp_path):
    """A file holding two trials back to back is lined up with the other
    path."""
    # DMD1's trial number failed to increment: its second file holds DMD2's
    # second and third trials back to back
    dmd1, dmd2 = session.triggered([100000, 220000], [100000, 100000, 120000])
    session.references()
    s = session.build(tmp_path / "out")
    assert s["filename"].tolist() == [[dmd1[0], dmd1[1], dmd1[1]], dmd2]
    s2 = s["slap2_info"]
    np.testing.assert_array_equal(
        s2["first_line"], [[1, 1, 100001], [1, 1, 1]]
    )
    np.testing.assert_array_equal(
        s2["last_line"], [[100000, 100000, 220000], [100000, 100000, 120000]]
    )
    dn1, dn2 = session.mtimes(dmd1), session.mtimes(dmd2)
    np.testing.assert_allclose(
        s2["trial_end_time_from_pc"].ravel(),
        [dn1[0], dn2[1], dn1[1]],
        rtol=0,
        atol=1e-9,
    )


def test_order_from_file_names_not_mod_times(session, tmp_path):
    """Acquisitions are ordered by the time in their names, not by file
    times."""
    # copies carry upload times: DMD1's are scrambled within a few seconds,
    # DMD2's identical; the earlier acquisition sorts last alphabetically
    acqs = [("spont_20250314_101500", 2), ("activity_20250314_111223", 12)]
    dmd1 = [
        f"{a}_DMD1-TRIAL{k:06d}.dat" for a, n in acqs for k in range(1, n + 1)
    ]
    dmd2 = [n.replace("DMD1", "DMD2") for n in dmd1]
    lines = 100000 + 100 * np.arange(1, len(dmd1) + 1)
    upload = datetime.datetime(2026, 8, 27, 10, 53, 33)
    for k, (a, b) in enumerate(zip(dmd1, dmd2), start=1):
        session.dat(
            a, lines[k - 1], upload + datetime.timedelta(seconds=(7 * k) % 5)
        )
        session.dat(b, lines[k - 1], upload)
    session.references()
    with pytest.warns(
        TrialTableWarning, match="do not look like acquisition times"
    ):
        t = session.build(tmp_path / "out")
    assert t["filename"].tolist() == [dmd1, dmd2]
    np.testing.assert_array_equal(
        t["true_trial_ix"][0], np.arange(1, len(dmd1) + 1)
    )
    np.testing.assert_array_equal(
        t["slap2_info"]["last_line"], np.tile(lines, (2, 1))
    )
    np.testing.assert_allclose(
        t["slap2_info"]["trial_end_time_from_pc"].ravel(),
        session.mtimes(dmd1),
        rtol=0,
        atol=1e-9,
    )


def test_file_order_falls_back_to_mod_time(session, tmp_path):
    """Without a time in the names, files are ordered by modification time."""
    names = ["b_DMD1.dat", "a_DMD1.dat"]
    for k, n in enumerate(names, start=1):
        t = START + datetime.timedelta(minutes=k)
        session.dat(n, 100000, t)
        session.dat(n.replace("DMD1", "DMD2"), 100000, t)
    session.references()
    with pytest.warns(TrialTableWarning, match="modification time"):
        t = session.build(tmp_path / "out")
    assert t["filename"].tolist() == [
        names,
        [n.replace("DMD1", "DMD2") for n in names],
    ]


def test_trial_times_from_line_clock(session, tmp_path):
    """Trial times come from the line clock, which shows gaps the line counts
    hide."""
    # DMD1's second file holds trials 2 and 3, with a 0.5 s gap that the line
    # count alone doesn't show; mtimes are identical upload times
    starts = [
        datetime.datetime(2025, 3, 14, 11, 12, 55)
        + datetime.timedelta(seconds=s)
        for s in (0, 10.5, 21)
    ]
    upload = datetime.datetime(2026, 8, 27, 10, 53, 33)
    dmd1 = [f"activity_20250314_111223_DMD1-TRIAL{k:06d}.dat" for k in (1, 2)]
    dmd2 = [
        f"activity_20250314_111223_DMD2-TRIAL{k:06d}.dat" for k in (1, 2, 3)
    ]
    session.dat(dmd1[0], 100000, upload, starts[0])
    session.dat(dmd1[1], 220000, upload, starts[1])
    for name, n, s in zip(dmd2, (100000, 100000, 120000), starts):
        session.dat(name, n, upload, s)
    session.references()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        t = session.build(tmp_path / "out")
    assert t["filename"].tolist() == [[dmd1[0], dmd1[1], dmd1[1]], dmd2]
    days = np.array([datenum(s) for s in starts])
    s2 = t["slap2_info"]
    np.testing.assert_allclose(
        s2["trial_start_time_inferred"].ravel(), days, rtol=0, atol=1e-9
    )
    np.testing.assert_allclose(
        s2["trial_end_time_from_pc"].ravel(),
        days + np.array([100000, 100000, 120000]) * LINE_PERIOD_S / 86400,
        rtol=0,
        atol=1e-9,
    )


def test_multicycle_trials_and_times(session, tmp_path):
    """A continuous recording is cut into trials with times from the line
    clock."""
    lines = 450000
    t0 = datetime.datetime(2025, 3, 14, 11, 12, 55)
    names = ["acq_DMD1-CYCLE-000000.dat", "acq_DMD2-CYCLE-000000.dat"]
    for n in names:
        session.dat(n, lines, START, t0)
    # later files of the recording are skipped (their line counts unknown)
    for n in ("acq_DMD1-CYCLE-000001.dat", "acq_DMD2-CYCLE-000001.dat"):
        (session.dir / n).write_bytes(b"")
    session.references()
    t = session.build(tmp_path / "out")
    edges = np.array([1, 150001, 300001, 450001], dtype=float)
    assert t["filename"].tolist() == [[names[0]] * 3, [names[1]] * 3]
    s2 = t["slap2_info"]
    np.testing.assert_array_equal(s2["first_line"], np.tile(edges[:3], (2, 1)))
    np.testing.assert_array_equal(s2["last_line"], np.tile(edges[1:], (2, 1)))
    days = datenum(t0) + (edges - 1) * LINE_PERIOD_S / 86400
    np.testing.assert_allclose(
        s2["trial_start_time_inferred"].ravel(), days[:3], rtol=0, atol=1e-9
    )
    np.testing.assert_allclose(
        s2["trial_end_time_from_pc"].ravel(), days[1:], rtol=0, atol=1e-9
    )


def test_multicycle_times_from_mtime(session, tmp_path):
    """Without the line clock, a continuous recording's times come from file
    times."""
    lines = 450000
    names = ["acq_DMD1-CYCLE-000000.dat", "acq_DMD2-CYCLE-000000.dat"]
    for n in names:
        session.dat(n, lines, START)
    session.references()
    s2 = session.build(tmp_path / "out")["slap2_info"]
    edges = np.array([1, 150001, 300001, 450001], dtype=float)
    dn = session.mtimes(names[:1])[0]
    to_days = LINE_PERIOD_S / 86400
    np.testing.assert_allclose(
        s2["trial_end_time_from_pc"].ravel(),
        dn - (lines - edges[1:]) * to_days,
        rtol=0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        s2["trial_start_time_inferred"].ravel(),
        dn - (lines - edges[:3]) * to_days,
        rtol=0,
        atol=1e-9,
    )


def test_multicycle_times_from_last_file_mtime(session, tmp_path):
    """Without the line clock, a CYCLE-split recording ends at its last
    file's modification time (MATLAB takes the first file's)."""
    lines = 450000
    names = ["acq_DMD1-CYCLE-000000.dat", "acq_DMD2-CYCLE-000000.dat"]
    for n in names:
        session.dat(n, lines, START)
    end = START + datetime.timedelta(seconds=30)
    for n in ("acq_DMD1-CYCLE-000001.dat", "acq_DMD2-CYCLE-000001.dat"):
        (session.dir / n).write_bytes(b"")
        os.utime(session.dir / n, (end.timestamp(), end.timestamp()))
    session.references()
    s2 = session.build(tmp_path / "out")["slap2_info"]
    last = session.mtimes(["acq_DMD1-CYCLE-000001.dat"])[0]
    edges = np.array([1, 150001, 300001, 450001], dtype=float)
    to_days = LINE_PERIOD_S / 86400
    np.testing.assert_allclose(
        s2["trial_end_time_from_pc"].ravel(),
        last - (lines - edges[1:]) * to_days,
        rtol=0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        s2["trial_start_time_inferred"].ravel(),
        last - (lines - edges[:3]) * to_days,
        rtol=0,
        atol=1e-9,
    )


def test_multicycle_edges_are_matlab_linspace(session, tmp_path):
    """The trial edges of a continuous recording are MATLAB's linspace
    values."""
    # 3 trials of 599838 lines: the edges are fractional, as MATLAB's
    lines = 599837
    names = ["acq_DMD1-CYCLE-000000.dat", "acq_DMD2-CYCLE-000000.dat"]
    for n in names:
        session.dat(n, lines, START)
    session.references()
    s2 = session.build(tmp_path / "out")["slap2_info"]
    edges = matlab_linspace(1, lines + 1, 4)
    assert edges[1] != round(edges[1])
    np.testing.assert_array_equal(s2["first_line"][0], edges[:3])
    np.testing.assert_array_equal(s2["last_line"][0], edges[1:])


def test_matlab_linspace_values():
    """matlab_linspace reproduces values computed in MATLAB."""
    # values from MATLAB R2023b: linspace(1, 599838, 4)
    y = matlab_linspace(1, 599838, 4)
    assert y[0] == 1 and y[-1] == 599838
    assert y[1] == 1 + (1 * (599838 - 1)) / 3
    assert y[2] == 1 + (2 * (599838 - 1)) / 3


def test_reference_stack(session, tmp_path):
    """Each path's reference stack is found and stored."""
    session.triggered([100000], [100000])
    session.references()
    t = session.build(tmp_path / "out")
    for path in ("Path1", "Path2"):
        ref = t["slap2_info"]["ref_stack"][path]
        np.testing.assert_array_equal(ref["channels"], [1, 2])
        np.testing.assert_array_equal(ref["Zs"], [0, 1, 2])
        np.testing.assert_array_equal(
            ref["dmdPixel2SampleTransform"], np.eye(3)
        )
        assert ref["IM"].shape == (
            6,
            8,
            6,
        )  # columns x rows x pages, as MATLAB
        assert ref["IM"].dtype == np.float32


def test_write_trial_table_h5(session, tmp_path):
    """The written trial_table.h5 holds the table in MATLAB's column-major
    layout."""
    dmd1, dmd2 = session.triggered([100000, 120000], [100000, 120000])
    session.references()
    out = tmp_path / "out"
    built = session.build(out)
    t = load_struct_h5(out / "trial_table.h5")
    assert t["datadr"] == str(session.dir)
    assert t["savedr"] == str(out)
    assert t["filename"].tolist() == [dmd1, dmd2]
    np.testing.assert_array_equal(
        t["slap2_info"]["last_line"], [[100000, 120000]] * 2
    )
    np.testing.assert_array_equal(
        t["slap2_info"]["ref_stack"]["Path1"]["IM"],
        built["slap2_info"]["ref_stack"]["Path1"]["IM"],
    )
    with h5py.File(out / "trial_table.h5", "r") as h:
        # stored column-major, as MATLAB's saveStructToH5 writes it
        assert h["filename"].shape == (2, 2)
        assert h["slap2_info/first_line"].shape == (2, 2)
        assert h["slap2_info/ref_stack/Path1/IM"].shape == (6, 8, 6)
        assert h["slap2_info/ref_stack/Path1/Zs"].shape == (3, 1)
        assert h["row_major"][()].item() == 0


def test_max_duration_trims_and_keeps_untrimmed(session, tmp_path):
    """max_duration_s trims the table and also writes the untrimmed one."""
    session.triggered([100000, 120000, 110000], [100000, 120000, 110000])
    session.references()
    out = tmp_path / "out"
    t = session.build(out, max_duration_s=15.0)  # 10 s + 5 s of the 12 s trial
    assert (out / "trial_table_untrimmed.h5").exists()
    s2 = load_struct_h5(out / "trial_table.h5")["slap2_info"]
    np.testing.assert_array_equal(s2["first_line"], np.ones((2, 2)))
    np.testing.assert_array_equal(s2["last_line"], [[100000, 50000]] * 2)
    assert t["filename"].shape == (2, 2)
    assert np.asarray(s2["trial_start_time_inferred"]).size == 2
    full = load_struct_h5(out / "trial_table_untrimmed.h5")
    assert full["filename"].shape == (2, 3)


def test_trim_keeps_whole_trials_before_the_limit():
    """Trimming keeps the trials that start before the limit and shortens the
    last."""
    table = {
        "filename": np.array([["a", "b"], ["c", "d"]], dtype=object),
        "true_trial_ix": np.array([[1.0, 2.0], [1.0, 2.0]]),
        "epoch": np.ones((2, 2)),
        "slap2_info": {
            "first_line": np.ones((2, 2)),
            "last_line": np.full((2, 2), 1000.0),
            "trial_end_time_from_pc": np.array([[1.0, 2.0]]),
            "trial_start_time_inferred": np.array([[0.5, 1.5]]),
        },
    }
    out, info = trim_trial_table(table, 0.15, 1e-4)  # 0.1 s trials
    assert info["n_trials_kept"] == 2 and info["last_trial_shortened"]
    keep = math.floor(
        (0.15 - 1000 * 1e-4) / 1e-4
    )  # 499 in floating point, as in MATLAB
    np.testing.assert_array_equal(
        out["slap2_info"]["last_line"], [[1000, keep]] * 2
    )
    out, info = trim_trial_table(table, 1.0, 1e-4)
    assert info["n_trials_kept"] == 2 and not info["last_trial_shortened"]
    with pytest.raises(ValueError):
        trim_trial_table(table, 0.0, 1e-4)


def test_no_reference_stack_error(session, tmp_path):
    """A session without a reference stack raises FileNotFoundError."""
    session.triggered([100000], [100000])
    with pytest.raises(FileNotFoundError):
        session.build(tmp_path / "out")


def test_unaligned_dmds_error(session, tmp_path):
    """Paths whose recordings do not line up raise an error."""
    session.triggered([100000], [150000])
    session.references()
    with pytest.raises(ValueError, match="lining up"):
        session.build(tmp_path / "out")


def test_unmatched_trailing_recordings_warn(session, tmp_path):
    """Data left after the last trial that lines up is left out, as in
    MATLAB, with a warning."""
    session.triggered([100000, 100000], [100000])
    session.references()
    with pytest.warns(TrialTableWarning, match="100000 lines of DMD1"):
        t = session.build(tmp_path / "out")
    assert t["filename"].shape == (2, 1)


def test_no_trials_error(session, tmp_path):
    """A session in which no trial lines up warns and raises."""
    session.triggered([100000], [])
    session.references()
    with pytest.warns(TrialTableWarning, match="left out"):
        with pytest.raises(ValueError, match="No trials"):
            session.build(tmp_path / "out")


@pytest.mark.filterwarnings("ignore:Could not read acquisition times")
def test_mixed_continuous_and_triggered_error(session, tmp_path):
    """Continuous and triggered recordings in one folder raise an error."""
    session.triggered([100000], [100000])
    for n in ("acq2_DMD1-CYCLE-000000.dat", "acq2_DMD2-CYCLE-000000.dat"):
        session.dat(n, 450000, START + datetime.timedelta(minutes=5))
    session.references()
    with pytest.raises(ValueError, match="both continuous"):
        session.build(tmp_path / "out")


def test_acquisition_order():
    """Files sort by acquisition time and trial number; ambiguous names are
    refused."""
    names = [
        "a_20250314_111223_DMD1-TRIAL000010.dat",
        "a_20250314_111223_DMD1-TRIAL000002.dat",
        "b_20250314_101500_DMD1-TRIAL000001.dat",
    ]
    order, ok = acquisition_order(names)
    assert ok and order.tolist() == [2, 1, 0]
    order, ok = acquisition_order(
        ["x_DMD1-TRIAL000002.dat", "x_DMD1-TRIAL000001.dat"]
    )
    assert ok and order.tolist() == [1, 0]
    _, ok = acquisition_order(["b_DMD1.dat", "a_DMD1.dat"])
    assert not ok
    _, ok = acquisition_order(["x_DMD1.dat", "x_DMD1-CYCLE-000000.dat"])
    assert not ok  # same position


def _write_mat_datetime(path: Path, ms: complex, tz: str = ""):
    """A v7.3 MAT file holding ``fpgaTimeReference`` as MATLAB writes it.

    The ``FileWrapper__`` blob lists strings, one class (datetime) and one
    object whose properties are ``data`` (cell 0) and, when given, ``tz``
    (cell 1).
    """
    strings = [b"data", b"datetime", b"fmt", b"tz"]
    names = b"".join(s + b"\0" for s in strings)
    names += b"\0" * (-len(names) % 8)
    classes = struct.pack("<8I", 0, 0, 0, 0, 0, 2, 0, 0)
    objects = struct.pack("<12I", 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1)
    props = [(1, 1, 0)] + ([(4, 1, 1)] if tz else [])
    words = [0, 0, len(props)] + [v for p in props for v in p]
    words += [0] * (len(words) % 2)
    sets = struct.pack(f"<{len(words)}I", *words)
    o0 = 40 + len(names)
    o1 = o0 + len(classes)
    o2 = o1
    o3 = o2 + len(objects)
    o4 = o3 + len(sets)
    header = struct.pack("<II", 4, len(strings)) + struct.pack(
        "<8I", o0, o1, o2, o3, o4, o4, o4, o4
    )
    blob = header + names + classes + objects + sets
    with h5py.File(path, "w") as h:
        refs = h.create_group("#refs#")
        b = refs.create_dataset(
            "a", data=np.frombuffer(blob, np.uint8)[None, :]
        )
        e = refs.create_dataset("b", data=np.zeros(2, np.uint64))
        cplx = np.array(
            [[(ms.real, ms.imag)]], dtype=[("real", "<f8"), ("imag", "<f8")]
        )
        d = refs.create_dataset("c", data=cplx)
        cells = [b.ref, e.ref, d.ref]
        if tz:
            z = refs.create_dataset(
                "d", data=np.array([[ord(c) for c in tz]], np.uint16)
            )
            cells.append(z.ref)
        sub = h.create_group("#subsystem#")
        sub.create_dataset(
            "MCOS", data=np.array([cells], dtype=h5py.ref_dtype)
        )
        ds = h.create_dataset(
            "fpgaTimeReference",
            data=np.array([[0xDD000000, 2, 1, 1, 1, 1]], np.uint32),
        )
        ds.attrs["MATLAB_class"] = np.bytes_(b"datetime")


def test_read_mat_datetime(tmp_path):
    """The .meta decoder reads an unzoned MATLAB datetime."""
    # 2025-03-14 09:30:20.5 wall clock, unzoned
    ms = (
        datetime.datetime(2025, 3, 14, 9, 30, 20, 500000)
        - datetime.datetime(1970, 1, 1)
    ).total_seconds() * 1000
    p = tmp_path / "a.meta"
    _write_mat_datetime(p, complex(ms, -8.6e-5))
    assert read_mat_datetime_ms(p, "fpgaTimeReference") == pytest.approx(
        ms - 8.6e-5, abs=1e-9
    )
    assert read_mat_datetime_ms(p, "missing") is None
    # zoned: data is UTC; the wall clock in Los Angeles is 7 h earlier in March
    _write_mat_datetime(p, complex(ms, 0), tz="America/Los_Angeles")
    assert read_mat_datetime_ms(p, "fpgaTimeReference") == pytest.approx(
        ms - 7 * 3600e3
    )


def test_read_mat_datetime_offset_across_dst(tmp_path):
    """The offset is elapsed time: 01:30 PST plus two hours on the day DST
    starts is 04:30 PDT (MATLAB: datetime + seconds)."""
    utc = datetime.datetime(2025, 3, 9, 9, 30)  # 01:30 in Los Angeles (PST)
    ms = (utc - datetime.datetime(1970, 1, 1)).total_seconds() * 1000
    p = tmp_path / "a.meta"
    _write_mat_datetime(p, complex(ms, 0), tz="America/Los_Angeles")
    wall = read_mat_datetime_ms(p, "fpgaTimeReference", offset_ms=2 * 3600e3)
    expected = datetime.datetime(2025, 3, 9, 4, 30) - datetime.datetime(
        1970, 1, 1
    )
    assert wall == pytest.approx(expected.total_seconds() * 1000, abs=1e-6)


@pytest.mark.parametrize("n", [0, 2])
def test_read_mat_datetime_needs_one_value(tmp_path, n):
    """An empty or multi-element reference gives None (file times are used
    instead), not an error or the first value."""
    p = tmp_path / "a.meta"
    _write_mat_datetime(p, complex(1.7e12, 0))
    with h5py.File(p, "r+") as h:
        del h["#refs#/c"]
        h["#refs#"].create_dataset(
            "c", data=np.zeros((1, n), [("real", "<f8"), ("imag", "<f8")])
        )
        cells = h["#subsystem#/MCOS"][()]
        cells[0, 2] = h["#refs#/c"].ref
        h["#subsystem#/MCOS"][...] = cells
    assert read_mat_datetime_ms(p, "fpgaTimeReference") is None


def test_first_line_timestamp(tmp_path):
    """The first line's timestamp is read from the .dat file, after a magic-
    number check."""
    class File:
        """A stand-in for the SLAP2 data-file object."""
        datFileName = str(tmp_path / "x.dat")
        header = {"firstCycleOffsetBytes": 16}

    line = struct.pack(
        "<IIIIIhhQ", 40, 322379495, 1, 1, 0, 0, 0, 1230711776925
    )
    Path(File.datFileName).write_bytes(b"\0" * 16 + line + b"\0" * 8)
    assert first_line_timestamp(File) == 1230711776925
    Path(File.datFileName).write_bytes(b"\0" * 16 + b"\1" * 40)
    with pytest.raises(ValueError, match="magic"):
        first_line_timestamp(File)
