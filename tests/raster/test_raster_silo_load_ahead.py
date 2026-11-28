"""Loading trials ahead of the solver in the streamed run
(``SiloExecution.load_workers``, ``load_memory_gb``)."""

import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from giant_python.models.params import SiloExecution
from giant_python.pipeline import silo
from giant_python.pipeline.silo_session import SiloSession

from .test_raster_silo_stages import (  # noqa: F401
    EX,
    PARAMS,
    _all_results,
    _equal,
    _summary,
    fake,
    make_session,
)


def _units(n, lines=100.0):
    """``n`` units of a path, each of ``lines`` lines."""
    return [
        silo._Unit(0, t, None, "", "", Path(f"r{t}.h5"), Path(""), lines)
        for t in range(n)
    ]


def test_loaded_keeps_order_and_limits(monkeypatch):
    """Units come back in order whatever finishes first; at most
    ``load_workers`` load at once; with a memory limit only one loads until
    a load has measured the bytes per line, then as many as fit."""
    active, most, started = [0], [0], []
    lock = threading.Lock()

    def load(u):
        """A load that takes longer for earlier units."""
        with lock:
            active[0] += 1
            most[0] = max(most[0], active[0])
            started.append(u.t)
        time.sleep(0.3 + 0.05 * (5 - u.t))  # earlier units finish last
        with lock:
            active[0] -= 1
        return {"t": u.t, "nbytes": 1000}

    units = _units(6)
    ex = SiloExecution(load_workers=3)
    got = [(u.t, item["t"]) for u, item in silo._loaded(units, load, ex)]
    assert got == [(t, t) for t in range(6)]
    assert most[0] == 3
    # 10 bytes per line: 1000 bytes per unit; a 2500-byte limit holds two
    active[0], most[0] = 0, 0
    ex = SiloExecution(load_workers=4, load_memory_gb=2.5e-6)
    got = [u.t for u, _ in silo._loaded(units, load, ex)]
    assert got == list(range(6)) and most[0] == 2
    # one worker: each unit is loaded when it is asked for
    started.clear()
    gen = silo._loaded(units, load, SiloExecution())
    next(gen)
    assert started == [0]


def test_loaded_waits_for_a_measured_load_and_takes_units_lazily():
    """A load without ``nbytes`` (a trial too short to extract) gives no
    estimate: with a memory limit the loads stay one at a time until a load
    reports its bytes. Units are taken from the iterable only when they are
    about to be loaded."""
    active, most, taken = [0], [0], []
    lock = threading.Lock()

    def load(u):
        """The first unit is short (no subproblems, no bytes)."""
        with lock:
            active[0] += 1
            most[0] = max(most[0], active[0])
        time.sleep(0.2)
        with lock:
            active[0] -= 1
            if u.t == 0:
                return {}
            return {"nbytes": 1000}

    def lazy(units):
        """The units, noting when each is taken."""
        for u in units:
            taken.append(u.t)
            yield u

    # 10 bytes per line from the second load on; the limit holds two
    ex = SiloExecution(load_workers=4, load_memory_gb=2.5e-6)
    gen = silo._loaded(lazy(_units(6)), load, ex)
    first = next(gen)
    assert first[0].t == 0 and most[0] == 1
    assert taken == [0, 1]  # the second is taken, waiting for an estimate
    rest = [u.t for u, _ in gen]
    assert rest == [1, 2, 3, 4, 5] and most[0] == 2
    # one worker: a unit is taken when it is asked for
    taken.clear()
    gen = silo._loaded(lazy(_units(3)), load, SiloExecution())
    next(gen)
    assert taken == [0]


def test_loaded_hands_on_earlier_units_before_a_failed_check():
    """With several loaders, an error raised while taking a unit (its
    check) reaches the caller after the units taken before it."""

    def units():
        """Two units, then a failing check."""
        yield from _units(2)
        raise FileNotFoundError("no baseline")

    gen = silo._loaded(
        units(), lambda u: {"t": u.t}, SiloExecution(load_workers=3)
    )
    got = []
    with pytest.raises(FileNotFoundError, match="no baseline"):
        for u, item in gen:
            got.append(item["t"])
    assert got == [0, 1]


@pytest.mark.parametrize("load_workers", [1, 3])
def test_streamed_run_checks_later_trials_when_it_reaches_them(
    tmp_path, fake, monkeypatch, load_workers  # noqa: F811
):
    """A later trial whose check fails (here its baseline) does not stop
    the earlier trials from being extracted and written, with one loader
    or several (the trial waiting for its results is written before the
    error is raised)."""
    s = make_session(tmp_path / "s")
    real = silo._extraction_baseline
    calls = []

    def failing(session, params, d, t, baseline_dir):
        """Fails for the last trial of the first path."""
        calls.append((d, t))
        if (d, t) == (0, session.n_trials - 1):
            raise FileNotFoundError("no baseline for this trial")
        return real(session, params, d, t, baseline_dir)

    monkeypatch.setattr(silo, "_extraction_baseline", failing)
    with pytest.raises(FileNotFoundError, match="no baseline"):
        silo.run_silo(s, PARAMS, replace(EX, load_workers=load_workers))
    session = SiloSession.open(s)
    written = [
        t
        for t in range(session.n_trials)
        if session.results_file(0, t).exists()
    ]
    assert written == list(range(session.n_trials - 1))


def test_loaded_passes_on_errors():
    """A load's error reaches the caller at its unit; later loads are not
    waited for beyond the running ones."""

    def load(u):
        """Fails on the second unit."""
        if u.t == 1:
            raise RuntimeError("bad trial")
        return {"nbytes": 10}

    gen = silo._loaded(_units(5), load, SiloExecution(load_workers=2))
    assert next(gen)[0].t == 0
    with pytest.raises(RuntimeError, match="bad trial"):
        next(gen)


def test_options_are_checked():
    """Worker counts are positive integers, the limit positive; the loader
    options come after the earlier fields, which keep their positions."""
    with pytest.raises(ValueError):
        SiloExecution(load_workers=0)
    with pytest.raises(ValueError):
        SiloExecution(load_memory_gb=0.0)
    names = list(SiloExecution.__dataclass_fields__)
    assert names[-2:] == ["load_workers", "load_memory_gb"]
    assert names[-4:-2] == ["overwrite", "verbose"]
    ex = SiloExecution(6, 3, 4, "cpu", "float64", None, "cpu", True, True)
    assert ex.overwrite and ex.verbose and ex.load_workers == 1


@pytest.mark.parametrize("memory", [None, 1e-9])
def test_streamed_run_loading_ahead_is_the_same(
    tmp_path, fake, memory  # noqa: F811
):
    """Loading three trials ahead gives the results files and summaries of
    loading one at a time."""
    one = make_session(tmp_path / "one")
    r1 = silo.run_silo(one, PARAMS, EX)
    ahead = make_session(tmp_path / "ahead")
    ex = replace(EX, load_workers=3, load_memory_gb=memory)
    r2 = silo.run_silo(ahead, PARAMS, ex)

    def loaded(report):
        """The units the run loaded."""
        return [tuple(e["unit"]) for e in report["trials"] if "load" in e]

    assert loaded(r1) == loaded(r2) and loaded(r1)
    # each loaded trial reports the loader's phases (none for the fake one)
    assert all(e["load_phases"] == {} for e in r2["trials"] if "load" in e)
    a, b = SiloSession.open(one), SiloSession.open(ahead)
    ra, rb = _all_results(a), _all_results(b)
    assert ra.keys() == rb.keys() and ra
    for key in ra:
        for name, value in ra[key].items():
            np.testing.assert_array_equal(value, rb[key][name])
    _equal(
        _summary(a.out_dir / "experiment_summary.h5"),
        _summary(b.out_dir / "experiment_summary.h5"),
    )


def test_held_bytes_count_what_a_loaded_trial_keeps():
    """The memory of a loaded trial waiting for the solver: its
    subproblems' data, variance multipliers and baselines, and its
    traces."""
    from types import SimpleNamespace

    y = np.zeros((2, 10, 50), np.float32)
    sub = SimpleNamespace(
        y=y, finv=np.zeros((10, 50), np.float32), baseline=None
    )
    with_base = SimpleNamespace(
        y=y, finv=sub.finv, baseline=np.zeros((10, 50), np.float32)
    )
    rec = SimpleNamespace(
        global_f=np.zeros((2, 50)),
        roi_f=np.zeros((1, 2, 50)),
        roi_fsvd=np.zeros((1, 2, 50)),
    )
    traces = 3 * 100 * 8
    assert silo._held_bytes([sub], rec) == 4000 + 2000 + traces
    assert silo._held_bytes([sub, with_base], rec) == 2 * 6000 + 2000 + traces
