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
    """Worker counts are positive integers, the limit positive."""
    with pytest.raises(ValueError):
        SiloExecution(load_workers=0)
    with pytest.raises(ValueError):
        SiloExecution(load_memory_gb=0.0)


@pytest.mark.parametrize("memory", [None, 1e-9])
def test_streamed_run_loading_ahead_is_the_same(tmp_path, fake, memory):
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
