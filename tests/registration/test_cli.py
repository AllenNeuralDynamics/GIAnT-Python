"""The ``giant register`` command's summary and exit code."""

import numpy as np
import pytest

from giant_python import cli


@pytest.mark.parametrize(
    "failed,names,recorded,code",
    [
        ([[False, False]], [["a.h5", "b.h5"]], [["x", "y"]], 0),
        ([[False, True]], [["a.h5", "b.h5"]], [["x", "y"]], 0),
        ([[True, True]], [["a.h5", ""]], [["x", "y"]], 1),
        ([[False, True]], [["a.h5", ""]], [["x", ""]], 0),
    ],
)
def test_register_reports_failed_trials(
    monkeypatch, capsys, failed, names, recorded, code
):
    """Flagged trials are listed; a trial that raised makes the exit code
    nonzero; a cell without a recording is listed as skipped."""
    from types import SimpleNamespace

    from giant_python.io import trial_table
    from giant_python.pipeline import register

    fields = {
        "registration_failed": np.array(failed),
        "fn_reg_ds": np.array(names, dtype=object),
        "fn_adata": np.array(names, dtype=object),
        "align_params": {},
    }
    monkeypatch.setattr(register, "register_multiroi", lambda *a, **k: fields)
    table = SimpleNamespace(filename=np.array(recorded, dtype=object))
    monkeypatch.setattr(trial_table, "read_trial_table", lambda path: table)
    assert cli.main(["register", "trial_table.h5"]) == code
    out, err = capsys.readouterr()
    n_rec = sum(r != "" for r in recorded[0])
    registered = sum(n != "" for n in names[0])
    assert f"{registered} of {n_rec} trials registered" in out
    flagged = any(f and n for f, n in zip(failed[0], names[0]))
    assert ("registrationFailed" in out) == flagged
    assert ("trial 2 (path 1)" in err) == (code == 1)
    assert ("No recording (skipped): trial 2 (path 1)" in out) == (n_rec == 1)
