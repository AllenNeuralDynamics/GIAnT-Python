"""The ``giant register`` command's summary and exit code."""

import numpy as np
import pytest

from giant_python import cli


@pytest.mark.parametrize(
    "failed,names,code",
    [
        ([[False, False]], [["a.h5", "b.h5"]], 0),
        ([[False, True]], [["a.h5", "b.h5"]], 0),
        ([[True, True]], [["a.h5", ""]], 1),
    ],
)
def test_register_reports_failed_trials(
    monkeypatch, capsys, failed, names, code
):
    """Flagged trials are listed; a trial that raised makes the exit code
    nonzero."""
    from giant_python.pipeline import register

    fields = {
        "registration_failed": np.array(failed),
        "fn_reg_ds": np.array(names, dtype=object),
        "fn_adata": np.array(names, dtype=object),
        "align_params": {},
    }
    monkeypatch.setattr(register, "register_multiroi", lambda *a, **k: fields)
    assert cli.main(["register", "trial_table.h5"]) == code
    out, err = capsys.readouterr()
    registered = 2 - sum(n == "" for n in names[0])
    assert f"{registered} of 2 trials registered" in out
    assert ("registrationFailed" in out) == (failed[0][0] or failed[0][1])
    assert ("trial 2 (path 1)" in err) == (code == 1)


@pytest.mark.parametrize(
    "flags,save_tif", [([], False), (["--h5"], False), (["--tif"], True)]
)
def test_register_movie_format(monkeypatch, flags, save_tif):
    """Registered movies are HDF5 unless ``--tif`` asks for TIFF."""
    from giant_python.pipeline import register

    seen = {}

    def fake(table, params, **kwargs):
        seen["params"] = params
        return {
            "registration_failed": np.array([[False]]),
            "fn_reg_ds": np.array([["a"]], dtype=object),
            "fn_adata": np.array([["a"]], dtype=object),
            "align_params": {},
        }

    monkeypatch.setattr(register, "register_multiroi", fake)
    assert cli.main(["register", "trial_table.h5", *flags]) == 0
    assert seen["params"].save_tif is save_tif


def test_register_movie_formats_exclusive():
    """``--tif`` and ``--h5`` cannot be combined."""
    with pytest.raises(SystemExit):
        cli.main(["register", "trial_table.h5", "--tif", "--h5"])
