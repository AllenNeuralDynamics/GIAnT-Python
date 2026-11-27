"""The network's baseline as rescaled by SILo's gain fit (``net·gain``).

In the 'gain' baseline mode (N1, the default with the network's baseline)
the solver fits one gain per selected pixel in each subproblem, after the
first activity fit: ``B = max(min_baseline, g * b_net)``. The final ``B`` is
not saved, but each subproblem's gains are, in its results file's ``info``
record (``baseline_step.gain``, one per subproblem pixel). A trial's mean of
``B`` is therefore ``g`` times the trial's mean of the network's baseline
(apart from the floor, which the network's baseline rarely reaches), and
the per-trial F0 image of the rescaled baseline is ``g`` times the network's
per-trial F0 image, at the selected pixels.

A pixel that belongs to overlapping subproblems has one gain in each; the
image takes their mean, and :func:`gain_maps` also returns their spread so
that the disagreement can be checked.

Results files: ``<work_dir>/results/path<d>_trial<tttt>[_s...].h5`` (path
and trial one-based); the plan ``<work_dir>/plan/path<d>.h5`` gives the
selected pixels (``sources/sel``, listed column by column, as MATLAB's
``find(selPix)``).
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


def selected_pixels(plan_path) -> tuple[np.ndarray, np.ndarray, tuple]:
    """Rows and columns of a plan's selected-pixel list, and the frame
    shape."""
    with h5py.File(plan_path, "r") as f:
        sel = f["sources/sel"][()].astype(bool)
    cols, rows = np.nonzero(sel.T)
    return rows, cols, sel.shape


def trial_gains(results_files) -> tuple[np.ndarray, np.ndarray]:
    """Every subproblem pixel's gain in one trial-path's results files.

    Returns
    -------
    pixels : ndarray of int
        Indices into the plan's selected-pixel list (repeated for a pixel
        in several subproblems).
    gains : ndarray of float
        The matching gains.

    Raises
    ------
    ValueError
        If a subproblem has no gain record (the run did not use a 'gain'
        mode).
    """
    pix, gains = [], []
    for path in results_files:
        with h5py.File(path, "r") as f:
            if "subproblems" not in f:
                continue
            for k in f["subproblems/index"][()]:
                g = f[f"subproblems/{int(k)}"]
                info = json.loads(g.attrs["info"])
                step = info.get("baseline_step")
                if not step or "gain" not in step:
                    raise ValueError(
                        f"{path}: subproblem {int(k)} has no gain record "
                        "(not a 'gain' baseline mode)"
                    )
                p = np.asarray(g["pixels"][()], dtype=np.int64).ravel()
                v = np.asarray(step["gain"], dtype=np.float64).ravel()
                if p.size != v.size:
                    raise ValueError(
                        f"{path}: subproblem {int(k)} has {p.size} pixels "
                        f"but {v.size} gains"
                    )
                pix.append(p)
                gains.append(v)
    if not pix:
        return np.zeros(0, np.int64), np.zeros(0)
    return np.concatenate(pix), np.concatenate(gains)


def results_files(work_dir, path: int, trial: int) -> list:
    """A trial-path's results files (``trial`` zero-based)."""
    d = Path(work_dir) / "results"
    return sorted(d.glob(f"path{path}_trial{trial + 1:04d}*.h5"))


def gain_maps(work_dir, path: int, n_trials: int):
    """Per-trial gain images of one path, in the plan's consensus frame.

    Returns
    -------
    gain : ndarray of float32, shape (trials, rows, cols)
        Mean gain per selected pixel; NaN elsewhere and for trials without
        results.
    spread : ndarray of float32, shape (trials, rows, cols)
        Largest minus smallest gain of a pixel over its subproblems (0 for a
        pixel in one subproblem).
    n_files : ndarray of int, shape (trials,)
        Results files read per trial.
    """
    plan = Path(work_dir) / "plan" / f"path{path}.h5"
    rows, cols, shape = selected_pixels(plan)
    gain = np.full((n_trials,) + shape, np.nan, np.float32)
    spread = np.full((n_trials,) + shape, np.nan, np.float32)
    n_files = np.zeros(n_trials, np.int64)
    for t in range(n_trials):
        files = results_files(work_dir, path, t)
        n_files[t] = len(files)
        if not files:
            continue
        pix, g = trial_gains(files)
        if pix.size == 0:
            continue
        n_sel = rows.size
        if pix.max() >= n_sel:
            raise ValueError(
                f"Path{path} trial {t + 1}: pixel index {pix.max()} beyond "
                f"the plan's {n_sel} selected pixels"
            )
        total = np.bincount(pix, weights=g, minlength=n_sel)
        count = np.bincount(pix, minlength=n_sel)
        hi = np.full(n_sel, -np.inf)
        lo = np.full(n_sel, np.inf)
        np.maximum.at(hi, pix, g)
        np.minimum.at(lo, pix, g)
        has = count > 0
        gain[t, rows[has], cols[has]] = total[has] / count[has]
        spread[t, rows[has], cols[has]] = hi[has] - lo[has]
    return gain, spread, n_files
