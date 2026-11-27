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
``find(selPix)``). As in SILo's collect stage, only results made from the
current plan are read. A work folder that holds results of more than one
solve for a trial (other solve settings or inputs) is an error, since the
summaries do not record which solve they used.

Only the gain modes whose baseline is ``g * b_net`` are read: 'gain' (N1),
and 'gain_shared' (N2) subproblems whose shared time course stayed at 1.
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


def _attr(f: h5py.File, name: str):
    """A file's attribute as a string (None if absent)."""
    v = f.attrs.get(name)
    if v is None:
        return None
    return v.decode() if isinstance(v, bytes) else str(v)


def plan_file(work_dir, path: int) -> Path:
    """The work folder's plan of a path."""
    return Path(work_dir) / "plan" / f"path{path}.h5"


def plan_hash(plan_path) -> str:
    """The plan's settings hash, which its results files record as
    ``plan_hash``."""
    with h5py.File(plan_path, "r") as f:
        h = _attr(f, "settings_hash")
    if h is None:
        raise ValueError(f"{plan_path}: no settings_hash")
    return h


def current_results(files, plan: str) -> list:
    """The results files of one trial-path that belong to the plan with
    settings hash ``plan``, as SILo's collect stage selects them.

    Files of other plans are left out. The rest must come from one solve
    (one ``solve_science`` and ``container_hash``); otherwise the work
    folder holds results of earlier runs that the summaries may not have
    used.

    Raises
    ------
    ValueError
        If the plan's files come from more than one solve.
    """
    groups: dict = {}
    for path in files:
        with h5py.File(path, "r") as f:
            if _attr(f, "plan_hash") != plan:
                continue
            key = (_attr(f, "solve_science"), _attr(f, "container_hash"))
        groups.setdefault(key, []).append(path)
    if len(groups) > 1:
        names = "; ".join(
            ", ".join(Path(p).name for p in v) for v in groups.values()
        )
        raise ValueError(
            "results of more than one solve of the same plan (other solve "
            f"settings or inputs): {names}. Remove the stale files."
        )
    return next(iter(groups.values()), [])


def trial_gains(results_files) -> tuple[np.ndarray, np.ndarray]:
    """Every subproblem pixel's gain in one trial-path's results files.

    A subproblem that is in several files (identical copies, which SILo's
    collect stage also allows) counts once.

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
        mode), has a shared time course other than 1 ('gain_shared',
        whose baseline is then not ``g * b_net``), or has two different
        copies.
    """
    seen: dict = {}
    for path in results_files:
        with h5py.File(path, "r") as f:
            if "subproblems" not in f:
                continue
            for k in f["subproblems/index"][()]:
                k = int(k)
                g = f[f"subproblems/{k}"]
                info = json.loads(g.attrs["info"])
                step = info.get("baseline_step")
                if not step or "gain" not in step:
                    raise ValueError(
                        f"{path}: subproblem {k} has no gain record "
                        "(not a 'gain' baseline mode)"
                    )
                shared = step.get("shared")
                if shared is not None and not np.allclose(
                    np.asarray(shared, dtype=np.float64), 1.0
                ):
                    raise ValueError(
                        f"{path}: subproblem {k} has a shared baseline time "
                        "course ('gain_shared'); its baseline is not "
                        "gain x network baseline"
                    )
                p = np.asarray(g["pixels"][()], dtype=np.int64).ravel()
                v = np.asarray(step["gain"], dtype=np.float64).ravel()
                if p.size != v.size:
                    raise ValueError(
                        f"{path}: subproblem {k} has {p.size} pixels "
                        f"but {v.size} gains"
                    )
                if k in seen:
                    p0, v0 = seen[k]
                    if not (np.array_equal(p0, p) and np.array_equal(v0, v)):
                        raise ValueError(
                            f"subproblem {k} has two different results "
                            f"({path})"
                        )
                    continue
                seen[k] = (p, v)
    if not seen:
        return np.zeros(0, np.int64), np.zeros(0)
    return (
        np.concatenate([p for p, _ in seen.values()]),
        np.concatenate([v for _, v in seen.values()]),
    )


def results_files(work_dir, path: int, trial: int) -> list:
    """A trial-path's results files (``trial`` zero-based) of the work
    folder's current plan: ``path<d>_trial<tttt>.h5`` and its ``_s...``
    shards, selected as SILo's collect stage does
    (:func:`current_results`)."""
    d = Path(work_dir) / "results"
    stem = f"path{path}_trial{trial + 1:04d}"
    files = sorted(
        p
        for p in d.glob(f"{stem}*.h5")
        if p.stem == stem or p.stem.startswith(stem + "_s")
    )
    if not files:
        return []
    return current_results(files, plan_hash(plan_file(work_dir, path)))


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
    rows, cols, shape = selected_pixels(plan_file(work_dir, path))
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
