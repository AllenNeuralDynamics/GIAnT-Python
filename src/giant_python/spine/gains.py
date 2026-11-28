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
summaries do not record which solve they used, and so are a trial's files
that leave out some of its subproblems (an incomplete set of shards), which
the collect stage also refuses.

Only the gain modes whose baseline is ``g * b_net`` are read: 'gain' (N1),
and 'gain_shared' (N2) subproblems whose shared time course stayed at 1.
The gains apply only to the baseline files they were fitted against: each
results file records that file's identity (``network_baseline``), and
:func:`check_gain_baselines` compares it with the files of the network's F0
images, as SILo's extraction compares the localization's baseline.
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
    (one ``solve_science``, ``solve_settings`` and ``container_hash``, as
    the collect stage requires of a container's files); otherwise the work
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
            key = (
                _attr(f, "solve_science"),
                _attr(f, "solve_settings"),
                _attr(f, "container_hash"),
            )
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


def _n_subproblems(path, f: h5py.File, n_sub):
    """The trial's number of subproblems as the file records it (or
    ``n_sub`` if it does not), checked against ``n_sub`` from its other
    files."""
    ns = f.attrs.get("n_subproblems")
    if ns is None:
        return n_sub
    ns = int(ns)
    if n_sub is not None and ns != n_sub:
        raise ValueError(
            f"{path}: the trial has {ns} subproblems here but {n_sub} in "
            "another of its results files"
        )
    return ns


def _subproblem_gains(path, k: int, g: h5py.Group):
    """A subproblem's pixels and gains (``baseline_step.gain`` of its
    ``info`` record), checked as :func:`trial_gains` describes."""
    step = json.loads(g.attrs["info"]).get("baseline_step")
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
            f"{path}: subproblem {k} has {p.size} pixels but {v.size} gains"
        )
    return p, v


def trial_gains(results_files) -> tuple[np.ndarray, np.ndarray]:
    """Every subproblem pixel's gain in one trial-path's results files.

    A subproblem that is in several files (identical copies, which SILo's
    collect stage also allows) counts once. As in the collect stage, every
    subproblem of the trial (``n_subproblems``, where the files record it)
    must be in the files.

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
        copies; or if the files disagree on the trial's number of
        subproblems or leave some out.
    """
    seen: dict = {}
    n_sub = None
    extracted = True
    for path in results_files:
        with h5py.File(path, "r") as f:
            n_sub = _n_subproblems(path, f, n_sub)
            if not int(f.attrs.get("extracted", 1)):
                extracted = False
            if "subproblems" not in f:
                continue
            for k in f["subproblems/index"][()]:
                k = int(k)
                p, v = _subproblem_gains(path, k, f[f"subproblems/{k}"])
                if k in seen:
                    p0, v0 = seen[k]
                    if not (np.array_equal(p0, p) and np.array_equal(v0, v)):
                        raise ValueError(
                            f"subproblem {k} has two different results "
                            f"({path})"
                        )
                    continue
                seen[k] = (p, v)
    if n_sub is not None and extracted:
        missing = sorted(set(range(n_sub)) - set(seen))
        if missing:
            names = ", ".join(Path(x).name for x in results_files)
            raise ValueError(
                f"{len(missing)} of {n_sub} subproblems have no results in "
                f"{names} (first: {missing[0]}): the trial's results are "
                "incomplete"
            )
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


_ID_KEYS = ("name", "size", "mtime_ns")
"""The fields of a file identity that SILo records."""


def recorded_baseline(files):
    """The baseline file a trial-path's results files record as fitted
    against (``network_baseline``: its name, size and modification time),
    or None. The files are of one solve, so they record the same file."""
    for path in files:
        with h5py.File(path, "r") as f:
            v = _attr(f, "network_baseline")
        if v:
            return json.loads(v).get("file")
    return None


def check_gain_baselines(work_dir, path: int, baselines) -> None:
    """Check that every trial's gains were fitted against the baseline file
    that its network F0 image is made from.

    Parameters
    ----------
    work_dir : str or Path
        The run's work folder.
    path : int
        Imaging path (one-based).
    baselines : list of dict or None
        Per trial, the identity of the network image's baseline file
        (:func:`~giant_python.spine.images.file_identity`); None for a
        trial without an image (nothing to check).

    Raises
    ------
    ValueError
        If a trial's results record no network baseline, or another file
        (other name, size or modification time: for example a baseline
        predicted again after the solve).
    """
    for t, ident in enumerate(baselines):
        if ident is None:
            continue
        files = results_files(work_dir, path, t)
        if not files:
            continue
        rec = recorded_baseline(files)
        if rec is None:
            raise ValueError(
                f"Path{path} trial {t + 1}: the results record no network "
                "baseline (not a run with the network's baseline)"
            )
        want = {k: ident.get(k) for k in _ID_KEYS}
        got = {k: rec.get(k) for k in _ID_KEYS}
        if got != want:
            raise ValueError(
                f"Path{path} trial {t + 1}: the gains were fitted against "
                f"another baseline file ({got}) than the network image's "
                f"({want}); give the baselines the run used, or solve again"
            )


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
