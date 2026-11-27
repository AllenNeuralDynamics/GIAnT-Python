"""A small synthetic session for the spine-size tests.

One run folder with ``experiment_summary.h5`` and ``per_trial_summary.h5``
in GIAnT-MATLAB's schema, a ``trial_table.h5``, a motion-correction folder
with ``<stem>_ALIGNMENTDATA.h5`` files (``recNegErr``) and a baseline folder
with ``<stem>_BASELINE-80Hz.h5`` files, for two imaging paths.

Each trial's registered image is the consensus scene shifted by a whole
number of pixels (trial pixel = consensus pixel + shift), so a trial's
image resampled into the consensus frame equals the scene exactly where it
is covered.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from giant_python.io.hdf5 import save_struct_h5

H, W = 24, 30
N_TRIALS = 4
N_FRAMES = 160
ALIGN_HZ = 80.0
SHIFTS = [(0, 0), (1, 0), (0, -2), (2, 1)]
"""Each trial's shift (rows, cols): trial pixel = consensus pixel + shift."""

SOURCES = [(8.0, 9.0), (15.0, 20.0), (12.0, 14.0)]
"""Centres (row, col) of the sources' footprints."""


def scene(path: int) -> np.ndarray:
    """The consensus F0 scene of a path: smooth and positive."""
    rr, cc = np.mgrid[0:H, 0:W].astype(float)
    return 50 + 10 * path + 2 * rr + 0.5 * cc


def trial_scale(t: int) -> float:
    """Each trial's brightness factor (e.g. bleaching)."""
    return 1.0 - 0.05 * t


def footprints() -> np.ndarray:
    """The sources' footprints ``(sources, rows, cols)``."""
    rr, cc = np.mgrid[0:H, 0:W].astype(float)
    out = np.zeros((len(SOURCES), H, W))
    for k, (r0, c0) in enumerate(SOURCES):
        f = np.exp(-((rr - r0) ** 2 + (cc - c0) ** 2) / (2 * 1.3**2))
        f[f < 0.01 * f.max()] = 0
        out[k] = f / f.sum()
    return out


def trial_image(path: int, t: int) -> np.ndarray:
    """A trial's registered F0 image (NaN where the shift leaves no
    data)."""
    s = scene(path) * trial_scale(t)
    dr, dc = SHIFTS[t]
    out = np.full((H, W), np.nan)
    # trial pixel (r + dr, c + dc) shows consensus pixel (r, c)
    rs = slice(max(0, dr), H + min(0, dr))
    cs = slice(max(0, dc), W + min(0, dc))
    rs0 = slice(max(0, -dr), H - max(0, dr))
    cs0 = slice(max(0, -dc), W - max(0, dc))
    out[rs, cs] = s[rs0, cs0]
    return out


def consensus_trial(path: int, t: int) -> np.ndarray:
    """What a trial's image looks like in the consensus frame."""
    out = scene(path) * trial_scale(t)
    dr, dc = SHIFTS[t]
    rr, cc = np.mgrid[0:H, 0:W]
    ok = (rr + dr >= 0) & (rr + dr < H) & (cc + dc >= 0) & (cc + dc < W)
    return np.where(ok, out, np.nan)


def make_session(
    root: Path,
    with_f0_silo: bool = True,
    activity_channel: int = 1,
    missing_trial: int | None = None,
) -> dict:
    """Write the synthetic session under ``root``; return its folders.

    ``dF_ls`` counts up through the frames in file channel 1 and from 1000
    in file channel 2; ``activity_channel`` is the run's
    ``params/activityChannel``. ``missing_trial`` (zero-based) gets NaN
    traces, as a trial without extracted sources.
    """
    root = Path(root)
    run = root / "run"
    moco = root / "motion_correction"
    base = root / "baselines"
    for d in (run, moco, base):
        d.mkdir(parents=True, exist_ok=True)
    affine = np.zeros((N_TRIALS, 2, 3))
    affine[:, :, :2] = np.eye(2)
    affine[:, :, 2] = np.asarray(SHIFTS, float)
    fp = footprints()
    exp: dict = {
        "params": {
            "activityChannel": float(activity_channel),
            "numChannels": 2.0,
        }
    }
    per: dict = {}
    dF = np.tile(np.arange(N_TRIALS * 10, dtype=float), (len(SOURCES), 2, 1))
    dF[:, 1] += 1000
    ev = np.zeros((len(SOURCES), 2, N_TRIALS * 10))
    ev[:, 1] = 1.0  # events only in file channel 2
    if missing_trial is not None:
        dF[:, :, missing_trial * 10 : (missing_trial + 1) * 10] = np.nan
        ev[:, :, missing_trial * 10 : (missing_trial + 1) * 10] = np.nan
    rng = np.random.default_rng(0)
    for p in (1, 2):
        imgs = np.stack([consensus_trial(p, t) for t in range(N_TRIALS)])
        mean_t = np.stack([imgs * 1.2, imgs * 0.1], axis=1)  # T x C x H x W
        exp[f"Path{p}"] = {
            "visualizations": {
                "mean_im": np.nanmean(mean_t, axis=0)[:, None],
                "act_im": rng.random((1, H, W)),
            },
            "sources": {
                "spatial": {"profiles": fp[:, None]},
                "temporal": {"dF_ls": dF, "events": ev},
            },
            "frame_info": {
                "trial_num_frames": np.full((1, N_TRIALS), 10.0),
                "trial_alignment_affine": affine,
            },
            "user_rois": {
                "labels": np.array([["roi A", "roi B"]], dtype=object),
                "mask": np.stack(
                    [
                        np.pad(np.ones((4, 4)), ((2, H - 6), (2, W - 6))),
                        np.pad(np.ones((3, 5)), ((10, H - 13), (20, W - 25))),
                    ]
                )[:, None]
                > 0,
            },
        }
        per[f"Path{p}"] = {
            "visualizations": {"per_trial_mean_im": mean_t[:, :, None]}
        }
        if with_f0_silo:
            f0 = np.stack([imgs, imgs * 0.1], axis=1)[:, :, None]
            per[f"Path{p}"]["visualizations"]["per_trial_F0_im"] = f0
        for t in range(N_TRIALS):
            stem = f"E1T{t + 1}DMD{p}"
            img = trial_image(p, t)
            frames = np.repeat(
                img.T[None], N_FRAMES, axis=0
            )  # frame, col, row
            frames[:5] = 1e4  # the trimmed start: must not enter
            with h5py.File(base / f"{stem}_BASELINE-80Hz.h5", "w") as f:
                f.create_dataset("baseline", data=frames.astype(np.float16))
                f.attrs["checkpoint"] = "synthetic.pt"
                f.attrs["step"] = 1
            with h5py.File(moco / f"{stem}_ALIGNMENTDATA.h5", "w") as f:
                f["recNegErr"] = np.zeros((N_FRAMES, 1))
    save_struct_h5(run / "experiment_summary.h5", exp)
    save_struct_h5(run / "per_trial_summary.h5", per)
    fn = np.array(
        [
            [
                f"E1T{t + 1}DMD{p}_REGISTERED_DOWNSAMPLED-80Hz.tif"
                for t in range(N_TRIALS)
            ]
            for p in (1, 2)
        ],
        dtype=object,
    )
    raw = np.array(
        [
            [
                f"activity_x_DMD{p}-TRIAL{t + 2:06d}.dat"
                for t in range(N_TRIALS)
            ]
            for p in (1, 2)
        ],
        dtype=object,
    )
    table = root / "trial_table.h5"
    save_struct_h5(
        table,
        {
            "datadr": str(root / "raw"),
            "savedr": str(root / "guard"),
            "filename": raw,
            "motion_correction": {"fn_reg_ds": fn},
        },
    )
    return {"run": run, "moco": moco, "baselines": base, "table": table}


PLAN_HASH = "plan-a"
"""The synthetic plans' settings hash."""

RESULT_ATTRS = {"plan_hash": PLAN_HASH, "solve_science": "science-a"}
"""Attributes of the synthetic results files (with a container hash)."""


def write_results(path: Path, subproblems: dict, **attrs) -> None:
    """A results file with ``subproblems`` (index -> (pixels, info))."""
    with h5py.File(path, "w") as f:
        for k, v in {**RESULT_ATTRS, "container_hash": "c", **attrs}.items():
            f.attrs[k] = v
        f["subproblems/index"] = np.array(sorted(subproblems))
        for k, (pix, info) in subproblems.items():
            g = f.create_group(f"subproblems/{k}")
            g["pixels"] = np.asarray(pix)
            g.attrs["info"] = json.dumps(info)


def make_gain_work(root: Path, gain_of_trial) -> Path:
    """A minimal 'gain'-mode work folder: a plan per path with the selected
    pixels, and per trial-path one results file with two overlapping
    subproblems whose gains are ``gain_of_trial(t)`` and twice that.

    Returns the work folder.
    """
    work = Path(root) / "work"
    (work / "plan").mkdir(parents=True, exist_ok=True)
    (work / "results").mkdir(parents=True, exist_ok=True)
    sel = np.zeros((H, W), bool)
    sel[4:14, 5:15] = True
    n_sel = int(sel.sum())
    for p in (1, 2):
        with h5py.File(work / "plan" / f"path{p}.h5", "w") as f:
            f["sources/sel"] = sel.astype(np.uint8)
            f.attrs["settings_hash"] = PLAN_HASH
        halves = (
            np.arange(0, n_sel // 2 + 1),
            np.arange(n_sel // 2, n_sel),
        )
        for t in range(N_TRIALS):
            g0 = float(gain_of_trial(t))
            write_results(
                work / "results" / f"path{p}_trial{t + 1:04d}.h5",
                {
                    k: (
                        pix,
                        {"baseline_step": {"gain": [g0 * (k + 1)] * pix.size}},
                    )
                    for k, pix in enumerate(halves)
                },
                container_hash=f"c{p}-{t}",
            )
    return work
