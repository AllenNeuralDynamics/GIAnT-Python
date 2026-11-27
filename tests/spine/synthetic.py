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


def make_session(root: Path, with_f0_silo: bool = True) -> dict:
    """Write the synthetic session under ``root``; return its folders."""
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
    exp, per = {}, {}
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
                "temporal": {
                    "dF_ls": np.tile(
                        np.arange(N_TRIALS * 10, dtype=float),
                        (len(SOURCES), 2, 1),
                    ),
                    "events": np.zeros((len(SOURCES), 2, N_TRIALS * 10)),
                },
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
        for t in range(N_TRIALS):
            g0 = float(gain_of_trial(t))
            name = work / "results" / f"path{p}_trial{t + 1:04d}.h5"
            with h5py.File(name, "w") as f:
                f["subproblems/index"] = np.array([0, 1])
                halves = (
                    np.arange(0, n_sel // 2 + 1),
                    np.arange(n_sel // 2, n_sel),
                )
                for k, pix in enumerate(halves):
                    g = f.create_group(f"subproblems/{k}")
                    g["pixels"] = pix
                    info = {
                        "baseline_step": {"gain": [g0 * (k + 1)] * pix.size}
                    }
                    g.attrs["info"] = json.dumps(info)
    return work
