"""A small synthetic SILo run for the viewer's tests.

Writes what ``giant view`` reads, in the layout ``giant silo`` writes it
(the summaries in MATLAB's orientation, h5py's axes reversed): a trial
table, two trials of one path with registered 80 Hz movies (HDF5) and
alignment files, network baselines, and both summaries. The movie is a
constant baseline plus every source's footprint in the trial's frame (the
canvas Gaussian evaluated, exactly, at the map's inverse of each trial
pixel: SILo reads a trial's data through the same map) times its 200 Hz
trace averaged into the 80 Hz frames with the given weights.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from giant_python.extraction.raster.alignment import TrialWarp
from giant_python.viewer.render import frame_weights

H, W = 40, 60
N_TRIALS = 2
DT80 = 134
SIGMA = 1.33
CENTRES = np.array([[12.3, 15.6], [20.0, 30.4], [28.7, 44.2]])
BASELINE = 10.0


def make_warp(mode: str) -> TrialWarp:
    """Each trial's map from the canvas to its registered frame."""
    shifts = np.array([[0.3, -0.4], [-0.6, 0.5]])
    if mode == "rigid":
        return TrialWarp.from_shifts(shifts)
    nodes = np.array([[10.0, 10.0], [10.0, 50.0], [30.0, 10.0], [30.0, 50.0]])
    rng = np.random.default_rng(1)
    disp = shifts[:, None, :] + rng.uniform(-0.6, 0.6, (N_TRIALS, 4, 2))
    affine = np.zeros((N_TRIALS, 2, 3))
    affine[:, :, :2] = np.eye(2)
    affine[:, :, 2] = shifts
    return TrialWarp(affine, nodes, 15.0, disp)


def gaussian(shape, centre, sigma=SIGMA, at=None):
    """A Gaussian image summing to 1 on the canvas grid (zero beyond 3
    sigma); with ``at = (rows, cols)``, the same function evaluated at
    those canvas positions instead of the grid."""
    rr, cc = np.mgrid[: shape[0], : shape[1]]

    def f(r, c):
        """The truncated, unnormalized Gaussian."""
        d2 = (r - centre[0]) ** 2 + (c - centre[1]) ** 2
        return np.where(
            d2 > (3 * sigma) ** 2, 0.0, np.exp(-d2 / (2 * sigma**2))
        )

    z = f(rr, cc).sum()
    return (f(rr, cc) if at is None else f(*at)) / z


def traces(n200: int, n_src: int) -> np.ndarray:
    """Denoised traces ``(n200, sources)``: decaying events."""
    rng = np.random.default_rng(2)
    x = np.zeros((n200, n_src))
    k = np.exp(-np.arange(40) / 4.8)
    for s in range(n_src):
        spikes = np.zeros(n200)
        spikes[rng.choice(np.arange(20, n200 - 50), 6, replace=False)] = (
            rng.uniform(20, 80, 6)
        )
        x[:, s] = np.convolve(spikes, k)[:n200]
    return x


def _put(g, name: str, a) -> None:
    """Write ``a`` as MATLAB would (h5py shows the axes reversed)."""
    a = np.asarray(a)
    g.create_dataset(name, data=a.T if a.ndim > 1 else a)


def _strings(a) -> np.ndarray:
    """A string array for the trial table."""
    return np.asarray(a, dtype=object).astype(h5py.string_dtype())


def write_session(root: Path, mode: str = "rigid", n80: int = 120):
    """Write the synthetic session under ``root``; returns a dict of its
    folders, the warp and the true trial-frame footprints and traces."""
    root = Path(root)
    moco, base, run = (root / x for x in ("moco", "base", "run"))
    for d in (moco, base, run / "source_extraction"):
        d.mkdir(parents=True, exist_ok=True)
    warp = make_warp(mode)
    n_src = CENTRES.shape[0]
    ds = 1.0 + DT80 * np.arange(n80)
    lines = np.ceil(1.0 + 53.4 * np.arange(int(ds[-1] / 53.4) + 1))
    n200 = lines.size
    x200 = [traces(n200, n_src) * (1 + 0.2 * t) for t in range(N_TRIALS)]
    true_fp, movies = [], []
    for t in range(N_TRIALS):
        w, _ = frame_weights(lines, np.zeros(n200, bool), ds, DT80, 3 * DT80)
        x80 = (w.T @ x200[t]).T  # (sources, frames)
        qr, qc = np.mgrid[:H, :W].astype(float)
        at = warp.inverse(t, qr, qc)  # canvas position of each trial pixel
        fps = np.stack([gaussian((H, W), c, at=at) for c in CENTRES])
        mov = BASELINE + np.einsum("sab,sf->fab", fps, x80)
        true_fp.append(fps)
        movies.append(mov.astype(np.float32))
        stem = f"E1T{t + 1}DMD1"
        with h5py.File(
            moco / f"{stem}_REGISTERED_DOWNSAMPLED-80Hz.h5", "w"
        ) as f:
            f["mov"] = movies[-1].transpose(0, 2, 1)  # (pages, cols, rows)
        with h5py.File(moco / f"{stem}_ALIGNMENTDATA.h5", "w") as f:
            f["DSframes"] = ds[:, None]
            f["alignHz"] = np.array([[80.0]])
            f["recNegErr"] = np.zeros((n80, 1))
        with h5py.File(base / f"{stem}_BASELINE-80Hz.h5", "w") as f:
            f["baseline"] = np.full((n80, W, H), BASELINE, np.float16)
    _write_table(run / "trial_table.h5", run, moco)
    _write_summaries(run / "source_extraction", warp, lines, x200, mode)
    return {
        "run": run,
        "moco": moco,
        "base": base,
        "warp": warp,
        "true_fp": true_fp,
        "movies": movies,
        "x200": x200,
        "lines": lines,
        "ds": ds,
    }


def _write_table(path: Path, savedr: Path, moco: Path) -> None:
    """A one-path, two-trial trial table (row-major)."""
    names = [f"E1T{t + 1}DMD1" for t in range(N_TRIALS)]
    with h5py.File(path, "w") as f:
        f["row_major"] = np.array([[1]], np.uint8)
        f["filename"] = _strings(
            [[f"x-TRIAL{t + 1:06d}.dat" for t in range(N_TRIALS)]]
        )
        g = f.create_group("motion_correction")
        g["fn_reg_ds"] = _strings(
            [[f"{n}_REGISTERED_DOWNSAMPLED-80Hz.h5" for n in names]]
        )
        g["fn_adata"] = _strings([[f"{n}_ALIGNMENTDATA.h5" for n in names]])
        f["savedr"] = _strings([[str(savedr)]])
        f["datadr"] = _strings([[str(savedr / "raw")]])


def _write_summaries(out: Path, warp, lines, x200, mode: str) -> None:
    """Both summaries of the synthetic run."""
    n_src, total = CENTRES.shape[0], lines.size * N_TRIALS
    fp = np.stack([gaussian((H, W), c) for c in CENTRES])  # canvas
    with h5py.File(out / "experiment_summary.h5", "w") as f:
        p = f.create_group("params")
        for k, v in (
            ("numChannels", 1),
            ("activityChannel", 1),
            ("alignHz", 80.0),
            ("analyzeHz", 200.0),
            ("motionThresh", 2.0),
            ("discardInitial_s", 0.0),
            ("nanThresh", 0.33),
            ("peakth", 8.0),
        ):
            p[k] = np.array([[v]])
        p["validKernel"] = np.ones((5, 5))
        p["interTrialAlignment"] = mode
        g = f.create_group("Path1")
        act = fp.sum(0)
        _put(g, "visualizations/act_im", act[None])
        _put(g, "visualizations/mean_im", (BASELINE + act)[None, None])
        rc = np.column_stack([np.zeros(n_src), CENTRES])
        _put(
            g,
            "visualizations/act_im_peaks",
            np.column_stack([np.zeros(n_src), np.round(CENTRES)]),
        )
        fi = "frame_info"
        _put(
            g,
            f"{fi}/trial_num_frames",
            np.full((N_TRIALS, 1), lines.size, np.int32),
        )
        _put(
            g,
            f"{fi}/frame_line_idxs",
            np.tile(lines, N_TRIALS)[:, None].astype(np.int32),
        )
        _put(g, f"{fi}/discard_frames", np.zeros((total, 1), np.int8))
        _put(g, f"{fi}/trial_alignment_affine", warp.affine)
        if mode == "patchwise":
            disp = np.zeros((N_TRIALS, 2, H, W), np.float32)
            rr, cc = np.mgrid[:H, :W]
            for t in range(N_TRIALS):
                r_t, c_t = warp.grid_coords(t, (H, W))
                disp[t, 0], disp[t, 1] = r_t - rr, c_t - cc
            _put(g, f"{fi}/trial_alignment_displacement", disp)
        _put(g, "sources/spatial/coords", rc)
        _put(g, "sources/spatial/profiles", fp[:, None].astype(np.float32))
        x = np.concatenate(x200, 0)  # (total, sources)
        for name, val in (
            ("dF_denoised", x),
            ("dF_ls", x + 1.0),
            ("events", np.zeros_like(x)),
            ("F0", np.full_like(x, 50.0)),
        ):
            _put(g, f"sources/temporal/{name}", val.T[:, None, :])
        _put(g, "sources/temporal/SNR", np.arange(1.0, n_src + 1)[:, None])
        _put(g, "sources/temporal/kinetic_score", np.full((n_src, 1), 0.8))
        _put(
            g,
            "sources/temporal/kinetic_n_events",
            np.full((n_src, 1), 6, np.int32),
        )
        _put(g, "sources/rejected/seed_coords", np.array([[0.0, 5.0, 5.0]]))
        _put(g, "sources/rejected/kinetic_score", np.array([[0.1]]))
        _put(g, "sources/rejected/kinetic_n_events", np.array([[3]], np.int32))
    with h5py.File(out / "per_trial_summary.h5", "w") as f:
        g = f.create_group("Path1")
        _put(
            g,
            "visualizations/per_trial_act_im",
            np.stack([fp.sum(0)] * N_TRIALS)[:, None],
        )
        _put(
            g,
            "visualizations/per_trial_mean_im",
            np.stack([BASELINE + fp.sum(0)] * N_TRIALS)[:, None, None],
        )
        _put(
            g,
            "sources/spatial/per_trial_profiles",
            np.stack([fp[:, None]] * N_TRIALS).astype(np.float32),
        )
        coords = np.zeros((N_TRIALS, n_src, 3))
        coords[:, :, 1:] = CENTRES
        _put(g, "sources/spatial/per_trial_coords", coords)
