"""Make a ``spine_summary`` file from a SILo run (``giant spine readout``).

Inputs:

* a run folder with ``experiment_summary.h5`` and ``per_trial_summary.h5``
  (GIAnT-MATLAB or GIAnT-Python raster SILo);
* the session's ``trial_table.h5`` (the registered movies' names, which
  give the baseline files, and the raw files' acquisition trial numbers);
* optionally the folder of the F0 network's ``<stem>_BASELINE-80Hz.h5``
  files and the motion-correction folder they were predicted from.

Images: the run's per-trial mean images (``mean_ch<c>``), the run's own
per-trial F0 images where it has them (``f0_silo_ch<c>``, GIAnT-MATLAB's
spine-size branch), and the network's per-trial F0 images (``f0_net``).
The network images take minutes (every trial's baseline is read); they are
cached per path under a key of their inputs' identities and settings, so a
rerun with other readout settings reuses them.

Every image is read out in the same windows: pixels with a value in every
trial used, in every image (:func:`~giant_python.spine.readout.
common_valid`), so the variants can be compared trial by trial.
"""

from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np

from .gains import gain_maps, results_files
from .images import F0Stack, file_identity, network_f0_stack, stack_key
from .product import FORMAT_VERSION, _code_version, write_product
from .readout import common_valid, mask_mean, norm_log, project
from .rois import masks_roiset, sources_roiset
from .summary_io import RunSummary

_TRIAL_RE = re.compile(r"TRIAL0*(\d+)")
_REGISTERED = "_REGISTERED"


def _stem(name) -> str | None:
    """A registered movie's trial stem (``E1T1DMD1``), or None."""
    name = str(name) if name is not None else ""
    if _REGISTERED not in name:
        return None
    return Path(name).name.split(_REGISTERED)[0]


def table_trials(table_path, path: int) -> tuple[list, np.ndarray]:
    """Each trial's stem and acquisition trial number for one path.

    Returns
    -------
    stems : list of (str or None)
    acquisition : ndarray of int (-1 where the raw name has no number)
    """
    from ..io.trial_table import read_trial_table

    tt = read_trial_table(table_path)
    mc = tt.motion_correction or {}
    fn = np.atleast_2d(np.asarray(mc.get("fn_reg_ds", []), dtype=object))
    raw = np.atleast_2d(np.asarray(tt.filename, dtype=object))
    if fn.shape[0] < path:
        raise ValueError(f"the trial table has no registered path {path}")
    stems = [_stem(x) for x in fn[path - 1]]
    acq = []
    for x in raw[path - 1] if raw.shape[0] >= path else [""] * len(stems):
        m = _TRIAL_RE.search(str(x))
        acq.append(int(m.group(1)) if m else -1)
    return stems, np.asarray(acq, np.int64)


def _f0_cache_key(stems, baseline_dir, moco_dir, tmap, shape, settings):
    """Cache key of a path's network F0 stack: its inputs' identities, the
    map, the frame and the settings."""
    ids = []
    for stem in stems:
        if stem is None:
            ids.append(None)
            continue
        b = Path(baseline_dir) / f"{stem}_BASELINE-80Hz.h5"
        a = Path(moco_dir) / f"{stem}_ALIGNMENTDATA.h5"
        ids.append(
            [
                file_identity(b) if b.is_file() else None,
                file_identity(a) if a.is_file() else None,
            ]
        )
    disp = tmap.displacement
    return stack_key(
        "f0net-v1",
        ids,
        list(shape),
        settings,
        tmap.affine,
        disp if disp is not None else np.zeros(0),
    )


def cached_f0_stack(
    cache_dir,
    path: int,
    stems,
    baseline_dir,
    moco_dir,
    tmap,
    shape,
    settings: dict,
    progress=None,
) -> tuple[F0Stack, str, bool]:
    """A path's network F0 stack, from the cache when its key matches.

    Returns
    -------
    stack : F0Stack
    key : str
        The cache key.
    reused : bool
        Whether it came from the cache.
    """
    import json

    key = _f0_cache_key(stems, baseline_dir, moco_dir, tmap, shape, settings)
    cache = Path(cache_dir) / f"f0net_Path{path}_{key}.h5"
    if cache.is_file():
        with h5py.File(cache, "r") as f:
            st = F0Stack(
                f["images"][()],
                f["n_frames"][()],
                json.loads(f.attrs["inputs"]),
                json.loads(f.attrs["attrs"]),
            )
        return st, key, True
    st = network_f0_stack(
        baseline_dir,
        moco_dir,
        stems,
        tmap,
        shape,
        align_hz=settings["align_hz"],
        trim_s=settings["trim_s"],
        motion_thresh=settings["motion_thresh"],
        discard_initial_s=settings["discard_initial_s"],
        progress=progress,
    )
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    with h5py.File(tmp, "w") as f:
        f.create_dataset(
            "images",
            data=st.images,
            chunks=(1,) + st.images.shape[1:],
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        f["n_frames"] = st.n_frames
        f.attrs["inputs"] = json.dumps(st.inputs)
        f.attrs["attrs"] = json.dumps(st.attrs, default=str)
    tmp.replace(cache)
    return st, key, False


def make_spine_summary(  # noqa: C901
    run_dir,
    table_path,
    out_dir,
    baseline_dir=None,
    moco_dir=None,
    paths=None,
    trim_s: float = 1.0,
    align_hz: float = 80.0,
    motion_thresh: float = 2.0,
    discard_initial_s: float = 0.0,
    cache_dir=None,
    overwrite: bool = False,
    images_only: bool = False,
    affine_base=None,
    gain_work_dir=None,
    progress=None,
) -> dict:
    """Read out spine sizes from a run and write a ``spine_summary`` file.

    Parameters
    ----------
    run_dir : str or Path
        Folder with the run's summaries.
    table_path : str or Path
        The session's ``trial_table.h5``.
    out_dir : str or Path
        Where the ``spine_summary_<key>.h5`` file goes.
    baseline_dir, moco_dir : str or Path, optional
        The network's baselines and the motion-correction folder; without
        them there is no ``f0_net`` image.
    paths : list of int, optional
        Imaging paths (default: all in the run).
    trim_s, align_hz, motion_thresh, discard_initial_s : float
        Settings of the network's per-trial F0 images
        (:func:`~giant_python.spine.images.kept_frames`).
    cache_dir : str or Path, optional
        Cache of the network's F0 stacks (default ``<out_dir>/cache``).
    overwrite : bool
        Rewrite an existing output with the same key.
    images_only : bool
        Only build the network's F0 stacks (the slow part).
    gain_work_dir : str or Path, optional
        Work folder of the run (a 'gain' baseline mode, N1): its results
        files' per-pixel gains give the ``f0_gain`` image, the network's F0
        image rescaled as the solver rescaled it (needs ``baseline_dir``).
    affine_base : {0, 1, None}
        Index base of the run's affine trial maps (see
        :meth:`~giant_python.spine.summary_io.RunSummary.trial_map`).
    progress : callable, optional
        Called with a message string.

    Returns
    -------
    dict
        A report: the output file, its key, and per path the images, trial
        and window counts.
    """
    if (baseline_dir is None) != (moco_dir is None):
        raise ValueError("give both --baseline-dir and --moco-dir, or neither")
    if gain_work_dir is not None and baseline_dir is None:
        raise ValueError("--gain-work-dir needs --baseline-dir")
    run = RunSummary(run_dir)
    out_dir = Path(out_dir)
    cache_dir = Path(cache_dir) if cache_dir else out_dir / "cache"
    paths = list(paths) if paths else run.paths
    settings = {
        "trim_s": float(trim_s),
        "align_hz": float(align_hz),
        "motion_thresh": float(motion_thresh),
        "discard_initial_s": float(discard_initial_s),
        "format_version": FORMAT_VERSION,
    }
    say = progress or (lambda msg: None)
    report: dict = {"paths": {}}
    inputs: dict = {
        "run": {
            "experiment_summary": file_identity(run.experiment_path),
            "per_trial_summary": (
                file_identity(run.per_trial_path)
                if run.per_trial_path.is_file()
                else None
            ),
        },
        "trial_table": file_identity(table_path),
        "baseline_dir": str(baseline_dir) if baseline_dir else None,
        "f0_keys": {},
        "gain_results": {},
    }
    stacks: dict = {}
    for n in paths:
        if n not in run.paths:
            raise ValueError(f"the run has no Path{n}")
        stems, acq = table_trials(table_path, n)
        if len(stems) != run.n_trials(n):
            raise ValueError(
                f"Path{n}: {len(stems)} trials in the table, "
                f"{run.n_trials(n)} in the run"
            )
        if baseline_dir is not None:
            shape = run.shape(n)

            def prog(t, total, stem, n=n):
                """Report a trial's start."""
                say(f"Path{n}: network F0 image {t + 1}/{total} ({stem})")

            st, key, reused = cached_f0_stack(
                cache_dir,
                n,
                stems,
                baseline_dir,
                moco_dir,
                run.trial_map(n, affine_base),
                shape,
                settings,
                prog,
            )
            stacks[n] = st
            inputs["f0_keys"][n] = key
            report["paths"][n] = {"f0_key": key, "f0_reused": reused}
            if gain_work_dir is not None:
                inputs["gain_results"][n] = [
                    [
                        file_identity(f)
                        for f in results_files(gain_work_dir, n, t)
                    ]
                    for t in range(len(stems))
                ]
    if images_only:
        report["cache_dir"] = str(cache_dir)
        return report

    key = stack_key("spine-summary", inputs, settings, _code_version())
    out = out_dir / f"spine_summary_{key}.h5"
    report["output"] = str(out)
    report["key"] = key
    if out.is_file() and not overwrite:
        report["reused"] = True
        return report

    content: dict = {}
    for n in paths:
        stems, acq = table_trials(table_path, n)
        shape = run.shape(n)
        nfr = run.trial_num_frames(n)
        n_ch = run.n_channels(n)
        images: dict = {}
        if run.has_per_trial(n, "visualizations/per_trial_mean_im"):
            for c in range(n_ch):
                say(f"Path{n}: per-trial mean images, channel {c + 1}")
                images[f"mean_ch{c + 1}"] = (
                    run.per_trial_images(n, "per_trial_mean_im", c),
                    {"source": "per_trial_mean_im", "units": "counts"},
                )
        if run.has_per_trial(n, "visualizations/per_trial_F0_im"):
            for c in range(n_ch):
                say(f"Path{n}: the run's per-trial F0 images, channel {c + 1}")
                images[f"f0_silo_ch{c + 1}"] = (
                    run.per_trial_images(n, "per_trial_F0_im", c),
                    {"source": "per_trial_F0_im", "units": "counts"},
                )
        if n in stacks and gain_work_dir is not None:
            say(f"Path{n}: gains of the run's baseline fit")
            gain, spread, _ = gain_maps(gain_work_dir, n, len(stems))
            if gain.shape[1:] != tuple(shape):
                raise ValueError(
                    f"Path{n}: the gain run's frame {gain.shape[1:]} is not "
                    f"the run's {tuple(shape)}"
                )
            images["f0_gain"] = (
                gain * stacks[n].images,
                {"source": "network baseline x SILo gain", "units": "counts"},
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                rel = spread / gain
            many = np.isfinite(rel) & (spread > 0)
            report["paths"].setdefault(n, {})["gain"] = {
                "median": float(np.nanmedian(gain)),
                "shared_pixel_share": float(
                    many.mean() / max(np.isfinite(gain).mean(), 1e-12)
                ),
                "median_rel_spread_shared": (
                    float(np.median(rel[many])) if many.any() else 0.0
                ),
            }
        if n in stacks:
            st = stacks[n]
            images["f0_net"] = (
                st.images,
                {
                    "source": "network baseline",
                    "units": "counts",
                    "n_frames": st.n_frames,
                    "settings": st.attrs,
                },
            )
        if not images:
            raise ValueError(f"Path{n}: no per-trial images in the run")
        has = np.ones(len(stems), bool)
        for stack, _ in images.values():
            flat = np.asarray(stack).reshape(stack.shape[0], -1)
            has &= np.isfinite(flat).any(axis=1)
        used = np.flatnonzero(has & (nfr > 0))
        valid = common_valid([s for s, _ in images.values()], used)
        rois = {}
        if run.has(n, "sources/spatial/profiles"):
            rois["sources"] = sources_roiset(run.profiles(n))
        labels, masks = run.user_rois(n)
        if len(labels):
            rois["user"] = masks_roiset(masks, labels)
        readout: dict = {}
        normalized: dict = {}
        for set_name, rs in rois.items():
            r: dict = {}
            if set_name == "user":
                for name, (stack, _) in images.items():
                    r[name], r[f"coverage_{name}"] = mask_mean(stack, rs)
            else:
                for name, (stack, _) in images.items():
                    r[name], r["n_pixels"] = project(stack, rs, valid)
            readout[set_name] = r
            if set_name != "sources" or used.size == 0:
                continue
            ok = np.ones(len(rs), bool)
            for name in images:
                ok &= np.all(r[name][used] > 0, axis=0)
            norm: dict = {"ok": ok}
            for name in images:
                full = np.full_like(r[name], np.nan)
                full[used] = norm_log(r[name][used], ok)
                norm[name] = full
            normalized[set_name] = norm
        activity = {}
        if "sources" in rois:
            activity["sources"] = run.activity(n)
        content[n] = {
            "shape": shape,
            "trials": {
                "table_index": np.arange(len(stems)),
                "acquisition": acq,
                "num_frames": nfr,
                "used": np.isin(np.arange(len(stems)), used),
            },
            "reference": {"mean_im": run.mean_im(n), "act_im": run.act_im(n)},
            "images": images,
            "valid": valid,
            "rois": rois,
            "readout": readout,
            "normalized": normalized,
            "activity": activity,
        }
        report["paths"].setdefault(n, {}).update(
            {
                "images": list(images),
                "trials_used": int(used.size),
                "valid_pixels": int(valid.sum()),
                "rois": {k: len(v) for k, v in rois.items()},
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    say(f"writing {out}")
    write_product(tmp, content, settings, inputs, run_dir)
    tmp.replace(out)
    report["reused"] = False
    return report
