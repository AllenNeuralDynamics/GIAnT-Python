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
trial used, in every image that covers the field of view (the mean images
and ``f0_net``; :func:`~giant_python.spine.readout.common_valid`), so the
variants can be compared trial by trial. The images that have values only
at the pixels SILo fitted (``f0_silo_ch<c>``, ``f0_gain``) are read out in
the same windows, with NaN where they do not cover one, so that they do not
take pixels or trials away from the other images; the report counts those
windows and trials. User ROIs are also read out as GIAnT-MATLAB's
``user_rois/per_trial_F0`` (``readout/user_any``), whose windows can differ
between trials and images.

Each image's normalized source sizes then go through the trend tests of
:mod:`~giant_python.spine.trends` (step, brief, and step beyond the shared
trend, with permutation p-values and Benjamini-Hochberg q-values per
path), over the trials in which the image has values.
"""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath

import h5py
import numpy as np

from .gains import (
    check_gain_baselines,
    gain_maps,
    plan_file,
    results_files,
)
from .images import F0Stack, file_identity, network_f0_stack, stack_key
from .product import FORMAT_VERSION, _code_version, write_product
from .readout import common_valid, mask_mean, norm_log, project
from .rois import masks_roiset, sources_roiset
from .summary_io import RunSummary
from .trends import trend_tests

_TRIAL_RE = re.compile(r"TRIAL0*(\d+)")
_REGISTERED = "_REGISTERED"
# version of the network F0 cache's layout (part of its key)
_F0_CACHE_FORMAT = 1


def _stem(name) -> str | None:
    """A registered movie's trial stem (``E1T1DMD1``), or None."""
    name = str(name) if name is not None else ""
    if _REGISTERED not in name:
        return None
    # tables written on Windows keep backslashes, which Linux paths do not
    # split
    return _base_name(name).split(_REGISTERED)[0]


def _replay_report(out: Path, report: dict, say) -> None:
    """Fill ``report`` from the report stored in a reused summary file
    (keeping what this call already found, such as the network images'
    cache keys) and pass its notes to ``say`` again."""
    with h5py.File(out, "r") as f:
        v = f.attrs.get("report")
    if v is None:
        return
    stored = json.loads(v.decode() if isinstance(v, bytes) else v)
    for n, rec in stored.get("paths", {}).items():
        mine = report["paths"].setdefault(int(n), {})
        for k, val in rec.items():
            mine.setdefault(k, val)
    for msg in stored.get("notes", []):
        say(msg)


def _trials_and_window(n: int, images: dict, nfr, say):
    """The trials read out and the images that set the windows.

    The images that cover the field of view (``covers`` 'field') set them:
    a trial is used if it has frames and each of them has values in it.
    An image without any values is left out (with a note), so that it does
    not empty every window. Without field images, all images set them.
    """
    frames = np.asarray(nfr) > 0
    has = {
        k: np.isfinite(np.asarray(s).reshape(s.shape[0], -1)).any(axis=1)
        for k, (s, _) in images.items()
    }
    window = [k for k, (_, a) in images.items() if a.get("covers") == "field"]
    if not window:
        window = list(images)
    for k in list(window):
        if not (has[k] & frames).any():
            say(f"Path{n}: {k} has no values; it does not set the windows")
            window.remove(k)
    if not window:
        raise ValueError(f"Path{n}: none of the per-trial images has values")
    ok = frames.copy()
    for k in window:
        lost = np.flatnonzero(frames & ~has[k])
        if lost.size:
            say(
                f"Path{n}: {k} has no values in trial(s) "
                f"{', '.join(str(t + 1) for t in lost)}; they are not read out"
            )
        ok &= has[k]
    if frames.any() and not ok.any():
        say(
            f"Path{n}: no trial has values in every one of {', '.join(window)}"
            "; nothing is read out"
        )
    return np.flatnonzero(ok), window


def _not_covered(r: dict, images: dict, window: list, used) -> dict:
    """Per image that does not set the windows: how many windows it leaves
    without a value in some trial used where the window images have one
    (only images that leave some)."""
    if used.size == 0:
        return {}
    ref = np.ones(r[window[0]][used].shape, bool)
    for k in window:
        ref &= np.isfinite(r[k][used])
    out = {}
    for k in images:
        if k in window:
            continue
        gap = (ref & ~np.isfinite(r[k][used])).any(axis=0)
        if gap.any():
            out[k] = int(gap.sum())
    return out


def _normalize(r: dict, images: dict, window: list, used) -> dict:
    """The normalized sizes (:func:`~giant_python.spine.readout.norm_log`)
    of every image's source readouts.

    The per-trial median is taken over one set of sources (``ok``): those
    whose readouts are positive in every trial used, in every image. An
    image that does not cover a source (or a trial) leaves it out of its
    own normalization only, so that images with values only at the pixels
    SILo fitted do not change the reference set of the others.
    """
    ok = np.ones(r[window[0]].shape[1], bool)
    cover = {}
    for name in images:
        vals = r[name][used]
        rows = np.isfinite(vals).any(axis=1)
        cov = np.isfinite(vals[rows]).all(axis=0) & rows.any()
        cover[name] = (rows, cov)
        pos = np.all(vals[rows] > 0, axis=0)
        ok &= pos if name in window else (pos | ~cov)
    norm: dict = {"ok": ok}
    for name in images:
        rows, cov = cover[name]
        full = np.full_like(r[name], np.nan)
        if rows.any():
            full[used[rows]] = norm_log(r[name][used[rows]], ok & cov)
        norm[name] = full
    return norm


def _trends(n, norm: dict, acq, opts: dict, workers, say) -> tuple:
    """The trend tests of every image's normalized source sizes
    (:func:`~giant_python.spine.trends.trend_tests`), over the trials in
    which the image has values.

    Returns
    -------
    trends : dict
        Per image, the arrays of the product's ``trends`` group (trial
        positions as zero-based trial-table indices, -1 where untested or
        where a test found no split or run) and its attributes.
    summary : dict
        Per image, the tested sources, the shared mode's variance share and
        the counts of the tests.
    """
    trends: dict = {}
    summary: dict = {}
    n_trials = len(acq)
    for name, vals in norm.items():
        if name == "ok":
            continue
        rows = np.flatnonzero(np.isfinite(vals).any(axis=1))
        if rows.size < 3:
            continue
        say(
            f"Path{n}: trend tests of {name} ({int(norm['ok'].sum())} "
            f"sources, {opts['n_perm']} trial orders)"
        )
        r = trend_tests(
            vals[rows],
            norm["ok"],
            acq[rows],
            n_perm=opts["n_perm"],
            fdr_q=opts["fdr_q"],
            max_brief=opts["max_brief"],
            seed=opts["seed"],
            workers=workers,
        )
        skip = ("counts", "var_share", "shared_v", "blocks", "brief_start")
        arrays = {
            k: r[k] for k in r if k not in skip and not k.endswith("_k")
        }
        # trial positions as trial-table indices
        for test in ("step", "resid"):
            k = r[f"{test}_k"]
            arrays[f"{test}_after"] = np.where(
                k > 0, rows[np.maximum(k - 1, 0)], -1
            )
        bs = r["brief_start"]
        arrays["brief_first"] = np.where(bs >= 0, rows[np.maximum(bs, 0)], -1)
        v = np.full(n_trials, np.nan)
        v[rows] = r["shared_v"]
        blocks = np.full(n_trials, -1, np.int64)
        blocks[rows] = r["blocks"]
        arrays.update({"shared_v": v, "blocks": blocks, "trials": rows})
        trends[name] = {
            "arrays": arrays,
            "attrs": dict(opts, var_share=r["var_share"]),
        }
        summary[name] = {
            "tested": int(r["tested"].sum()),
            "var_share": round(float(r["var_share"]), 4),
            "counts": r["counts"],
        }
    return trends, summary


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
    names = raw[path - 1] if raw.shape[0] >= path else [""] * len(stems)
    if len(names) != len(stems):
        raise ValueError(
            f"the trial table lists {len(names)} raw files but {len(stems)} "
            f"registered movies for path {path}"
        )
    acq = []
    for x in names:
        # the file's own name: a folder may also say TRIAL
        m = _TRIAL_RE.search(_base_name(x))
        acq.append(int(m.group(1)) if m else -1)
    return stems, np.asarray(acq, np.int64)


def _base_name(name) -> str:
    """The last part of a path written on any system."""
    name = str(name) if name is not None else ""
    return PurePosixPath(name.replace("\\", "/")).name


def _write_atomic(dest, write) -> None:
    """Write ``dest`` through a temporary file of its own in the same folder
    (``write(tmp)``), then move it into place.

    Concurrent writers of the same file each have their own temporary file.
    If the move is refused because another writer's file is in place and
    open (Windows), that file is kept: the same name means the same key and
    so the same content.
    """
    import os
    import tempfile

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f"{dest.name}.", suffix=".tmp", dir=dest.parent
    )
    os.close(fd)
    tmp = Path(name)
    try:
        write(tmp)
        try:
            os.replace(tmp, dest)
        except PermissionError:
            if not dest.is_file():
                raise
            tmp.unlink(missing_ok=True)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


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
            # caches written before the untrimmed flags are made again
            st = None
            if "untrimmed" in f:
                st = F0Stack(
                    f["images"][()],
                    f["n_frames"][()],
                    f["untrimmed"][()].astype(bool),
                    json.loads(f.attrs["inputs"]),
                    json.loads(f.attrs["attrs"]),
                )
        if st is not None:
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

    def write(tmp):
        """Write the stack to ``tmp``."""
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
            f["untrimmed"] = st.untrimmed.astype(np.uint8)
            f.attrs["inputs"] = json.dumps(st.inputs)
            f.attrs["attrs"] = json.dumps(st.attrs, default=str)

    _write_atomic(cache, write)
    return st, key, False


def make_spine_summary(  # noqa: C901
    run_dir,
    table_path,
    out_dir,
    baseline_dir=None,
    moco_dir=None,
    paths=None,
    trim_s: float = 1.0,
    align_hz: float | None = None,
    motion_thresh: float | None = None,
    discard_initial_s: float | None = None,
    cache_dir=None,
    overwrite: bool = False,
    images_only: bool = False,
    affine_base=None,
    gain_work_dir=None,
    n_perm: int = 20000,
    fdr_q: float = 0.1,
    max_brief: int = 3,
    seed: int = 1,
    workers: int | None = None,
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
    trim_s : float
        Seconds left out at each end of a trial in the network's per-trial
        F0 images (:func:`~giant_python.spine.images.kept_frames`).
    align_hz, motion_thresh, discard_initial_s : float, optional
        The registered frame rate and SILo's motion discards for those
        images; by default the run's own (``params/alignHz``,
        ``params/motionThresh``, ``params/discardInitial_s``; 80, 2 and 0
        where the run does not record them), so that the images leave out
        the frames that SILo left out.
    cache_dir : str or Path, optional
        Cache of the network's F0 stacks (default ``<out_dir>/cache``).
    overwrite : bool
        Rewrite an existing output with the same key.
    images_only : bool
        Only build the network's F0 stacks (the slow part).
    gain_work_dir : str or Path, optional
        Work folder of the run (a 'gain' baseline mode, N1): its results
        files' per-pixel gains give the ``f0_gain`` image, the network's F0
        image rescaled as the solver rescaled it (needs ``baseline_dir``,
        with the baseline files the run's gains were fitted against).
    affine_base : {0, 1, None}
        Index base of the run's affine trial maps (see
        :meth:`~giant_python.spine.summary_io.RunSummary.trial_map`).
    n_perm, fdr_q, max_brief, seed : int, float, int, int
        The trend tests' random trial orders per source (0: statistics
        only), FDR level, longest brief run and seed
        (:func:`~giant_python.spine.trends.trend_tests`).
    workers : int, optional
        Threads for the permutations (the results do not depend on it).
    progress : callable, optional
        Called with a message string.

    Returns
    -------
    dict
        A report: the output file, its key, and per path the images, trial
        and window counts and the windows that images leave without a
        value. The file stores this report and its notes; when an existing
        file is reused, they are read back and the notes passed to
        ``progress`` again.
    """
    if (baseline_dir is None) != (moco_dir is None):
        raise ValueError("give both --baseline-dir and --moco-dir, or neither")
    if gain_work_dir is not None and baseline_dir is None:
        raise ValueError("--gain-work-dir needs --baseline-dir")
    run = RunSummary(run_dir)
    # the run's own discards unless overridden (before the cache keys)
    if align_hz is None:
        align_hz = run.param("alignHz", 80.0)
    if motion_thresh is None:
        motion_thresh = run.param("motionThresh", 2.0)
    if discard_initial_s is None:
        discard_initial_s = run.param("discardInitial_s", 0.0)
    out_dir = Path(out_dir)
    cache_dir = Path(cache_dir) if cache_dir else out_dir / "cache"
    paths = list(paths) if paths else run.paths
    f0_settings = {
        "trim_s": float(trim_s),
        "align_hz": float(align_hz),
        "motion_thresh": float(motion_thresh),
        "discard_initial_s": float(discard_initial_s),
        "format_version": _F0_CACHE_FORMAT,
    }
    trend_opts = {
        "n_perm": int(n_perm),
        "fdr_q": float(fdr_q),
        "max_brief": int(max_brief),
        "seed": int(seed),
    }
    settings = dict(
        f0_settings, format_version=FORMAT_VERSION, trends=trend_opts
    )
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
        "paths": [int(n) for n in paths],
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
                f0_settings,
                prog,
            )
            stacks[n] = st
            inputs["f0_keys"][n] = key
            report["paths"][n] = {"f0_key": key, "f0_reused": reused}
            if gain_work_dir is not None:
                # the gains apply only to the baselines they were fitted to
                check_gain_baselines(
                    gain_work_dir,
                    n,
                    [None if i is None else i["baseline"] for i in st.inputs],
                )
                inputs["gain_results"][n] = {
                    "plan": file_identity(plan_file(gain_work_dir, n)),
                    "results": [
                        [
                            file_identity(f)
                            for f in results_files(gain_work_dir, n, t)
                        ]
                        for t in range(len(stems))
                    ],
                }
    if images_only:
        report["cache_dir"] = str(cache_dir)
        return report

    key = stack_key("spine-summary", inputs, settings, _code_version())
    out = out_dir / f"spine_summary_{key}.h5"
    report["output"] = str(out)
    report["key"] = key
    if out.is_file() and not overwrite:
        report["reused"] = True
        _replay_report(out, report, say)
        return report

    content: dict = {}
    notes: list = []

    def note(msg):
        """A note about the readout: shown, and stored with it."""
        notes.append(msg)
        say(msg)

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
                    {
                        "source": "per_trial_mean_im",
                        "units": "counts",
                        "covers": "field",
                    },
                )
        if run.has_per_trial(n, "visualizations/per_trial_F0_im"):
            for c in range(n_ch):
                say(f"Path{n}: the run's per-trial F0 images, channel {c + 1}")
                images[f"f0_silo_ch{c + 1}"] = (
                    run.per_trial_images(n, "per_trial_F0_im", c),
                    {
                        "source": "per_trial_F0_im",
                        "units": "counts",
                        "covers": "fitted",
                    },
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
                {
                    "source": "network baseline x SILo gain",
                    "units": "counts",
                    "covers": "fitted",
                },
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
                    "covers": "field",
                    "n_frames": st.n_frames,
                    "untrimmed": st.untrimmed.astype(np.uint8),
                    "settings": st.attrs,
                },
            )
            short = np.flatnonzero(st.untrimmed)
            if short.size:
                report["paths"].setdefault(n, {})["f0_untrimmed"] = [
                    int(t) + 1 for t in short
                ]
                note(
                    f"Path{n}: f0_net of trial(s) "
                    f"{', '.join(str(t + 1) for t in short)} averages the "
                    "whole trial, too short for the trim"
                )
        if not images:
            raise ValueError(f"Path{n}: no per-trial images in the run")
        used, window = _trials_and_window(n, images, nfr, note)
        valid = common_valid([images[k][0] for k in window], used)
        rois = {}
        if run.has(n, "sources/spatial/profiles"):
            rois["sources"] = sources_roiset(run.profiles(n))
        labels, masks = run.user_rois(n)
        if len(labels):
            rois["user"] = masks_roiset(masks, labels)
        readout: dict = {}
        normalized: dict = {}
        trends: dict = {}
        for set_name, rs in rois.items():
            r: dict = {}
            if set_name == "user":
                # the same windows as the sources: valid pixels only
                vflat = valid.ravel()
                r["n_pixels"] = np.array(
                    [
                        np.count_nonzero(vflat[rs.pixels(k)[0]])
                        for k in range(len(rs))
                    ],
                    np.int64,
                )
                for name, (stack, _) in images.items():
                    r[name], _ = mask_mean(stack, rs, valid)
                # GIAnT-MATLAB's user_rois/per_trial_F0: every pixel with a
                # value in that trial (windows differ between trials and
                # images)
                anyp: dict = {}
                for name, (stack, _) in images.items():
                    anyp[name], anyp[f"coverage_{name}"] = mask_mean(stack, rs)
                readout["user_any"] = anyp
            else:
                for name, (stack, _) in images.items():
                    r[name], r["n_pixels"] = project(stack, rs, valid)
            readout[set_name] = r
            gaps = _not_covered(r, images, window, used)
            if gaps:
                rep_n = report["paths"].setdefault(n, {})
                rep_n.setdefault("not_covered", {})[set_name] = gaps
                for name, k in gaps.items():
                    note(
                        f"Path{n}: {name} does not cover {k} of {len(rs)} "
                        f"{set_name} windows in every trial used (NaN there)"
                    )
            if set_name != "sources" or used.size == 0:
                continue
            normalized[set_name] = _normalize(r, images, window, used)
            trends[set_name], summ = _trends(
                n, normalized[set_name], acq, trend_opts, workers, say
            )
            report["paths"].setdefault(n, {})["trends"] = summ
        activity = {}
        if "sources" in rois:
            activity["sources"] = run.activity(n)
        content[n] = {
            "shape": shape,
            "activity_channel": run.activity_channel(),
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
            "trends": trends,
            "activity": activity,
        }
        report["paths"].setdefault(n, {}).update(
            {
                "images": list(images),
                "window_images": list(window),
                "trials_used": int(used.size),
                "valid_pixels": int(valid.sum()),
                "rois": {k: len(v) for k, v in rois.items()},
            }
        )
    say(f"writing {out}")
    stored = {"paths": report["paths"], "notes": notes}
    _write_atomic(
        out,
        lambda tmp: write_product(
            tmp, content, settings, inputs, run_dir, report=stored
        ),
    )
    report["reused"] = False
    return report
