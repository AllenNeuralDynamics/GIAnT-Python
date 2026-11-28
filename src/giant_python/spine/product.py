"""The ``spine_summary`` file: everything the spine-size viewer and later
analyses need from one run, in one compact HDF5 file.

Layout (row-major, numpy orientation; images ``(trials, rows, cols)`` in the
run's consensus frame, zero-based coordinates)::

    /                     attrs: format, format_version, code_version,
                          created, run_dir, settings (JSON), inputs (JSON)
    /Path<n>/             attrs: rows, cols, n_trials, activity_channel
                          (one-based file channel of the run's activity)
      trials/table_index  (T,) zero-based trial of the trial table
      trials/acquisition  (T,) acquisition trial number (from the raw file)
      trials/num_frames   (T,) extraction frames of the run (0 = not
                          extracted)
      reference/mean_im   (C, H, W) the run's mean image
      reference/act_im    (H, W) the run's activity image
      images/<name>       (T, H, W) float32 per-trial images; attrs: source,
                          units, covers ('field': sets the windows;
                          'fitted': values only at the pixels SILo
                          fitted); n_frames (T,) where known. ``mean_ch<c>``
                          and ``f0_silo_ch<c>`` are of file channel <c>
                          (one-based; the activity channel is the group's
                          activity_channel); ``f0_net`` and ``f0_gain`` are
                          of the channel the baselines were predicted from
      valid               (H, W) uint8 pixels the windows are restricted to
      rois/<set>/...      ids, labels, rows, cols, ptr, index, weight
                          (:class:`~giant_python.spine.rois.RoiSet`)
      readout/<set>/<name>      (T, R) readout of image <name>
      readout/<set>/n_pixels    (R,) valid pixels per window
      readout/user_any/<name>   (T, R) user ROIs as GIAnT-MATLAB's
                                user_rois/per_trial_F0 (every pixel with a
                                value in the trial)
      readout/user_any/coverage_<name>  (T, R) share of the ROI's pixels
      normalized/<set>/<name>   (T, R) ``norm_log`` of the readout
      normalized/<set>/ok       (R,) windows in the per-trial median
      activity/<set>/<name>     (T, R) per-trial activity (sources)
"""

from __future__ import annotations

import json
import time

import h5py
import numpy as np

from .rois import RoiSet

FORMAT = "giant_python.spine_summary"
FORMAT_VERSION = 1

_IMAGE_OPTS = {"compression": "gzip", "compression_opts": 4, "shuffle": True}


def _code_version() -> str:
    """The installed GIAnT-Python version."""
    try:
        from importlib.metadata import version

        return version("giant-python")
    except Exception:  # pragma: no cover - not installed as a package
        return "unknown"


def write_roiset(g: h5py.Group, rois: RoiSet) -> None:
    """Store an :class:`RoiSet` in a group."""
    g.attrs["kind"] = rois.kind
    g.attrs["meta"] = json.dumps(rois.meta)
    g["ids"] = rois.ids
    g["labels"] = np.asarray(rois.labels, dtype=h5py.string_dtype())
    g["rows"] = rois.rows
    g["cols"] = rois.cols
    g["ptr"] = rois.ptr
    g.create_dataset("index", data=rois.index, **_IMAGE_OPTS)
    g.create_dataset("weight", data=rois.weight.astype(np.float32))


def read_roiset(g: h5py.Group, shape: tuple) -> RoiSet:
    """Read an :class:`RoiSet` stored by :func:`write_roiset`."""
    labels = [
        x.decode() if isinstance(x, bytes) else str(x) for x in g["labels"][()]
    ]
    return RoiSet(
        kind=str(g.attrs["kind"]),
        shape=tuple(shape),
        ids=g["ids"][()],
        labels=labels,
        rows=g["rows"][()],
        cols=g["cols"][()],
        ptr=g["ptr"][()],
        index=g["index"][()],
        weight=g["weight"][()].astype(np.float64),
        meta=json.loads(g.attrs.get("meta", "{}")),
    )


def write_product(path, paths: dict, settings: dict, inputs: dict, run_dir):
    """Write a ``spine_summary`` file.

    Parameters
    ----------
    path : str or Path
        Output file (overwritten).
    paths : dict
        Per imaging path number, a dict with ``shape``, ``trials`` (dict of
        arrays), ``reference`` (dict), ``images`` (name -> (stack, attrs)),
        ``valid``, ``rois`` (set -> RoiSet), ``readout`` (set -> name ->
        array), ``normalized`` (set -> name -> array, plus ``ok``) and
        ``activity`` (set -> name -> array).
    settings, inputs : dict
        JSON-serializable records of the settings and the input files.
    run_dir : str or Path
        The run the readout was made from.
    """
    with h5py.File(path, "w") as f:
        f.attrs["format"] = FORMAT
        f.attrs["format_version"] = FORMAT_VERSION
        f.attrs["code_version"] = _code_version()
        f.attrs["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        f.attrs["run_dir"] = str(run_dir)
        f.attrs["settings"] = json.dumps(settings, sort_keys=True)
        f.attrs["inputs"] = json.dumps(inputs, sort_keys=True, default=str)
        for n, p in sorted(paths.items()):
            g = f.create_group(f"Path{n}")
            g.attrs["rows"], g.attrs["cols"] = p["shape"]
            g.attrs["n_trials"] = len(p["trials"]["table_index"])
            g.attrs["activity_channel"] = int(p.get("activity_channel", 1))
            for k, v in p["trials"].items():
                g[f"trials/{k}"] = np.asarray(v)
            for k, v in p["reference"].items():
                g.create_dataset(
                    f"reference/{k}",
                    data=np.asarray(v, np.float32),
                    **_IMAGE_OPTS,
                )
            for name, (stack, attrs) in p["images"].items():
                stack = np.asarray(stack, np.float32)
                d = g.create_dataset(
                    f"images/{name}",
                    data=stack,
                    chunks=(1,) + stack.shape[1:],
                    **_IMAGE_OPTS,
                )
                for ak, av in attrs.items():
                    d.attrs[ak] = (
                        av if not isinstance(av, dict) else json.dumps(av)
                    )
            g.create_dataset(
                "valid", data=np.asarray(p["valid"], np.uint8), **_IMAGE_OPTS
            )
            for set_name, rois in p["rois"].items():
                write_roiset(g.create_group(f"rois/{set_name}"), rois)
            for group in ("readout", "normalized", "activity"):
                for set_name, arrays in p.get(group, {}).items():
                    for name, arr in arrays.items():
                        g[f"{group}/{set_name}/{name}"] = np.asarray(arr)


def read_product(path) -> dict:
    """Read a ``spine_summary`` file into nested dicts (images included).

    Returns
    -------
    dict
        ``attrs`` (with ``settings`` and ``inputs`` decoded) and, per path
        number, the groups of :func:`write_product` (``rois`` as
        :class:`RoiSet`).
    """
    out: dict = {"paths": {}}
    with h5py.File(path, "r") as f:
        if f.attrs.get("format") != FORMAT:
            raise ValueError(f"{path} is not a spine_summary file")
        attrs = {k: f.attrs[k] for k in f.attrs}
        attrs["settings"] = json.loads(attrs.get("settings", "{}"))
        attrs["inputs"] = json.loads(attrs.get("inputs", "{}"))
        out["attrs"] = attrs
        for key in f.keys():
            if not key.startswith("Path"):
                continue
            g = f[key]
            shape = (int(g.attrs["rows"]), int(g.attrs["cols"]))
            p: dict = {
                "shape": shape,
                "activity_channel": int(g.attrs.get("activity_channel", 1)),
            }
            for grp in ("trials", "reference", "images"):
                p[grp] = {k: g[grp][k][()] for k in g.get(grp, {})}
            p["image_attrs"] = {
                k: dict(g["images"][k].attrs) for k in g.get("images", {})
            }
            p["valid"] = g["valid"][()].astype(bool)
            p["rois"] = {
                k: read_roiset(g["rois"][k], shape) for k in g.get("rois", {})
            }
            for grp in ("readout", "normalized", "activity"):
                p[grp] = {
                    s: {k: g[grp][s][k][()] for k in g[grp][s]}
                    for s in g.get(grp, {})
                }
            out["paths"][int(key[4:])] = p
    return out
