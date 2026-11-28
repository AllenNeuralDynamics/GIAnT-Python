"""What the spine viewer shows, read from ``spine_summary`` files.

A :class:`SpineStore` holds one or more products (``spine_summary_*.h5``)
and the marks file. Products are read lazily and again when they change on
disk, so a rerun of ``giant spine readout`` shows up on the page's next
request. The per-trial images are read per image and kept in a small
cache, with what the views derive from them:

* **image** view: each trial divided by its median over the pixels that
  have a value in every trial (bleach-normalized), or their mean;
* **change** view (``spineViewer.m``): the normalized trials blurred with
  a 0.7 px Gaussian that ignores NaN; a trial's change against the mean of
  all trials or against the trial before it (in the same block of
  consecutive acquisition numbers), and for the mean, the last block
  against the earlier ones (one block: the second half against the first);
  with a brightness that follows the mean image and saturates at the 75th
  percentile of the pixels near sources, so that dim pixels fade;
* **activity**: the run's activity image as a robust z-score (median and
  MAD, as source detection sets its threshold).

Marks (``spine_marks.csv``: Path, Source, Label, Note, Row0, Col0, Updated)
belong to a position in the consensus frame and attach to the nearest
source within 2.5 px, so marks made on one run show on another run of the
same session.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np

from ..product import FORMAT, read_roiset
from ..rois import MARK_COLUMNS, nearest, read_marks, write_marks

MARK_LABELS = ("Growing", "Shrinking", "Unsure", "Not a spine")
MARK_RADIUS = 2.5
CROP_HALF = 9
BLUR_SIGMA = 0.7
NEAR_PX = 4.0
# MAD of a normal variable, in SDs (as source detection's threshold)
_MAD_NORMAL = 0.6741891400433162


def _decode(v):
    """A string attribute as str."""
    return v.decode() if isinstance(v, bytes) else str(v)


def _nanmean(a: np.ndarray, axis=0) -> np.ndarray:
    """``nanmean`` without the warning for all-NaN slices (NaN there)."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def blur_nan(img: np.ndarray, sigma: float = BLUR_SIGMA) -> np.ndarray:
    """Gaussian blur that ignores NaN (normalized convolution), with
    MATLAB ``imgaussfilt``'s radius ``ceil(2 sigma)``; NaN stays NaN."""
    from scipy.ndimage import gaussian_filter

    m = np.isfinite(img)
    a = np.where(m, img, 0.0).astype(np.float64)
    trunc = math.ceil(2 * sigma) / sigma
    num = gaussian_filter(a, sigma, mode="nearest", truncate=trunc)
    den = gaussian_filter(
        m.astype(np.float64), sigma, mode="nearest", truncate=trunc
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~m] = np.nan
    return out.astype(np.float32)


def robust_z(img: np.ndarray) -> np.ndarray:
    """``(x - median) / (MAD / 0.674)`` over the finite pixels."""
    v = img[np.isfinite(img)]
    if v.size == 0:
        return np.full(img.shape, np.nan, np.float32)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) / _MAD_NORMAL
    return ((img - med) / (mad if mad > 0 else 1.0)).astype(np.float32)


def late_trials(blocks: np.ndarray) -> np.ndarray:
    """The 'late' trials of the mean's change view: the last block of
    consecutive acquisition numbers, or with one block the second half."""
    blocks = np.asarray(blocks)
    if blocks.size and blocks.max() > blocks.min():
        return blocks == blocks.max()
    return np.arange(blocks.size) >= blocks.size // 2


def trial_blocks_of(acquisition: np.ndarray) -> np.ndarray:
    """Blocks of consecutive acquisition numbers (as the trend tests)."""
    from ..trends import trial_blocks

    return trial_blocks(acquisition)


class _Lru:
    """A small least-recently-used cache."""

    def __init__(self, size: int):
        """Keep up to ``size`` entries."""
        self.size = size
        self.d: OrderedDict = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key, make):
        """The value of ``key``, made with ``make()`` if absent."""
        with self.lock:
            if key in self.d:
                self.d.move_to_end(key)
                return self.d[key]
        val = make()
        with self.lock:
            self.d[key] = val
            self.d.move_to_end(key)
            while len(self.d) > self.size:
                self.d.popitem(last=False)
        return val

    def clear(self):
        """Forget everything."""
        with self.lock:
            self.d.clear()


class SpineProduct:
    """One ``spine_summary`` file, re-read when it changes on disk.

    The small groups (trials, ROIs, readouts, normalized sizes, trends,
    activity) are read at once; the per-trial images per request.
    """

    def __init__(self, path, name: str | None = None, cache: int = 8):
        """Open ``path`` (checked for the format)."""
        self.file = Path(path)
        self.name = name or self.file.stem
        self._mtime = None
        self._small: dict = {}
        self._lock = threading.Lock()
        self._cache = _Lru(cache)
        # bumped when the file is read again: cache keys carry it, so a
        # result computed from the old file is never served for the new
        self._gen = 0
        self.small()

    def small(self) -> dict:
        """The product without its images, re-read if the file changed."""
        mt = self.file.stat().st_mtime_ns
        with self._lock:
            if mt != self._mtime:
                self._small = self._read_small()
                self._gen += 1
                self._cache.clear()
                self._mtime = mt
            return self._small

    def _read_small(self) -> dict:
        """Read everything but the per-trial images."""
        out: dict = {"paths": {}}
        with h5py.File(self.file, "r") as f:
            if f.attrs.get("format") != FORMAT:
                raise ValueError(f"{self.file} is not a spine_summary file")
            a = {k: f.attrs[k] for k in f.attrs}
            out["attrs"] = {
                "run_dir": _decode(a.get("run_dir", "")),
                "created": _decode(a.get("created", "")),
                "code_version": _decode(a.get("code_version", "")),
                "format_version": int(a.get("format_version", 0)),
                "settings": json.loads(_decode(a.get("settings", "{}"))),
                "report": json.loads(_decode(a.get("report", "{}"))),
            }
            for key in f:
                if key.startswith("Path"):
                    out["paths"][int(key[4:])] = self._read_path(f[key])
        return out

    @staticmethod
    def _read_path(g) -> dict:
        """One path's small groups."""
        shape = (int(g.attrs["rows"]), int(g.attrs["cols"]))
        p: dict = {
            "shape": shape,
            "activity_channel": int(g.attrs.get("activity_channel", 1)),
            "trials": {k: g["trials"][k][()] for k in g["trials"]},
            "images": {},
            "valid": g["valid"][()].astype(bool),
        }
        for name, d in g.get("images", {}).items():
            at = {}
            for k, v in d.attrs.items():
                if isinstance(v, np.ndarray):
                    at[k] = v.tolist()
                elif isinstance(v, (bytes, str)):
                    at[k] = _decode(v)
                else:
                    at[k] = v.item() if hasattr(v, "item") else v
            p["images"][name] = at
        rois = g.get("rois", {})
        p["rois"] = {k: read_roiset(rois[k], shape) for k in rois}
        for grp in ("readout", "normalized", "activity"):
            p[grp] = {
                s: {k: g[grp][s][k][()] for k in g[grp][s]}
                for s in g.get(grp, {})
            }
        p["trends"] = {
            s: {
                name: dict({k: t[k][()] for k in t}, attrs=dict(t.attrs))
                for name, t in g["trends"][s].items()
            }
            for s in g.get("trends", {})
        }
        p["reference"] = {
            k: g["reference"][k][()] for k in g.get("reference", {})
        }
        return p

    def path(self, n: int) -> dict:
        """Path ``n``'s small groups."""
        paths = self.small()["paths"]
        if n not in paths:
            raise LookupError(f"{self.name} has no Path{n}")
        return paths[n]

    # ---- images and what the views derive from them ----
    def _stack(self, n: int, image: str) -> np.ndarray:
        """A per-trial image stack (float32)."""
        with h5py.File(self.file, "r") as f:
            key = f"Path{n}/images/{image}"
            if key not in f:
                raise LookupError(f"Path{n} has no image {image}")
            return f[key][()].astype(np.float32)

    def normalized(self, n: int, image: str) -> dict:
        """Path ``n``'s image stack, each trial divided by its median over
        the pixels with a value in every trial that has values, and the
        mean of those trials (``stack``, ``mean``, ``medians``,
        ``trials``)."""

        def make():
            """Read and normalize."""
            st = self._stack(n, image)
            has = np.isfinite(st.reshape(st.shape[0], -1)).any(axis=1)
            trials = np.flatnonzero(has)
            common = (
                np.isfinite(st[trials]).all(axis=0)
                if trials.size
                else np.zeros(st.shape[1:], bool)
            )
            med = np.full(st.shape[0], np.nan, np.float32)
            for t in trials:
                v = st[t][common]
                if v.size:
                    med[t] = np.median(v)
            with np.errstate(invalid="ignore", divide="ignore"):
                norm = st / med[:, None, None]
            norm[~has] = np.nan
            with np.errstate(invalid="ignore"):
                mean = (
                    _nanmean(norm[trials], axis=0)
                    if trials.size
                    else np.full(st.shape[1:], np.nan, np.float32)
                )
            return {
                "stack": norm.astype(np.float32),
                "mean": mean.astype(np.float32),
                "medians": med,
                "trials": trials,
            }

        self.small()
        return self._cache.get((self._gen, "norm", n, image), make)

    def change_prep(self, n: int, image: str) -> dict:
        """The change views' inputs: the blurred normalized trials, their
        mean and the brightness (``blurred``, ``mean``, ``brightness``)."""

        def make():
            """Blur and set the brightness."""
            nz = self.normalized(n, image)
            st = nz["stack"]
            blurred = np.stack([blur_nan(st[t]) for t in range(st.shape[0])])
            with np.errstate(invalid="ignore"):
                mean = (
                    _nanmean(blurred[nz["trials"]], axis=0)
                    if nz["trials"].size
                    else np.full(st.shape[1:], np.nan, np.float32)
                )
            b = np.sqrt(np.maximum(mean, 0))
            b[~np.isfinite(mean)] = np.nan
            near = self.near_sources(n) & np.isfinite(b)
            ref = b[near] if near.any() else b[np.isfinite(b)]
            top = np.percentile(ref, 75) if ref.size else 1.0
            bright = np.minimum(b / (top if top > 0 else 1.0), 1.0)
            bright[~np.isfinite(bright)] = 0
            return {
                "blurred": blurred.astype(np.float32),
                "mean": mean.astype(np.float32),
                "brightness": bright.astype(np.float32),
            }

        self.small()
        return self._cache.get((self._gen, "change", n, image), make)

    def near_sources(self, n: int) -> np.ndarray:
        """Pixels within 4 px of a source's centre."""
        p = self.path(n)
        h, w = p["shape"]
        out = np.zeros((h, w), bool)
        rs = p["rois"].get("sources")
        if rs is None:
            return out
        rr, cc = np.mgrid[0:h, 0:w]
        for r0, c0 in zip(rs.rows, rs.cols):
            if not (np.isfinite(r0) and np.isfinite(c0)):
                continue
            r1, r2 = int(max(0, r0 - NEAR_PX)), int(min(h, r0 + NEAR_PX + 1))
            c1, c2 = int(max(0, c0 - NEAR_PX)), int(min(w, c0 + NEAR_PX + 1))
            sub = (rr[r1:r2, c1:c2] - r0) ** 2 + (cc[r1:r2, c1:c2] - c0) ** 2
            out[r1:r2, c1:c2] |= sub <= NEAR_PX**2
        return out

    def blocks(self, n: int) -> np.ndarray:
        """Blocks of consecutive acquisition numbers of all trials."""
        return trial_blocks_of(self.path(n)["trials"]["acquisition"])

    def change(self, n: int, image: str, trial: int, vs: str) -> np.ndarray:
        """``(2, rows, cols)``: the relative change (NaN where there is
        none) and the brightness. ``trial`` -1: the late trials against
        the earlier ones; ``vs`` 'average' or 'previous'."""
        cp = self.change_prep(n, image)
        bl = cp["blurred"]
        trials = self.normalized(n, image)["trials"]
        blocks = self.blocks(n)
        with np.errstate(invalid="ignore", divide="ignore"):
            if trial < 0:
                late = late_trials(blocks[trials])
                a = _nanmean(bl[trials[late]], axis=0)
                b = _nanmean(bl[trials[~late]], axis=0)
                c = a / b - 1
            elif vs == "previous":
                prev = trial - 1
                if prev >= 0 and blocks[prev] == blocks[trial]:
                    c = bl[trial] / bl[prev] - 1
                else:
                    c = np.full(bl.shape[1:], np.nan, np.float32)
            else:
                c = bl[trial] / cp["mean"] - 1
        c = np.where(np.isfinite(cp["mean"]), c, np.nan)
        return np.stack([c, cp["brightness"]]).astype(np.float32)

    def field(self, n: int, image: str, trial: int, view: str, vs: str):
        """The field image of a view: ``(layers, rows, cols)`` float32.

        ``view`` 'image' (one layer: the normalized trial, or their mean
        for ``trial`` -1), 'change' (two layers, :meth:`change`) or
        'activity' (one layer: the run's activity image as a robust z;
        ``image`` and ``trial`` are ignored).
        """
        if view == "activity":
            act = self.path(n)["reference"].get("act_im")
            if act is None:
                raise LookupError(f"Path{n} has no activity image")
            return robust_z(np.asarray(act, np.float32))[None]
        if view == "change":
            return self.change(n, image, trial, vs)
        nz = self.normalized(n, image)
        img = nz["mean"] if trial < 0 else nz["stack"][trial]
        return img[None].astype(np.float32)


class SpineStore:
    """The products and marks behind the viewer's API."""

    def __init__(self, products, marks=None, crop_half: int = CROP_HALF):
        """``products``: (name or None, file) pairs; ``marks``: the marks
        CSV (created on the first mark; None: no marks)."""
        self.products = [SpineProduct(f, name) for name, f in products]
        names = [p.name for p in self.products]
        if len(set(names)) != len(names):
            for i, p in enumerate(self.products):
                p.name = f"{i + 1}: {p.name}"
        self.marks_path = Path(marks) if marks else None
        self.crop_half = int(crop_half)
        self._marks_lock = threading.Lock()

    def product(self, i: int) -> SpineProduct:
        """The ``i``-th product."""
        if not 0 <= i < len(self.products):
            raise LookupError(f"no product {i}")
        return self.products[i]

    # ---- metadata and per-source data ----
    def meta(self) -> dict:
        """Products, paths, images and trend tests (JSON)."""
        out = []
        for i, pr in enumerate(self.products):
            sm = pr.small()
            paths = {}
            for n, p in sorted(sm["paths"].items()):
                tr = p["trials"]
                paths[n] = {
                    "shape": list(p["shape"]),
                    "acquisition": tr["acquisition"].tolist(),
                    "used": tr.get("used", np.ones(len(tr["acquisition"])))
                    .astype(bool)
                    .tolist(),
                    "blocks": pr.blocks(n).tolist(),
                    "images": [
                        {
                            "name": k,
                            "covers": a.get("covers", ""),
                            "source": a.get("source", ""),
                        }
                        for k, a in p["images"].items()
                    ],
                    "normalized": sorted(
                        k
                        for k in p["normalized"].get("sources", {})
                        if k != "ok"
                    ),
                    "trends": sorted(p["trends"].get("sources", {})),
                    "n_sources": len(p["rois"].get("sources", [])),
                    "n_user": len(p["rois"].get("user", [])),
                    "activity_channel": p["activity_channel"],
                    "has_activity_image": "act_im" in p["reference"],
                }
            out.append(
                {
                    "index": i,
                    "name": pr.name,
                    "file": str(pr.file),
                    "run_dir": sm["attrs"]["run_dir"],
                    "created": sm["attrs"]["created"],
                    "settings": sm["attrs"]["settings"],
                    "notes": sm["attrs"]["report"].get("notes", []),
                    "paths": paths,
                }
            )
        return {
            "products": out,
            "marks_file": str(self.marks_path) if self.marks_path else None,
            "mark_labels": list(MARK_LABELS),
            "crop_half": self.crop_half,
        }

    def sources(self, i: int, n: int) -> dict:
        """Path ``n``'s sources: positions, normalized sizes per image,
        trend tests, activity and marks (JSON, (trials, sources) arrays as
        lists of rows)."""
        p = self.product(i).path(n)
        rs = p["rois"].get("sources")
        if rs is None:
            # no sources: the same fields, empty, so that the page loads
            return {
                "n": 0,
                "ids": [],
                "rows": [],
                "cols": [],
                "n_pixels": [],
                "ok": [],
                "normalized": {},
                "trends": {},
                "activity": {},
                "marks": {"by_source": {}, "unattached": 0},
            }
        norm = p["normalized"].get("sources", {})
        trends = {}
        for name, t in p["trends"].get("sources", {}).items():
            trends[name] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in t.items()
                if k != "attrs"
            }
            trends[name]["attrs"] = {
                k: (v.item() if hasattr(v, "item") else v)
                for k, v in t["attrs"].items()
            }
        return {
            "n": len(rs),
            "ids": rs.ids.tolist(),
            "rows": rs.rows.tolist(),
            "cols": rs.cols.tolist(),
            "n_pixels": p["readout"]
            .get("sources", {})
            .get("n_pixels", np.zeros(len(rs)))
            .tolist(),
            "ok": norm.get("ok", np.zeros(len(rs), bool)).tolist(),
            "normalized": {
                k: v.tolist() for k, v in norm.items() if k != "ok"
            },
            "trends": trends,
            "activity": {
                k: v.tolist()
                for k, v in p["activity"].get("sources", {}).items()
            },
            "marks": self.marks_for(i, n),
        }

    def roi(self, i: int, n: int, k: int) -> dict:
        """Source ``k``'s window around its centre: weights and the pixels
        the windows are restricted to, on a ``2 h + 1`` square grid whose
        top-left pixel is ``origin`` (pixels outside the frame: weight 0,
        not valid)."""
        p = self.product(i).path(n)
        rs = p["rois"]["sources"]
        if not 0 <= k < len(rs):
            raise LookupError(f"no source {k}")
        hw = self.crop_half
        r0, c0 = _centre(rs.rows[k], rs.cols[k])
        w = rs.image(k)
        out_w = _crop(w, r0, c0, hw, 0.0)
        valid = _crop(p["valid"].astype(np.float64), r0, c0, hw, 0.0)
        return {
            "id": int(rs.ids[k]),
            "row": float(rs.rows[k]),
            "col": float(rs.cols[k]),
            "origin": [r0 - hw, c0 - hw],
            "size": 2 * hw + 1,
            "weights": np.round(out_w, 6).tolist(),
            "valid": (valid > 0).astype(int).tolist(),
        }

    def crops(self, i, n, k, image, view, vs) -> np.ndarray:
        """Source ``k``'s crops of an image: ``(layers, 1 + trials, s, s)``,
        the mean first (for the change view the late against the early
        trials), then each trial (NaN where it has no value)."""
        pr = self.product(i)
        p = pr.path(n)
        rs = p["rois"]["sources"]
        if not 0 <= k < len(rs):
            raise LookupError(f"no source {k}")
        hw = self.crop_half
        r0, c0 = _centre(rs.rows[k], rs.cols[k])
        n_t = len(p["trials"]["acquisition"])
        layers = 2 if view == "change" else 1
        s = 2 * hw + 1
        out = np.full((layers, n_t + 1, s, s), np.nan, np.float32)
        # each frame is cropped as it is made: only the tiles are kept
        for j, t in enumerate([-1] + list(range(n_t))):
            if view == "change":
                frame = pr.change(n, image, t, vs)
            else:
                frame = pr.field(n, image, t, "image", vs)
            for layer in range(layers):
                out[layer, j] = _crop(frame[layer], r0, c0, hw, np.nan)
        return out

    # ---- marks ----
    def _read_marks(self) -> list:
        """All marks in the file (none without one)."""
        if self.marks_path is None or not self.marks_path.is_file():
            return []
        return read_marks(self.marks_path)

    def marks_for(self, i: int, n: int) -> dict:
        """Path ``n``'s marks, attached to the product's sources: per
        source index the label, note and time (the newest mark wins when
        two land on one source), and the marks with no source here."""
        p = self.product(i).path(n)
        rs = p["rois"].get("sources")
        mine = [m for m in self._read_marks() if m["Path"] == n]
        if not mine or rs is None:
            return {"by_source": {}, "unattached": len(mine)}
        k = nearest(
            rs,
            [m["Row0"] for m in mine],
            [m["Col0"] for m in mine],
            MARK_RADIUS,
        )
        by: dict = {}
        off = 0
        for m, s in zip(mine, k):
            if s < 0:
                off += 1
                continue
            cur = by.get(int(s))
            if cur is None or m.get("Updated", "") >= cur["updated"]:
                by[int(s)] = {
                    "label": m.get("Label", ""),
                    "note": m.get("Note", ""),
                    "updated": m.get("Updated", ""),
                }
        return {"by_source": by, "unattached": off}

    def set_mark(self, i: int, n: int, k: int, label: str, note: str):
        """Mark source ``k`` (``label`` '' clears its marks), keyed by its
        position; returns the path's marks."""
        if self.marks_path is None:
            raise ValueError("no marks file (give --marks)")
        if label and label not in MARK_LABELS:
            raise ValueError(f"unknown mark {label!r}")
        p = self.product(i).path(n)
        rs = p["rois"]["sources"]
        if not 0 <= k < len(rs):
            raise LookupError(f"no source {k}")
        with self._marks_lock:
            marks = self._read_marks()
            here = [m for m in marks if m["Path"] == n]
            at = nearest(
                rs,
                [m["Row0"] for m in here],
                [m["Col0"] for m in here],
                MARK_RADIUS,
            )
            drop = {id(m) for m, s in zip(here, at) if s == k}
            marks = [m for m in marks if id(m) not in drop]
            if label:
                marks.append(
                    {
                        "Path": n,
                        "Source": int(rs.ids[k]) + 1,
                        "Label": label,
                        "Note": note or "",
                        "Row0": float(rs.rows[k]),
                        "Col0": float(rs.cols[k]),
                        "Updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
            # by path, in the order written within it (stable), so that of
            # two marks with the same time the later one still wins
            marks.sort(key=lambda m: m["Path"])
            _write_marks_atomic(self.marks_path, marks)
        return self.marks_for(i, n)


def _centre(r, c) -> tuple:
    """The integer pixel nearest a centroid."""
    return int(np.floor(r + 0.5)), int(np.floor(c + 0.5))


def _crop(img: np.ndarray, r0: int, c0: int, hw: int, fill) -> np.ndarray:
    """The ``2 hw + 1`` square of ``img`` centred on ``(r0, c0)``, with
    ``fill`` outside the frame."""
    s = 2 * hw + 1
    out = np.full((s, s), fill, dtype=np.float64)
    h, w = img.shape
    r1, r2 = max(0, r0 - hw), min(h, r0 + hw + 1)
    c1, c2 = max(0, c0 - hw), min(w, c0 + hw + 1)
    if r1 < r2 and c1 < c2:
        out[
            r1 - (r0 - hw) : r2 - (r0 - hw), c1 - (c0 - hw) : c2 - (c0 - hw)
        ] = img[r1:r2, c1:c2]
    return out


def _write_marks_atomic(path: Path, marks: list) -> None:
    """Write the marks through a temporary file in the same folder."""
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=f"{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    try:
        write_marks(tmp, marks)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


__all__ = [
    "MARK_COLUMNS",
    "MARK_LABELS",
    "SpineProduct",
    "SpineStore",
    "blur_nan",
    "late_trials",
    "robust_z",
]
