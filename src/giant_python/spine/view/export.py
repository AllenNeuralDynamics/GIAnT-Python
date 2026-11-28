"""``giant spine export-html``: one self-contained page of a
``spine_summary`` file, for reading away from the data (a private web page,
a file to send).

The page is the viewer's own (:mod:`.server`'s ``page.html``) with the data
embedded, gzip-compressed and base64-encoded: per path the sources' sizes,
trend tests and marks; the field's mean of trials in each image (image and
change views) and the activity image; and, for selected sources, their
windows and per-trial crops. Images are stored as uint8 codes per layer
(``value = offset + scale * code``, 255 = NaN): the image view between its
0.05th and 99.95th percentiles, changes between -50% and +50%, brightness
between 0 and 1. Trials other than the mean, and crops of other sources,
are not in the page; the page says so where they would be.
"""

from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

import numpy as np

from .data import SpineStore
from .server import PAGE, json_safe, product_args

# the largest page an Artifact takes
MAX_BYTES = 16 * 1024 * 1024
_PREFER = ("f0_gain", "f0_net", "f0_silo_ch1", "mean_ch1")


def field_key(n: int, image: str, trial: int, view: str, vs: str) -> str:
    """Key of a field image in the page's data (as the page's
    ``fieldKey``)."""
    if view == "activity":
        return f"{n}||-1|activity|"
    v = vs if view == "change" and trial >= 0 else ""
    return f"{n}|{image}|{trial}|{view}|{v}"


def crop_key(n: int, k: int, image: str, view: str, vs: str) -> str:
    """Key of a source's crops in the page's data."""
    return f"{n}|{k}|{image}|{view}|{vs if view == 'change' else ''}"


def _pack_layer(a: np.ndarray, lo=None, hi=None) -> dict:
    """One layer as uint8 codes (255 = NaN) with offset and scale."""
    a = np.asarray(a, np.float32)
    fin = np.isfinite(a)
    if lo is None or hi is None:
        v = a[fin]
        lo, hi = (
            (float(np.percentile(v, 0.05)), float(np.percentile(v, 99.95)))
            if v.size
            else (0.0, 1.0)
        )
    if not hi > lo:
        hi = lo + 1.0
    scale = (hi - lo) / 254.0
    with np.errstate(invalid="ignore"):
        q = np.clip(np.round((a - lo) / scale), 0, 254)
    q = np.where(fin, q, 255).astype(np.uint8)
    return {
        "offset": float(lo),
        "scale": float(scale),
        "b64": base64.b64encode(q.tobytes()).decode("ascii"),
    }


def pack(arr: np.ndarray, kind: str) -> dict:
    """An array ``(layers, ...)`` for the page: ``kind`` 'image' (one
    layer, percentile range), 'change' (relative change and brightness)
    or 'activity' (z, -5 to 30)."""
    arr = np.asarray(arr, np.float32)
    if kind == "change":
        layers = [_pack_layer(arr[0], -0.5, 0.5), _pack_layer(arr[1], 0, 1)]
    elif kind == "activity":
        layers = [_pack_layer(arr[0], -5.0, 30.0)]
    else:
        layers = [_pack_layer(arr[0])]
    return {"shape": list(arr.shape), "layers": layers}


def primary_image(names) -> str | None:
    """The image the table leads with (the viewer's default)."""
    for nm in _PREFER:
        if nm in names:
            return nm
    return sorted(names)[0] if names else None


def pick_sources(src: dict, image: str | None, top: int) -> list:
    """Source indices for the crops: the marked ones and the ``top`` with
    the smallest q over the step, brief and beyond-trend tests of
    ``image``; ties (q at the floor of the permutations) go to the larger
    statistic of the test that gives that q."""
    picked = [int(k) for k in src.get("marks", {}).get("by_source", {})]
    tr = src.get("trends", {}).get(image) if image else None
    if tr and top > 0:

        def arr(key):
            """A per-source list as floats (None and NaN as inf)."""
            v = np.array([np.inf if x is None else x for x in tr[key]], float)
            return np.where(np.isfinite(v), v, np.inf)

        tests = ("step", "brief", "resid")
        q = np.stack([arr(f"{t}_q") for t in tests])
        t_abs = np.stack([np.abs(arr(f"{t}_t")) for t in tests])
        t_abs[~np.isfinite(t_abs)] = 0
        best = np.argmin(q, axis=0)
        cols = np.arange(q.shape[1])
        q_best, t_best = q[best, cols], t_abs[best, cols]
        order = np.lexsort((-t_best, q_best))
        picked += [int(k) for k in order if np.isfinite(q_best[k])][:top]
    seen: list = []
    for k in picked:
        if k not in seen:
            seen.append(k)
    return seen


def export_html(
    product,
    out,
    paths=None,
    sources=None,
    top: int = 30,
    crop_images=None,
    marks=None,
    title: str | None = None,
    name: str | None = None,
    crop_half: int = 9,
    progress=None,
) -> dict:
    """Write one self-contained viewer page of a ``spine_summary`` file.

    Parameters
    ----------
    product : str or Path
        The ``spine_summary`` file.
    out : str or Path
        The page to write.
    paths : list of int, optional
        Paths to include (default: all).
    sources : list of int, optional
        One-based source numbers to include crops of (default: the marked
        sources and the ``top`` by q).
    top : int
        Sources with the smallest q added per path when ``sources`` is not
        given.
    crop_images : list of str, optional
        Images of the crops (default: the F0 images, else all).
    marks : str or Path, optional
        Marks CSV whose marks are shown (not editable in the page).
    title, name : str, optional
        The page's title and the product's name on it.
    crop_half : int
        Half-width of the crops, px.
    progress : callable, optional
        Called with a message string.

    Returns
    -------
    dict
        The output file, its size, and the sources included per path.
    """
    say = progress or (lambda msg: None)
    store = SpineStore([(name, product)], marks, crop_half=crop_half)
    pr = store.products[0]
    meta = store.meta()
    pm = meta["products"][0]
    if paths:
        pm["paths"] = {n: pm["paths"][n] for n in paths if n in pm["paths"]}
    data: dict = {
        "meta": meta,
        "sources": {},
        "fields": {},
        "rois": {},
        "crops": {},
    }
    meta["marks_file"] = None
    included = {}
    for n, info in pm["paths"].items():
        src = store.sources(0, n)
        data["sources"][n] = src
        names = [x["name"] for x in info["images"]]
        for image in names:
            say(f"Path{n}: {image}")
            data["fields"][field_key(n, image, -1, "image", "")] = pack(
                pr.field(n, image, -1, "image", ""), "image"
            )
            data["fields"][field_key(n, image, -1, "change", "")] = pack(
                pr.field(n, image, -1, "change", "average"), "change"
            )
        if info["has_activity_image"]:
            data["fields"][field_key(n, "", -1, "activity", "")] = pack(
                pr.field(n, "", -1, "activity", ""), "activity"
            )
        if src.get("n", 0) == 0:
            included[n] = []
            continue
        if sources:
            ids = list(src["ids"])
            ks = [ids.index(s - 1) for s in sources if s - 1 in ids]
        else:
            ks = pick_sources(src, primary_image(info["normalized"]), top)
        cimgs = crop_images or [x for x in names if x.startswith("f0")]
        cimgs = [x for x in cimgs if x in names] or names
        for k in ks:
            data["rois"][f"{n}|{k}"] = store.roi(0, n, k)
            for image in cimgs:
                data["crops"][crop_key(n, k, image, "image", "")] = pack(
                    store.crops(0, n, k, image, "image", "average"), "image"
                )
                data["crops"][crop_key(n, k, image, "change", "average")] = (
                    pack(
                        store.crops(0, n, k, image, "change", "average"),
                        "change",
                    )
                )
        included[n] = [src["ids"][k] + 1 for k in ks]
        say(f"Path{n}: crops of {len(ks)} sources in {', '.join(cimgs)}")
    blob = json.dumps(json_safe(data), allow_nan=False).encode()
    b64 = base64.b64encode(gzip.compress(blob, 6)).decode("ascii")
    page = PAGE.read_text(encoding="utf-8")
    t = title or f"Spine sizes {pr.name}"
    page = page.replace(
        "<title>Spine size viewer</title>", f"<title>{_esc(t)}</title>", 1
    )
    tag = (
        '<script id="spine-static" type="application/octet-stream">'
        f"{b64}</script>\n<script>"
    )
    page = page.replace("<script>", tag, 1)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    size = out.stat().st_size
    if size > MAX_BYTES:
        say(
            f"{out} is {size / 2**20:.1f} MB, over the 16 MB of an Artifact: "
            "fewer sources (--top, --sources) or crop images make it smaller"
        )
    return {"output": str(out), "bytes": size, "sources": included}


def _esc(s: str) -> str:
    """HTML-escape a title."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def add_arguments(p) -> None:
    """``giant spine export-html``'s options."""
    p.add_argument("product", help="a spine_summary_<key>.h5 file")
    p.add_argument("--out", required=True, help="the page to write (.html)")
    p.add_argument(
        "--paths", default=None, help="paths, e.g. 1,2 (default: all)"
    )
    p.add_argument(
        "--sources",
        default=None,
        help="one-based source numbers to include crops of, e.g. 59,100 "
        "(default: the marked sources and the --top by q)",
    )
    p.add_argument(
        "--top",
        type=int,
        default=30,
        help="sources with the smallest q added per path (default 30)",
    )
    p.add_argument(
        "--crop-images",
        default=None,
        help="images of the crops, e.g. f0_net,f0_gain (default: the F0 "
        "images)",
    )
    p.add_argument(
        "--marks", default=None, help="spine_marks.csv whose marks to show"
    )
    p.add_argument("--title", default=None, help="the page's title")
    p.add_argument(
        "--name", default=None, help="the product's name on the page"
    )
    p.add_argument(
        "--crop-half",
        type=int,
        default=9,
        help="half-width of the crops, px (default 9)",
    )


def export_html_from_args(args) -> int:
    """Run ``giant spine export-html``; print a JSON report."""
    import sys

    def ints(s):
        """Comma-separated integers."""
        return [int(x) for x in s.split(",") if x.strip()] if s else None

    name, f = product_args([args.product])[0]
    rep = export_html(
        f,
        args.out,
        paths=ints(args.paths),
        sources=ints(args.sources),
        top=args.top,
        crop_images=(
            [x for x in args.crop_images.split(",") if x.strip()]
            if args.crop_images
            else None
        ),
        marks=args.marks,
        title=args.title,
        name=args.name or name,
        crop_half=args.crop_half,
        progress=lambda m: print(m, file=sys.stderr, flush=True),
    )
    print(json.dumps(rep, indent=1))
    return 0
