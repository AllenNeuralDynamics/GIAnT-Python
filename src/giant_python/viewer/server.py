"""``giant view``: a local page for inspecting SILo runs by eye.

``giant view RUN [RUN ...]`` serves http://127.0.0.1:8766 with two views:

* **Sources**: per path, the combined activity image, each trial's
  activity image and the mean image of one or two runs, with the runs'
  sources overlaid (kept, kinetics-rejected and, when the run kept its
  plan files, seeds without a source), synchronized zoom and pan, a
  flicker between two panels, and a source's footprint, traces (denoised
  and least-squares dF, events, F0 over all trials), SNR, kinetic score
  and footprint width. With two runs, sources are marked by whether the
  other run has one within 1.5 px (one to one, closest first).
* **Movie**: one trial-path at a time, frame by frame: the registered
  80 Hz movie, each run's fitted activity rendered into it
  (:mod:`~giant_python.viewer.render`) and the residual movie - activity;
  with ``--baseline-dir`` also the network's baseline and movie -
  activity - baseline. Scroll or play through time, average frames, zoom
  and pan with all panels in step, traces of a pixel or small region.

A trial-path is loaded when it is shown (``--cache-trials`` at once).
Memory: the movie (float32) and baseline (float16) at the valid pixels,
about 6 bytes x frames x valid pixels (0.5 GB for a 780125 trial of 2254
frames x 36k pixels); the activity is computed per request from sparse
footprints.

API (GET; used by the page): ``/api/meta``, ``/api/sources?path=``,
``/api/image?path=&run=&kind=act|mean|trial_act|trial_mean&trial=``
(float32 rows x cols), ``/api/source?path=&run=&id=`` (JSON),
``/api/source_traces?path=&run=&id=`` (float32 4 x frames),
``/api/trial?path=&trial=`` (JSON),
``/api/frame?path=&trial=&t=&avg=&kernel=`` (float32 layers x valid
pixels), ``/api/trace?path=&trial=&r=&c=&rad=&kernel=`` (float32 layers x
frames; ``kernel`` exporter, matched or exact, default ``--kernel``),
``/api/progress?key=``.
"""

from __future__ import annotations

import json
import threading
import time
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import numpy as np

from .run import TRACES, SiloRun
from .trial import Session, load_trial, source_events

PAGE = Path(__file__).with_name("page.html")
DEFAULT_PORT = 8766


def _json_safe(o):
    """``o`` with non-finite floats as None (the browser rejects NaN)."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return _json_safe(o.tolist())
    if isinstance(o, np.generic):
        return _json_safe(o.item())
    return o


def _r(a, nd: int = 3) -> list:
    """An array as nested lists rounded to ``nd`` decimals."""
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


class Store:
    """The runs of one session and the trial-paths loaded from them.

    Parameters
    ----------
    runs : list of SiloRun
        One or more runs of the same session (same paths and trials).
    session : Session
        The trial table and folders.
    label : str
        Session label for the page.
    cache_trials : int
        Trial-paths kept in memory.
    blur : float
        Gaussian blur (px) of the rendered footprints.
    """

    def __init__(
        self,
        runs,
        session,
        label="",
        cache_trials=2,
        blur=0.0,
        kernel="matched",
    ):
        """Check that the runs fit together."""
        if not runs:
            raise ValueError("give at least one run")
        names = [r.name for r in runs]
        if len(set(names)) != len(names):
            raise ValueError(f"run names must differ: {names}")
        self.runs, self.session, self.label = runs, session, label
        self.cache_trials, self.blur = max(1, int(cache_trials)), blur
        self.kernel = kernel
        self.paths = runs[0].paths
        for r in runs[1:]:
            if r.paths != self.paths:
                raise ValueError(
                    f"runs have different paths: {runs[0].name} "
                    f"{self.paths}, {r.name} {r.paths}"
                )
            # the movie is read once, as the first run's activity channel
            # with the trial loader's defaults (trial._settings)
            for key, default in (("numChannels", 2), ("activityChannel", 1)):
                v0, v = runs[0].param(key, default), r.param(key, default)
                if int(v) != int(v0):
                    raise ValueError(
                        f"runs differ in {key}: {runs[0].name} {v0}, "
                        f"{r.name} {v}"
                    )
        self._cache: OrderedDict = OrderedDict()
        self._loading: dict = {}
        # one trial-path loads at a time: each needs a whole movie in memory
        self._load_slot = threading.Semaphore(1)
        self._lock = threading.Lock()
        self.progress: dict = {}

    def run(self, name: str) -> SiloRun:
        """The run called ``name``."""
        for r in self.runs:
            if r.name == name:
                return r
        raise KeyError(name)

    def meta(self) -> dict:
        """What the page lists: runs, paths, trials."""
        paths = []
        for d in self.paths:
            p = self.runs[0].path(d)
            trials = []
            for t in range(p.n_frames.size):
                trials.append(
                    {
                        "trial": t,
                        "frames200": int(p.n_frames[t]),
                        "movie": bool(self.session.has_movie(d - 1, t)),
                        "baseline": self.session.baseline_file(d - 1, t)
                        is not None,
                        "aligned": [
                            r.path(d).maps[t] is not None for r in self.runs
                        ],
                    }
                )
            paths.append(
                {
                    "path": d,
                    "shape": list(p.shape),
                    "trials": trials,
                    "alignment": [r.path(d).alignment for r in self.runs],
                }
            )
        return {
            "label": self.label,
            "blur_px": self.blur,
            "kernel": self.kernel,
            "runs": [
                {
                    "name": r.name,
                    "dir": str(r.run_dir),
                    "settings": r.describe(),
                }
                for r in self.runs
            ],
            "table": str(self.session.table),
            "moco_dir": str(self.session.moco_dir),
            "baseline_dir": str(self.session.baseline_dir or ""),
            "paths": paths,
        }

    def sources(self, d: int) -> dict:
        """Every run's sources of path ``d`` (canvas, zero-based)."""
        out = {}
        for r in self.runs:
            s = r.path(d).sources
            kept = (
                np.column_stack(
                    [
                        s.rc,
                        s.snr,
                        s.kinetic_score,
                        s.kinetic_n_events,
                        s.width,
                        s.pixel,
                    ]
                )
                if s.n
                else np.zeros((0, 8))
            )
            rej = np.column_stack(
                [s.rejected_rc, s.rejected_score, s.rejected_n_events]
            )
            seeds = None if s.unused_seeds is None else _r(s.unused_seeds, 2)
            out[r.name] = {
                "kept": _r(kept),
                "rejected": _r(rej),
                "seeds": seeds,
                "trials": int(r.path(d).n_frames.size),
            }
        return {"shape": list(self.runs[0].path(d).shape), "runs": out}

    def image(self, d: int, run: str, kind: str, trial: int) -> np.ndarray:
        """An image of the canvas as float32 ``(rows, cols)``."""
        p = self.run(run).path(d)
        if kind == "act":
            img = p.act_im
        elif kind == "mean":
            img = p.mean_im
        elif kind in ("trial_act", "trial_mean"):
            stack = p.trial_act if kind == "trial_act" else p.trial_mean
            if stack is None or not 0 <= trial < stack.shape[0]:
                raise KeyError(f"{kind} of trial {trial}")
            img = stack[trial]
        else:
            raise KeyError(kind)
        return np.ascontiguousarray(img, np.float32)

    def source(self, d: int, run: str, s: int) -> dict:
        """One kept source: values, footprint, trial positions, events."""
        r = self.run(run)
        p = r.path(d)
        src = p.sources
        if not 0 <= s < src.n:
            raise KeyError(f"source {s}")
        patch, r0, c0 = r.profile_patch(d, s)
        trc = r.trial_centroids(d)
        pos = []
        for t, m in enumerate(p.maps):
            rc = trc[t, s] if t < trc.shape[0] else src.rc[s]
            if np.isnan(rc).any():
                rc = src.rc[s]
            pos.append(None if m is None else _r(m.forward(*rc), 2))
        return {
            "id": s,
            "rc": _r(src.rc[s], 2),
            "pixel": _r(src.pixel[s]),
            "snr": float(src.snr[s]),
            "kinetic_score": float(src.kinetic_score[s]),
            "kinetic_n_events": float(src.kinetic_n_events[s]),
            "width": float(src.width[s]),
            "footprint": {"r0": r0, "c0": c0, "values": _r(patch, 5)},
            "trial_rc": pos,
            "offsets": p.offsets.tolist(),
            "hz": float(r.param("analyzeHz", 200.0)),
            "events": source_events(r, self.session, d, s),
        }

    def source_traces(self, d: int, run: str, s: int) -> np.ndarray:
        """A source's four traces ``(4, frames)`` float32."""
        tr = self.run(run).traces(d)
        n = self.run(run).path(d).lines.size
        out = np.full((len(TRACES), n), np.nan, np.float32)
        for i, name in enumerate(TRACES):
            if name in tr and 0 <= s < tr[name].shape[1]:
                out[i] = tr[name][:, s]
        return out

    def trial(self, d: int, t: int):
        """Trial-path ``(d, t)``, loaded once (a second request waits)."""
        key = (d, t)
        while True:
            with self._lock:
                if key in self._cache:
                    self._cache.move_to_end(key)
                    return self._cache[key]
                ev = self._loading.get(key)
                mine = ev is None
                if mine:
                    ev = self._loading[key] = threading.Event()
            if not mine:
                ev.wait()
                continue
            try:
                with self._load_slot:
                    tr = self._load(d, t)
                with self._lock:
                    self._cache[key] = tr
                    while len(self._cache) > self.cache_trials:
                        self._cache.popitem(last=False)
            finally:
                with self._lock:
                    self._loading.pop(key, None)
                ev.set()
            return tr

    def _load(self, d: int, t: int):
        """Load a trial-path and report the time."""
        if not self.session.has_movie(d - 1, t):
            raise KeyError(f"no movie for path {d} trial {t + 1}")
        key = f"{d}/{t}"
        self.progress[key] = "reading the movie and rendering"
        t0 = time.time()
        try:
            tr = load_trial(
                self.runs, self.session, d, t, self.blur, self.kernel
            )
        finally:
            self.progress.pop(key, None)
        print(
            f"loaded path {d} trial {t + 1}: {tr.T} frames x {tr.N} px "
            f"in {time.time() - t0:.1f} s",
            flush=True,
        )
        return tr

    def trial_static(self, d: int, t: int) -> dict:
        """What the page needs once per trial-path."""
        tr = self.trial(d, t)
        runs = {}
        for r in tr.runs:
            runs[r.name] = {
                "kept": _r(r.kept_rc, 2),
                "rejected": _r(r.rejected_rc, 2),
                "seeds": _r(r.seeds_rc, 2),
                "rendered": r.rendered,
            }
        return {
            "rows": tr.rows.tolist(),
            "cols": tr.cols.tolist(),
            "mean": _r(tr.mean, 2),
            "discard": np.flatnonzero(tr.discard).tolist(),
            "layers": tr.layers,
            "shape": [tr.T, *tr.shape],
            "markers": runs,
            "info": tr.info,
            "busiest": [tr.busiest_frame(i) for i in range(len(tr.runs))],
        }


def _ints(q: dict, *names) -> list:
    """Integer query values."""
    return [int(q[n]) for n in names]


def _route(store: Store, path: str, q: dict):
    """Answer one API request: ``(kind, payload, headers)``."""
    if path == "/api/meta":
        return "json", store.meta(), {}
    if path == "/api/sources":
        return "json", store.sources(int(q["path"])), {}
    if path == "/api/image":
        img = store.image(
            int(q["path"]), q["run"], q["kind"], int(q.get("trial", -1))
        )
        return "bin", img, {"X-Shape": json.dumps(list(img.shape))}
    if path == "/api/source":
        d, s = _ints(q, "path", "id")
        return "json", store.source(d, q["run"], s), {}
    if path == "/api/source_traces":
        d, s = _ints(q, "path", "id")
        return (
            "bin",
            store.source_traces(d, q["run"], s),
            {"X-Names": json.dumps(list(TRACES))},
        )
    if path == "/api/trial":
        d, t = _ints(q, "path", "trial")
        return "json", store.trial_static(d, t), {}
    if path == "/api/frame":
        d, t, f = _ints(q, "path", "trial", "t")
        tr = store.trial(d, t)
        f = min(max(f, 0), tr.T - 1)
        a = tr.frame(f, max(1, int(q.get("avg", 1))), q.get("kernel"))
        return "bin", a, {"X-Frame": str(f)}
    if path == "/api/trace":
        d, t, r, c = _ints(q, "path", "trial", "r", "c")
        rad = max(0, int(q.get("rad", 0)))
        a, n = store.trial(d, t).trace(r, c, rad, q.get("kernel"))
        return "bin", a, {"X-Pixels": str(n)}
    if path == "/api/progress":
        return "json", {"msg": store.progress.get(q.get("key", ""), "")}, {}
    raise LookupError(path)


class _Handler(BaseHTTPRequestHandler):
    """Serves the page and the API of ``store`` (set by
    :func:`make_handler`)."""

    store: Store

    def log_message(self, *a):
        """Quiet."""

    def _send(self, code: int, body: bytes, ctype: str, headers=None):
        """Send one response."""
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200, headers=None):
        """Send JSON (non-finite floats as null)."""
        body = json.dumps(_json_safe(obj), allow_nan=False).encode()
        self._send(code, body, "application/json", headers)

    def do_GET(self):
        """Route a GET request."""
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                return
            kind, payload, headers = _route(self.store, u.path, q)
            if kind == "json":
                self._json(payload, headers=headers)
                return
            body = np.ascontiguousarray(payload, np.float32).tobytes()
            self._send(200, body, "application/octet-stream", headers)
        except KeyError as e:
            self._json({"error": f"missing or unknown {e}"}, 400)
        except LookupError:
            self._json({"error": f"not found: {u.path}"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001  (shown on the page)
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def make_handler(store: Store):
    """The request handler class serving ``store``."""
    return type("Handler", (_Handler,), {"store": store})


def _run_args(items) -> list:
    """``NAME=DIR`` or ``DIR`` arguments as ``(name, dir)``."""
    out = []
    for item in items:
        name, sep, d = str(item).partition("=")
        if sep and name and not Path(item).exists():
            out.append((name, d))
        else:
            out.append((None, item))
    return out


def _default_table(run_dir: Path) -> Optional[Path]:
    """A trial table in or above a run folder."""
    for d in (run_dir, run_dir.parent, run_dir.parent.parent):
        if (d / "trial_table.h5").exists():
            return d / "trial_table.h5"
    return None


def make_store(args) -> Store:
    """The store for parsed ``giant view`` arguments."""
    runs = [SiloRun(d, name) for name, d in _run_args(args.runs)]
    table = args.table or _default_table(runs[0].run_dir)
    if table is None:
        raise ValueError(
            f"no trial_table.h5 in or above {runs[0].run_dir}: give --table"
        )
    session = Session(table, args.moco_dir, args.baseline_dir, args.data_dir)
    label = args.label or Path(session.table).parent.name
    return Store(
        runs, session, label, args.cache_trials, args.blur, args.kernel
    )


def make_server(store: Store, host="127.0.0.1", port=DEFAULT_PORT):
    """An HTTP server for ``store`` (not started)."""
    return ThreadingHTTPServer((host, port), make_handler(store))


def add_arguments(p) -> None:
    """``giant view``'s options."""
    p.add_argument(
        "runs",
        nargs="+",
        metavar="RUN",
        help="SILo run folders (holding experiment_summary.h5, "
        "directly or in source_extraction/), optionally as "
        "NAME=FOLDER; two or more runs of the same session are "
        "compared",
    )
    p.add_argument(
        "--table",
        default=None,
        help="the session's trial_table.h5 (default: in or "
        "above the first run's folder)",
    )
    p.add_argument(
        "--moco-dir",
        default=None,
        help="registered movies and alignment files (default: "
        "the table's savedr/motion_correction)",
    )
    p.add_argument(
        "--baseline-dir",
        default=None,
        help="F0 network baselines (<stem>_BASELINE-80Hz.h5); "
        "adds the baseline layers",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help="raw .dat/.meta folder, for the lines per cycle "
        "(default: the table's datadr; without it the frame "
        "kernel reaches 3 dt)",
    )
    p.add_argument("--label", default="", help="session label on the page")
    p.add_argument(
        "--blur",
        type=float,
        default=0.0,
        help="Gaussian blur (px) of the rendered footprints " "(default 0)",
    )
    p.add_argument(
        "--kernel",
        choices=("matched", "exporter", "exact"),
        default="matched",
        help="how SILo's 200 Hz traces become the movie's 80 Hz "
        "frames: matched (default; sqrt(dt80^2 - dt200^2), for the "
        "200 Hz frames' own smoothing), the exporter's weights (the "
        "movie's dt), or exact (r x200 + (1 - r) exporter's, r = "
        "(dt200/dt80)^2); the page switches between them",
    )
    p.add_argument(
        "--cache-trials",
        type=int,
        default=2,
        help="trial-paths kept in memory",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument(
        "--no-browser", action="store_true", help="do not open a browser tab"
    )


def serve(args) -> int:
    """Run ``giant view`` until interrupted."""
    store = make_store(args)
    server = make_server(store, args.host, args.port)
    url = f"http://{args.host}:{server.server_address[1]}/"
    names = ", ".join(r.name for r in store.runs)
    print(
        f"{store.label}: runs {names}; paths {store.paths}; serving {url} "
        "(Ctrl+C to stop)",
        flush=True,
    )
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
