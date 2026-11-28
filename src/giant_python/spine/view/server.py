"""``giant spine view``: a local page for spine sizes, trends and marks.

``giant spine view PRODUCT [PRODUCT ...] --marks spine_marks.csv`` serves
http://127.0.0.1:8790 (one page, :data:`PAGE`) for one or more
``spine_summary`` files of a session (:mod:`.data`): the field with the
sources, per-trial crops of the selected source, its size over the trials
in every image with the shared trend and its activity, a sortable table of
the trend tests, and marks written at once to the CSV. Several products
(runs or readout variants) are compared by switching between them at the
same path, trial, view and nearest source.

API (used by the page): GET ``/api/meta``, ``/api/sources?prod=&path=``,
``/api/field?prod=&path=&image=&trial=&view=image|change|activity&vs=``
(float32 ``layers x rows x cols``, shape in ``X-Shape``),
``/api/roi?prod=&path=&id=`` (JSON), ``/api/crops?prod=&path=&id=&image=
&view=&vs=`` (float32 ``layers x tiles x s x s``); POST ``/api/mark``
(JSON ``{prod, path, id, label, note}``; ``label`` null keeps the
source's label and sets the note).
"""

from __future__ import annotations

import json
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from .data import SpineStore

PAGE = Path(__file__).with_name("page.html")
DEFAULT_PORT = 8790


def json_safe(o):
    """``o`` with non-finite floats as None (the browser rejects NaN)."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, np.generic):
        return json_safe(o.item())
    return o


def route(store: SpineStore, path: str, q: dict):
    """Answer one GET request: ``(kind, payload, headers)``."""
    if path == "/api/meta":
        return "json", store.meta(), {}
    i = int(q.get("prod", 0))
    if path == "/api/sources":
        return "json", store.sources(i, int(q["path"])), {}
    if path == "/api/field":
        img = store.product(i).field(
            int(q["path"]),
            q.get("image", ""),
            int(q.get("trial", -1)),
            q.get("view", "image"),
            q.get("vs", "average"),
        )
        return "bin", img, {"X-Shape": json.dumps(list(img.shape))}
    if path == "/api/roi":
        return "json", store.roi(i, int(q["path"]), int(q["id"])), {}
    if path == "/api/crops":
        a = store.crops(
            i,
            int(q["path"]),
            int(q["id"]),
            q["image"],
            q.get("view", "image"),
            q.get("vs", "average"),
        )
        return "bin", a, {"X-Shape": json.dumps(list(a.shape))}
    raise LookupError(path)


class _Handler(BaseHTTPRequestHandler):
    """Serves the page and the API of ``store`` (set by
    :func:`make_handler`).

    Every request must name a trusted host (``allowed``, set by
    :func:`make_server`: the loopback names and the bound host at the
    server's port), so that a page reaching the server through DNS
    rebinding (its own name resolving to 127.0.0.1) gets nothing; marks
    must also come as JSON from a trusted origin.
    """

    store: SpineStore
    allowed: frozenset = frozenset()

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
        body = json.dumps(json_safe(obj), allow_nan=False).encode()
        self._send(code, body, "application/json", headers)

    def _guard(self, fn):
        """Run ``fn`` and answer errors as JSON."""
        try:
            fn()
        except (KeyError, ValueError) as e:
            self._json({"error": f"{type(e).__name__}: {e}"}, 400)
        except LookupError as e:
            self._json({"error": f"not found: {e}"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001  (shown on the page)
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def _untrusted_host(self) -> bool:
        """Whether the request's ``Host`` is not one of ``allowed`` (an
        empty ``allowed`` trusts every host: a server bound to all
        interfaces)."""
        if not self.allowed:
            return False
        return self.headers.get("Host", "").lower() not in self.allowed

    def do_GET(self):
        """Route a GET request."""
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}

        def go():
            """Answer it."""
            if self._untrusted_host():
                self._json({"error": "untrusted Host"}, 403)
                return
            if u.path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                return
            kind, payload, headers = route(self.store, u.path, q)
            if kind == "json":
                self._json(payload, headers=headers)
                return
            body = np.ascontiguousarray(payload, np.float32).tobytes()
            self._send(200, body, "application/octet-stream", headers)

        self._guard(go)

    def _cross_origin(self) -> bool:
        """Whether the request comes from a page of another origin: its
        ``Origin`` (or ``Referer``) is not ``http://`` and a trusted host
        (with ``allowed`` empty, not the request's own ``Host``)."""
        src = self.headers.get("Origin") or self.headers.get("Referer")
        if not src:
            return False
        o = urlparse(src)
        if o.scheme != "http":
            return True
        if self.allowed:
            return o.netloc.lower() not in self.allowed
        return o.netloc.lower() != self.headers.get("Host", "").lower()

    def do_POST(self):
        """Set a mark (JSON from a page of this server only: a page of
        another origin cannot send JSON without a preflight, which this
        server does not answer, and its Origin is refused)."""
        u = urlparse(self.path)

        def go():
            """Answer it."""
            if self._untrusted_host():
                self._json({"error": "untrusted Host"}, 403)
                return
            if u.path != "/api/mark":
                raise LookupError(u.path)
            ctype = self.headers.get("Content-Type", "")
            if ctype.split(";")[0].strip().lower() != "application/json":
                self._json({"error": "marks take application/json"}, 415)
                return
            if self._cross_origin():
                self._json(
                    {"error": "marks only from this server's page"}, 403
                )
                return
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            marks = self.store.set_mark(
                int(body.get("prod", 0)),
                int(body["path"]),
                int(body["id"]),
                (
                    None
                    if body.get("label", "") is None
                    else str(body.get("label", ""))
                ),
                str(body.get("note", "")),
            )
            self._json(marks)

        self._guard(go)


def make_handler(store: SpineStore):
    """The request handler class serving ``store``."""
    return type("Handler", (_Handler,), {"store": store})


_LOOPBACK = ("127.0.0.1", "localhost", "[::1]")


def make_server(store: SpineStore, host="127.0.0.1", port=DEFAULT_PORT):
    """An HTTP server for ``store`` (not started).

    It answers only requests whose ``Host`` is a loopback name or ``host``
    at the bound port; bound to all interfaces (``0.0.0.0``, ``::``), it
    trusts every host name.
    """
    handler = make_handler(store)
    cls = _Server6 if ":" in host else ThreadingHTTPServer
    srv = cls((host.strip("[]"), port), handler)
    handler.allowed = allowed_hosts(host, srv.server_address[1])
    return srv


def allowed_hosts(host: str, port: int) -> frozenset:
    """The ``Host`` values a server bound to ``host`` and ``port``
    trusts: the loopback names and ``host``, with the port (and without
    it at port 80, which browsers leave out); empty (every name) when
    bound to all interfaces."""
    if host in ("0.0.0.0", "::", "[::]", ""):
        return frozenset()
    names = {_bracket(h) for h in set(_LOOPBACK) | {host.lower()}}
    allowed = {f"{h}:{port}" for h in names}
    if port == 80:
        allowed |= names
    return frozenset(allowed)


class _Server6(ThreadingHTTPServer):
    """The threading server on an IPv6 address."""

    address_family = socket.AF_INET6


def _bracket(host: str) -> str:
    """An IPv6 literal in brackets, as URLs and Host headers write it."""
    host = host.strip("[]")
    return f"[{host}]" if ":" in host else host


def product_args(items) -> list:
    """``NAME=FILE`` or ``FILE`` arguments as ``(name, file)``."""
    out = []
    for item in items:
        name, sep, f = str(item).partition("=")
        if sep and name and not Path(item).exists():
            out.append((name, f))
        else:
            out.append((None, item))
    return out


def add_arguments(p) -> None:
    """``giant spine view``'s options."""
    p.add_argument(
        "products",
        nargs="+",
        metavar="PRODUCT",
        help="spine_summary_<key>.h5 files of one session, optionally as "
        "NAME=FILE; several are compared",
    )
    p.add_argument(
        "--marks",
        default=None,
        help="spine_marks.csv to read and write (created on the first "
        "mark; without it the page cannot mark)",
    )
    p.add_argument(
        "--crop-half",
        type=int,
        default=9,
        help="half-width of the per-trial crops, px (default 9)",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument(
        "--no-browser", action="store_true", help="do not open a browser tab"
    )


def serve(args) -> int:
    """Run ``giant spine view`` until interrupted."""
    store = SpineStore(
        product_args(args.products), args.marks, crop_half=args.crop_half
    )
    server = make_server(store, args.host, args.port)
    url = f"http://{_bracket(args.host)}:{server.server_address[1]}/"
    names = ", ".join(p.name for p in store.products)
    print(f"spine viewer: {names}; serving {url} (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
