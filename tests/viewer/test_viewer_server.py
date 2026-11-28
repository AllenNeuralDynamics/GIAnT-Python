"""API smoke test of the giant view server."""

import argparse
import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from giant_python.cli import build_parser
from giant_python.viewer.server import make_server, make_store

from .synthetic import H, N_TRIALS, W, write_session


@pytest.fixture
def server(tmp_path):
    """A running server on a synthetic session with two runs."""
    syn = write_session(tmp_path, "patchwise")
    args = build_parser().parse_args(
        [
            "view",
            f"net={syn['run']}",
            f"other={syn['run']}",
            "--moco-dir",
            str(syn["moco"]),
            "--baseline-dir",
            str(syn["base"]),
            "--port",
            "0",
            "--no-browser",
        ]
    )
    assert isinstance(args, argparse.Namespace) and args.command == "view"
    store = make_store(args)
    srv = make_server(store, "127.0.0.1", 0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", store
    srv.shutdown()
    srv.server_close()


def _get(url):
    """Body and headers of a GET."""
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read(), r.headers


def test_api(server):
    """Every endpoint answers with the expected shapes."""
    base, store = server
    body, _ = _get(base + "/")
    assert b"SILo run viewer" in body
    meta = json.loads(_get(base + "/api/meta")[0])
    assert [r["name"] for r in meta["runs"]] == ["net", "other"]
    assert meta["paths"][0]["alignment"] == ["patchwise", "patchwise"]
    assert len(meta["paths"][0]["trials"]) == N_TRIALS
    src = json.loads(_get(base + "/api/sources?path=1")[0])
    assert src["shape"] == [H, W] and len(src["runs"]["net"]["kept"]) == 3
    img, hd = _get(base + "/api/image?path=1&run=net&kind=trial_act&trial=1")
    assert json.loads(hd["X-Shape"]) == [H, W]
    assert np.frombuffer(img, np.float32).size == H * W
    s = json.loads(_get(base + "/api/source?path=1&run=net&id=1")[0])
    assert len(s["trial_rc"]) == N_TRIALS and s["events"]
    tb, hd = _get(base + "/api/source_traces?path=1&run=net&id=1")
    assert (
        np.frombuffer(tb, np.float32).size
        == 4 * store.runs[0].path(1).lines.size
    )
    st = json.loads(_get(base + "/api/trial?path=1&trial=0")[0])
    n_layers = len(st["layers"])
    assert n_layers == 1 + 2 * 2 + 1 + 2  # movie, 2 runs x 2, base, 2 res
    n = len(st["rows"])
    fr, hd = _get(base + "/api/frame?path=1&trial=0&t=40&avg=4")
    assert np.frombuffer(fr, np.float32).size == n_layers * n
    fm, _ = _get(base + "/api/frame?path=1&trial=0&t=40&avg=4&kernel=matched")
    assert np.frombuffer(fm, np.float32).size == n_layers * n
    fx, _ = _get(base + "/api/frame?path=1&trial=0&t=40&avg=4&kernel=exact")
    assert np.frombuffer(fx, np.float32).size == n_layers * n
    tc, hd = _get(base + "/api/trace?path=1&trial=0&r=20&c=30&rad=1")
    assert int(hd["X-Pixels"]) == 9
    assert np.frombuffer(tc, np.float32).size == n_layers * st["shape"][0]
    assert json.loads(_get(base + "/api/progress?key=1/0")[0]) == {"msg": ""}


def test_api_errors(server):
    """Unknown paths and bad keys are errors, not crashes."""
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(base + "/api/nothing")
    assert e.value.code == 404
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(base + "/api/source?path=1&run=nobody&id=0")
    assert e.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(base + "/api/frame?path=1&trial=0")
    assert e.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(base + "/api/frame?path=1&trial=0&t=3&kernel=other")
    assert e.value.code == 400
