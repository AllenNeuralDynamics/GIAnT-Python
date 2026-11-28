"""Tests of the spine viewer's data, API and HTML export
(:mod:`giant_python.spine.view`), on a synthetic session."""

import base64
import gzip
import json
import re
import threading
import urllib.request

import numpy as np
import pytest

from giant_python.cli import main
from giant_python.spine.pipeline import make_spine_summary
from giant_python.spine.product import read_product
from giant_python.spine.rois import read_marks
from giant_python.spine.view.data import (
    SpineStore,
    blur_nan,
    late_trials,
    robust_z,
)
from giant_python.spine.view.export import export_html, field_key, pack
from giant_python.spine.view.server import make_server

from . import synthetic as syn


@pytest.fixture
def product(tmp_path):
    """A synthetic session's spine summary (with network images)."""
    s = syn.make_session(tmp_path)
    rep = make_spine_summary(
        s["run"],
        s["table"],
        tmp_path / "out",
        baseline_dir=s["baselines"],
        moco_dir=s["moco"],
        trim_s=0.1,
        n_perm=100,
    )
    return rep["output"]


def test_blur_nan_ignores_nan():
    """NaN stays NaN and does not pull its neighbours down."""
    img = np.ones((9, 9), np.float32)
    img[4, 4] = np.nan
    out = blur_nan(img)
    assert np.isnan(out[4, 4])
    np.testing.assert_allclose(out[np.isfinite(out)], 1.0, rtol=1e-6)


def test_robust_z_and_late_trials():
    """Median and MAD z-scores; late trials are the last block, or the
    second half of one block."""
    z = robust_z(np.array([[0.0, 1.0, 2.0, np.nan]]))
    assert z[0, 1] == 0 and np.isnan(z[0, 3])
    np.testing.assert_array_equal(
        late_trials([0, 0, 1, 1, 1]), [0, 0, 1, 1, 1]
    )
    np.testing.assert_array_equal(late_trials([0, 0, 0, 0]), [0, 0, 1, 1])


def test_normalized_field_and_change(product):
    """Each trial is divided by its median over the common pixels; the
    change view compares blurred trials, with a brightness in [0, 1]."""
    store = SpineStore([(None, product)])
    pr = store.products[0]
    stack = read_product(product)["paths"][1]["images"]["mean_ch1"]
    nz = pr.normalized(1, "mean_ch1")
    common = np.isfinite(stack).all(axis=0)
    for t in range(stack.shape[0]):
        med = np.median(stack[t][common])
        np.testing.assert_allclose(nz["stack"][t], stack[t] / med, rtol=1e-6)
    np.testing.assert_allclose(
        pr.field(1, "mean_ch1", -1, "image", "")[0],
        np.nanmean(nz["stack"], axis=0),
        rtol=1e-6,
    )
    ch = pr.field(1, "mean_ch1", 2, "change", "average")
    assert ch.shape == (2,) + stack.shape[1:]
    b = ch[1]
    assert np.nanmin(b) >= 0 and np.nanmax(b) <= 1
    cp = pr.change_prep(1, "mean_ch1")
    expect = cp["blurred"][2] / cp["mean"] - 1
    np.testing.assert_allclose(ch[0], expect, rtol=1e-5, equal_nan=True)
    prev = pr.field(1, "mean_ch1", 2, "change", "previous")[0]
    np.testing.assert_allclose(
        prev,
        cp["blurred"][2] / cp["blurred"][1] - 1,
        rtol=1e-5,
        equal_nan=True,
    )
    # the first trial of a block has no previous trial
    assert np.isnan(pr.field(1, "mean_ch1", 0, "change", "previous")[0]).all()
    act = pr.field(1, "", -1, "activity", "")
    assert act.shape == (1,) + stack.shape[1:]


def test_store_sources_roi_and_crops(product):
    """Sources carry sizes and trends; a window and its crops sit on the
    same grid around the source."""
    store = SpineStore([("syn", product)], crop_half=4)
    meta = store.meta()
    p1 = meta["products"][0]["paths"][1]
    assert meta["products"][0]["name"] == "syn"
    assert "f0_net" in p1["trends"] and p1["n_sources"] == len(syn.SOURCES)
    src = store.sources(0, 1)
    prod = read_product(product)["paths"][1]
    np.testing.assert_allclose(
        np.array(src["normalized"]["f0_net"], float),
        prod["normalized"]["sources"]["f0_net"],
    )
    assert src["trends"]["f0_net"]["attrs"]["n_perm"] == 100
    roi = store.roi(0, 1, 0)
    assert roi["size"] == 9 and np.array(roi["weights"]).shape == (9, 9)
    r0, c0 = roi["origin"]
    w = prod["rois"]["sources"].image(0)
    np.testing.assert_allclose(
        np.array(roi["weights"]), w[r0 : r0 + 9, c0 : c0 + 9], atol=1e-6
    )
    crops = store.crops(0, 1, 0, "mean_ch1", "image", "average")
    n_t = len(prod["trials"]["acquisition"])
    assert crops.shape == (1, n_t + 1, 9, 9)
    full = store.products[0].normalized(1, "mean_ch1")["stack"]
    np.testing.assert_allclose(
        crops[0, 1:], full[:, r0 : r0 + 9, c0 : c0 + 9], rtol=1e-6
    )
    ch = store.crops(0, 1, 0, "mean_ch1", "change", "average")
    assert ch.shape == (2, n_t + 1, 9, 9)


def test_marks_attach_by_position(product, tmp_path):
    """A mark is stored with its source's position and attaches to the
    nearest source within 2.5 px; clearing removes it."""
    marks = tmp_path / "marks.csv"
    store = SpineStore([(None, product)], marks)
    got = store.set_mark(0, 1, 1, "Growing", "a note")
    assert got["by_source"][1]["label"] == "Growing"
    rows = read_marks(marks)
    assert rows[0]["Source"] == "2" and rows[0]["Note"] == "a note"
    rs = read_product(product)["paths"][1]["rois"]["sources"]
    assert rows[0]["Row0"] == pytest.approx(rs.rows[1])
    # a second mark on the same source replaces the first
    store.set_mark(0, 1, 1, "Shrinking", "")
    assert len(read_marks(marks)) == 1
    assert store.marks_for(0, 1)["by_source"][1]["label"] == "Shrinking"
    # a mark far from every source does not attach
    rows = read_marks(marks) + [
        dict(rows[0], Row0=0.0, Col0=0.0, Label="Unsure")
    ]
    from giant_python.spine.rois import write_marks

    write_marks(marks, rows)
    assert store.marks_for(0, 1)["unattached"] == 1
    store.set_mark(0, 1, 1, "", "")
    assert 1 not in store.marks_for(0, 1)["by_source"]
    with pytest.raises(ValueError, match="unknown mark"):
        store.set_mark(0, 1, 1, "Big", "")


def test_product_is_read_again_when_it_changes(product, tmp_path):
    """A rewritten product shows on the next request."""
    import os
    import shutil

    copy = tmp_path / "copy.h5"
    shutil.copy(product, copy)
    store = SpineStore([(None, copy)])
    first = store.sources(0, 1)["normalized"]["f0_net"]
    import h5py

    with h5py.File(copy, "r+") as f:
        d = f["Path1/normalized/sources/f0_net"]
        d[...] = d[()] + 1
    st = os.stat(copy)
    os.utime(copy, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    again = store.sources(0, 1)["normalized"]["f0_net"]
    a = np.array(first, float)
    np.testing.assert_allclose(np.array(again, float), a + 1)


def _get(url):
    """GET and decode JSON or float32."""
    with urllib.request.urlopen(url) as r:
        body = r.read()
        if r.headers.get("Content-Type") == "application/json":
            return json.loads(body)
        shape = json.loads(r.headers["X-Shape"])
        return np.frombuffer(body, np.float32).reshape(shape)


def test_server_answers_the_api(product, tmp_path):
    """The page and every API route answer; a mark is written by POST."""
    marks = tmp_path / "marks.csv"
    store = SpineStore([(None, product)], marks)
    srv = make_server(store, port=0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/") as r:
            assert b"Spine size viewer" in r.read()
        meta = _get(base + "/api/meta")
        assert meta["products"][0]["paths"]["1"]["n_sources"] == 3
        src = _get(base + "/api/sources?prod=0&path=1")
        assert src["n"] == 3
        f = _get(base + "/api/field?prod=0&path=1&image=f0_net&trial=1")
        assert f.shape == (1, syn.H, syn.W)
        c = _get(
            base + "/api/crops?prod=0&path=1&id=0&image=f0_net&view=change"
        )
        assert c.shape[0] == 2
        roi = _get(base + "/api/roi?prod=0&path=1&id=2")
        assert roi["id"] == 2
        req = urllib.request.Request(
            base + "/api/mark",
            data=json.dumps(
                {"prod": 0, "path": 1, "id": 0, "label": "Unsure"}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as r:
            assert json.loads(r.read())["by_source"]["0"]["label"] == "Unsure"
        assert read_marks(marks)[0]["Label"] == "Unsure"
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(base + "/api/field?prod=0&path=7&image=x")
        assert e.value.code == 404
    finally:
        srv.shutdown()
        srv.server_close()


def _static(page: str) -> dict:
    """The data embedded in an exported page."""
    m = re.search(
        r'<script id="spine-static" type="application/octet-stream">'
        r"([^<]*)</script>",
        page,
    )
    return json.loads(gzip.decompress(base64.b64decode(m.group(1))))


def _unpack(e) -> np.ndarray:
    """An embedded array (as the page's ``unpack``)."""
    parts = []
    for layer in e["layers"]:
        q = np.frombuffer(base64.b64decode(layer["b64"]), np.uint8)
        v = layer["offset"] + layer["scale"] * q.astype(np.float64)
        parts.append(np.where(q == 255, np.nan, v))
    return np.concatenate(parts).reshape(e["shape"])


def test_pack_round_trips_within_a_code():
    """uint8 codes keep values within half a step and NaN as NaN."""
    rng = np.random.default_rng(0)
    a = rng.normal(1, 0.1, (1, 20, 30)).astype(np.float32)
    a[0, 3, 4] = np.nan
    e = pack(a, "image")
    back = _unpack(e)
    assert np.isnan(back[0, 3, 4])
    step = e["layers"][0]["scale"]
    inside = np.isfinite(a) & (a >= e["layers"][0]["offset"])
    inside &= a <= e["layers"][0]["offset"] + 254 * step
    np.testing.assert_allclose(back[inside], a[inside], atol=step / 2 + 1e-7)
    ch = np.stack([np.full((4, 4), 0.1), np.full((4, 4), 0.5)])
    back = _unpack(pack(ch, "change"))
    np.testing.assert_allclose(back, ch, atol=0.003)


def test_export_html_embeds_the_viewer_data(product, tmp_path):
    """The page holds the meta, the sources, the mean fields and the crops
    of the chosen sources, and nothing that needs the server."""
    out = tmp_path / "page.html"
    rep = export_html(product, out, sources=[2], name="syn")
    page = out.read_text(encoding="utf-8")
    assert "<title>Spine sizes syn</title>" in page
    data = _static(page)
    assert data["meta"]["marks_file"] is None
    assert data["sources"]["1"]["n"] == 3
    key = field_key(1, "f0_net", -1, "image", "")
    assert key in data["fields"]
    assert field_key(1, "f0_net", -1, "change", "") in data["fields"]
    assert rep["sources"][1] == [2]
    assert "1|1" in data["rois"] and "1|0" not in data["rois"]
    crop_keys = [k for k in data["crops"] if k.startswith("1|1|")]
    assert crop_keys and all("f0_" in k for k in crop_keys)
    store = SpineStore([(None, product)])
    mean = store.products[0].field(1, "f0_net", -1, "image", "")
    e = data["fields"][key]
    back = _unpack(e)
    fin = np.isfinite(mean)
    step = e["layers"][0]["scale"]
    lo = e["layers"][0]["offset"]
    inside = fin & (mean >= lo) & (mean <= lo + 254 * step)
    np.testing.assert_allclose(back[inside], mean[inside], atol=step)


def test_cli_export(product, tmp_path, capsys):
    """giant spine export-html writes the page and reports it."""
    out = tmp_path / "e.html"
    assert (
        main(
            ["spine", "export-html", product, "--out", str(out), "--top", "1"]
        )
        == 0
    )
    rep = json.loads(capsys.readouterr().out)
    assert rep["output"] == str(out) and out.is_file()


def test_pick_sources_marked_first_then_by_q_and_size():
    """The export's crops go to the marked sources, then by q, with ties at
    the permutation floor going to the larger statistic."""
    from giant_python.spine.view.export import pick_sources

    nan = float("nan")
    tr = {
        "step_q": [0.5, 1e-4, 1e-4, nan, 0.02],
        "step_t": [1.0, 3.0, 6.0, nan, 2.5],
        "brief_q": [0.9, 0.9, 0.9, nan, 0.9],
        "brief_t": [1.0, 1.0, 1.0, nan, 1.0],
        "resid_q": [0.04, 0.9, 0.9, nan, 0.9],
        "resid_t": [2.0, 1.0, 1.0, nan, 1.0],
    }
    src = {"n": 5, "trends": {"f0": tr}, "marks": {"by_source": {"0": {}}}}
    assert pick_sources(src, "f0", 3) == [0, 2, 1, 4]
    assert pick_sources(src, "f0", 0) == [0]


def test_marks_need_json_from_this_server(product, tmp_path):
    """The mark endpoint refuses bodies that are not JSON (a cross-origin
    page could send those without a preflight) and requests whose Origin
    is another site."""
    marks = tmp_path / "marks.csv"
    store = SpineStore([(None, product)], marks)
    srv = make_server(store, port=0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    body = json.dumps({"path": 1, "id": 0, "label": "Unsure"}).encode()
    try:
        for headers, code in (
            ({"Content-Type": "text/plain"}, 415),
            (
                {
                    "Content-Type": "application/json",
                    "Origin": "http://example.com",
                },
                403,
            ),
        ):
            req = urllib.request.Request(
                base + "/api/mark", data=body, headers=headers
            )
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(req)
            assert e.value.code == code
        assert not marks.exists()
        req = urllib.request.Request(
            base + "/api/mark",
            data=body,
            headers={"Content-Type": "application/json", "Origin": base},
        )
        with urllib.request.urlopen(req) as r:
            assert r.status == 200
        assert read_marks(marks)[0]["Label"] == "Unsure"
    finally:
        srv.shutdown()
        srv.server_close()


def test_sources_without_sources_keep_the_fields(product, tmp_path):
    """A path without sources answers with the same, empty, fields."""
    store = SpineStore([(None, product)])
    pr = store.products[0]
    pr.small()["paths"][1]["rois"].pop("sources")
    src = store.sources(0, 1)
    assert src["n"] == 0 and src["trends"] == {} and src["ids"] == []
    assert src["marks"] == {"by_source": {}, "unattached": 0}


def test_cache_keys_follow_the_file(product, tmp_path):
    """A result made from the old file is not served after the file
    changes, even if it lands in the cache after the reload."""
    import os
    import shutil

    copy = tmp_path / "copy.h5"
    shutil.copy(product, copy)
    store = SpineStore([(None, copy)])
    pr = store.products[0]
    old = pr.normalized(1, "mean_ch1")
    gen = pr._gen
    st = os.stat(copy)
    os.utime(copy, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    pr.small()
    assert pr._gen == gen + 1
    # a late insert under the old generation's key is never read
    pr._cache.get((gen, "norm", 1, "mean_ch1"), lambda: "stale")
    new = pr.normalized(1, "mean_ch1")
    assert new is not old and not isinstance(new, str)


def test_server_refuses_untrusted_hosts(product, tmp_path):
    """Requests naming another host (DNS rebinding) get nothing, and marks
    from an origin that is not this server's trusted names are refused."""
    store = SpineStore([(None, product)], tmp_path / "marks.csv")
    srv = make_server(store, port=0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    port = srv.server_address[1]
    base = f"http://127.0.0.1:{port}"
    try:
        req = urllib.request.Request(
            base + "/api/meta", headers={"Host": f"evil.example:{port}"}
        )
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req)
        assert e.value.code == 403
        req = urllib.request.Request(
            base + "/api/meta", headers={"Host": f"localhost:{port}"}
        )
        with urllib.request.urlopen(req) as r:
            assert r.status == 200
        body = json.dumps({"path": 1, "id": 0, "label": "Unsure"}).encode()
        for origin in (
            f"http://evil.example:{port}",
            f"https://127.0.0.1:{port}",
        ):
            req = urllib.request.Request(
                base + "/api/mark",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Origin": origin,
                    "Host": f"127.0.0.1:{port}",
                },
            )
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(req)
            assert e.value.code == 403
    finally:
        srv.shutdown()
        srv.server_close()


def test_export_refuses_unknown_paths(product, tmp_path):
    """Asking for a path the summary does not have is an error, not an
    empty page."""
    with pytest.raises(ValueError, match="no path"):
        export_html(product, tmp_path / "p.html", paths=[7])
    assert not (tmp_path / "p.html").exists()


def test_pick_sources_uses_the_strongest_test_at_the_smallest_q():
    """Among tests tied at a source's smallest q, the largest statistic
    breaks ties between sources."""
    from giant_python.spine.view.export import pick_sources

    tr = {
        "step_q": [1e-3, 1e-3],
        "step_t": [2.0, 6.0],
        "brief_q": [1e-3, 0.5],
        "brief_t": [20.0, 1.0],
        "resid_q": [0.9, 0.9],
        "resid_t": [1.0, 1.0],
    }
    src = {"n": 2, "trends": {"f0": tr}, "marks": {}}
    assert pick_sources(src, "f0", 1) == [0]


def test_marks_keep_their_order_within_a_path(product, tmp_path):
    """Two marks with the same time that land on one source: the one
    written later wins, whatever their rows."""
    from giant_python.spine.rois import write_marks

    store = SpineStore([(None, product)], tmp_path / "marks.csv")
    rs = read_product(product)["paths"][1]["rois"]["sources"]
    r, c = float(rs.rows[0]), float(rs.cols[0])
    base = {
        "Path": 1,
        "Source": 1,
        "Note": "",
        "Col0": c,
        "Updated": "2026-01-01 00:00:00",
    }
    write_marks(
        tmp_path / "marks.csv",
        [
            dict(base, Label="Growing", Row0=r + 0.5),
            dict(base, Label="Shrinking", Row0=r - 0.5),
        ],
    )
    assert store.marks_for(0, 1)["by_source"][0]["label"] == "Shrinking"
    # a write through the store keeps the order within the path
    store.set_mark(0, 1, 1, "Unsure", "")
    rows = read_marks(tmp_path / "marks.csv")
    assert [m["Label"] for m in rows] == ["Growing", "Shrinking", "Unsure"]
    assert store.marks_for(0, 1)["by_source"][0]["label"] == "Shrinking"


def test_pick_sources_keeps_infinite_statistics_first():
    """An infinite statistic (a change without noise) ranks first among
    equal q-values."""
    from giant_python.spine.view.export import pick_sources

    inf = float("inf")
    tr = {
        f"{t}_{f}": [None, None]
        for t in ("brief", "resid")
        for f in ("q", "t")
    }
    tr.update({"step_q": [1e-4, 1e-4], "step_t": [6.0, inf]})
    src = {"n": 2, "trends": {"f0": tr}, "marks": {}}
    assert pick_sources(src, "f0", 1) == [1]


def test_summary_without_rois_loads(product, tmp_path):
    """A summary with no sources and no user ROIs (no rois group) loads."""
    import shutil

    import h5py

    copy = tmp_path / "copy.h5"
    shutil.copy(product, copy)
    with h5py.File(copy, "r+") as f:
        for n in ("Path1", "Path2"):
            if f"{n}/rois" in f:
                del f[f"{n}/rois"]
    store = SpineStore([(None, copy)])
    assert store.sources(0, 1)["n"] == 0
    assert store.meta()["products"][0]["paths"][1]["n_sources"] == 0


def test_server_on_ipv6_loopback_and_port_80_names(product):
    """The server binds an IPv6 literal, and at port 80 trusts the names
    without a port, as browsers send them."""
    import socket as so

    from giant_python.spine.view.server import _bracket

    from giant_python.spine.view.server import allowed_hosts

    assert _bracket("::1") == "[::1]" and _bracket("localhost") == "localhost"
    at80 = allowed_hosts("127.0.0.1", 80)
    assert "127.0.0.1" in at80 and "localhost:80" in at80
    assert "127.0.0.1" not in allowed_hosts("127.0.0.1", 8790)
    assert allowed_hosts("0.0.0.0", 8790) == frozenset()
    if not so.has_ipv6:
        pytest.skip("no IPv6")
    store = SpineStore([(None, product)])
    try:
        srv = make_server(store, host="::1", port=0)
    except OSError:
        pytest.skip("no IPv6 loopback")
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    port = srv.server_address[1]
    try:
        assert f"[::1]:{port}" in srv.RequestHandlerClass.allowed
        with urllib.request.urlopen(f"http://[::1]:{port}/api/meta") as r:
            assert r.status == 200
    finally:
        srv.shutdown()
        srv.server_close()


def test_a_note_alone_keeps_the_label(product, tmp_path):
    """A note saved on its own keeps the label the source has at that
    moment (Unsure without one)."""
    store = SpineStore([(None, product)], tmp_path / "marks.csv")
    got = store.set_mark(0, 1, 0, None, "first note")
    assert got["by_source"][0]["label"] == "Unsure"
    store.set_mark(0, 1, 0, "Growing", "first note")
    got = store.set_mark(0, 1, 0, None, "second note")
    assert got["by_source"][0] == {
        "label": "Growing",
        "note": "second note",
        "updated": got["by_source"][0]["updated"],
    }
    assert len(read_marks(tmp_path / "marks.csv")) == 1
