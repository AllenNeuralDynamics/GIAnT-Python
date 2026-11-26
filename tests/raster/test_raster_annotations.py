"""Tests of reading SILo's annotations.h5 and moving ROIs to the consensus
frame (loadAnnotationsH5, roisToConsensusFrame, shiftRoiData)."""

import h5py
import numpy as np
import pytest

from giant_python.extraction.raster.alignment import TrialWarp
from giant_python.extraction.raster.annotations import (
    PathAnnotations,
    UserRoi,
    annotation_trial,
    read_annotations,
    rois_to_consensus,
    session_annotations,
    soma_mask,
)

SHAPE = (12, 16)  # MATLAB rows x cols of the annotation image


def _text(f, path, value):
    """A string as MATLAB's saveAnnotationsH5 writes it (uint16 codes)."""
    f.create_dataset(
        path, data=np.array([[ord(c)] for c in value], dtype=np.uint16)
    )


def _matlab(f, path, value):
    """A numeric array as MATLAB's h5write stores it (dimensions
    reversed)."""
    f.create_dataset(path, data=np.ascontiguousarray(np.asarray(value).T))


def write_annotations(path, zero_indexed=True):
    """A file with two paths in MATLAB's layout; returns the masks."""
    masks = []
    with h5py.File(path, "w") as f:
        _matlab(f, "row_major", np.uint8([[0]]))
        if zero_indexed is not None:
            _matlab(f, "coords_zero_indexed", np.uint8([[int(zero_indexed)]]))
            _matlab(f, "schema_version", np.uint32([[2]]))
        for d, (label, fn) in enumerate(
            [("Soma 1", "E1T2DMD1_REGISTERED_DOWNSAMPLED-80Hz"), ("spine", "")]
        ):
            root = f"Path{d + 1}"
            _text(f, f"{root}/dr", "C:\\data\\moco")
            if fn:
                _text(f, f"{root}/fn", fn)
            _matlab(f, f"{root}/n_rois", np.uint32([[2 if d == 0 else 0]]))
            if d:
                continue
            for k, (lab, r0, c0) in enumerate([(label, 2, 3), ("sp2", 6, 9)]):
                mask = np.zeros(SHAPE, np.uint8)
                mask[r0 : r0 + 3, c0 : c0 + 2] = 1
                masks.append(mask.astype(bool))
                roi = f"{root}/roi_{k + 1:03d}"
                _text(f, f"{roi}/type", "circle")
                _text(f, f"{roi}/label", lab)
                _matlab(f, f"{roi}/mask", mask)
                # zero-based [row col] in the file when zero-indexed
                center = (
                    np.array([[r0 + 1.0, c0 + 0.5]])
                    if zero_indexed
                    else np.array([[c0 + 1.5, r0 + 2.0]])
                )
                _matlab(f, f"{roi}/center", center)
                _matlab(f, f"{roi}/radius", np.array([[1.5]]))
    return masks


@pytest.mark.parametrize("zero_indexed", [True, False])
def test_read_annotations(tmp_path, zero_indexed):
    """Strings, masks (MATLAB orientation) and geometry are read back."""
    path = tmp_path / "annotations.h5"
    masks = write_annotations(path, zero_indexed)
    paths = read_annotations(path)
    assert len(paths) == 2
    a = paths[0]
    assert a.fn == "E1T2DMD1_REGISTERED_DOWNSAMPLED-80Hz"
    assert a.dr == "C:\\data\\moco"
    assert [r.label for r in a.rois] == ["Soma 1", "sp2"]
    assert [r.type for r in a.rois] == ["circle", "circle"]
    np.testing.assert_array_equal(a.rois[0].mask, masks[0])
    # images.roi [x, y], one-based
    np.testing.assert_allclose(a.rois[0].center, [4.5, 4.0])
    assert a.rois[0].radius == 1.5
    assert a.rois[0].is_soma and not a.rois[1].is_soma
    assert paths[1].fn == "" and paths[1].rois == []
    assert [a.path for a in paths] == [1, 2]


def test_annotation_trial():
    """The annotation's file is matched with or without its extension."""
    fns = ["E1T1DMD1_REGISTERED_DOWNSAMPLED-80Hz.tif", "E1T2DMD1_X.h5"]
    assert annotation_trial("E1T1DMD1_REGISTERED_DOWNSAMPLED-80Hz", fns) == 0
    assert annotation_trial("E1T2DMD1_X.tif", fns) == 1
    assert annotation_trial("C:\\moco\\E1T2DMD1_X", fns) == 1
    assert annotation_trial("other", fns) is None
    assert annotation_trial("", fns) is None


def test_rois_to_consensus_shift(tmp_path):
    """Masks move by minus the trial's rounded shift, geometry exactly; a
    trial that is not known or not aligned leaves them in place."""
    path = tmp_path / "annotations.h5"
    masks = write_annotations(path)
    ann = read_annotations(path)[0]
    fns = ["t1.tif", "E1T2DMD1_REGISTERED_DOWNSAMPLED-80Hz.tif"]
    warp = TrialWarp.from_shifts(np.array([[0.0, 0.0], [1.4, -2.0]]))
    shape = (14, 15)
    moved = rois_to_consensus(ann, fns, warp, shape)
    want = np.zeros(shape, bool)
    r, c = np.nonzero(masks[0])
    keep = (r - 1 >= 0) & (c + 2 < shape[1])
    want[r[keep] - 1, c[keep] + 2] = True
    np.testing.assert_array_equal(moved[0].mask, want)
    np.testing.assert_allclose(moved[0].center, ann.rois[0].center + [2, -1.4])
    assert ann.rois[0].mask.shape == SHAPE  # the input is unchanged
    unaligned = TrialWarp.from_shifts(np.array([[0, 0], [np.nan, np.nan]]))
    with pytest.warns(UserWarning, match="not aligned"):
        same = rois_to_consensus(ann, fns, unaligned, shape)
    np.testing.assert_array_equal(same[0].mask[:12, :15], masks[0][:, :15])
    with pytest.warns(UserWarning, match="not in the trial table"):
        rois_to_consensus(ann, ["a.tif", "b.tif"], warp, shape)


def test_soma_mask_and_session(tmp_path):
    """Soma masks are the union of the ROIs labelled SOMA; a session
    without a file has no annotations."""
    rois = [
        UserRoi("soma", "polygon", np.eye(4, dtype=bool)),
        UserRoi("Spine", "circle", np.ones((4, 4), bool)),
        UserRoi("SOMA2", "circle", np.zeros((0, 0), bool)),
    ]
    np.testing.assert_array_equal(soma_mask(rois, (3, 5))[:, :3], np.eye(3))
    assert session_annotations(None, 2) == [None, None]
    assert session_annotations(tmp_path / "missing.h5", 1) == [None]
    path = tmp_path / "annotations.h5"
    write_annotations(path)
    got = session_annotations(path, 3)
    assert isinstance(got[0], PathAnnotations) and got[2] is None
    # paths are placed by their numbers, not by their order in the file
    with h5py.File(path, "a") as f:
        f.move("Path1", "Path3")
    got = session_annotations(path, 3)
    assert got[0] is None and got[1].path == 2 and got[2].path == 3


def test_text_is_utf16(tmp_path):
    """MATLAB stores text as UTF-16 code units, surrogate pairs included."""
    label = "soma é😀"
    codes = np.frombuffer(label.encode("utf-16-le"), dtype="<u2")
    with h5py.File(tmp_path / "a.h5", "w") as f:
        f.create_dataset("Path1/n_rois", data=np.uint32([[1]]))
        f.create_dataset("Path1/roi_001/label", data=codes[:, None])
        f.create_dataset("Path1/roi_001/mask", data=np.ones((2, 2), np.uint8))
    roi = read_annotations(tmp_path / "a.h5")[0].rois[0]
    assert roi.label == label
