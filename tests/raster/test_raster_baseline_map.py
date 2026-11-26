"""Tests of the mapping of the F0 network's baseline onto the extraction's
pixels, frames and units."""

import h5py
import numpy as np
import pytest

from giant_python.extraction.raster.baseline_map import (
    MappedBaseline,
    baseline_file,
    interp_frames,
    map_baseline,
    read_network_baseline,
    sample_bilinear,
)
from giant_python.extraction.raster.solver_baseline import initial_baseline
from giant_python.extraction.raster.solver_setup import SolverParams


def _plane(n_t, n_r, n_c, a=100.0, br=10.0, bc=5.0):
    """A stack that is linear in row and column, the same in every frame."""
    r, c = np.meshgrid(np.arange(n_r), np.arange(n_c), indexing="ij")
    return np.broadcast_to(a + br * r + bc * c, (n_t, n_r, n_c)).copy()


def test_baseline_file_names():
    """The file is named from the trial's stem or its movie's name."""
    assert baseline_file("d", "E1T2DMD1").name == "E1T2DMD1_BASELINE-80Hz.h5"
    name = "E1T2DMD1_REGISTERED_DOWNSAMPLED-80Hz.h5"
    assert baseline_file("d", name).name == "E1T2DMD1_BASELINE-80Hz.h5"


def test_read_network_baseline_orientation(tmp_path):
    """slap2-f0 writes (frame, MATLAB col, MATLAB row); reading gives
    (frame, row, col)."""
    stack = np.random.default_rng(0).random((4, 3, 5)).astype(np.float32)
    path = tmp_path / "E1T1DMD1_BASELINE-80Hz.h5"
    with h5py.File(path, "w") as f:
        f["baseline"] = stack.transpose(0, 2, 1).astype(np.float16)
        f.attrs["checkpoint"] = "x.pt"
    values, attrs = read_network_baseline(path)
    assert values.shape == (4, 3, 5)
    np.testing.assert_allclose(values, stack, rtol=1e-3)
    assert attrs["checkpoint"] == "x.pt"


def test_read_network_baseline_rejects_other_shapes(tmp_path):
    """A /baseline that is not three-dimensional is refused."""
    path = tmp_path / "b.h5"
    with h5py.File(path, "w") as f:
        f["baseline"] = np.zeros((4, 3))
    with pytest.raises(ValueError, match="frame, col, row"):
        read_network_baseline(path)


def test_bilinear_is_exact_on_a_plane():
    """A plane is sampled exactly at sub-pixel positions."""
    stack = _plane(3, 6, 7)
    rc = np.array([[0.0, 0.0], [2.5, 3.25], [4.9, 5.1], [5.0, 6.0]])
    values, weight = sample_bilinear(stack, rc)
    want = 100 + 10 * rc[:, 0] + 5 * rc[:, 1]
    np.testing.assert_allclose(values, np.repeat(want[:, None], 3, 1))
    np.testing.assert_allclose(weight, 1.0)


def test_bilinear_renormalizes_over_neighbours_with_values():
    """Neighbours without a value are left out; too little weight left
    gives NaN."""
    stack = np.full((2, 4, 4), 7.0)
    stack[:, 1, 2] = np.nan
    # (1.2, 1.2): the NaN corner (1, 2) carries 0.8 * 0.2 of the weight
    values, weight = sample_bilinear(stack, np.array([[1.2, 1.2]]))
    np.testing.assert_allclose(values, 7.0)
    np.testing.assert_allclose(weight, 1 - 0.16)
    # (1.1, 1.9): it carries 0.9 * 0.9 = 0.81
    values, weight = sample_bilinear(stack, np.array([[1.1, 1.9]]))
    assert np.isnan(values).all()
    np.testing.assert_allclose(weight, 0.19)
    values, _ = sample_bilinear(stack, np.array([[1.1, 1.9]]), 0.1)
    np.testing.assert_allclose(values, 7.0)


def test_bilinear_outside_the_frame():
    """Neighbours outside the stack have no value; a whole-pixel position
    on the last row needs no neighbour beyond it."""
    stack = _plane(1, 3, 3)
    values, weight = sample_bilinear(
        stack, np.array([[2.0, 1.0], [2.4, 1.0], [-1.0, -1.0]])
    )
    np.testing.assert_allclose(values[0], 100 + 20 + 5)
    np.testing.assert_allclose(weight[0], 1.0)
    np.testing.assert_allclose(values[1], 125)
    np.testing.assert_allclose(weight[1], 0.6)
    assert np.isnan(values[2]).all()
    assert weight[2] == 0


def test_bilinear_per_frame_nans():
    """A neighbour missing in one frame only is left out of that frame."""
    stack = _plane(3, 3, 3)
    stack[1, 1, 1] = np.nan
    values, weight = sample_bilinear(stack, np.array([[1.75, 1.0]]))
    np.testing.assert_allclose(values[0, [0, 2]], 100 + 17.5 + 5)
    np.testing.assert_allclose(values[0, 1], 100 + 20 + 5)
    np.testing.assert_allclose(weight, 0.75)
    with pytest.raises(ValueError):
        sample_bilinear(stack, np.array([[np.nan, 1.0]]))
    with pytest.raises(ValueError):
        sample_bilinear(stack, np.array([[1.0, 1.0]]), 0.0)


def test_interp_frames_linear_in_lines():
    """Linear in line number, NaN samples left out, ends held."""
    src = np.array([10.0, 30.0, 50.0, 70.0])
    vals = np.array([[1.0, 3.0, 5.0, 7.0], [1.0, np.nan, 5.0, 7.0]])
    dst = np.array([0.0, 10.0, 20.0, 45.0, 70.0, 90.0])
    out = interp_frames(vals, src, dst)
    np.testing.assert_allclose(out[0], [1, 1, 2, 4.5, 7, 7])
    np.testing.assert_allclose(out[1], [1, 1, 2, 4.5, 7, 7])
    nan_row = interp_frames(np.full((1, 4), np.nan), src, dst)
    assert np.isnan(nan_row).all()
    one = interp_frames(np.array([[4.0]]), np.array([5.0]), dst)
    np.testing.assert_allclose(one, 4.0)
    with pytest.raises(ValueError):
        interp_frames(vals, src[::-1], dst)
    with pytest.raises(ValueError):
        interp_frames(vals[:, :3], src, dst)


def _registration(n_r=12, n_c=14, lines_80=100, lines_200=40, n_lines=8000):
    """A synthetic network baseline: a plane times a slow time course.

    Returns the stack (counts), the registered frames' lines, the
    extraction frames' lines and the true value as a function of
    (row, col, line).
    """
    ds = np.arange(1, n_lines, lines_80, dtype=np.float64)
    fl = np.arange(1, n_lines, lines_200, dtype=np.float64)

    def truth(r, c, line):
        """The baseline in counts."""
        return (200 + 8 * r + 3 * c) * (1 + 0.1 * np.sin(line / 2000.0))

    r, c = np.meshgrid(np.arange(n_r), np.arange(n_c), indexing="ij")
    stack = truth(r[None], c[None], ds[:, None, None]).astype(np.float32)
    return stack, ds, fl, truth


def test_map_baseline_space_time_units():
    """Space, time and units of the mapping, against the known field."""
    stack, ds, fl, truth = _registration()
    rc = np.array([[3.4, 5.7], [0.0, 0.0], [11.0, 13.0], [6.5, 2.25]])
    mapped = map_baseline(stack, rc, ds, fl, photon_scale=25.0)
    assert isinstance(mapped, MappedBaseline)
    assert mapped.values.shape == (4, fl.size)
    assert not mapped.missing.any()
    want = truth(rc[:, :1], rc[:, 1:], fl[None, :]) / 25.0
    inner = fl <= ds[-1]
    # linear interpolation of a slow sine: error below 1e-4 relative
    np.testing.assert_allclose(mapped.values[:, inner], want[:, inner], 2e-4)
    # after the last registered frame: its value
    last = truth(rc[:, 0], rc[:, 1], ds[-1]) / 25.0
    tail = np.broadcast_to(last[:, None], mapped.values[:, ~inner].shape)
    np.testing.assert_allclose(mapped.values[:, ~inner], tail, 1e-6)


def test_map_baseline_missing_pixels_and_fallback():
    """Pixels without a network value are flagged; with the data given
    they get its starting baseline."""
    stack, ds, fl, _ = _registration()
    stack[:, :, :4] = np.nan
    rc = np.array([[5.0, 1.0], [5.0, 3.6], [5.0, 3.4], [5.0, 8.0]])
    mapped = map_baseline(stack, rc, ds, fl, 25.0)
    # (5, 3.6) has 0.6 of its weight on column 4, (5, 3.4) only 0.4
    np.testing.assert_array_equal(mapped.missing, [True, False, True, False])
    assert np.isnan(mapped.values[mapped.missing]).all()
    rng = np.random.default_rng(1)
    data = 8 + rng.normal(0, 0.1, (4, fl.size))
    params = SolverParams()
    filled = map_baseline(stack, rc, ds, fl, 25.0, fallback_data=data)
    np.testing.assert_array_equal(filled.missing, mapped.missing)
    miss = mapped.missing
    np.testing.assert_allclose(
        filled.values[miss], initial_baseline(data[miss], params)
    )
    np.testing.assert_allclose(filled.values[~miss], mapped.values[~miss])
    with pytest.raises(ValueError, match="fallback_data"):
        map_baseline(stack, rc, ds, fl, 25.0, fallback_data=data[:, :5])


def test_map_baseline_dropped_frames_are_interpolated_over():
    """Dropped registered frames do not enter the time interpolation."""
    stack, ds, fl, truth = _registration()
    bad = stack.copy()
    drop = np.zeros(ds.size, bool)
    drop[20:25] = True
    bad[drop] = 1e6
    rc = np.array([[4.0, 4.0]])
    mapped = map_baseline(bad, rc, ds, fl, 1.0, drop_frames=drop)
    want = truth(4.0, 4.0, fl)
    np.testing.assert_allclose(mapped.values[0], want, rtol=5e-3)
    kept = map_baseline(stack, rc, ds, fl, 1.0)
    none = map_baseline(
        stack, rc, ds, fl, 1.0, drop_frames=np.zeros_like(drop)
    )
    np.testing.assert_array_equal(kept.values, none.values)
    with pytest.raises(ValueError, match="drop_frames"):
        map_baseline(bad, rc, ds, fl, 1.0, drop_frames=drop[:-1])
    with pytest.raises(ValueError, match="every registered frame"):
        map_baseline(bad, rc, ds, fl, 1.0, drop_frames=np.ones_like(drop))


def test_map_baseline_checks_inputs():
    """Mismatched frames and a bad photon scale are refused."""
    stack, ds, fl, _ = _registration()
    rc = np.zeros((1, 2))
    with pytest.raises(ValueError, match="registration has"):
        map_baseline(stack, rc, ds[:-1], fl, 25.0)
    for scale in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError, match="photon_scale"):
            map_baseline(stack, rc, ds, fl, scale)
