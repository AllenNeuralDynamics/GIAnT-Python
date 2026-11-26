"""Tests of the subproblem solver's parameters, filters and inputs."""

import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy import signal

from giant_python.extraction.raster.solver_setup import (
    SolverParams,
    Subproblem,
    conv_operator,
    initial_footprints,
    load_matlab_subproblem,
    params_from_matlab,
    sel_coords,
    temporal_kernel,
)

DATA = Path(__file__).parent / "data"
HELPERS = DATA / "matlab_solver_helpers.h5"
SUBPROBLEM = DATA / "matlab_solver_subproblem.h5"

# the two parameter sets of the MATLAB references (make_helper_refs.m)
PARAMS_A = SolverParams(
    sigma_px=1.6,
    sel_radius=4,
    tau_full=3.7,
    tau2_s=0.05,
    analyze_hz=150,
    baseline_window_s=2,
    denoise_window_s=0.1,
)
PARAMS_B = SolverParams()


def _close(a, b, tol=1e-13):
    """Largest difference relative to the largest reference value."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    assert a.shape == b.shape
    assert np.abs(a - b).max() <= tol * np.abs(b).max()


@pytest.mark.parametrize("name, params", [("a", PARAMS_A), ("b", PARAMS_B)])
def test_filters_match_matlab(name, params):
    """Footprint filter, surround filters and kernels of
    setParamsExtractTrial."""
    with h5py.File(HELPERS, "r") as f:
        g = f["params"][name]
        _close(params.hfilter, g["Hfilter"][()])
        # MATLAB stores the three filters along the third dimension
        _close(params.bfilter, g["Bfilter"][()].transpose(2, 0, 1))
        _close(params.k, g["k"][()].ravel())
        _close(params.k2, g["k2"][()].ravel())
        if "baselineWindow_samps" in g:
            assert params.baseline_window == g["baselineWindow_samps"][()]
            assert params.denoise_window == g["denoiseWindow_samps"][()]


def test_kernel_and_windows():
    """The kernel is causal, sums to 1 and peaks at its centre; the windows
    follow MATLAB's rounding."""
    k = temporal_kernel(4.8)
    n = int(np.ceil(6 * 4.8))
    assert k.size == 2 * n + 1
    assert np.all(k[:n] == 0) and np.argmax(k) == n
    assert k.sum() == pytest.approx(1.0, abs=1e-15)
    assert k[n + 1] / k[n] == pytest.approx(np.exp(-1 / 4.8))
    p = SolverParams(analyze_hz=434.7826087, tau_s=0.024)
    assert p.tau_frames == pytest.approx(10.4348, abs=1e-4)
    assert p.denoise_window == 87  # ceil(86.96)
    assert p.baseline_window == pytest.approx(1739.13, abs=1e-2)  # not rounded
    assert p.lp_factor == 20
    assert SolverParams(tau_full=7.0).tau_frames == 7.0


def test_params_reject_bad_values():
    """Values the solver cannot run with are refused."""
    with pytest.raises(ValueError, match="debias_variance"):
        SolverParams(debias_variance="other")
    with pytest.raises(ValueError, match="nmf_iter"):
        SolverParams(nmf_iter=0)
    with pytest.raises(ValueError, match="sigma_px"):
        SolverParams(sigma_px=0)
    # divisors: a zero would give a NaN kernel or an endless spline
    with pytest.raises(ValueError, match="tau_full"):
        SolverParams(tau_full=0)
    with pytest.raises(ValueError, match="baseline_knot_factor"):
        SolverParams(baseline_knot_factor=0)
    with pytest.raises(ValueError, match="tau_s"):
        SolverParams(tau_s=float("nan"))
    with pytest.raises(ValueError, match="photon_scale"):
        SolverParams(photon_scale=float("inf"))
    with pytest.raises(ValueError, match="lam"):
        SolverParams(lam=-0.1)
    assert SolverParams(photon_scale=None, lam=0, phi=0).tau_full is None
    # the baseline needs at least two denoise windows per baseline window
    with pytest.raises(ValueError, match="baseline_window_s"):
        SolverParams(baseline_window_s=0.1, denoise_window_s=0.1)
    with pytest.raises(ValueError, match="baseline_window_s"):
        SolverParams(baseline_window_s=0.2, denoise_window_s=0.196)
    assert (
        SolverParams(baseline_window_s=0.3, denoise_window_s=0.2).lp_factor
        == 2
    )


def test_initial_footprints_match_matlab():
    """Starting footprints, also for sources at the map's edge and next
    to an unselected pixel."""
    with h5py.File(HELPERS, "r") as f:
        g = f["foot"]
        sel = g["selPix"][()].astype(bool)
        rc = np.stack([g["R"][()].ravel(), g["C"][()].ravel()], 1) - 1
        hs, h, valid = initial_footprints(sel, rc, float(g["sigma"][()]))
        assert np.array_equal(hs, g["Hs"][()])
        _close(h, g["H"][()])
        assert np.array_equal(valid, g["Hvalid"][()].astype(bool))


def test_sel_coords_are_column_major():
    """Pixels are listed as MATLAB's find(sel) lists them."""
    sel = np.array([[1, 0, 1], [1, 1, 0]], dtype=bool)
    rows, cols = sel_coords(sel)
    assert rows.tolist() == [0, 1, 1, 0]
    assert cols.tolist() == [0, 0, 1, 2]


def test_conv_operator_is_same_convolution():
    """The operator equals a 'same' convolution of the map, zero outside
    the selected pixels, read at the selected pixels."""
    rng = np.random.default_rng(0)
    sel = rng.random((9, 11)) < 0.7
    rows, cols = sel_coords(sel)
    values = rng.standard_normal(rows.size)
    for shape in ((5, 5), (3, 7), (4, 5)):
        filt = rng.standard_normal(shape)
        filt[0, 0] = 0  # a zero tap is left out of the sparse matrix
        image = np.zeros(sel.shape)
        image[rows, cols] = values
        # MATLAB's convn 'same': the centre is floor(size / 2)
        full = signal.convolve2d(image, filt, mode="full")
        r0, c0 = shape[0] // 2, shape[1] // 2
        same = full[r0 : r0 + sel.shape[0], c0 : c0 + sel.shape[1]]
        got = conv_operator(sel, filt) @ values
        np.testing.assert_allclose(got, same[rows, cols], atol=1e-13)


def test_subproblem_checks_its_arrays():
    """A subproblem whose arrays do not fit together is refused."""
    sel = np.ones((3, 3), dtype=bool)
    y = np.zeros((1, 9, 20), np.float32)
    finv = np.ones((9, 20), np.float32)
    sub = Subproblem(y=y, finv=finv, sel=sel, sources_rc=[[1, 1]])
    assert (sub.n_pixels, sub.n_frames, sub.n_sources) == (9, 20, 1)
    assert sub.coords[0].tolist() == [0, 1, 2] * 3
    with pytest.raises(ValueError, match="pixels"):
        Subproblem(y=y[:, :8], finv=finv, sel=sel, sources_rc=[[1, 1]])
    with pytest.raises(ValueError, match="outside"):
        Subproblem(y=y, finv=finv, sel=sel, sources_rc=[[1, 3]])
    with pytest.raises(ValueError, match="channels"):
        Subproblem(y=y[0], finv=finv, sel=sel, sources_rc=[[1, 1]])


def _write_matlab_params(g, params: SolverParams, **extra):
    """A MATLAB params struct as a v7.3 file holds it."""
    fields = dict(
        analyzeHz=params.analyze_hz,
        tau_s=params.tau_s,
        tau_full=params.tau_frames,
        tau2_s=params.tau2_s,
        sigma_px=params.sigma_px,
        selRadius=params.sel_radius,
        baselineWindow_Glu_s=params.baseline_window_s,
        denoiseWindow_s=params.denoise_window_s,
        minBaseline=params.min_baseline,
        nmfIter=params.nmf_iter,
        phi=params.phi,
    )
    fields["lambda"] = params.lam
    for name, v in fields.items():
        g.create_dataset(name, data=np.array([[float(v)]]))
    g.create_dataset("k", data=params.k[None, :])
    g.create_dataset("Hfilter", data=params.hfilter.T)
    # MATLAB's (rows, cols, 3) array, dimensions reversed in the file
    g.create_dataset("Bfilter", data=params.bfilter.transpose(0, 2, 1))
    for name, v in extra.items():
        g.create_dataset(name, data=v)


def _write_matlab_subproblem(folder, y, finv, sel, rc, params, row0, col0):
    """sub_001.mat and trial.mat as extractTrial saves them (v7.3)."""
    full = np.zeros((sel.shape[0] + row0 + 2, sel.shape[1] + col0 + 1), bool)
    full[0, 0] = full[1, 0] = True  # two pixels of another subproblem
    full[row0 : row0 + sel.shape[0], col0 : col0 + sel.shape[1]] = sel
    with h5py.File(folder / "trial.mat", "w") as f:
        f.create_dataset("selPix", data=full.T.astype(np.uint8))
    with h5py.File(folder / "sub_001.mat", "w") as f:
        # MATLAB's pixels x frames x channels, dimensions reversed
        f.create_dataset("Y_p", data=y.transpose(0, 2, 1))
        f.create_dataset("F_inv_p", data=finv.T)
        f.create_dataset("selPix_p", data=sel.T.astype(np.uint8))
        g = f.create_group("sources_p")
        g.create_dataset("R", data=rc[None, :, 0] + 1.0)
        g.create_dataset("C", data=rc[None, :, 1] + 1.0)
        f.create_dataset("rowSupport", data=np.array([[row0 + 1.0, 0]]).T)
        f.create_dataset("colSupport", data=np.array([[col0 + 1.0, 0]]).T)
        f.create_dataset("srcIdx", data=np.array([[4.0, 9.0]]))
        debias = np.array([[ord(c)] for c in "Model"], dtype=np.uint16)
        empty = f.create_group("params").create_dataset(
            "photonScale", data=np.zeros(2, np.uint64)
        )
        empty.attrs["MATLAB_empty"] = 1
        _write_matlab_params(f["params"], params, debiasVariance=debias)
    return folder / "sub_001.mat"


def test_load_matlab_subproblem(tmp_path):
    """A saved subproblem is read in the port's convention: channels x
    pixels x frames, zero-based sources, pixel indices in the trial's
    list, and its parameters."""
    rng = np.random.default_rng(1)
    sel = np.ones((4, 5), dtype=bool)
    sel[0, 0] = sel[3, 4] = False
    n_px = int(sel.sum())
    y = rng.random((2, n_px, 30))
    finv = 0.5 + rng.random((n_px, 30))
    rc = np.array([[1, 1], [2, 3]])
    params = SolverParams(sigma_px=1.6, sel_radius=4, tau_s=0.03, lam=0.7)
    path = _write_matlab_subproblem(tmp_path, y, finv, sel, rc, params, 3, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sub, p = load_matlab_subproblem(path)
    assert sub.y.dtype == np.float64
    assert np.array_equal(sub.y, y) and np.array_equal(sub.finv, finv)
    assert np.array_equal(sub.sel, sel)
    assert np.array_equal(sub.sources_rc, rc)
    assert (sub.row0, sub.col0) == (3, 2)
    assert sub.sources.tolist() == [3, 8]
    # the trial's map has two selected pixels before this crop's
    assert np.array_equal(sub.pixels, 2 + np.arange(n_px))
    assert p.photon_scale is None and p.debias_variance == "model"
    assert (p.sigma_px, p.sel_radius, p.lam) == (1.6, 4, 0.7)
    assert p.tau_frames == pytest.approx(0.03 * 200)
    assert load_matlab_subproblem(path, np.float32)[0].y.dtype == np.float32


def test_trial_file_of_another_trial_is_refused(tmp_path):
    """A trial.mat that does not select every pixel of the subproblem
    cannot give its pixel indices."""
    sel = np.ones((3, 3), dtype=bool)
    path = _write_matlab_subproblem(
        tmp_path,
        np.zeros((1, 9, 5)),
        np.ones((9, 5)),
        sel,
        np.array([[1, 1]]),
        SolverParams(),
        2,
        1,
    )
    with h5py.File(tmp_path / "trial.mat", "r+") as f:
        full = f["selPix"][()]
        full[2, 3] = 0  # one of the crop's pixels, in MATLAB's file order
        f["selPix"][...] = full
    with pytest.raises(ValueError, match="not from the same trial"):
        load_matlab_subproblem(path)
    with h5py.File(tmp_path / "trial.mat", "w") as f:
        f.create_dataset("selPix", data=np.ones((2, 2), np.uint8))
    with pytest.raises(ValueError, match="not from the same trial"):
        load_matlab_subproblem(path)


def test_single_channel_subproblem_without_trial_file(tmp_path):
    """One channel is saved as a 2-D array; without trial.mat the pixel
    indices are unknown."""
    sel = np.ones((2, 3), dtype=bool)
    y = np.arange(6 * 7, dtype=np.float64).reshape(1, 6, 7)
    path = _write_matlab_subproblem(
        tmp_path,
        y,
        np.ones((6, 7)),
        sel,
        np.array([[0, 1]]),
        SolverParams(),
        0,
        0,
    )
    (tmp_path / "trial.mat").unlink()
    with h5py.File(path, "r+") as f:
        del f["Y_p"]
        f.create_dataset("Y_p", data=y[0].T)
    sub, _ = load_matlab_subproblem(path)
    assert sub.y.shape == (1, 6, 7) and np.array_equal(sub.y, y)
    assert sub.pixels is None


def test_params_warn_when_saved_filters_differ(tmp_path):
    """A saved kernel that the scalar parameters do not give is reported."""
    with h5py.File(tmp_path / "p.mat", "w") as f:
        g = f.create_group("params")
        _write_matlab_params(g, SolverParams())
        g["k"][...] = g["k"][()] * 1.01
        with pytest.warns(UserWarning, match="params.k differs"):
            p = params_from_matlab(g)
    assert p == SolverParams(tau_full=4.8, photon_scale=None)


def test_subproblem_fixture_starts_as_matlab():
    """The fixture's starting footprints are MATLAB's."""
    with h5py.File(SUBPROBLEM, "r") as f:
        sel = f["in/sel"][()].astype(bool)
        hs, h, valid = initial_footprints(
            sel, f["in/sources_rc"][()], PARAMS_B.sigma_px
        )
        assert np.array_equal(hs, f["init/Hs"][()])
        _close(h, f["init/H"][()])
        assert np.array_equal(valid, f["init/Hvalid"][()].astype(bool))
