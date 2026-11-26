"""Tests of SILo's one parameter set and the types shared by its stages."""

import h5py
import numpy as np
import pytest

from giant_python.extraction.raster.alignment import TrialWarp
from giant_python.extraction.raster.container import (
    read_container,
    read_subproblem,
    write_container,
)
from giant_python.extraction.raster.localization import (
    RasterLocalizationParams,
)
from giant_python.extraction.raster.plan import (
    PlanParams,
    group_sources,
    loader_warp,
)
from giant_python.extraction.raster.solver import SolverOptions
from giant_python.extraction.raster.solver_setup import (
    SolverParams,
    Subproblem,
)
from giant_python.extraction.raster.trial_data import (
    RigidShift,
    TrialData,
    TrialLoadParams,
)
from giant_python.io.hdf5 import write_matlab_value
from giant_python.models.params import (
    SiloExecution,
    SiloParams,
    resolve_silo_options,
)

# GIAnT-MATLAB's /params of a SLAP2 SILo run (main 5a0673b): name, MATLAB
# size and class. Values of the scalars are checked below.
MATLAB_FIELDS = {
    "Bfilter": ((25, 25, 3), np.float64),
    "Hfilter": ((9, 9), np.float64),
    "VIF": ((1, 1), np.float64),
    "activityChannel": ((1, 1), np.float64),
    "alignHz": ((1, 1), np.float64),
    "alignPatchPx": ((1, 1), np.float64),
    "allowAnnotationGUI": ((1, 1), np.int8),
    "analyzeHz": ((1, 1), np.float64),
    "baselineWindow_Ca_s": ((1, 1), np.float64),
    "baselineWindow_Glu_s": ((1, 1), np.float64),
    "baselineWindow_samps": ((1, 1), np.float64),
    "dXY": ((1, 1), np.float64),
    "debiasVariance": ((1, 1), object),
    "denoiseWindow_s": ((1, 1), np.float64),
    "denoiseWindow_samps": ((1, 1), np.float64),
    "discardInitial_s": ((1, 1), np.float64),
    "drawUserRois": ((1, 1), np.int8),
    "endTime": ((1, 1), object),
    "includeIntegrationROIs": ((1, 1), np.int8),
    "interTrialAlignment": ((1, 1), object),
    "isSLAP2": ((1, 1), np.int8),
    "k": ((1, 59), np.float64),
    "k2": ((1, 361), np.float64),
    "lambda": ((1, 1), np.float64),
    "minBaseline": ((1, 1), np.float64),
    "minKineticScore": ((1, 1), np.float64),
    "minPeakDistance": ((1, 1), np.float64),
    "motionThresh": ((1, 1), np.float64),
    "nWorkers": ((1, 1), np.float64),
    "nanThresh": ((1, 1), np.float64),
    "nmfIter": ((1, 1), np.float64),
    "numChannels": ((1, 1), np.float64),
    "peakMethod": ((1, 1), object),
    "peakth": ((1, 1), np.float64),
    "phi": ((1, 1), np.float64),
    "photonScale": ((1, 1), np.float64),
    "saveSubproblemsDir": ((1, 1), object),
    "selRadius": ((1, 1), np.float64),
    "sigma_px": ((1, 1), np.float64),
    "startTime": ((1, 1), object),
    "tau2_s": ((1, 1), np.float64),
    "tau2_samps": ((1, 1), np.float64),
    "tau_full": ((1, 1), np.float64),
    "tau_s": ((1, 1), np.float64),
    "validKernel": ((3, 3), np.int8),
    "wmfDeblendth": ((1, 1), np.float64),
    "wmfDip": ((1, 1), np.float64),
    "wmfPeakth": ((1, 1), np.float64),
}

MATLAB_VALUES = {
    "VIF": 1.67,
    "activityChannel": 1,
    "alignHz": 80,
    "alignPatchPx": 1500,
    "analyzeHz": 200,
    "baselineWindow_Glu_s": 4,
    "baselineWindow_samps": 800,
    "dXY": 3,
    "denoiseWindow_samps": 40,
    "lambda": 0.9,
    "minBaseline": 0.01,
    "minKineticScore": 0.2,
    "nanThresh": 0.33,
    "nmfIter": 2,
    "numChannels": 2,
    "peakth": 3,
    "photonScale": 25,
    "selRadius": 6,
    "tau2_samps": 30,
    "tau_full": 4.8,
    "wmfDeblendth": 8,
}


def test_defaults_are_matlabs_slap2_silo():
    """The derived stage parameters keep each stage's MATLAB defaults."""
    p = SiloParams()
    assert p.plan_params() == PlanParams()
    assert p.load_params() == TrialLoadParams()
    assert p.solver_params() == SolverParams()
    loc = p.localization_params(80.0)
    # peakth is 3 for SLAP2 in every stage (RasterLocalizationParams' own
    # default, 8, is MATLAB's for other recordings)
    assert loc.peak_th == 3.0
    from dataclasses import replace

    assert replace(loc, peak_th=8.0) == RasterLocalizationParams()
    assert p.sel_radius == 6


def test_one_setting_reaches_every_stage():
    """A shared setting is passed to each stage that uses it."""
    p = SiloParams(sigma_px=1.5, nan_thresh=0.2, peak_th=4.0)
    assert p.plan_params().sigma_px == 1.5
    assert p.solver_params().sigma_px == 1.5
    assert p.localization_params(80).sigma_px == 1.5
    assert p.plan_params().nan_thresh == 0.2
    assert p.load_params().nan_thresh == 0.2
    assert p.localization_params(80).nan_thresh == 0.2
    assert p.localization_params(80).peak_th == 4.0
    assert (
        SiloParams(localization_peak_th=8.0).localization_params(80).peak_th
        == 8.0
    )
    assert p.localization_params(123.0).align_hz == 123.0


def test_to_matlab_has_matlabs_fields_types_and_values(tmp_path):
    """Written as saveStructToH5 would, the struct has MATLAB's schema."""
    m = SiloParams().to_matlab(2, 80.0, 12, "a", "b")
    assert set(m) == set(MATLAB_FIELDS)
    path = tmp_path / "p.h5"
    with h5py.File(path, "w") as f:
        write_matlab_value(f, "params", m)
    with h5py.File(path) as f:
        g = f["params"]
        for name, (size, cls) in MATLAB_FIELDS.items():
            # h5py shows MATLAB's dimensions reversed
            assert g[name].shape == size[::-1], name
            if cls is object:
                assert g[name].dtype.kind == "O", name
            else:
                assert g[name].dtype == cls, name
        for name, value in MATLAB_VALUES.items():
            assert g[name][()].item() == pytest.approx(value), name
        assert g["debiasVariance"][()].item() == b"fixed"
        assert g["peakMethod"][()].item() == b"getactimpeaks"
        assert g["saveSubproblemsDir"][()].item() == b""
        assert g["isSLAP2"][()].item() == 1
        assert g["allowAnnotationGUI"][()].item() == 0
    sp = SolverParams()
    np.testing.assert_array_equal(m["k"], sp.k)
    np.testing.assert_array_equal(m["Bfilter"][:, :, 2], sp.bfilter[2])


def test_python_only_settings_are_recorded_when_not_matlabs():
    """A default struct has MATLAB's fields only; other settings add one."""
    m = SiloParams(
        footprint_variance="fixed",
        localization_baseline="network",
        trial_peaks=False,
        localization_peak_th=8.0,
    ).to_matlab(2, 80.0)
    assert m["footprintVariance"] == "fixed"
    assert m["localizationBaseline"] == "network"
    assert m["trialPeaks"] == np.bool_(False)
    assert m["localizationPeakth"] == 8.0
    assert set(m) - set(MATLAB_FIELDS) == {
        "footprintVariance",
        "localizationBaseline",
        "trialPeaks",
        "localizationPeakth",
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma_px": 0},
        {"activity_channel": 0},
        {"activity_channel": 1.5},
        {"lam": -1},
        {"debias_variance": "other"},
        {"footprint_variance": "other"},
        {"localization_baseline": "other"},
        {"localization_null": "other"},
        {"trial_peaks": 1},
        {"min_frames": 0},
        {"min_kinetic_score": float("nan")},
        # the solver needs a baseline window longer than the denoise window
        {"baseline_window_s": 0.1},
    ],
)
def test_invalid_values_are_rejected(kwargs):
    """Values no stage can run with raise ValueError."""
    with pytest.raises(ValueError):
        SiloParams(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [{"inter_trial_alignment": "affine"}, {"peak_method": "wmf"}],
)
def test_modes_not_ported_raise(kwargs):
    """Affine alignment and WMF seeding are not ported yet."""
    with pytest.raises(NotImplementedError):
        SiloParams(**kwargs)


def test_stage_settings_change_only_with_their_fields():
    """A solver setting changes the solve stage's settings only."""
    a, b = SiloParams(), SiloParams(lam=0.5)
    for stage in ("localize", "plan", "prepare", "collect"):
        assert a.stage_settings(stage) == b.stage_settings(stage)
    assert a.stage_settings("solve") != b.stage_settings("solve")
    assert a.stage_settings("plan") != SiloParams(peak_th=4.0).stage_settings(
        "plan"
    )
    with pytest.raises(ValueError):
        a.stage_settings("other")


def test_execution_options():
    """Execution options validate and give the solver's options."""
    e = SiloExecution(device="cuda", dtype="float32", batch_bytes=1e8)
    assert e.solver_options() == SolverOptions(
        device="cuda", dtype="float32", batch_bytes=1e8
    )
    for bad in (
        {"solver_threads": 0},
        {"device": "gpu"},
        {"dtype": "float16"},
        {"batch_bytes": -1.0},
        {"overwrite": 1},
    ):
        with pytest.raises(ValueError):
            SiloExecution(**bad)
    p, e = resolve_silo_options({"lam": 0.5}, {"workers": 2})
    assert p.lam == 0.5 and e.workers == 2
    with pytest.raises(TypeError):
        resolve_silo_options({"nWorkers": 2})


def _trial(sel, n_frames=30, seed=0):
    """A trial with random data at the selected pixels of ``sel``."""
    rng = np.random.default_rng(seed)
    cols, rows = np.nonzero(sel.T)
    rc = np.stack([rows, cols], 1)
    return TrialData(
        y=rng.random((2, rows.size, n_frames)).astype(np.float32),
        finv=(1 + rng.random((rows.size, n_frames))).astype(np.float32),
        frame_lines=np.arange(1, n_frames + 1),
        motion_r=np.zeros(n_frames),
        motion_c=np.zeros(n_frames),
        discard=np.zeros(n_frames, bool),
        global_f=rng.random((2, n_frames)),
        roi_f=np.zeros((0, 2, n_frames)),
        roi_fsvd=np.zeros((0, 2, n_frames)),
        pixel_rc=rc,
        trial_rc=rc.astype(float),
        photon_scale=25.0,
    )


def test_plan_groups_container_and_solver_share_one_type(tmp_path):
    """A plan group's subproblem equals the container's: the solver's type."""
    shape = (20, 24)
    rows, cols = np.array([5, 6, 14]), np.array([5, 8, 17])
    sel = np.zeros(shape, bool)
    for r, c in zip(rows, cols):
        sel[max(r - 4, 0) : r + 5, max(c - 4, 0) : c + 5] = True
    groups = group_sources(rows, cols, sel, zone_radius=3, sel_radius=6)
    assert len(groups) == 2
    trial = _trial(sel)
    direct = [g.subproblem(trial.y, trial.finv) for g in groups]
    assert all(isinstance(s, Subproblem) for s in direct)
    path = tmp_path / "c.h5"
    write_container(
        path,
        trial,
        np.stack([rows, cols], 1),
        [g.pixels for g in groups],
        [g.sources for g in groups],
        shape,
    )
    container = read_container(path)
    for k, sub in enumerate(direct):
        for got in (container.subproblem(k), read_subproblem(path, k)):
            assert isinstance(got, Subproblem)
            for name in ("y", "finv", "sel", "sources_rc", "pixels"):
                np.testing.assert_array_equal(
                    getattr(got, name), getattr(sub, name)
                )
            np.testing.assert_array_equal(got.sources, sub.sources)
            assert (got.row0, got.col0) == (sub.row0, sub.col0)


def test_loader_warp():
    """A shift reaches the loader as a RigidShift, an affine map as a
    function of zero-based coordinates; an unaligned trial raises."""
    warp = TrialWarp.from_shifts(np.array([[1.5, -2.0], [np.nan, np.nan]]))
    w = loader_warp(warp, 0)
    assert w == RigidShift(1.5, -2.0)
    with pytest.raises(ValueError):
        loader_warp(warp, 1)
    affine = np.array([[[1.0, 0.1, 0.5], [0.0, 1.0, -1.0]]])
    warp = TrialWarp(affine)
    f = loader_warp(warp, 0)
    r, c = f(np.array([2.0]), np.array([3.0]))
    np.testing.assert_allclose(r, [2 + 0.3 + 0.5])
    np.testing.assert_allclose(c, [3 - 1.0])
