"""Tests for giant_python.bandsilo.traces (Phase 7 high-res traces).

Covers the per-trial compute (motion interp/binning, per-motion least-squares
``phi``/``F0``, global and per-ROI fluorescence) and the IO wrapper's
skip/monkeypatched-read paths. ``compute_high_res_traces`` is additionally
cross-checked against a verbatim copy of the reference in the project's
development notes; here we assert structural behavior on synthetic data.
"""

import unittest

import numpy as np
import torch

from giant_python.bandsilo import geometry as geo
from giant_python.bandsilo import traces as tr


def _geometry(npc=15, npr=15, num_fast_zs=2):
    """Build a small consistent geometry (subsample map + sparse H + PSF)."""
    plane = npc * npr
    positions = [(r, c) for r in range(4, 11) for c in range(4, 11)]
    rows = []
    for z in range(num_fast_zs):
        for r, c in positions:
            rows.append([z * plane + r * npr + c, len(rows) + 1])
    smi = np.array(rows, dtype=np.int32)
    yy, xx = np.mgrid[-2:3, -2:3]
    psf2d = np.exp(-(yy**2 + xx**2) / 2.0).astype(np.float32)
    sh_inds, sh_vals = geo.build_sparse_h(smi, psf2d, npc, npr)
    return dict(
        npc=npc,
        npr=npr,
        nz=num_fast_zs,
        nsp=smi.shape[0],
        smi=smi,
        psf2d=psf2d,
        sh_inds=sh_inds,
        sh_vals=sh_vals,
    )


def _trial_inputs(num_channels=1, unique_motion_ds=None, seed=2):
    """Synthetic loaded-trial arrays + geometry for compute_high_res_traces."""
    g = _geometry()
    rng = np.random.default_rng(seed)
    nsp = g["nsp"]
    n_ds = 60
    n_frames = 40
    ds_frames = np.arange(n_ds, dtype=float)
    frames = np.linspace(1, n_ds - 2, n_frames)
    a_data = dict(
        DSframes=ds_frames,
        motionDSr=(np.arange(n_ds) % 2).astype(float),  # bins 0 and 1
        motionDSc=np.zeros(n_ds),
        motionDSz=np.zeros(n_ds),
        onlineYshift=rng.standard_normal(n_ds),
        onlineXshift=rng.standard_normal(n_ds),
        onlineZshift=rng.standard_normal(n_ds),
    )
    data = rng.standard_normal((nsp, n_frames)).astype(np.float32) + 2
    data2 = (
        rng.standard_normal((nsp, n_frames)).astype(np.float32) + 1
        if num_channels >= 2
        else None
    )
    background_ds = rng.standard_normal((nsp, n_ds)).astype(np.float32) + 1
    if unique_motion_ds is None:
        unique_motion_ds = np.array([[0, 0], [1, 0]], dtype=float)
    mot_keep_ds = np.arange(unique_motion_ds.shape[0])
    n_pixels = g["nz"] * g["npc"] * g["npr"]
    a_final = torch.rand((n_pixels, 3), dtype=torch.float32)
    soma_sps = [np.array([0, 1, 2]), np.array([5, 6, 7, 8])]
    return dict(
        g=g,
        data=data,
        data2=data2,
        background_ds=background_ds,
        frames=frames,
        a_data=a_data,
        a_final=a_final,
        unique_motion_ds=unique_motion_ds,
        mot_keep_ds=mot_keep_ds,
        soma_sps=soma_sps,
        num_channels=num_channels,
        n_frames=n_frames,
    )


def _compute(inp):
    """Call compute_high_res_traces with the assembled inputs dict."""
    g = inp["g"]
    return tr.compute_high_res_traces(
        inp["data"],
        inp["data2"],
        inp["background_ds"],
        inp["frames"],
        inp["a_data"],
        g["smi"],
        g["sh_inds"],
        g["sh_vals"],
        inp["a_final"],
        inp["unique_motion_ds"],
        inp["mot_keep_ds"],
        0.0,
        g["psf2d"],
        g["nsp"],
        g["nz"],
        g["npc"],
        g["npr"],
        inp["num_channels"],
        inp["soma_sps"],
    )


class TestComputeHighResTraces(unittest.TestCase):
    """compute_high_res_traces per-trial numerics."""

    def test_single_channel_shapes_and_fit(self):
        """Returns the 8-tuple; phi/F0 are per-frame per-source with fits."""
        inp = _trial_inputs(num_channels=1)
        out = _compute(inp)
        phi, f0, frames, sel, global_f, motion, online, f_soma = out
        n_sources = inp["a_final"].shape[1]
        self.assertEqual(phi.shape, (inp["n_frames"], n_sources))
        self.assertEqual(f0.shape, (inp["n_frames"], n_sources))
        self.assertEqual(frames.shape, (inp["n_frames"],))
        self.assertEqual(global_f.shape, (inp["n_frames"], 1))
        self.assertEqual(
            f_soma.shape, (inp["n_frames"], len(inp["soma_sps"]), 1)
        )
        self.assertEqual(len(motion), 3)
        self.assertEqual(len(online), 3)
        self.assertTrue(np.any(np.isfinite(phi)))

    def test_two_channels(self):
        """A second channel adds columns to global and soma fluorescence."""
        inp = _trial_inputs(num_channels=2)
        out = _compute(inp)
        _, _, _, _, global_f, _, _, f_soma = out
        self.assertEqual(global_f.shape, (inp["n_frames"], 2))
        self.assertEqual(
            f_soma.shape, (inp["n_frames"], len(inp["soma_sps"]), 2)
        )
        # both channels populated where frames are kept
        self.assertTrue(np.any(np.isfinite(f_soma[:, :, 1])))

    def test_unmatched_ds_bin_skipped(self):
        """A low-res motion bin absent from the trial is skipped."""
        umd = np.array([[0, 0], [1, 0], [5, 5]], dtype=float)  # [5,5] absent
        inp = _trial_inputs(num_channels=1, unique_motion_ds=umd)
        out = _compute(inp)
        # still produces finite fits for the two present bins
        self.assertTrue(np.any(np.isfinite(out[0])))


class TestGetHighResTraces(unittest.TestCase):
    """get_high_res_traces IO wrapper (skip + monkeypatched read)."""

    def test_skipped_trial_returns_empty(self):
        """keep_trial=False returns the empty 8-tuple without reading."""
        inp = _trial_inputs(num_channels=2)
        g = inp["g"]
        out = tr.get_high_res_traces(
            (0, False, inp["background_ds"]),
            0,
            100.0,
            np.zeros((g["nsp"], 1), dtype=np.int32),
            "unused",
            {},
            g["smi"],
            g["sh_inds"],
            g["sh_vals"],
            inp["a_final"].numpy(),
            inp["unique_motion_ds"],
            inp["mot_keep_ds"],
            0.0,
            g["psf2d"],
            g["nsp"],
            g["nz"],
            g["npc"],
            g["npr"],
            2,
            inp["soma_sps"],
        )
        self.assertEqual(out[0].shape, (0, inp["a_final"].shape[1]))
        self.assertEqual(out[4].shape, (0, 2))  # globalF (0, channels)
        self.assertEqual(out[2].shape, (0,))  # frames

    def test_kept_trial_reads_and_computes(self):
        """A kept trial reads (monkeypatched) then computes the traces."""
        inp = _trial_inputs(num_channels=1)
        g = inp["g"]

        def fake_load(*args, **kwargs):
            """Return the synthetic loaded-trial arrays."""
            return (
                inp["data"],
                inp["data2"],
                inp["a_data"],
                inp["frames"],
            )

        orig = tr._load_high_res_trial_data
        tr._load_high_res_trial_data = fake_load
        try:
            out = tr.get_high_res_traces(
                (0, True, inp["background_ds"]),
                0,
                100.0,
                np.zeros((g["nsp"], 1), dtype=np.int32),
                "unused",
                {},
                g["smi"],
                g["sh_inds"],
                g["sh_vals"],
                inp["a_final"].numpy(),  # numpy -> exercises tensor conversion
                inp["unique_motion_ds"],
                inp["mot_keep_ds"],
                0.0,
                g["psf2d"],
                g["nsp"],
                g["nz"],
                g["npc"],
                g["npr"],
                1,
                inp["soma_sps"],
            )
        finally:
            tr._load_high_res_trial_data = orig

        # matches a direct compute with the same inputs
        expected = _compute(inp)
        np.testing.assert_allclose(
            np.nan_to_num(out[0]), np.nan_to_num(expected[0]), rtol=1e-6
        )
        self.assertEqual(out[0].shape, (inp["n_frames"], 3))


if __name__ == "__main__":
    unittest.main()
