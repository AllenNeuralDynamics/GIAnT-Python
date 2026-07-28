"""Tests for giant_python.bandsilo.trial_data (pure + mocked-reader paths)."""

import os
import tempfile
import unittest

import numpy as np

from giant_python.bandsilo import trial_data as td


class _MockDataFile:
    """Minimal SLAP2 data-file stub for accumulate_superpixel_data.

    Models a single cycle with ``lines_per_cycle`` lines. Each line maps to a
    superpixel id / fast-Z via ``line_super_pixel_ids`` and ``line_fastz`` and
    returns a fixed per-channel value from ``get_line_data``.
    """

    def __init__(self, lines_per_cycle, num_cycles, line_specs, channels=2):
        """Build a stub from per-line ``(sp_ids, fastz, values)`` specs."""
        self.header = {"linesPerCycle": lines_per_cycle}
        self.numCycles = num_cycles
        self._line_specs = line_specs
        self._channels = channels

        class _Meta:
            """Minimal metadata stub exposing ``linePeriod_s``."""

            linePeriod_s = 1.0

        self.metaData = _Meta()
        self.lineDataNumElements = [len(spec[0]) for spec in line_specs]
        self.lineSuperPixelIDs = [np.asarray(spec[0]) for spec in line_specs]
        self.lineFastZIdxs = [np.asarray(spec[1]) for spec in line_specs]

    def getLineData(self, lines, cycles, channel):
        """Return a per-line list of (n_positions, n_channels) arrays."""
        out = []
        for li in lines:
            spec = self._line_specs[int(li) - 1]
            values = np.asarray(spec[2], dtype=np.float32)
            out.append(np.tile(values.reshape(-1, 1), (1, self._channels)))
        return out


class TestNearestInterp(unittest.TestCase):
    """nearest_interp mirrors MATLAB nearest interpolation."""

    def test_single_point(self):
        """A single sample returns yp unchanged."""
        yp = np.array([7.0])
        out = td.nearest_interp(np.array([0.0, 5.0]), np.array([2.0]), yp)
        np.testing.assert_array_equal(out, yp)

    def test_midpoint_rounding(self):
        """Query points snap to the nearest sample boundary."""
        xp = np.array([0.0, 10.0])
        yp = np.array([1.0, 2.0])
        out = td.nearest_interp(np.array([4.0, 6.0]), xp, yp)
        np.testing.assert_array_equal(out, np.array([1.0, 2.0]))


class TestFastDilation(unittest.TestCase):
    """fast_dilation covers the 3x3 fast path and the cv2 fallback."""

    def test_3x3_fast_path(self):
        """A single point dilates to its 8-neighborhood."""
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        out = td.fast_dilation(mask, np.ones((3, 3), np.uint8))
        self.assertEqual(out[1:4, 1:4].sum(), 9)
        self.assertEqual(out.sum(), 9)

    def test_leading_axes_fast_path(self):
        """The fast path dilates only the trailing two axes."""
        mask = np.zeros((2, 5, 5), dtype=bool)
        mask[0, 2, 2] = True
        out = td.fast_dilation(mask, np.ones((3, 3), np.uint8))
        self.assertEqual(out[0].sum(), 9)
        self.assertEqual(out[1].sum(), 0)

    def test_generic_cv2_path(self):
        """A non-3x3 kernel routes through the cv2 fallback."""
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        out = td.fast_dilation(mask, np.ones((1, 3), np.uint8))
        # 1x3 horizontal kernel dilates left/right only.
        self.assertTrue(out[2, 1] and out[2, 3] and out[2, 2])
        self.assertFalse(out[1, 2] or out[3, 2])

    def test_default_kernel(self):
        """A None kernel uses the 7x7 default (generic path)."""
        mask = np.zeros((9, 9), dtype=bool)
        mask[4, 4] = True
        out = td.fast_dilation(mask)
        self.assertEqual(out[1:8, 1:8].sum(), 49)


class TestComputeDsFrames(unittest.TestCase):
    """compute_ds_frames builds the downsampled line grid."""

    def test_grid(self):
        """The grid is ceil of arange(first, last+1, dt)."""
        out = td.compute_ds_frames(1, 5, 2.0)
        np.testing.assert_array_equal(out, np.array([1.0, 3.0, 5.0]))


class TestAccumulate(unittest.TestCase):
    """accumulate_superpixel_data scatters weighted line data."""

    def _super_pixel_ids(self):
        """Two superpixels: sp1 z1 -> 101, sp2 z1 -> 201."""
        return np.array([[101], [201]], dtype=np.int32)

    def _data_file(self, channels=2):
        """One cycle, two lines each mapping to one superpixel."""
        line_specs = [
            ([1], [1], [10.0]),  # line 1 -> sp id 1*100+1 = 101, value 10
            ([2], [1], [20.0]),  # line 2 -> sp id 2*100+1 = 201, value 20
        ]
        return _MockDataFile(2, 1, line_specs, channels=channels)

    def test_single_channel(self):
        """Channel-1 read populates data/data_count; no second channel."""
        df = self._data_file()
        ds_frames = np.array([1.0, 2.0])
        acc = td.accumulate_superpixel_data(
            df, ds_frames, 1.0, self._super_pixel_ids(), 1, False
        )
        self.assertEqual(acc["data"].shape, (2, 2))
        self.assertIsNone(acc["data2"])
        # sp1 gets its strongest weight at frame 0 (line 1), sp2 at frame 1.
        self.assertGreater(acc["data"][0, 0], 0)
        self.assertGreater(acc["data"][1, 1], 0)
        # Normalized value recovers the line value.
        norm = acc["data"][0, 0] / acc["data_count"][0, 0]
        self.assertAlmostEqual(norm, 10.0, places=4)

    def test_two_channels(self):
        """all_channels + num_channels>=2 fills the second-channel arrays."""
        df = self._data_file()
        ds_frames = np.array([1.0, 2.0])
        acc = td.accumulate_superpixel_data(
            df, ds_frames, 1.0, self._super_pixel_ids(), 2, True
        )
        self.assertIsNotNone(acc["data2"])
        self.assertEqual(acc["data2"].shape, (2, 2))
        self.assertGreater(acc["data_count2"][0, 0], 0)

    def test_zero_element_line_skipped(self):
        """A line with zero data elements is skipped."""
        line_specs = [
            ([], [], []),  # line 1: empty -> skipped
            ([2], [1], [20.0]),
        ]
        df = _MockDataFile(2, 1, line_specs, channels=1)
        acc = td.accumulate_superpixel_data(
            df, np.array([1.0, 2.0]), 1.0, self._super_pixel_ids(), 1, False
        )
        # sp1 (id 101) never accumulates because its only line is empty.
        self.assertEqual(acc["data"][0].sum(), 0)

    def test_line_with_no_matching_superpixels_skipped(self):
        """A non-empty line matching none of the ids is skipped."""
        line_specs = [
            ([9], [9], [10.0]),  # lookup 909 -> matches neither 101 nor 201
            ([2], [1], [20.0]),  # sp2 (id 201) matches
        ]
        df = _MockDataFile(2, 1, line_specs, channels=1)
        acc = td.accumulate_superpixel_data(
            df, np.array([1.0, 2.0]), 1.0, self._super_pixel_ids(), 1, False
        )
        # sp1 (101) never matches any line; sp2 (201) accumulates.
        self.assertEqual(acc["data"][0].sum(), 0)
        self.assertGreater(acc["data"][1].sum(), 0)


class TestReadBandTrialData(unittest.TestCase):
    """read_band_trial_data orchestrates open+accumulate+alignment."""

    def test_skipped_trial(self):
        """keep_trial=False returns None without opening anything."""
        out = td.read_band_trial_data(
            0, False, 0, 100.0, np.zeros((1, 1)), "/data", {}, 1
        )
        self.assertIsNone(out)

    def test_full_read_with_monkeypatched_open(self):
        """A kept trial reads, scales by 1/100, and loads alignment."""
        sp_ids = np.array([[101], [201]], dtype=np.int32)
        df = _MockDataFile(
            2, 1, [([1], [1], [10.0]), ([2], [1], [20.0])], channels=1
        )
        trial_table = {
            "filename": np.array([["t1.dat"]], dtype=object),
            "first_line": np.array([[1]]),
            "last_line": np.array([[2]]),
            "fn_adata": np.array([["a.h5"]], dtype=object),
        }

        orig_open = td._open_slap2_file
        orig_align = td.load_alignment_data_h5
        td._open_slap2_file = lambda path: df
        td.load_alignment_data_h5 = lambda path: {
            "motionDSr": np.zeros(2),
            "motionDSc": np.zeros(2),
            "motionDSz": np.zeros(2),
        }
        try:
            out = td.read_band_trial_data(
                0,
                True,
                0,
                1.0,
                sp_ids,
                "/data",
                trial_table,
                1,
                all_channels=False,
            )
        finally:
            td._open_slap2_file = orig_open
            td.load_alignment_data_h5 = orig_align

        self.assertIsNotNone(out)
        self.assertEqual(out["data"].shape, (2, 2))
        self.assertIsNone(out["data2"])
        norm = out["data"][0, 0] / out["data_count"][0, 0]
        self.assertAlmostEqual(norm, 10.0 / 100.0, places=6)


class TestAssembleAndCache(unittest.TestCase):
    """assemble_lowres_data + save/load npz round trip."""

    def _results(self, num_channels):
        """Two synthetic trial results (one None) for assembly."""
        r0 = {
            "data": np.ones((2, 3), dtype=np.float32),
            "data_count": np.full((2, 3), 2.0, dtype=np.float32),
            "alignment": {
                "motionDSr": np.zeros(3),
                "motionDSc": np.ones(3),
                "motionDSz": np.full(3, 2.0),
            },
            "ds_frames": np.arange(3),
            "data2": np.full((2, 3), 5.0, dtype=np.float32),
            "data_count2": np.full((2, 3), 2.0, dtype=np.float32),
        }
        r1 = {
            "data": np.ones((2, 2), dtype=np.float32),
            "data_count": np.full((2, 2), 2.0, dtype=np.float32),
            "alignment": {
                "motionDSr": np.zeros(2),
                "motionDSc": np.ones(2),
                "motionDSz": np.full(2, 2.0),
            },
            "ds_frames": np.arange(2),
            "data2": np.full((2, 2), 5.0, dtype=np.float32),
            "data_count2": np.full((2, 2), 2.0, dtype=np.float32),
        }
        return [r0, None, r1]

    def test_assemble_single_channel(self):
        """Single-channel assembly concatenates and builds trial IDs."""
        arrays = td.assemble_lowres_data(self._results(1), 1)
        self.assertEqual(arrays["lowResData"].shape, (2, 5))
        self.assertNotIn("lowResData2", arrays)
        # trial IDs: three 0s (r0) then two 2s (r1, original index 2).
        np.testing.assert_array_equal(
            arrays["lowResTrialID"], np.array([0, 0, 0, 2, 2])
        )

    def test_assemble_two_channels(self):
        """Two-channel assembly adds the second-channel arrays."""
        arrays = td.assemble_lowres_data(self._results(2), 2)
        self.assertEqual(arrays["lowResData2"].shape, (2, 5))
        self.assertEqual(arrays["lowResDataCt2"].shape, (2, 5))

    def test_save_load_round_trip(self):
        """save_lowres_data + load_lowres_data preserve arrays."""
        arrays = td.assemble_lowres_data(self._results(2), 2)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "lowres_data_DMD1.npz")
            td.save_lowres_data(path, arrays)
            loaded = td.load_lowres_data(path, 2)
        np.testing.assert_array_equal(
            loaded["lowResData"], arrays["lowResData"]
        )
        np.testing.assert_array_equal(
            loaded["lowResData2"], arrays["lowResData2"]
        )


if __name__ == "__main__":
    unittest.main()
