"""Regression tests for MATLAB-compatible public offline displacement."""

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from giant_python.extraction.band.inputs import load_alignment_data_h5
from giant_python.extraction.band.result_assembly import assemble_path_summary
from giant_python.extraction.band.traces import _interp_trial_alignment
from giant_python.io.experiment_summary import write_summary
from giant_python.models.experiment_summary import ExperimentSummary
from tests.test_bandsilo_pipeline import _path_result


class TestMotionConvention(unittest.TestCase):
    """Convert private reference offsets back to public displacement once."""

    def test_alignment_to_summary_preserves_saved_displacement(self):
        """Signed fractional motion survives loading, interpolation and IO."""
        with tempfile.TemporaryDirectory() as directory:
            alignment = Path(directory) / "alignment.h5"
            ds = np.array([1.0, 3.0, 5.0, 7.0])
            frames = np.arange(1.0, 8.0)
            displacement = (
                np.array([1.5, -2.0, 0.0, 3.0]),
                np.array([-3.0, 2.5, 1.0, -1.0]),
                np.array([0.5, -0.5, 1.5, 0.0]),
            )
            with h5py.File(alignment, "w") as f:
                f["row_major"] = 1
                f["DSframes"] = ds
                f["numChannels"] = 1
                f["alignHz"] = 10.0
                for axis, values in zip("rcz", displacement):
                    f[f"motionDS{axis}"] = values
                for axis, value in zip("YXZ", (2, -3, 1)):
                    f[f"slap2/onlineMotion{axis}shift"] = np.full(4, value)
            loaded = load_alignment_data_h5(alignment)
            motion = _interp_trial_alignment(
                frames, loaded, np.zeros((1, ds.size))
            )
            pr = _path_result()
            trial = pr.trace_results[1]._replace(
                frame_line_idxs=frames,
                reference_offsets=motion[:3],
                online_shifts=motion[4:],
            )
            pr.trace_results = [trial]
            original = [a.copy() for a in trial.reference_offsets]
            path = assemble_path_summary(pr)
            summary = ExperimentSummary(paths=[path])
            output = Path(directory) / "summary.h5"
            write_summary(summary, output)
            with h5py.File(output, "r") as f:
                for axis, values in zip("YXZ", displacement):
                    key = f"offline{axis}shifts"
                    expected = np.interp(frames, ds, values)[:, None]
                    np.testing.assert_allclose(
                        getattr(summary.paths[0].frame_info, key), expected
                    )
                    np.testing.assert_allclose(
                        f[f"Path1/frame_info/{key}"], expected
                    )
                for axis, value in zip("YXZ", (2, -3, 1)):
                    key = f"online{axis}shifts"
                    np.testing.assert_array_equal(
                        getattr(path.frame_info, key), np.full((7, 1), value)
                    )
            for before, after in zip(original, trial.reference_offsets):
                np.testing.assert_array_equal(before, after)

    def test_output_negation_preserves_nan_and_input_arrays(self):
        """NaNs and trial order survive without modifying numerical inputs."""
        pr = _path_result()
        pr.trace_results[0].reference_offsets[0][0] = np.nan
        original = np.concatenate(
            [r.reference_offsets[0] for r in pr.trace_results]
        )
        path = assemble_path_summary(pr)
        np.testing.assert_array_equal(
            path.frame_info.offlineYshifts.ravel(), -original
        )
        np.testing.assert_array_equal(
            np.concatenate([r.reference_offsets[0] for r in pr.trace_results]),
            original,
        )
