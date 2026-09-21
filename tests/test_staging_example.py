"""Safety and relocation contracts for the standalone staging example."""

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

from examples import extract_band as example
from giant_python import AnnotationOptions, BandSiloParams, ExecutionOptions
from giant_python.models.trial_table import Slap2Info, TrialTable

ROOT = Path(__file__).resolve().parents[1]


class TestExampleImport(unittest.TestCase):
    """Import and help work without loading the scientific backend."""

    def test_import_has_no_filesystem_environment_or_backend_side_effects(
        self,
    ):
        """Execute an import in a fresh interpreter with mutation guards."""
        code = textwrap.dedent("""
            import argparse
            import builtins
            import contextlib
            import dataclasses
            import os
            import pathlib
            import shutil
            import sys
            import types
            import typing
            from unittest.mock import patch

            path = pathlib.Path('examples/extract_band.py')
            source = compile(path.read_text(), str(path), 'exec')
            module = types.ModuleType('staging_import_check')
            sys.modules[module.__name__] = module
            before = dict(os.environ)
            original_import = builtins.__import__
            forbidden = {
                'giant_python', 'numpy', 'scipy', 'matplotlib', 'cv2',
                'tkinter', 'joblib', 'slap2_utils', 'torch', 'h5py',
            }

            def checked_import(name, *args, **kwargs):
                assert name.split('.')[0] not in forbidden, name
                return original_import(name, *args, **kwargs)

            def audit(event, args):
                assert event not in {
                    'open', 'os.mkdir', 'os.remove', 'os.rmdir', 'os.rename',
                    'os.symlink', 'os.link', 'os.putenv', 'os.unsetenv',
                    'os.chdir', 'os.chmod', 'os.truncate', 'os.utime',
                    'os.listdir', 'os.scandir',
                }, (event, args)

            sys.addaudithook(audit)
            with patch('builtins.__import__', side_effect=checked_import):
                exec(source, module.__dict__)
            assert dict(os.environ) == before
            assert not forbidden.intersection(sys.modules)
            assert callable(module.main)
            """)
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_help_exits_before_validation(self):
        """Help requires neither a dataset nor GIAnT imports."""
        with (
            patch.object(example, "prepare_run") as prepare,
            contextlib.redirect_stdout(io.StringIO()) as output,
            self.assertRaises(SystemExit) as raised,
        ):
            example.main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("--stage-data", output.getvalue())
        self.assertNotIn("--operator", output.getvalue())
        prepare.assert_not_called()


class TestRelativeNames(unittest.TestCase):
    """Metadata cannot bypass input roots on either path convention."""

    def test_relative_nested_and_empty_entries_are_accepted(self):
        """Preserve relative subdirectories and absent per-trial entries."""
        example.validate_relative_names(
            [
                [None, "", "trial.dat"],
                ["raw/trial.dat", r"raw\trial.dat", "."],
            ],
            "filename",
        )

    def test_absolute_drive_unc_and_parent_names_are_rejected(self):
        """POSIX and Windows escapes fail regardless of the host OS."""
        for name in (
            "/raw/trial.dat",
            "../trial.dat",
            "raw/../trial.dat",
            r"..\trial.dat",
            r"raw\..\trial.dat",
            r"raw/..\trial.dat",
            r"C:\raw\trial.dat",
            "C:/raw/trial.dat",
            "C:trial.dat",
            r"\raw\trial.dat",
            r"\\server\share\trial.dat",
            "//server/share/trial.dat",
        ):
            for field in ("filename", "motion_correction/fn_adata"):
                with self.subTest(name=name, field=field):
                    with self.assertRaisesRegex(ValueError, field):
                        example.validate_relative_names([["ok", name]], field)


class TestStagingExample(unittest.TestCase):
    """Use tiny fixture files and a mocked backend, never a real dataset."""

    def setUp(self):
        """Create independent input trees and a portable loaded table."""
        temporary = tempfile.TemporaryDirectory(prefix="giant-example-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.data = self.root / "raw"
        self.motion = self.root / "registered"
        self.annotations = self.root / "rois"
        self.scratch = self.root / "scratch"
        self.results = self.root / "run"
        self.table_path = self.root / "selected-table.h5"
        for directory in (
            self.data,
            self.motion,
            self.annotations,
            self.scratch,
        ):
            directory.mkdir()
        self.table_path.write_bytes(b"unchanged table provenance")
        (self.data / "trial.dat").write_bytes(b"raw")
        (self.data / "trial.meta").write_bytes(b"sidecar")
        (self.data / "reference.tif").write_bytes(b"reference")
        (self.motion / "bandRegLookupTable.h5").write_bytes(b"lookup")
        (self.motion / "alignment.h5").write_bytes(b"alignment")
        (self.annotations / "annotations.h5").write_bytes(b"annotations")
        self.table = TrialTable(
            datadr=Path("old/raw"),
            savedr=Path("old/results"),
            filename=[["trial.dat"]],
            slap2_info=Slap2Info(),
            motion_correction={"fn_adata": [["alignment.h5"]]},
            source_path=Path("old/table.h5"),
        )
        self.load = self.enterContext(
            patch.object(TrialTable, "from_h5", return_value=self.table)
        )
        self.extract = self.enterContext(
            patch(
                "giant_python.extract_band_sources",
                return_value=SimpleNamespace(paths=[object()]),
            )
        )

    def argv(self, *extra):
        """Build explicit input arguments, allowing final flag overrides."""
        return [
            "--trial-table",
            str(self.table_path),
            "--data-dir",
            str(self.data),
            "--motion-correction",
            str(self.motion),
            "--results-dir",
            str(self.results),
            *map(str, extra),
        ]

    def invoke(self, *extra):
        """Run quietly with the mocked extraction service."""
        with contextlib.redirect_stdout(io.StringIO()):
            return example.main(self.argv(*extra))

    def snapshot(self):
        """Capture fixture contents, including links without following them."""
        return {
            str(path.relative_to(self.root)): (
                ("link", str(path.readlink()))
                if path.is_symlink()
                else (
                    ("directory", None)
                    if path.is_dir()
                    else ("file", path.read_bytes())
                )
            )
            for path in self.root.rglob("*")
        }

    def assert_rejected(self, error, *extra):
        """Rejected inputs must not stage, extract, write or change env."""
        before = self.snapshot()
        environment = dict(os.environ)
        extraction_calls = list(self.extract.mock_calls)
        with (
            patch.object(example, "stage_directory") as stage,
            patch.object(example.shutil, "copy2") as copy,
            patch.object(Path, "mkdir") as mkdir,
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(error),
        ):
            self.invoke("--joblib-temp", self.scratch, *extra)
        stage.assert_not_called()
        copy.assert_not_called()
        mkdir.assert_not_called()
        self.assertEqual(self.extract.mock_calls, extraction_calls)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(dict(os.environ), environment)

    def test_default_run_relocates_loaded_model_without_rewriting_source(self):
        """Default staging changes only the copied model's effective roots."""
        original = asdict(self.table)
        self.assertEqual(self.invoke(), 0)
        self.load.assert_called_once_with(self.table_path)
        self.extract.assert_called_once()
        args, kwargs = self.extract.call_args
        table, science = args
        self.assertIsInstance(table, TrialTable)
        self.assertIsNot(table, self.table)
        self.assertEqual(table.datadr, self.data)
        self.assertEqual(table.savedr, self.results)
        self.assertEqual(
            table.source_path, self.results / "input_trial_table.h5"
        )
        self.assertEqual(asdict(self.table), original)
        self.assertEqual(science, BandSiloParams())
        self.assertEqual(science.peakth, 7.0)
        self.assertEqual(kwargs["execution"], ExecutionOptions())
        self.assertEqual(
            kwargs["annotations"],
            AnnotationOptions(False, False),
        )
        self.assertEqual(
            table.source_path.read_bytes(), self.table_path.read_bytes()
        )
        self.assertEqual(
            (self.results / "motion_correction/alignment.h5").read_bytes(),
            b"alignment",
        )
        self.assertFalse((self.results / "input").exists())

    def test_operator_option_is_rejected_before_staging(self):
        """The removed CLI option fails before any extraction or staging."""
        before = self.snapshot()
        with (
            contextlib.redirect_stderr(io.StringIO()) as stderr,
            self.assertRaises(SystemExit) as raised,
        ):
            self.invoke("--operator", "Ada")
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("unrecognized arguments: --operator", stderr.getvalue())
        self.assertEqual(self.snapshot(), before)
        self.extract.assert_not_called()

    def test_copy_stages_sidecars_references_annotations_and_options(self):
        """Copy input trees and forward split policy/science overrides."""
        self.invoke(
            "--stage-data",
            "copy",
            "--use-rois",
            "--annotations-dir",
            self.annotations,
            "--max-workers",
            2,
            "--max-trials",
            3,
            "--verbose",
            "--denoise-window-s",
            2,
            "--vif",
            1.67,
        )
        table, params = self.extract.call_args.args
        self.assertEqual(table.datadr, self.results / "input")
        for name in ("trial.dat", "trial.meta", "reference.tif"):
            self.assertEqual(
                (table.datadr / name).read_bytes(),
                (self.data / name).read_bytes(),
            )
        self.assertEqual(
            (self.results / "annotations/annotations.h5").read_bytes(),
            b"annotations",
        )
        self.assertEqual(params, BandSiloParams(denoise_window_s=2, vif=1.67))
        self.assertEqual(
            self.extract.call_args.kwargs,
            {
                "execution": ExecutionOptions(2, True, 3),
                "annotations": AnnotationOptions(True, False),
            },
        )

    def test_symlink_staging_uses_correct_roots_without_link_privileges(self):
        """Mock OS link creation while verifying real staging dispatch."""
        with patch.object(Path, "symlink_to", autospec=True) as link:
            self.invoke("--stage-data", "symlink", "--stage-motion", "symlink")
        self.assertEqual(
            link.call_args_list,
            [
                call(
                    self.results / "motion_correction",
                    self.motion,
                    target_is_directory=True,
                ),
                call(
                    self.results / "input",
                    self.data,
                    target_is_directory=True,
                ),
            ],
        )
        table = self.extract.call_args.args[0]
        self.assertEqual(table.datadr, self.results / "input")
        self.assertEqual(table.savedr, self.results)

    def test_existing_directory_and_file_are_never_replaced(self):
        """Existing contents survive even when output is empty or a file."""
        for destination in (self.data, self.table_path, self.scratch):
            with self.subTest(destination=destination):
                self.assert_rejected(
                    FileExistsError, "--results-dir", destination
                )

    def test_existing_and_dangling_results_links_are_never_replaced(self):
        """Check live/dead link preflight without requiring link privilege."""
        exists, is_symlink = Path.exists, Path.is_symlink
        for name, live in (("live", True), ("dead", False)):
            link = self.root / name
            with (
                self.subTest(name=name),
                patch.object(
                    Path,
                    "exists",
                    autospec=True,
                    side_effect=lambda p: live if p == link else exists(p),
                ),
                patch.object(
                    Path,
                    "is_symlink",
                    autospec=True,
                    side_effect=lambda p: p == link or is_symlink(p),
                ),
                patch.object(Path, "unlink", autospec=True) as unlink,
            ):
                self.assert_rejected(FileExistsError, "--results-dir", link)
                unlink.assert_not_called()

    def test_dangling_link_guard_without_os_link_privileges(self):
        """The link predicate is checked even when exists() is false."""
        with (
            patch.object(Path, "exists", return_value=False),
            patch.object(Path, "is_symlink", return_value=True),
            self.assertRaises(FileExistsError),
        ):
            example.validate_destination(self.results, (self.data,))

    def test_output_nested_in_any_input_is_rejected(self):
        """Reject output within raw, motion, or annotation input roots."""
        for source in (self.data, self.motion, self.annotations):
            with self.subTest(source=source):
                self.assert_rejected(
                    ValueError,
                    "--results-dir",
                    source / "new/run",
                    "--annotations-dir",
                    self.annotations,
                )

    def test_missing_and_wrong_kind_paths_fail_before_staging(self):
        """Check input paths and optional scratch roots before staging."""
        for flag, path, error in (
            ("--trial-table", self.root / "missing.h5", FileNotFoundError),
            ("--trial-table", self.data, FileNotFoundError),
            ("--data-dir", self.table_path, NotADirectoryError),
            ("--data-dir", self.root / "absent", FileNotFoundError),
            ("--motion-correction", self.scratch, FileNotFoundError),
            ("--annotations-dir", self.scratch, FileNotFoundError),
            ("--joblib-temp", self.table_path, NotADirectoryError),
            ("--joblib-temp", self.root / "absent", FileNotFoundError),
        ):
            with self.subTest(flag=flag, path=path):
                self.assert_rejected(error, flag, path)
        self.load.assert_not_called()

    def test_missing_registration_metadata_fails_before_staging(self):
        """A loaded table must identify SLAP2 data and alignment filenames."""
        for slap2, motion in (
            (None, {"fn_adata": [["alignment.h5"]]}),
            (Slap2Info(), None),
            (Slap2Info(), {}),
            (Slap2Info(), {"alignHz": 80}),
        ):
            with self.subTest(slap2=slap2, motion=motion):
                self.table.slap2_info = slap2
                self.table.motion_correction = motion
                self.assert_rejected(ValueError)

    def test_unsafe_loaded_names_fail_before_staging(self):
        """Both metadata fields are validated on the loaded table."""
        for field in ("filename", "fn_adata"):
            with self.subTest(field=field):
                self.table.filename = [["trial.dat"]]
                self.table.motion_correction = {"fn_adata": [["alignment.h5"]]}
                if field == "filename":
                    self.table.filename = [["../trial.dat"]]
                else:
                    self.table.motion_correction[field] = [
                        [r"..\alignment.h5"]
                    ]
                self.assert_rejected(ValueError)

    def test_bad_science_and_execution_options_cannot_mutate_run(self):
        """Validate finite positive settings before mkdir/copy/env changes."""
        for flag in (
            "--analyze-hz",
            "--decay-tau-s",
            "--baseline-window-s",
            "--denoise-window-s",
            "--vif",
            "--d-xy",
            "--peak-buffer",
            "--psf-dilation",
            "--max-workers",
            "--max-trials",
        ):
            with self.subTest(flag=flag):
                self.assert_rejected(ValueError, flag, 0)
        for value in ("nan", "inf", "-1"):
            with self.subTest(value=value):
                self.assert_rejected(ValueError, "--analyze-hz", value)

    def test_invalid_roi_policy_fails_before_reading_table(self):
        """Drawing requires ROI use; headless ROI use needs an input file."""
        for flags in (("--interactive",), ("--use-rois",)):
            with self.subTest(flags=flags):
                self.assert_rejected(SystemExit, *flags)
        self.load.assert_not_called()

    def test_joblib_override_is_visible_only_during_execution(self):
        """Restore both absent and pre-existing environment values."""
        for previous in (None, "", "original-scratch"):
            for fails in (False, True):
                with self.subTest(previous=previous, fails=fails):
                    destination = self.root / f"run-{previous}-{fails}"
                    with patch.dict(os.environ):
                        if previous is None:
                            os.environ.pop("JOBLIB_TEMP_FOLDER", None)
                        else:
                            os.environ["JOBLIB_TEMP_FOLDER"] = previous
                        environment = dict(os.environ)

                        def extract(*args, **kwargs):
                            """Observe the worker setting in the backend."""
                            self.assertEqual(
                                os.environ["JOBLIB_TEMP_FOLDER"],
                                str(self.scratch),
                            )
                            if fails:
                                raise RuntimeError("backend failed")
                            return SimpleNamespace(paths=[])

                        self.extract.side_effect = extract
                        flags = (
                            "--results-dir",
                            destination,
                            "--joblib-temp",
                            self.scratch,
                        )
                        if fails:
                            with self.assertRaisesRegex(
                                RuntimeError, "backend failed"
                            ):
                                self.invoke(*flags)
                        else:
                            self.invoke(*flags)
                        self.assertEqual(dict(os.environ), environment)
                        self.assertTrue(
                            (destination / "input_trial_table.h5").is_file()
                        )
                        self.assert_rejected(
                            FileExistsError, "--results-dir", destination
                        )

    def test_no_joblib_flag_preserves_environment_during_extraction(self):
        """Extraction observes the caller's environment without overrides."""
        environment = dict(os.environ)

        def extract(*args, **kwargs):
            """Check the environment at the point workers would be launched."""
            self.assertEqual(dict(os.environ), environment)
            return SimpleNamespace(paths=[])

        self.extract.side_effect = extract
        self.invoke()
        self.assertEqual(dict(os.environ), environment)

    def test_destination_created_after_validation_is_not_overwritten(self):
        """Exclusive mkdir refuses an output that appeared after preflight."""
        args = example.build_parser().parse_args(self.argv())
        run = example.prepare_run(args)
        self.results.mkdir()
        marker = self.results / "keep.txt"
        marker.write_bytes(b"keep")
        with (
            patch.object(example.shutil, "copy2") as copy,
            self.assertRaises(FileExistsError),
        ):
            example.stage_and_extract(run, args)
        copy.assert_not_called()
        self.extract.assert_not_called()
        self.assertEqual(marker.read_bytes(), b"keep")

    def test_stage_directory_guards_existing_destinations_and_invalid_mode(
        self,
    ):
        """Staging never invokes copy/link on an existing or invalid target."""
        with (
            patch.object(example.shutil, "copytree") as copy,
            patch.object(Path, "symlink_to") as link,
        ):
            for mode in ("copy", "symlink"):
                with self.subTest(mode=mode):
                    with self.assertRaises(FileExistsError):
                        example.stage_directory(self.data, self.scratch, mode)
                    with (
                        patch.object(Path, "exists", return_value=False),
                        patch.object(Path, "is_symlink", return_value=True),
                        self.assertRaises(FileExistsError),
                    ):
                        example.stage_directory(self.data, self.results, mode)
            with self.assertRaisesRegex(ValueError, "Unknown staging mode"):
                example.stage_directory(self.data, self.results, "invalid")
        copy.assert_not_called()
        link.assert_not_called()


if __name__ == "__main__":
    unittest.main()
