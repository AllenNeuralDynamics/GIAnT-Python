"""Tests for the giant_python.cli argument parser and dispatch."""

import io
import subprocess
import sys
import unittest
from unittest import mock

from giant_python import cli


class TestBuildParser(unittest.TestCase):
    """build_parser exposes only annotation and extraction with valid modes."""

    def setUp(self):
        """Build a fresh parser for each test."""
        self.parser = cli.build_parser()

    def test_no_command(self):
        """With no subcommand, command is None."""
        args = self.parser.parse_args([])
        self.assertIsNone(args.command)

    def test_removed_commands_are_absent(self):
        """Removed stages have neither CLI help entries nor handlers."""
        help_text = self.parser.format_help()
        for command in ("annotate", "extract"):
            self.assertIn(command, help_text)
        for command in ("organize", "register"):
            with self.subTest(command=command):
                self.assertNotIn(command, help_text)
                self.assertFalse(hasattr(cli, "_cmd_" + command))
                with mock.patch(
                    "sys.stderr", new_callable=io.StringIO
                ) as stderr:
                    with self.assertRaises(SystemExit) as error:
                        self.parser.parse_args([command])
                self.assertEqual(error.exception.code, 2)
                self.assertIn("invalid choice", stderr.getvalue())
                self.assertIn(command, stderr.getvalue())

    def test_annotate_defaults(self):
        """annotate defaults to band scan, no draw, auto interactivity."""
        args = self.parser.parse_args(["annotate", "tt.h5"])
        self.assertEqual(args.command, "annotate")
        self.assertEqual(args.trial_table, "tt.h5")
        self.assertEqual(args.microscope, "slap2")
        self.assertEqual(args.scan_mode, "band")
        self.assertFalse(args.draw_user_rois)
        self.assertIsNone(args.interactive)
        self.assertFalse(args.verbose)

    def test_extract_flags(self):
        """extract parses the draw and headless flags."""
        args = self.parser.parse_args(
            ["extract", "tt.h5", "--draw-user-rois", "--headless"]
        )
        self.assertTrue(args.draw_user_rois)
        self.assertFalse(args.interactive)

    def test_interactive_flag(self):
        """--interactive resolves to True."""
        args = self.parser.parse_args(["annotate", "tt.h5", "--interactive"])
        self.assertTrue(args.interactive)

    def test_supported_routing_choices(self):
        """Both implemented commands accept explicit SLAP2 band routing."""
        for command in ("annotate", "extract"):
            with self.subTest(command=command):
                args = self.parser.parse_args(
                    [
                        command,
                        "tt.h5",
                        "--microscope",
                        "slap2",
                        "--scan-mode",
                        "band",
                    ]
                )
                self.assertEqual(args.microscope, "slap2")
                self.assertEqual(args.scan_mode, "band")


class TestOptionsFromArgs(unittest.TestCase):
    """CLI separates science, execution and annotation configuration."""

    def test_maps_fields(self):
        """Run and annotation flags do not change scientific defaults."""
        args = cli.build_parser().parse_args(
            [
                "extract",
                "tt.h5",
                "--scan-mode",
                "band",
                "--draw-user-rois",
                "--interactive",
                "--operator",
                "Alice",
                "--verbose",
            ]
        )
        params, execution, annotations = cli._options_from_args(args)
        self.assertEqual(params.peakth, 7.0)
        self.assertFalse(hasattr(params, "scan_mode"))
        self.assertTrue(annotations.enabled)
        self.assertTrue(annotations.interactive)
        self.assertEqual(annotations.operator, "Alice")
        self.assertTrue(execution.verbose)
        self.assertEqual(execution.max_workers, 6)
        self.assertIsNone(execution.max_trials)

    def test_explicit_annotation_enables_rois(self):
        """The annotate command itself enables ROI handling."""
        args = cli.build_parser().parse_args(["annotate", "tt.h5"])
        _, _, annotations = cli._options_from_args(args)
        self.assertTrue(annotations.enabled)

    def test_run_limits(self):
        """Execution limits map to validated options."""
        args = cli.build_parser().parse_args(
            ["extract", "tt.h5", "--max-workers", "2", "--max-trials", "3"]
        )
        _, execution, _ = cli._options_from_args(args)
        self.assertEqual(execution.max_workers, 2)
        self.assertEqual(execution.max_trials, 3)


class TestMain(unittest.TestCase):
    """main dispatches to handlers and handles the no-command case."""

    def test_help_does_not_import_extraction_or_gui_backends(self):
        """Fresh-process CLI help exits successfully without heavy imports."""
        code = """
import sys
from giant_python.cli import main
try:
    main(sys.argv[1:])
except SystemExit as error:
    assert error.code == 0, error.code
else:
    raise AssertionError('help did not exit')
blocked = ('torch', 'cv2', 'tkinter', 'slap2_utils',
           'giant_python.extraction.band.workflow',
           'giant_python.gui')
loaded = [name for name in sys.modules
          if any(name == prefix or name.startswith(prefix + '.')
                 for prefix in blocked)]
assert not loaded, loaded
"""
        for args in (["-h"], ["extract", "-h"], ["annotate", "-h"]):
            with self.subTest(args=args):
                result = subprocess.run(
                    [sys.executable, "-c", code, *args],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_no_command_prints_help(self):
        """No subcommand prints help and returns 2."""
        parser = cli.build_parser()
        with (
            mock.patch.object(cli, "build_parser", return_value=parser),
            mock.patch.object(parser, "print_help") as help_,
        ):
            code = cli.main([])
        help_.assert_called_once()
        self.assertEqual(code, 2)

    def test_dispatches_annotate(self):
        """main routes the annotate command to its handler."""
        with mock.patch.object(
            cli, "_cmd_annotate", return_value=0
        ) as handler:
            code = cli.main(["annotate", "tt.h5"])
        handler.assert_called_once()
        self.assertEqual(code, 0)


class TestCommandServices(unittest.TestCase):
    """Handlers delegate to public services and report invalid requests."""

    def test_extract_service(self):
        """The extraction handler passes a filename, not a fake table."""
        with mock.patch(
            "giant_python.pipeline.extract.extract_sources"
        ) as extract:
            code = cli.main(
                [
                    "extract",
                    "curated_v2.h5",
                    "--max-trials",
                    "4",
                    "--headless",
                    "--draw-user-rois",
                ]
            )
        self.assertEqual(code, 0)
        self.assertEqual(extract.call_args.args[0], "curated_v2.h5")
        self.assertEqual(extract.call_args.kwargs["scan_mode"], "band")
        self.assertEqual(extract.call_args.kwargs["execution"].max_trials, 4)
        self.assertTrue(extract.call_args.kwargs["annotations"].enabled)
        self.assertFalse(extract.call_args.kwargs["annotations"].interactive)

    def test_annotation_service(self):
        """Annotation routes through the same public function as Pipeline."""
        with mock.patch(
            "giant_python.pipeline.annotate.annotate_rois"
        ) as annotate:
            code = cli.main(["annotate", "curated_v2.h5", "--headless"])
        self.assertEqual(code, 0)
        self.assertEqual(annotate.call_args.args[0], "curated_v2.h5")
        self.assertTrue(annotate.call_args.kwargs["annotations"].enabled)
        self.assertFalse(annotate.call_args.kwargs["annotations"].interactive)

    def test_removed_commands_fail_before_dispatch(self):
        """Old commands cannot fall through to either implemented handler."""
        with (
            mock.patch.object(cli, "_cmd_annotate") as annotate,
            mock.patch.object(cli, "_cmd_extract") as extract,
        ):
            for argv in (
                ["organize", "/data", "/results"],
                ["register", "curated_v2.h5"],
            ):
                with self.subTest(argv=argv):
                    with mock.patch(
                        "sys.stderr", new_callable=io.StringIO
                    ) as stderr:
                        with self.assertRaises(SystemExit) as error:
                            cli.main(argv)
                    self.assertEqual(error.exception.code, 2)
                    self.assertIn("invalid choice", stderr.getvalue())
                    self.assertIn(argv[0], stderr.getvalue())
        annotate.assert_not_called()
        extract.assert_not_called()

    def test_unsupported_routing_reports_parser_error(self):
        """Unsupported routing is rejected by argparse before dispatch."""
        with (
            mock.patch.object(cli, "_cmd_annotate") as annotate,
            mock.patch.object(cli, "_cmd_extract") as extract,
        ):
            for command in ("extract", "annotate"):
                for flag, mode in (
                    ("--scan-mode", "standard"),
                    ("--scan-mode", "unknown"),
                    ("--microscope", "bergamo"),
                    ("--microscope", "unknown"),
                ):
                    with self.subTest(command=command, flag=flag, mode=mode):
                        with mock.patch(
                            "sys.stderr", new_callable=io.StringIO
                        ) as stderr:
                            with self.assertRaises(SystemExit) as error:
                                cli.main([command, "tt.h5", flag, mode])
                        self.assertEqual(error.exception.code, 2)
                        self.assertIn("invalid choice", stderr.getvalue())
                        self.assertIn(flag, stderr.getvalue())
                        self.assertIn(mode, stderr.getvalue())
        annotate.assert_not_called()
        extract.assert_not_called()

    def test_invalid_execution_limit_is_reported(self):
        """Bad run options fail before the service can load any data."""
        with mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
            with self.assertRaises(SystemExit) as error:
                cli.main(["extract", "tt.h5", "--max-workers", "0"])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("max_workers", stderr.getvalue())

    def test_service_value_error_is_actionable(self):
        """ValueError from either public service becomes a CLI usage error."""
        services = {
            "annotate": "giant_python.pipeline.annotate.annotate_rois",
            "extract": "giant_python.pipeline.extract.extract_sources",
        }
        for command, target in services.items():
            with self.subTest(command=command):
                with (
                    mock.patch(
                        target, side_effect=ValueError("Invalid trial table")
                    ) as service,
                    mock.patch("sys.stderr", new_callable=io.StringIO) as err,
                ):
                    with self.assertRaises(SystemExit) as error:
                        cli.main([command, "tt.h5"])
                service.assert_called_once()
                self.assertEqual(error.exception.code, 2)
                self.assertIn("Invalid trial table", err.getvalue())


if __name__ == "__main__":
    unittest.main()
