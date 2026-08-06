"""Tests for the giant_python.cli argument parser and dispatch."""

import unittest
from unittest import mock

from giant_python import cli


class TestBuildParser(unittest.TestCase):
    """build_parser wires the four subcommands and their options."""

    def setUp(self):
        """Build a fresh parser for each test."""
        self.parser = cli.build_parser()

    def test_no_command(self):
        """With no subcommand, command is None."""
        args = self.parser.parse_args([])
        self.assertIsNone(args.command)

    def test_organize(self):
        """organize captures the data and save dirs."""
        args = self.parser.parse_args(["organize", "/data", "/results"])
        self.assertEqual(args.command, "organize")
        self.assertEqual(args.data_dir, "/data")
        self.assertEqual(args.save_dir, "/results")

    def test_annotate_defaults(self):
        """annotate defaults to band scan, no draw, auto interactivity."""
        args = self.parser.parse_args(["annotate", "tt.h5"])
        self.assertEqual(args.command, "annotate")
        self.assertEqual(args.trial_table, "tt.h5")
        self.assertEqual(args.scan_mode, "band")
        self.assertFalse(args.draw_user_rois)
        self.assertIsNone(args.interactive)

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


class TestParamsFromArgs(unittest.TestCase):
    """_params_from_args builds a SiloParams from the namespace."""

    def test_maps_fields(self):
        """Each parsed field maps onto the SiloParams."""
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
            ]
        )
        params = cli._params_from_args(args)
        self.assertEqual(params.scan_mode, "band")
        self.assertTrue(params.draw_user_rois)
        self.assertTrue(params.interactive)
        self.assertEqual(params.operator, "Alice")


class TestMain(unittest.TestCase):
    """main dispatches to handlers and handles the no-command case."""

    def test_no_command_prints_help(self):
        """No subcommand prints help and returns 2."""
        parser = cli.build_parser()
        with mock.patch.object(
            cli, "build_parser", return_value=parser
        ), mock.patch.object(parser, "print_help") as help_:
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


if __name__ == "__main__":
    unittest.main()
