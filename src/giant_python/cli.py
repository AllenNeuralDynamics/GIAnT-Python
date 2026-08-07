"""Command-line interface: ``giant organize|register|annotate|extract``.

Thin wrapper over the pipeline stages for batch / cluster use. Wired as the
``giant`` console script in pyproject.toml. The ``annotate`` subcommand is the
standalone ROI-annotation step (the BandSILo analog of the GIAnT-MATLAB
annotation capsule): it writes ``annotations.h5``, which a later ``extract``
run consumes without opening a GUI.

The argument parser and the ``SiloParams`` builder are pure and unit-tested;
the individual command handlers drive heavy IO / GUI code and are excluded
from coverage.
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from .models.params import SiloParams


def _add_silo_options(parser: argparse.ArgumentParser) -> None:
    """Attach the shared source-extraction options to a subparser."""
    parser.add_argument(
        "--scan-mode",
        choices=("standard", "band"),
        default="band",
        help="SLAP2 scan mode selecting the extraction backend.",
    )
    parser.add_argument(
        "--draw-user-rois",
        action="store_true",
        help="Prompt for / use manual user ROIs.",
    )
    interactive = parser.add_mutually_exclusive_group()
    interactive.add_argument(
        "--interactive",
        dest="interactive",
        action="store_true",
        default=None,
        help="Force the ROI-drawing GUI when annotations are missing.",
    )
    interactive.add_argument(
        "--headless",
        dest="interactive",
        action="store_false",
        default=None,
        help="Never open a GUI; fail fast if annotations are missing.",
    )
    parser.add_argument(
        "--operator",
        default=SiloParams.operator,
        help="Operator name recorded in the output metadata.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-stage status messages and progress bars.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the ``giant`` argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with ``organize``, ``register``, ``annotate``, and ``extract``
        subcommands. Each subcommand stores its 0-based ``command`` name in the
        parsed namespace.
    """
    parser = argparse.ArgumentParser(prog="giant")
    subparsers = parser.add_subparsers(dest="command")

    organize = subparsers.add_parser(
        "organize", help="Build the trial table from raw data."
    )
    organize.add_argument("data_dir", help="Raw recording directory.")
    organize.add_argument("save_dir", help="Results directory.")

    register = subparsers.add_parser("register", help="Run motion correction.")
    register.add_argument("trial_table", help="Path to trial_table.h5.")

    annotate = subparsers.add_parser(
        "annotate", help="Draw / save user ROIs (standalone step)."
    )
    annotate.add_argument("trial_table", help="Path to trial_table.h5.")
    _add_silo_options(annotate)

    extract = subparsers.add_parser("extract", help="Run source extraction.")
    extract.add_argument("trial_table", help="Path to trial_table.h5.")
    _add_silo_options(extract)

    return parser


def _params_from_args(args: argparse.Namespace) -> SiloParams:
    """Build a :class:`SiloParams` from parsed annotate/extract args."""
    return SiloParams(
        scan_mode=args.scan_mode,
        draw_user_rois=args.draw_user_rois,
        interactive=args.interactive,
        operator=args.operator,
        verbose=args.verbose,
    )


def _cmd_organize(args: argparse.Namespace) -> int:  # pragma: no cover - IO
    """Handle ``giant organize`` (delegates to the trial-table builder)."""
    from .pipeline import TrialTableBuilder

    TrialTableBuilder("slap2").run(args.data_dir, args.save_dir)
    return 0


def _cmd_register(args: argparse.Namespace) -> int:  # pragma: no cover - IO
    """Handle ``giant register`` (delegates to motion correction)."""
    from .pipeline import MotionCorrector

    MotionCorrector.for_microscope("slap2").run(args.trial_table)
    return 0


def _cmd_annotate(
    args: argparse.Namespace,
) -> int:  # pragma: no cover - GUI/IO
    """Handle ``giant annotate`` (standalone ROI annotation)."""
    from .bandsilo.annotate import annotate_band_rois

    annotate_band_rois(args.trial_table, _params_from_args(args))
    return 0


def _cmd_extract(args: argparse.Namespace) -> int:  # pragma: no cover - IO
    """Handle ``giant extract`` (source extraction)."""
    from .pipeline import SourceExtractor

    params = _params_from_args(args)
    SourceExtractor.for_scan_mode(params.scan_mode, params).run(
        args.trial_table
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``giant`` console script.

    Dispatches the ``organize``, ``register``, ``annotate``, and ``extract``
    subcommands to the corresponding pipeline stages.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Process exit code (``2`` when no subcommand is given).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "organize": _cmd_organize,
        "register": _cmd_register,
        "annotate": _cmd_annotate,
        "extract": _cmd_extract,
    }
    if args.command is None:
        parser.print_help()
        return 2
    return handlers[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
