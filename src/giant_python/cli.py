"""Command-line interface: ``giant annotate|extract``.

Thin wrapper over the pipeline stages for batch / cluster use. Wired as the
``giant`` console script in pyproject.toml. The ``annotate`` subcommand is the
standalone ROI-annotation step (the BandSILo analog of the GIAnT-MATLAB
annotation capsule): it writes ``annotations.h5``, which a later ``extract``
run consumes without opening a GUI.

CLI, standalone functions and Pipeline share the same workflow services.
Scientific defaults are unchanged; execution and annotation flags build
separate options. Heavy acquisition and GUI modules are imported only by
the selected backend when needed.
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from .models.params import (
    AnnotationOptions,
    BandSiloParams,
    ExecutionOptions,
)


def _add_silo_options(parser: argparse.ArgumentParser) -> None:
    """Attach the shared source-extraction options to a subparser."""
    parser.add_argument(
        "--microscope",
        choices=("slap2",),
        default="slap2",
        help="Acquisition type (currently only SLAP2 band is implemented).",
    )
    parser.add_argument(
        "--scan-mode",
        choices=("band",),
        default="band",
        help="Supported scan mode.",
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
        default=AnnotationOptions.operator,
        help="Operator name recorded in the output metadata.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-stage status messages and progress bars.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=ExecutionOptions.max_workers,
        help="Positive worker count (default: 6).",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Positive debug trial limit per DMD (default: all trials).",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the ``giant`` argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with ``annotate`` and ``extract``
        subcommands. Each subcommand stores its ``command`` name in the
        parsed namespace.
    """
    parser = argparse.ArgumentParser(prog="giant")
    subparsers = parser.add_subparsers(dest="command")

    annotate = subparsers.add_parser(
        "annotate", help="Draw / save user ROIs (standalone step)."
    )
    annotate.add_argument("trial_table", help="Path to trial_table.h5.")
    _add_silo_options(annotate)

    extract = subparsers.add_parser("extract", help="Run source extraction.")
    extract.add_argument("trial_table", help="Path to trial_table.h5.")
    _add_silo_options(extract)

    return parser


def _options_from_args(
    args: argparse.Namespace,
) -> tuple[BandSiloParams, ExecutionOptions, AnnotationOptions]:
    """Build independent science, execution and annotation configuration."""
    return (
        BandSiloParams(),
        ExecutionOptions(
            max_workers=args.max_workers,
            verbose=args.verbose,
            max_trials=args.max_trials,
        ),
        AnnotationOptions(
            enabled=args.draw_user_rois or args.command == "annotate",
            interactive=args.interactive,
            operator=args.operator,
        ),
    )


def _cmd_annotate(
    args: argparse.Namespace,
) -> int:
    """Handle ``giant annotate`` (standalone ROI annotation)."""
    from .pipeline.annotate import annotate_rois

    params, execution, annotations = _options_from_args(args)
    annotate_rois(
        args.trial_table,
        params,
        microscope=args.microscope,
        scan_mode=args.scan_mode,
        execution=execution,
        annotations=annotations,
    )
    return 0


def _cmd_extract(args: argparse.Namespace) -> int:
    """Handle ``giant extract`` (source extraction)."""
    from .pipeline.extract import extract_sources

    params, execution, annotations = _options_from_args(args)
    extract_sources(
        args.trial_table,
        params,
        microscope=args.microscope,
        scan_mode=args.scan_mode,
        execution=execution,
        annotations=annotations,
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``giant`` console script.

    Dispatches ``annotate`` and ``extract`` to their workflow services.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Process exit code (``2`` when no subcommand is given).

    Raises
    ------
    SystemExit
        Invalid options and unsupported workflows print an argparse error
        and exit with code two, without claiming the operation succeeded.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "annotate": _cmd_annotate,
        "extract": _cmd_extract,
    }
    if args.command is None:
        parser.print_help()
        return 2
    try:
        return handlers[args.command](args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
