"""CLI for annotation and extraction from registered SLAP2 band inputs.

``giant annotate`` loads or draws ROIs; ``giant extract`` runs BandSILo.
Both delegate to the public workflow services with separate scientific,
execution and annotation options. GUI imports occur only when needed.
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
    for name, kind, help_text in (
        (
            "decay_tau_s",
            float,
            "Shared localization and trace decay time in seconds.",
        ),
        (
            "trace_beta_fraction",
            float,
            "Trial-relative nonnegative event penalty.",
        ),
        ("trace_max_iter", int, "Maximum iterations per transient fit stage."),
        ("trace_tolerance", float, "Transient optimizer tolerance."),
        (
            "trace_event_threshold",
            float,
            "Absolute event threshold for debias support.",
        ),
        (
            "random_seed",
            int,
            "Nonnegative seed for the localization motion order.",
        ),
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=kind,
            default=getattr(BandSiloParams, name),
            help=help_text,
        )
    parser.add_argument(
        "--trace-debias",
        action="store_true",
        help="Refit selected transient events without the L1 penalty.",
    )
    parser.add_argument(
        "--z-tol",
        type=float,
        default=BandSiloParams.z_tol,
        help="Z tolerance for motion-bin selection in microns (default: 1.5).",
    )
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
        BandSiloParams(
            z_tol=args.z_tol,
            decay_tau_s=args.decay_tau_s,
            trace_beta_fraction=args.trace_beta_fraction,
            trace_max_iter=args.trace_max_iter,
            trace_tolerance=args.trace_tolerance,
            trace_debias=args.trace_debias,
            trace_event_threshold=args.trace_event_threshold,
            random_seed=args.random_seed,
        ),
        ExecutionOptions(
            max_workers=args.max_workers,
            verbose=args.verbose,
            max_trials=args.max_trials,
        ),
        AnnotationOptions(
            enabled=args.draw_user_rois or args.command == "annotate",
            interactive=args.interactive,
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


if __name__ == "__main__":
    raise SystemExit(main())
