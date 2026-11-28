"""``giant spine``: spine-size readout from a SILo run.

``giant spine readout RUN_DIR --table TABLE [--baseline-dir DIR --moco-dir
DIR]`` reads out every source's (and user ROI's) size in every trial and
writes a ``spine_summary_<key>.h5`` file (:mod:`giant_python.spine`);
``giant spine images`` only builds the network's per-trial F0 images (the
slow part), which ``readout`` then reuses from its cache.
"""

from __future__ import annotations

import argparse
import json
import sys

SPINE_COMMANDS = ("spine",)


def _add_common(p: argparse.ArgumentParser) -> None:
    """Options shared by the spine commands."""
    p.add_argument(
        "run_dir", help="Folder with experiment_summary.h5 (a SILo run)."
    )
    p.add_argument(
        "--table", required=True, help="The session's trial_table.h5."
    )
    p.add_argument(
        "--baseline-dir",
        default=None,
        help="Folder of the F0 network's <stem>_BASELINE-80Hz.h5 files.",
    )
    p.add_argument(
        "--moco-dir",
        default=None,
        help="Motion-correction folder the baselines were predicted from "
        "(its ALIGNMENTDATA files give the discarded frames).",
    )
    p.add_argument("--out-dir", required=True, help="Folder of the outputs.")
    p.add_argument(
        "--cache-dir",
        default=None,
        help="Cache of the network's F0 images (default <out-dir>/cache).",
    )
    p.add_argument(
        "--paths",
        default=None,
        help="Imaging paths, e.g. 1,2 (default: all in the run).",
    )
    p.add_argument(
        "--trim-s",
        type=float,
        default=1.0,
        help="Seconds left out at each end of a trial (default 1).",
    )
    p.add_argument(
        "--align-hz",
        type=float,
        default=None,
        help="Frame rate of the registered movies (default: the run's "
        "alignHz, else 80).",
    )
    p.add_argument(
        "--motion-thresh",
        type=float,
        default=None,
        help="SiLo's motion threshold for discarded frames (default: the "
        "run's motionThresh, else 2).",
    )
    p.add_argument(
        "--discard-initial-s",
        type=float,
        default=None,
        help="Seconds discarded at the start of each trial (default: the "
        "run's discardInitial_s, else 0).",
    )
    p.add_argument(
        "--affine-base",
        type=int,
        choices=(0, 1),
        default=None,
        help="Index base of the run's affine trial maps: 0 (GIAnT-Python) "
        "or 1 (GIAnT-MATLAB); needed only for affine alignment.",
    )
    p.add_argument(
        "--gain-work-dir",
        default=None,
        help="Work folder of a run with the 'gain' baseline mode (N1): adds "
        "the network's F0 rescaled by the solver's per-pixel gains.",
    )
    p.add_argument("--quiet", action="store_true", help="No progress lines.")


def add_spine_commands(subparsers) -> None:
    """Add ``giant spine`` and its subcommands."""
    spine = subparsers.add_parser(
        "spine", help="Spine-size readout from a SILo run."
    )
    sub = spine.add_subparsers(dest="spine_command")
    readout = sub.add_parser(
        "readout",
        help="Read out sizes in fixed windows; write spine_summary_<key>.h5.",
    )
    _add_common(readout)
    readout.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite an output that exists with the same key.",
    )
    images = sub.add_parser(
        "images",
        help="Only build the network's per-trial F0 images (cached).",
    )
    _add_common(images)


def run_spine_command(args: argparse.Namespace) -> int:
    """Run a ``giant spine`` subcommand; print a JSON report."""
    from .spine.pipeline import make_spine_summary

    if args.spine_command not in ("readout", "images"):
        raise ValueError("give a spine subcommand: readout or images")
    if args.spine_command == "images" and args.baseline_dir is None:
        raise ValueError("giant spine images needs --baseline-dir")
    paths = (
        [int(x) for x in args.paths.split(",") if x.strip()]
        if args.paths
        else None
    )

    def say(msg: str) -> None:
        """Print a progress line to stderr."""
        if not args.quiet:
            print(msg, file=sys.stderr, flush=True)

    report = make_spine_summary(
        args.run_dir,
        args.table,
        args.out_dir,
        baseline_dir=args.baseline_dir,
        moco_dir=args.moco_dir,
        paths=paths,
        trim_s=args.trim_s,
        align_hz=args.align_hz,
        motion_thresh=args.motion_thresh,
        discard_initial_s=args.discard_initial_s,
        cache_dir=args.cache_dir,
        overwrite=getattr(args, "overwrite", False),
        images_only=args.spine_command == "images",
        affine_base=args.affine_base,
        gain_work_dir=args.gain_work_dir,
        progress=say,
    )
    print(json.dumps(report, indent=1, default=str))
    return 0
