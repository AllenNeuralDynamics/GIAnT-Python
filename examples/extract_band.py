"""Extract registered band data, optionally staging inputs in a fresh run.

Use --help for arguments. No files, environment variables, GUI toolkits, or
GIAnT modules are touched on import. This example copies existing motion
correction; it does not run the unimplemented Python registration stage.

The source table must use relative recording/alignment filenames. Include
raw acquisition sidecars and reference TIFFs in --data-dir. A copied table
retains its original on-disk metadata: effective directory overrides are
made on the loaded model, not saved with the unsupported TrialTable writer.
"""

from __future__ import annotations

import argparse
import os
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Generator

if TYPE_CHECKING:
    from giant_python import (
        AnnotationOptions,
        BandSiloParams,
        ExecutionOptions,
        TrialTable,
    )


@dataclass(frozen=True)
class InputPaths:
    """Resolved input roots and a fresh destination, before any staging."""

    table: Path
    data: Path
    motion: Path
    results: Path
    annotations: Path | None
    joblib_temp: Path | None


@dataclass(frozen=True)
class PreparedRun:
    """Validated paths, loaded metadata and independent run settings."""

    paths: InputPaths
    table: TrialTable
    params: BandSiloParams
    execution: ExecutionOptions
    annotations: AnnotationOptions


def build_parser() -> argparse.ArgumentParser:
    """Describe explicit input roots, safe staging, and split run options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-table", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Raw recordings, sidecars, and reference TIFF directory",
    )
    parser.add_argument(
        "--motion-correction",
        type=Path,
        required=True,
        help="Existing alignment and band registration lookup directory",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="New directory; an existing file, directory, or link is refused",
    )
    parser.add_argument(
        "--stage-data",
        choices=("none", "copy", "symlink"),
        default="none",
        help="Keep raw inputs in place (default), copy, or link the full tree",
    )
    parser.add_argument(
        "--stage-motion",
        choices=("copy", "symlink"),
        default="copy",
        help="Copy existing registration assets (default), or link them",
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        help="Optional existing annotation directory to copy into the run",
    )
    parser.add_argument(
        "--joblib-temp",
        type=Path,
        help="Existing temporary directory; overrides JOBLIB_TEMP_FOLDER",
    )
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--use-rois", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--analyze-hz", type=float, default=100.0)
    parser.add_argument("--decay-tau-s", type=float, default=0.15)
    parser.add_argument("--baseline-window-s", type=float, default=4.0)
    parser.add_argument("--denoise-window-s", type=float, default=1.0)
    parser.add_argument("--vif", type=float, default=1.38)
    parser.add_argument("--d-xy", type=int, default=5)
    parser.add_argument("--peak-buffer", type=int, default=3)
    parser.add_argument("--psf-dilation", type=int, default=17)
    return parser


def require_directory(path: Path) -> Path:
    """Resolve a source directory and fail before staging if it is missing."""
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(path)
    return resolved


def validate_relative_names(values: object, field: str) -> None:
    """Reject names that would bypass the explicitly selected input roots."""
    import numpy as np

    for entry in np.asarray(values, dtype=object).flat:
        if entry is None or str(entry) == "":
            continue
        name = str(entry)
        posix = PurePosixPath(name)
        windows = PureWindowsPath(name)
        if (
            posix.is_absolute()
            or windows.anchor
            or ".." in posix.parts
            or ".." in windows.parts
        ):
            raise ValueError(
                f"{field} must contain relative filenames without '..': "
                f"{name!r}. Rebase the metadata explicitly before staging; "
                "this example will not guess a replacement basename."
            )


def stage_directory(source: Path, destination: Path, mode: str) -> None:
    """Copy or link a directory without replacing any existing destination."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if mode == "copy":
        shutil.copytree(source, destination, symlinks=True)
    elif mode == "symlink":
        destination.symlink_to(source, target_is_directory=True)
    else:
        raise ValueError(f"Unknown staging mode: {mode}")


def validate_annotation_arguments(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    """Reject inconsistent GUI policy before touching inputs."""
    if args.interactive and not args.use_rois:
        parser.error("--interactive requires --use-rois")
    if args.use_rois and not args.interactive and not args.annotations_dir:
        parser.error("Headless --use-rois requires --annotations-dir")


def require_file(path: Path) -> Path:
    """Resolve a required metadata file without creating anything."""
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(path)
    return resolved


def validate_destination(requested: Path, sources: tuple[Path, ...]) -> Path:
    """Refuse existing destinations and recursive staging into input trees."""
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(f"Choose a fresh --results-dir: {requested}")
    results = requested.resolve()
    for source in sources:
        if results.is_relative_to(source):
            raise ValueError("Results must be outside all input directories")
    return results


def validate_paths(args: argparse.Namespace) -> InputPaths:
    """Check all paths without modifying files or the environment."""
    table = require_file(args.trial_table)
    data = require_directory(args.data_dir)
    motion = require_directory(args.motion_correction)
    require_file(motion / "bandRegLookupTable.h5")
    annotations = None
    sources = (data, motion)
    if args.annotations_dir is not None:
        annotations = require_directory(args.annotations_dir)
        require_file(annotations / "annotations.h5")
        sources += (annotations,)
    results = validate_destination(args.results_dir, sources)
    joblib_temp = (
        require_directory(args.joblib_temp)
        if args.joblib_temp is not None
        else None
    )
    return InputPaths(table, data, motion, results, annotations, joblib_temp)


def prepare_run(args: argparse.Namespace) -> PreparedRun:
    """Validate metadata and options before staging or environment changes."""
    paths = validate_paths(args)
    from giant_python import (
        AnnotationOptions,
        BandSiloParams,
        ExecutionOptions,
        TrialTable,
    )

    table = TrialTable.from_h5(paths.table)
    if table.slap2_info is None or not table.motion_correction:
        raise ValueError("A pre-registered SLAP2 band trial table is required")
    if "fn_adata" not in table.motion_correction:
        raise ValueError("The table requires motion_correction/fn_adata")
    validate_relative_names(table.filename, "filename")
    validate_relative_names(
        table.motion_correction["fn_adata"], "motion_correction/fn_adata"
    )
    params = BandSiloParams(
        analyze_hz=args.analyze_hz,
        decay_tau_s=args.decay_tau_s,
        baseline_window_s=args.baseline_window_s,
        denoise_window_s=args.denoise_window_s,
        vif=args.vif,
        d_xy=args.d_xy,
        peak_buffer=args.peak_buffer,
        psf_dilation=args.psf_dilation,
    )
    execution = ExecutionOptions(
        max_workers=args.max_workers,
        max_trials=args.max_trials,
        verbose=args.verbose,
    )
    annotations = AnnotationOptions(
        enabled=args.use_rois,
        interactive=args.interactive,
    )
    return PreparedRun(paths, table, params, execution, annotations)


@contextmanager
def joblib_environment(
    directory: Path | None,
) -> Generator[None, None, None]:
    """Temporarily override worker scratch space only for a validated run."""
    if directory is None:
        yield
        return
    previous = os.environ.get("JOBLIB_TEMP_FOLDER")
    os.environ["JOBLIB_TEMP_FOLDER"] = str(directory)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("JOBLIB_TEMP_FOLDER", None)
        else:
            os.environ["JOBLIB_TEMP_FOLDER"] = previous


def stage_and_extract(run: PreparedRun, args: argparse.Namespace) -> None:
    """Stage validated inputs and pass the relocated model to extraction."""
    from giant_python import extract_band_sources

    paths = run.paths
    results_dir = paths.results
    data_dir = paths.data

    # No overwrite mode and no automatic cleanup: a failed run leaves its
    # partial directory for inspection, never removes an existing tree/link.
    results_dir.mkdir(parents=True, exist_ok=False)
    staged_table = results_dir / "input_trial_table.h5"
    shutil.copy2(paths.table, staged_table)
    stage_directory(
        paths.motion, results_dir / "motion_correction", args.stage_motion
    )
    if args.stage_data != "none":
        staged_data = results_dir / "input"
        stage_directory(data_dir, staged_data, args.stage_data)
        data_dir = staged_data
    if paths.annotations is not None:
        stage_directory(paths.annotations, results_dir / "annotations", "copy")

    # Copying the HDF5 file does not rewrite the paths stored inside it.
    # Pass this overridden model, not staged_table, to make staging effective.
    table = replace(
        run.table,
        datadr=data_dir,
        savedr=results_dir,
        source_path=staged_table,
    )
    summary = extract_band_sources(
        table, run.params, execution=run.execution, annotations=run.annotations
    )
    print(f"Extracted {len(summary.paths)} imaging path(s)")
    print(results_dir / "source_extraction" / "experiment_summary.h5")


def main(argv: list[str] | None = None) -> int:
    """Validate first, then stage a fresh run and extract registered bands."""
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_annotation_arguments(args, parser)
    run = prepare_run(args)
    with joblib_environment(run.paths.joblib_temp):
        stage_and_extract(run, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
