"""Standalone band annotation and shared load/draw/headless orchestration.

Raw persistence lives in io.annotations, pure geometry in band.rois, and
toolkit code in gui.roi_editor. No toolkit is loaded on the headless path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ...io.annotations import read_annotations_h5, save_annotations_h5
from ...models.params import (
    AnnotationInput,
    BandParamsInput,
    ExecutionInput,
    resolve_band_options,
)
from ...models.trial_table import TrialTable
from ...progress import log
from . import geometry as geo
from . import inputs
from .inputs import (
    compute_keep_trials,
    load_alignment_data_h5,
    load_trial_table,
)
from .rois import compute_user_roi_geometry, resolve_annotation_records


def first_valid_trial(keep_trials: np.ndarray, dmd_ix: int) -> int:
    """Return this 0-based DMD's first kept trial, or raise ValueError."""
    indexes = np.flatnonzero(np.asarray(keep_trials)[dmd_ix])
    if indexes.size == 0:
        raise ValueError(f"No valid trials for DMD {dmd_ix + 1}")
    return int(indexes[0])


def motion_median_from_adata(a_data: dict) -> tuple:
    """Round internal band motion then take per-axis (row,column,z) medians."""
    return (
        int(np.nanmedian(np.round(a_data["motionDSr"]))),
        int(np.nanmedian(np.round(a_data["motionDSc"]))),
        int(np.nanmedian(np.round(a_data["motionDSz"]))),
    )


def _stdin_is_tty() -> bool:  # pragma: no cover - runtime stdin
    """Check for interactive stdin without importing a GUI toolkit."""
    return bool(getattr(sys.stdin, "isatty", lambda: False)())


def resolve_interactivity(override: Optional[bool] = None) -> bool:
    """Resolve override, then GIANT_HEADLESS, then terminal detection."""
    if override is not None:
        return bool(override)
    if os.environ.get("GIANT_HEADLESS"):
        return False
    return _stdin_is_tty()


def _empty_user_rois(n_dmds: int) -> dict:
    """Create neutral per-path ROI selections without implying annotations."""
    keys = [f"DMD{d + 1}" for d in range(n_dmds)]
    return {
        "user_roi_masks": {k: [] for k in keys},
        "user_roi_superpixels": {k: [] for k in keys},
        "user_roi_labels": {k: [] for k in keys},
        "roi_records": {k: [] for k in keys},
        "annotated": False,
    }


def load_annotations_h5(
    annotations_path: str, n_dmds: int, user_roi_geo: dict
) -> tuple:
    """Read and resolve band annotations; return the historical five-tuple.

    Missing/unreadable files and selections with no usable ROIs return
    (False, {}, {}, {}, {}). Geometry validation is delegated to pure helpers.
    """
    if not os.path.exists(annotations_path):
        return False, {}, {}, {}, {}
    try:
        raw = read_annotations_h5(annotations_path)
    except OSError as error:
        print(f"Could not read {annotations_path}: {error}")
        return False, {}, {}, {}, {}
    return resolve_annotation_records(raw, n_dmds, user_roi_geo)


def resolve_user_rois(
    annotation_dir: Union[str, Path],
    n_dmds: int,
    user_roi_geo: dict,
    *,
    interactive: bool,
    ref_files: Optional[dict] = None,
) -> dict:
    """Load usable annotations, or draw interactively or fail promptly."""
    path = os.path.join(str(annotation_dir), "annotations.h5")
    skip, masks, superpixels, labels, records = load_annotations_h5(
        path, n_dmds, user_roi_geo
    )
    if skip:
        return {
            "user_roi_masks": masks,
            "user_roi_superpixels": superpixels,
            "user_roi_labels": labels,
            "roi_records": records,
            "annotated": True,
        }
    if not interactive:
        raise RuntimeError(
            "Annotations are enabled but no usable ROIs were found at "
            f"{path!r}, and this process is not interactive. Run "
            "`giant annotate <trial_table.h5>` first, or run in an "
            "interactive session (or pass interactive=True) to draw now."
        )
    return _draw_and_save_user_rois(
        annotation_dir, n_dmds, user_roi_geo, ref_files
    )


def _draw_and_save_user_rois(
    annotation_dir: Union[str, Path],
    n_dmds: int,
    user_roi_geo: dict,
    ref_files: Optional[dict],
) -> dict:  # pragma: no cover - GUI
    """Launch the lazy ROI editor and save its records and masks."""
    from ...gui.roi_editor import run_user_roi_selection

    os.makedirs(str(annotation_dir), exist_ok=True)
    selection = run_user_roi_selection(user_roi_geo, n_dmds)
    save_annotations_h5(
        str(annotation_dir),
        selection["roi_records"],
        selection["user_roi_masks"],
        n_dmds,
        ref_files=ref_files,
    )
    selection["annotated"] = True
    return selection


def build_user_roi_geometry(trial_table: dict, lookup: dict) -> tuple:
    """Prepare per-DMD reference geometry and reference-file provenance."""
    user_roi_geo, ref_files = {}, {}
    datadr = str(trial_table["datadr"])
    keep_trials = trial_table["keep_trials"]
    for dmd_ix in range(int(trial_table["n_dmds"])):
        key = f"DMD{dmd_ix + 1}"
        ref_stack, _, ref_file = inputs.load_reference_stack(
            datadr, trial_table["ref_stack"], dmd_ix
        )
        smi = geo.build_subsample_matrix_inds(
            lookup["allSuperPixelIDs"][key], lookup["sparseMaskInds"][key]
        )
        fvt = first_valid_trial(keep_trials, dmd_ix)
        a_data = load_alignment_data_h5(trial_table["fn_adata"][dmd_ix, fvt])
        user_roi_geo[key] = compute_user_roi_geometry(
            ref_stack,
            lookup["fastZ2RefZ"][key],
            smi,
            motion_median_from_adata(a_data),
        )
        ref_files[key] = ref_file
    return user_roi_geo, ref_files


def annotate_band_rois(
    input: Union[str, Path, TrialTable],
    params: BandParamsInput = None,
    *,
    execution: ExecutionInput = None,
    annotations: AnnotationInput = None,
) -> str:  # pragma: no cover - full reference IO and optional GUI
    """Annotate a path or in-memory TrialTable; return annotations.h5's path.

    Existing usable annotations are loaded headlessly. No extraction or
    source fitting occurs and in-memory trial tables are never reloaded.
    Science, execution and annotation inputs are independently validated.
    This explicit step requests annotation regardless of ``enabled``;
    ``interactive`` still controls whether drawing is permitted.
    """
    _, run, policy = resolve_band_options(params, execution, annotations)
    log(f"Loading trial table {input}", run.verbose)
    trial_table = load_trial_table(input)
    trial_table["keep_trials"] = compute_keep_trials(
        trial_table["fn_adata"], trial_table["filename"], trial_table["datadr"]
    )
    annotation_dir = trial_table["annotation_save_dr"]
    lookup = inputs.load_lookup_table(
        Path(trial_table["moco_save_dr"]) / "bandRegLookupTable.h5",
        trial_table["n_dmds"],
    )
    log(
        f"Loading reference geometry for {trial_table['n_dmds']} DMD path(s)",
        run.verbose,
    )
    user_roi_geo, ref_files = build_user_roi_geometry(trial_table, lookup)
    resolve_user_rois(
        annotation_dir,
        trial_table["n_dmds"],
        user_roi_geo,
        interactive=resolve_interactivity(policy.interactive),
        ref_files=ref_files,
    )
    out_path = os.path.join(str(annotation_dir), "annotations.h5")
    log(f"Annotations at {out_path}", run.verbose)
    return out_path
