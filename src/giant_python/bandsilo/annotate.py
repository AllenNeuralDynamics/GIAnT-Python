"""Standalone band-scan ROI annotation step (decoupled from extraction).

The GIAnT-MATLAB workflow runs ROI annotation as a *separate* CodeOcean
capsule (``run_capsule`` + ``annotateROIs.m`` /
``loadSessionImagesForAnnotation``) that writes an ``annotations.h5`` the
extraction step later consumes. This
module is the BandSILo analog: :func:`annotate_band_rois` loads just enough of
a ``trial_table.h5`` to display the per-DMD reference images (no
background/rho/NMF compute), launches the ROI selector, and writes
``<savedr>/annotations/annotations.h5``.

The same load-or-draw seam (:func:`resolve_user_rois`) is reused by the
headless extraction driver, so annotation geometry is computed identically in
both places. Extraction never draws unless it resolves to an interactive
session (:func:`resolve_interactivity`); otherwise it fails fast with guidance.

The interactive GUI and disk/``slap2_utils`` IO cannot run headless and are
excluded from coverage; the pure helpers (first-valid-trial, motion medians,
interactivity resolution, and the load/fail-fast branches of
:func:`resolve_user_rois`) are unit-tested.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ..models.params import SiloParams
from . import geometry as geo
from .gui import (
    compute_user_roi_geometry,
    load_annotations_h5,
    run_user_roi_selection,
    save_annotations_h5,
)
from .hdf5 import compute_keep_trials, load_alignment_data_h5, load_trial_table
from .progress import log


def first_valid_trial(keep_trials: np.ndarray, dmd_ix: int) -> int:
    """Return the first kept trial index for a DMD.

    Parameters
    ----------
    keep_trials : ndarray of bool, shape (n_dmds, n_trials)
        Per-DMD/-trial keep mask.
    dmd_ix : int
        0-based DMD index.

    Returns
    -------
    int
        0-based index of the first ``True`` entry in ``keep_trials[dmd_ix]``.

    Raises
    ------
    ValueError
        If no trial is kept for this DMD.
    """
    row = np.asarray(keep_trials)[dmd_ix]
    idxs = np.flatnonzero(row)
    if idxs.size == 0:
        raise ValueError(f"No valid trials for DMD {dmd_ix + 1}")
    return int(idxs[0])


def motion_median_from_adata(a_data: dict) -> tuple:
    """Return the median ``(row, column, z)`` motion from alignment data.

    Mirrors the ``motion_medians`` computation in
    ``extractSLAP2IntegrationSources.py``: the per-axis median of the rounded
    downsampled motion estimates for one trial's alignment data. Both the
    annotate step (drawing) and the extract step (loading) use this so the ROI
    geometry is identical.

    Parameters
    ----------
    a_data : dict
        Alignment data from :func:`giant_python.bandsilo.hdf5\
.load_alignment_data_h5` (``motionDSr``/``motionDSc``/``motionDSz``).

    Returns
    -------
    tuple of int
        ``(median_row, median_column, median_z)``.
    """
    return (
        int(np.nanmedian(np.round(a_data["motionDSr"]))),
        int(np.nanmedian(np.round(a_data["motionDSc"]))),
        int(np.nanmedian(np.round(a_data["motionDSz"]))),
    )


def _stdin_is_tty() -> bool:  # pragma: no cover - depends on runtime stdin
    """Return True if stdin is attached to an interactive terminal."""
    return bool(getattr(sys.stdin, "isatty", lambda: False)())


def resolve_interactivity(override: Optional[bool] = None) -> bool:
    """Decide whether a drawing GUI may be launched in this process.

    Resolution order: an explicit ``override`` wins; otherwise the
    ``GIANT_HEADLESS`` environment variable forces headless; otherwise fall
    back to whether stdin is a TTY. A display probe (e.g. ``DISPLAY`` on Linux)
    can be added here later without changing callers.

    Parameters
    ----------
    override : bool or None
        ``SiloParams.interactive`` (``True``/``False`` force the answer;
        ``None`` auto-detects).

    Returns
    -------
    bool
        True if the process may open the ROI-drawing GUI.
    """
    if override is not None:
        return bool(override)
    if os.environ.get("GIANT_HEADLESS"):
        return False
    return _stdin_is_tty()


def _empty_user_rois(n_dmds: int) -> dict:
    """Return an all-empty per-DMD user-ROI selection for ``n_dmds`` paths."""
    keys = [f"DMD{d + 1}" for d in range(n_dmds)]
    return {
        "user_roi_masks": {k: [] for k in keys},
        "user_roi_superpixels": {k: [] for k in keys},
        "user_roi_labels": {k: [] for k in keys},
        "roi_records": {k: [] for k in keys},
        "annotated": False,
    }


def resolve_user_rois(
    annotation_dir: Union[str, Path],
    n_dmds: int,
    user_roi_geo: dict,
    *,
    interactive: bool,
    ref_files: Optional[dict] = None,
) -> dict:
    """Load ``annotations.h5`` if present, else draw (interactive) or fail.

    This is the shared seam between the standalone annotate step and the
    headless extract step. When ``annotations.h5`` holds at least one ROI it is
    loaded and returned. Otherwise, if ``interactive`` the drawing GUI runs and
    the result is saved; if not, a :class:`RuntimeError` is raised telling the
    user to run ``giant annotate`` first (so batch/cluster ``extract`` jobs
    never block on a GUI).

    Parameters
    ----------
    annotation_dir : str or Path
        Directory holding (or to hold) ``annotations.h5``.
    n_dmds : int
        Number of DMD paths.
    user_roi_geo : dict
        Per-DMD geometry from :func:`compute_user_roi_geometry`.
    interactive : bool
        Whether a GUI may be launched (from :func:`resolve_interactivity`).
    ref_files : dict, optional
        ``{"DMD{N}": reference_file}`` provenance written into the file.

    Returns
    -------
    dict
        ``{"user_roi_masks", "user_roi_superpixels", "user_roi_labels",
        "roi_records", "annotated"}``.

    Raises
    ------
    RuntimeError
        If drawing is required but the process is not interactive.
    """
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
            "draw_user_rois is set but no usable ROIs were found at "
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
) -> dict:  # pragma: no cover - interactive GUI + disk IO
    """Run the ROI selector, persist ``annotations.h5``, and return it."""
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


def build_user_roi_geometry(
    trial_table: dict,
    lookup: dict,
) -> tuple:  # pragma: no cover - reference-stack/aData disk IO
    """Build per-DMD user-ROI geometry and reference files for a session.

    Loads each DMD's reference stack, superpixel index map, and
    first-valid-trial motion medians, then derives the display geometry via
    :func:`compute_user_roi_geometry`. Mirrors the ``soma_geo`` setup block of
    ``main`` in ``extractSLAP2IntegrationSources.py``.

    Parameters
    ----------
    trial_table : dict
        Normalized trial table (provides ``datadr``, ``ref_stack``,
        ``keep_trials``, ``fn_adata``, ``n_dmds``).
    lookup : dict
        Band-registration lookup table from
        :func:`giant_python.bandsilo.geometry.load_lookup_table`.

    Returns
    -------
    tuple
        ``(user_roi_geo, ref_files)`` dicts keyed ``DMD{N}``.
    """
    user_roi_geo, ref_files = {}, {}
    datadr = str(trial_table["datadr"])
    keep_trials = trial_table["keep_trials"]
    for dmd_ix in range(int(trial_table["n_dmds"])):
        key = f"DMD{dmd_ix + 1}"
        ref_stack, _, ref_file = geo.load_reference_stack(
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


def _resolve_params(
    params_in,
) -> SiloParams:  # pragma: no cover - trivial glue
    """Coerce ``params_in`` (None / dict / SiloParams) to a SiloParams."""
    if params_in is None:
        return SiloParams(scan_mode="band")
    if isinstance(params_in, SiloParams):
        return params_in
    return SiloParams(**params_in)


def annotate_band_rois(
    path_to_trial_table: Union[str, Path],
    params_in=None,
) -> str:  # pragma: no cover - full GUI + reference-stack/aData IO
    """Annotate user ROIs for a band-scan session (standalone step).

    The BandSILo analog of the GIAnT-MATLAB annotation capsule: loads the
    ``trial_table.h5`` and per-DMD reference geometry (no extraction compute),
    then loads an existing ``annotations.h5`` or launches the ROI selector and
    writes ``<savedr>/annotations/annotations.h5``. Idempotent -- a session
    that already has annotations short-circuits (matching the MATLAB
    ``if exist(fnAnnH5)`` guard).

    Parameters
    ----------
    path_to_trial_table : str or Path
        Path to the ``trial_table.h5`` produced by BandRegistration.
    params_in : SiloParams or dict, optional
        Parameter overrides (``interactive`` gates the GUI).

    Returns
    -------
    str
        Path to the written (or existing) ``annotations.h5``.
    """
    params = _resolve_params(params_in)
    log(f"Loading trial table {path_to_trial_table}", params.verbose)
    trial_table = load_trial_table(path_to_trial_table)
    trial_table["keep_trials"] = compute_keep_trials(
        trial_table["fn_adata"],
        trial_table["filename"],
        trial_table["datadr"],
    )
    annotation_dir = trial_table["annotation_save_dr"]

    lookup = geo.load_lookup_table(
        Path(trial_table["moco_save_dr"]) / "bandRegLookupTable.h5",
        trial_table["n_dmds"],
    )
    log(
        f"Loading reference geometry for {trial_table['n_dmds']} "
        "DMD path(s)",
        params.verbose,
    )
    user_roi_geo, ref_files = build_user_roi_geometry(trial_table, lookup)

    resolve_user_rois(
        annotation_dir,
        trial_table["n_dmds"],
        user_roi_geo,
        interactive=resolve_interactivity(params.interactive),
        ref_files=ref_files,
    )
    out_path = os.path.join(str(annotation_dir), "annotations.h5")
    log(f"Annotations at {out_path}", params.verbose)
    return out_path
