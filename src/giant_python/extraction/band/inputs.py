"""Band runtime input preparation and displacement-to-offset normalization.

Shared IO is faithful to saved displacement. Only this backend boundary
negates offline motion into the reference offsets used by band numerics.
Lookup/reference/PSF loading and asset discovery also belong at this boundary;
PSF padding and cropping delegate to pure geometry functions.
"""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from ...io.hdf5 import load_struct_h5, to_serializable
from ...io.slap2 import load_alignment_data_h5 as read_alignment_data_h5
from ...io.tiff import read_tiff
from ...models.trial_table import TrialTable
from .geometry import build_combined_psf, threshold_and_crop_psf


def load_lookup_table(path: Union[str, Path], n_dmds: int) -> dict:
    """Load the band-registration lookup table (``bandRegLookupTable.h5``).

    Parameters
    ----------
    path : str or Path
        Path to the lookup-table HDF5 file.
    n_dmds : int
        Number of DMD paths.

    Returns
    -------
    dict
        ``{"allSuperPixelIDs": {...}, "sparseMaskInds": {...},
        "fastZ2RefZ": {...}}``, each an inner dict keyed ``DMD{N}``.

    Raises
    ------
    ValueError
        If a requested DMD has no Path/DMD lookup group.
    """
    lt = load_struct_h5(path)
    all_super_pixel_ids = {}
    sparse_mask_inds = {}
    fastz_to_refz = {}
    for d in range(n_dmds):
        p = lt.get(f"Path{d + 1}", lt.get(f"DMD{d + 1}"))
        if p is None:
            raise ValueError(
                f"Missing lookup-table group Path{d + 1} or DMD{d + 1}"
            )
        key = f"DMD{d + 1}"
        all_super_pixel_ids[key] = (
            np.asarray(p["allSuperPixelIDs"]).reshape(-1, 1).astype(np.int32)
        )
        sparse_mask_inds[key] = np.asarray(p["sparseMaskInds"]).astype(
            np.int32
        )
        fastz_to_refz[key] = (
            np.asarray(p["fastZ2RefZ"]).reshape(-1, 1).astype(np.int32)
        )
    return {
        "allSuperPixelIDs": all_super_pixel_ids,
        "sparseMaskInds": sparse_mask_inds,
        "fastZ2RefZ": fastz_to_refz,
    }


def _ref_stack_group(ref_stack: dict, dmd_ix: int) -> dict:
    """Return a Path/DMD reference subgroup; raise ValueError if missing."""
    group = ref_stack.get(
        f"Path{dmd_ix + 1}", ref_stack.get(f"DMD{dmd_ix + 1}")
    )
    if group is None:
        raise ValueError(
            f"Missing reference-stack group Path{dmd_ix + 1} "
            f"or DMD{dmd_ix + 1}"
        )
    return group


def find_reference_file(
    datadr: Union[str, Path], dmd_ix: int
) -> Optional[str]:
    """Find the REFERENCE tif for a DMD under ``datadr`` (recursive glob).

    Parameters
    ----------
    datadr : str or Path
        Raw-data directory.
    dmd_ix : int
        0-based DMD index.

    Returns
    -------
    str or None
        Path to the first matching REFERENCE tif, or ``None`` if none found.
    """
    patterns = (
        f"**/*DMD{dmd_ix + 1}_CONFIG2-REFERENCE*",
        f"**/*DMD{dmd_ix + 1}-REFERENCE*",
    )
    for pattern in patterns:
        matches = list(Path(datadr).glob(pattern))
        if matches:
            return str(matches[0])
    return None


def load_reference_stack(
    datadr: Union[str, Path],
    ref_stack_meta: dict,
    dmd_ix: int,
) -> Tuple[Optional[np.ndarray], np.ndarray, Optional[str]]:
    """Load the band reference stack for one DMD.

    Reads the REFERENCE tif from ``datadr`` but takes the channel list from the
    trial table's embedded ref_stack metadata. The raw counts are scaled by
    ``1/100`` and reshaped to ``[channels, z, y, x]``.

    Parameters
    ----------
    datadr : str or Path
        Raw-data directory.
    ref_stack_meta : dict
        The trial table's ``ref_stack`` sub-struct.
    dmd_ix : int
        0-based DMD index.

    Returns
    -------
    ref_stack : ndarray or None
        ``[channels, z, y, x]`` stack, or ``None`` if no REFERENCE file found.
    channels : ndarray
        Channel indices for this DMD.
    ref_file : str or None
        The REFERENCE file used, or ``None``.

    Raises
    ------
    ValueError
        If the requested DMD has no Path/DMD reference metadata group.
    """
    grp = _ref_stack_group(ref_stack_meta, dmd_ix)
    channels = np.asarray(grp["channels"]).reshape(-1)
    num_channels = len(channels)

    ref_file = find_reference_file(datadr, dmd_ix)
    if ref_file is None:
        return None, channels, None

    ref = read_tiff(ref_file) / 100
    ref = ref.reshape(-1, num_channels, ref.shape[1], ref.shape[2])
    ref = ref.transpose(1, 0, 2, 3)
    return ref, channels, ref_file


def default_psf(dilation: int) -> np.ndarray:
    """Load a bundled default PSF template (``assets/psfs/dil-NN.tif``).

    Parameters
    ----------
    dilation : int
        Dilation size selecting ``dil-{dilation:02d}.tif`` (e.g. ``17``).

    Returns
    -------
    ndarray of float32
        The PSF template image.
    """
    name = f"dil-{int(dilation):02d}.tif"
    resource = (
        resources.files("giant_python")
        .joinpath("assets")
        .joinpath("psfs")
        .joinpath(name)
    )
    with resources.as_file(resource) as p:
        return read_tiff(str(p)).astype(np.float32)


def load_psf(dilation: int, n_dmds: int) -> dict:
    """Load per-DMD PSFs from bundled assets, thresholded and cropped.

    Each DMD uses the same ``dil-NN.tif`` template from
    ``assets/psfs/``, center-padded to a common size when shapes differ.

    Parameters
    ----------
    dilation : int
        Dilation size selecting the bundled PSF template.
    n_dmds : int
        Number of DMD paths.

    Returns
    -------
    dict
        ``{"DMD{N}": psf2d}`` thresholded/cropped PSFs.
    """
    psfs = [default_psf(dilation) for _ in range(n_dmds)]
    psf_combined = build_combined_psf(psfs)
    return {
        f"DMD{i + 1}": threshold_and_crop_psf(psf_combined[i])
        for i in range(n_dmds)
    }


def load_alignment_data_h5(path: Union[str, Path]) -> dict:
    """Read band alignment, negating saved offline shifts exactly once.

    Online shifts and frame indexes are unchanged. Negation allocates arrays
    and does not modify values owned by the raw reader.
    """
    data = read_alignment_data_h5(path)
    return {
        **data,
        **{
            key: None if data[key] is None else -data[key]
            for key in ("motionDSr", "motionDSc", "motionDSz")
        },
    }


def _resolve_fn_adata(fn_adata: np.ndarray, moco_save_dr: str) -> np.ndarray:
    """Resolve alignment basenames, preserving empty entries and array rank."""
    n_dmds, n_trials = fn_adata.shape
    out = np.empty((n_dmds, n_trials), dtype=object)
    for d in range(n_dmds):
        for t in range(n_trials):
            entry = fn_adata[d, t]
            out[d, t] = (
                ""
                if entry in (None, "")
                else os.path.join(moco_save_dr, str(entry))
            )
    return out


def load_trial_table(path: Union[str, Path, TrialTable]) -> dict:
    """Prepare a structural band view from a path OR an already loaded table.

    Never reload a TrialTable. This does not probe files or read alignments;
    callers retain control of ``compute_keep_trials`` and ``read_align_info``.
    The view carries the original model as ``trial_table`` and its provenance
    as ``source_path``; all historical flattened runtime keys are retained.
    """
    tt = path if isinstance(path, TrialTable) else TrialTable.from_h5(path)
    slap2 = tt.slap2_info
    motion_correction = tt.motion_correction or {}
    if slap2 is None or "fn_adata" not in motion_correction:
        raise ValueError(
            "load_trial_table expects a registered SLAP2 band trial table "
            "(slap2_info + motion_correction/fn_adata); "
            f"got {str(path)!r}."
        )
    datadr = str(tt.datadr) if tt.datadr is not None else ""
    result_dr = str(tt.savedr) if tt.savedr is not None else ""
    moco_save_dr = os.path.join(result_dr, "motion_correction")
    align_params = {
        key: to_serializable(val)
        for key, val in (
            motion_correction.get("align_params", {}) or {}
        ).items()
    }
    filenames = np.atleast_2d(np.asarray(tt.filename, dtype=object))
    first_line = np.atleast_2d(np.asarray(slap2.first_line))
    last_line = np.atleast_2d(np.asarray(slap2.last_line))
    fn_adata = np.atleast_2d(
        np.asarray(motion_correction["fn_adata"], dtype=object)
    )
    n_dmds, n_trials = filenames.shape
    return {
        "trial_table": tt,
        "source_path": tt.source_path,
        "datadr": datadr,
        "savedr": result_dr,
        "moco_save_dr": moco_save_dr,
        "annotation_save_dr": os.path.join(result_dr, "annotations"),
        "src_extr_save_dr": os.path.join(result_dr, "source_extraction"),
        "align_params": align_params,
        "n_dmds": int(n_dmds),
        "n_trials": int(n_trials),
        "filename": filenames,
        "first_line": first_line,
        "last_line": last_line,
        "fn_adata": _resolve_fn_adata(fn_adata, moco_save_dr),
        "ref_stack": slap2.ref_stack or {},
    }


def compute_keep_trials(
    fn_adata_abs: np.ndarray,
    filename: np.ndarray,
    datadr: Union[str, Path],
) -> np.ndarray:
    """Keep trials only when both alignment and raw source files exist."""
    fn_adata_abs = np.atleast_2d(np.asarray(fn_adata_abs, dtype=object))
    filename = np.atleast_2d(np.asarray(filename, dtype=object))
    n_dmds, n_trials = fn_adata_abs.shape
    keep = np.ones((n_dmds, n_trials), dtype=bool)
    for d in range(n_dmds):
        for t in range(n_trials):
            adata = fn_adata_abs[d, t]
            src = filename[d, t]
            src_path = os.path.join(str(datadr), str(src)) if src else ""
            if (
                not adata
                or not os.path.exists(str(adata))
                or not src
                or not os.path.exists(src_path)
            ):
                keep[d, t] = False
    return keep


def read_align_info(
    fn_adata_abs: np.ndarray,
    keep_trials: np.ndarray,
    n_dmds: int,
) -> tuple:
    """Read first-kept-trial alignHz per DMD and the last channel count."""
    fn_adata_abs = np.atleast_2d(np.asarray(fn_adata_abs, dtype=object))
    keep_trials = np.atleast_2d(np.asarray(keep_trials))
    align_hz = {}
    num_channels = None
    for d in range(n_dmds):
        valid = np.flatnonzero(keep_trials[d])
        if valid.size == 0:
            continue
        a_data = load_alignment_data_h5(fn_adata_abs[d, valid[0]])
        align_hz[f"DMD{d + 1}"] = a_data["alignHz"]
        num_channels = a_data["numChannels"]
    return align_hz, num_channels
