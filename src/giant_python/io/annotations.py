"""Raw ROI records, masks and provenance IO without band geometry or UI."""

from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np

from .hdf5 import load_struct_h5


def _h5_scalar_str(ds: h5py.Dataset) -> str:
    """Read a scalar string, including legacy one-element string arrays."""
    value = ds[()]
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def read_annotations_h5(path: Union[str, Path]) -> dict:
    """Read raw Path groups including n_rois, masks, records and provenance.

    No geometry validation or superpixel mapping occurs here. Missing and
    unreadable files raise OSError; annotation orchestration decides policy.
    The shared decoder honors row_major and leaves coordinate indexes alone.
    """
    return load_struct_h5(path)


def save_annotations_h5(
    dr: str,
    roi_records_by_dmd: dict,
    user_roi_masks: dict,
    n_dmds: int,
    ref_files: Optional[dict] = None,
) -> str:
    """Write dr/annotations.h5 and return its path, preserving the band schema.

    Rectangles are polygon records with zero-based [y,x] vertices. Masks are
    (fast-Z,Y,X), may overlap, and are stored as compressed uint8. The caller
    owns directory creation and the decision to replace an existing file.
    """
    path = str(Path(dr) / "annotations.h5")
    str_dt = h5py.string_dtype(encoding="utf-8")
    ref_files = ref_files or {}
    with h5py.File(path, "w") as f:
        f["row_major"] = 1
        f["coords_zero_indexed"] = 1
        for dmd_ix in range(n_dmds):
            key = f"DMD{dmd_ix + 1}"
            group = f.create_group(f"Path{dmd_ix + 1}")
            ref_file = ref_files.get(key)
            group.create_dataset(
                "dr",
                data=str(Path(ref_file).parent) if ref_file else "",
                dtype=str_dt,
            )
            group.create_dataset(
                "fn",
                data=Path(ref_file).name if ref_file else "",
                dtype=str_dt,
            )
            records = roi_records_by_dmd.get(key, [])
            group.create_dataset("n_rois", data=len(records))
            mask_list = user_roi_masks.get(key, [])
            for i, record in enumerate(records):
                _write_roi_group(group, i, record, mask_list, str_dt)
    return path


def _write_roi_group(grp, i: int, rec: dict, mask_list: list, str_dt) -> None:
    """Write one roi_### group with the established optional-field policy."""
    group = grp.create_group(f"roi_{i:03d}")
    roi_type = rec.get("type", "polygon")
    group.create_dataset("type", data=roi_type, dtype=str_dt)
    group.create_dataset(
        "label", data=rec.get("label", f"ROI{i + 1}"), dtype=str_dt
    )
    if i < len(mask_list):
        group.create_dataset(
            "mask",
            data=np.asarray(mask_list[i]).astype(np.uint8),
            compression="gzip",
            shuffle=True,
        )
    if roi_type == "polygon" and "position" in rec:
        group.create_dataset(
            "position", data=np.asarray(rec["position"], dtype=np.float64)
        )
