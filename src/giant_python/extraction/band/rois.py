"""Pure band ROI geometry, annotation validation, and superpixel mapping."""

import numpy as np


def user_roi_superpixel_lists_from_masks(
    roi_masks: list,
    sp_fastz: np.ndarray,
    sp_rows: np.ndarray,
    sp_cols: np.ndarray,
) -> list:
    """Return indices of superpixels in each ROI, preserving overlap/order."""
    return [
        np.flatnonzero(np.asarray(mask)[sp_fastz, sp_rows, sp_cols])
        for mask in roi_masks
    ]


def compute_user_roi_geometry(
    ref: np.ndarray,
    fastz_to_refz: np.ndarray,
    subsample_matrix_inds: np.ndarray,
    motion_median: tuple,
) -> dict:
    """Prepare reference display and mask geometry with internal band offsets.

    Reference is (channels,z,Y,X); fast-Z lookup is 1-based. Reference pixel
    decomposition retains the original column-major convention and indexes.
    """
    avg_motion_r, avg_motion_c, avg_motion_z = motion_median
    num_ref_z = ref.shape[1]
    yx_shape = (ref.shape[2], ref.shape[3])
    plane = yx_shape[0] * yx_shape[1]
    ch_means = [np.nanmean(ref[c]) for c in range(ref.shape[0])]
    best_ch = int(np.argmax(ch_means)) if len(ch_means) > 0 else 0
    z_map = np.array(fastz_to_refz + avg_motion_z).reshape(-1) - 1
    num_fast_z = z_map.shape[0]
    ref_pix = subsample_matrix_inds[:, 0]
    sp_fastz = ref_pix // plane
    sp_cols = avg_motion_c + (ref_pix - sp_fastz * plane) // yx_shape[0]
    sp_rows = avg_motion_r + ref_pix % yx_shape[0]
    sp_mask = np.zeros((num_fast_z, *yx_shape), dtype=bool)
    valid = (
        (sp_rows >= 0)
        & (sp_rows < yx_shape[0])
        & (sp_cols >= 0)
        & (sp_cols < yx_shape[1])
        & (sp_fastz >= 0)
        & (sp_fastz < num_fast_z)
    )
    if np.any(valid):
        sp_mask[sp_fastz[valid], sp_rows[valid], sp_cols[valid]] = True
    return {
        "ref": ref,
        "num_ref_z": num_ref_z,
        "yx_shape": yx_shape,
        "best_ch": best_ch,
        "z_map": z_map,
        "num_fast_z": num_fast_z,
        "sp_fastz": sp_fastz,
        "sp_rows": sp_rows,
        "sp_cols": sp_cols,
        "sp_mask": sp_mask,
    }


def _empty_path_selection(dmd_key: str, out: dict) -> None:
    """Record a neutral selection for one invalid/missing path."""
    for values in out.values():
        values[dmd_key] = []


def _read_roi_records(grp: dict, expected: tuple) -> tuple:
    """Validate decoded ROI records against the current band mask geometry.

    Takes a plain decoded mapping, NOT an HDF5 handle. Missing masks become
    empty masks, missing records are skipped, and bad shapes reject the path.
    """
    n_rois = int(np.asarray(grp.get("n_rois", 0)).reshape(-1)[0])
    roi_masks, labels, recs = [], [], []
    for i in range(n_rois):
        record = grp.get(f"roi_{i:03d}")
        if record is None:
            continue
        label = str(record.get("label", f"ROI{i+1}"))
        roi_type = str(record.get("type", "polygon"))
        if "mask" in record:
            mask = np.asarray(record["mask"])
            if mask.shape != expected:
                return roi_masks, labels, recs, True
            roi_masks.append(mask > 0)
        else:
            roi_masks.append(np.zeros(expected, dtype=bool))
        labels.append(label)
        rec: dict[str, str | np.ndarray] = {"type": roi_type, "label": label}
        if "position" in record:
            rec["position"] = np.asarray(record["position"])
        recs.append(rec)
    return roi_masks, labels, recs, False


def _load_one_path(
    raw: dict, dmd_ix: int, user_roi_geo: dict, out: dict
) -> bool:
    """Resolve one decoded path against current geometry, without disk IO."""
    key = f"DMD{dmd_ix + 1}"
    path_key = f"Path{dmd_ix + 1}"
    geo = user_roi_geo[key]
    expected = (geo["num_fast_z"], geo["yx_shape"][0], geo["yx_shape"][1])
    group = raw.get(path_key)
    if group is None:
        _empty_path_selection(key, out)
        return False
    masks, labels, records, bad_shape = _read_roi_records(group, expected)
    if bad_shape:
        print(
            f"annotations.h5: {path_key} ROI mask shape mismatch "
            f"(expected {expected}); treating this path as no selection."
        )
        _empty_path_selection(key, out)
        return False
    out["user_roi_masks"][key] = masks
    out["user_roi_superpixels"][key] = user_roi_superpixel_lists_from_masks(
        masks, geo["sp_fastz"], geo["sp_rows"], geo["sp_cols"]
    )
    out["user_roi_labels"][key] = labels
    out["roi_records"][key] = records
    if len(records) > 0:
        print(
            f"Loaded {len(records)} ROI(s) for {path_key} from annotations.h5"
        )
        return True
    return False


def resolve_annotation_records(
    raw: dict, n_dmds: int, user_roi_geo: dict
) -> tuple:
    """Return selection status, masks, superpixels, labels, and ROI records."""
    out = {
        "user_roi_masks": {},
        "user_roi_superpixels": {},
        "user_roi_labels": {},
        "roi_records": {},
    }
    any_valid = False
    for dmd_ix in range(n_dmds):
        if _load_one_path(raw, dmd_ix, user_roi_geo, out):
            any_valid = True
    if not any_valid:
        return False, {}, {}, {}, {}
    return (
        True,
        out["user_roi_masks"],
        out["user_roi_superpixels"],
        out["user_roi_labels"],
        out["roi_records"],
    )
