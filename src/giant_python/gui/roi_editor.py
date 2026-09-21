"""The existing OpenCV rectangle selector and tkinter label prompt.

Accepts prepared band reference geometry and returns in-memory records/masks.
There is no file IO here, and tkinter/cv2 are imported only on drawing paths.
"""

import numpy as np

from ..extraction.band.rois import user_roi_superpixel_lists_from_masks


def _ask_roi_label(
    roi_num: int,
    path_num: int,
    plane_num: int,
) -> str:  # pragma: no cover - interactive tkinter dialog
    """Prompt for one ROI label, defaulting to ROI{n} on cancel or failure."""
    import tkinter as tk
    from tkinter import simpledialog

    text = None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        text = simpledialog.askstring(
            "ROI label",
            f"Label for ROI {roi_num} "
            f"(Path {path_num}, fast-Z plane {plane_num}):",
            parent=root,
        )
        root.destroy()
    except Exception as error:  # noqa: BLE001 - GUI best-effort
        print(f"Could not prompt for ROI label ({error}); using default.")
    if not text or not text.strip():
        return f"ROI{roi_num}"
    return text.strip()


def run_user_roi_selection(
    user_roi_geo: dict, n_dmds: int
) -> dict:  # pragma: no cover - GUI
    """Select ROIs on reference images and return per-DMD selections."""
    import cv2

    out = {
        "user_roi_masks": {},
        "user_roi_superpixels": {},
        "user_roi_labels": {},
        "roi_records": {},
    }
    for dmd_ix in range(n_dmds):
        key = f"DMD{dmd_ix + 1}"
        geo = user_roi_geo[key]
        roi_masks, roi_records = _select_rois_for_dmd(cv2, dmd_ix, geo)
        out["user_roi_masks"][key] = roi_masks
        out["user_roi_superpixels"][key] = (
            user_roi_superpixel_lists_from_masks(
                roi_masks, geo["sp_fastz"], geo["sp_rows"], geo["sp_cols"]
            )
        )
        out["user_roi_labels"][key] = [
            record["label"] for record in roi_records
        ]
        out["roi_records"][key] = roi_records
    return out


def _normalize_plane(
    plane: np.ndarray,
) -> np.ndarray:  # pragma: no cover - display
    """Percentile-normalize a reference plane to an 8-bit image."""
    im = np.nan_to_num(plane, nan=0.0)
    vmin = np.percentile(im, 1)
    vmax = np.percentile(im, 99.5)
    if not np.isfinite(vmin):
        vmin = float(np.nanmin(im)) if np.any(np.isfinite(im)) else 0.0
    if not np.isfinite(vmax):
        vmax = float(np.nanmax(im)) if np.any(np.isfinite(im)) else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    im8 = np.clip((im - vmin) / (vmax - vmin), 0, 1)
    return (im8 * 255).astype(np.uint8)


def _select_rois_for_dmd(cv2, dmd_ix: int, g: dict):  # pragma: no cover - GUI
    """Run the per-DMD interactive loop, returning masks and records."""
    ref = g["ref"]
    num_ref_z = g["num_ref_z"]
    yx_shape = g["yx_shape"]
    best_ch = g["best_ch"]
    z_map = g["z_map"]
    num_fast_z = g["num_fast_z"]
    sp_mask = g["sp_mask"]
    roi_masks = []
    roi_records = []
    roi_union = np.zeros((num_fast_z, *yx_shape), dtype=bool)
    window_name = f"Select user ROI(s) DMD{dmd_ix + 1}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 500)
    has_z_trackbar = num_fast_z > 1
    if has_z_trackbar:
        cv2.createTrackbar("z", window_name, 0, num_fast_z - 1, lambda v: None)
    curr_fz = 0
    while True:
        if has_z_trackbar:
            curr_fz = int(
                np.clip(
                    cv2.getTrackbarPos("z", window_name), 0, num_fast_z - 1
                )
            )
        refz = int(np.clip(z_map[curr_fz], 0, max(0, num_ref_z - 1)))
        im8 = _normalize_plane(ref[best_ch, refz])
        disp = _compose_display(cv2, im8, sp_mask[curr_fz], roi_union[curr_fz])
        cv2.imshow(window_name, disp)
        keycode = cv2.waitKey(50) & 0xFF
        if keycode == ord("e"):
            _edit_rois(
                cv2,
                im8,
                sp_mask[curr_fz],
                curr_fz,
                dmd_ix,
                num_fast_z,
                yx_shape,
                roi_masks,
                roi_union,
                roi_records,
            )
        elif keycode == ord("n"):
            curr_fz = min(curr_fz + 1, num_fast_z - 1)
            if has_z_trackbar:
                cv2.setTrackbarPos("z", window_name, curr_fz)
        elif keycode == ord("p"):
            curr_fz = max(curr_fz - 1, 0)
            if has_z_trackbar:
                cv2.setTrackbarPos("z", window_name, curr_fz)
        elif keycode in (27, ord("q")):
            break
    cv2.destroyWindow(window_name)
    return roi_masks, roi_records


def _compose_display(
    cv2, im8, sp_plane, roi_plane
):  # pragma: no cover - display
    """Overlay the superpixel mask in green and ROI union in red."""
    disp = cv2.cvtColor(im8, cv2.COLOR_GRAY2BGR)
    if np.any(sp_plane):
        sp_color = np.zeros_like(disp)
        sp_color[sp_plane] = (0, 255, 0)
        disp = cv2.addWeighted(disp, 0.75, sp_color, 0.25, 0)
    if roi_plane.any():
        roi_color = np.zeros_like(disp)
        roi_color[roi_plane] = (0, 0, 255)
        disp = cv2.addWeighted(disp, 0.7, roi_color, 0.3, 0)
    return disp


def _edit_rois(
    cv2,
    im8,
    sp_plane,
    curr_fz,
    dmd_ix,
    num_fast_z,
    yx_shape,
    roi_masks,
    roi_union,
    roi_records,
):  # pragma: no cover - GUI
    """Append selected rectangles as polygon records and overlapping masks."""
    edit_name = f"Edit ROIs z={curr_fz + 1}"
    edit_disp = _compose_display(
        cv2, im8, sp_plane, np.zeros(yx_shape, dtype=bool)
    )
    rois = cv2.selectROIs(
        edit_name, edit_disp, showCrosshair=True, fromCenter=False
    )
    cv2.resizeWindow(edit_name, 800, 500)
    if rois is not None and len(rois) > 0:
        for x, y, w, h in rois:
            roi_num = len(roi_masks) + 1
            y_stop, x_stop = y + h, x + w
            mask = np.zeros((num_fast_z, *yx_shape), dtype=bool)
            mask[curr_fz, y:y_stop, x:x_stop] = True
            roi_masks.append(mask)
            roi_union[curr_fz, y:y_stop, x:x_stop] = True
            label = _ask_roi_label(roi_num, dmd_ix + 1, curr_fz + 1)
            position = np.array(
                [[y, x], [y, x + w], [y + h, x + w], [y + h, x]],
                dtype=np.float64,
            )
            roi_records.append(
                {"type": "polygon", "label": label, "position": position}
            )
        print(f"Added {len(rois)} ROI(s) at fast-Z {curr_fz + 1}")
    cv2.destroyWindow(edit_name)
