"""Parameter GUI, user-ROI annotation GUI, and annotation HDF5 IO.

Ported from ``extractSLAP2IntegrationSources.py`` (parameter dialog ref
1013-1152, annotation helpers/IO ref 1155-1357, user-ROI selection GUI ref
1567-1741). The interactive dialogs (``tkinter`` parameter form, ``tkinter``
label prompt, ``cv2`` ROI selector) import their GUI toolkits lazily so that
importing this module never requires a display; they are excluded from coverage
because they cannot run headless. The pure geometry/superpixel helpers and the
annotation reader/writer are fully testable and carry the numerical contract.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from ..models import SiloParams

# Mapping of parameter-GUI fields to SiloParams attributes. ``sparse_fac`` is
# handled separately (the GUI exposes its natural log).
_PARAM_GUI_FIELDS = (
    ("analyze_hz", "Analyze Hz:"),
    ("decay_tau_s", "Decay Tau (s):"),
    ("baseline_window_s", "Baseline Window (s):"),
    ("denoise_window_s", "Denoise Window (s):"),
    ("vif", "Variance Inflation Factor (VIF):"),
    ("d_xy", "dXY:"),
    ("peakth", "Peak Threshold:"),
    ("peak_buffer", "Peak Buffer:"),
    ("max_workers", "Max Workers:"),
    ("operator", "Operator:"),
)


def _h5_scalar_str(ds: h5py.Dataset) -> str:
    """Read an h5 scalar string dataset as a python ``str``."""
    v = ds[()]
    if isinstance(v, np.ndarray):
        v = v.reshape(-1)[0]
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return str(v)


def user_roi_superpixel_lists_from_masks(
    roi_masks: list,
    sp_fastz: np.ndarray,
    sp_rows: np.ndarray,
    sp_cols: np.ndarray,
) -> list:
    """Return one superpixel-index list per user ROI, from boolean masks.

    ROIs may overlap: a superpixel inside several ROIs appears in each list.

    Parameters
    ----------
    roi_masks : list of ndarray
        Per-ROI boolean masks of shape ``(num_fast_z, Y, X)``.
    sp_fastz, sp_rows, sp_cols : ndarray
        Per-superpixel fast-Z / row / column indices into the mask.

    Returns
    -------
    list of ndarray
        For each ROI, the flat indices of superpixels falling inside it.
    """
    return [
        np.flatnonzero(np.asarray(m)[sp_fastz, sp_rows, sp_cols])
        for m in roi_masks
    ]


def compute_user_roi_geometry(
    ref: np.ndarray,
    fastz_to_refz: np.ndarray,
    subsample_matrix_inds: np.ndarray,
    motion_median: tuple,
) -> dict:
    """Compute reference-image geometry for the user-ROI selection GUI.

    Derives the best display channel, the fast-Z -> reference-Z map (shifted by
    the median Z motion), and the per-superpixel (fast-Z, row, column) indices
    (shifted by the median row/column motion), plus a boolean superpixel mask.

    Parameters
    ----------
    ref : ndarray
        Reference stack for one DMD, shape ``[channels, z, Y, X]``.
    fastz_to_refz : ndarray
        Fast-Z -> reference-Z lookup (1-based), shape ``(num_fast_z, 1)``.
    subsample_matrix_inds : ndarray
        ``[ref_pixel (0-based), superpixel_id]`` rows for this DMD.
    motion_median : tuple of int
        Median ``(row, column, z)`` motion for this DMD.

    Returns
    -------
    dict
        Geometry fields consumed by the user-ROI selection GUI and the
        annotation reader: ``ref``, ``num_ref_z``, ``yx_shape``, ``best_ch``,
        ``z_map``, ``num_fast_z``, ``sp_fastz``, ``sp_rows``, ``sp_cols``,
        ``sp_mask``.
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


def save_annotations_h5(
    dr: str,
    roi_records_by_dmd: dict,
    user_roi_masks: dict,
    n_dmds: int,
    ref_files: Optional[dict] = None,
) -> str:
    """Write user ROI annotations to ``dr/annotations.h5``.

    Rectangles drawn in the OpenCV selector are stored as ``type='polygon'``
    with a 4-vertex ``position`` (``[y, x]``, 0-indexed) plus a per-ROI binary
    ``mask`` (fast-Z, Y, X). ``roi_records_by_dmd[key]`` is a list (in ROI
    order) of dicts with keys ``type``, ``label``, ``position``; the per-ROI
    mask is taken from ``user_roi_masks[key][i]`` (ROIs may overlap).

    Parameters
    ----------
    dr : str
        Destination directory (``annotations.h5`` is written inside it).
    roi_records_by_dmd : dict
        ``{"DMD{N}": [record, ...]}`` ROI metadata.
    user_roi_masks : dict
        ``{"DMD{N}": [mask, ...]}`` per-ROI boolean masks.
    n_dmds : int
        Number of DMD paths.
    ref_files : dict, optional
        ``{"DMD{N}": reference_file_path}`` for provenance.

    Returns
    -------
    str
        The path to the written ``annotations.h5``.
    """
    path = os.path.join(dr, "annotations.h5")
    str_dt = h5py.string_dtype(encoding="utf-8")
    ref_files = ref_files or {}
    with h5py.File(path, "w") as f:
        f["row_major"] = 1
        f["coords_zero_indexed"] = 1
        for dmd_ix in range(n_dmds):
            dmd_key = f"DMD{dmd_ix + 1}"
            grp = f.create_group(f"Path{dmd_ix + 1}")

            ref_file = ref_files.get(dmd_key)
            grp.create_dataset(
                "dr",
                data=(str(Path(ref_file).parent) if ref_file else ""),
                dtype=str_dt,
            )
            grp.create_dataset(
                "fn",
                data=(Path(ref_file).name if ref_file else ""),
                dtype=str_dt,
            )

            records = roi_records_by_dmd.get(dmd_key, [])
            grp.create_dataset("n_rois", data=len(records))

            mask_list = user_roi_masks.get(dmd_key, [])
            for i, rec in enumerate(records):
                _write_roi_group(grp, i, rec, mask_list, str_dt)
    return path


def _write_roi_group(
    grp: h5py.Group,
    i: int,
    rec: dict,
    mask_list: list,
    str_dt,
) -> None:
    """Write a single ``roi_###`` subgroup (type/label/mask/position)."""
    rgrp = grp.create_group(f"roi_{i:03d}")
    rtype = rec.get("type", "polygon")
    rgrp.create_dataset("type", data=rtype, dtype=str_dt)
    rgrp.create_dataset(
        "label", data=rec.get("label", f"ROI{i + 1}"), dtype=str_dt
    )
    if i < len(mask_list):
        roi_mask = np.asarray(mask_list[i]).astype(np.uint8)
        rgrp.create_dataset(
            "mask", data=roi_mask, compression="gzip", shuffle=True
        )
    if rtype == "polygon" and "position" in rec:
        rgrp.create_dataset(
            "position", data=np.asarray(rec["position"], dtype=np.float64)
        )


def _empty_path_selection(dmd_key: str, out: dict) -> None:
    """Record an empty (no-ROI) selection for one DMD across all outputs."""
    out["user_roi_masks"][dmd_key] = []
    out["user_roi_superpixels"][dmd_key] = []
    out["user_roi_labels"][dmd_key] = []
    out["roi_records"][dmd_key] = []


def _read_roi_records(grp: h5py.Group, expected: tuple):
    """Read all ``roi_###`` subgroups for one path.

    Parameters
    ----------
    grp : h5py.Group
        The ``Path{N}`` group.
    expected : tuple
        Expected mask shape ``(num_fast_z, Y, X)``.

    Returns
    -------
    tuple
        ``(roi_masks, labels, recs, bad_shape)`` where ``bad_shape`` is True if
        any ROI mask shape did not match ``expected``.
    """
    n_rois = 0
    if "n_rois" in grp:
        n_rois = int(np.asarray(grp["n_rois"][()]).reshape(-1)[0])
    roi_masks, labels, recs = [], [], []
    for i in range(n_rois):
        rgrp = grp.get(f"roi_{i:03d}")
        if rgrp is None:
            continue
        lbl = _h5_scalar_str(rgrp["label"]) if "label" in rgrp else f"ROI{i+1}"
        typ = _h5_scalar_str(rgrp["type"]) if "type" in rgrp else "polygon"
        if "mask" in rgrp:
            m = np.asarray(rgrp["mask"][()])
            if m.shape != expected:
                return roi_masks, labels, recs, True
            roi_masks.append(m > 0)
        else:
            roi_masks.append(np.zeros(expected, dtype=bool))
        labels.append(lbl)
        rec = {"type": typ, "label": lbl}
        if "position" in rgrp:
            rec["position"] = np.asarray(rgrp["position"][()])
        recs.append(rec)
    return roi_masks, labels, recs, False


def load_annotations_h5(
    annotations_path: str,
    n_dmds: int,
    user_roi_geo: dict,
) -> tuple:
    """Load user ROI annotations from ``annotations.h5`` when the file exists.

    Reconstructs, per path, the per-ROI superpixel lists, text labels, and raw
    ROI records. Each ``Path`` group may be absent (treated as no selection).
    If a per-ROI mask shape does not match the current geometry, that path is
    treated as no selection.

    Parameters
    ----------
    annotations_path : str
        Path to ``annotations.h5``.
    n_dmds : int
        Number of DMD paths.
    user_roi_geo : dict
        Per-DMD geometry from :func:`compute_user_roi_geometry`.

    Returns
    -------
    tuple
        ``(skip_manual, user_roi_masks, user_roi_superpixels,
        user_roi_labels, roi_records)``. ``skip_manual`` is True iff the file
        was read and at least one path had >= 1 ROI. On a missing or unreadable
        file, returns ``(False, {}, {}, {}, {})`` and the caller runs manual
        selection.
    """
    empty = (False, {}, {}, {}, {})
    if not os.path.exists(annotations_path):
        return empty

    out = {
        "user_roi_masks": {},
        "user_roi_superpixels": {},
        "user_roi_labels": {},
        "roi_records": {},
    }
    any_valid = False
    try:
        with h5py.File(annotations_path, "r") as hf:
            for dmd_ix in range(n_dmds):
                if _load_one_path(hf, dmd_ix, user_roi_geo, out):
                    any_valid = True
    except OSError as e:
        print(f"Could not read {annotations_path}: {e}")
        return empty

    if not any_valid:
        return empty
    return (
        True,
        out["user_roi_masks"],
        out["user_roi_superpixels"],
        out["user_roi_labels"],
        out["roi_records"],
    )


def _load_one_path(
    hf: h5py.File,
    dmd_ix: int,
    user_roi_geo: dict,
    out: dict,
) -> bool:
    """Load one path's annotations into ``out``; return True if it had ROIs."""
    dmd_key = f"DMD{dmd_ix + 1}"
    path_key = f"Path{dmd_ix + 1}"
    geo = user_roi_geo[dmd_key]
    expected = (geo["num_fast_z"], geo["yx_shape"][0], geo["yx_shape"][1])

    grp = hf.get(path_key)
    if grp is None:
        _empty_path_selection(dmd_key, out)
        return False

    roi_masks, labels, recs, bad_shape = _read_roi_records(grp, expected)
    if bad_shape:
        print(
            f"annotations.h5: {path_key} ROI mask shape mismatch "
            f"(expected {expected}); treating this path as no selection."
        )
        _empty_path_selection(dmd_key, out)
        return False

    out["user_roi_masks"][dmd_key] = roi_masks
    out["user_roi_superpixels"][
        dmd_key
    ] = user_roi_superpixel_lists_from_masks(
        roi_masks, geo["sp_fastz"], geo["sp_rows"], geo["sp_cols"]
    )
    out["user_roi_labels"][dmd_key] = labels
    out["roi_records"][dmd_key] = recs
    if len(recs) > 0:
        print(f"Loaded {len(recs)} ROI(s) for {path_key} from annotations.h5")
        return True
    return False


def run_parameter_gui(
    params: Optional[SiloParams] = None,
) -> Optional[SiloParams]:  # pragma: no cover - interactive tkinter dialog
    """Open the tkinter parameter form and return an updated ``SiloParams``.

    The form is pre-filled from ``params`` (or :class:`SiloParams` defaults).
    ``sparse_fac`` is edited on a natural-log scale. Returns ``None`` if the
    user cancels.

    Parameters
    ----------
    params : SiloParams, optional
        Parameters to pre-fill; defaults to a fresh :class:`SiloParams`.

    Returns
    -------
    SiloParams or None
        The updated parameters, or ``None`` if cancelled.
    """
    import tkinter as tk
    from tkinter import messagebox, ttk

    base = params or SiloParams()
    root = tk.Tk()
    root.title("SLAP2 Analysis Parameters")
    frame = ttk.Frame(root, padding="10")
    frame.grid(row=0, column=0)

    entries = {}
    row = 0
    for attr, label in _PARAM_GUI_FIELDS:
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky=tk.W)
        var = tk.StringVar(value=str(getattr(base, attr)))
        ttk.Entry(frame, textvariable=var, width=15).grid(row=row, column=1)
        entries[attr] = var
        row += 1

    ttk.Label(frame, text="Sparse Factor (log):").grid(
        row=row, column=0, sticky=tk.W
    )
    sparse_var = tk.StringVar(value=str(float(np.log(base.sparse_fac))))
    ttk.Entry(frame, textvariable=sparse_var, width=15).grid(row=row, column=1)
    row += 1

    ttk.Label(frame, text="Draw User ROIs?").grid(
        row=row, column=0, sticky=tk.W
    )
    draw_rois_var = tk.BooleanVar(value=base.draw_user_rois)
    ttk.Checkbutton(frame, variable=draw_rois_var).grid(row=row, column=1)
    row += 1

    result = {"params": None}

    def on_ok():
        """Validate entries and store an updated SiloParams, then close."""
        try:
            updates = {
                "analyze_hz": float(entries["analyze_hz"].get()),
                "decay_tau_s": float(entries["decay_tau_s"].get()),
                "baseline_window_s": float(entries["baseline_window_s"].get()),
                "denoise_window_s": float(entries["denoise_window_s"].get()),
                "vif": float(entries["vif"].get()),
                "d_xy": int(entries["d_xy"].get()),
                "peakth": float(entries["peakth"].get()),
                "peak_buffer": int(entries["peak_buffer"].get()),
                "max_workers": int(entries["max_workers"].get()),
                "operator": entries["operator"].get(),
                "sparse_fac": float(np.exp(float(sparse_var.get()))),
                "draw_user_rois": bool(draw_rois_var.get()),
            }
            result["params"] = replace(base, **updates)
            root.destroy()
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter values: {e}")

    def on_cancel():
        """Discard edits and close the dialog."""
        root.destroy()

    button_frame = ttk.Frame(frame)
    button_frame.grid(row=row, column=0, columnspan=2, pady=10)
    ttk.Button(button_frame, text="OK", command=on_ok).pack(side=tk.LEFT)
    ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(
        side=tk.LEFT
    )

    root.grab_set()
    root.mainloop()
    return result["params"]


def _ask_roi_label(
    roi_num: int, path_num: int, plane_num: int
) -> str:  # pragma: no cover - interactive tkinter dialog
    """Modal tkinter prompt for one ROI's label; defaults to ``ROI{n}``."""
    import tkinter as tk
    from tkinter import simpledialog

    text = None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        text = simpledialog.askstring(
            "ROI label",
            f"Label for ROI {roi_num} (Path {path_num}, "
            f"fast-Z plane {plane_num}):",
            parent=root,
        )
        root.destroy()
    except Exception as e:  # noqa: BLE001 - GUI best-effort
        print(f"Could not prompt for ROI label ({e}); using default.")
    if not text or not text.strip():
        return f"ROI{roi_num}"
    return text.strip()


def run_user_roi_selection(
    user_roi_geo: dict,
    n_dmds: int,
) -> dict:  # pragma: no cover - interactive cv2 GUI
    """Interactively select user ROIs on each DMD's reference image.

    Scrolls through fast-Z planes and lets the user draw rectangles (stored as
    polygons). Returns per-DMD ``user_roi_masks``, ``user_roi_superpixels``,
    ``user_roi_labels``, and ``roi_records`` for writing via
    :func:`save_annotations_h5`.

    Parameters
    ----------
    user_roi_geo : dict
        Per-DMD geometry from :func:`compute_user_roi_geometry`.
    n_dmds : int
        Number of DMD paths.

    Returns
    -------
    dict
        ``{"user_roi_masks", "user_roi_superpixels", "user_roi_labels",
        "roi_records"}``.
    """
    import cv2

    out = {
        "user_roi_masks": {},
        "user_roi_superpixels": {},
        "user_roi_labels": {},
        "roi_records": {},
    }
    for dmd_ix in range(n_dmds):
        dmd_key = f"DMD{dmd_ix + 1}"
        g = user_roi_geo[dmd_key]
        roi_masks, roi_records = _select_rois_for_dmd(cv2, dmd_ix, g)
        out["user_roi_masks"][dmd_key] = roi_masks
        out["user_roi_superpixels"][
            dmd_key
        ] = user_roi_superpixel_lists_from_masks(
            roi_masks, g["sp_fastz"], g["sp_rows"], g["sp_cols"]
        )
        out["user_roi_labels"][dmd_key] = [r["label"] for r in roi_records]
        out["roi_records"][dmd_key] = roi_records
    return out


def _normalize_plane(
    plane: np.ndarray,
) -> np.ndarray:  # pragma: no cover - display helper
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


def _select_rois_for_dmd(
    cv2, dmd_ix: int, g: dict
):  # pragma: no cover - interactive cv2 GUI
    """Run the interactive ROI selector for one DMD; return masks + records."""
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
    cv2, im8: np.ndarray, sp_plane: np.ndarray, roi_plane: np.ndarray
):  # pragma: no cover - display helper
    """Overlay superpixel (green) and ROI-union (red) masks on the plane."""
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
):  # pragma: no cover - interactive cv2 GUI
    """Prompt the user to draw ROI rectangles and append them in place."""
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
            m = np.zeros((num_fast_z, *yx_shape), dtype=bool)
            m[curr_fz, y:y_stop, x:x_stop] = True
            roi_masks.append(m)
            roi_union[curr_fz, y:y_stop, x:x_stop] = True
            label_text = _ask_roi_label(roi_num, dmd_ix + 1, curr_fz + 1)
            position = np.array(
                [[y, x], [y, x + w], [y + h, x + w], [y + h, x]],
                dtype=np.float64,
            )
            roi_records.append(
                {
                    "type": "polygon",
                    "label": label_text,
                    "position": position,
                }
            )
        print(f"Added {len(rois)} ROI(s) at fast-Z {curr_fz + 1}")
    cv2.destroyWindow(edit_name)
