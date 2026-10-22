"""Codec for the experiment-summary schema emitted by band extraction.

The canonical interface is ``write_summary(summary, path) -> None`` and
``read_summary(path) -> ExperimentSummary``.
There is no motion-sign conversion here. Existing params/row_major are left
untouched; supplied paths are replaced, and unrelated paths are retained.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Union

import h5py
import numpy as np

from ..models.experiment_summary import (
    FRAME_INFO_KEYS,
    ExperimentSummary,
    FrameInfo,
    PathSummary,
    Source,
    UserRoi,
    Visualizations,
)
from .hdf5 import load_struct_h5, write_dict_to_h5group

_TEMPORAL_KEYS = ("dF_ls", "dF_denoised", "events", "F0", "SNR")
_SOURCE_KEYS = ("profiles", "coords", *_TEMPORAL_KEYS)


def _decode_path(assembled: dict, path_index: int) -> PathSummary:
    """Decode schema arrays into an eager path, sharing populated arrays.

    This schema-only adapter does not concatenate traces or compute baselines.
    Empty-array templates preserve dimensions/dtypes absent from empty lists.
    """
    sources = [
        Source(
            profile=assembled["profiles"][j],
            coords=assembled["coords"][j],
            df_ls=assembled["dF_ls"][j],
            df_denoised=(
                assembled["dF_denoised"][j]
                if "dF_denoised" in assembled
                else None
            ),
            events=assembled["events"][j] if "events" in assembled else None,
            f0=assembled["F0"][j],
            snr=assembled["SNR"][j, 0],
        )
        for j in range(assembled["profiles"].shape[0])
    ]
    rois = assembled.get("user_rois")
    user_rois = []
    if rois is not None:
        labels = np.asarray(rois["labels"], dtype=object).reshape(-1)
        user_rois = [
            UserRoi(
                label=str(label),
                mask=rois["mask"][j] if j < rois["mask"].shape[0] else None,
                f=rois["F"][j] if j < rois["F"].shape[0] else None,
                fsvd=(
                    rois["Fsvd"][j]
                    if "Fsvd" in rois and j < rois["Fsvd"].shape[0]
                    else None
                ),
            )
            for j, label in enumerate(labels)
        ]
    return PathSummary(
        path_index=path_index,
        sources=sources,
        user_rois=user_rois,
        visualizations=Visualizations(
            mean_im=assembled["mean_im"],
            act_im=assembled["act_im"],
            act_im_peaks=assembled["act_im_peaks"],
        ),
        frame_info=FrameInfo(
            **{key: assembled[key] for key in FRAME_INFO_KEYS}
        ),
        global_f=assembled["global_f"],
        z_depths=assembled["Z_depths"],
        annotation_enabled=assembled.get(
            "annotation_enabled", rois is not None
        ),
        _empty_source_arrays=(
            {}
            if sources
            else {
                key: assembled[key] for key in _SOURCE_KEYS if key in assembled
            }
        ),
        _empty_roi_arrays=(
            {key: rois[key] for key in ("mask", "F", "Fsvd") if key in rois}
            if rois is not None and not user_rois
            else {}
        ),
    )


def _stack_required_arrays(
    arrays: Iterable[np.ndarray | None], field: str
) -> np.ndarray:
    """Stack arrays; identify missing fields by their 0-based index."""
    required: list[np.ndarray] = []
    for index, array in enumerate(arrays):
        if array is None:
            raise ValueError(f"{field}[{index}] is required; got None")
        required.append(array)
    return np.stack(required)


def _optional_array(array, reference: np.ndarray, field: str) -> np.ndarray:
    """Validate a supplied trace, or mark unavailable samples with NaNs."""
    if array is None:
        dtype = (
            reference.dtype
            if np.issubdtype(reference.dtype, np.inexact)
            else np.dtype(float)
        )
        return np.full(reference.shape, np.nan, dtype=dtype)
    array = np.asarray(array)
    if array.shape != reference.shape:
        raise ValueError(
            f"{field} must have shape {reference.shape}; got {array.shape}"
        )
    return array


def _stack_optional_arrays(
    arrays: Iterable[np.ndarray | None], reference: np.ndarray, field: str
) -> np.ndarray:
    """Check each optional entry against its corresponding required trace."""
    return np.stack(
        [
            _optional_array(array, reference[index], f"{field}[{index}]")
            for index, array in enumerate(arrays)
        ]
    )


def _empty_geometry(
    path: PathSummary, field: str
) -> tuple[tuple[int, ...], np.ndarray]:
    """Infer empty tensor dimensions from the saved images and global trace."""
    mean = np.asarray(path.visualizations.mean_im)
    global_f = np.asarray(path.global_f)
    if mean.ndim != 4 or global_f.ndim != 2:
        raise ValueError(
            f"Empty {field} require shape templates or "
            "mean_im and global_f geometry"
        )
    return mean.shape[1:], global_f


def _encode_path(path: PathSummary) -> dict:
    """Expose band-schema arrays for persistence, without scientific changes.

    Populated lists are stacked so edits to public Source/UserRoi entries are
    reflected. Empty results retain their original tensor dimensions/dtypes.
    Missing optional traces are NaN, never inferred from other signals.
    Required fields and optional shapes are checked before path replacement.
    """
    out = {
        "Z_depths": path.z_depths,
        "global_f": path.global_f,
        "mean_im": path.visualizations.mean_im,
        "act_im": path.visualizations.act_im,
        "act_im_peaks": path.visualizations.act_im_peaks,
        **path.frame_info.to_dict(),
    }
    if path.sources:
        out.update(
            {
                "profiles": _stack_required_arrays(
                    (s.profile for s in path.sources), "Source.profile"
                ),
                "coords": _stack_required_arrays(
                    (s.coords for s in path.sources), "Source.coords"
                ),
                "dF_ls": _stack_required_arrays(
                    (s.df_ls for s in path.sources), "Source.df_ls"
                ),
                "F0": _stack_required_arrays(
                    (s.f0 for s in path.sources), "Source.f0"
                ),
                "SNR": np.asarray(
                    [np.nan if s.snr is None else s.snr for s in path.sources]
                ).reshape(-1, 1),
            }
        )
        for key, field in (
            ("dF_denoised", "df_denoised"),
            ("events", "events"),
        ):
            out[key] = _stack_optional_arrays(
                (getattr(s, field) for s in path.sources),
                out["dF_ls"],
                f"Source.{field}",
            )
    elif path._empty_source_arrays:
        out.update(path._empty_source_arrays)
    else:
        spatial_shape, global_f = _empty_geometry(path, "sources")
        out.update(
            {
                "profiles": np.empty((0, *spatial_shape), dtype=np.float32),
                "coords": np.empty((0, 3), dtype=np.int32),
                "dF_ls": np.empty((0, *global_f.shape), dtype=global_f.dtype),
                "F0": np.empty((0, *global_f.shape), dtype=global_f.dtype),
                "SNR": np.empty((0, 1), dtype=np.float32),
            }
        )
    if not path.sources:
        for key in ("dF_denoised", "events"):
            out[key] = _optional_array(out.get(key), out["dF_ls"], key)
    if not path.annotation_enabled and path.user_rois:
        raise ValueError("Set annotation_enabled=True to serialize user ROIs")
    if path.user_rois:
        out["user_rois"] = {
            "mask": _stack_required_arrays(
                (roi.mask for roi in path.user_rois), "UserRoi.mask"
            ),
            "labels": [roi.label for roi in path.user_rois],
            "F": _stack_required_arrays(
                (roi.f for roi in path.user_rois), "UserRoi.f"
            ),
        }
        out["user_rois"]["Fsvd"] = _stack_optional_arrays(
            (roi.fsvd for roi in path.user_rois),
            out["user_rois"]["F"],
            "UserRoi.fsvd",
        )
    else:
        if path._empty_roi_arrays:
            empty = path._empty_roi_arrays
        else:
            # Both disabled and enabled-but-empty annotations have full schema.
            spatial_shape, global_f = _empty_geometry(path, "ROIs")
            empty = {
                "mask": np.zeros((0, *spatial_shape), dtype=bool),
                "F": np.empty((0, *global_f.shape), dtype=global_f.dtype),
            }
        out["user_rois"] = {
            **empty,
            "labels": [],
            "Fsvd": _optional_array(
                empty.get("Fsvd"), empty["F"], "UserRoi.Fsvd"
            ),
        }
    if path.z_depths is None:
        raise ValueError("Z_depths is required; got None")
    out["Z_depths"] = np.asarray(path.z_depths).reshape(-1, 1)
    return out


def _write_path_group(f: h5py.File, path: PathSummary) -> None:
    """Replace one Path{index+1} group without array sign conversion."""
    assembled = _encode_path(path)
    name = f"Path{path.path_index + 1}"
    if name in f:
        del f[name]
    group = f.create_group(name)
    group.create_dataset("Z_depths", data=assembled["Z_depths"])
    sources = group.create_group("sources")
    spatial = sources.create_group("spatial")
    for key in ("profiles", "coords"):
        spatial.create_dataset(key, data=assembled[key])
    temporal = sources.create_group("temporal")
    for key in _TEMPORAL_KEYS:
        temporal.create_dataset(key, data=assembled[key])
    frame_group = group.create_group("frame_info")
    for key in FRAME_INFO_KEYS:
        frame_group.create_dataset(key, data=assembled[key])
    group.create_group("global").create_dataset(
        "F", data=assembled["global_f"]
    )
    vis = group.create_group("visualizations")
    for key in ("act_im", "mean_im", "act_im_peaks"):
        vis.create_dataset(key, data=assembled[key])
    rois = assembled["user_rois"]
    roi_group = group.create_group("user_rois")
    roi_group.attrs["annotation_enabled"] = bool(path.annotation_enabled)
    for key in ("mask", "F", "Fsvd"):
        roi_group.create_dataset(key, data=rois[key])
    roi_group.create_dataset(
        "labels",
        data=np.asarray(rois["labels"], dtype=object).reshape(-1, 1),
        dtype=h5py.string_dtype(encoding="utf-8"),
    )


def write_summary(summary: ExperimentSummary, path: Union[str, Path]) -> None:
    """Persist models, retaining existing params, flags and unrelated paths.

    Path identity comes exclusively from PathSummary.path_index. Required
    source/ROI arrays and supplied optional trace shapes are validated before
    replacing each existing path group. Disabled annotations still emit empty
    ROI datasets; their state is recorded in a group attribute.
    """
    indexes = [item.path_index for item in summary.paths]
    if len(set(indexes)) != len(indexes):
        raise ValueError("Summary contains duplicate path_index values")
    with h5py.File(path, "a") as f:
        if "row_major" not in f:
            f["row_major"] = 1
        if "params" not in f:
            write_dict_to_h5group(
                f.create_group("params"),
                summary.params or {},
            )
        for item in summary.paths:
            _write_path_group(f, item)


def read_summary(path: Union[str, Path]) -> ExperimentSummary:
    """Read emitted band paths eagerly, respecting row_major and saved signs.

    Unrecognized root groups are ignored. Paths are sorted by their numeric
    identity, not lexical order. This is not a general standard-SILo codec.
    Legacy missing optional traces remain None; without annotation metadata,
    ROI group presence retains its historical enabled/disabled meaning.
    """
    raw = load_struct_h5(path)
    names = sorted(
        (key for key in raw if key.startswith("Path") and key[4:].isdigit()),
        key=lambda key: int(key[4:]),
    )
    # The shared struct loader intentionally ignores attributes. Legacy files
    # used group presence alone to distinguish disabled/empty annotations.
    with h5py.File(path, "r") as f:
        annotation_enabled = {
            name: (
                bool(
                    f[name]["user_rois"].attrs.get("annotation_enabled", True)
                )
                if "user_rois" in f[name]
                else False
            )
            for name in names
        }
    paths = []
    for name in names:
        group = raw[name]
        sources = group["sources"]
        rois = group.get("user_rois")
        assembled = {
            "Z_depths": group["Z_depths"],
            **sources["spatial"],
            **sources["temporal"],
            **group["frame_info"],
            **group["visualizations"],
            "global_f": group["global"]["F"],
            "user_rois": rois,
            "annotation_enabled": annotation_enabled[name],
        }
        paths.append(_decode_path(assembled, int(name[4:]) - 1))
    return ExperimentSummary(paths=paths, params=raw.get("params"))
