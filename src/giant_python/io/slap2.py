"""SLAP2 binary reader access and saved alignment-data decoding.

Band geometry and superpixel reduction belong to the extraction.band backend,
not this raw acquisition adapter.
"""

from pathlib import Path
from typing import Any, Protocol, Union

import numpy as np

from .hdf5 import load_struct_h5


def _reshape_1d(src: dict, key: str):
    """Flatten an optional raw alignment vector without altering its sign."""
    value = src.get(key)
    return None if value is None else np.asarray(value).reshape(-1)


def load_alignment_data_h5(path: Union[str, Path]) -> dict:
    """Read alignment vectors with the original saved displacement sign.

    Flatten DS/offline/online vectors and expose numChannels/alignHz scalars.
    No coordinate normalization or motion negation belongs in this reader.
    Band numerical consumers must use extraction.band.inputs instead.
    """
    data = load_struct_h5(path)
    online = data.get("slap2", {}) or {}
    return {
        "DSframes": _reshape_1d(data, "DSframes"),
        "motionDSr": _reshape_1d(data, "motionDSr"),
        "motionDSc": _reshape_1d(data, "motionDSc"),
        "motionDSz": _reshape_1d(data, "motionDSz"),
        "onlineYshift": _reshape_1d(online, "onlineMotionYshift"),
        "onlineXshift": _reshape_1d(online, "onlineMotionXshift"),
        "onlineZshift": _reshape_1d(online, "onlineMotionZshift"),
        "numChannels": int(np.asarray(data["numChannels"]).reshape(-1)[0]),
        "alignHz": float(np.asarray(data["alignHz"]).reshape(-1)[0]),
    }


class Slap2Reader(Protocol):
    """Minimal raw-reader surface consumed by the existing band reducer.

    Line/cycle indexes passed to getLineData remain 1-based. The owning
    caller retains the reader for the entire batched reduction; this protocol
    deliberately promises neither a close method nor context-manager support.
    """

    header: dict
    metaData: Any
    numCycles: int
    lineDataNumElements: Any
    lineSuperPixelIDs: Any
    lineFastZIdxs: Any

    def getLineData(self, line_indices, cycle_indices, channels=None):
        """Read a batch using the vendor reader's unchanged contract."""


def open_slap2_file(path: Union[str, Path]) -> Slap2Reader:
    """Lazily open DataFile/MultiDataFiles; return ownership to the caller.

    Retains the original CYCLE detection and module reload. No reads, close,
    copying, or early resource disposal are performed here.
    """
    import importlib
    import re

    import slap2_utils

    importlib.reload(slap2_utils)
    path = str(path)
    if re.search(r"CYCLE\d+", path):
        return slap2_utils.MultiDataFiles(path)
    return slap2_utils.DataFile(path)
