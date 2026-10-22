"""Faithful trial-table schema decoding, without runtime file discovery."""

from pathlib import Path
from typing import Optional, Union

import numpy as np

from .hdf5 import load_struct_h5


def slap2_info_from_raw(slap2: Optional[dict]):
    """Decode recognized SLAP2 metadata; keep absent metadata absent."""
    from ..models.trial_table import Slap2Info

    if not slap2:
        return None
    names = (
        "ref_stack",
        "first_line",
        "last_line",
        "trial_start_time_inferred",
        "trial_end_time_from_pc",
    )
    return Slap2Info(**{key: slap2[key] for key in names if key in slap2})


def read_trial_table(path: Union[str, Path]):
    """Return an eager TrialTable with source provenance, no path resolution.

    ``source_path`` records the input filename and is not part of disk data.
    Numeric ranks are retained; the filename string grid is always rank two.
    """
    from ..models.trial_table import TrialTable

    raw = load_struct_h5(path)
    return TrialTable(
        datadr=Path(raw["datadr"]) if raw.get("datadr") else None,
        savedr=Path(raw["savedr"]) if raw.get("savedr") else None,
        filename=np.atleast_2d(
            np.asarray(raw.get("filename", []), dtype=object)
        ),
        true_trial_ix=raw.get("true_trial_ix", []),
        epoch=raw.get("epoch", []),
        slap2_info=slap2_info_from_raw(raw.get("slap2_info")),
        motion_correction=raw.get("motion_correction"),
        source_extraction=raw.get("source_extraction"),
        source_path=Path(path).absolute(),
    )
