"""Small runtime records used by band extraction, without copying arrays."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class TrialTraceResult(NamedTuple):
    """One trial's named traces in a compact, immutable record.

    Source arrays are (frames, sources); global fluorescence is
    (frames, channels) and ROI fluorescence is (frames, ROIs, channels).
    Reference offsets and online shifts are (row, column, z) arrays, each
    with one value per frame. Only output assembly negates reference offsets
    into saved MATLAB displacement. Skipped trials have zero frames.
    """

    d_f: np.ndarray
    f0_ls: np.ndarray
    frame_line_idxs: np.ndarray
    selected_pixels: np.ndarray
    global_f: np.ndarray
    reference_offsets: tuple[np.ndarray, np.ndarray, np.ndarray]
    online_shifts: tuple[np.ndarray, np.ndarray, np.ndarray]
    user_roi_f: np.ndarray


@dataclass(frozen=True)
class ResolvedAcquisition:
    """Acquisition metadata resolved once without mutating science options."""

    num_channels: int
    align_hz: dict[str, float]
