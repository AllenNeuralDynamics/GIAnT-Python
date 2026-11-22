"""Undoing SLAP2's online motion correction line by line (compensateOnline).

The online correction moves the DMD pattern by integer offsets recorded in
each line's header. GIAnT-MATLAB (``getLineOnlineOffsets``,
``frameOnlineOffsets``, ``slap2FrameSums`` with ``CompensateOnline``) undoes
it within each frame: every sample is moved by its line's offset minus the
frame's own offset (rows follow y, columns follow x), so a frame spanning an
x/y switch is not blurred, and the frame stays in the DMD coordinates of the
uncompensated frame.

A new y reaches the DMD before the header records it: ``y_lead[0]`` lines
earlier when x does not change on the same line, ``y_lead[1]`` lines when it
does, but never before the previous x/y switch.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..io.slap2_raster import RasterLayout, read_line_offsets

Y_LEAD = (2, 6)
"""Lines by which a new y precedes its header (x unchanged, x changed)."""

MAPPING = np.array([[0.0, 1.0], [1.0, 0.0]])
"""Header offset difference [dx; dy] to [row; col] sample displacement."""


def _mround(x):
    """Round half away from zero (MATLAB ``round``)."""
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


@dataclass
class LineOffsets:
    """Online offsets of every line of a recording.

    Attributes
    ----------
    x, y : ndarray of int
        Header ``xOffset_pix`` / ``yOffset_pix`` of each line (index
        ``line - 1``). Lines outside ``lines_read`` repeat the first or last
        value read, as in MATLAB.
    y_eff : ndarray of int
        y on the DMD while each line was acquired.
    median_xy : ndarray of float
        Rounded median ``[x, y]`` over the lines read (MATLAB medianXY).
    switch_lines : ndarray of int
        One-based lines whose x or y differs from the previous line's.
    lines_read : (int, int)
        First and last line whose header was read.
    y_lead : (int, int)
        Lead of y switches used for ``y_eff``.
    """

    x: np.ndarray
    y: np.ndarray
    y_eff: np.ndarray
    median_xy: np.ndarray
    switch_lines: np.ndarray
    lines_read: Tuple[int, int]
    y_lead: Tuple[int, int] = Y_LEAD


def dmd_y(x, y, switch_lines, lead=Y_LEAD) -> np.ndarray:
    """y on the DMD: each change of the header y applied ``lead`` earlier.

    Parameters
    ----------
    x, y : ndarray
        Header offsets of every line.
    switch_lines : ndarray of int
        One-based x/y switch lines.
    lead : (int, int)
        Lead when x is unchanged / changes on the same line.

    Returns
    -------
    ndarray
        ``yEff`` as GIAnT-MATLAB's ``dmdY`` computes it.
    """
    y_eff = y.copy()
    y_sw = np.flatnonzero(np.diff(y) != 0) + 1  # zero-based switch lines
    sw0 = np.asarray(switch_lines) - 1
    for s in y_sw:
        k = lead[1] if x[s] != x[s - 1] else lead[0]
        before = sw0[sw0 < s]
        prev = before[-1] if before.size else 0
        y_eff[max(s - k, prev) : s] = y[s]
    return y_eff


def line_offsets(
    layout: RasterLayout, lines: Tuple[int, int], lead=Y_LEAD
) -> LineOffsets:
    """Read the headers covering ``lines`` (``getLineOnlineOffsets``).

    Parameters
    ----------
    layout : RasterLayout
        The recording's layout.
    lines : (int, int)
        First and last line needed (may extend past the recording).
    lead : (int, int)
        Lead of y switches.

    Returns
    -------
    LineOffsets
        Offsets of every line of the recording, exact within the cycles
        covering ``lines`` widened by ``max(lead) + 1`` lines.
    """
    p = layout.lines_per_cycle
    n_c = layout.num_cycles
    pad = max(lead) + 1
    first_c = max(1, int(np.floor((lines[0] - pad - 1) / p)) + 1)
    last_c = min(n_c, int(np.ceil((lines[1] + pad) / p)))
    if first_c > last_c:
        raise ValueError(f"lines {lines} are outside the recording")
    read = (int((first_c - 1) * p + 1), int(last_c * p))
    x, y, _ = read_line_offsets(layout, np.arange(read[0], read[1] + 1))
    x, y = x.astype(np.int64), y.astype(np.int64)
    median_xy = _mround([np.median(x), np.median(y)])
    n_before, n_after = read[0] - 1, layout.total_lines - read[1]
    x = np.pad(x, (n_before, n_after), mode="edge")
    y = np.pad(y, (n_before, n_after), mode="edge")
    switch = np.flatnonzero((np.diff(x) != 0) | (np.diff(y) != 0)) + 2
    return LineOffsets(
        x=x,
        y=y,
        y_eff=dmd_y(x, y, switch, lead),
        median_xy=median_xy,
        switch_lines=switch,
        lines_read=read,
        y_lead=tuple(lead),
    )


def frame_offsets(
    offsets: LineOffsets, times, dt: int, lines_per_cycle: int
) -> np.ndarray:
    """The offset each frame mostly saw (``frameOnlineOffsets``).

    Parameters
    ----------
    offsets : LineOffsets
        Line offsets of the recording.
    times : array_like of int
        One-based frame times.
    dt : int
        Kernel time constant in lines.
    lines_per_cycle : int
        ``P``; the kernel reaches ``max(3 dt, P)`` lines.

    Returns
    -------
    ndarray of int
        ``(frames, 2)`` [x, y]: kernel-weighted means of ``x`` and ``y_eff``
        over each frame's lines, rounded.
    """
    times = np.asarray(times, dtype=np.int64).ravel()
    n = offsets.x.size
    dt_read = max(3 * dt, lines_per_cycle)
    out = np.zeros((times.size, 2), dtype=np.int64)
    for f, t in enumerate(times):
        ln = np.arange(max(1, t - dt_read), min(n, t + dt_read) + 1)
        w = np.exp(-np.abs(t - ln) / dt)
        out[f] = _mround(
            [w @ offsets.x[ln - 1], w @ offsets.y_eff[ln - 1]] / w.sum()
        )
    return out


def line_shifts(offsets: LineOffsets, lines: np.ndarray, refs: np.ndarray):
    """Row and column displacement of samples on ``lines`` for each frame.

    Parameters
    ----------
    offsets : LineOffsets
        Line offsets.
    lines : ndarray of int
        ``(frames, ...)`` one-based lines (any trailing shape).
    refs : ndarray of int
        ``(frames, 2)`` frame reference offsets [x, y].

    Returns
    -------
    dr, dc : ndarray of int
        ``MAPPING @ [x - ref_x; y_eff - ref_y]``: rows follow y, columns x.
    """
    extra = (1,) * (lines.ndim - 1)
    idx = np.clip(lines, 1, offsets.x.size) - 1
    dx = offsets.x[idx] - refs[:, 0].reshape((-1,) + extra)
    dy = offsets.y_eff[idx] - refs[:, 1].reshape((-1,) + extra)
    return dy, dx


def compensation_record(offsets: LineOffsets, refs: np.ndarray) -> dict:
    """The ``slap2/onlineCompensation`` group of the alignment file.

    Parameters
    ----------
    offsets : LineOffsets
        Line offsets (gives ``refOffset`` and ``yLead``).
    refs : ndarray of int
        ``(frames, 2)`` reference offsets [x, y] of the DS frames.

    Returns
    -------
    dict
        Fields as MATLAB's ``compensationRecord`` saves them.
    """
    return {
        "enabled": np.int8(1),
        "refOffset": offsets.median_xy.astype(np.float64),
        "frameRefX": refs[:, 0].astype(np.int16),
        "frameRefY": refs[:, 1].astype(np.int16),
        "mapping": MAPPING.copy(),
        "yLead": np.asarray(offsets.y_lead, dtype=np.float64),
    }
