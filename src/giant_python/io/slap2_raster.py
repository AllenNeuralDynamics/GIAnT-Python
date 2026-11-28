"""Raster sample layout, raw sample reads and line headers of SLAP2 files.

The vendor reader (``slap2_utils``) parses the file header and parse plan;
this module turns them into flat tables for reading every raster sample of a
range of cycles at once, as GIAnT-MATLAB's ``slap2FrameSums`` does with its
file reader, and decodes the online-motion fields of line headers, as
``getOnlineMotion`` does.

Conventions: line indices within a cycle (``slot_line``) and global line
numbers are one-based, as in MATLAB. Cycle indices are zero-based. Pixel IDs
are one-based and row-major over the DMD (``(row - 1) * n_cols + col``).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple, Union

import numpy as np

from .slap2 import open_slap2_file

INVALID_SAMPLE = -32768
"""Raw value marking a sample the acquisition did not record."""

_PLACEHOLDER_Z = -10203
"""First entry of ``slap2_utils`` ``fastZs`` for old-format metadata."""


@dataclass(frozen=True)
class RasterLayout:
    """Where every raster sample of a SLAP2 recording lies in its files.

    Attributes
    ----------
    files : tuple of str
        Data files in cycle order (several for CYCLE-split recordings).
    file_cycles : ndarray of int
        Number of cycles in each file.
    file_offset_words : ndarray of int
        Word (int16) offset of the first cycle in each file.
    words_per_cycle : int
        Words per cycle, identical in every file.
    lines_per_cycle : int
        Lines per cycle (``P``).
    n_rows, n_cols : int
        DMD image rows (``dmdPixelsPerColumn``) and columns
        (``dmdPixelsPerRow``).
    num_channels : int
        Recorded channels.
    line_period_s : float
        Line period from the metadata file.
    line_header_bytes : int
        Size of each line header.
    line_header_words : ndarray of int
        Zero-based word offset of each line's header within a cycle.
    row_base : ndarray of int
        Zero-based word offset of each line's samples within a cycle.
    n_per_channel : ndarray of int
        Samples per channel on each line.
    line_fast_z : ndarray of int
        Fast-z plane index (one-based, 0 for none) of each line.
    fast_zs : ndarray of float
        Fast-z positions of the planes.
    slot_line : ndarray of int
        One-based line index of each raster slot.
    slot_pos : ndarray of int
        Zero-based position of each raster slot within its line's samples.
    slot_pixel : ndarray of int
        One-based DMD pixel ID of each raster slot.
    """

    files: Tuple[str, ...]
    file_cycles: np.ndarray
    file_offset_words: np.ndarray
    words_per_cycle: int
    lines_per_cycle: int
    n_rows: int
    n_cols: int
    num_channels: int
    line_period_s: float
    line_header_bytes: int
    line_header_words: np.ndarray
    row_base: np.ndarray
    n_per_channel: np.ndarray
    line_fast_z: np.ndarray
    fast_zs: np.ndarray
    slot_line: np.ndarray
    slot_pos: np.ndarray
    slot_pixel: np.ndarray

    @property
    def num_cycles(self) -> int:
        """Total number of cycles over all files."""
        return int(np.sum(self.file_cycles))

    @property
    def total_lines(self) -> int:
        """Total number of lines over all files."""
        return self.num_cycles * self.lines_per_cycle


def _line_ids(ids, n_samples: int) -> np.ndarray:
    """Superpixel IDs of one line, checked against its sample count.

    ``slap2_utils`` decodes an empty MATLAB array of the parse plan as its
    dimensions (``[0, 1]``); a line without samples has no superpixels.
    """
    ids = np.asarray(ids).ravel().astype(np.int64)
    if n_samples == 0:
        return ids[:0]
    if ids.size != n_samples:
        raise ValueError(
            f"parse plan lists {ids.size} superpixels for a line with "
            f"{n_samples} samples per channel"
        )
    return ids


def _data_files(reader) -> list:
    """The ``DataFile`` objects of a reader, in cycle order."""
    return list(getattr(reader, "hDataFiles", None) or [reader])


def raster_layout(path: Union[str, Path]) -> RasterLayout:
    """Read the raster sample layout of a SLAP2 recording.

    Parameters
    ----------
    path : str or Path
        A ``.dat`` file; CYCLE-split recordings are opened as one recording,
        as ``open_slap2_file`` does.

    Returns
    -------
    RasterLayout
        The layout; only raster superpixels (IDs within the DMD) are listed,
        as with ``getImage(..., spTypeFlag = 1)``.
    """
    reader = open_slap2_file(path)
    files = _data_files(reader)
    first = files[0]
    hdr = first.header
    p = int(hdr["linesPerCycle"])
    n_cols = int(hdr["dmdPixelsPerRow"])
    n_rows = int(hdr["dmdPixelsPerColumn"])
    n_ch = int(hdr["numChannels"])
    if int(hdr.get("channelsInterleave", 0)):
        raise ValueError("only deinterleaved SLAP2 data are supported")
    header_idx = np.asarray(first.lineHeaderIdxs, dtype=np.int64)
    start_idx = np.asarray(first.lineDataStartIdxs, dtype=np.int64)
    n_elem = np.asarray(first.lineDataNumElements, dtype=np.int64)
    n_per_ch = n_elem // n_ch
    words = int(hdr["bytesPerCycle"]) // 2
    for df in files[1:]:
        if int(df.header["bytesPerCycle"]) // 2 != words or not np.array_equal(
            np.asarray(df.lineDataStartIdxs), start_idx
        ):
            raise ValueError("the files of a recording differ in layout")
    slot_line, slot_pos, slot_pixel = [], [], []
    n_px = n_rows * n_cols
    for line in range(1, p + 1):
        ids = _line_ids(first.lineSuperPixelIDs[line - 1], n_per_ch[line - 1])
        pix = ids + 1
        sel = np.flatnonzero(pix <= n_px)
        slot_line.append(np.full(sel.size, line, dtype=np.int64))
        slot_pos.append(sel.astype(np.int64))
        slot_pixel.append(pix[sel])
    fast_zs = np.asarray(first.fastZs, dtype=np.float64).ravel()
    if fast_zs.size and fast_zs[0] == _PLACEHOLDER_Z:
        fast_zs = fast_zs[1:]
    return RasterLayout(
        files=tuple(str(df.datFileName) for df in files),
        file_cycles=np.array([int(df.numCycles) for df in files]),
        file_offset_words=np.array(
            [int(df.header["firstCycleOffsetBytes"]) // 2 for df in files]
        ),
        words_per_cycle=words,
        lines_per_cycle=p,
        n_rows=n_rows,
        n_cols=n_cols,
        num_channels=n_ch,
        line_period_s=float(first.metaData.linePeriod_s),
        line_header_bytes=int(hdr["lineHeaderSizeBytes"]),
        line_header_words=header_idx - header_idx[0],
        row_base=start_idx - header_idx[0],
        n_per_channel=n_per_ch,
        line_fast_z=np.asarray(first.lineFastZIdxs).astype(np.int64),
        fast_zs=fast_zs,
        slot_line=np.concatenate(slot_line),
        slot_pos=np.concatenate(slot_pos),
        slot_pixel=np.concatenate(slot_pixel),
    )


def _file_spans(layout: RasterLayout, c0: int, c1: int):
    """Yield (file index, first local cycle, count, output offset)."""
    ends = np.cumsum(layout.file_cycles)
    starts = ends - layout.file_cycles
    for f, (s, e) in enumerate(zip(starts, ends)):
        lo, hi = max(c0, s), min(c1, e)
        if lo < hi:
            yield f, int(lo - s), int(hi - lo), int(lo - c0)


def read_raster_samples(
    layout: RasterLayout,
    cycles: Tuple[int, int],
    slots: Union[np.ndarray, None] = None,
    chunk_cycles: int = 256,
) -> np.ndarray:
    """Read raw raster samples of a range of cycles.

    Parameters
    ----------
    layout : RasterLayout
        The recording's layout.
    cycles : (int, int)
        Zero-based cycle range ``[c0, c1)``.
    slots : ndarray of int, optional
        Raster slots to read (default: all).
    chunk_cycles : int
        Cycles read from a file at a time, bounding temporary memory.

    Returns
    -------
    ndarray of int16
        Samples, shaped ``(channels, slots, cycles)``.
    """
    c0, c1 = int(cycles[0]), int(cycles[1])
    if not 0 <= c0 <= c1 <= layout.num_cycles:
        raise ValueError(f"cycle range {cycles} outside the recording")
    if slots is None:
        slots = np.arange(layout.slot_line.size)
    lines = layout.slot_line[slots] - 1
    word = [
        layout.row_base[lines]
        + ch * layout.n_per_channel[lines]
        + layout.slot_pos[slots]
        for ch in range(layout.num_channels)
    ]
    out = np.empty((layout.num_channels, slots.size, c1 - c0), dtype=np.int16)
    wpc = layout.words_per_cycle
    for f, local, count, dst in _file_spans(layout, c0, c1):
        mm = np.memmap(
            layout.files[f],
            dtype=np.int16,
            mode="r",
            offset=2 * int(layout.file_offset_words[f]),
            shape=(int(layout.file_cycles[f]), wpc),
        )
        for a in range(0, count, chunk_cycles):
            b = min(count, a + chunk_cycles)
            block = np.asarray(mm[local + a : local + b])
            for ch in range(layout.num_channels):
                out[ch, :, dst + a : dst + b] = block[:, word[ch]].T
        del mm
    return out


def read_line_offsets(
    layout: RasterLayout, lines: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Online-motion offsets recorded in the headers of given lines.

    Line header layout (bytes): 20-21 ``xOffset_pix`` (int16), 22-23
    ``yOffset_pix`` (int16), 32-35 ``zOffset_um`` (float32, present only
    when headers are longer than 32 bytes; 0 otherwise).

    Parameters
    ----------
    layout : RasterLayout
        The recording's layout.
    lines : sequence of int
        One-based global line numbers.

    Returns
    -------
    x, y, z : ndarray of float
        Offsets of each line, as GIAnT-MATLAB ``getOnlineMotion`` returns.
    """
    lines = np.asarray(lines, dtype=np.int64).ravel()
    p = layout.lines_per_cycle
    if lines.size and (lines.min() < 1 or lines.max() > layout.total_lines):
        raise ValueError("line numbers outside the recording")
    cyc = (lines - 1) // p
    li = (lines - 1) % p
    x = np.full(lines.size, np.nan)
    y = np.full(lines.size, np.nan)
    z = np.zeros(lines.size)
    ends = np.cumsum(layout.file_cycles)
    starts = ends - layout.file_cycles
    for f in range(len(layout.files)):
        sel = np.flatnonzero((cyc >= starts[f]) & (cyc < ends[f]))
        if sel.size == 0:
            continue
        byte = 2 * (
            layout.file_offset_words[f]
            + (cyc[sel] - starts[f]) * layout.words_per_cycle
            + layout.line_header_words[li[sel]]
        )
        mm = np.memmap(layout.files[f], dtype=np.uint8, mode="r")
        raw = mm[byte[:, None] + np.arange(min(40, layout.line_header_bytes))]
        del mm
        x[sel] = raw[:, 20:22].copy().view(np.int16).ravel()
        y[sel] = raw[:, 22:24].copy().view(np.int16).ravel()
        if layout.line_header_bytes > 32:
            z[sel] = raw[:, 32:36].copy().view(np.float32).ravel()
    return x, y, z
