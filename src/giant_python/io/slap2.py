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


def is_cycle_split(path: Union[str, Path]) -> bool:
    """Whether a ``.dat`` file name marks one file of a CYCLE-split recording.

    SLAP2 writes long recordings as ``<name>-CYCLE-<first cycle>.dat`` files.
    The hyphen after ``CYCLE`` is optional, as in the ``-CYCLE-?[0-9]+$``
    suffix that ``MultiDataFiles`` (MATLAB and slap2_utils) strips to find
    the other files.
    """
    import re

    return (
        re.search(r"CYCLE-?\d+$", Path(path).stem, re.IGNORECASE) is not None
    )


_LINE_MAGIC = 322379495
"""``magicNumber`` field of every SLAP2 line header."""


def first_line_timestamp(datafile) -> int:
    """FPGA clock tick of the first line of a ``slap2_utils`` ``DataFile``.

    Line headers hold the tick as a uint64 at bytes 24-31; the first line's
    header starts the file's first cycle (``firstCycleOffsetBytes``). This is
    GIAnT-MATLAB's reader's ``firstLineTimestamp``.

    Parameters
    ----------
    datafile : slap2_utils.DataFile
        One file of a recording.

    Returns
    -------
    int
        The tick count.
    """
    offset = int(datafile.header["firstCycleOffsetBytes"])
    with open(datafile.datFileName, "rb") as f:
        f.seek(offset)
        raw = f.read(32)
    if len(raw) < 32:
        raise ValueError(f"no line header in {datafile.datFileName}")
    magic = int(np.frombuffer(raw[4:8], "<u4")[0])
    if magic != _LINE_MAGIC:
        raise ValueError(
            f"corrupt line header in {datafile.datFileName}: magic number "
            f"{magic:#x}, expected {_LINE_MAGIC:#x}"
        )
    return int(np.frombuffer(raw[24:32], "<u8")[0])


def _mcos_property_sets(blob: bytes, start: int, end: int) -> list:
    """Property sets of an MCOS ``FileWrapper__`` segment.

    Each set is a uint32 count followed by (name, type, value) triples,
    padded to 8 bytes.
    """
    words = np.frombuffer(blob[start:end], "<u4")
    sets, i = [], 0
    while i < words.size:
        n = int(words[i])
        props = words[i + 1 : i + 1 + 3 * n].reshape(n, 3).astype(int)
        sets.append(props)
        i += 1 + 3 * n
        i += i % 2
    return sets


def _mcos_object_properties(h, object_id: int) -> dict:
    """Saved properties of one MCOS object of an open v7.3 MAT file.

    The ``FileWrapper__`` blob (first cell of ``#subsystem#/MCOS``) holds a
    string table and segments for classes, objects and property sets; each
    object row gives the indexes of its two property sets.

    Returns
    -------
    dict
        Property name -> its value: an array for values saved as cells,
        a string or an integer for values saved inline.
    """
    import struct

    refs = np.asarray(h["#subsystem#/MCOS"][()]).ravel()
    blob = np.asarray(h[refs[0]][()]).ravel().tobytes()
    _, n_str = struct.unpack_from("<II", blob, 0)
    offs = struct.unpack_from("<8I", blob, 8)
    strings = [s.decode() for s in blob[40 : offs[0]].split(b"\0")[:n_str]]
    objects = np.frombuffer(blob[offs[2] : offs[3]], "<u4").reshape(-1, 6)
    row = objects[objects[:, 5] == object_id]
    if row.shape[0] != 1:
        raise ValueError(f"MCOS object {object_id} not found")
    set1, set3 = int(row[0, 3]), int(row[0, 4])
    sets1 = _mcos_property_sets(blob, offs[1], offs[2])
    sets3 = _mcos_property_sets(blob, offs[3], offs[4])
    props = {}
    for sets, idx in ((sets1, set1), (sets3, set3)):
        for name_ix, kind, value in sets[idx] if 0 < idx < len(sets) else []:
            if kind == 1:  # a cell of #subsystem#/MCOS, after two fixed ones
                val = np.asarray(h[refs[value + 2]][()])
            elif kind == 0:
                val = strings[value - 1]
            else:
                val = value
            props[strings[name_ix - 1]] = val
    return props


def _wall_clock_ms(ms: float, tz: str):
    """UTC milliseconds since 1970 as the wall clock of time zone ``tz``."""
    import datetime
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    try:
        zone = ZoneInfo("UTC" if tz.upper() == "UTC" else tz)
    except (ZoneInfoNotFoundError, ValueError):
        return None
    epoch = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
    local = (epoch + datetime.timedelta(milliseconds=ms)).astimezone(zone)
    wall = local.replace(tzinfo=None) - epoch.replace(tzinfo=None)
    return wall / datetime.timedelta(milliseconds=1)


def _text(value) -> str:
    """A MATLAB char array (uint16 codes) or string as ``str``."""
    if isinstance(value, np.ndarray):
        return "".join(map(chr, value.ravel())) if value.size else ""
    return str(value)


def read_mat_datetime_ms(path: Union[str, Path], name: str):
    """A MATLAB ``datetime`` saved in a v7.3 MAT file, as wall-clock ms.

    MATLAB stores a ``datetime`` as an MCOS object whose ``data`` property is
    milliseconds since 1970-01-01 (a complex double: whole and fractional
    parts) and whose ``tz`` property, when set, names the time zone in which
    ``data`` is UTC.

    Parameters
    ----------
    path : str or Path
        The MAT (v7.3, HDF5) file, e.g. a SLAP2 ``.meta`` file.
    name : str
        Top-level variable name, e.g. ``"fpgaTimeReference"``.

    Returns
    -------
    float or None
        Milliseconds since 1970-01-01 of the local (wall-clock) time, which
        MATLAB's ``datenum`` of the value uses; None when the value is
        missing or not a ``datetime`` this reader understands.
    """
    import struct

    import h5py

    try:
        with h5py.File(path, "r") as h:
            ds = h.get(name)
            if ds is None or ds.attrs.get("MATLAB_class") not in (
                b"datetime",
                "datetime",
            ):
                return None
            object_id = int(np.asarray(ds[()]).ravel()[-2])
            props = _mcos_object_properties(h, object_id)
    except (OSError, KeyError, ValueError, IndexError, struct.error):
        return None
    if not isinstance(props.get("data"), np.ndarray):
        return None
    data = props["data"].ravel()[0]
    if data.dtype.names:
        ms = float(data["real"]) + float(data["imag"])
    else:
        ms = float(np.real(data)) + float(np.imag(data))
    tz = _text(props.get("tz", ""))
    if not np.isfinite(ms):
        return None
    return _wall_clock_ms(ms, tz) if tz else ms


def open_slap2_file(path: Union[str, Path]) -> Slap2Reader:
    """Lazily open DataFile/MultiDataFiles; return ownership to the caller.

    A file of a CYCLE-split recording opens the whole recording (all its
    files), so line and cycle indexes run over the recording, as in
    GIAnT-MATLAB. Keeps the module reload. No reads, close, copying, or early
    resource disposal are performed here.
    """
    import importlib

    import slap2_utils

    importlib.reload(slap2_utils)
    path = str(path)
    if is_cycle_split(path):
        return slap2_utils.MultiDataFiles(path)
    return slap2_utils.DataFile(path)
