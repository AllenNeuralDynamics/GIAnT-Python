"""Registered frames of any raw samples, with a recording's registration.

The frames registration would make of a recording if its samples were
``samples``, moved by the recorded motion instead of a new estimate: frames
are formed at the given line times with the exponential kernel, the online
correction undone as the alignment file records it (``compensateOnline``),
and resampled onto the registered grid with freshness-weighted bilinear
interpolation, as :func:`~giant_python.registration.multiroi.register_trial`
does. Samples can be any array of raw counts in the recording's slot order,
for example the injected copy minus the original, or the expected injected
counts of :class:`~giant_python.simulation.injection.SampleModel`.

A frame is moved by the continuous motion interpolated to its time as SILo's
``loadTrial`` does (``pchip``, which passes through the recorded motion at the
registered frames), less its own online reference offset when compensated: a
frame formed with another kernel than the registration's can have another
reference than the registered frame at the same time.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterator, Optional, Tuple

import numpy as np
import torch

from ..io.slap2_raster import RasterLayout
from ..registration.compensation import line_offsets
from ..registration.frames import FrameFormer
from ..registration.interpolation import interp_frame
from ..registration.sources import CompensatedFrames, PlainFrames
from .injection import Alignment, motion_at


def _source(former: FrameFormer, alignment: Alignment, times, dt: int):
    """The frame source matching the alignment's window and compensation."""
    layout = former.layout
    m = alignment.margin
    h, w = (s - 2 * m for s in alignment.shape)
    comp = alignment.compensation
    if comp is not None:
        reach = 2 * max(3 * dt, layout.lines_per_cycle) + 8
        offs = line_offsets(
            layout,
            (int(times.min()) - reach, int(times.max()) + reach),
            lead=tuple(int(v) for v in comp["yLead"]),
        )
        src = CompensatedFrames(former, offs, dt, times)
    else:
        r0, c0 = alignment.trim_r0 - 1, alignment.trim_c0 - 1
        inside = (
            (former.rows >= r0)
            & (former.rows < r0 + h)
            & (former.cols >= c0)
            & (former.cols < c0 + w)
        )
        src = PlainFrames(former, inside, dt)
    win = src.window
    if (
        int(win.trim_rows[0]) != alignment.trim_r0
        or int(win.trim_cols[0]) != alignment.trim_c0
        or win.shape != (h, w)
    ):
        raise ValueError(
            "the samples' raster does not match the registration window of "
            f"{alignment.path}"
        )
    return src


def frame_motion(
    alignment: Alignment, times: np.ndarray, refs: Optional[np.ndarray]
) -> np.ndarray:
    """Motion to resample frames at ``times`` with (rows, cols per frame).

    Parameters
    ----------
    alignment : Alignment
        The registration.
    times : ndarray of int
        One-based frame line times.
    refs : ndarray or None
        ``(frames, 2)`` online offset [x, y] each frame is formed around
        (compensated registrations), else None.

    Returns
    -------
    ndarray
        ``(frames, 2)``: the interpolated continuous motion, less ``mapping @
        (ref - refOffset)`` when compensated.
    """
    move = motion_at(alignment, times).T
    comp = alignment.compensation
    if comp is not None:
        d = comp["mapping"] @ (refs.T - comp["refOffset"][:, None])
        move = move - d.T
    return move


def iter_registered_frames(
    layout: RasterLayout,
    samples: np.ndarray,
    cycle0: int,
    alignment: Alignment,
    times,
    dt: int,
    device="cpu",
    chunk: int = 64,
    offset_rc=(0.0, 0.0),
) -> Iterator[Tuple[np.ndarray, torch.Tensor, torch.Tensor]]:
    """Registered frames of ``samples``, a chunk of frames at a time.

    Parameters
    ----------
    layout : RasterLayout
        The recording the samples belong to (slot order, lines, headers).
    samples : ndarray
        ``(channels, slots, cycles)`` raw counts (any real dtype; -32768
        marks invalid samples) of the cycles ``cycle0 ...``; the channels
        need not be the recording's.
    cycle0 : int
        Zero-based cycle of ``samples[:, :, 0]``.
    alignment : Alignment
        The recording's registration of this trial.
    times : array_like of int
        One-based frame line times.
    dt : int
        Kernel time constant in lines.
    device : str or torch.device
        Where frames are formed.
    chunk : int
        Frames per yielded chunk.
    offset_rc : (float, float)
        Frames are rendered in the trial's registered frame moved by this
        offset: pixel ``p`` shows the trial frame's ``p + offset_rc`` (the
        way SILo reads a trial on its consensus grid, with the trial shift
        ``motOutput``). Default: the trial's own registered frame.

    Yields
    ------
    index : ndarray of int
        Positions in ``times`` of the chunk's frames.
    frames : torch.Tensor
        ``(channels, n, rows, cols)`` registered frames (NaN without data).
    var_factor : torch.Tensor
        ``(n, rows, cols)`` variance factor (``1 / freshness``).
    """
    times = np.asarray(times, dtype=np.int64).ravel()
    n_ch = samples.shape[0]
    lay = replace(layout, num_channels=n_ch)
    former = FrameFormer(lay, samples, int(cycle0), 1, device)
    src = _source(former, alignment, times, int(dt))
    refs = src.references(times)
    move = frame_motion(alignment, times, refs) + np.asarray(offset_rc)
    m = alignment.margin
    win = src.window
    for a in range(0, times.size, chunk):
        idx = np.arange(a, min(times.size, a + chunk))
        vals, fresh = src.frames(times[idx])
        out, var = [], []
        for i, k in enumerate(idx):
            fimg = win.image(fresh[i], 0.0)
            regs = [
                interp_frame(
                    win.image(vals[ch, i], float("nan")),
                    fimg,
                    float(move[k, 0]),
                    float(move[k, 1]),
                    m,
                )
                for ch in range(n_ch)
            ]
            out.append(torch.stack([r for r, _ in regs]))
            var.append(regs[0][1])
        yield idx, torch.stack(out, 1), torch.stack(var)
