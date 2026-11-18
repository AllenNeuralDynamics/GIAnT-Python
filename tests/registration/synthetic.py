"""A small synthetic SLAP2 raster recording with known motion.

The file mimics the raw layout that ``giant_python.io.slap2_raster`` reads:
a header, then cycles of ``P`` lines, each line a header (with online-motion
fields) followed by its samples, channel by channel. The parse-plan side is
given directly as a :class:`RasterLayout`.
"""

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from giant_python.io.slap2_raster import RasterLayout

HEADER_WORDS = 68
LINE_HEADER_BYTES = 40


def make_layout(path: Path, n_cycles: int, seed: int = 0, p: int = 7):
    """Layout of a synthetic recording (the data file is not written).

    Pixels of the raster block (rows 8-27, columns 10-39 of a 36 x 50 DMD)
    are each assigned to 1-3 of the lines 2..p (line 1 has no samples), in
    random order within a line.
    """
    rng = np.random.default_rng(seed)
    n_rows, n_cols, n_ch = 36, 50, 2
    rr, cc = np.meshgrid(np.arange(8, 28), np.arange(10, 40), indexing="ij")
    pix = (rr * n_cols + cc).ravel() + 1
    lines = {line: [] for line in range(1, p + 1)}
    for px in pix:
        for line in rng.choice(np.arange(2, p + 1), rng.integers(1, 4), False):
            lines[int(line)].append(px)
    slot_line, slot_pos, slot_pixel, n_per = [], [], [], []
    for line in range(1, p + 1):
        ids = np.array(lines[line], dtype=np.int64)
        rng.shuffle(ids)
        slot_line += [line] * ids.size
        slot_pos += list(range(ids.size))
        slot_pixel += list(ids)
        n_per.append(ids.size)
    n_per = np.array(n_per)
    header_words = LINE_HEADER_BYTES // 2
    line_words = header_words + n_ch * n_per
    line_header = np.concatenate([[0], np.cumsum(line_words)[:-1]])
    return RasterLayout(
        files=(str(path),),
        file_cycles=np.array([n_cycles]),
        file_offset_words=np.array([HEADER_WORDS]),
        words_per_cycle=int(line_words.sum()),
        lines_per_cycle=p,
        n_rows=n_rows,
        n_cols=n_cols,
        num_channels=n_ch,
        line_period_s=1 / 1600.0,
        line_header_bytes=LINE_HEADER_BYTES,
        line_header_words=line_header,
        row_base=line_header + header_words,
        n_per_channel=n_per,
        line_fast_z=np.ones(p, dtype=np.int64),
        fast_zs=np.array([-20.0]),
        slot_line=np.array(slot_line),
        slot_pos=np.array(slot_pos),
        slot_pixel=np.array(slot_pixel),
    )


def scene(layout: RasterLayout, seed: int = 1) -> np.ndarray:
    """Smooth random brightness pattern over the DMD (photons/sample)."""
    rng = np.random.default_rng(seed)
    img = gaussian_filter(rng.random((layout.n_rows, layout.n_cols)), 1.5)
    img = (img - img.min()) / (img.max() - img.min())
    return 2.0 + 60.0 * img**2


def motion(n_lines: int, seed: int = 2) -> np.ndarray:
    """Known tissue motion ``(lines, 2)`` [rows, cols], within +-2 px."""
    t = np.arange(n_lines) / n_lines
    rng = np.random.default_rng(seed)
    ph = rng.random(2) * 2 * np.pi
    return np.stack(
        [
            1.6 * np.sin(2 * np.pi * 1.3 * t + ph[0]),
            1.2 * np.sin(2 * np.pi * 0.7 * t + ph[1]),
        ],
        1,
    )


def write_recording(layout: RasterLayout, seed: int = 3, invalid=0.0):
    """Write Poisson samples of the moving scene; return the true motion.

    The sample of pixel (r, c) on line l is the scene at
    ``(r - m_r(l), c - m_c(l))``; channel 2 sees 0.5 x the scene. Line
    headers record online offsets x = cycle % 3, y = -(cycle % 2) and
    z = 0.25 * cycle. A fraction ``invalid`` of samples is -32768.
    """
    rng = np.random.default_rng(seed)
    p = layout.lines_per_cycle
    n_c = layout.num_cycles
    n_lines = n_c * p
    m = motion(n_lines)
    img = scene(layout)
    words = np.zeros((n_c, layout.words_per_cycle), dtype=np.int16)
    pr = (layout.slot_pixel - 1) // layout.n_cols
    pc = (layout.slot_pixel - 1) % layout.n_cols
    for c in range(n_c):
        for line in range(1, p + 1):
            hdr = layout.line_header_words[line - 1]
            raw = np.zeros(LINE_HEADER_BYTES, dtype=np.uint8)
            raw[20:22] = np.array([c % 3], np.int16).view(np.uint8)
            raw[22:24] = np.array([-(c % 2)], np.int16).view(np.uint8)
            raw[32:36] = np.array([0.25 * c], np.float32).view(np.uint8)
            words[c, hdr : hdr + LINE_HEADER_BYTES // 2] = raw.view(np.int16)
            sel = np.flatnonzero(layout.slot_line == line)
            if sel.size == 0:
                continue
            ln = c * p + line - 1
            lam = map_coordinates(
                img,
                [pr[sel] - m[ln, 0], pc[sel] - m[ln, 1]],
                order=1,
                mode="nearest",
            )
            for ch, gain in enumerate((1.0, 0.5)):
                x = rng.poisson(gain * lam).astype(np.int16)
                x[rng.random(x.size) < invalid] = -32768
                base = (
                    layout.row_base[line - 1]
                    + ch * layout.n_per_channel[line - 1]
                )
                words[c, base + layout.slot_pos[sel]] = x
    with open(layout.files[0], "wb") as f:
        f.write(np.zeros(HEADER_WORDS, np.int16).tobytes())
        f.write(words.tobytes())
    return m
