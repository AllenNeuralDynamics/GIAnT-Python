"""Tests of source seeding and pixel selection of SiLo's plan stage."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from giant_python.extraction.raster.seeding import (
    disk_strel,
    seed_sources,
    select_source_pixels,
    sources_from_seeds,
)

FIXTURE = Path(__file__).parent / "data" / "matlab_plan.h5"


def test_disk_strel_matches_matlab():
    """strel('disk', r).Neighborhood of MATLAB R2024b for r = 0 .. 39."""
    with h5py.File(FIXTURE, "r") as f:
        g = f["strel_disk"]
        for r in range(40):
            np.testing.assert_array_equal(
                disk_strel(r), g[str(r)][()].astype(bool), err_msg=f"r={r}"
            )


def test_disk_strel_shapes():
    """MATLAB's disk is smaller than its radius says: radius 3 is a full
    5 x 5 square, radius 6 an 11 x 11 octagon (SiLo's selRadius)."""
    assert disk_strel(0).shape == (1, 1)
    np.testing.assert_array_equal(
        disk_strel(1), [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    )
    np.testing.assert_array_equal(disk_strel(3), np.ones((5, 5), bool))
    d6 = disk_strel(6)
    assert d6.shape == (11, 11) and d6.sum() == 109
    np.testing.assert_array_equal(d6[0], [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0])
    np.testing.assert_array_equal(d6[1], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    assert d6[2:9].all()
    np.testing.assert_array_equal(d6, d6.T)
    np.testing.assert_array_equal(d6, d6[::-1, ::-1])
    # n = 0 is the Euclidean disk
    assert disk_strel(6, 0).shape == (13, 13)
    with pytest.raises(ValueError):
        disk_strel(-1)
    with pytest.raises(ValueError):
        disk_strel(5, 5)


def test_sources_from_seeds_rounds_and_prunes():
    """Positions round as MATLAB's (halves up in its one-based
    coordinates), the first seed of a pixel wins, order is kept, and
    positions are clamped to the image."""
    seeds = np.array(
        [
            [5.0, 3.5, 7.49, 0.4],  # MATLAB 4.5, 8.49 -> pixel (5, 8)
            [2.0, 9.2, 1.0, 0.4],
            [9.0, 3.6, 7.2, 0.4],  # same pixel as the first: dropped
            [1.0, -0.5, 19.5, 0.4],  # MATLAB 0.5, 20.5 -> (1, 21): clamped
            [4.0, 2.5 - 1e-12, 0.0, 0.4],
        ]
    )
    rows, cols, values = sources_from_seeds(seeds, (12, 20))
    np.testing.assert_array_equal(rows, [4, 9, 0, 2])
    np.testing.assert_array_equal(cols, [7, 1, 19, 0])
    np.testing.assert_array_equal(values, [5.0, 2.0, 1.0, 4.0])
    empty = sources_from_seeds(np.zeros((0, 4)), (12, 20))
    assert all(x.size == 0 for x in empty)


def test_select_source_pixels():
    """A source's pixels are MATLAB's disk within the image and the valid
    pixels, in column-major order; sources that lose their own pixel or
    nearly all pixels are pruned."""
    valid = np.ones((30, 40), bool)
    valid[:, 25:] = False
    valid[12, 10] = False
    rows = np.array([12, 2, 12, 12, 20])
    cols = np.array([8, 3, 29, 10, 30])
    sel = select_source_pixels(rows, cols, valid, sel_radius=6)
    np.testing.assert_array_equal(sel.keep, [True, True, False, False, False])
    # an interior source has the whole disk but the invalid pixel
    disk = np.zeros((30, 40), bool)
    disk[7:18, 3:14] = disk_strel(6)
    disk[12, 10] = False
    cc, rr = np.nonzero(disk.T)
    np.testing.assert_array_equal(sel.pixels[0], rr + cc * 30)
    assert (np.diff(sel.pixels[0]) > 0).all()
    # near the corner the disk is clipped by the image
    assert sel.pixels[1].size < 109 and sel.pixels[1].min() == 0
    # a source on an invalid pixel is pruned although pixels remain: the
    # third keeps the disk's first column (7 pixels), the fourth nearly all
    assert sel.pixels[2].size == 7 and sel.pixels[3].size == 108
    assert sel.pixels[4].size == 0
    # a source needs more than 5 pixels
    island = np.zeros((20, 20), bool)
    island[5:7, 5:7] = True  # 4 pixels
    few = select_source_pixels(np.array([5]), np.array([5]), island, 6)
    assert few.pixels[0].size == 4 and not few.keep[0]
    island[5:7, 7] = True  # 6 pixels
    more = select_source_pixels(np.array([5]), np.array([5]), island, 6)
    assert more.pixels[0].size == 6 and more.keep[0]


def test_seed_sources():
    """The default method seeds blobs; the soma mask excludes them; unknown
    and unported methods are refused; an image without usable pixels gives
    no seeds."""
    rng = np.random.default_rng(0)
    im = rng.normal(0, 0.3, (40, 50))
    rr, cc = np.mgrid[:40, :50]
    for r0, c0 in [(10, 12), (28, 35)]:
        im += 20 * np.exp(-((rr - r0) ** 2 + (cc - c0) ** 2) / (2 * 1.3**2))
    seeds = seed_sources(im, 8.0)
    assert seeds.shape == (2, 4)
    np.testing.assert_allclose(
        seeds[np.argsort(seeds[:, 1]), 1:3], [(10, 12), (28, 35)], atol=0.3
    )
    soma = np.zeros(im.shape, bool)
    soma[20:, 25:] = True
    masked = seed_sources(im, 8.0, soma)
    assert masked.shape == (1, 4) and abs(masked[0, 1] - 10) < 0.3
    # an all-false mask is no mask
    np.testing.assert_array_equal(
        seed_sources(im, 8.0, np.zeros(im.shape, bool)), seeds
    )
    assert seed_sources(np.full((5, 5), np.nan), 3.0).shape == (0, 4)
    assert seed_sources(im, 8.0, np.ones(im.shape, bool)).shape == (0, 4)
    with pytest.raises(NotImplementedError, match="WMF"):
        seed_sources(im, 8.0, method="wmf")
    with pytest.raises(ValueError, match="method"):
        seed_sources(im, 8.0, method="other")
