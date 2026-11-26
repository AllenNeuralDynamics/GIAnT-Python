"""Convert make_matlab_kinetics.m's output into matlab_kinetics.h5.

usage: python make_matlab_kinetics_h5.py kinetics_fixture.mat OUT.h5

The MATLAB script saves a v7.3 .mat (HDF5 with MATLAB's dimensions reversed
and vectors as 2-D arrays). This writes the flat datasets the test reads,
in numpy orientation: ``Y`` (rows, n), ``M`` (separations, rows, n),
``seps``, ``LS`` (trials, sources, frames), ``score``, ``nEvents``,
``keep`` (vectors 1-D). A subset of the islocalmax rows is kept (60 with
noise, 60 of whole numbers) to keep the file small.
"""

import sys

import h5py
import numpy as np


def main(src: str, dst: str) -> None:
    """Read the .mat and write the fixture."""
    with h5py.File(src, "r") as f, h5py.File(dst, "w") as g:
        y = f["Y"][()].T
        m = f["M"][()].transpose(2, 1, 0).astype(np.uint8)
        rows = np.r_[0:60, 100:160]
        g.attrs["source"] = (
            "make_matlab_kinetics.m (GIAnT-MATLAB main 5a0673b, MATLAB R2024b)"
        )
        g["Y"] = y[rows]
        g["M"] = m[:, rows]
        g["seps"] = f["seps"][()].ravel()
        g.create_dataset(
            "LS",
            data=f["LS"][()].transpose(2, 1, 0),
            compression="gzip",
            compression_opts=9,
            shuffle=True,
        )
        g["score"] = f["score"][()].ravel()
        g["nEvents"] = f["nEvents"][()].ravel().astype(np.int64)
        g["keep"] = f["keep"][()].ravel().astype(np.uint8)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
