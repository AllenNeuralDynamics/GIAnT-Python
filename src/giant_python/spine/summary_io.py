"""Read what the spine-size readout needs from a SILo run's summaries.

A run folder holds ``experiment_summary.h5`` and ``per_trial_summary.h5`` in
GIAnT-MATLAB's schema: per imaging path ``Path<n>``, arrays stored
column-major (``/row_major`` 0, so h5py shows MATLAB's dimensions reversed)
or row-major (``/row_major`` 1). GIAnT-MATLAB writes them, and so does
GIAnT-Python's raster SILo. Arrays here are returned in MATLAB's dimension
order (``size()`` in MATLAB), with images as ``(rows, cols)``; coordinates
are zero-based.

Only the fields used are read, one at a time, because the per-trial file
can hold tens of gigabytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

EXPERIMENT_SUMMARY = "experiment_summary.h5"
PER_TRIAL_SUMMARY = "per_trial_summary.h5"


def _row_major(f: h5py.File) -> bool:
    """Whether a file stores arrays row-major (``/row_major`` 1)."""
    if "row_major" not in f:
        return False
    return bool(np.asarray(f["row_major"][()]).reshape(-1)[0])


def read_array(ds: h5py.Dataset, row_major: bool, index=()) -> np.ndarray:
    """A dataset in MATLAB's dimension order.

    Parameters
    ----------
    ds : h5py.Dataset
        The dataset.
    row_major : bool
        Whether the file stores arrays row-major.
    index : tuple
        Selection in MATLAB's dimension order (slices or integers), applied
        before reading; empty for the whole array.

    Returns
    -------
    ndarray
    """
    if not index:
        data = ds[()]
        return np.asarray(data if row_major else np.transpose(data))
    if row_major:
        return np.asarray(ds[tuple(index)])
    index = tuple(index) + (slice(None),) * (ds.ndim - len(index))
    data = ds[tuple(reversed(index))]
    return np.asarray(np.transpose(data))


def _per_trial(read, edges, n_sources: int, mean: bool) -> np.ndarray:
    """Per-trial mean (``mean``) or number of positive frames of a
    ``(sources, frames)`` trace, reading each trial's frames with
    ``read(start, stop)``; ``(trials, sources)``, NaN for a trial without
    frames and for a source without any value in a trial."""
    res = np.full((len(edges) - 1, n_sources), np.nan)
    for t in range(len(edges) - 1):
        if edges[t + 1] == edges[t]:
            continue
        seg = np.asarray(read(int(edges[t]), int(edges[t + 1])), np.float64)
        ok = np.isfinite(seg)
        has = ok.any(axis=1)
        if mean:
            total = np.where(ok, seg, 0.0).sum(axis=1)
            cnt = np.maximum(ok.sum(axis=1), 1)
            res[t] = np.where(has, total / cnt, np.nan)
        else:
            res[t] = np.where(has, (seg > 0).sum(axis=1), np.nan)
    return res


def _strings(ds: h5py.Dataset) -> list:
    """A cell array or array of strings as a list of str."""
    data = np.asarray(ds[()]).ravel()
    out = []
    for v in data:
        if isinstance(v, bytes):
            out.append(v.decode("utf-8", "replace"))
        elif isinstance(v, h5py.Reference):
            out.append(str(v))
        else:
            out.append(str(v))
    return out


@dataclass
class TrialMap:
    """Each trial's map from the consensus frame to its registered image.

    Built from a summary's ``frame_info``: the displacement field when the
    run saved one (patchwise alignment: trial pixel = consensus pixel +
    displacement), else the affine maps.

    Attributes
    ----------
    affine : ndarray of shape (trials, 2, 3)
        Zero-based ``[row_t, col_t] = affine[t] @ [row, col, 1]``; NaN for
        trials that were not aligned.
    displacement : ndarray of shape (trials, 2, rows, cols), or None
        ``(row, col)`` displacement of every consensus pixel.
    """

    affine: np.ndarray
    displacement: np.ndarray | None = None

    @property
    def n_trials(self) -> int:
        """Number of trials."""
        return self.affine.shape[0]

    @property
    def aligned(self) -> np.ndarray:
        """Which trials have a map (bool per trial), as
        :attr:`~giant_python.extraction.raster.alignment.TrialWarp.aligned`.
        """
        if self.displacement is not None:
            d = self.displacement.reshape(self.displacement.shape[0], -1)
            return np.isfinite(d).any(axis=1)
        return np.isfinite(self.affine).all(axis=(1, 2))

    def grid_coords(
        self, trial: int, shape: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Positions in the trial's registered image of every pixel of the
        ``(rows, cols)`` consensus grid (zero-based)."""
        rr = np.arange(shape[0], dtype=np.float64)[:, None]
        cc = np.arange(shape[1], dtype=np.float64)[None, :]
        rr, cc = np.broadcast_arrays(rr, cc)
        if self.displacement is not None:
            d = self.displacement[trial].astype(np.float64)
            if d.shape[1:] != tuple(shape):
                raise ValueError(
                    f"displacement field {d.shape[1:]} does not match the "
                    f"frame {tuple(shape)}"
                )
            return rr + d[0], cc + d[1]
        a = self.affine[trial]
        return (
            a[0, 0] * rr + a[0, 1] * cc + a[0, 2],
            a[1, 0] * rr + a[1, 1] * cc + a[1, 2],
        )


class RunSummary:
    """One SILo run's summaries (``experiment_summary.h5`` and, when there,
    ``per_trial_summary.h5``).

    Parameters
    ----------
    run_dir : str or Path
        Folder with the summaries.
    """

    def __init__(self, run_dir) -> None:
        """Open a run folder; raise ValueError without an experiment
        summary."""
        self.run_dir = Path(run_dir)
        self.experiment_path = self.run_dir / EXPERIMENT_SUMMARY
        self.per_trial_path = self.run_dir / PER_TRIAL_SUMMARY
        if not self.experiment_path.is_file():
            raise ValueError(f"no {EXPERIMENT_SUMMARY} in {self.run_dir}")
        with h5py.File(self.experiment_path, "r") as f:
            self._row_major = _row_major(f)
            self.paths = sorted(
                int(k[4:])
                for k in f.keys()
                if k.startswith("Path") and k[4:].isdigit()
            )

    def _exp(self) -> h5py.File:
        """The experiment summary, open for reading."""
        return h5py.File(self.experiment_path, "r")

    def _per_trial(self) -> h5py.File:
        """The per-trial summary, open for reading."""
        if not self.per_trial_path.is_file():
            raise ValueError(f"no {PER_TRIAL_SUMMARY} in {self.run_dir}")
        return h5py.File(self.per_trial_path, "r")

    def has_per_trial(self, path: int, name: str) -> bool:
        """Whether the per-trial summary has ``Path<path>/<name>``."""
        if not self.per_trial_path.is_file():
            return False
        with self._per_trial() as f:
            return f"Path{path}/{name}" in f

    def has(self, path: int, name: str) -> bool:
        """Whether the experiment summary has ``Path<path>/<name>``."""
        with self._exp() as f:
            return f"Path{path}/{name}" in f

    def array(self, path: int, name: str, index=()) -> np.ndarray:
        """``Path<path>/<name>`` of the experiment summary, in MATLAB's
        dimension order."""
        with self._exp() as f:
            return read_array(f[f"Path{path}/{name}"], self._row_major, index)

    def per_trial_array(self, path: int, name: str, index=()) -> np.ndarray:
        """``Path<path>/<name>`` of the per-trial summary, in MATLAB's
        dimension order."""
        with self._per_trial() as f:
            row_major = _row_major(f)
            return read_array(f[f"Path{path}/{name}"], row_major, index)

    def shape(self, path: int) -> tuple[int, int]:
        """``(rows, cols)`` of a path's consensus frame."""
        m = self.array(path, "visualizations/mean_im")
        return int(m.shape[-2]), int(m.shape[-1])

    def n_channels(self, path: int) -> int:
        """Number of channels of a path's images."""
        return int(self.array(path, "visualizations/mean_im").shape[0])

    def trial_num_frames(self, path: int) -> np.ndarray:
        """Each trial's number of extraction frames (0 for trials that were
        not extracted)."""
        return self.array(path, "frame_info/trial_num_frames").ravel()

    def n_trials(self, path: int) -> int:
        """Number of trials of a path."""
        return int(self.trial_num_frames(path).size)

    def mean_im(self, path: int) -> np.ndarray:
        """The path's mean image, ``(channels, rows, cols)``."""
        return self.array(path, "visualizations/mean_im")[:, 0]

    def act_im(self, path: int) -> np.ndarray:
        """The path's activity image, ``(rows, cols)``."""
        return self.array(path, "visualizations/act_im")[0]

    def per_trial_images(self, path: int, name: str, channel=None):
        """A per-trial image stack, ``(trials, rows, cols)``.

        Parameters
        ----------
        path : int
            Imaging path.
        name : str
            ``"per_trial_mean_im"``, ``"per_trial_F0_im"`` (both ``T x C x 1
            x H x W``) or ``"per_trial_act_im"`` (``T x 1 x H x W``).
        channel : int, optional
            Zero-based channel, for the images that have channels.
        """
        key = f"visualizations/{name}"
        if channel is None:
            return self.per_trial_array(path, key, (slice(None), 0))
        return self.per_trial_array(path, key, (slice(None), int(channel), 0))

    def profiles(self, path: int) -> np.ndarray:
        """The sources' mean footprints, ``(sources, rows, cols)``."""
        p = self.array(path, "sources/spatial/profiles")
        return np.asarray(p[:, 0], dtype=np.float32)

    def user_rois(self, path: int) -> tuple[list, np.ndarray]:
        """User ROIs: labels and masks ``(rois, rows, cols)`` in the
        consensus frame (empty when the run has none)."""
        if not self.has(path, "user_rois/mask"):
            return [], np.zeros((0,) + self.shape(path), bool)
        with self._exp() as f:
            labels = _strings(f[f"Path{path}/user_rois/labels"])
        masks = self.array(path, "user_rois/mask")[:, 0] > 0
        return labels, masks

    def trial_map(self, path: int, affine_base: int | None = None) -> TrialMap:
        """Each trial's map from the consensus frame to its registered
        image (from ``frame_info``).

        Parameters
        ----------
        path : int
            Imaging path.
        affine_base : {0, 1, None}
            Index base of ``trial_alignment_affine``: GIAnT-Python writes it
            zero-based, GIAnT-MATLAB one-based. It matters only for maps
            that are neither a displacement field nor shifts; None accepts
            those two cases and raises for a true affine map.

        Raises
        ------
        ValueError
            If the maps are affine and ``affine_base`` is None.
        """
        affine = self.array(path, "frame_info/trial_alignment_affine")
        affine = np.asarray(affine, dtype=np.float64).reshape(-1, 2, 3)
        disp = None
        if self.has(path, "frame_info/trial_alignment_displacement"):
            disp = self.array(path, "frame_info/trial_alignment_displacement")
            return TrialMap(affine, disp)
        lin = affine[:, :, :2]
        ok = np.isfinite(lin).all(axis=(1, 2))
        shifts_only = bool(np.allclose(lin[ok], np.eye(2), atol=0, rtol=0))
        if not shifts_only:
            if affine_base is None:
                raise ValueError(
                    "the run's trial maps are affine; give their index base "
                    "(0 for GIAnT-Python, 1 for GIAnT-MATLAB)"
                )
            if int(affine_base) == 1:
                affine = affine.copy()
                affine[:, :, 2] += affine[:, :, 0] + affine[:, :, 1] - 1
        return TrialMap(affine, None)

    def param(self, name: str, default=None):
        """A scalar setting of the run (the summary's ``params/<name>``),
        or ``default`` if it is not recorded."""
        with self._exp() as f:
            key = f"params/{name}"
            if key not in f:
                return default
            v = np.asarray(f[key][()]).ravel()
            return float(v[0]) if v.size else default

    def activity_channel(self) -> int:
        """The run's activity channel (one-based file channel, the
        summary's ``params/activityChannel``; 1 if it is not recorded)."""
        return int(self.param("activityChannel", 1))

    def activity(self, path: int) -> dict:
        """Per-trial activity of each source: the mean of ``dF_ls`` and the
        number of frames with an event, in the run's activity channel,
        ``(trials, sources)``.

        The summaries keep the temporal traces in file-channel order, so the
        channel is :meth:`activity_channel`. NaN for trials that were not
        extracted and for a source without any value in a trial (a trial
        without extracted sources), so that missing traces do not read as
        no activity.
        """
        n = self.trial_num_frames(path).astype(np.int64)
        edges = np.concatenate([[0], np.cumsum(n)])
        ch = self.activity_channel() - 1
        out = {}
        for name, key in (
            ("mean_dF", "sources/temporal/dF_ls"),
            ("event_frames", "sources/temporal/events"),
        ):
            if not self.has(path, key):
                continue
            with self._exp() as f:
                ds = f[f"Path{path}/{key}"]
                n_src = ds.shape[0] if self._row_major else ds.shape[-1]
                # one trial at a time: a session's traces can take GBs
                out[name] = _per_trial(
                    lambda a, b: read_array(
                        ds, self._row_major, (slice(None), ch, slice(a, b))
                    ),
                    edges,
                    n_src,
                    name == "mean_dF",
                )
        return out
