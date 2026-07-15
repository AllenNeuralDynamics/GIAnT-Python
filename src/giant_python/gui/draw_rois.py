"""ROI annotation GUI (port of drawROIs.m / annotateROIs.m).

Intended to be backed by napari. Kept in an isolated subpackage so the
headless pipeline never imports GUI dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union


class DrawROIs:
    """Interactive ROI annotation tool (port of the drawROIs.m class).

    Lets the user draw polygon / circle / ellipse ROIs over the mean and
    activity images and saves them to annotations.h5.

    Parameters
    ----------
    trial_table : TrialTable
        The session trial table (provides image paths and save directory).
    """

    def __init__(self, trial_table: "object") -> None:
        self.trial_table = trial_table

    def show(self) -> None:
        """Launch the annotation GUI."""
        raise NotImplementedError

    def save_rois(self, path: Union[str, Path]) -> None:
        """Save drawn ROIs to annotations.h5.

        Parameters
        ----------
        path : str or Path
            Destination path for annotations.h5.
        """
        raise NotImplementedError


def annotate_rois(trial_table: "object") -> None:
    """Launch the ROI annotation GUI for a session (port of annotateROIs.m).

    Parameters
    ----------
    trial_table : TrialTable
        The session trial table.
    """
    raise NotImplementedError
