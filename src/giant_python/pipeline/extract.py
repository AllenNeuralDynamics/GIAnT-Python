"""Stage 3 - source extraction / SILo (port of SILo.m).

Exposes the porting-faithful extraction functions plus a
:class:`SourceExtractor` base with one backend per SLAP2 scan mode
(standard pixel-movie SILo / integration band-scan SILo), selected via
:meth:`SourceExtractor.for_scan_mode` -- mirroring the
:class:`~giant_python.pipeline.register.MotionCorrector` backend pattern.

Both backends share the same algorithmic skeleton (activity image -> peak
detection -> source localization -> trace extraction -> baseline/dF ->
deconvolution); the shared steps live as pure kernels in
:mod:`giant_python.math` (``peaks``, ``baseline``, ``deconv``) so that only
the scan-mode-specific data loading and geometry differ between backends.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from ..models.params import SiloParams
from .base import Stage


def silo(
    dr_or_path_to_trial_table: Optional[Union[str, Path]] = None,
    params_in: Optional[dict] = None,
) -> None:
    """Perform standard (pixel-movie) source extraction and localization.

    Detects active synaptic boutons or spines from registered movies using
    peak-finding on a per-trial activity image followed by NMF-based trace
    refinement across trials. Results are saved as an experiment summary
    file. Corresponds to SILo.m in GIAnT-MATLAB.

    Parameters
    ----------
    dr_or_path_to_trial_table : str or Path, optional
        Either a directory containing a trial table file, or the full path
        to a trial table file.
    params_in : dict, optional
        Parameter overrides. Keys and default values are defined in
        :func:`giant_python.models.params.set_params` under ``"SILo"``.
    """
    raise NotImplementedError


def extract_integration_sources(
    path_to_trial_table: Optional[Union[str, Path]] = None,
    params_in: Optional[dict] = None,
) -> None:
    """Perform integration (band-scan) source extraction for SLAP2.

    The band-scan variant of SILo: operates on superpixel / DMD-geometry
    data and the integration lookup table produced by
    :func:`giant_python.pipeline.register.integration_registration`
    (``fnAdataInt``), rather than on a reconstructed pixel movie. Shares the
    peak-detection, baseline, and deconvolution kernels with :func:`silo`.
    Corresponds to ``extractSLAP2IntegrationSources.py`` in
    ophys-slap2-analysis.

    Parameters
    ----------
    path_to_trial_table : str or Path, optional
        Full path to the trial table file (with integration alignment data).
    params_in : dict, optional
        Parameter overrides. Keys and default values are defined in
        :func:`giant_python.models.params.set_params`.
    """
    raise NotImplementedError


class SourceExtractor(Stage):
    """Stage 3 base class: extract sources from motion-corrected trials.

    Use :meth:`for_scan_mode` to obtain the appropriate backend. Concrete
    subclasses wrap the corresponding extraction function.

    Parameters
    ----------
    params : SiloParams, optional
        Source-extraction parameters; defaults to :class:`SiloParams`
        defaults.
    """

    def __init__(self, params: Optional[SiloParams] = None) -> None:
        self.params = params or SiloParams()

    @staticmethod
    def for_scan_mode(
        scan_mode: str,
        params: Optional[SiloParams] = None,
    ) -> "SourceExtractor":
        """Return the extraction backend for the given SLAP2 scan mode.

        Parameters
        ----------
        scan_mode : str
            ``"standard"`` (-> :class:`StandardSourceExtractor`) or
            ``"integration"`` (-> :class:`IntegrationSourceExtractor`).
        params : SiloParams, optional
            Source-extraction parameters.

        Returns
        -------
        SourceExtractor
            The appropriate backend instance.
        """
        raise NotImplementedError

    def run(self, trial_table: "object") -> "object":
        """Extract sources and write experiment_summary.h5.

        Parameters
        ----------
        trial_table : TrialTable
            A trial table whose trials have been motion-corrected.

        Returns
        -------
        ExperimentSummary
            The extracted sources and summary (also persisted to disk).
        """
        raise NotImplementedError


class StandardSourceExtractor(SourceExtractor):
    """Standard pixel-movie SILo backend (wraps :func:`silo`)."""

    def run(self, trial_table: "object") -> "object":
        """Extract from pixel movies. See :meth:`SourceExtractor.run`."""
        raise NotImplementedError


class IntegrationSourceExtractor(SourceExtractor):
    """Integration (band-scan) SILo backend.

    Wraps :func:`extract_integration_sources`.
    """

    def run(self, trial_table: "object") -> "object":
        """Extract from integration data. See :meth:`SourceExtractor.run`."""
        raise NotImplementedError
