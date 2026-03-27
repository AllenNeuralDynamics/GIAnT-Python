"""Source extraction and localization of synaptic activity signals."""

from pathlib import Path
from typing import Optional, Union


def sel_act(
    dr_or_path_to_trial_table: Optional[Union[str, Path]] = None,
    params_in: Optional[dict] = None,
) -> None:
    """Perform source extraction and localization of synaptic activity.

    Detects active synaptic boutons or spines from registered movies using
    peak-finding on a per-trial activity image followed by NMF-based trace
    refinement across trials. Results are saved as an experiment summary
    file. Corresponds to SELAct.m in GIAnT-MATLAB.

    Parameters
    ----------
    dr_or_path_to_trial_table : str or Path, optional
        Either a directory containing a trial table file, or the full path
        to a trial table file.
    params_in : dict, optional
        Parameter overrides. Keys and default values are defined in
        :func:`giant_python.utils.set_params` under ``"SELAct"``.
    """
    raise NotImplementedError
