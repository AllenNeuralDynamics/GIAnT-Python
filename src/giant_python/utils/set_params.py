"""Shared parameter configuration for GIAnT-Python functions."""

from typing import Optional


def set_params(
    fn_name: str,
    params_in: Optional[dict] = None,
) -> dict:
    """Return a parameter dictionary for the named GIAnT function.

    Populates default parameters for the specified function and merges any
    caller-supplied overrides. Corresponds to setParams.m in GIAnT-MATLAB.

    Parameters
    ----------
    fn_name : str
        Name of the GIAnT function whose parameters should be configured.
        Supported values: ``"SELAct"``, ``"MultiRoiRegistration"``,
        ``"StripRegistration"``, ``"BandRegistration"``.
    params_in : dict, optional
        Caller-supplied parameter overrides. Any key present in *params_in*
        takes precedence over the corresponding default value.

    Returns
    -------
    dict
        Parameter dictionary with defaults merged with *params_in*.

    Raises
    ------
    ValueError
        If *fn_name* is not a recognised GIAnT function name.
    """
    raise NotImplementedError
