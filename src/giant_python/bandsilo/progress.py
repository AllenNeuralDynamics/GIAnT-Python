"""Opt-in verbose logging + progress bars for the band pipeline.

The BandSILo port stays quiet by default (headless / batch friendly); the
reference's unconditional ``print``s and ``tqdm`` bars were dropped. These
helpers restore that feedback opt-in: when ``verbose`` is set
(``SiloParams.verbose`` / ``giant ... --verbose``) stage messages are printed
and long-running loops show a progress bar. ``tqdm`` is imported lazily, so a
non-verbose run neither prints nor imports it.
"""

from __future__ import annotations

from typing import Iterable, Optional


def log(message: str, verbose: bool) -> None:
    """Print ``message`` when ``verbose`` is set; otherwise do nothing.

    Parameters
    ----------
    message : str
        The status message to display.
    verbose : bool
        Whether verbose output is enabled.
    """
    if verbose:
        print(message)


def progress(
    iterable: Iterable,
    *,
    desc: str,
    verbose: bool,
    total: Optional[int] = None,
) -> Iterable:
    """Wrap ``iterable`` in a tqdm progress bar when ``verbose`` is set.

    Returns the iterable unchanged when ``verbose`` is ``False`` so callers can
    wrap any loop unconditionally at zero cost (and without importing
    ``tqdm``).

    Parameters
    ----------
    iterable : iterable
        The loop iterable to (optionally) wrap.
    desc : str
        Progress-bar label.
    verbose : bool
        Whether to show the bar.
    total : int, optional
        Item count, for iterables without a ``len`` (e.g. a generator).

    Returns
    -------
    iterable
        The tqdm-wrapped iterable, or the original iterable.
    """
    if not verbose:
        return iterable
    from tqdm import tqdm

    return tqdm(iterable, desc=desc, total=total)
