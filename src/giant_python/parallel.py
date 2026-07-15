"""Per-trial parallelism helper (replacement for MATLAB parfor/parfeval).

Centralizes the choice of parallel backend (e.g. joblib /
``concurrent.futures``) so stages can map a pure per-trial function over
trials without each stage reimplementing the plumbing.
"""

from typing import Callable, Iterable, List, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def map_trials(
    func: Callable[[T], R],
    trials: Iterable[T],
    n_workers: int = 1,
) -> List[R]:
    """Apply *func* to each trial, optionally in parallel.

    Parameters
    ----------
    func : callable
        A pure, picklable function applied to a single trial.
    trials : iterable
        The trials to process.
    n_workers : int
        Number of parallel workers. ``1`` runs serially.

    Returns
    -------
    list
        Results in input order.
    """
    raise NotImplementedError
