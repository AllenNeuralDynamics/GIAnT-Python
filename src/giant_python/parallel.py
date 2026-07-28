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

    Runs serially when ``n_workers <= 1`` (or a single trial), otherwise fans
    the trials out over a joblib process pool. Results are returned in input
    order regardless of backend.

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
    trials = list(trials)
    if n_workers <= 1 or len(trials) <= 1:
        return [func(t) for t in trials]

    from joblib import Parallel, delayed

    return Parallel(n_jobs=n_workers)(delayed(func)(t) for t in trials)
