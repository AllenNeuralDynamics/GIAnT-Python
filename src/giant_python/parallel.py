"""Per-trial parallelism helper (replacement for MATLAB parfor/parfeval).

Centralizes the choice of parallel backend (e.g. joblib /
``concurrent.futures``) so stages can map a pure per-trial function over
trials without each stage reimplementing the plumbing.
"""

from typing import Callable, Iterable, List, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def _maybe_bar(trials: list, desc: Optional[str]):
    """Wrap ``trials`` in a tqdm bar when ``desc`` is set, else as-is."""
    if desc is None:
        return trials
    from tqdm import tqdm

    return tqdm(trials, desc=desc)


def map_trials(
    func: Callable[[T], R],
    trials: Iterable[T],
    n_workers: int = 1,
    desc: Optional[str] = None,
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
    desc : str, optional
        When set, display a tqdm progress bar with this label as the trials are
        dispatched (used by the verbose band pipeline). ``None`` shows nothing.

    Returns
    -------
    list
        Results in input order.
    """
    trials = list(trials)
    if n_workers <= 1 or len(trials) <= 1:
        return [func(t) for t in _maybe_bar(trials, desc)]

    from joblib import Parallel, delayed

    # ``return_as="generator"`` yields results (in submission order) as each
    # trial *completes*, so wrapping this generator in tqdm tracks real
    # progress. Wrapping the dispatch generator instead would fill the bar as
    # soon as tasks are submitted, long before the workers finish.
    results = Parallel(n_jobs=n_workers, return_as="generator")(
        delayed(func)(t) for t in trials
    )
    if desc is not None:
        from tqdm import tqdm

        results = tqdm(results, desc=desc, total=len(trials))
    return list(results)
