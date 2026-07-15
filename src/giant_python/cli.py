"""Command-line interface: ``giant organize|register|annotate|extract``.

Thin wrapper over the pipeline stages for batch / cluster use. Wired as the
``giant`` console script in pyproject.toml.
"""

from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``giant`` console script.

    Dispatches the ``organize``, ``register``, ``annotate``, and ``extract``
    subcommands to the corresponding pipeline stages.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Process exit code.
    """
    raise NotImplementedError


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
