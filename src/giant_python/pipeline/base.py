"""Abstract base class for pipeline stages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Stage(ABC):
    """Base class for a single pipeline stage.

    A stage bundles its parameters and exposes a single :meth:`run` method.
    Concrete stages (``TrialTableBuilder``, ``MotionCorrector``,
    ``SourceExtractor``) delegate the actual computation to the functional
    core and the math kernels. Stages are independently usable for partial
    re-runs; the :class:`~giant_python.pipeline.pipeline.Pipeline` facade
    composes them for the common end-to-end case.
    """

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the stage and return its result."""
        raise NotImplementedError  # pragma: no cover - abstract stub
