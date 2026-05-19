"""Lightweight timing utility.

Used by the pipeline to report CPU-based feature-extraction time and
per-subject prediction time, matching the ``Time (s/subject)`` column in
Table 2 of the manuscript.
"""

from __future__ import annotations

import time
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Timer(ContextDecorator):
    """Context manager that records wall-clock time for a code block.

    Examples
    --------
    >>> with Timer("feature_extraction") as t:
    ...     run_feature_extraction()
    >>> t.elapsed_s
    23.7
    """

    label: str = "block"
    elapsed_s: float = field(default=0.0, init=False)
    _start: Optional[float] = field(default=None, init=False, repr=False)

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self._start is not None
        self.elapsed_s = time.perf_counter() - self._start
