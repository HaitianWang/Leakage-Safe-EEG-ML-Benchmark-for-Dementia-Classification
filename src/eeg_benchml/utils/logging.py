"""Minimal structured logging helpers.

The package-wide logger uses a deterministic single-line format so that
benchmark runs remain diffable across machines. The logger name is hierarchical
and matches the package layout (``eeg_benchml.<submodule>``).
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

_DEFAULT_FORMAT = "[%(asctime)s] %(levelname)-7s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%dT%H:%M:%S"


def get_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger for the given module name.

    Parameters
    ----------
    name : str, optional
        Logger name. ``None`` returns the root package logger.
    level : int
        Logging level. Defaults to :data:`logging.INFO`.

    Returns
    -------
    logger : logging.Logger
        A logger with a single ``stderr`` handler attached. Repeated calls
        with the same name return the same instance.
    """
    logger = logging.getLogger(name if name else "eeg_benchml")

    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(
            logging.Formatter(fmt=_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)
        )
        logger.addHandler(handler)
        logger.propagate = False

    logger.setLevel(level)
    return logger
