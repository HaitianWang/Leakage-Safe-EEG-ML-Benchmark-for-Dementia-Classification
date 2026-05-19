"""Deterministic seeding utilities.

The classical ML pipeline contains multiple stochastic components (ICA random
initialisation, sigmoid calibration shuffling, bootstrap resampling). A single
``set_global_seed`` call at the start of an experiment fixes the random state
for ``random``, ``numpy``, and (optionally) any framework-specific RNG.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python's :mod:`random`, NumPy, and the ``PYTHONHASHSEED`` env var.

    Parameters
    ----------
    seed : int
        Non-negative integer seed value.
    """
    if seed < 0:
        raise ValueError("Seed must be a non-negative integer.")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
