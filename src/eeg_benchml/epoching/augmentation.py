"""Training-only EEG epoch augmentation.

Equation 2 of the manuscript defines the augmentation rule:

.. math::

   \\widetilde{E}_{i,j,c}(t) = a_c \\, E_{i,j,c}(t) + \\epsilon_c(t),
   \\quad a_c \\sim \\mathcal{U}(0.95, 1.05),
   \\quad \\epsilon_c(t) \\sim \\mathcal{N}(0, (0.01 \\sigma_c)^2).

The augmentation is applied per training epoch and per channel, generating one
additional augmented copy of each training epoch. Validation and test subjects
are never modified, which guarantees leakage safety.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass
class AugmentationConfig:
    """Configuration for the training-only augmentation step."""

    enabled: bool = True
    amplitude_scaling_range: Tuple[float, float] = (0.95, 1.05)
    gaussian_noise_sigma: float = 0.01
    random_state: int = 42

    def rng(self) -> np.random.Generator:
        """Return a deterministic NumPy RNG for this configuration."""
        return np.random.default_rng(self.random_state)


@dataclass
class AugmentedEpochArrays:
    """Container returned by :func:`augment_training_epochs`."""

    data: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    labels: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    groups: np.ndarray = field(default_factory=lambda: np.empty((0,)))


def augment_training_epochs(
    data: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    config: AugmentationConfig,
) -> AugmentedEpochArrays:
    """Generate one augmented copy of each training epoch.

    Parameters
    ----------
    data : ndarray of shape ``(n_epochs, n_channels, n_times)``
        Training epochs (raw waveforms after preprocessing).
    labels : ndarray of shape ``(n_epochs,)``
        Epoch-level labels.
    groups : ndarray of shape ``(n_epochs,)``
        Subject identifiers used to keep epochs grouped during validation.
    config : AugmentationConfig
        Augmentation hyperparameters. When ``config.enabled`` is ``False``,
        the function returns the input arrays unchanged.

    Returns
    -------
    augmented : AugmentedEpochArrays
        Concatenation of the original epochs and their augmented copies.
    """
    if not config.enabled or data.size == 0:
        return AugmentedEpochArrays(data=data, labels=labels, groups=groups)

    rng = config.rng()
    n_epochs, n_channels, _ = data.shape

    low, high = config.amplitude_scaling_range
    scaling = rng.uniform(low=low, high=high, size=(n_epochs, n_channels, 1))

    sigma_per_channel = data.std(axis=-1, keepdims=True)
    noise = rng.normal(
        loc=0.0,
        scale=config.gaussian_noise_sigma * sigma_per_channel,
        size=data.shape,
    )

    augmented = data * scaling + noise
    return AugmentedEpochArrays(
        data=np.concatenate([data, augmented], axis=0),
        labels=np.concatenate([labels, labels], axis=0),
        groups=np.concatenate([groups, groups], axis=0),
    )
