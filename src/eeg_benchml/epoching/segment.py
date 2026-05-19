"""Fixed-length epoching utilities.

Equation 1 of the manuscript defines the per-subject epoch set:

.. math::

   \\mathcal{E}_i =
   \\{ E_{i,j} = X_i[:, jh \\ldots jh + L - 1]
        \\mid j = 0, \\ldots, \\lfloor (T_i - L) / h \\rfloor \\},
   \\quad L = f_s \\tau, \\quad h = L / 2.

The reference configuration uses :math:`\\tau = 10` s with 50 % overlap.
Epoch lengths of 5, 20, and 30 s are exposed for the Stage B ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mne


@dataclass
class EpochingConfig:
    """Parameters controlling fixed-length epoching."""

    duration_s: float = 10.0
    overlap: float = 0.5

    def stride_s(self) -> float:
        """Stride between consecutive epoch starts, in seconds."""
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError(
                f"Epoch overlap must be in [0, 1); got {self.overlap}."
            )
        return self.duration_s * (1.0 - self.overlap)


def make_fixed_length_epochs(
    raw: mne.io.BaseRaw, config: EpochingConfig
) -> Optional[mne.Epochs]:
    """Segment a continuous recording into fixed-length epochs.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Cleaned recording returned by the preprocessing stage.
    config : EpochingConfig
        Epoch duration and overlap.

    Returns
    -------
    epochs : mne.Epochs or None
        ``None`` when the recording is too short to yield any epoch.
    """
    try:
        epochs = mne.make_fixed_length_epochs(
            raw,
            duration=config.duration_s,
            overlap=config.duration_s * config.overlap,
            preload=True,
            verbose=False,
        )
    except Exception:
        return None
    if len(epochs) == 0:
        return None
    return epochs
