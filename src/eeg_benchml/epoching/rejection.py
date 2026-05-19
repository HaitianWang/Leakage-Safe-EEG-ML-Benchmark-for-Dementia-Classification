"""Automatic epoch rejection based on amplitude and log-variance criteria.

Section 4.2 of the manuscript defines the post-segmentation rejection rule:

* Reject any epoch whose peak-to-peak amplitude exceeds 150 micro-volts on any
  channel.
* Reject any epoch whose log-variance deviates by more than 3.5 standard
  deviations from the subject-level median.

Both criteria are applied within each subject and never use information from
other subjects, so they are leakage-safe.
"""

from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np


@dataclass
class RejectionConfig:
    """Configuration for the rejection step."""

    peak_to_peak_uv: float = 150.0
    log_variance_z_threshold: float = 3.5


def reject_bad_epochs(epochs: mne.Epochs, config: RejectionConfig) -> mne.Epochs:
    """Apply the manuscript's amplitude + log-variance rejection rule.

    Parameters
    ----------
    epochs : mne.Epochs
        Output of :func:`eeg_benchml.epoching.segment.make_fixed_length_epochs`.
    config : RejectionConfig
        Amplitude and log-variance thresholds.

    Returns
    -------
    cleaned : mne.Epochs
        The same object with bad epochs dropped in place.
    """
    if config.peak_to_peak_uv is not None and config.peak_to_peak_uv > 0:
        reject = dict(eeg=config.peak_to_peak_uv * 1.0e-6)
        epochs.drop_bad(reject=reject, verbose=False)

    if len(epochs) == 0:
        return epochs

    data = epochs.get_data(picks="eeg")
    log_var = np.log(np.var(data, axis=-1) + 1e-12).mean(axis=-1)
    median = float(np.median(log_var))
    mad = float(np.median(np.abs(log_var - median)))
    spread = mad * 1.4826 if mad > 0 else float(np.std(log_var) + 1e-12)
    z = (log_var - median) / (spread + 1e-12)

    keep_mask = np.abs(z) <= config.log_variance_z_threshold
    if not np.all(keep_mask):
        drop_indices = np.where(~keep_mask)[0]
        epochs.drop(drop_indices, verbose=False)
    return epochs
