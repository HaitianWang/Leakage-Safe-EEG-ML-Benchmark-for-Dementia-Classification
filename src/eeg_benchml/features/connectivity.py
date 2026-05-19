"""Pairwise connectivity features.

The weighted phase lag index (wPLI) is computed for each unordered channel
pair and each frequency band (Equation 5 of the manuscript):

.. math::

   \\mathrm{wPLI}_{cd}^{(b)} =
   \\frac{
       \\left| \\mathbb{E}\\left[ |\\Im(Z_{cd}^{(b)})| \\cdot
                 \\mathrm{sign}(\\Im(Z_{cd}^{(b)})) \\right] \\right|
   }{
       \\mathbb{E}\\left[ |\\Im(Z_{cd}^{(b)})| \\right] + \\epsilon
   }.

For the 19-channel 10--20 montage this yields :math:`171 \\times 5 = 855`
descriptors per epoch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy import signal as sp_signal

from ..constants import BAND_ORDER, BANDS, CHANNELS_10_20, list_channel_pairs


@dataclass
class ConnectivityConfig:
    """Configuration for the wPLI extractor."""

    metric: str = "wpli"
    eps: float = 1.0e-8


def _band_filter(
    data: np.ndarray, sfreq: float, low: float, high: float
) -> np.ndarray:
    """Apply a zero-phase Butterworth band-pass for connectivity estimation."""
    nyq = 0.5 * sfreq
    low_n = max(low / nyq, 1e-4)
    high_n = min(high / nyq, 0.999)
    if low_n >= high_n:
        return data
    sos = sp_signal.butter(N=4, Wn=(low_n, high_n), btype="bandpass", output="sos")
    return sp_signal.sosfiltfilt(sos, data, axis=-1)


def _wpli_for_band(
    data: np.ndarray, eps: float
) -> np.ndarray:
    """Compute wPLI for every pair of channels for a band-limited signal.

    Parameters
    ----------
    data : ndarray of shape ``(n_epochs, n_channels, n_times)``
        Band-limited EEG signal.
    eps : float
        Numerical stabiliser added to the denominator.

    Returns
    -------
    wpli : ndarray of shape ``(n_epochs, n_pairs)``
    """
    analytic = sp_signal.hilbert(data, axis=-1)
    n_epochs, n_channels, _ = analytic.shape
    pairs = [
        (i, j)
        for i in range(n_channels)
        for j in range(i + 1, n_channels)
    ]
    out = np.zeros((n_epochs, len(pairs)), dtype=float)
    for k, (i, j) in enumerate(pairs):
        cross = analytic[:, i, :] * np.conj(analytic[:, j, :])
        imag = cross.imag
        num = np.abs(np.mean(np.abs(imag) * np.sign(imag), axis=-1))
        denom = np.mean(np.abs(imag), axis=-1) + eps
        out[:, k] = num / denom
    return out


def compute_connectivity_features(
    data: np.ndarray, sfreq: float, config: ConnectivityConfig
) -> Tuple[np.ndarray, List[str]]:
    """Compute pairwise wPLI features for all channel pairs and frequency bands.

    Parameters
    ----------
    data : ndarray of shape ``(n_epochs, n_channels, n_times)``
    sfreq : float
    config : ConnectivityConfig

    Returns
    -------
    features : ndarray of shape ``(n_epochs, 855)`` with 5 bands x 171 pairs.
    feature_names : list[str]
    """
    channel_pairs = list_channel_pairs()
    n_epochs = data.shape[0]
    blocks = []
    feature_names: List[str] = []
    for band in BAND_ORDER:
        low, high = BANDS[band]
        filtered = _band_filter(data, sfreq=sfreq, low=low, high=high)
        wpli = _wpli_for_band(filtered, eps=config.eps)
        blocks.append(wpli)
        feature_names.extend(
            [f"conn_wpli_{band}_{a}-{b}" for (a, b) in channel_pairs]
        )

    feature_matrix = np.concatenate(blocks, axis=-1)
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    assert feature_matrix.shape == (n_epochs, len(feature_names))
    assert feature_matrix.shape[1] == len(CHANNELS_10_20) * (len(CHANNELS_10_20) - 1) // 2 * len(BAND_ORDER)
    return feature_matrix, feature_names
