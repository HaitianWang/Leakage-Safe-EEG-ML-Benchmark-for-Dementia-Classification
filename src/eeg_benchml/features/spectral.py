"""Spectral feature extraction.

Power spectral density is estimated by Welch's method with a Hann window of
4 seconds, 50 % overlap, and ``n_fft = 1024`` (Section 4.3.1 of the manuscript).
For each channel, the extractor produces a compact 14-dimensional descriptor:

* 5 absolute log band powers (delta, theta, alpha, beta, low-gamma).
* 5 relative band powers.
* 1 spectral entropy.
* 1 alpha peak frequency.
* 2 clinically motivated rhythm ratios (theta/alpha, (theta+delta)/(alpha+beta)).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import entropy as scipy_entropy

from ..constants import BAND_ORDER, BANDS, CHANNELS_10_20


@dataclass
class SpectralConfig:
    """Welch PSD configuration."""

    welch_window_s: float = 4.0
    welch_overlap: float = 0.5
    n_fft: int = 1024


def _bandpower(psd: np.ndarray, freqs: np.ndarray, low: float, high: float) -> np.ndarray:
    """Numerically integrate ``psd`` over ``[low, high]`` Hz.

    Parameters
    ----------
    psd : ndarray of shape ``(n_epochs, n_channels, n_freqs)``
        Power spectral density estimates.
    freqs : ndarray of shape ``(n_freqs,)``
        Frequency bins associated with ``psd``.
    """
    mask = (freqs >= low) & (freqs < high)
    if mask.sum() < 2:
        return np.zeros(psd.shape[:2], dtype=float)
    return np.trapz(psd[..., mask], freqs[mask], axis=-1)


def _alpha_peak_frequency(psd: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Per-channel peak frequency within the alpha band (8--13 Hz)."""
    low, high = BANDS["alpha"]
    mask = (freqs >= low) & (freqs <= high)
    if mask.sum() == 0:
        return np.zeros(psd.shape[:2], dtype=float)
    band_psd = psd[..., mask]
    band_freqs = freqs[mask]
    peak_idx = np.argmax(band_psd, axis=-1)
    return band_freqs[peak_idx]


def compute_spectral_features(
    data: np.ndarray, sfreq: float, config: SpectralConfig
) -> Tuple[np.ndarray, List[str]]:
    """Compute per-channel spectral features for a batch of epochs.

    Parameters
    ----------
    data : ndarray of shape ``(n_epochs, n_channels, n_times)``
        EEG epochs.
    sfreq : float
        Sampling frequency in Hz.
    config : SpectralConfig
        Welch PSD configuration.

    Returns
    -------
    features : ndarray of shape ``(n_epochs, n_features)``
    feature_names : list[str]
    """
    n_per_seg = int(round(config.welch_window_s * sfreq))
    n_per_seg = min(n_per_seg, data.shape[-1])
    n_overlap = int(round(n_per_seg * config.welch_overlap))

    freqs, psd = sp_signal.welch(
        data,
        fs=sfreq,
        window="hann",
        nperseg=n_per_seg,
        noverlap=n_overlap,
        nfft=max(config.n_fft, n_per_seg),
        average="mean",
        axis=-1,
    )

    eps = 1e-12
    abs_powers: Dict[str, np.ndarray] = {}
    rel_powers: Dict[str, np.ndarray] = {}
    total_power = _bandpower(psd, freqs, 0.5, 45.0) + eps
    for band in BAND_ORDER:
        low, high = BANDS[band]
        bp = _bandpower(psd, freqs, low, high) + eps
        abs_powers[band] = bp
        rel_powers[band] = bp / total_power

    # Spectral entropy per channel (normalised).
    psd_norm = psd / (np.sum(psd, axis=-1, keepdims=True) + eps)
    spec_entropy = scipy_entropy(psd_norm, axis=-1) / np.log(psd_norm.shape[-1])

    alpha_peak = _alpha_peak_frequency(psd, freqs)

    slow_fast_ratio = abs_powers["theta"] / (abs_powers["alpha"] + eps)
    slowing_index = (
        (abs_powers["theta"] + abs_powers["delta"])
        / (abs_powers["alpha"] + abs_powers["beta"] + eps)
    )

    blocks = []
    feature_names: List[str] = []

    for band in BAND_ORDER:
        blocks.append(np.log10(abs_powers[band]))
        feature_names.extend([f"spec_{ch}_log_power_{band}" for ch in CHANNELS_10_20])
    for band in BAND_ORDER:
        blocks.append(rel_powers[band])
        feature_names.extend([f"spec_{ch}_rel_power_{band}" for ch in CHANNELS_10_20])

    blocks.append(spec_entropy)
    feature_names.extend([f"spec_{ch}_entropy" for ch in CHANNELS_10_20])

    blocks.append(alpha_peak)
    feature_names.extend([f"spec_{ch}_alpha_peak_hz" for ch in CHANNELS_10_20])

    blocks.append(slow_fast_ratio)
    feature_names.extend([f"spec_{ch}_theta_over_alpha" for ch in CHANNELS_10_20])

    blocks.append(slowing_index)
    feature_names.extend(
        [f"spec_{ch}_slowing_index_td_over_ab" for ch in CHANNELS_10_20]
    )

    feature_matrix = np.concatenate(
        [block.reshape(block.shape[0], -1) for block in blocks], axis=-1
    )
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return feature_matrix, feature_names
