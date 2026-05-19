"""Time-domain complexity features.

For each channel and epoch, this module produces a compact 25-dimensional
descriptor (Section 4.3.2 of the manuscript):

* Hjorth activity (log), mobility, and complexity.
* Zero-crossing rate.
* Sample entropy (Equation 4 of the manuscript).
* Higuchi fractal dimension with :math:`k_{\\max} = 8`.
* Multi-scale entropy at scales :math:`\\{1, 2, 3, 4, 5\\}` (kept as a compact
  summary by averaging across scales when needed downstream).

The implementations below favour clarity over raw speed; they rely solely on
NumPy / SciPy and are deterministic given a fixed input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from ..constants import CHANNELS_10_20


@dataclass
class ComplexityConfig:
    """Configuration for the complexity feature extractor."""

    sample_entropy_m: int = 2
    sample_entropy_r: float = 0.2
    higuchi_kmax: int = 8
    multi_scale_entropy_scales: Tuple[int, ...] = (1, 2, 3, 4, 5)


def _hjorth(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hjorth activity, mobility, and complexity per epoch and channel."""
    eps = 1e-12
    diff1 = np.diff(data, axis=-1)
    diff2 = np.diff(diff1, axis=-1)

    var0 = np.var(data, axis=-1)
    var1 = np.var(diff1, axis=-1)
    var2 = np.var(diff2, axis=-1)

    activity = var0
    mobility = np.sqrt(var1 / (var0 + eps))
    complexity = np.sqrt(var2 / (var1 + eps)) / (mobility + eps)
    return activity, mobility, complexity


def _zero_crossing_rate(data: np.ndarray) -> np.ndarray:
    """Number of sign changes per sample, per epoch and channel."""
    centred = data - data.mean(axis=-1, keepdims=True)
    signs = np.sign(centred)
    return (np.diff(signs, axis=-1) != 0).mean(axis=-1)


def _sample_entropy_1d(signal: np.ndarray, m: int, r_abs: float) -> float:
    """Sample entropy for a 1-D signal (used inside per-channel loops)."""
    n = signal.shape[0]
    if n <= m + 1:
        return 0.0

    def _phi(length: int) -> float:
        templates = np.lib.stride_tricks.sliding_window_view(signal, length)
        diff = np.abs(templates[:, None, :] - templates[None, :, :]).max(axis=-1)
        np.fill_diagonal(diff, np.inf)
        matches = (diff <= r_abs).sum()
        return float(matches) / 2.0  # symmetric pairs counted once

    b_m = _phi(m)
    b_m1 = _phi(m + 1)
    if b_m == 0 or b_m1 == 0:
        return 0.0
    return float(-np.log(b_m1 / b_m))


def _multiscale_entropy_1d(
    signal: np.ndarray, scales: Tuple[int, ...], m: int, r_abs: float
) -> np.ndarray:
    """Compact multi-scale sample-entropy vector."""
    out = np.zeros(len(scales), dtype=float)
    for idx, scale in enumerate(scales):
        if scale == 1:
            coarse = signal
        else:
            usable_length = (signal.shape[0] // scale) * scale
            if usable_length == 0:
                out[idx] = 0.0
                continue
            coarse = signal[:usable_length].reshape(-1, scale).mean(axis=1)
        out[idx] = _sample_entropy_1d(coarse, m=m, r_abs=r_abs)
    return out


def _higuchi_fractal_dimension(signal: np.ndarray, kmax: int) -> float:
    """Higuchi fractal dimension of a 1-D signal."""
    n = signal.shape[0]
    if n < 2 * kmax:
        return 0.0
    lk = np.zeros(kmax, dtype=float)
    for k in range(1, kmax + 1):
        lm = np.zeros(k, dtype=float)
        for m in range(k):
            indices = np.arange(1, int(np.floor((n - m - 1) / k)) + 1)
            if indices.size < 2:
                lm[m] = 0.0
                continue
            diffs = np.abs(signal[m + indices * k] - signal[m + (indices - 1) * k])
            length = diffs.sum() * (n - 1) / (k * indices.size * k)
            lm[m] = length
        lk[k - 1] = lm.mean()
    log_k = np.log(np.arange(1, kmax + 1))
    log_lk = np.log(lk + 1e-12)
    slope, _ = np.polyfit(log_k, log_lk, 1)
    return float(-slope)


def compute_complexity_features(
    data: np.ndarray, config: ComplexityConfig
) -> Tuple[np.ndarray, List[str]]:
    """Compute the per-channel complexity descriptor for a batch of epochs.

    Parameters
    ----------
    data : ndarray of shape ``(n_epochs, n_channels, n_times)``
    config : ComplexityConfig

    Returns
    -------
    features : ndarray of shape ``(n_epochs, n_features)``
    feature_names : list[str]
    """
    n_epochs, n_channels, _ = data.shape
    feature_names: List[str] = []

    activity, mobility, complexity = _hjorth(data)
    zcr = _zero_crossing_rate(data)

    sample_entropy = np.zeros((n_epochs, n_channels), dtype=float)
    fractal_dim = np.zeros((n_epochs, n_channels), dtype=float)
    mse = np.zeros(
        (n_epochs, n_channels, len(config.multi_scale_entropy_scales)),
        dtype=float,
    )

    for i in range(n_epochs):
        epoch_std = data[i].std(axis=-1, keepdims=True)
        for c in range(n_channels):
            signal = data[i, c]
            r_abs = float(config.sample_entropy_r * (epoch_std[c, 0] + 1e-12))
            sample_entropy[i, c] = _sample_entropy_1d(
                signal, m=config.sample_entropy_m, r_abs=r_abs
            )
            fractal_dim[i, c] = _higuchi_fractal_dimension(
                signal, kmax=config.higuchi_kmax
            )
            mse[i, c, :] = _multiscale_entropy_1d(
                signal,
                scales=tuple(config.multi_scale_entropy_scales),
                m=config.sample_entropy_m,
                r_abs=r_abs,
            )

    blocks = [
        np.log10(activity + 1e-12),
        mobility,
        complexity,
        zcr,
        sample_entropy,
        fractal_dim,
    ]
    base_names = [
        "hjorth_log_activity",
        "hjorth_mobility",
        "hjorth_complexity",
        "zero_crossing_rate",
        "sample_entropy",
        "higuchi_fd",
    ]
    for name in base_names:
        feature_names.extend([f"cmpl_{ch}_{name}" for ch in CHANNELS_10_20])

    for scale_idx, scale in enumerate(config.multi_scale_entropy_scales):
        blocks.append(mse[..., scale_idx])
        feature_names.extend(
            [f"cmpl_{ch}_mse_scale{scale}" for ch in CHANNELS_10_20]
        )

    feature_matrix = np.concatenate(
        [block.reshape(n_epochs, -1) for block in blocks], axis=-1
    )
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return feature_matrix, feature_names
