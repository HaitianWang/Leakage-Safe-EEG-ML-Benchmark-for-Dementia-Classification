"""Time-domain complexity features.

For each channel and epoch, this module produces a compact **25-dimensional**
descriptor (Section 4.3.2 of the manuscript) computed from both broadband and
band-limited epochs:

* **Broadband descriptors (10)**: Hjorth activity (log), Hjorth mobility,
  Hjorth complexity, sample entropy, Higuchi fractal dimension, and
  multi-scale sample entropy at scales :math:`\\{1, 2, 3, 4, 5\\}`.
* **Band-limited descriptors (15)**: Hjorth mobility, Hjorth complexity, and
  sample entropy computed on each of the five clinical EEG bands (delta,
  theta, alpha, beta, low-gamma) for :math:`5 \\times 3 = 15` additional
  features.

With 19 channels of the 10--20 montage this yields :math:`25 \\times 19 = 475`
complexity features, matching the ``Dim.`` column for the
``Complexity`` and ``Spectral + complexity`` ablation rows in Table 2.

Sample entropy is parameterised with embedding dimension :math:`m = 2` and
tolerance :math:`r = 0.2 \\sigma`, where :math:`\\sigma` is the standard
deviation of the analysed segment (Equation 4 of the manuscript). Higuchi
fractal dimension uses :math:`k_{\\max} = 8`.

The implementations below favour clarity over raw speed; they rely solely on
NumPy / SciPy and are deterministic given a fixed input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy import signal as sp_signal

from ..constants import BAND_ORDER, BANDS, CHANNELS_10_20

# Bands used for the band-limited complexity descriptors. Following the
# manuscript, we include all five canonical EEG bands.
_BAND_LIMITED_KEYS: Tuple[str, ...] = BAND_ORDER


@dataclass
class ComplexityConfig:
    """Configuration for the complexity feature extractor."""

    sample_entropy_m: int = 2
    sample_entropy_r: float = 0.2
    higuchi_kmax: int = 8
    multi_scale_entropy_scales: Tuple[int, ...] = (1, 2, 3, 4, 5)


# ---------------------------------------------------------------------------
# Primitive feature implementations (per-channel, per-epoch).
# ---------------------------------------------------------------------------
def _hjorth(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hjorth activity, mobility, and complexity per epoch and channel.

    Activity is the variance of the signal; mobility is
    :math:`\\sqrt{\\mathrm{Var}(\\dot{x}) / \\mathrm{Var}(x)}`; complexity is
    the ratio between the mobility of :math:`\\dot{x}` and the mobility of
    :math:`x` (Section 4.3.2 of the manuscript).
    """
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


def _sample_entropy_1d(signal: np.ndarray, m: int, r_abs: float) -> float:
    """Sample entropy for a 1-D signal (Equation 4 of the manuscript)."""
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
    """Compact multi-scale sample-entropy vector across the requested scales."""
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
    """Higuchi fractal dimension with cap :math:`k_{\\max}`."""
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


def _band_filter(
    data: np.ndarray, sfreq: float, low: float, high: float
) -> np.ndarray:
    """Apply a zero-phase Butterworth band-pass to a batch of epochs."""
    nyq = 0.5 * sfreq
    low_n = max(low / nyq, 1e-4)
    high_n = min(high / nyq, 0.999)
    if low_n >= high_n:
        return data
    sos = sp_signal.butter(N=4, Wn=(low_n, high_n), btype="bandpass", output="sos")
    return sp_signal.sosfiltfilt(sos, data, axis=-1)


# ---------------------------------------------------------------------------
# Composite feature extractor.
# ---------------------------------------------------------------------------
def compute_complexity_features(
    data: np.ndarray,
    config: ComplexityConfig,
    sfreq: float = 250.0,
) -> Tuple[np.ndarray, List[str]]:
    """Compute the per-channel 25-dimensional complexity descriptor.

    Parameters
    ----------
    data : ndarray of shape ``(n_epochs, n_channels, n_times)``
        Preprocessed EEG epochs.
    config : ComplexityConfig
        Complexity feature configuration (Section 4.3.2 of the manuscript).
    sfreq : float
        Sampling frequency, required by the band-pass filters for the
        band-limited Hjorth and sample-entropy descriptors.

    Returns
    -------
    features : ndarray of shape ``(n_epochs, 25 * n_channels)``
    feature_names : list[str]
        Column names with the prefix ``cmpl_<channel>_<descriptor>[_<band>]``,
        so that downstream interpretation can recover both the channel and
        the optional band.
    """
    n_epochs, n_channels, _ = data.shape
    feature_names: List[str] = []
    blocks: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Block 1: broadband descriptors (10 features per channel).
    # ------------------------------------------------------------------
    activity, mobility, complexity = _hjorth(data)
    blocks.extend([np.log10(activity + 1e-12), mobility, complexity])
    for descriptor in ("hjorth_log_activity", "hjorth_mobility", "hjorth_complexity"):
        feature_names.extend([f"cmpl_{ch}_{descriptor}" for ch in CHANNELS_10_20])

    sample_entropy = np.zeros((n_epochs, n_channels), dtype=float)
    higuchi = np.zeros((n_epochs, n_channels), dtype=float)
    mse = np.zeros(
        (n_epochs, n_channels, len(config.multi_scale_entropy_scales)),
        dtype=float,
    )
    for i in range(n_epochs):
        epoch_std = data[i].std(axis=-1, keepdims=True)
        for c in range(n_channels):
            r_abs = float(config.sample_entropy_r * (epoch_std[c, 0] + 1e-12))
            sample_entropy[i, c] = _sample_entropy_1d(
                data[i, c], m=config.sample_entropy_m, r_abs=r_abs
            )
            higuchi[i, c] = _higuchi_fractal_dimension(
                data[i, c], kmax=config.higuchi_kmax
            )
            mse[i, c, :] = _multiscale_entropy_1d(
                data[i, c],
                scales=tuple(config.multi_scale_entropy_scales),
                m=config.sample_entropy_m,
                r_abs=r_abs,
            )

    blocks.append(sample_entropy)
    feature_names.extend([f"cmpl_{ch}_sample_entropy" for ch in CHANNELS_10_20])
    blocks.append(higuchi)
    feature_names.extend([f"cmpl_{ch}_higuchi_fd" for ch in CHANNELS_10_20])
    for scale_idx, scale in enumerate(config.multi_scale_entropy_scales):
        blocks.append(mse[..., scale_idx])
        feature_names.extend(
            [f"cmpl_{ch}_mse_scale{scale}" for ch in CHANNELS_10_20]
        )

    # ------------------------------------------------------------------
    # Block 2: band-limited descriptors (15 features per channel).
    #
    # For each of the five clinical EEG bands we re-compute Hjorth mobility,
    # Hjorth complexity, and sample entropy on the band-limited signal. This
    # gives ``5 bands x 3 descriptors = 15`` additional descriptors per
    # channel, matching the manuscript's "compact 25-dimensional descriptor
    # set per channel from broadband and selected band-limited epochs".
    # ------------------------------------------------------------------
    for band in _BAND_LIMITED_KEYS:
        low, high = BANDS[band]
        band_data = _band_filter(data, sfreq=sfreq, low=low, high=high)
        _, band_mobility, band_complexity = _hjorth(band_data)
        blocks.append(band_mobility)
        feature_names.extend(
            [f"cmpl_{ch}_hjorth_mobility_{band}" for ch in CHANNELS_10_20]
        )
        blocks.append(band_complexity)
        feature_names.extend(
            [f"cmpl_{ch}_hjorth_complexity_{band}" for ch in CHANNELS_10_20]
        )

        band_sampen = np.zeros((n_epochs, n_channels), dtype=float)
        for i in range(n_epochs):
            band_std = band_data[i].std(axis=-1, keepdims=True)
            for c in range(n_channels):
                r_abs = float(
                    config.sample_entropy_r * (band_std[c, 0] + 1e-12)
                )
                band_sampen[i, c] = _sample_entropy_1d(
                    band_data[i, c],
                    m=config.sample_entropy_m,
                    r_abs=r_abs,
                )
        blocks.append(band_sampen)
        feature_names.extend(
            [f"cmpl_{ch}_sample_entropy_{band}" for ch in CHANNELS_10_20]
        )

    feature_matrix = np.concatenate(
        [block.reshape(n_epochs, -1) for block in blocks], axis=-1
    )
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    # Sanity check: the descriptor count must match the manuscript's 25 per
    # channel reporting.
    expected = 25 * n_channels
    if feature_matrix.shape[1] != expected:
        raise RuntimeError(
            f"Complexity feature dimensionality mismatch: produced "
            f"{feature_matrix.shape[1]} columns, expected {expected} "
            f"(25 descriptors per channel x {n_channels} channels)."
        )
    return feature_matrix, feature_names
