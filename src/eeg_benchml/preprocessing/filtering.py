"""Linear filtering, resampling, and referencing helpers.

The functions in this module are thin wrappers around the MNE-Python API.
They are factored out so that the higher-level
:func:`eeg_benchml.preprocessing.pipeline.preprocess_subject` orchestrator
remains readable and so that unit tests can exercise individual steps.
"""

from __future__ import annotations

from typing import Optional

import mne
import numpy as np


def maybe_rescale_to_volts(raw: mne.io.BaseRaw, abs_threshold_v: float = 1.0e-2) -> mne.io.BaseRaw:
    """Heuristically rescale EEG channels stored in micro-volts to volts.

    MNE-Python expects EEG amplitudes to be expressed in volts. When a file
    is read with the wrong unit, the median absolute amplitude becomes
    unrealistically large. If that happens, channels are multiplied by
    ``1e-6`` to obtain volts.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Preloaded MNE Raw object.
    abs_threshold_v : float
        Median absolute amplitude (in volts) above which the data is
        considered to be in micro-volts.
    """
    picks = mne.pick_types(raw.info, eeg=True, eog=False, exclude=[])
    if len(picks) == 0:
        return raw
    data = raw.get_data(picks=picks)
    median_abs = float(np.nanmedian(np.abs(data)))
    if median_abs > abs_threshold_v:
        raw.apply_function(lambda x: x * 1.0e-6, picks=picks, channel_wise=False)
    return raw


def resample(raw: mne.io.BaseRaw, target_sfreq: float) -> mne.io.BaseRaw:
    """Resample the recording to ``target_sfreq`` Hz if necessary."""
    if raw.info["sfreq"] == target_sfreq:
        return raw
    raw.resample(sfreq=target_sfreq, npad="auto", verbose=False)
    return raw


def bandpass_filter(
    raw: mne.io.BaseRaw,
    l_freq: float,
    h_freq: float,
    notch_freq: Optional[float] = None,
    design: str = "firwin",
) -> mne.io.BaseRaw:
    """Apply optional notch filtering followed by a zero-phase FIR band-pass.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Preloaded raw recording (modified in place and also returned).
    l_freq, h_freq : float
        Low and high cutoffs in Hz.
    notch_freq : float, optional
        If provided, a single-frequency notch filter is applied first. The
        manuscript's reference pipeline leaves this as ``None`` because the
        upper cutoff is below the 50 Hz line frequency.
    design : str
        FIR filter design passed to MNE. ``"firwin"`` matches the manuscript.
    """
    if notch_freq is not None and notch_freq > 0:
        raw.notch_filter(freqs=[notch_freq], verbose=False)
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design=design, verbose=False)
    return raw


def average_reference(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Apply the common average reference."""
    raw.set_eeg_reference("average", projection=False, verbose=False)
    return raw


def interpolate_bads(raw: mne.io.BaseRaw, max_interpolated: int) -> mne.io.BaseRaw:
    """Spherically interpolate up to ``max_interpolated`` bad channels.

    Recordings with more than ``max_interpolated`` unusable channels are
    returned unchanged so that the caller can decide to drop them.
    """
    if not raw.info["bads"]:
        return raw
    if len(raw.info["bads"]) > max_interpolated:
        return raw
    raw.interpolate_bads(reset_bads=True, verbose=False)
    return raw
