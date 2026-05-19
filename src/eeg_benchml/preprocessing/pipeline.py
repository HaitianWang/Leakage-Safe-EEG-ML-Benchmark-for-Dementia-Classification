"""Top-level preprocessing orchestrator.

This module exposes :func:`preprocess_subject`, which converts one subject's
raw recording into the cleaned recording used by the epoching and feature
engineering stages.

Four variants are supported through the ``artifact_correction`` field of
:class:`PreprocessConfig`:

* ``filtering_only``: band-pass and average reference only.
* ``ica_only``:       filtering + ICLabel-driven ICA.
* ``asr_only``:       filtering + ASR.
* ``asr_ica``:        filtering + ASR followed by ICA (reference pipeline).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import mne

from ..utils import get_logger
from .asr import ASRConfig, apply_asr
from .filtering import (
    average_reference,
    bandpass_filter,
    interpolate_bads,
    maybe_rescale_to_volts,
    resample,
)
from .ica import ICAConfig, fit_apply_ica

_LOGGER = get_logger(__name__)

_VALID_VARIANTS = (
    "filtering_only",
    "ica_only",
    "asr_only",
    "asr_ica",
)


@dataclass
class PreprocessConfig:
    """Aggregate configuration for the preprocessing stage."""

    resample_hz: float = 250.0
    l_freq: float = 0.5
    h_freq: float = 45.0
    notch_freq: Optional[float] = None
    filter_design: str = "firwin"
    reference: str = "average"
    artifact_correction: str = "asr_ica"

    ica: ICAConfig = field(default_factory=ICAConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)

    max_interpolated_channels: int = 2

    def __post_init__(self) -> None:
        if self.artifact_correction not in _VALID_VARIANTS:
            raise ValueError(
                f"Unknown artifact_correction='{self.artifact_correction}'. "
                f"Expected one of {_VALID_VARIANTS}."
            )


def preprocess_subject(raw: mne.io.BaseRaw, config: PreprocessConfig) -> mne.io.BaseRaw:
    """Run the full preprocessing chain on a raw recording.

    The order of operations follows Section 4.2 of the manuscript:

    1. Pick EEG channels and rescale to volts if needed.
    2. Resample to ``config.resample_hz`` Hz.
    3. Apply a zero-phase FIR band-pass filter (with optional notch).
    4. Run the configured artefact-correction variant.
    5. Apply the common average reference.
    6. Interpolate up to ``max_interpolated_channels`` bad channels.

    The function operates on a copy of the input. The caller therefore retains
    the original recording untouched, which is important during ablation runs.
    """
    work = raw.copy()
    work.pick_types(eeg=True, eog=True, exclude="bads")
    work = maybe_rescale_to_volts(work)
    work = resample(work, target_sfreq=config.resample_hz)
    work = bandpass_filter(
        raw=work,
        l_freq=config.l_freq,
        h_freq=config.h_freq,
        notch_freq=config.notch_freq,
        design=config.filter_design,
    )

    variant = config.artifact_correction
    if variant == "asr_only":
        work = apply_asr(work, config.asr)
    elif variant == "ica_only":
        work = fit_apply_ica(work, config.ica)
    elif variant == "asr_ica":
        work = apply_asr(work, config.asr)
        work = fit_apply_ica(work, config.ica)
    elif variant == "filtering_only":
        pass  # filtering only is exactly this chain without any source-based step
    else:  # pragma: no cover - guarded by __post_init__
        raise ValueError(f"Unsupported artifact_correction='{variant}'.")

    if config.reference == "average":
        work = average_reference(work)
    work = interpolate_bads(work, max_interpolated=config.max_interpolated_channels)
    _LOGGER.info(
        "Preprocessed recording: variant=%s, sfreq=%.1f, n_channels=%d.",
        variant, work.info["sfreq"], len(mne.pick_types(work.info, eeg=True)),
    )
    return work
