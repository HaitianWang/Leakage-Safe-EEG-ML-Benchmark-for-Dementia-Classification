"""Signal preprocessing variants used in Stage A of the benchmark.

The reference pipeline corresponds to :func:`asr_then_ica` and follows the
manuscript's Section 4.2 specification:

* Resample to 250 Hz.
* Zero-phase FIR band-pass filter 0.5--45 Hz (no notch required because the
  upper cutoff already excludes the 50 Hz line component).
* ASR (clean_rawdata) with burst cutoff 20 sigma, flatline threshold 5 s,
  channel correlation threshold 0.80, and maximum bad-window proportion 0.25.
* ICA with the number of components equal to the data rank, up to 1000
  iterations, fixed random seed, and ICLabel-based component removal at
  probability greater than 0.80.
* Average reference and spherical interpolation of up to two bad channels.

The remaining variants (``filtering_only``, ``ica_only``, ``asr_only``)
expose the same interface to support Stage A ablation.
"""

from .filtering import bandpass_filter, average_reference
from .ica import ICAConfig, fit_apply_ica
from .asr import ASRConfig, apply_asr
from .pipeline import (
    PreprocessConfig,
    preprocess_subject,
)

__all__ = [
    "bandpass_filter",
    "average_reference",
    "ICAConfig",
    "fit_apply_ica",
    "ASRConfig",
    "apply_asr",
    "PreprocessConfig",
    "preprocess_subject",
]
