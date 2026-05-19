"""EEG-BenchML: leakage-safe empirical benchmark for EEG-based dementia
classification.

This package provides the reference implementation of the end-to-end classical
machine learning pipeline described in the accompanying manuscript. Submodules
mirror the methodology section:

* :mod:`eeg_benchml.data` -- BIDS-style data loading and label normalisation.
* :mod:`eeg_benchml.preprocessing` -- filtering, ICA, ASR, ASR + ICA pipelines.
* :mod:`eeg_benchml.epoching` -- fixed-length epoching, rejection, and
  training-only amplitude / Gaussian-noise augmentation.
* :mod:`eeg_benchml.features` -- spectral, complexity, connectivity, and graph
  feature engineering.
* :mod:`eeg_benchml.selection` -- fold-internal feature selection.
* :mod:`eeg_benchml.models` -- classical classifiers with sigmoid calibration.
* :mod:`eeg_benchml.evaluation` -- LOSO cross-validation, subject-level
  aggregation, metrics, and uncertainty analysis.
* :mod:`eeg_benchml.interpretation` -- permutation importance with region /
  band aggregation.
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.1.0"
