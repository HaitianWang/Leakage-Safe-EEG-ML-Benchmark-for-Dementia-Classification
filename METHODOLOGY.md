# Methodology Mapping

This document maps every methodological component described in the manuscript
to the corresponding modules of the codebase. It is provided as supporting
material for reviewers who want to verify that the implementation follows the
manuscript.

## 1. Dataset and Materials (Manuscript Section 3)

* `src/eeg_benchml/data/loader.py` --- BIDS-style discovery of EEG files
  (`discover_subjects`) and reader dispatch (`read_raw_eeg`).
* `src/eeg_benchml/data/labels.py` --- robust normalisation of the
  participants table to the canonical `{AD, FTD, CN}` labels.
* `src/eeg_benchml/constants.py` --- the 19 channels of the 10--20 montage,
  the 5 anatomical regions (frontal / temporal / central / parietal /
  occipital), and the 5 frequency bands.

## 2. Data Preparation, Tasks, Augmentation, and Subject-Level Evaluation
   (Manuscript Section 4.1)

| Manuscript element | Code module |
|---|---|
| Equation (1): fixed-length epoching | `eeg_benchml.epoching.segment.make_fixed_length_epochs` |
| Equation (2): training-only augmentation | `eeg_benchml.epoching.augmentation.augment_training_epochs` |
| Equation (3): mean-probability aggregation | `eeg_benchml.evaluation.aggregation.aggregate_subject_predictions` |
| LOSO outer protocol + inner GroupKFold | `eeg_benchml.evaluation.loso.LOSOEvaluator` |
| Subject-level metrics | `eeg_benchml.evaluation.metrics.compute_subject_level_metrics` |
| Bootstrap CIs + paired tests | `eeg_benchml.evaluation.uncertainty` |
| Runtime measurement | `eeg_benchml.utils.timing.Timer` |
| Benchmark tasks (AD vs CN, FTD vs CN, AD vs FTD, three-class) | `eeg_benchml.constants.TASKS` |

## 3. Signal Preprocessing (Manuscript Section 4.2)

| Manuscript element | Code module |
|---|---|
| Resample 500 -> 250 Hz | `eeg_benchml.preprocessing.filtering.resample` |
| 0.5--45 Hz FIR band-pass (no notch) | `eeg_benchml.preprocessing.filtering.bandpass_filter` |
| Average referencing + bad-channel interpolation | `eeg_benchml.preprocessing.filtering.average_reference`, `.interpolate_bads` |
| ICA decomposition + ICLabel-based removal | `eeg_benchml.preprocessing.ica.fit_apply_ica` |
| ASR (`clean_rawdata` equivalent) | `eeg_benchml.preprocessing.asr.apply_asr` |
| Filtering only / ICA only / ASR only / ASR + ICA variants | `eeg_benchml.preprocessing.pipeline.preprocess_subject` |
| Amplitude / log-variance rejection | `eeg_benchml.epoching.rejection.reject_bad_epochs` |

## 4. Feature Engineering (Manuscript Section 4.3)

| Manuscript family | Code module | Notes |
|---|---|---|
| Spectral (266 features) | `eeg_benchml.features.spectral.compute_spectral_features` | Welch PSD, 14 descriptors/channel including alpha peak and slowing ratios. |
| Complexity (475 features) | `eeg_benchml.features.complexity.compute_complexity_features` | Hjorth parameters, sample entropy (Equation 4), Higuchi FD, MSE (scales 1--5), and ZCR. |
| Connectivity (855 features) | `eeg_benchml.features.connectivity.compute_connectivity_features` | wPLI (Equation 5) over 171 pairs and 5 bands. |
| Graph (optional 20 features) | `eeg_benchml.features.graph.compute_graph_features` | Region-level network with mean strength, clustering, global efficiency, and characteristic path length. |
| Family composition | `eeg_benchml.features.extractor.FeatureExtractor` | Stable column order, family slice indices. |

## 5. Feature Selection and Dimensionality Control (Manuscript Section 4.4)

* `eeg_benchml.selection.selector.LeakageSafeSelector`:
  1. Variance filter (`variance_threshold = 1e-6`).
  2. Spearman correlation pruning (`correlation_threshold = 0.95`).
  3. Z-score standardisation with training-fold mean and standard deviation.
  4. Mutual-information top-`k` selection (default) or
     `l1`-regularised logistic regression.

## 6. Classical ML Models (Manuscript Section 4.5)

* `eeg_benchml.models.classifiers.build_classifier` --- factory for shrinkage
  LDA, logistic regression, linear SVM, RBF-SVM, KNN, random forest,
  ExtraTrees, and gradient boosting, including the exact hyper-parameter
  grids reported in the manuscript.
* `eeg_benchml.models.calibration.wrap_with_sigmoid_calibration` --- sigmoid
  (Platt) calibration applied to every classifier so that the mean-probability
  aggregation can be reused across models.

## 7. Experimental Evaluation (Manuscript Section 5)

* `scripts/run_experiment.py` --- single-configuration runner.
* `scripts/run_staged_benchmark.py` --- reproduces Stages A--D of Table 2.
* `scripts/run_interpretation.py` --- replicates the region / band
  permutation-importance interpretation (Fig. 4).

## 8. Reference Configuration

The final reference pipeline matches the manuscript:

* ASR + ICA artifact correction,
* 10 s epochs with 50 % overlap and training-only amplitude scaling +
  Gaussian noise,
* spectral + complexity + pairwise connectivity features (1596 raw),
* mutual-information top-100 selection,
* linear SVM with sigmoid calibration,
* mean-probability aggregation.
