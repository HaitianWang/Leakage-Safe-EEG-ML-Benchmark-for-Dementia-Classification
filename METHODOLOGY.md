# Methodology Mapping

This document maps every methodological component described in the
manuscript to the corresponding modules of the codebase. It is provided as
supporting material for reviewers who want to verify that the implementation
follows the manuscript point by point.

## 1. Dataset and Materials (Manuscript Section 3)

| Manuscript element | Code module |
|---|---|
| BIDS-style file discovery | [`src/eeg_benchml/data/loader.py`](src/eeg_benchml/data/loader.py) |
| Diagnostic label normalisation to `{AD, FTD, CN}` | [`src/eeg_benchml/data/labels.py`](src/eeg_benchml/data/labels.py) |
| 19-channel 10--20 montage and region groups | [`src/eeg_benchml/constants.py`](src/eeg_benchml/constants.py) |
| 5 anatomical regions (frontal, temporal, central, parietal, occipital) | `eeg_benchml.constants.REGIONS` |

## 2. Data Preparation, Tasks, Augmentation, and Subject-Level Evaluation
   (Manuscript Section 4.1)

| Manuscript element | Code module |
|---|---|
| Equation (1): fixed-length epoching, 10 s, 50 % overlap | `eeg_benchml.epoching.segment.make_fixed_length_epochs` |
| Equation (2): training-only augmentation, amplitude scaling + Gaussian noise | `eeg_benchml.epoching.augmentation.augment_training_epochs` |
| Equation (3): mean-probability aggregation, majority voting ablation | `eeg_benchml.evaluation.aggregation.aggregate_subject_predictions` |
| LOSO outer protocol + inner 5-fold GroupKFold | `eeg_benchml.evaluation.loso.LOSOEvaluator` |
| Subject-level metrics (Acc / BAC / F1 / Sens. / Spec. / MCC / AUC) | `eeg_benchml.evaluation.metrics.compute_subject_level_metrics` |
| Bootstrap CIs on accuracy AND AUC (Table 3) | `eeg_benchml.evaluation.uncertainty.bootstrap_confidence_interval` |
| Paired bootstrap p-values (Table 3) | `eeg_benchml.evaluation.uncertainty.paired_bootstrap_test` |
| Runtime measurement (feature time + per-subject prediction time) | `eeg_benchml.utils.timing.Timer`, used in `LOSOEvaluator` |
| Benchmark tasks (AD vs CN, FTD vs CN, AD vs FTD, three-class) | `eeg_benchml.constants.TASKS` |

## 3. Signal Preprocessing (Manuscript Section 4.2)

| Manuscript element | Code module |
|---|---|
| Resample 500 -> 250 Hz | `eeg_benchml.preprocessing.filtering.resample` |
| 0.5--45 Hz zero-phase FIR band-pass (no notch) | `eeg_benchml.preprocessing.filtering.bandpass_filter` |
| Average referencing + up to two spherical interpolations | `eeg_benchml.preprocessing.filtering.average_reference`, `interpolate_bads` |
| ICA fitted on filtered signal, components = data rank, max_iter = 1000, fixed seed | `eeg_benchml.preprocessing.ica.fit_apply_ica` |
| ICLabel-based component removal at probability > 0.80 | same module |
| ASR (`clean_rawdata` equivalent) with cutoff 20σ, flatline 5 s, channel corr 0.80, max bad-window 0.25 | `eeg_benchml.preprocessing.asr.apply_asr` |
| Filtering-only / ICA-only / ASR-only / ASR + ICA variants | `eeg_benchml.preprocessing.pipeline.preprocess_subject` |
| Epoch rejection (peak-to-peak > 150 μV, log-variance > 3.5σ from median) | `eeg_benchml.epoching.rejection.reject_bad_epochs` |

## 4. Feature Engineering (Manuscript Section 4.3)

The exact dimensionalities below match Table 2 of the manuscript (``Dim.`` column).

### 4.1 Spectral features (266 total = 14 descriptors × 19 channels)

| Descriptor group | Count per channel |
|---|---|
| 5 absolute log band powers (delta, theta, alpha, beta, low-gamma) | 5 |
| 5 relative band powers | 5 |
| Spectral entropy | 1 |
| Alpha peak frequency | 1 |
| Theta / alpha rhythm ratio | 1 |
| Slowing index (theta + delta) / (alpha + beta) | 1 |
| **Per-channel total** | **14** |

Module: [`src/eeg_benchml/features/spectral.py`](src/eeg_benchml/features/spectral.py).
Welch PSD: 4 s Hann window, 50 % overlap, ``n_fft = 1024`` (Equation 4 of the
manuscript).

### 4.2 Complexity features (475 total = 25 descriptors × 19 channels)

| Descriptor group | Per channel |
|---|---|
| **Broadband (10):** Hjorth activity (log), Hjorth mobility, Hjorth complexity, sample entropy, Higuchi FD, MSE at scales 1--5 | 10 |
| **Band-limited (15):** Hjorth mobility, Hjorth complexity, sample entropy, computed on each of the 5 EEG bands (3 × 5) | 15 |
| **Per-channel total** | **25** |

Module: [`src/eeg_benchml/features/complexity.py`](src/eeg_benchml/features/complexity.py).
Sample entropy uses :math:`m = 2` and :math:`r = 0.2\sigma` (Equation 5);
Higuchi FD uses :math:`k_{\max} = 8`.

### 4.3 Pairwise connectivity features (855 total = 171 pairs × 5 bands)

Weighted phase lag index (Equation 6 of the manuscript) computed for all
unordered channel pairs and frequency bands. Module:
[`src/eeg_benchml/features/connectivity.py`](src/eeg_benchml/features/connectivity.py).

### 4.4 Graph features (70 total = 5 bands × (10 between-region wPLI + 4 graph descriptors))

| Descriptor group | Per band | Count |
|---|---|---|
| Between-region wPLI means (frontal-temporal, frontal-central, ..., parietal-occipital) | 10 | 50 |
| Weighted mean node strength | 1 | 5 |
| Weighted clustering coefficient | 1 | 5 |
| Global efficiency (inverse-weight graph) | 1 | 5 |
| Characteristic path length (inverse-weight graph) | 1 | 5 |
| **Per-band total** | **14** | **70** |

Module: [`src/eeg_benchml/features/graph.py`](src/eeg_benchml/features/graph.py).
The weakest 70 % of edges are pruned before global graph descriptors are
computed.

### 4.5 Combined feature sets used in Table 2

| Row | Families | Dim. |
|---|---|---|
| ``Spectral`` | spectral | 266 |
| ``Complexity`` | complexity | 475 |
| ``Connectivity`` | connectivity | 855 |
| ``Connectivity + graph`` | connectivity + graph | 925 |
| ``Spectral + complexity`` | spectral + complexity | 741 |
| ``Spectral + complexity + connectivity`` (selected) | spectral + complexity + connectivity | **1596** |

These numbers are enforced by ``assert`` statements in the feature extractor
and by `tests/test_features_shapes.py`.

## 5. Feature Selection and Dimensionality Control (Manuscript Section 4.4)

Module: [`src/eeg_benchml/selection/selector.py`](src/eeg_benchml/selection/selector.py).

| Step | Threshold / behaviour |
|---|---|
| Variance filter | ``threshold = 1e-6`` |
| Spearman correlation pruning | retain MI-larger feature when ``|ρ| > 0.95`` |
| Z-score standardisation | per-column mean and standard deviation from training fold only |
| Mutual-information selector | ``k ∈ {50, 100, 200, 400}`` chosen by inner GroupKFold |
| L1-logistic selector | ``C ∈ {0.01, 0.1, 1, 10}``, iterates from large to small ``C`` |

## 6. Classical ML Models (Manuscript Section 4.5)

Module: [`src/eeg_benchml/models/classifiers.py`](src/eeg_benchml/models/classifiers.py)
and [`src/eeg_benchml/models/calibration.py`](src/eeg_benchml/models/calibration.py).

| Classifier | Hyper-parameter grid (matches manuscript) |
|---|---|
| Shrinkage LDA | ``shrinkage = 'auto'`` |
| Logistic regression | ``C ∈ {0.01, 0.1, 1, 10}``, ``max_iter = 5000``, ``l2`` |
| Linear SVM (selected) | ``C ∈ {0.01, 0.1, 1, 10}`` |
| RBF-SVM | ``C ∈ {0.1, 1, 10, 100}``, ``γ ∈ {scale, 0.001, 0.01, 0.1}`` |
| KNN | ``k ∈ {3, 5, 7, 11}``, distance-weighted |
| Random forest | 500 trees, ``min_samples_leaf ∈ {1, 2, 4}``, ``max_features = sqrt`` |
| ExtraTrees | 500 trees, same as random forest |
| Gradient boosting | ``n_estimators ∈ {100, 300}``, ``lr ∈ {0.05, 0.1}``, ``depth ∈ {2, 3}`` |

All classifiers are wrapped in
:func:`eeg_benchml.models.calibration.wrap_with_sigmoid_calibration` so that
the same mean-probability aggregation rule applies uniformly across the
benchmark.

## 7. Staged Benchmark (Manuscript Section 5.2)

Module: [`scripts/run_staged_benchmark.py`](scripts/run_staged_benchmark.py).

| Stage | Reproduces |
|---|---|
| **A** | preprocessing variants (filtering / ICA / ASR / ASR + ICA) |
| **B** | epoch length 5 / 10 / 20 / 30 s, augmentation full / control / off |
| **B (Aug. ctrl.)** | no-aug + widened SVM grid, Gaussian-noise only, amplitude-scaling only |
| **C** | feature-family ablation (spectral / complexity / connectivity / connectivity + graph / pairs / full) |
| **D** | no-selection, MI top-50/100/200, L1-logistic, and the eight classifiers |

## 8. Fair LOSO Baselines (Manuscript Table 4)

Module: [`src/eeg_benchml/baselines/recipes.py`](src/eeg_benchml/baselines/recipes.py)
and [`scripts/run_baselines.py`](scripts/run_baselines.py).

Each baseline is declared as a :class:`BaselineRecipe` and expanded into a
plain override list. Because the same ``run_pipeline`` orchestrator is used,
all baselines share the leakage-safe LOSO protocol, the subject-level
aggregation rule, and the training-fold feature-selection procedure.

| Baseline | Feature families | Classifier | Task |
|---|---|---|---|
| Band power + coherence | spectral + connectivity | linear SVM | AD vs CN |
| Spectrum + complexity + synchronisation | spectral + complexity + connectivity | random forest | AD vs CN |
| Rhythm-ratio discriminative | spectral | linear SVM | AD vs FTD |
| Electrode-pair communication | connectivity | random forest | AD vs FTD |

## 9. Interpretation (Manuscript Section 5.5 and Fig. 4)

Module: [`src/eeg_benchml/interpretation/`](src/eeg_benchml/interpretation/).

* :func:`compute_permutation_importance` computes the per-feature accuracy
  drop using subject-level aggregation.
* :func:`aggregate_importance_by_family` / `_band` / `_channel` / `_region`
  produce the per-family, per-band, per-channel, and per-region summaries
  shown in Fig. 4.

## 10. Reference Configuration

The final reference pipeline matches the manuscript's selected configuration:

* ASR + ICA artifact correction.
* 10 s epochs with 50 % overlap and training-only amplitude scaling +
  Gaussian noise.
* Spectral + complexity + pairwise connectivity features (1596 raw).
* Mutual-information top-100 selection.
* Linear SVM with sigmoid calibration.
* Mean-probability aggregation.

These defaults are encoded in [`configs/base.yaml`](configs/base.yaml) and
the four group files (`preprocessing.yaml`, `features.yaml`,
`selection.yaml`, `classifiers.yaml`).

## 11. Static Description Tool

The script [`scripts/describe_pipeline.py`](scripts/describe_pipeline.py)
prints a structured report of the configured pipeline **without** reading
any EEG data. It is intended as a quick auditing tool for reviewers:

```bash
python scripts/describe_pipeline.py                           # reference pipeline
python scripts/describe_pipeline.py --preprocessing ica_only  # Stage A ablation
python scripts/describe_pipeline.py --features connectivity_graph  # Stage C ablation
python scripts/describe_pipeline.py --classifier rbf_svm      # Stage D ablation
```
