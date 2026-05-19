# EEG-BenchML: Leakage-Safe Empirical Benchmark for EEG-Based Dementia Classification

Reference implementation of the end-to-end classical machine learning pipeline
described in the manuscript *"Empirical Evaluation of EEG-Based Machine Learning
Pipelines for Dementia Classification"*. The codebase covers the complete
workflow from raw resting-state EEG to subject-level prediction:

1. BIDS-compatible data loading and label normalisation.
2. Signal preprocessing: band-pass filtering, ICA-based artifact removal,
   Artifact Subspace Reconstruction (ASR), and the combined ASR + ICA pipeline.
3. Fixed-length epoching with optional training-only amplitude scaling and
   Gaussian-noise augmentation.
4. Multi-domain feature engineering: spectral, complexity, pairwise
   connectivity (weighted phase lag index), and graph-derived descriptors.
5. Leakage-safe feature selection: variance filtering, Spearman correlation
   pruning, mutual-information ranking, and `l1`-regularised logistic
   selection.
6. Classical ML classification with sigmoid calibration: linear discriminant
   analysis, logistic regression, linear / RBF SVM, k-nearest neighbours,
   random forest, ExtraTrees, and gradient boosting.
7. Subject-level probability aggregation (mean probability and majority voting)
   under leave-one-subject-out cross-validation with an inner GroupKFold loop.
8. Evaluation: accuracy, balanced accuracy, F1, sensitivity, specificity,
   Matthews correlation coefficient, ROC-AUC, subject-level bootstrap
   confidence intervals, and paired bootstrap tests.
9. Interpretation: permutation importance aggregated by feature family,
   frequency band, channel, and anatomical region.

The implementation is provided as a self-contained engineering project
accompanying the manuscript. The reference dataset is the publicly available
*OpenNeuro ds004504* resting-state EEG corpus (Alzheimer's disease,
frontotemporal dementia, and cognitively normal subjects). The repository
contains code only and does not redistribute the dataset.

## Repository Layout

```
eeg_benchml_dementia/
├── README.md
├── LICENSE
├── METHODOLOGY.md              # Section-by-section map between code and paper
├── pyproject.toml
├── requirements.txt
├── environment.yml
├── configs/                    # Hydra/OmegaConf style configuration tree
│   ├── base.yaml
│   ├── preprocessing.yaml
│   ├── features.yaml
│   ├── selection.yaml
│   └── classifiers.yaml
├── src/eeg_benchml/
│   ├── constants.py            # Channel groups, frequency bands, label sets
│   ├── data/                   # Dataset, BIDS loader, label normalisation
│   ├── preprocessing/          # Filtering, ICA, ASR, ASR+ICA pipelines
│   ├── epoching/               # Fixed-length epochs, rejection, augmentation
│   ├── features/               # Spectral / complexity / connectivity / graph
│   ├── selection/              # Fold-internal feature selection
│   ├── models/                 # Classifiers + sigmoid calibration
│   ├── evaluation/             # LOSO, aggregation, metrics, uncertainty
│   ├── interpretation/         # Permutation importance + region aggregation
│   ├── pipeline.py             # End-to-end pipeline orchestrator
│   └── utils/                  # Logging, seeding, I/O, timing helpers
├── scripts/                    # Command-line entry points
│   ├── run_experiment.py
│   ├── run_staged_benchmark.py
│   └── run_interpretation.py
└── tests/                      # Smoke / consistency tests for the main blocks
```

## Quick Start

1. **Create the environment.**

   ```bash
   conda env create -f environment.yml
   conda activate eeg-benchml
   # or, with pip:
   pip install -r requirements.txt
   ```

2. **Point the configuration at a BIDS-style EEG dataset.** Edit
   `configs/base.yaml` and set `data.root` to the dataset root directory.
   The dataset is expected to follow the BIDS convention with a
   `participants.tsv` file containing a diagnostic group column (e.g.
   ``Group`` / ``diagnosis`` / ``condition``).

3. **Run the selected end-to-end pipeline (final reference configuration).**

   ```bash
   python scripts/run_experiment.py task=ad_cn
   ```

4. **Reproduce the staged benchmark in Table 2 of the manuscript.**

   ```bash
   python scripts/run_staged_benchmark.py task=ad_cn
   ```

5. **Run the region / band permutation-importance interpretation.**

   ```bash
   python scripts/run_interpretation.py task=ad_cn
   ```

## Reference Configuration

The final pipeline selected in the manuscript corresponds to the default
configuration values shipped in `configs/`:

| Stage | Setting |
| --- | --- |
| Resampling | 250 Hz |
| Band-pass filter | 0.5 -- 45 Hz, zero-phase FIR |
| Artifact correction | ASR + ICA (ICLabel-driven removal) |
| Epoching | 10 s, 50 % overlap |
| Training-only augmentation | amplitude scaling + Gaussian noise |
| Features | spectral + complexity + pairwise connectivity (1596 raw) |
| Feature selection | mutual information, top-100 |
| Classifier | linear SVM with sigmoid calibration |
| Aggregation | mean class probability over epochs |
| Validation | leave-one-subject-out, inner 5-fold GroupKFold |

## Anonymity Notice

All files in this repository are anonymised. Author names, institutional
affiliations, contact information, and dataset-specific local paths have been
intentionally removed and replaced with neutral placeholders to support
double-blind peer review.

## License

This codebase is released under the terms specified in `LICENSE`.
