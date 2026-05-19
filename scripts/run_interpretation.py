"""Interpretation runner.

Replicates Fig. 4 of the manuscript by computing permutation importance for
the selected pipeline and aggregating the scores by feature family, band,
channel, and region.

Usage
-----
.. code:: bash

   python scripts/run_interpretation.py experiment.task=ad_cn \
       data.root=/path/to/dataset
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_benchml.constants import TASKS  # noqa: E402
from eeg_benchml.data import (  # noqa: E402
    discover_subjects,
    load_participant_labels,
    read_raw_eeg,
)
from eeg_benchml.epoching import (  # noqa: E402
    EpochingConfig,
    RejectionConfig,
    make_fixed_length_epochs,
    reject_bad_epochs,
)
from eeg_benchml.evaluation import compute_subject_level_metrics  # noqa: E402
from eeg_benchml.evaluation.loso import LOSOConfig, LOSOEvaluator  # noqa: E402
from eeg_benchml.features import FeatureExtractor  # noqa: E402
from eeg_benchml.interpretation import (  # noqa: E402
    aggregate_importance_by_band,
    aggregate_importance_by_channel,
    aggregate_importance_by_family,
    aggregate_importance_by_region,
    compute_permutation_importance,
)
from eeg_benchml.models.calibration import wrap_with_sigmoid_calibration  # noqa: E402
from eeg_benchml.models.classifiers import build_classifier  # noqa: E402
from eeg_benchml.preprocessing import preprocess_subject  # noqa: E402
from eeg_benchml.utils import get_logger, set_global_seed  # noqa: E402
from eeg_benchml.utils.io import dump_json, ensure_dir  # noqa: E402
from eeg_benchml.pipeline import (  # noqa: E402
    _build_augmentation_config,
    _build_classifier_config,
    _build_epoching_config,
    _build_feature_config,
    _build_preprocess_config,
    _build_rejection_config,
    _build_selection_config,
)

from _config import load_config, parse_kv_overrides  # noqa: E402

_LOGGER = get_logger("scripts.run_interpretation")


def _prepare_epochs(record, preprocess_cfg, epoch_cfg, rejection_cfg):
    raw = read_raw_eeg(record.eeg_path)
    cleaned = preprocess_subject(raw=raw, config=preprocess_cfg)
    epochs = make_fixed_length_epochs(raw=cleaned, config=epoch_cfg)
    if epochs is None:
        return np.empty((0, 0, 0)), float(cleaned.info["sfreq"])
    epochs = reject_bad_epochs(epochs=epochs, config=rejection_cfg)
    if len(epochs) == 0:
        return np.empty((0, 0, 0)), float(epochs.info["sfreq"])
    return epochs.get_data(picks="eeg"), float(epochs.info["sfreq"])


def main() -> None:
    overrides = parse_kv_overrides(sys.argv[1:])
    cfg = load_config(overrides=overrides)
    set_global_seed(int(cfg.experiment.seed))

    task = str(cfg.experiment.task)
    classes = list(TASKS[task])
    dataset_root = Path(str(cfg.data.root)).resolve()
    labels = load_participant_labels(
        dataset_root=dataset_root,
        participants_file=str(cfg.data.participants_file),
        label_columns=list(cfg.data.label_columns),
    )
    subjects = discover_subjects(
        dataset_root=dataset_root,
        label_map=labels,
        extensions=tuple(cfg.data.eeg_extensions),
        exclude_derivatives=bool(cfg.data.exclude_bids_derivatives),
        task_labels=classes,
    )

    preprocess_cfg = _build_preprocess_config(cfg)
    epoch_cfg = _build_epoching_config(cfg)
    rejection_cfg = _build_rejection_config(cfg)
    feature_cfg = _build_feature_config(cfg)
    selection_cfg = _build_selection_config(cfg)
    classifier_cfg = _build_classifier_config(cfg)
    augmentation_cfg = _build_augmentation_config(cfg)

    subject_epoch_arrays: Dict[str, Tuple[np.ndarray, str]] = {}
    sfreq = float(preprocess_cfg.resample_hz)
    for record in subjects:
        data, sub_sfreq = _prepare_epochs(
            record=record,
            preprocess_cfg=preprocess_cfg,
            epoch_cfg=epoch_cfg,
            rejection_cfg=rejection_cfg,
        )
        if data.size == 0:
            continue
        subject_epoch_arrays[record.subject_id] = (data, record.label)
        sfreq = sub_sfreq

    # Run the standard LOSO loop first to obtain a fitted classifier on the
    # final fold; the interpretation step then attaches a global classifier to
    # the pooled feature matrix to obtain permutation-importance estimates.
    evaluator = LOSOEvaluator(
        feature_extractor_config=feature_cfg,
        selection_config=selection_cfg,
        classifier_config=classifier_cfg,
        augmentation_config=augmentation_cfg,
        evaluation_config=LOSOConfig(
            inner_folds=int(cfg.evaluation.inner_cv_folds),
            aggregation_rule=str(cfg.evaluation.aggregation),
            random_state=int(cfg.experiment.seed),
        ),
        classes=classes,
    )
    result = evaluator.run(
        subject_epoch_arrays=subject_epoch_arrays, sfreq=sfreq
    )
    metrics = compute_subject_level_metrics(
        y_true=result.y_true_subject,
        y_pred=result.y_pred_subject,
        mean_probabilities=np.array([
            result.subject_probabilities[sid]
            for sid in sorted(result.subject_probabilities)
        ]),
        classes=classes,
    )
    _LOGGER.info("Subject-level metrics: %s", metrics)

    # Pool features across subjects so permutation importance can be computed
    # on a single matrix. We re-extract features with the same configuration
    # for clarity.
    extractor = FeatureExtractor(feature_cfg)
    X_blocks: List[np.ndarray] = []
    groups: List[str] = []
    y_subject: List[str] = []
    feature_names: List[str] = []
    for subject_id in sorted(subject_epoch_arrays):
        data, label = subject_epoch_arrays[subject_id]
        bundle = extractor.transform_array(data=data, sfreq=sfreq)
        X_blocks.append(bundle.features)
        feature_names = bundle.feature_names
        groups.extend([subject_id] * bundle.features.shape[0])
        y_subject.append(label)
    if not X_blocks:
        raise RuntimeError("No features could be extracted for the interpretation step.")
    X_full = np.concatenate(X_blocks, axis=0)

    estimator = build_classifier(classifier_cfg)
    estimator.fit(X_full, np.array([labels.get(sub, "") for sub in groups]))
    calibrated = wrap_with_sigmoid_calibration(
        classifier=estimator,
        inner_folds=3,
        random_state=int(cfg.experiment.seed),
    )
    calibrated.fit(X_full, np.array([labels.get(sub, "") for sub in groups]))

    importance_mean, _ = compute_permutation_importance(
        estimator=calibrated,
        X=X_full,
        y_subject=y_subject,
        groups=groups,
        classes=classes,
        feature_names=feature_names,
        n_repeats=5,
        rule=str(cfg.evaluation.aggregation),
        random_state=int(cfg.experiment.seed),
    )

    interpretation = {
        "by_family": aggregate_importance_by_family(feature_names, importance_mean),
        "by_band": aggregate_importance_by_band(feature_names, importance_mean),
        "by_channel": aggregate_importance_by_channel(feature_names, importance_mean),
        "by_region": aggregate_importance_by_region(feature_names, importance_mean),
    }

    out_dir = ensure_dir(Path(str(cfg.experiment.output_dir)) / "interpretation")
    dump_json(
        payload={"metrics": metrics, "interpretation": interpretation},
        path=out_dir / "interpretation.json",
    )
    _LOGGER.info("Interpretation results written to %s", out_dir)


if __name__ == "__main__":
    main()
