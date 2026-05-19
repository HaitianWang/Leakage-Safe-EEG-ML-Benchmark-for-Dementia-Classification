"""End-to-end pipeline orchestrator.

This module ties every component together. Given an OmegaConf configuration
node, :func:`run_pipeline` performs the full benchmark loop and writes the
resulting metrics, selected features, and uncertainty estimates under
``experiment.output_dir``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from omegaconf import DictConfig, OmegaConf

from .constants import TASKS
from .data import (
    SubjectRecord,
    discover_subjects,
    load_participant_labels,
    read_raw_eeg,
)
from .epoching import (
    AugmentationConfig,
    EpochingConfig,
    RejectionConfig,
    make_fixed_length_epochs,
    reject_bad_epochs,
)
from .evaluation import (
    LOSOResult,
    bootstrap_confidence_interval,
    compute_subject_level_metrics,
)
from .evaluation.loso import LOSOConfig, LOSOEvaluator
from .evaluation.uncertainty import (
    metric_accuracy,
    metric_auc_binary,
    metric_auc_ovr,
)
from .features import FeatureExtractorConfig
from .features.complexity import ComplexityConfig
from .features.connectivity import ConnectivityConfig
from .features.graph import GraphConfig
from .features.spectral import SpectralConfig
from .models.classifiers import ClassifierConfig
from .preprocessing import PreprocessConfig, preprocess_subject
from .preprocessing.asr import ASRConfig
from .preprocessing.ica import ICAConfig
from .selection import FeatureSelectionConfig
from .utils import get_logger, set_global_seed
from .utils.io import dump_json, ensure_dir

_LOGGER = get_logger(__name__)


@dataclass
class PipelineArtefacts:
    """Lightweight container for the values returned by :func:`run_pipeline`."""

    config: Dict[str, Any]
    classes: List[str]
    metrics: Dict[str, float]
    uncertainty: Dict[str, Tuple[float, float]]
    feature_time_s: float
    predict_time_s_per_subject: float
    selected_feature_names_per_fold: List[List[str]]


# ---------------------------------------------------------------------------
# Configuration adapters.
# ---------------------------------------------------------------------------
def _build_preprocess_config(cfg: DictConfig) -> PreprocessConfig:
    ica_cfg = cfg.preprocessing.get("ica", {}) or {}
    asr_cfg = cfg.preprocessing.get("asr", {}) or {}
    return PreprocessConfig(
        resample_hz=float(cfg.preprocessing.get("resample_hz", 250.0)),
        l_freq=float(cfg.preprocessing.get("l_freq", 0.5)),
        h_freq=float(cfg.preprocessing.get("h_freq", 45.0)),
        notch_freq=cfg.preprocessing.get("notch_freq"),
        filter_design=str(cfg.preprocessing.get("filter_design", "firwin")),
        reference=str(cfg.preprocessing.get("reference", "average")),
        artifact_correction=str(cfg.preprocessing.get("artifact_correction", "asr_ica")),
        ica=ICAConfig(
            n_components=ica_cfg.get("n_components"),
            max_iter=int(ica_cfg.get("max_iter", 1000)),
            random_state=int(ica_cfg.get("random_state", 42)),
            iclabel_threshold=float(ica_cfg.get("iclabel_threshold", 0.80)),
            artifact_labels=tuple(
                ica_cfg.get("artifact_labels",
                            ("eye", "muscle", "heart", "line_noise", "channel_noise"))
            ),
        ),
        asr=ASRConfig(
            burst_cutoff=float(asr_cfg.get("burst_cutoff", 20.0)),
            flatline_threshold_s=float(asr_cfg.get("flatline_threshold_s", 5.0)),
            channel_correlation_threshold=float(
                asr_cfg.get("channel_correlation_threshold", 0.80)
            ),
            max_bad_window_proportion=float(
                asr_cfg.get("max_bad_window_proportion", 0.25)
            ),
        ),
    )


def _build_feature_config(cfg: DictConfig) -> FeatureExtractorConfig:
    families = OmegaConf.to_container(cfg.features.families, resolve=True)
    spectral_cfg = cfg.features.get("spectral", {}) or {}
    complexity_cfg = cfg.features.get("complexity", {}) or {}
    connectivity_cfg = cfg.features.get("connectivity", {}) or {}
    graph_cfg = cfg.features.get("graph", {}) or {}
    return FeatureExtractorConfig(
        families=dict(families),
        spectral=SpectralConfig(
            welch_window_s=float(spectral_cfg.get("welch_window_s", 4.0)),
            welch_overlap=float(spectral_cfg.get("welch_overlap", 0.5)),
            n_fft=int(spectral_cfg.get("n_fft", 1024)),
        ),
        complexity=ComplexityConfig(
            sample_entropy_m=int(complexity_cfg.get("sample_entropy_m", 2)),
            sample_entropy_r=float(complexity_cfg.get("sample_entropy_r", 0.2)),
            higuchi_kmax=int(complexity_cfg.get("higuchi_kmax", 8)),
            multi_scale_entropy_scales=tuple(
                int(s)
                for s in complexity_cfg.get(
                    "multi_scale_entropy_scales", (1, 2, 3, 4, 5)
                )
            ),
        ),
        connectivity=ConnectivityConfig(
            metric=str(connectivity_cfg.get("metric", "wpli")),
            eps=float(connectivity_cfg.get("eps", 1.0e-8)),
        ),
        graph=GraphConfig(
            edge_keep_proportion=float(graph_cfg.get("edge_keep_proportion", 0.30)),
            descriptors=tuple(
                graph_cfg.get(
                    "descriptors",
                    ("mean_strength", "clustering", "global_efficiency", "char_path_length"),
                )
            ),
        ),
    )


def _build_selection_config(cfg: DictConfig) -> FeatureSelectionConfig:
    selection_cfg = cfg.selection
    mi_cfg = selection_cfg.get("mutual_information", {}) or {}
    l1_cfg = selection_cfg.get("l1_logistic", {}) or {}
    return FeatureSelectionConfig(
        variance_threshold=float(selection_cfg.get("variance_threshold", 1e-6)),
        correlation_threshold=float(selection_cfg.get("correlation_threshold", 0.95)),
        selector=str(selection_cfg.get("selector", "mutual_information")),
        candidate_k=tuple(int(k) for k in mi_cfg.get("candidate_k", (50, 100, 200, 400))),
        candidate_C=tuple(float(c) for c in l1_cfg.get("candidate_C", (0.01, 0.1, 1.0, 10.0))),
    )


def _build_classifier_config(cfg: DictConfig) -> ClassifierConfig:
    return ClassifierConfig(
        name=str(cfg.classifier.name),
        params=dict(OmegaConf.to_container(cfg.classifier.params, resolve=True))
        if cfg.classifier.params is not None
        else {},
    )


def _build_augmentation_config(cfg: DictConfig) -> AugmentationConfig:
    aug_cfg = cfg.features.get("augmentation", {}) or {}
    return AugmentationConfig(
        enabled=bool(aug_cfg.get("enabled", True)),
        amplitude_scaling_range=tuple(
            aug_cfg.get("amplitude_scaling", (0.95, 1.05))
        ),
        gaussian_noise_sigma=float(aug_cfg.get("gaussian_noise_sigma", 0.01)),
        random_state=int(cfg.experiment.get("seed", 42)),
    )


def _build_epoching_config(cfg: DictConfig) -> EpochingConfig:
    epoch_cfg = cfg.features.get("epoch", {}) or {}
    return EpochingConfig(
        duration_s=float(epoch_cfg.get("duration_s", 10.0)),
        overlap=float(epoch_cfg.get("overlap", 0.5)),
    )


def _build_rejection_config(cfg: DictConfig) -> RejectionConfig:
    epoch_cfg = cfg.features.get("epoch", {}) or {}
    return RejectionConfig(
        peak_to_peak_uv=float(epoch_cfg.get("reject_peak_to_peak_uv", 150.0)),
        log_variance_z_threshold=float(epoch_cfg.get("log_variance_z_threshold", 3.5)),
    )


def _prepare_subject_epochs(
    record: SubjectRecord,
    preprocess_cfg: PreprocessConfig,
    epoch_cfg: EpochingConfig,
    rejection_cfg: RejectionConfig,
) -> Tuple[np.ndarray, float]:
    """Run preprocessing + epoching + rejection for a single subject."""
    raw = read_raw_eeg(record.eeg_path)
    cleaned = preprocess_subject(raw=raw, config=preprocess_cfg)
    epochs = make_fixed_length_epochs(raw=cleaned, config=epoch_cfg)
    if epochs is None:
        return np.empty((0, 0, 0)), float(cleaned.info["sfreq"])
    epochs = reject_bad_epochs(epochs=epochs, config=rejection_cfg)
    if len(epochs) == 0:
        return np.empty((0, 0, 0)), float(cleaned.info["sfreq"])
    return epochs.get_data(picks="eeg"), float(epochs.info["sfreq"])


def run_pipeline(cfg: DictConfig) -> PipelineArtefacts:
    """Execute the configured pipeline and return :class:`PipelineArtefacts`."""
    set_global_seed(int(cfg.experiment.seed))
    task = str(cfg.experiment.task)
    if task not in TASKS:
        raise ValueError(f"Unknown task '{task}'. Expected one of {list(TASKS)}.")
    classes = list(TASKS[task])

    dataset_root = Path(str(cfg.data.root)).expanduser().resolve()
    label_columns = list(cfg.data.label_columns)
    labels = load_participant_labels(
        dataset_root=dataset_root,
        participants_file=str(cfg.data.participants_file),
        label_columns=label_columns,
    )
    subjects = discover_subjects(
        dataset_root=dataset_root,
        label_map=labels,
        extensions=tuple(cfg.data.eeg_extensions),
        exclude_derivatives=bool(cfg.data.exclude_bids_derivatives),
        task_labels=classes,
    )
    if not subjects:
        raise RuntimeError(
            "No subjects matched the task labels. Check the dataset path "
            "and participants table."
        )

    preprocess_cfg = _build_preprocess_config(cfg)
    epoch_cfg = _build_epoching_config(cfg)
    rejection_cfg = _build_rejection_config(cfg)
    feature_cfg = _build_feature_config(cfg)
    selection_cfg = _build_selection_config(cfg)
    classifier_cfg = _build_classifier_config(cfg)
    augmentation_cfg = _build_augmentation_config(cfg)

    subject_epoch_arrays: Dict[str, Tuple[np.ndarray, str]] = {}
    sfreq: float = float(preprocess_cfg.resample_hz)
    for record in subjects:
        _LOGGER.info("Preparing subject %s (label=%s)", record.subject_id, record.label)
        data, sub_sfreq = _prepare_subject_epochs(
            record=record,
            preprocess_cfg=preprocess_cfg,
            epoch_cfg=epoch_cfg,
            rejection_cfg=rejection_cfg,
        )
        if data.size == 0:
            _LOGGER.warning("Skipping %s: no valid epochs.", record.subject_id)
            continue
        subject_epoch_arrays[record.subject_id] = (data, record.label)
        sfreq = sub_sfreq

    if len(subject_epoch_arrays) == 0:
        raise RuntimeError("All subjects were dropped during preprocessing.")

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
    result: LOSOResult = evaluator.run(
        subject_epoch_arrays=subject_epoch_arrays, sfreq=sfreq
    )

    # Realign the subject-level probability matrix with ``y_true_subject``
    # so the AUC bootstrap operates on matched rows.
    ordered_subjects = sorted(result.subject_probabilities)
    probability_matrix = np.array(
        [result.subject_probabilities[sid] for sid in ordered_subjects]
    )
    subject_to_truth = {
        sid: lbl for sid, lbl in zip(
            sorted(subject_epoch_arrays.keys()), result.y_true_subject
        )
    }
    aligned_truth = np.array([subject_to_truth[sid] for sid in ordered_subjects])
    aligned_pred = np.array(
        [
            result.y_pred_subject[
                sorted(subject_epoch_arrays.keys()).index(sid)
            ]
            for sid in ordered_subjects
        ]
    )

    metrics = compute_subject_level_metrics(
        y_true=aligned_truth.tolist(),
        y_pred=aligned_pred.tolist(),
        mean_probabilities=probability_matrix,
        classes=classes,
    )

    # Bootstrap CIs on subject-level accuracy AND AUC, matching Table 3.
    n_resamples = int(cfg.evaluation.uncertainty.bootstrap_resamples)
    confidence_level = float(cfg.evaluation.uncertainty.confidence_level)
    seed = int(cfg.experiment.seed)

    acc_lo, acc_hi = bootstrap_confidence_interval(
        metric_fn=metric_accuracy,
        y_true=aligned_truth,
        y_pred=aligned_pred,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        random_state=seed,
    )
    if len(classes) == 2:
        positive_label = classes[0]
        positive_idx = classes.index(positive_label)
        auc_lo, auc_hi = bootstrap_confidence_interval(
            metric_fn=metric_auc_binary(positive_label),
            y_true=aligned_truth,
            y_pred=probability_matrix[:, positive_idx],
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            random_state=seed,
        )
    else:
        auc_lo, auc_hi = bootstrap_confidence_interval(
            metric_fn=metric_auc_ovr(classes),
            y_true=aligned_truth,
            y_pred=probability_matrix,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            random_state=seed,
        )
    uncertainty: Dict[str, Tuple[float, float]] = {
        "accuracy_ci_95": (acc_lo, acc_hi),
        "auc_ci_95": (auc_lo, auc_hi),
    }

    artefacts = PipelineArtefacts(
        config=OmegaConf.to_container(cfg, resolve=True),
        classes=classes,
        metrics=metrics,
        uncertainty=uncertainty,
        feature_time_s=result.feature_time_s,
        predict_time_s_per_subject=result.predict_time_s_per_subject,
        selected_feature_names_per_fold=result.selected_feature_names_per_fold,
    )

    out_dir = ensure_dir(Path(str(cfg.experiment.output_dir)) / str(cfg.experiment.name))
    dump_json(
        payload={
            "config": artefacts.config,
            "classes": artefacts.classes,
            "metrics": artefacts.metrics,
            "uncertainty": {
                key: {"lower": low, "upper": high}
                for key, (low, high) in artefacts.uncertainty.items()
            },
            "feature_time_s": artefacts.feature_time_s,
            "predict_time_s_per_subject": artefacts.predict_time_s_per_subject,
        },
        path=out_dir / "metrics.json",
    )
    return artefacts
