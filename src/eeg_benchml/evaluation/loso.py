"""Leave-one-subject-out evaluation with an inner GroupKFold loop.

This module is the orchestration backbone of the benchmark. For each outer
fold it:

1. Splits the dataset into a single held-out subject and the remaining
   training subjects.
2. Builds the training-fold feature matrix using the previously extracted
   epochs (training-only augmentation is applied here so that augmented
   epochs never leak into the validation or test fold).
3. Fits the leakage-safe feature selector on the training fold.
4. Performs hyper-parameter tuning of the classifier on the training fold
   using an inner :class:`~sklearn.model_selection.GroupKFold` split.
5. Calibrates the best classifier with sigmoid scaling and predicts the
   held-out subject's class probabilities.
6. Aggregates the epoch-level probabilities into a single subject-level
   prediction.

The output of :meth:`LOSOEvaluator.run` is a :class:`LOSOResult` ready to be
forwarded to the metric and uncertainty helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut

from ..epoching.augmentation import AugmentationConfig, augment_training_epochs
from ..features import FeatureExtractor, FeatureExtractorConfig
from ..models.calibration import wrap_with_sigmoid_calibration
from ..models.classifiers import ClassifierConfig, build_classifier
from ..selection import FeatureSelectionConfig, LeakageSafeSelector
from ..utils import Timer, get_logger
from .aggregation import aggregate_subject_predictions

_LOGGER = get_logger(__name__)


@dataclass
class LOSOConfig:
    """Top-level evaluation configuration."""

    inner_folds: int = 5
    aggregation_rule: str = "mean_probability"
    random_state: int = 42


@dataclass
class LOSOResult:
    """Aggregated results from a LOSO benchmark run."""

    classes: List[str]
    y_true_subject: List[str]
    y_pred_subject: List[str]
    subject_probabilities: Dict[str, np.ndarray]
    selected_feature_indices_per_fold: List[np.ndarray] = field(default_factory=list)
    selected_feature_names_per_fold: List[List[str]] = field(default_factory=list)
    feature_time_s: float = 0.0
    predict_time_s_per_subject: float = 0.0
    held_out_predictions: Dict[str, str] = field(default_factory=dict)


class LOSOEvaluator:
    """Run the LOSO benchmark over a list of per-subject epoch arrays."""

    def __init__(
        self,
        feature_extractor_config: FeatureExtractorConfig,
        selection_config: FeatureSelectionConfig,
        classifier_config: ClassifierConfig,
        augmentation_config: AugmentationConfig,
        evaluation_config: LOSOConfig,
        classes: Sequence[str],
    ) -> None:
        self.feature_extractor = FeatureExtractor(feature_extractor_config)
        self.selection_config = selection_config
        self.classifier_config = classifier_config
        self.augmentation_config = augmentation_config
        self.evaluation_config = evaluation_config
        self.classes = list(classes)

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------
    def run(
        self,
        subject_epoch_arrays: Dict[str, Tuple[np.ndarray, str]],
        sfreq: float,
    ) -> LOSOResult:
        """Execute the full LOSO benchmark.

        Parameters
        ----------
        subject_epoch_arrays : dict[str, (ndarray, str)]
            Per-subject ``(epochs_array, label)`` tuples. ``epochs_array`` has
            shape ``(n_epochs, n_channels, n_times)``.
        sfreq : float
            Sampling frequency in Hz (assumed identical across subjects after
            preprocessing).
        """
        subject_ids = sorted(subject_epoch_arrays.keys())
        subject_labels = {
            sub: subject_epoch_arrays[sub][1] for sub in subject_ids
        }

        # Stage 1: extract features for every subject. We cache subject-level
        # feature bundles so that the augmentation step can recompute features
        # for augmented training epochs without re-running the (slow)
        # connectivity transform on test subjects.
        precomputed: Dict[str, np.ndarray] = {}
        feature_names: Optional[List[str]] = None
        feature_time_total = 0.0
        for sub in subject_ids:
            data = subject_epoch_arrays[sub][0]
            with Timer("feature_extraction") as t:
                bundle = self.feature_extractor.transform_array(
                    data=data, sfreq=sfreq
                )
            precomputed[sub] = bundle.features
            feature_names = bundle.feature_names
            feature_time_total += t.elapsed_s
        if feature_names is None:
            raise RuntimeError("No subject features could be extracted.")

        loso = LeaveOneGroupOut()
        groups = np.array(subject_ids)
        # Build a "subject groups" array of length equal to the number of
        # subjects. The LOSO splitter only needs subject-level groups, so we
        # operate at the subject level and let the per-subject expansion to
        # epochs happen inside the loop.
        dummy_X = np.zeros((len(subject_ids), 1))
        y_subject = np.array([subject_labels[s] for s in subject_ids])

        y_true_subject: List[str] = []
        y_pred_subject: List[str] = []
        subject_proba: Dict[str, np.ndarray] = {}
        selected_indices_per_fold: List[np.ndarray] = []
        selected_names_per_fold: List[List[str]] = []
        prediction_times: List[float] = []

        for train_idx, test_idx in loso.split(dummy_X, y_subject, groups=groups):
            test_subject = groups[test_idx][0]
            train_subjects = groups[train_idx]

            # ---- Build the training-fold matrices ----
            train_data_blocks: List[np.ndarray] = []
            train_labels: List[str] = []
            train_groups: List[str] = []
            for sub in train_subjects:
                arr = subject_epoch_arrays[sub][0]
                train_data_blocks.append(arr)
                lbl = subject_labels[sub]
                train_labels.extend([lbl] * arr.shape[0])
                train_groups.extend([sub] * arr.shape[0])
            train_data = np.concatenate(train_data_blocks, axis=0)
            train_labels_arr = np.array(train_labels)
            train_groups_arr = np.array(train_groups)

            # Training-only augmentation works on raw epoch arrays.
            aug = augment_training_epochs(
                data=train_data,
                labels=train_labels_arr,
                groups=train_groups_arr,
                config=self.augmentation_config,
            )
            with Timer("feature_extraction_train"):
                train_bundle = self.feature_extractor.transform_array(
                    data=aug.data, sfreq=sfreq
                )
            X_train = train_bundle.features
            y_train = aug.labels
            groups_train = aug.groups

            # ---- Test-fold features come from the precomputed cache ----
            X_test = precomputed[test_subject]
            test_groups = np.array([test_subject] * X_test.shape[0])

            # ---- Leakage-safe selection ----
            selector = LeakageSafeSelector(self.selection_config)
            inner_k = int(
                self.selection_config.candidate_k[0]
                if len(self.selection_config.candidate_k) == 1
                else self._choose_k_via_inner_cv(
                    X_train=X_train,
                    y_train=y_train,
                    groups_train=groups_train,
                    feature_names=train_bundle.feature_names,
                )
            )
            selection = selector.fit(
                X=X_train,
                y=y_train,
                feature_names=train_bundle.feature_names,
                k=inner_k,
            )
            X_train_sel = selector.transform(X_train)
            X_test_sel = selector.transform(X_test)
            selected_indices_per_fold.append(selection.selected_indices)
            selected_names_per_fold.append(selection.selected_names)

            # ---- Inner GroupKFold hyperparameter tuning ----
            best_estimator = self._fit_classifier_with_inner_cv(
                X_train_sel=X_train_sel,
                y_train=y_train,
                groups_train=groups_train,
            )

            # ---- Sigmoid calibration on training-fold predictions ----
            calibrated = wrap_with_sigmoid_calibration(
                classifier=best_estimator,
                inner_folds=3,
                random_state=self.evaluation_config.random_state,
            )
            calibrated.fit(X_train_sel, y_train)

            # ---- Predict held-out subject ----
            with Timer("predict_subject") as t:
                proba = calibrated.predict_proba(X_test_sel)
            prediction_times.append(t.elapsed_s)

            classes_aligned = list(calibrated.classes_)
            # Realign columns to the configured class order so downstream
            # code can safely index by ``self.classes``.
            proba_aligned = self._align_probabilities(
                proba=proba, source_classes=classes_aligned
            )

            predictions, mean_probs = aggregate_subject_predictions(
                proba=proba_aligned,
                groups=test_groups,
                classes=self.classes,
                rule=self.evaluation_config.aggregation_rule,
            )

            y_true_subject.append(subject_labels[test_subject])
            y_pred_subject.append(predictions[test_subject])
            subject_proba[test_subject] = mean_probs[test_subject]
            _LOGGER.info(
                "LOSO fold complete: subject=%s, true=%s, pred=%s",
                test_subject, subject_labels[test_subject], predictions[test_subject],
            )

        return LOSOResult(
            classes=self.classes,
            y_true_subject=y_true_subject,
            y_pred_subject=y_pred_subject,
            subject_probabilities=subject_proba,
            selected_feature_indices_per_fold=selected_indices_per_fold,
            selected_feature_names_per_fold=selected_names_per_fold,
            feature_time_s=feature_time_total,
            predict_time_s_per_subject=float(np.mean(prediction_times)) if prediction_times else 0.0,
            held_out_predictions={
                sid: pred for sid, pred in zip(subject_ids, y_pred_subject)
                if sid not in subject_proba
            },
        )

    # ------------------------------------------------------------------
    # Helpers.
    # ------------------------------------------------------------------
    def _choose_k_via_inner_cv(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        groups_train: np.ndarray,
        feature_names: Sequence[str],
    ) -> int:
        """Pick the best MI ``k`` via inner GroupKFold accuracy."""
        n_folds = max(2, self.evaluation_config.inner_folds)
        unique_groups = np.unique(groups_train)
        n_folds = min(n_folds, len(unique_groups))
        if n_folds < 2:
            return int(self.selection_config.candidate_k[0])
        inner = GroupKFold(n_splits=n_folds)
        scores: Dict[int, List[float]] = {
            k: [] for k in self.selection_config.candidate_k
        }
        for tr_idx, va_idx in inner.split(X_train, y_train, groups=groups_train):
            for k in self.selection_config.candidate_k:
                selector = LeakageSafeSelector(self.selection_config)
                selection = selector.fit(
                    X=X_train[tr_idx],
                    y=y_train[tr_idx],
                    feature_names=feature_names,
                    k=int(k),
                )
                X_tr_sel = selector.transform(X_train[tr_idx])
                X_va_sel = selector.transform(X_train[va_idx])
                clf = build_classifier(self.classifier_config)
                clf.fit(X_tr_sel, y_train[tr_idx])
                preds = clf.predict(X_va_sel)
                scores[k].append(float((preds == y_train[va_idx]).mean()))
                # Avoid touching huge memory blocks more than necessary.
                del selection
        average_scores = {k: float(np.mean(v)) for k, v in scores.items() if v}
        best_k = max(average_scores, key=average_scores.get)
        return int(best_k)

    def _fit_classifier_with_inner_cv(
        self,
        X_train_sel: np.ndarray,
        y_train: np.ndarray,
        groups_train: np.ndarray,
    ) -> BaseEstimator:
        """Fit the classifier with inner GroupKFold hyperparameter tuning."""
        estimator = build_classifier(self.classifier_config)
        grid = getattr(estimator, "param_grid", {})
        unique_groups = np.unique(groups_train)
        if not grid or len(unique_groups) < 2:
            estimator.fit(X_train_sel, y_train)
            return estimator

        n_folds = min(self.evaluation_config.inner_folds, len(unique_groups))
        inner_cv = GroupKFold(n_splits=max(2, n_folds))
        search = GridSearchCV(
            estimator=estimator,
            param_grid=grid,
            cv=inner_cv.split(X_train_sel, y_train, groups=groups_train),
            scoring="accuracy",
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train_sel, y_train)
        return search.best_estimator_

    def _align_probabilities(
        self,
        proba: np.ndarray,
        source_classes: Sequence[str],
    ) -> np.ndarray:
        """Reorder classifier probability columns to match ``self.classes``."""
        source_classes = list(source_classes)
        index = [source_classes.index(cls) for cls in self.classes]
        return proba[:, index]
