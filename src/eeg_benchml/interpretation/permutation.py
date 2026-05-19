"""Subject-level permutation importance.

The interpretation step described in Section 4 of the manuscript estimates a
permutation importance score per feature using held-out subject-level
predictions. The score is the drop in accuracy caused by randomly permuting
the values of a single feature column across all epochs of the held-out
subjects.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

from ..evaluation.aggregation import aggregate_subject_predictions


def compute_permutation_importance(
    estimator: BaseEstimator,
    X: np.ndarray,
    y_subject: Sequence[str],
    groups: Sequence[str],
    classes: Sequence[str],
    feature_names: Sequence[str],
    n_repeats: int = 10,
    rule: str = "mean_probability",
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute permutation importance from epoch-level features.

    Parameters
    ----------
    estimator : sklearn estimator
        Already-fitted, calibrated classifier (e.g. the output of the LOSO
        loop). Must expose ``predict_proba``.
    X : ndarray of shape ``(n_epochs, n_features)``
        Epoch-level feature matrix for the held-out subjects.
    y_subject : sequence of str
        Subject-level ground-truth labels, indexed by the unique values of
        ``groups``.
    groups : sequence of str
        Subject identifier per epoch row.
    classes : sequence of str
        Class label order used by the classifier.
    feature_names : sequence of str
        Column names for ``X``.
    n_repeats : int
        Number of permutation repetitions.
    rule : str
        Subject-level aggregation rule.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    importance_mean, importance_std : ndarray of shape ``(n_features,)``
        Mean and standard deviation of the accuracy drop caused by permuting
        each feature column. Values are expressed in percentage points.
    """
    rng = np.random.default_rng(random_state)
    groups = np.asarray(groups)
    classes_list = list(classes)
    feature_names = list(feature_names)
    y_subject_arr = np.asarray(y_subject)

    base_proba = estimator.predict_proba(X)
    base_predictions, _ = aggregate_subject_predictions(
        proba=base_proba, groups=groups, classes=classes_list, rule=rule
    )
    unique_groups = sorted(set(groups))
    if len(unique_groups) != len(y_subject_arr):
        raise ValueError(
            "Mismatch between number of held-out subjects and length of y_subject."
        )
    base_score = accuracy_score(
        y_subject_arr,
        np.array([base_predictions[s] for s in unique_groups]),
    )

    n_features = X.shape[1]
    importance = np.zeros((n_repeats, n_features), dtype=float)
    for rep in range(n_repeats):
        for j in range(n_features):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            proba = estimator.predict_proba(X_perm)
            preds, _ = aggregate_subject_predictions(
                proba=proba, groups=groups, classes=classes_list, rule=rule
            )
            score = accuracy_score(
                y_subject_arr,
                np.array([preds[s] for s in unique_groups]),
            )
            importance[rep, j] = (base_score - score) * 100.0
    return importance.mean(axis=0), importance.std(axis=0)
