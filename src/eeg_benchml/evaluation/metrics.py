"""Subject-level metric computation.

Section 4.1 of the manuscript reports accuracy, balanced accuracy, F1-score,
sensitivity, specificity, Matthews correlation coefficient, and ROC-AUC. For
binary tasks, sensitivity and F1-score correspond to the first class named in
the task; specificity corresponds to the second class. For the three-class
task, sensitivity, specificity, F1-score, and AUC are macro-averaged using a
one-vs-rest formulation.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)


def compute_subject_level_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    mean_probabilities: np.ndarray,
    classes: Sequence[str],
) -> Dict[str, float]:
    """Compute the metric dictionary reported in the manuscript.

    Parameters
    ----------
    y_true, y_pred : sequence of str
        Ground-truth and predicted subject-level labels.
    mean_probabilities : ndarray of shape ``(n_subjects, n_classes)``
        Mean class probabilities per subject (matching the order of ``y_true``).
    classes : sequence of str
        Class labels aligned with the columns of ``mean_probabilities``.

    Returns
    -------
    metrics : dict[str, float]
        Per-metric scalar values. All values are returned as percentages so
        they can be directly compared with the manuscript's reporting.
    """
    classes = list(classes)
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    probs = np.asarray(mean_probabilities)

    metrics: Dict[str, float] = {}
    metrics["accuracy"] = 100.0 * accuracy_score(y_true_arr, y_pred_arr)
    metrics["balanced_accuracy"] = 100.0 * balanced_accuracy_score(
        y_true_arr, y_pred_arr
    )

    if len(classes) == 2:
        positive_label = classes[0]
        negative_label = classes[1]
        metrics["sensitivity"] = 100.0 * recall_score(
            y_true_arr, y_pred_arr, pos_label=positive_label, zero_division=0.0
        )
        metrics["specificity"] = 100.0 * recall_score(
            y_true_arr, y_pred_arr, pos_label=negative_label, zero_division=0.0
        )
        metrics["f1"] = 100.0 * f1_score(
            y_true_arr, y_pred_arr, pos_label=positive_label, zero_division=0.0
        )
        positive_idx = classes.index(positive_label)
        binary_truth = (y_true_arr == positive_label).astype(int)
        try:
            metrics["auc"] = 100.0 * roc_auc_score(binary_truth, probs[:, positive_idx])
        except ValueError:
            metrics["auc"] = float("nan")
    else:
        metrics["sensitivity"] = 100.0 * recall_score(
            y_true_arr, y_pred_arr, average="macro", zero_division=0.0
        )
        # Per-class specificity is computed manually since sklearn lacks a
        # direct helper for multi-class one-vs-rest specificity.
        per_class_specificity = []
        for label in classes:
            negative_mask = y_true_arr != label
            if negative_mask.sum() == 0:
                continue
            negative_pred_mask = y_pred_arr != label
            per_class_specificity.append(
                float((negative_mask & negative_pred_mask).sum() / negative_mask.sum())
            )
        metrics["specificity"] = (
            100.0 * float(np.mean(per_class_specificity))
            if per_class_specificity
            else float("nan")
        )
        metrics["f1"] = 100.0 * f1_score(
            y_true_arr, y_pred_arr, average="macro", zero_division=0.0
        )
        try:
            metrics["auc"] = 100.0 * roc_auc_score(
                y_true_arr,
                probs,
                multi_class="ovr",
                labels=classes,
                average="macro",
            )
        except ValueError:
            metrics["auc"] = float("nan")

    try:
        metrics["mcc"] = 100.0 * matthews_corrcoef(y_true_arr, y_pred_arr)
    except ValueError:
        metrics["mcc"] = float("nan")
    return metrics
