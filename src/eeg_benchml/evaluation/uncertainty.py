"""Subject-level bootstrap confidence intervals and paired bootstrap tests.

Table 3 of the manuscript reports 95 % bootstrap confidence intervals on
subject-level accuracy and AUC and a paired bootstrap p-value against the
selected reference pipeline. Both routines resample **subjects** (not
epochs), preserving the leakage-safe subject-level evaluation guarantee.

The metric helpers (:func:`metric_accuracy`, :func:`metric_auc_binary`,
:func:`metric_auc_ovr`) are deliberately plain functions so they can be
plugged directly into :func:`bootstrap_confidence_interval` and
:func:`paired_bootstrap_test` without additional wiring.
"""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

MetricFn = Callable[[np.ndarray, np.ndarray], float]


def bootstrap_confidence_interval(
    metric_fn: MetricFn,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float]:
    """Compute a percentile-based bootstrap confidence interval.

    Parameters
    ----------
    metric_fn : callable
        ``metric_fn(y_true, y_pred) -> float``. ``y_pred`` may hold
        predicted labels (for accuracy) or class scores (for AUC).
    y_true, y_pred : ndarray of shape ``(n_subjects,)`` or ``(n_subjects, n_classes)``
        Subject-level ground-truth and predictions / scores.
    n_resamples : int
        Number of bootstrap resamples. The manuscript uses 1000.
    confidence_level : float
        Desired confidence level (default 0.95).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    lower, upper : tuple of float
        Lower and upper percentiles, in the same units as ``metric_fn``.
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = y_true.shape[0]
    if n == 0:
        return float("nan"), float("nan")

    scores = np.zeros(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(low=0, high=n, size=n)
        try:
            scores[i] = metric_fn(y_true[idx], y_pred[idx])
        except Exception:
            scores[i] = np.nan
    alpha = (1.0 - confidence_level) / 2.0
    lower = float(np.nanpercentile(scores, 100.0 * alpha))
    upper = float(np.nanpercentile(scores, 100.0 * (1.0 - alpha)))
    return lower, upper


def paired_bootstrap_test(
    metric_fn: MetricFn,
    y_true: np.ndarray,
    y_pred_reference: np.ndarray,
    y_pred_other: np.ndarray,
    n_resamples: int = 1000,
    random_state: int = 42,
) -> float:
    """Two-sided paired bootstrap p-value for ``other`` vs ``reference``.

    For each bootstrap resample we compute the metric difference
    ``metric(other) - metric(reference)`` and return the two-sided p-value
    obtained by comparing the centred null distribution against the observed
    difference. This is the convention used by Table 3 of the manuscript.
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_pred_reference = np.asarray(y_pred_reference)
    y_pred_other = np.asarray(y_pred_other)
    n = y_true.shape[0]
    if n == 0:
        return float("nan")

    observed = metric_fn(y_true, y_pred_other) - metric_fn(y_true, y_pred_reference)
    differences = np.zeros(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(low=0, high=n, size=n)
        ref = metric_fn(y_true[idx], y_pred_reference[idx])
        oth = metric_fn(y_true[idx], y_pred_other[idx])
        differences[i] = oth - ref

    centred = differences - differences.mean()
    return float(np.mean(np.abs(centred) >= abs(observed)))


# ---------------------------------------------------------------------------
# Metric helpers.
# ---------------------------------------------------------------------------
def metric_accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Plain accuracy as a percentage, ready for the bootstrap helpers."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return float("nan")
    return 100.0 * float((y_true == y_pred).mean())


def metric_auc_binary(positive_label: str) -> MetricFn:
    """Return a binary-AUC metric callable bound to ``positive_label``.

    The returned function accepts a 1-D probability vector ``y_pred`` of the
    positive class. Resampling that produces a single-class subset returns
    ``nan`` so that downstream percentile aggregation skips the resample.
    """

    def _auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        binary_truth = (np.asarray(y_true) == positive_label).astype(int)
        if binary_truth.sum() in (0, binary_truth.size):
            return float("nan")
        try:
            return 100.0 * float(roc_auc_score(binary_truth, np.asarray(y_pred)))
        except ValueError:
            return float("nan")

    _auc.__name__ = f"auc_binary[{positive_label}]"
    return _auc


def metric_auc_ovr(classes: Sequence[str]) -> MetricFn:
    """Return a macro-averaged one-vs-rest AUC metric callable.

    Parameters
    ----------
    classes : sequence of str
        Class order matching the columns of the probability matrix passed in
        as ``y_pred``.
    """
    classes_list = list(classes)

    def _auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            return 100.0 * float(
                roc_auc_score(
                    np.asarray(y_true),
                    np.asarray(y_pred),
                    multi_class="ovr",
                    labels=classes_list,
                    average="macro",
                )
            )
        except ValueError:
            return float("nan")

    _auc.__name__ = "auc_ovr"
    return _auc
