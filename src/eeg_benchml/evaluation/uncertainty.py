"""Subject-level bootstrap confidence intervals and paired bootstrap tests.

Table 3 of the manuscript reports 95 % bootstrap confidence intervals on
accuracy / AUC and a paired bootstrap p-value against the selected reference
pipeline. Both routines resample subjects (not epochs) so that they preserve
the subject-level evaluation guarantee.
"""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np

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
        Function with signature ``metric_fn(y_true, y_pred) -> float`` that
        returns a scalar score (e.g. accuracy as a percentage).
    y_true, y_pred : ndarray of shape ``(n_subjects,)``
        Subject-level ground-truth and predicted labels (or, for AUC,
        subject-level scores aligned with ``y_true``).
    n_resamples : int
        Number of bootstrap resamples.
    confidence_level : float
        Desired confidence level, e.g. 0.95.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    lower, upper : tuple of float
        Lower and upper bootstrap percentiles. The returned values follow the
        same units as ``metric_fn``.
    """
    rng = np.random.default_rng(random_state)
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
    obtained from the proportion of resamples with the opposite sign of the
    observed difference. The implementation follows the same convention as
    Table 3 of the manuscript.
    """
    rng = np.random.default_rng(random_state)
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

    # Centre the differences before computing the p-value; this yields a valid
    # null distribution under the assumption of no systematic difference.
    centred = differences - differences.mean()
    p_value = float(np.mean(np.abs(centred) >= abs(observed)))
    return p_value


def metric_accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Plain accuracy as a percentage, ready to be plugged into the bootstrap."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return float("nan")
    return 100.0 * float((y_true == y_pred).mean())
