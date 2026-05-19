"""Sigmoid calibration wrapper.

The manuscript reports that classifiers without native probability outputs
(e.g. linear SVM and shrinkage LDA) are wrapped by a sigmoid (Platt) scaler
fitted only on training subjects. This allows the same mean-probability
aggregation rule to be applied across all classifiers.
"""

from __future__ import annotations

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold


def wrap_with_sigmoid_calibration(
    classifier: BaseEstimator,
    inner_folds: int = 3,
    random_state: int = 42,
) -> ClassifierMixin:
    """Return ``classifier`` wrapped in a sigmoid-calibrated ensemble.

    Parameters
    ----------
    classifier : sklearn estimator
        Any sklearn-compatible classifier. Estimators that already expose a
        well-calibrated :meth:`predict_proba` (logistic regression, random
        forest) still benefit from the wrapper because the same code path is
        used for all classifiers in the benchmark.
    inner_folds : int
        Number of folds used by :class:`CalibratedClassifierCV` for the
        sigmoid fit. The split is stratified to keep both classes inside each
        fold.
    random_state : int
        Seed for reproducibility.
    """
    splitter = StratifiedKFold(
        n_splits=inner_folds, shuffle=True, random_state=random_state
    )
    return CalibratedClassifierCV(
        estimator=classifier,
        method="sigmoid",
        cv=splitter,
    )
