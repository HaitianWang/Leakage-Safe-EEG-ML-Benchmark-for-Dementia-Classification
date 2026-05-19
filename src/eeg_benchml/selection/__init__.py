"""Fold-internal feature selection.

This package implements the leakage-safe selection cascade described in
Section 4.4 of the manuscript:

1. Variance filtering (threshold :math:`10^{-6}`).
2. Spearman correlation pruning (threshold :math:`0.95`).
3. Z-score standardisation on training-fold statistics.
4. A primary feature ranking step: mutual information top-:math:`k` or
   :math:`\\ell_1`-regularised logistic regression.

The selector is always fitted on training subjects only.
"""

from .selector import (
    FeatureSelectionConfig,
    FeatureSelectionResult,
    LeakageSafeSelector,
)

__all__ = [
    "FeatureSelectionConfig",
    "FeatureSelectionResult",
    "LeakageSafeSelector",
]
