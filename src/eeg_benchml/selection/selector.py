"""Leakage-safe feature selection pipeline.

The :class:`LeakageSafeSelector` implements the cascade described in
Section 4.4 of the manuscript. Every step is fitted on training-fold data
only; no statistic estimated from test subjects is ever used for selection,
scaling, or hyper-parameter tuning.

Cascade
-------
1. **Variance filter**: features with variance below
   :math:`10^{-6}` are dropped.
2. **Spearman correlation pruning**: for every pair of features with
   :math:`|\\rho| > 0.95`, the feature with the smaller mutual-information
   score with the training labels is dropped.
3. **Z-score standardisation**: using training-fold mean and standard
   deviation.
4. **Primary ranking**: mutual-information top-:math:`k` (default) or
   :math:`\\ell_1`-regularised logistic regression.

The :meth:`LeakageSafeSelector.transform` method reapplies the fitted cascade
to a held-out feature matrix using the stored per-column statistics for the
selected columns only. This keeps the transform numerically identical to the
training-time selection while remaining trivial to audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression


@dataclass
class FeatureSelectionConfig:
    """Configuration for the leakage-safe selection cascade."""

    variance_threshold: float = 1.0e-6
    correlation_threshold: float = 0.95
    selector: str = "mutual_information"
    candidate_k: Sequence[int] = field(default_factory=lambda: (50, 100, 200, 400))
    candidate_C: Sequence[float] = field(
        default_factory=lambda: (0.01, 0.1, 1.0, 10.0)
    )
    random_state: int = 42


@dataclass
class FeatureSelectionResult:
    """Output of :meth:`LeakageSafeSelector.fit`.

    The selector stores the original-column indices of the retained features
    together with their per-column mean and scale. This makes the
    :meth:`LeakageSafeSelector.transform` reduction unambiguous and the
    fitted state directly inspectable.
    """

    selected_indices: np.ndarray
    selected_names: List[str]
    selected_mean: np.ndarray
    selected_scale: np.ndarray


class LeakageSafeSelector:
    """Apply the variance / correlation / scaling / ranking cascade."""

    def __init__(self, config: FeatureSelectionConfig) -> None:
        self.config = config
        self._result: Optional[FeatureSelectionResult] = None

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str],
        k: Optional[int] = None,
    ) -> FeatureSelectionResult:
        """Fit the cascade on training-fold features.

        Parameters
        ----------
        X : ndarray of shape ``(n_samples, n_features)``
            Training-fold feature matrix. Columns are indexed in the order
            produced by :class:`~eeg_benchml.features.FeatureExtractor`.
        y : ndarray of shape ``(n_samples,)``
            Training-fold class labels.
        feature_names : sequence of str
            Names matching the columns of ``X``.
        k : int, optional
            Number of features to retain (mutual-information selector only).
            When ``None``, the smallest candidate in
            :attr:`FeatureSelectionConfig.candidate_k` is used.
        """
        feature_names = list(feature_names)
        n_features = X.shape[1]
        keep_mask = np.ones(n_features, dtype=bool)

        # 1. Variance filter.
        variances = X.var(axis=0)
        keep_mask &= variances > self.config.variance_threshold

        # 2. Spearman correlation pruning.
        if keep_mask.sum() > 1:
            keep_mask = self._prune_correlated(X, y, keep_mask)

        # 3. Standardisation on the surviving columns.
        post_correlation_indices = np.where(keep_mask)[0]
        X_red = X[:, post_correlation_indices]
        mean_ = X_red.mean(axis=0)
        scale_ = X_red.std(axis=0)
        scale_[scale_ < 1e-12] = 1.0  # protect against constant columns
        X_scaled = (X_red - mean_) / scale_

        # 4. Final ranking.
        if self.config.selector == "none":
            local_indices = np.arange(X_scaled.shape[1])
        elif self.config.selector == "mutual_information":
            k_to_use = k if k is not None else int(min(self.config.candidate_k))
            k_to_use = min(k_to_use, X_scaled.shape[1])
            local_indices = self._select_by_mutual_information(
                X_scaled, y, k_to_use
            )
        elif self.config.selector == "l1_logistic":
            local_indices = self._select_by_l1_logistic(X_scaled, y)
        else:
            raise ValueError(
                f"Unknown feature selector '{self.config.selector}'. "
                "Expected one of: none, mutual_information, l1_logistic."
            )

        # Map local indices (within the post-correlation reduced matrix) back
        # to the original-column indices, and retain only the corresponding
        # scaler statistics. This eliminates the position mismatch that
        # plagues naive implementations and keeps ``transform`` trivial.
        global_indices = post_correlation_indices[local_indices]
        selected_mean = mean_[local_indices]
        selected_scale = scale_[local_indices]
        selected_names = [feature_names[idx] for idx in global_indices]

        self._result = FeatureSelectionResult(
            selected_indices=global_indices,
            selected_names=selected_names,
            selected_mean=selected_mean,
            selected_scale=selected_scale,
        )
        return self._result

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted cascade to a held-out feature matrix.

        Parameters
        ----------
        X : ndarray of shape ``(n_samples, n_features)``
            Feature matrix in the same column order as the training fold.

        Returns
        -------
        X_selected : ndarray of shape ``(n_samples, len(selected_indices))``
            Z-scored values for the selected columns.
        """
        if self._result is None:
            raise RuntimeError("LeakageSafeSelector.fit must be called first.")
        X_selected = X[:, self._result.selected_indices]
        return (X_selected - self._result.selected_mean) / self._result.selected_scale

    # ------------------------------------------------------------------
    # Internal helpers.
    # ------------------------------------------------------------------
    def _prune_correlated(
        self,
        X: np.ndarray,
        y: np.ndarray,
        keep_mask: np.ndarray,
    ) -> np.ndarray:
        """Drop features whose absolute Spearman correlation exceeds the threshold.

        For every pair of correlated columns we retain the one with the
        larger mutual-information score with the training labels, matching
        the policy in Section 4.4 of the manuscript.
        """
        active_indices = np.where(keep_mask)[0]
        if active_indices.size < 2:
            return keep_mask
        X_active = X[:, active_indices]
        rho, _ = spearmanr(X_active, axis=0)
        rho = np.atleast_2d(np.abs(np.asarray(rho)))
        if rho.shape != (X_active.shape[1], X_active.shape[1]):
            # ``spearmanr`` returns a scalar for two columns.
            rho = np.array([[1.0, float(rho)], [float(rho), 1.0]])

        mi_scores = mutual_info_classif(
            X_active,
            y,
            discrete_features=False,
            random_state=self.config.random_state,
        )

        drop_mask = np.zeros(X_active.shape[1], dtype=bool)
        for i in range(X_active.shape[1]):
            if drop_mask[i]:
                continue
            for j in range(i + 1, X_active.shape[1]):
                if drop_mask[j]:
                    continue
                if rho[i, j] > self.config.correlation_threshold:
                    if mi_scores[i] >= mi_scores[j]:
                        drop_mask[j] = True
                    else:
                        drop_mask[i] = True
                        break

        new_keep_mask = keep_mask.copy()
        new_keep_mask[active_indices[drop_mask]] = False
        return new_keep_mask

    def _select_by_mutual_information(
        self, X: np.ndarray, y: np.ndarray, k: int
    ) -> np.ndarray:
        """Return indices of the ``k`` features with the highest MI score."""
        scores = mutual_info_classif(
            X,
            y,
            discrete_features=False,
            random_state=self.config.random_state,
        )
        order = np.argsort(-scores)
        return order[:k]

    def _select_by_l1_logistic(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Select the non-zero columns of an ``l1``-regularised logistic regression.

        We iterate through ``candidate_C`` from large to small and return the
        first sparse solution that retains at least one feature. The outer
        LOSO orchestrator can wrap this routine in an inner GroupKFold loop
        to choose ``C`` directly, but the simple heuristic is sufficient when
        only one ``C`` is provided.
        """
        for c_value in sorted(self.config.candidate_C, reverse=True):
            model = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=c_value,
                max_iter=5000,
                random_state=self.config.random_state,
            )
            model.fit(X, y)
            coefs = np.abs(model.coef_).max(axis=0)
            nonzero = np.where(coefs > 0)[0]
            if nonzero.size > 0:
                return nonzero
        return np.arange(X.shape[1])

    @property
    def result(self) -> FeatureSelectionResult:
        """Return the last fitted :class:`FeatureSelectionResult`."""
        if self._result is None:
            raise RuntimeError("LeakageSafeSelector.fit must be called first.")
        return self._result


# Convenience alias retained for legacy imports.
SubjectStratifiedSelector = LeakageSafeSelector
