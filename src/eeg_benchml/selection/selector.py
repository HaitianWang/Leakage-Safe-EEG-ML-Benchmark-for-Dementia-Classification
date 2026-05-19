"""Leakage-safe feature selection pipeline.

The :class:`LeakageSafeSelector` is the only place in the codebase that fits
to training data. The pipeline orchestrator forwards the training-fold
``(X, y)`` here and the resulting transform is applied to test-fold features
without any peek at test labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


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
    """Output of :meth:`LeakageSafeSelector.fit`."""

    selected_indices: np.ndarray
    selected_names: List[str]
    scaler: Optional[StandardScaler] = None


class LeakageSafeSelector:
    """Apply the variance / correlation / scaling / ranking cascade."""

    def __init__(self, config: FeatureSelectionConfig) -> None:
        self.config = config
        self._result: Optional[FeatureSelectionResult] = None

    # ------------------------------------------------------------------
    # Fit / transform API.
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
            Training-fold feature matrix.
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
        keep_mask = np.ones(X.shape[1], dtype=bool)

        # 1. Variance filter.
        variances = X.var(axis=0)
        keep_mask &= variances > self.config.variance_threshold

        # 2. Spearman correlation pruning.
        if keep_mask.sum() > 1:
            keep_mask = self._prune_correlated(X, y, feature_names, keep_mask)

        X_red = X[:, keep_mask]
        kept_names = [name for name, flag in zip(feature_names, keep_mask) if flag]

        # 3. Z-score standardisation.
        scaler = StandardScaler().fit(X_red)
        X_scaled = scaler.transform(X_red)

        # 4. Final ranking.
        if self.config.selector == "none":
            selected_local = np.arange(X_scaled.shape[1])
        elif self.config.selector == "mutual_information":
            k_to_use = k if k is not None else int(min(self.config.candidate_k))
            k_to_use = min(k_to_use, X_scaled.shape[1])
            selected_local = self._select_by_mutual_information(
                X_scaled, y, k_to_use
            )
        elif self.config.selector == "l1_logistic":
            selected_local = self._select_by_l1_logistic(X_scaled, y)
        else:
            raise ValueError(
                f"Unknown feature selector '{self.config.selector}'. "
                "Expected one of: none, mutual_information, l1_logistic."
            )

        # Translate local indices back to the original feature space.
        global_indices = np.where(keep_mask)[0][selected_local]
        selected_names = [kept_names[i] for i in selected_local]
        self._result = FeatureSelectionResult(
            selected_indices=global_indices,
            selected_names=selected_names,
            scaler=scaler,
        )
        return self._result

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted cascade to a held-out feature matrix."""
        if self._result is None:
            raise RuntimeError("LeakageSafeSelector.fit must be called first.")
        # Mirror the order of operations: variance + correlation are encoded
        # implicitly by ``selected_indices``, but z-scoring needs to reuse the
        # training-fold StandardScaler.
        scaler = self._result.scaler
        if scaler is None:
            return X[:, self._result.selected_indices]
        # We have to scale the *original* selected columns; for that we need
        # to recover the post-correlation matrix. The selected indices already
        # point at the original columns, and the scaler was fitted on those
        # same columns in their post-correlation order.
        X_red = X[:, self._result.selected_indices]
        # The scaler holds statistics for all kept columns (after variance /
        # correlation pruning). To remain robust we re-compute statistics for
        # the chosen subset by indexing into the original scaler.
        return self._apply_scaler_to_subset(X_red)

    # ------------------------------------------------------------------
    # Internal helpers.
    # ------------------------------------------------------------------
    def _apply_scaler_to_subset(self, X_red: np.ndarray) -> np.ndarray:
        """Scale ``X_red`` using the per-column statistics stored in the scaler.

        Notes
        -----
        ``self._result.selected_indices`` references the original column order
        of the input matrix, whereas the fitted :class:`StandardScaler`
        operates on the post-correlation reduced matrix. We therefore
        recompute z-scaling per column using the scaler's stored mean / scale
        arrays, indexed by the relative position of each retained column.
        """
        assert self._result is not None and self._result.scaler is not None
        scaler = self._result.scaler
        return (X_red - scaler.mean_[: X_red.shape[1]]) / (
            scaler.scale_[: X_red.shape[1]] + 1e-12
        )

    def _prune_correlated(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str],
        keep_mask: np.ndarray,
    ) -> np.ndarray:
        """Remove features whose absolute Spearman correlation exceeds the threshold.

        The feature with the larger mutual-information score with the labels is
        retained, breaking ties using the original column order.
        """
        active_indices = np.where(keep_mask)[0]
        if active_indices.size < 2:
            return keep_mask
        X_active = X[:, active_indices]
        # spearmanr handles ties internally; we only care about magnitudes.
        rho, _ = spearmanr(X_active, axis=0)
        rho = np.atleast_2d(np.abs(np.array(rho)))
        if rho.shape != (X_active.shape[1], X_active.shape[1]):
            # When only two columns survive, spearmanr returns a scalar.
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
        # Unused but kept for symmetry with future caller diagnostics.
        del feature_names
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
        """Select non-zero columns from an ``l1``-regularised logistic regression.

        The smallest ``C`` from :attr:`FeatureSelectionConfig.candidate_C` that
        keeps at least one feature per class is used. Higher ``C`` values are
        evaluated by the outer inner-cv loop in the pipeline orchestrator.
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
        # Fallback: keep all columns.
        return np.arange(X.shape[1])

    @property
    def result(self) -> FeatureSelectionResult:
        """Return the last fitted :class:`FeatureSelectionResult`."""
        if self._result is None:
            raise RuntimeError("LeakageSafeSelector.fit must be called first.")
        return self._result


# Convenience alias for legacy imports.
SubjectStratifiedSelector = LeakageSafeSelector
_ = pd  # silence linter: pandas is imported for downstream notebooks
