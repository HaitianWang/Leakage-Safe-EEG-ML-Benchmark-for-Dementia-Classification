"""Leakage-safe selection sanity tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_benchml.selection import (  # noqa: E402
    FeatureSelectionConfig,
    LeakageSafeSelector,
)


def test_variance_filter_drops_constant_columns() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal(size=(40, 8))
    X[:, 3] = 0.0  # constant feature should be removed
    y = (X[:, 0] > 0).astype(int).astype(str)

    config = FeatureSelectionConfig(
        variance_threshold=1e-6,
        correlation_threshold=0.99,
        selector="mutual_information",
        candidate_k=(4,),
    )
    selector = LeakageSafeSelector(config)
    result = selector.fit(X=X, y=y, feature_names=[f"f{i}" for i in range(8)], k=4)
    assert 3 not in result.selected_indices.tolist()


def test_correlation_filter_keeps_only_one_of_perfectly_correlated_pair() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal(size=(50, 5))
    X[:, 1] = X[:, 0] * 1.0001  # near-perfectly correlated with column 0
    y = (X[:, 2] > 0).astype(int).astype(str)
    config = FeatureSelectionConfig(
        variance_threshold=1e-6,
        correlation_threshold=0.95,
        selector="none",
    )
    selector = LeakageSafeSelector(config)
    result = selector.fit(X=X, y=y, feature_names=[f"f{i}" for i in range(5)])
    assert not (0 in result.selected_indices and 1 in result.selected_indices)
