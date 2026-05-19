"""Permutation-importance interpretation aggregated by region and band."""

from .permutation import compute_permutation_importance
from .regions import (
    aggregate_importance_by_band,
    aggregate_importance_by_channel,
    aggregate_importance_by_family,
    aggregate_importance_by_region,
)

__all__ = [
    "compute_permutation_importance",
    "aggregate_importance_by_band",
    "aggregate_importance_by_channel",
    "aggregate_importance_by_family",
    "aggregate_importance_by_region",
]
