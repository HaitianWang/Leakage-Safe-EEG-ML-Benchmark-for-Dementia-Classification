"""Subject-level evaluation: LOSO, aggregation, metrics, and uncertainty."""

from .aggregation import aggregate_subject_predictions
from .loso import LOSOEvaluator, LOSOResult
from .metrics import compute_subject_level_metrics
from .uncertainty import bootstrap_confidence_interval, paired_bootstrap_test

__all__ = [
    "aggregate_subject_predictions",
    "LOSOEvaluator",
    "LOSOResult",
    "compute_subject_level_metrics",
    "bootstrap_confidence_interval",
    "paired_bootstrap_test",
]
