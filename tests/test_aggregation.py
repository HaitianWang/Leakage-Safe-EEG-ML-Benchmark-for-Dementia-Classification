"""Aggregation rule sanity tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_benchml.evaluation.aggregation import aggregate_subject_predictions  # noqa: E402


def test_mean_probability_aggregation() -> None:
    proba = np.array([
        [0.2, 0.8],
        [0.4, 0.6],
        [0.7, 0.3],
        [0.9, 0.1],
    ])
    groups = ["sub-001", "sub-001", "sub-002", "sub-002"]
    classes = ["AD", "CN"]
    preds, mean_probs = aggregate_subject_predictions(
        proba=proba, groups=groups, classes=classes, rule="mean_probability"
    )
    assert preds["sub-001"] == "CN"
    assert preds["sub-002"] == "AD"
    assert np.isclose(mean_probs["sub-001"][1], 0.7)
    assert np.isclose(mean_probs["sub-002"][0], 0.8)


def test_majority_voting_aggregation() -> None:
    proba = np.array([
        [0.4, 0.6],
        [0.55, 0.45],
        [0.6, 0.4],
        [0.51, 0.49],
    ])
    groups = ["sub-001"] * 4
    classes = ["AD", "CN"]
    preds, _ = aggregate_subject_predictions(
        proba=proba, groups=groups, classes=classes, rule="majority_voting"
    )
    # Three of four epochs vote for AD, one for CN.
    assert preds["sub-001"] == "AD"
