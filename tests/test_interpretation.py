"""Interpretation aggregation tests for the new feature names."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_benchml.constants import REGIONS  # noqa: E402
from eeg_benchml.interpretation.regions import (  # noqa: E402
    aggregate_importance_by_band,
    aggregate_importance_by_channel,
    aggregate_importance_by_family,
    aggregate_importance_by_region,
)


def test_family_aggregation() -> None:
    feature_names = [
        "spec_Fp1_log_power_delta",
        "cmpl_Fp1_hjorth_complexity_alpha",
        "conn_wpli_alpha_Fp1-Fp2",
        "graph_global_efficiency_beta",
        "graph_wpli_theta_frontal-temporal",
    ]
    importance = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    by_family = aggregate_importance_by_family(feature_names, importance)
    assert by_family["spectral"] == 1.0
    assert by_family["complexity"] == 2.0
    assert by_family["connectivity"] == 3.0
    assert by_family["graph"] == 4.0 + 5.0


def test_band_aggregation_handles_complexity_band_suffix() -> None:
    feature_names = [
        "cmpl_Fp1_hjorth_mobility_delta",
        "cmpl_Fp1_sample_entropy_alpha",
        "spec_Fp1_rel_power_beta",
        "graph_clustering_theta",
    ]
    importance = np.array([1.0, 1.0, 1.0, 1.0])
    by_band = aggregate_importance_by_band(feature_names, importance)
    assert by_band["delta"] == 1.0
    assert by_band["alpha"] == 1.0
    assert by_band["beta"] == 1.0
    assert by_band["theta"] == 1.0


def test_region_aggregation_uses_channel_and_region_tokens() -> None:
    feature_names = [
        "spec_Fp1_log_power_alpha",            # frontal channel
        "spec_O1_log_power_alpha",              # occipital channel
        "graph_wpli_alpha_frontal-temporal",   # region-pair feature
    ]
    importance = np.array([2.0, 4.0, 6.0])
    by_region = aggregate_importance_by_region(feature_names, importance)
    # The graph between-region feature splits its importance between the two
    # named regions.
    assert by_region["frontal"] == 2.0 + 3.0
    assert by_region["temporal"] == 3.0
    assert by_region["occipital"] == 4.0
    # Every region appears in the output (zero default).
    for region in REGIONS:
        assert region in by_region


def test_channel_aggregation_splits_connectivity_pairs() -> None:
    feature_names = ["conn_wpli_alpha_Fp1-Fp2"]
    importance = np.array([10.0])
    by_channel = aggregate_importance_by_channel(feature_names, importance)
    assert by_channel["Fp1"] == 5.0
    assert by_channel["Fp2"] == 5.0
