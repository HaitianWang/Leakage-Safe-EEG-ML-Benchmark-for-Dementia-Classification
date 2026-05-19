"""Aggregate per-feature importance by family, band, channel, and region.

The aggregation logic uses the structured feature-name prefix produced by the
extractor:

* ``spec_<channel>_<descriptor>_<band>?``
* ``cmpl_<channel>_<descriptor>``
* ``conn_wpli_<band>_<channel-pair>``
* ``graph_<descriptor>_<band>``

When the prefix does not encode a band or channel, that feature is skipped in
the corresponding aggregation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Sequence

import numpy as np

from ..constants import BAND_ORDER, CHANNEL_TO_REGION, CHANNELS_10_20, REGIONS
from ..features.extractor import feature_family_of


def _channels_in_name(name: str) -> Sequence[str]:
    """Return the channel(s) referenced by a feature name."""
    family = feature_family_of(name)
    if family in {"spectral", "complexity"}:
        # spec_<CH>_... or cmpl_<CH>_...
        parts = name.split("_", 2)
        if len(parts) < 3:
            return ()
        return (parts[1],)
    if family == "connectivity":
        # conn_wpli_<band>_<chA>-<chB>
        pair_token = name.split("_")[-1]
        if "-" in pair_token:
            return tuple(pair_token.split("-"))
        return ()
    return ()


def _band_in_name(name: str) -> str:
    """Return the band identifier referenced by a feature name (or empty string)."""
    for band in BAND_ORDER:
        if name.endswith(f"_{band}") or f"_{band}_" in name:
            return band
    return ""


def aggregate_importance_by_family(
    feature_names: Iterable[str], importance: np.ndarray
) -> Dict[str, float]:
    """Sum permutation importance scores per feature family."""
    family_totals: Dict[str, float] = defaultdict(float)
    for name, value in zip(feature_names, importance):
        try:
            family_totals[feature_family_of(name)] += float(value)
        except ValueError:
            continue
    return dict(family_totals)


def aggregate_importance_by_band(
    feature_names: Iterable[str], importance: np.ndarray
) -> Dict[str, float]:
    """Sum permutation importance scores per frequency band."""
    band_totals: Dict[str, float] = defaultdict(float)
    for name, value in zip(feature_names, importance):
        band = _band_in_name(name)
        if not band:
            continue
        band_totals[band] += float(value)
    return dict(band_totals)


def aggregate_importance_by_channel(
    feature_names: Iterable[str], importance: np.ndarray
) -> Dict[str, float]:
    """Sum permutation importance scores per channel.

    For connectivity features, the importance is split evenly between the
    two channels involved in the pair.
    """
    channel_totals: Dict[str, float] = defaultdict(float)
    for name, value in zip(feature_names, importance):
        channels = _channels_in_name(name)
        if not channels:
            continue
        share = float(value) / max(1, len(channels))
        for channel in channels:
            if channel in CHANNELS_10_20:
                channel_totals[channel] += share
    return dict(channel_totals)


def aggregate_importance_by_region(
    feature_names: Iterable[str], importance: np.ndarray
) -> Dict[str, float]:
    """Sum permutation importance per anatomical region."""
    region_totals: Dict[str, float] = defaultdict(float)
    for name, value in zip(feature_names, importance):
        channels = _channels_in_name(name)
        if not channels:
            continue
        share = float(value) / max(1, len(channels))
        for channel in channels:
            region = CHANNEL_TO_REGION.get(channel)
            if region:
                region_totals[region] += share
    # Ensure every region appears in the output for downstream plotting.
    for region in REGIONS:
        region_totals.setdefault(region, 0.0)
    return dict(region_totals)
