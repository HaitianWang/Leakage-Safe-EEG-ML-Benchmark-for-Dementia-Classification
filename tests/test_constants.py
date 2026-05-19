"""Lightweight sanity checks on the constant tables.

These tests do not require any EEG data or external dependencies; they verify
that the channel / region / band tables stay consistent with the manuscript
specification.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_benchml.constants import (  # noqa: E402
    BAND_ORDER,
    BANDS,
    CHANNELS_10_20,
    CHANNEL_TO_REGION,
    REGIONS,
    list_channel_pairs,
)


def test_channels_count() -> None:
    """The reference dataset uses the standard 19-channel 10--20 montage."""
    assert len(CHANNELS_10_20) == 19


def test_regions_partition_channels() -> None:
    """Every 10--20 channel must belong to exactly one anatomical region."""
    in_region = sorted(
        ch for chs in REGIONS.values() for ch in chs
    )
    assert sorted(CHANNELS_10_20) == in_region
    for channel in CHANNELS_10_20:
        assert channel in CHANNEL_TO_REGION


def test_band_order_matches_band_dict() -> None:
    """The exposed band order has to match the band definition table."""
    assert tuple(BANDS.keys()) == BAND_ORDER


def test_pairwise_count() -> None:
    """19 channels yield 171 unordered pairs (Section 4.3.3 of the manuscript)."""
    pairs = list_channel_pairs()
    assert len(pairs) == 19 * 18 // 2
