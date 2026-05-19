"""Constants shared across the EEG-BenchML pipeline.

The values defined here intentionally mirror the manuscript so that any
downstream code can be cross-checked against Section 3 (dataset description),
Section 4.3 (feature engineering), and Section 4.4 (feature aggregation).

Notes
-----
The expected dataset is the publicly available OpenNeuro ds004504 resting-state
EEG corpus. All identifiers, channel names, and anatomical groupings are taken
from the corresponding BIDS specification and are therefore not subject-
identifying. No subject-level metadata is stored in this file.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Diagnostic labels.
# ---------------------------------------------------------------------------
#: Canonical labels used by every internal module.
LABELS: Tuple[str, ...] = ("AD", "FTD", "CN")

#: Mapping from raw participant-table strings to canonical labels.
LABEL_ALIASES: Dict[str, str] = {
    "a": "AD",
    "ad": "AD",
    "alzheimer": "AD",
    "alzheimers": "AD",
    "alzheimer disease": "AD",
    "alzheimers disease": "AD",
    "alzheimer's disease": "AD",
    "f": "FTD",
    "ftd": "FTD",
    "frontotemporal dementia": "FTD",
    "c": "CN",
    "cn": "CN",
    "hc": "CN",
    "nc": "CN",
    "control": "CN",
    "healthy": "CN",
    "healthy control": "CN",
    "normal": "CN",
}

#: Benchmark classification tasks evaluated in the manuscript.
TASKS: Dict[str, Tuple[str, ...]] = {
    "ad_cn": ("AD", "CN"),
    "ftd_cn": ("FTD", "CN"),
    "ad_ftd": ("AD", "FTD"),
    "three": ("AD", "FTD", "CN"),
}

# ---------------------------------------------------------------------------
# EEG montage.
# ---------------------------------------------------------------------------
#: The 19 scalp channels of the international 10--20 montage used by the
#: reference dataset, in the order expected by the connectivity feature
#: extractor (so that pairwise indices remain stable across subjects).
CHANNELS_10_20: Tuple[str, ...] = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "O2",
)

#: Anatomical region grouping used by the region-level interpretation step.
#:
#: Although frontopolar and frontal electrodes are visually separated in
#: Fig. 2 of the manuscript for clarity, they are merged here into a single
#: ``frontal`` region for feature aggregation.
REGIONS: Dict[str, Tuple[str, ...]] = {
    "frontal":  ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"),
    "temporal": ("T3", "T4", "T5", "T6"),
    "central":  ("C3", "Cz", "C4"),
    "parietal": ("P3", "Pz", "P4"),
    "occipital": ("O1", "O2"),
}

#: Inverse channel -> region lookup, built once at import time.
CHANNEL_TO_REGION: Dict[str, str] = {
    ch: region for region, chs in REGIONS.items() for ch in chs
}

# ---------------------------------------------------------------------------
# Frequency bands.
# ---------------------------------------------------------------------------
#: Canonical EEG frequency bands used by the spectral and connectivity
#: feature extractors. Tuples follow ``(low_hz, high_hz)``.
BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma_low": (30.0, 45.0),
}

#: Ordered band labels, exposed separately so feature matrices keep a stable
#: column order across modules.
BAND_ORDER: Tuple[str, ...] = ("delta", "theta", "alpha", "beta", "gamma_low")

# ---------------------------------------------------------------------------
# Feature family identifiers.
# ---------------------------------------------------------------------------
#: Feature families used in the staged benchmark and the interpretation step.
FEATURE_FAMILIES: Tuple[str, ...] = ("spectral", "complexity", "connectivity", "graph")


def list_channel_pairs() -> List[Tuple[str, str]]:
    """Return all unordered channel pairs of the 19-channel 10--20 montage.

    Returns
    -------
    pairs : list of tuple[str, str]
        Pairwise tuples ``(c1, c2)`` with ``c1`` preceding ``c2`` in the
        canonical channel ordering. With 19 channels this returns 171 pairs.
    """
    pairs: List[Tuple[str, str]] = []
    n = len(CHANNELS_10_20)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((CHANNELS_10_20[i], CHANNELS_10_20[j]))
    return pairs
