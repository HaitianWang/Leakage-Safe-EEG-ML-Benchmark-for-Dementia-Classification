"""Graph-derived features from the region-level connectivity matrix.

Following Section 4.3.3 of the manuscript, the pairwise wPLI features are
first aggregated into the five anatomical regions (frontal, temporal,
central, parietal, occipital). For each band we then derive two groups of
descriptors from the resulting :math:`5 \\times 5` region-level connectivity
matrix:

* **Between-region wPLI means (10 per band)**: average wPLI for each of the
  :math:`\\binom{5}{2} = 10` unordered region pairs. These descriptors
  summarise inter-regional functional coordination beyond the raw 171-pair
  representation.
* **Global graph descriptors (4 per band)**: weighted mean node strength,
  weighted clustering coefficient, global efficiency, and characteristic
  path length, computed after pruning the weakest 70 % of edges.

With five frequency bands this yields :math:`5 \\times (10 + 4) = 70` graph
features per epoch, matching the dimensionality of the ``Connectivity +
graph`` row in Table 2 of the manuscript (:math:`925 - 855 = 70`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Tuple

import networkx as nx
import numpy as np

from ..constants import BAND_ORDER, CHANNEL_TO_REGION, REGIONS, list_channel_pairs

# Region order is fixed at import time so the column layout is stable across
# subjects and folds.
_REGION_ORDER: Tuple[str, ...] = tuple(REGIONS.keys())


@dataclass
class GraphConfig:
    """Configuration for graph-feature extraction."""

    edge_keep_proportion: float = 0.30
    descriptors: Iterable[str] = field(
        default_factory=lambda: (
            "mean_strength",
            "clustering",
            "global_efficiency",
            "char_path_length",
        )
    )


# ---------------------------------------------------------------------------
# Internal helpers.
# ---------------------------------------------------------------------------
def _region_matrix_from_pairs(
    wpli_per_band: np.ndarray, channel_pairs: List[Tuple[str, str]]
) -> np.ndarray:
    """Aggregate channel-pair wPLI into a symmetric region-level matrix.

    Parameters
    ----------
    wpli_per_band : ndarray of shape ``(n_epochs, n_pairs)``
        wPLI values for one frequency band.
    channel_pairs : list[tuple[str, str]]
        Channel pairs corresponding to the second dimension of
        ``wpli_per_band``.

    Returns
    -------
    matrix : ndarray of shape ``(n_epochs, n_regions, n_regions)``
        Symmetric region-level matrix per epoch. Diagonal entries hold the
        within-region wPLI mean; off-diagonal entries hold the between-region
        wPLI mean.
    """
    region_to_idx = {name: idx for idx, name in enumerate(_REGION_ORDER)}
    n_regions = len(_REGION_ORDER)
    n_epochs, _ = wpli_per_band.shape
    accum = np.zeros((n_epochs, n_regions, n_regions), dtype=float)
    counts = np.zeros((n_regions, n_regions), dtype=float)
    for k, (a, b) in enumerate(channel_pairs):
        ra = region_to_idx[CHANNEL_TO_REGION[a]]
        rb = region_to_idx[CHANNEL_TO_REGION[b]]
        i, j = (ra, rb) if ra <= rb else (rb, ra)
        accum[:, i, j] += wpli_per_band[:, k]
        if i != j:
            accum[:, j, i] += wpli_per_band[:, k]
        counts[i, j] += 1
        if i != j:
            counts[j, i] += 1
    counts[counts == 0] = 1.0
    return accum / counts


def _between_region_pairs() -> List[Tuple[int, int]]:
    """Return the 10 unordered between-region index pairs in a stable order."""
    n = len(_REGION_ORDER)
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _prune_weakest_edges(matrix: np.ndarray, keep_proportion: float) -> np.ndarray:
    """Zero-out the weakest edges so only the top ``keep_proportion`` remain.

    The mask is recomputed per epoch so that the resulting graph descriptors
    operate on the same proportion of edges across the cohort.
    """
    pruned = matrix.copy()
    n = matrix.shape[-1]
    triu_indices = np.triu_indices(n, k=1)
    upper = pruned[..., triu_indices[0], triu_indices[1]]
    if upper.size == 0:
        return pruned
    threshold = np.quantile(upper, 1.0 - keep_proportion, axis=-1, keepdims=True)
    mask = upper >= threshold
    upper = upper * mask
    pruned[..., triu_indices[0], triu_indices[1]] = upper
    pruned[..., triu_indices[1], triu_indices[0]] = upper
    return pruned


def _graph_descriptors(
    matrix: np.ndarray, descriptors: Iterable[str]
) -> Tuple[np.ndarray, List[str]]:
    """Compute a fixed set of weighted graph descriptors per epoch."""
    n_epochs, _, _ = matrix.shape
    descriptor_list = list(descriptors)
    out = np.zeros((n_epochs, len(descriptor_list)), dtype=float)
    names = list(descriptor_list)
    for i in range(n_epochs):
        graph = nx.from_numpy_array(matrix[i])
        for j, descriptor in enumerate(descriptor_list):
            out[i, j] = _compute_descriptor(graph, descriptor)
    return out, names


def _compute_descriptor(graph: "nx.Graph", descriptor: str) -> float:
    """Dispatch table for the weighted graph descriptors."""
    if descriptor == "mean_strength":
        strengths = dict(graph.degree(weight="weight"))
        return float(np.mean(list(strengths.values()))) if strengths else 0.0
    if descriptor == "clustering":
        clustering = nx.clustering(graph, weight="weight")
        return float(np.mean(list(clustering.values()))) if clustering else 0.0
    if descriptor in {"global_efficiency", "char_path_length"}:
        # For distance-based descriptors we invert weights so that strong
        # connectivity corresponds to a short distance (standard convention
        # for functional connectivity graphs).
        inv = graph.copy()
        for _, _, attrs in inv.edges(data=True):
            weight = attrs.get("weight", 1.0)
            attrs["weight"] = 1.0 / (weight + 1e-9)
        try:
            if descriptor == "global_efficiency":
                return float(nx.global_efficiency(inv))
            return float(nx.average_shortest_path_length(inv, weight="weight"))
        except Exception:
            return 0.0
    raise ValueError(f"Unknown graph descriptor '{descriptor}'.")


# ---------------------------------------------------------------------------
# Public extractor.
# ---------------------------------------------------------------------------
def compute_graph_features(
    wpli_features: np.ndarray, config: GraphConfig
) -> Tuple[np.ndarray, List[str]]:
    """Compute graph-derived features from pairwise wPLI features.

    Parameters
    ----------
    wpli_features : ndarray of shape ``(n_epochs, 855)``
        Pairwise connectivity features in the order produced by
        :mod:`eeg_benchml.features.connectivity`: 171 channel pairs concatenated
        across the 5 frequency bands.
    config : GraphConfig
        Configuration controlling edge pruning and the global descriptors.

    Returns
    -------
    features : ndarray of shape ``(n_epochs, 70)``
        :math:`5 \\text{ bands} \\times (10 \\text{ between-region means} + 4
        \\text{ graph descriptors}) = 70` graph features.
    feature_names : list[str]
        Column names prefixed by ``graph_``, suitable for
        :func:`eeg_benchml.interpretation.regions.aggregate_importance_by_band`.
    """
    channel_pairs = list_channel_pairs()
    n_pairs = len(channel_pairs)
    n_epochs = wpli_features.shape[0]
    expected_pair_columns = n_pairs * len(BAND_ORDER)
    if wpli_features.shape[1] != expected_pair_columns:
        raise ValueError(
            f"Expected {expected_pair_columns} pairwise wPLI features, "
            f"got {wpli_features.shape[1]}."
        )

    between_pairs = _between_region_pairs()
    feature_names: List[str] = []
    blocks: List[np.ndarray] = []

    for band_idx, band in enumerate(BAND_ORDER):
        start = band_idx * n_pairs
        stop = start + n_pairs
        band_wpli = wpli_features[:, start:stop]
        region_matrix = _region_matrix_from_pairs(band_wpli, channel_pairs)

        # Between-region wPLI means (10 features per band).
        between_block = np.stack(
            [region_matrix[:, i, j] for (i, j) in between_pairs], axis=-1
        )
        blocks.append(between_block)
        for (i, j) in between_pairs:
            feature_names.append(
                f"graph_wpli_{band}_{_REGION_ORDER[i]}-{_REGION_ORDER[j]}"
            )

        # Weighted graph descriptors (4 features per band).
        pruned = _prune_weakest_edges(region_matrix, config.edge_keep_proportion)
        descriptors, names = _graph_descriptors(pruned, config.descriptors)
        blocks.append(descriptors)
        feature_names.extend([f"graph_{name}_{band}" for name in names])

    feature_matrix = np.concatenate(blocks, axis=-1)
    if feature_matrix.shape != (n_epochs, 70):
        raise RuntimeError(
            f"Graph feature dimensionality mismatch: produced "
            f"{feature_matrix.shape[1]} columns, expected 70 "
            f"(5 bands x (10 between-region means + 4 graph descriptors))."
        )
    return feature_matrix, feature_names
