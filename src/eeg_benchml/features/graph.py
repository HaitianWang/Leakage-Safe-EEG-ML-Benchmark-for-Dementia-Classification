"""Graph-derived features from the region-level connectivity matrix.

Following Section 4.3.3 of the manuscript, pairwise wPLI features are first
aggregated into anatomical regions (frontal, temporal, central, parietal,
occipital). The within- and between-region wPLI means form a 5x5 connectivity
matrix per band. We then prune the weakest 70 % of edges and compute four
compact graph descriptors on the remaining weighted graph:

* mean node strength,
* weighted clustering coefficient,
* global efficiency,
* characteristic path length.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Tuple

import networkx as nx
import numpy as np

from ..constants import BAND_ORDER, CHANNEL_TO_REGION, REGIONS, list_channel_pairs


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


def _region_matrix_from_pairs(
    wpli_per_band: np.ndarray, channel_pairs: List[Tuple[str, str]]
) -> np.ndarray:
    """Aggregate channel-pair wPLI into a region-level matrix.

    Parameters
    ----------
    wpli_per_band : ndarray of shape ``(n_epochs, n_pairs)``
        wPLI values for one frequency band.
    channel_pairs : list[tuple[str, str]]
        Channel pairs corresponding to the second dimension of ``wpli_per_band``.

    Returns
    -------
    matrix : ndarray of shape ``(n_epochs, n_regions, n_regions)``
        Symmetric region-level connectivity matrix per epoch.
    """
    region_order = list(REGIONS.keys())
    region_to_idx = {name: idx for idx, name in enumerate(region_order)}
    n_regions = len(region_order)
    n_epochs, n_pairs = wpli_per_band.shape
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
    matrix = accum / counts
    return matrix


def _prune_weakest_edges(matrix: np.ndarray, keep_proportion: float) -> np.ndarray:
    """Zero-out the weakest edges so only the top ``keep_proportion`` remain."""
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
    """Compute a fixed set of graph descriptors per epoch."""
    n_epochs, n_regions, _ = matrix.shape
    region_order = list(REGIONS.keys())
    descriptor_list = list(descriptors)
    out = np.zeros((n_epochs, len(descriptor_list)), dtype=float)
    names = list(descriptor_list)
    for i in range(n_epochs):
        graph = nx.from_numpy_array(matrix[i])
        for j, descriptor in enumerate(descriptor_list):
            if descriptor == "mean_strength":
                strengths = dict(graph.degree(weight="weight"))
                out[i, j] = float(np.mean(list(strengths.values())))
            elif descriptor == "clustering":
                clustering = nx.clustering(graph, weight="weight")
                out[i, j] = float(np.mean(list(clustering.values())))
            elif descriptor == "global_efficiency":
                # Use inverse weights so that strong connectivity becomes a
                # short distance, matching the standard definition.
                inv_graph = graph.copy()
                for u, v, attrs in inv_graph.edges(data=True):
                    weight = attrs.get("weight", 1.0)
                    attrs["weight"] = 1.0 / (weight + 1e-9)
                try:
                    out[i, j] = float(nx.global_efficiency(inv_graph))
                except Exception:
                    out[i, j] = 0.0
            elif descriptor == "char_path_length":
                inv_graph = graph.copy()
                for u, v, attrs in inv_graph.edges(data=True):
                    weight = attrs.get("weight", 1.0)
                    attrs["weight"] = 1.0 / (weight + 1e-9)
                try:
                    out[i, j] = float(
                        nx.average_shortest_path_length(inv_graph, weight="weight")
                    )
                except Exception:
                    out[i, j] = 0.0
            else:
                raise ValueError(f"Unknown graph descriptor '{descriptor}'.")
    # Unused but kept for downstream interpretation if needed.
    del region_order
    return out, names


def compute_graph_features(
    wpli_features: np.ndarray, config: GraphConfig
) -> Tuple[np.ndarray, List[str]]:
    """Compute compact graph descriptors per band from pairwise wPLI features.

    Parameters
    ----------
    wpli_features : ndarray of shape ``(n_epochs, 855)``
        Pairwise connectivity features, ordered band-by-band.
    config : GraphConfig

    Returns
    -------
    features : ndarray of shape ``(n_epochs, n_descriptors * n_bands)``
    feature_names : list[str]
    """
    channel_pairs = list_channel_pairs()
    n_pairs = len(channel_pairs)
    n_epochs = wpli_features.shape[0]
    expected = n_pairs * len(BAND_ORDER)
    if wpli_features.shape[1] != expected:
        raise ValueError(
            f"Expected {expected} pairwise wPLI features, got {wpli_features.shape[1]}."
        )
    blocks: List[np.ndarray] = []
    feature_names: List[str] = []
    for band_idx, band in enumerate(BAND_ORDER):
        start = band_idx * n_pairs
        stop = start + n_pairs
        band_wpli = wpli_features[:, start:stop]
        region_matrix = _region_matrix_from_pairs(band_wpli, channel_pairs)
        pruned = _prune_weakest_edges(region_matrix, config.edge_keep_proportion)
        descriptors, names = _graph_descriptors(pruned, config.descriptors)
        blocks.append(descriptors)
        feature_names.extend([f"graph_{name}_{band}" for name in names])
    feature_matrix = np.concatenate(blocks, axis=-1)
    assert feature_matrix.shape[0] == n_epochs
    return feature_matrix, feature_names
