"""Composite feature extractor.

The :class:`FeatureExtractor` produces, for each epoch, a single feature
vector by concatenating the family-specific descriptors selected via
:class:`FeatureExtractorConfig`. The output column ordering is stable and
matches the manuscript's reporting order:

* ``spectral`` (14 descriptors per channel x 19 channels = 266 features)
* ``complexity`` (25 descriptors per channel x 19 channels = 475 features)
* ``connectivity`` (171 channel pairs x 5 bands = 855 features)
* ``graph`` (optional, 4 descriptors x 5 bands = 20 features)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import mne
import numpy as np

from ..constants import FEATURE_FAMILIES
from ..utils import get_logger
from .complexity import ComplexityConfig, compute_complexity_features
from .connectivity import ConnectivityConfig, compute_connectivity_features
from .graph import GraphConfig, compute_graph_features
from .spectral import SpectralConfig, compute_spectral_features

_LOGGER = get_logger(__name__)


@dataclass
class FeatureExtractorConfig:
    """Top-level configuration for the composite feature extractor."""

    families: Dict[str, bool] = field(
        default_factory=lambda: {
            "spectral": True,
            "complexity": True,
            "connectivity": True,
            "graph": False,
        }
    )
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    complexity: ComplexityConfig = field(default_factory=ComplexityConfig)
    connectivity: ConnectivityConfig = field(default_factory=ConnectivityConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)


@dataclass
class FeatureBundle:
    """Container returned by :meth:`FeatureExtractor.transform`."""

    features: np.ndarray
    feature_names: List[str]
    family_indices: Dict[str, slice]


class FeatureExtractor:
    """Compose family-specific feature extractors into a single transform."""

    def __init__(self, config: FeatureExtractorConfig) -> None:
        self.config = config

    def transform(self, epochs: mne.Epochs) -> FeatureBundle:
        """Compute the feature matrix for ``epochs``.

        Parameters
        ----------
        epochs : mne.Epochs
            Preprocessed and rejected epochs for a single subject. The
            extractor consumes the underlying NumPy array and is therefore
            agnostic to MNE-specific metadata.
        """
        data = epochs.get_data(picks="eeg")
        sfreq = float(epochs.info["sfreq"])
        return self.transform_array(data=data, sfreq=sfreq)

    def transform_array(self, data: np.ndarray, sfreq: float) -> FeatureBundle:
        """Compute the feature matrix from a raw ``(epochs, channels, time)`` array.

        This entry point is used by the augmentation step, which already
        operates on plain NumPy arrays.
        """
        n_epochs = data.shape[0]
        blocks: List[np.ndarray] = []
        all_names: List[str] = []
        family_indices: Dict[str, slice] = {}
        cursor = 0

        if self.config.families.get("spectral", False):
            feats, names = compute_spectral_features(
                data=data, sfreq=sfreq, config=self.config.spectral
            )
            family_indices["spectral"] = slice(cursor, cursor + feats.shape[1])
            cursor += feats.shape[1]
            blocks.append(feats)
            all_names.extend(names)

        if self.config.families.get("complexity", False):
            feats, names = compute_complexity_features(
                data=data, config=self.config.complexity, sfreq=sfreq
            )
            family_indices["complexity"] = slice(cursor, cursor + feats.shape[1])
            cursor += feats.shape[1]
            blocks.append(feats)
            all_names.extend(names)

        wpli_block: np.ndarray = np.empty((n_epochs, 0))
        wpli_names: List[str] = []
        if self.config.families.get("connectivity", False) or self.config.families.get("graph", False):
            wpli_block, wpli_names = compute_connectivity_features(
                data=data, sfreq=sfreq, config=self.config.connectivity
            )
        if self.config.families.get("connectivity", False):
            family_indices["connectivity"] = slice(cursor, cursor + wpli_block.shape[1])
            cursor += wpli_block.shape[1]
            blocks.append(wpli_block)
            all_names.extend(wpli_names)

        if self.config.families.get("graph", False):
            if wpli_block.size == 0:
                # If the user requested graph features without enabling the
                # raw connectivity family, we still need the wPLI tensor to
                # derive the region-level matrix. We compute it here without
                # adding the 855 pairwise descriptors to the output.
                wpli_block, _ = compute_connectivity_features(
                    data=data, sfreq=sfreq, config=self.config.connectivity
                )
            graph_feats, graph_names = compute_graph_features(
                wpli_features=wpli_block, config=self.config.graph
            )
            family_indices["graph"] = slice(cursor, cursor + graph_feats.shape[1])
            cursor += graph_feats.shape[1]
            blocks.append(graph_feats)
            all_names.extend(graph_names)

        if not blocks:
            raise ValueError(
                "FeatureExtractor was configured without any feature family enabled."
            )

        features = np.concatenate(blocks, axis=-1)
        _LOGGER.debug(
            "Extracted feature matrix with %d epochs x %d features.",
            features.shape[0], features.shape[1],
        )
        return FeatureBundle(
            features=features, feature_names=all_names, family_indices=family_indices
        )


def feature_family_of(feature_name: str) -> str:
    """Return the family identifier (``spectral``/``complexity``/...) of a feature.

    The function only looks at the ``feature_name`` prefix, which is set by the
    family-specific extractor: ``spec_``, ``cmpl_``, ``conn_``, or ``graph_``.
    Unknown prefixes raise :class:`ValueError`.
    """
    for prefix, family in (
        ("spec_", "spectral"),
        ("cmpl_", "complexity"),
        ("conn_", "connectivity"),
        ("graph_", "graph"),
    ):
        if feature_name.startswith(prefix):
            return family
    raise ValueError(
        f"Unknown feature family for '{feature_name}'. "
        f"Expected one of {FEATURE_FAMILIES}."
    )
