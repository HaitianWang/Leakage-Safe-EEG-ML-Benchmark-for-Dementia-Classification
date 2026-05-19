"""Feature extractor shape tests using synthetic random EEG.

The tests do not require the real dataset. They generate a small batch of
synthetic 10 s epochs and verify that the feature dimensions match the
manuscript's reporting (Table 2 ``Dim.`` column).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_benchml.features import (  # noqa: E402
    FeatureExtractor,
    FeatureExtractorConfig,
)


def _make_synthetic(n_epochs: int = 4, n_times: int = 2500) -> np.ndarray:
    """Return a deterministic ``(n_epochs, 19, n_times)`` synthetic array."""
    rng = np.random.default_rng(0)
    return rng.standard_normal(size=(n_epochs, 19, n_times)).astype(float)


def test_full_feature_vector_has_1596_components() -> None:
    """The combined feature vector for 19 channels has 1596 columns."""
    config = FeatureExtractorConfig(
        families={
            "spectral": True,
            "complexity": True,
            "connectivity": True,
            "graph": False,
        }
    )
    config.complexity.multi_scale_entropy_scales = (1, 2, 3, 4, 5)
    extractor = FeatureExtractor(config)
    data = _make_synthetic()
    bundle = extractor.transform_array(data=data, sfreq=250.0)
    assert bundle.features.shape == (data.shape[0], 1596)
    assert len(bundle.feature_names) == 1596
    assert set(bundle.family_indices.keys()) == {"spectral", "complexity", "connectivity"}


def test_spectral_only_has_266_components() -> None:
    config = FeatureExtractorConfig(
        families={
            "spectral": True,
            "complexity": False,
            "connectivity": False,
            "graph": False,
        }
    )
    extractor = FeatureExtractor(config)
    data = _make_synthetic()
    bundle = extractor.transform_array(data=data, sfreq=250.0)
    assert bundle.features.shape == (data.shape[0], 266)


def test_complexity_only_has_475_components() -> None:
    config = FeatureExtractorConfig(
        families={
            "spectral": False,
            "complexity": True,
            "connectivity": False,
            "graph": False,
        }
    )
    extractor = FeatureExtractor(config)
    data = _make_synthetic()
    bundle = extractor.transform_array(data=data, sfreq=250.0)
    assert bundle.features.shape == (data.shape[0], 475)


def test_connectivity_only_has_855_components() -> None:
    config = FeatureExtractorConfig(
        families={
            "spectral": False,
            "complexity": False,
            "connectivity": True,
            "graph": False,
        }
    )
    extractor = FeatureExtractor(config)
    data = _make_synthetic()
    bundle = extractor.transform_array(data=data, sfreq=250.0)
    assert bundle.features.shape == (data.shape[0], 855)
