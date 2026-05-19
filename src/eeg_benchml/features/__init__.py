"""Feature engineering: spectral, complexity, connectivity, and graph features.

The feature extractor is the central component of the manuscript's
methodology. It produces, for each epoch, a tabular feature vector that
combines:

* 14 spectral descriptors per channel (band powers, ratios, spectral entropy,
  alpha peak frequency).
* 25 complexity descriptors per channel (Hjorth parameters, sample entropy,
  multi-scale entropy, Higuchi fractal dimension).
* 855 pairwise connectivity descriptors (weighted phase lag index per band).
* Optional graph descriptors (mean strength, clustering, global efficiency,
  characteristic path length) on the region-level connectivity matrix.

With all families enabled and the 19-channel 10--20 montage, the raw feature
vector has 1596 components, matching the ``Dim.`` column of Table 2.
"""

from .extractor import (
    FeatureBundle,
    FeatureExtractor,
    FeatureExtractorConfig,
    feature_family_of,
)

__all__ = [
    "FeatureBundle",
    "FeatureExtractor",
    "FeatureExtractorConfig",
    "feature_family_of",
]
