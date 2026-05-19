"""Baseline recipe declarations.

The recipes below describe the four classical counterparts evaluated in
Table 4 of the manuscript. They are inspired by prior EEG dementia studies
and intentionally use the same leakage-safe LOSO protocol, subject-level
aggregation rule, and training-fold feature-selection procedure as the
proposed pipeline. The recipes do not attempt to reproduce the original
publications -- they only fix the feature family, classifier, and target
task that characterise each baseline.

Each :class:`BaselineRecipe` captures:

* The classification task (``ad_cn``, ``ftd_cn``, ``ad_ftd``, or ``three``).
* The active feature families (subset of
  ``{spectral, complexity, connectivity, graph}``).
* The classifier name (must be registered in
  :func:`eeg_benchml.models.classifiers.available_classifiers`).
* Optional overrides applied on top of the defaults in ``configs/``.

The helper :func:`expand_recipe` turns a recipe into a plain dictionary of
``(key, value)`` overrides that can be applied to a base configuration via
``OmegaConf.from_dotlist``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class BaselineRecipe:
    """Declarative description of a single Table 4 baseline."""

    name: str
    task: str
    families: Tuple[str, ...]
    classifier: str
    description: str
    extra_overrides: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)


def _families_overrides(families: Tuple[str, ...]) -> List[Tuple[str, str]]:
    """Build OmegaConf-compatible overrides for ``features.families.*``."""
    overrides: List[Tuple[str, str]] = []
    for family in ("spectral", "complexity", "connectivity", "graph"):
        overrides.append(
            (f"features.families.{family}", "true" if family in families else "false")
        )
    return overrides


def expand_recipe(recipe: BaselineRecipe) -> List[str]:
    """Translate a :class:`BaselineRecipe` into ``key=value`` overrides.

    The returned list is ready to be passed to
    :func:`scripts._config.load_config` via the ``overrides`` argument.

    Examples
    --------
    >>> overrides = expand_recipe(BASELINE_REGISTRY["band_power_coherence_svm"])
    >>> "experiment.task=ad_cn" in overrides
    True
    """
    overrides = [
        f"experiment.task={recipe.task}",
        f"experiment.name=baseline_{recipe.name}",
    ]
    for key, value in _families_overrides(recipe.families):
        overrides.append(f"{key}={value}")
    overrides.extend([f"{k}={v}" for k, v in recipe.extra_overrides])
    return overrides


# ---------------------------------------------------------------------------
# Table 4 baselines.
# ---------------------------------------------------------------------------
BASELINE_REGISTRY: Dict[str, BaselineRecipe] = {
    "band_power_coherence_svm": BaselineRecipe(
        name="band_power_coherence_svm",
        task="ad_cn",
        families=("spectral", "connectivity"),
        classifier="linear_svm",
        description=(
            "Spectral band power combined with phase-based connectivity, "
            "classified by a linear SVM. Matches the band-power + coherence "
            "baseline reported in Table 4 (AD vs CN)."
        ),
    ),
    "spectrum_complexity_synchronisation_rf": BaselineRecipe(
        name="spectrum_complexity_synchronisation_rf",
        task="ad_cn",
        families=("spectral", "complexity", "connectivity"),
        classifier="random_forest",
        description=(
            "Spectral, complexity, and synchronisation descriptors classified "
            "by a random forest. Matches the multi-domain random-forest "
            "baseline reported in Table 4 (AD vs CN)."
        ),
    ),
    "rhythm_ratio_svm": BaselineRecipe(
        name="rhythm_ratio_svm",
        task="ad_ftd",
        families=("spectral",),
        classifier="linear_svm",
        description=(
            "Rhythm-ratio discriminative spectral features classified by a "
            "linear SVM. Matches the AD vs FTD rhythm-ratio baseline."
        ),
    ),
    "electrode_pair_communication_rf": BaselineRecipe(
        name="electrode_pair_communication_rf",
        task="ad_ftd",
        families=("connectivity",),
        classifier="random_forest",
        description=(
            "Electrode-pair communication (connectivity) features classified "
            "by a random forest. Matches the AD vs FTD electrode-pair "
            "baseline."
        ),
    ),
}
