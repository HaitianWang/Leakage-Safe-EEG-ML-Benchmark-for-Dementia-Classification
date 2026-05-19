"""Classical counterpart baselines for the Table 4 fair LOSO comparison.

Section 5.3 of the manuscript reports a fair baseline comparison in which
classical counterparts inspired by prior EEG dementia studies are evaluated
under the **same** leakage-safe LOSO protocol used by the proposed pipeline.

This package exposes :data:`BASELINE_REGISTRY`, a mapping of baseline name to
:class:`BaselineRecipe`. Each recipe captures the preprocessing, feature,
selection, and classifier choices for the corresponding baseline. Recipes are
purely declarative -- they can be expanded back into a full
:class:`omegaconf.DictConfig` and handed to
:func:`eeg_benchml.pipeline.run_pipeline` without any extra glue code.
"""

from .recipes import BASELINE_REGISTRY, BaselineRecipe, expand_recipe

__all__ = ["BASELINE_REGISTRY", "BaselineRecipe", "expand_recipe"]
