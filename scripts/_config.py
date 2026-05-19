"""Configuration loading helpers shared by the CLI scripts.

The scripts use OmegaConf for configuration management because it is
lightweight and supports both file-based composition and command-line overrides
in a way that matches the project's research workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

from omegaconf import DictConfig, OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

_VARIANT_KEYS = {
    "preprocessing": "preprocessing.yaml",
    "features": "features.yaml",
    "selection": "selection.yaml",
    "classifier": "classifiers.yaml",
}


def load_config(
    overrides: Optional[Sequence[str]] = None,
    *,
    preprocessing_variant: str = "default",
    features_variant: str = "default",
    selection_variant: str = "default",
    classifier_name: str = "linear_svm",
) -> DictConfig:
    """Compose the four configuration groups into one :class:`DictConfig`.

    Parameters
    ----------
    overrides : sequence of str, optional
        Dot-notated overrides such as ``["experiment.seed=0"]``.
    preprocessing_variant, features_variant, selection_variant, classifier_name : str
        Variant keys inside the corresponding YAML file.
    """
    base = OmegaConf.load(CONFIG_DIR / "base.yaml")
    base = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
    base["preprocessing"] = _load_variant("preprocessing", preprocessing_variant)
    base["features"] = _load_variant("features", features_variant)
    base["selection"] = _load_variant("selection", selection_variant)
    base["classifier"] = _load_variant("classifier", classifier_name)
    base = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
    if overrides:
        base = OmegaConf.merge(base, OmegaConf.from_dotlist(list(overrides)))
    return base  # type: ignore[return-value]


def _load_variant(group: str, variant: str) -> DictConfig:
    path = CONFIG_DIR / _VARIANT_KEYS[group]
    if not path.exists():
        raise FileNotFoundError(f"Missing configuration group file: {path}")
    full = OmegaConf.load(path)
    if variant not in full:
        raise KeyError(
            f"Variant '{variant}' not found in {path}. "
            f"Available variants: {list(full.keys())}"
        )
    return OmegaConf.create(OmegaConf.to_container(full[variant], resolve=True))


def parse_kv_overrides(arguments: Iterable[str]) -> Sequence[str]:
    """Collect ``key=value`` style overrides from command-line arguments."""
    overrides = []
    for arg in arguments:
        if "=" in arg:
            overrides.append(arg)
    return overrides
