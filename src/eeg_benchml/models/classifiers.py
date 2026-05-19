"""Classical ML classifier registry.

Each entry returns an sklearn-compatible estimator with the hyper-parameter
grid described in Section 4.5 of the manuscript. The grids are kept compact
to make the inner GroupKFold tuning tractable on a single workstation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

CLASSIFIER_NAMES = (
    "shrinkage_lda",
    "logistic_regression",
    "linear_svm",
    "rbf_svm",
    "knn",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
)


@dataclass
class ClassifierConfig:
    """Configuration container for a single classifier and its grid."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    random_state: int = 42


def available_classifiers() -> Iterable[str]:
    """Return the registry of classifier names."""
    return CLASSIFIER_NAMES


def build_classifier(config: ClassifierConfig):
    """Instantiate an sklearn classifier from a :class:`ClassifierConfig`.

    The returned estimator exposes a ``param_grid`` attribute that the LOSO
    orchestrator hands to ``GridSearchCV`` together with an inner GroupKFold
    split.
    """
    name = config.name
    params = config.params
    rs = config.random_state

    if name == "shrinkage_lda":
        estimator = LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage=params.get("shrinkage", "auto")
        )
        estimator.param_grid = {}
    elif name == "logistic_regression":
        estimator = LogisticRegression(
            penalty=params.get("penalty", "l2"),
            max_iter=int(params.get("max_iter", 5000)),
            class_weight=params.get("class_weight", "balanced"),
            random_state=rs,
        )
        estimator.param_grid = {"C": list(params.get("C_grid", (0.01, 0.1, 1.0, 10.0)))}
    elif name == "linear_svm":
        estimator = SVC(
            kernel="linear",
            class_weight=params.get("class_weight", "balanced"),
            probability=False,
            random_state=rs,
        )
        estimator.param_grid = {"C": list(params.get("C_grid", (0.01, 0.1, 1.0, 10.0)))}
    elif name == "rbf_svm":
        estimator = SVC(
            kernel="rbf",
            class_weight=params.get("class_weight", "balanced"),
            probability=False,
            random_state=rs,
        )
        estimator.param_grid = {
            "C": list(params.get("C_grid", (0.1, 1.0, 10.0, 100.0))),
            "gamma": list(params.get("gamma_grid", ("scale", 0.001, 0.01, 0.1))),
        }
    elif name == "knn":
        estimator = KNeighborsClassifier(weights=params.get("weights", "distance"))
        estimator.param_grid = {
            "n_neighbors": list(params.get("k_grid", (3, 5, 7, 11)))
        }
    elif name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 500)),
            max_features=params.get("max_features", "sqrt"),
            class_weight=params.get("class_weight", "balanced"),
            random_state=rs,
        )
        estimator.param_grid = {
            "min_samples_leaf": list(params.get("min_samples_leaf_grid", (1, 2, 4)))
        }
    elif name == "extra_trees":
        estimator = ExtraTreesClassifier(
            n_estimators=int(params.get("n_estimators", 500)),
            max_features=params.get("max_features", "sqrt"),
            class_weight=params.get("class_weight", "balanced"),
            random_state=rs,
        )
        estimator.param_grid = {
            "min_samples_leaf": list(params.get("min_samples_leaf_grid", (1, 2, 4)))
        }
    elif name == "gradient_boosting":
        estimator = GradientBoostingClassifier(random_state=rs)
        estimator.param_grid = {
            "n_estimators": list(params.get("n_estimators_grid", (100, 300))),
            "learning_rate": list(params.get("learning_rate_grid", (0.05, 0.1))),
            "max_depth": list(params.get("max_depth_grid", (2, 3))),
        }
    else:
        raise ValueError(
            f"Unknown classifier '{name}'. Expected one of {CLASSIFIER_NAMES}."
        )

    estimator.classifier_name = name
    return estimator
