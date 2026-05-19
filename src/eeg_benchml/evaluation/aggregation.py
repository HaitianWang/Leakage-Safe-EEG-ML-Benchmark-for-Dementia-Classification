"""Subject-level aggregation of epoch-level predictions.

Equation 3 of the manuscript defines the mean-probability aggregation rule:

.. math::

   \\widehat{y}_i =
   \\operatorname*{arg\\,max}_q
   \\frac{1}{|\\mathcal{E}_i|}
   \\sum_{E_{i,j} \\in \\mathcal{E}_i}
   p_\\theta(y = q \\mid E_{i,j}).

Majority voting is provided as an aggregation ablation.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


def aggregate_subject_predictions(
    proba: np.ndarray,
    groups: Sequence[str],
    classes: Sequence[str],
    rule: str = "mean_probability",
) -> Tuple[Dict[str, str], Dict[str, np.ndarray]]:
    """Aggregate epoch-level probabilities to subject-level predictions.

    Parameters
    ----------
    proba : ndarray of shape ``(n_epochs, n_classes)``
        Calibrated class probabilities returned by the classifier.
    groups : sequence of str
        Subject identifier per epoch.
    classes : sequence of str
        Class labels aligned with the columns of ``proba``.
    rule : str
        ``"mean_probability"`` (Equation 3) or ``"majority_voting"`` ablation.

    Returns
    -------
    predictions : dict[str, str]
        ``subject_id -> predicted_label``.
    mean_probabilities : dict[str, ndarray]
        ``subject_id -> per-class probability vector``.
    """
    proba = np.asarray(proba)
    classes = list(classes)
    groups = np.asarray(groups)

    unique_groups = np.unique(groups)
    predictions: Dict[str, str] = {}
    mean_probs: Dict[str, np.ndarray] = {}

    for sub in unique_groups:
        mask = groups == sub
        subject_proba = proba[mask]
        if rule == "mean_probability":
            mean_vec = subject_proba.mean(axis=0)
            predicted = classes[int(np.argmax(mean_vec))]
        elif rule == "majority_voting":
            votes = np.argmax(subject_proba, axis=1)
            counts = np.bincount(votes, minlength=len(classes))
            predicted = classes[int(np.argmax(counts))]
            mean_vec = subject_proba.mean(axis=0)
        else:
            raise ValueError(
                f"Unknown aggregation rule '{rule}'. "
                "Expected 'mean_probability' or 'majority_voting'."
            )
        predictions[str(sub)] = predicted
        mean_probs[str(sub)] = mean_vec
    return predictions, mean_probs
