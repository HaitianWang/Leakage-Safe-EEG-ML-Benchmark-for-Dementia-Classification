"""ICA-based artifact removal with ICLabel scoring.

This module implements the ICA-only variant of the artifact-correction stage
and is also reused as the second step of the ASR + ICA pipeline. Components
labelled as eye, muscle, heart, line-noise, or channel-noise with probability
greater than the configured threshold are removed before reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

import mne
import numpy as np

from ..utils import get_logger

_LOGGER = get_logger(__name__)

# ICLabel is an optional dependency. We import it lazily so that the rest of
# the package can still be imported in environments without it.
try:  # pragma: no cover - import side effect only
    from mne_icalabel import label_components

    _HAS_ICLABEL = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_ICLABEL = False


@dataclass
class ICAConfig:
    """Configuration for the ICA artifact-removal step."""

    n_components: Optional[int] = None
    max_iter: int = 1000
    random_state: int = 42
    iclabel_threshold: float = 0.80
    artifact_labels: Sequence[str] = field(
        default_factory=lambda: (
            "eye",
            "muscle",
            "heart",
            "line_noise",
            "channel_noise",
        )
    )


def _data_rank(raw: mne.io.BaseRaw) -> int:
    """Return the numerical rank of the EEG matrix.

    ICA components beyond the data rank are unstable. The reference pipeline
    therefore sets ``n_components`` equal to this rank when no explicit value
    is provided.
    """
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    data = raw.get_data(picks=picks)
    if data.size == 0:
        return 1
    try:
        return int(np.linalg.matrix_rank(data @ data.T))
    except Exception:
        return max(1, len(picks) - 1)


def fit_apply_ica(raw: mne.io.BaseRaw, config: ICAConfig) -> mne.io.BaseRaw:
    """Fit an ICA decomposition and remove ICLabel-flagged artefact components.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Filtered raw recording. The signal is expected to have been
        band-pass filtered before this function is called.
    config : ICAConfig
        Tunable parameters of the ICA step.

    Returns
    -------
    raw : mne.io.BaseRaw
        Recording reconstructed without the artefact components. If ICLabel
        is unavailable, the recording is returned unchanged with a warning.
    """
    if not _HAS_ICLABEL:
        _LOGGER.warning(
            "mne-icalabel is not installed. ICA components cannot be auto-"
            "labelled and the raw signal is returned unchanged."
        )
        return raw

    n_components = config.n_components if config.n_components else _data_rank(raw)
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method="infomax",
        fit_params={"extended": True},
        max_iter=config.max_iter,
        random_state=config.random_state,
        verbose=False,
    )
    ica.fit(raw, verbose=False)

    labels = label_components(raw, ica, method="iclabel")
    probabilities = np.asarray(labels["y_pred_proba"])
    component_labels: List[str] = list(labels["labels"])
    artifact_set = {label.lower() for label in config.artifact_labels}

    to_exclude = [
        comp_idx
        for comp_idx, (lab, prob) in enumerate(zip(component_labels, probabilities))
        if lab.lower() in artifact_set and float(prob) >= config.iclabel_threshold
    ]
    ica.exclude = to_exclude
    _LOGGER.info(
        "ICA removed %d / %d components (threshold=%.2f).",
        len(to_exclude), n_components, config.iclabel_threshold,
    )
    return ica.apply(raw, verbose=False)


def list_artifact_components(component_labels: Iterable[str]) -> List[int]:
    """Return indices of components matching any of the artefact label tokens.

    Convenience function exposed for unit tests; not used in the main pipeline.
    """
    artifact_set = {"eye", "muscle", "heart", "line_noise", "channel_noise"}
    return [i for i, lab in enumerate(component_labels) if lab.lower() in artifact_set]
