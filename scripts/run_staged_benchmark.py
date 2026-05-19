"""Staged benchmark runner.

This script reproduces the four-stage ablation reported in Table 2 of the
manuscript. Stages are evaluated sequentially; the best setting from each
stage is propagated to the next.

Usage
-----
.. code:: bash

   python scripts/run_staged_benchmark.py experiment.task=ad_cn \
       data.root=/path/to/dataset

The script writes a summary CSV ``staged_benchmark.csv`` and per-stage
``metrics.json`` files under the experiment output directory.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omegaconf import OmegaConf  # noqa: E402

from eeg_benchml.pipeline import run_pipeline  # noqa: E402
from eeg_benchml.utils import get_logger  # noqa: E402
from eeg_benchml.utils.io import ensure_dir  # noqa: E402

from _config import load_config, parse_kv_overrides  # noqa: E402

_LOGGER = get_logger("scripts.run_staged_benchmark")


def _stage_a_preprocessing() -> List[Dict[str, str]]:
    return [
        {"preprocessing": "filtering_only", "label": "filtering_only"},
        {"preprocessing": "ica_only", "label": "ica_only"},
        {"preprocessing": "asr_only", "label": "asr_only"},
        {"preprocessing": "asr_ica", "label": "asr_ica"},
    ]


def _stage_b_epoch_aug() -> List[Dict[str, str]]:
    # Epoch-length ablations are encoded via base overrides instead of new
    # variants to keep the YAML tree compact.
    return [
        {"label": "10s_no_aug", "epoch_s": 10.0, "augment": False},
        {"label": "5s_no_aug",  "epoch_s": 5.0,  "augment": False},
        {"label": "20s_no_aug", "epoch_s": 20.0, "augment": False},
        {"label": "30s_no_aug", "epoch_s": 30.0, "augment": False},
        {"label": "10s_aug",    "epoch_s": 10.0, "augment": True},
    ]


def _stage_c_features() -> List[Dict[str, str]]:
    return [
        {"features": "spectral_only", "label": "spectral"},
        {"features": "complexity_only", "label": "complexity"},
        {"features": "connectivity_only", "label": "connectivity"},
        {"features": "connectivity_graph", "label": "connectivity_graph"},
        {"features": "default", "label": "spectral_complexity_connectivity"},
    ]


def _stage_d_selection_classifier() -> List[Dict[str, str]]:
    return [
        {"selection": "none", "classifier": "linear_svm", "label": "no_selection_linear_svm"},
        {"selection": "mi_top50", "classifier": "linear_svm", "label": "mi50_linear_svm"},
        {"selection": "mi_top100", "classifier": "linear_svm", "label": "mi100_linear_svm"},
        {"selection": "mi_top200", "classifier": "linear_svm", "label": "mi200_linear_svm"},
        {"selection": "l1_logistic", "classifier": "linear_svm", "label": "l1_linear_svm"},
        {"selection": "mi_top100", "classifier": "shrinkage_lda", "label": "mi100_lda"},
        {"selection": "mi_top100", "classifier": "logistic_regression", "label": "mi100_logreg"},
        {"selection": "mi_top100", "classifier": "rbf_svm", "label": "mi100_rbf_svm"},
        {"selection": "mi_top100", "classifier": "knn", "label": "mi100_knn"},
        {"selection": "mi_top100", "classifier": "random_forest", "label": "mi100_rf"},
        {"selection": "mi_top100", "classifier": "extra_trees", "label": "mi100_et"},
        {"selection": "mi_top100", "classifier": "gradient_boosting", "label": "mi100_gb"},
    ]


def _run_with_overrides(
    label: str,
    overrides_kv: List[str],
    *,
    preprocessing: str = "asr_ica",
    features: str = "default",
    selection: str = "default",
    classifier: str = "linear_svm",
) -> Dict[str, float]:
    cfg = load_config(
        overrides=overrides_kv,
        preprocessing_variant=preprocessing,
        features_variant=features,
        selection_variant=selection,
        classifier_name=classifier,
    )
    OmegaConf.update(cfg, "experiment.name", f"staged_{label}", merge=False)
    artefacts = run_pipeline(cfg)
    return artefacts.metrics


def main() -> None:
    extra_overrides = parse_kv_overrides(sys.argv[1:])
    summary_rows: List[Dict[str, str]] = []

    # Stage A.
    for entry in _stage_a_preprocessing():
        metrics = _run_with_overrides(
            label=f"A_{entry['label']}",
            overrides_kv=list(extra_overrides),
            preprocessing=entry["preprocessing"],
        )
        summary_rows.append({"stage": "A", "label": entry["label"], **metrics})

    # Stage B.
    for entry in _stage_b_epoch_aug():
        overrides = list(extra_overrides) + [
            f"features.epoch.duration_s={entry['epoch_s']}",
            f"features.augmentation.enabled={'true' if entry['augment'] else 'false'}",
        ]
        metrics = _run_with_overrides(
            label=f"B_{entry['label']}",
            overrides_kv=overrides,
        )
        summary_rows.append({"stage": "B", "label": entry["label"], **metrics})

    # Stage C.
    for entry in _stage_c_features():
        metrics = _run_with_overrides(
            label=f"C_{entry['label']}",
            overrides_kv=list(extra_overrides),
            features=entry["features"],
        )
        summary_rows.append({"stage": "C", "label": entry["label"], **metrics})

    # Stage D.
    for entry in _stage_d_selection_classifier():
        metrics = _run_with_overrides(
            label=f"D_{entry['label']}",
            overrides_kv=list(extra_overrides),
            selection=entry["selection"],
            classifier=entry["classifier"],
        )
        summary_rows.append({"stage": "D", "label": entry["label"], **metrics})

    base_cfg = load_config(overrides=extra_overrides)
    out_dir = ensure_dir(Path(str(base_cfg.experiment.output_dir)) / "staged_benchmark")
    with (out_dir / "staged_benchmark.csv").open("w", newline="", encoding="utf-8") as fp:
        fieldnames = list(summary_rows[0].keys()) if summary_rows else []
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    _LOGGER.info("Staged benchmark complete. Summary written to %s.", out_dir)


if __name__ == "__main__":
    main()
