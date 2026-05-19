"""Static pipeline description.

This script does **not** read EEG data or run any model. Instead, it
introspects the configuration tree and emits a structured human-readable
report that mirrors the manuscript's methodology section. It is intended as
a quick auditing tool for reviewers who want to confirm that the codebase
faithfully implements every component described in the paper.

Usage
-----
.. code:: bash

   python scripts/describe_pipeline.py

   # or, to inspect a specific variant:
   python scripts/describe_pipeline.py --preprocessing asr_only \
       --features connectivity_graph --classifier rbf_svm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omegaconf import DictConfig, OmegaConf  # noqa: E402

from eeg_benchml.constants import (  # noqa: E402
    BANDS,
    BAND_ORDER,
    CHANNELS_10_20,
    LABELS,
    REGIONS,
    TASKS,
    list_channel_pairs,
)

from _config import load_config, parse_kv_overrides  # noqa: E402


def _hline(title: str) -> str:
    return f"\n========== {title} ==========\n"


def _format_dict(d: Dict[str, Any], indent: int = 2) -> str:
    spaces = " " * indent
    lines = []
    for key, value in d.items():
        if isinstance(value, dict):
            lines.append(f"{spaces}{key}:")
            lines.append(_format_dict(value, indent=indent + 2))
        else:
            lines.append(f"{spaces}{key}: {value}")
    return "\n".join(lines)


def _describe_dataset() -> str:
    out = [
        "Reference dataset: OpenNeuro ds004504 (resting-state EEG, eyes-closed).",
        f"Diagnostic classes: {', '.join(LABELS)}.",
        f"EEG montage (10-20): {len(CHANNELS_10_20)} channels -> "
        f"{', '.join(CHANNELS_10_20)}",
        "Region grouping for interpretation:",
    ]
    for region, channels in REGIONS.items():
        out.append(f"  {region:9s}: {', '.join(channels)}")
    out.append(
        f"Number of unordered channel pairs (for connectivity features): "
        f"{len(list_channel_pairs())}"
    )
    return "\n".join(out)


def _describe_bands() -> str:
    lines = ["Frequency bands used by the spectral / connectivity / graph extractors:"]
    for band in BAND_ORDER:
        low, high = BANDS[band]
        lines.append(f"  {band:10s}: [{low:>4.1f}, {high:>4.1f}] Hz")
    return "\n".join(lines)


def _describe_tasks() -> str:
    lines = ["Benchmark tasks:"]
    for key, classes in TASKS.items():
        lines.append(f"  {key:7s}: classes = {classes}")
    return "\n".join(lines)


def _describe_preprocessing(cfg: DictConfig) -> str:
    pc = cfg.preprocessing
    rows: List[Tuple[str, str]] = [
        ("artifact_correction", str(pc.get("artifact_correction"))),
        ("resample_hz", str(pc.get("resample_hz"))),
        ("band-pass", f"{pc.get('l_freq')}--{pc.get('h_freq')} Hz, design={pc.get('filter_design')}"),
        ("notch_freq", str(pc.get("notch_freq"))),
        ("reference", str(pc.get("reference"))),
    ]
    if pc.get("ica") is not None:
        ica = pc["ica"]
        rows.extend([
            ("ica.iclabel_threshold", str(ica.get("iclabel_threshold"))),
            ("ica.artifact_labels", ", ".join(list(ica.get("artifact_labels", [])))),
            ("ica.max_iter", str(ica.get("max_iter"))),
        ])
    if pc.get("asr") is not None:
        asr = pc["asr"]
        rows.extend([
            ("asr.burst_cutoff", str(asr.get("burst_cutoff"))),
            ("asr.flatline_threshold_s", str(asr.get("flatline_threshold_s"))),
            ("asr.channel_correlation_threshold",
             str(asr.get("channel_correlation_threshold"))),
            ("asr.max_bad_window_proportion",
             str(asr.get("max_bad_window_proportion"))),
        ])
    width = max(len(key) for key, _ in rows)
    lines = [f"{key.ljust(width)} : {value}" for key, value in rows]
    return "\n".join(lines)


def _describe_features(cfg: DictConfig) -> str:
    families = OmegaConf.to_container(cfg.features.families, resolve=True)
    epoch = cfg.features.get("epoch", {})
    aug = cfg.features.get("augmentation", {})

    family_dim = {
        "spectral": 266,        # 14 descriptors x 19 channels
        "complexity": 475,      # 25 descriptors x 19 channels
        "connectivity": 855,    # 171 pairs x 5 bands
        "graph": 70,            # 5 bands x (10 between-region means + 4 graph descriptors)
    }
    enabled = [name for name, on in families.items() if on]
    total_dim = sum(family_dim[name] for name in enabled if name in family_dim)
    lines = [
        f"Enabled families: {', '.join(enabled) if enabled else '<none>'}",
        f"Per-family raw dimensionality (Table 2 'Dim.' column):",
    ]
    for name in ("spectral", "complexity", "connectivity", "graph"):
        flag = "ON " if families.get(name) else "off"
        lines.append(f"  [{flag}] {name:13s} -> {family_dim[name]:5d}")
    lines.append(f"Combined raw dimensionality: {total_dim}")
    lines.append("")
    lines.append("Epoching:")
    lines.append(_format_dict(dict(OmegaConf.to_container(epoch, resolve=True))))
    lines.append("")
    lines.append("Training-only augmentation:")
    lines.append(_format_dict(dict(OmegaConf.to_container(aug, resolve=True))))
    return "\n".join(lines)


def _describe_selection(cfg: DictConfig) -> str:
    sc = cfg.selection
    return _format_dict(dict(OmegaConf.to_container(sc, resolve=True)))


def _describe_classifier(cfg: DictConfig) -> str:
    classifier = cfg.classifier
    return _format_dict(dict(OmegaConf.to_container(classifier, resolve=True)))


def _describe_evaluation(cfg: DictConfig) -> str:
    return _format_dict(
        dict(OmegaConf.to_container(cfg.evaluation, resolve=True))
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a static description of the configured pipeline."
    )
    parser.add_argument("--preprocessing", default="default")
    parser.add_argument("--features", default="default")
    parser.add_argument("--selection", default="default")
    parser.add_argument("--classifier", default="linear_svm")
    args, extra = parser.parse_known_args()
    overrides = parse_kv_overrides(extra)
    cfg = load_config(
        overrides=overrides,
        preprocessing_variant=args.preprocessing,
        features_variant=args.features,
        selection_variant=args.selection,
        classifier_name=args.classifier,
    )

    print(_hline("EEG-BenchML pipeline description"))
    print(_describe_dataset())
    print(_describe_bands())
    print()
    print(_describe_tasks())

    print(_hline("Stage A: Preprocessing"))
    print(_describe_preprocessing(cfg))

    print(_hline("Stage B/C: Epoching, augmentation, feature engineering"))
    print(_describe_features(cfg))

    print(_hline("Stage D: Feature selection"))
    print(_describe_selection(cfg))

    print(_hline("Stage D: Classifier"))
    print(_describe_classifier(cfg))

    print(_hline("Evaluation protocol"))
    print(_describe_evaluation(cfg))

    print(_hline("Done"))


if __name__ == "__main__":
    main()
