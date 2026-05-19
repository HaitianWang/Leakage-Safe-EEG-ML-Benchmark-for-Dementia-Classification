"""Run the Table 4 fair-LOSO baselines.

Each baseline registered in :data:`eeg_benchml.baselines.BASELINE_REGISTRY` is
expanded into a configuration override list and executed through the same
:func:`eeg_benchml.pipeline.run_pipeline` orchestrator used by the proposed
pipeline. This guarantees that the baselines and the proposed pipeline share
the same leakage-safe LOSO protocol, subject-level aggregation rule, and
training-fold feature-selection procedure.

Usage
-----
.. code:: bash

   python scripts/run_baselines.py data.root=/path/to/dataset

The script writes ``baselines_summary.csv`` to the experiment output
directory.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_benchml.baselines import BASELINE_REGISTRY, expand_recipe  # noqa: E402
from eeg_benchml.pipeline import run_pipeline  # noqa: E402
from eeg_benchml.utils import get_logger  # noqa: E402
from eeg_benchml.utils.io import ensure_dir  # noqa: E402

from _config import load_config, parse_kv_overrides  # noqa: E402

_LOGGER = get_logger("scripts.run_baselines")


def main() -> None:
    extra_overrides = list(parse_kv_overrides(sys.argv[1:]))
    summary_rows = []

    for name, recipe in BASELINE_REGISTRY.items():
        recipe_overrides = expand_recipe(recipe)
        full_overrides = recipe_overrides + extra_overrides
        cfg = load_config(
            overrides=full_overrides,
            classifier_name=recipe.classifier,
        )
        _LOGGER.info(
            "Running baseline '%s' (%s) on task=%s with families=%s",
            name, recipe.description, recipe.task, recipe.families,
        )
        artefacts = run_pipeline(cfg)
        row = {
            "baseline": name,
            "task": recipe.task,
            "families": "+".join(recipe.families),
            "classifier": recipe.classifier,
            **artefacts.metrics,
        }
        summary_rows.append(row)

    base_cfg = load_config(overrides=extra_overrides)
    out_dir = ensure_dir(
        Path(str(base_cfg.experiment.output_dir)) / "baselines"
    )
    out_path = out_dir / "baselines_summary.csv"
    fieldnames = list(summary_rows[0].keys()) if summary_rows else []
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    _LOGGER.info("Baseline benchmark complete. Summary written to %s.", out_path)


if __name__ == "__main__":
    main()
