"""Single-configuration runner.

Usage
-----
.. code:: bash

   python scripts/run_experiment.py experiment.task=ad_cn \
       data.root=/path/to/dataset

Any setting in ``configs/base.yaml`` (and the composed variant files) can be
overridden via ``key=value`` arguments on the command line. The script writes
``metrics.json`` to ``<output_dir>/<experiment.name>/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add the ``src`` directory to ``sys.path`` so the script can be run without
# installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_benchml.pipeline import run_pipeline  # noqa: E402
from eeg_benchml.utils import get_logger  # noqa: E402

from _config import load_config, parse_kv_overrides  # noqa: E402

_LOGGER = get_logger("scripts.run_experiment")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single EEG-BenchML experiment configuration."
    )
    parser.add_argument(
        "--preprocessing", default="default",
        help="Preprocessing variant key (see configs/preprocessing.yaml).",
    )
    parser.add_argument(
        "--features", default="default",
        help="Feature variant key (see configs/features.yaml).",
    )
    parser.add_argument(
        "--selection", default="default",
        help="Selection variant key (see configs/selection.yaml).",
    )
    parser.add_argument(
        "--classifier", default="linear_svm",
        help="Classifier variant key (see configs/classifiers.yaml).",
    )
    args, extra = parser.parse_known_args()

    overrides = parse_kv_overrides(extra)
    cfg = load_config(
        overrides=overrides,
        preprocessing_variant=args.preprocessing,
        features_variant=args.features,
        selection_variant=args.selection,
        classifier_name=args.classifier,
    )
    _LOGGER.info("Running configuration: %s", cfg.experiment.name)
    artefacts = run_pipeline(cfg)
    _LOGGER.info("Subject-level metrics: %s", artefacts.metrics)


if __name__ == "__main__":
    main()
