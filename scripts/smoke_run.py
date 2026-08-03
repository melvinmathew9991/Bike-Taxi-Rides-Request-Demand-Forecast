#!/usr/bin/env python
"""
Manual smoke run of the full pipeline against real data.

This is a script, not a test. It previously lived at the repository root as
`test_pipeline.py`, where pytest collected it by name and executed a complete
training run - including `sys.exit()` - during test collection. The automated
suite lives in `tests/` and never touches real data or fits a model on it.

Usage:
    python scripts/smoke_run.py [--raw-data PATH] [--test-data PATH]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ML_Pipeline.config import PipelineConfig  # noqa: E402
from ML_Pipeline.pipeline import MLPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data", default="data/raw_data.csv")
    parser.add_argument(
        "--test-data", default="data/test_dataset/cleaned_test_booking_data.csv"
    )
    parser.add_argument("--output", default="output")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    log = logging.getLogger("smoke_run")

    if not Path(args.raw_data).exists():
        log.error(
            "Raw data not found at %s. This script needs the real dataset, which "
            "is git-ignored because it carries personal data. "
            "See docs/DATA_GOVERNANCE.md.",
            args.raw_data,
        )
        return 2

    config = PipelineConfig(
        raw_data_path=args.raw_data,
        test_data_path=args.test_data,
        output_dir=args.output,
    )
    pipeline = MLPipeline(config=config)

    log.info("Stage 1 only, to check the data loads...")
    df = pipeline.stage_1_load_data()
    log.info("Loaded %s rows, columns: %s", f"{len(df):,}", list(df.columns))

    log.info("Running the full pipeline...")
    results = pipeline.run_full_pipeline()
    log.info("Pipeline finished with status: %s", results["status"])
    for name, metrics in results.get("metrics", {}).items():
        log.info("  %s: %s", name, metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
