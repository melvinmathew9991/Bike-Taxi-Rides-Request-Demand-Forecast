#!/usr/bin/env python
"""
Check the serving model's health, for a scheduler to run.

Scores the promoted model on the most recent week of demand it has not been
fitted on, and fails when it is stale, loses to seasonal-naive, or its level
ratio leaves the band set from the model card's staleness curve. The checks and
their thresholds are documented in `ML_Pipeline.monitoring`.

Exit codes, so a scheduler can act on the result:
    0  healthy (per-cluster warnings may still be printed)
    2  nothing to check: no promoted model, no demand grid, or a feature the
       grid cannot supply
    3  unhealthy: at least one check failed - retrain, or ship the baseline
    1  the check itself crashed

Usage:
    python scripts/monitor_model.py
    python scripts/monitor_model.py --output-dir output --json output/health.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ML_Pipeline.config import PipelineConfig  # noqa: E402
from ML_Pipeline.monitoring import (  # noqa: E402
    DEFAULT_WINDOW_DAYS,
    MAX_LEVEL_RATIO,
    MIN_LEVEL_RATIO,
    check_health,
)
from ML_Pipeline.serving import ServingState  # noqa: E402

logger = logging.getLogger("monitor_model")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--output-dir", default=os.environ.get("BIKETAXI_OUTPUT_DIR", "output"),
        help="Pipeline output directory holding the registry and demand grid "
             "(default: $BIKETAXI_OUTPUT_DIR or ./output)",
    )
    parser.add_argument(
        "--days", type=int, default=DEFAULT_WINDOW_DAYS,
        help=f"Days of recent unseen demand to score (default {DEFAULT_WINDOW_DAYS})",
    )
    parser.add_argument("--min-level-ratio", type=float, default=MIN_LEVEL_RATIO)
    parser.add_argument("--max-level-ratio", type=float, default=MAX_LEVEL_RATIO)
    parser.add_argument(
        "--json", type=Path, default=None,
        help="Also write the full report, per-cluster rows included, to this file",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        state = ServingState(PipelineConfig(output_dir=args.output_dir))
        report = check_health(
            state, days=args.days,
            min_level_ratio=args.min_level_ratio,
            max_level_ratio=args.max_level_ratio,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    except Exception:
        logger.exception("Health check crashed.")
        return 1

    print(f"Model: {report.model_name}")
    if report.window_start:
        print(
            f"Scored {report.rows_scored:,} rows, "
            f"{report.window_start} to {report.window_end}"
        )
    for check in report.checks:
        print(f"  [{check.status.upper():7}] {check.name}: {check.detail}")
    print("HEALTHY" if report.healthy else "UNHEALTHY")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_dict(), indent=2))
    return 0 if report.healthy else 3


if __name__ == "__main__":
    sys.exit(main())
