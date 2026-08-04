#!/usr/bin/env python
"""
Command-line entry point for the demand-forecast pipeline.

Every flag here reaches the code it names. `--n-clusters` in particular used to
be parsed, logged, written into the saved config snapshot, and then discarded
while the clustering stage hardcoded 50 - and the troubleshooting guide told
users to lower it to fix out-of-memory errors.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ML_Pipeline.config import ModelRegistry, PipelineConfig  # noqa: E402
from ML_Pipeline.pipeline import MLPipeline  # noqa: E402

logger = logging.getLogger("run_pipeline")


def setup_logging(log_file: str | None = None, level: str = "INFO") -> None:
    """Configure root logging to file and console, without duplicate handlers."""
    if log_file is None:
        log_file = f"logs/pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Re-running in one process (notebook, tests) would otherwise stack handlers
    # and multiply every log line.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    logger.info("Logging to %s", log_file)


def run_pipeline(
    config: PipelineConfig | None = None,
    full_run: bool = True,
    stages: list[str] | None = None,
) -> dict:
    """
    Run the pipeline and register the resulting models.

    Args:
        config: Pipeline configuration.
        full_run: Run every stage.
        stages: Subset to run: any of `data`, `features`, `model`, `predict`.
    """
    config = config or PipelineConfig()
    logger.info("Configuration:\n%s", json.dumps(config.to_dict(), indent=2, default=str))

    pipeline = MLPipeline(config=config)
    results: dict = {}

    if full_run or not stages:
        logger.info("Running the full pipeline.")
        results = pipeline.run_full_pipeline()
    else:
        logger.info("Running stages: %s", stages)
        if "data" in stages:
            pipeline.stage_1_load_data()
            pipeline.stage_2_basic_preprocessing()
        if "features" in stages:
            if pipeline.df_processed is None:
                pipeline.stage_2_basic_preprocessing()
            pipeline.stage_3_advanced_preprocessing()
            pipeline.stage_4_geospatial_clustering()
        if "model" in stages:
            if pipeline.df_processed is None:
                pipeline.stage_4_geospatial_clustering()
            pipeline.stage_5_model_training()
        if "predict" in stages:
            results["predictions"] = pipeline.stage_6_predictions()
        results.setdefault("status", "success")
        results["metrics"] = pipeline.metrics

    config.save_config()

    # Register with the metrics the models actually scored. Previously
    # `register_model` was called without `metrics=`, so every entry stored `{}`
    # and `get_best_model` could never return anything.
    registry = ModelRegistry(str(Path(config.output_dir) / "model_registry.json"))
    for name, bundle in pipeline.models.items():
        registry.register_model(
            model_name=f"xgb_{name}_{config.model_version}",
            model_path=config.get_model_path(name),
            model_type="xgboost",
            metrics=bundle.metrics,
            parameters=bundle.params,
            metadata={
                "version": config.model_version,
                "features": bundle.feature_names,
                "uses_lags": bundle.uses_lags,
                "training_rows": bundle.training_rows,
                "trained_at": bundle.trained_at,
                "notes": bundle.notes,
            },
        )

    best = registry.get_best_model("xgboost", metric="test_rmse")
    if best:
        logger.info("Best model by test RMSE: %s (%.4f)", best[0], best[1]["metrics"]["test_rmse"])

    logger.info("Done. Outputs in %s", config.output_dir)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ML pipeline runner for bike-taxi demand forecasting"
    )
    parser.add_argument("--config", help="Path to a configuration JSON file")
    parser.add_argument(
        "--stages", nargs="+", choices=["data", "features", "model", "predict"],
        help="Run only these stages",
    )
    parser.add_argument("--raw-data", default="data/raw_data.csv")
    parser.add_argument(
        "--test-data", default="data/test_dataset/cleaned_test_booking_data.csv"
    )
    parser.add_argument("--output", default="output")
    parser.add_argument(
        "--n-clusters", type=int, default=50, help="Number of geographic clusters"
    )
    parser.add_argument(
        "--test-fraction", type=float, default=0.2,
        help="Chronological share of the timeline held out for testing",
    )
    parser.add_argument(
        "--horizon-steps", type=int, default=None,
        help="Intervals to forecast (default: one day)",
    )
    parser.add_argument(
        "--no-centroids", action="store_true",
        help="Use the raw cluster label instead of centroid coordinates",
    )
    parser.add_argument(
        "--cluster-diagnostics", action="store_true",
        help="Run the (expensive) cluster-count sweep before fitting",
    )
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    setup_logging(args.log_file, args.log_level)

    if args.config:
        logger.info("Loading configuration from %s", args.config)
        config = PipelineConfig.load_config(args.config)
    else:
        config = PipelineConfig(
            raw_data_path=args.raw_data,
            test_data_path=args.test_data,
            output_dir=args.output,
            n_clusters=args.n_clusters,
            test_fraction=args.test_fraction,
            horizon_steps=args.horizon_steps,
            use_cluster_centroids=not args.no_centroids,
            run_cluster_diagnostics=args.cluster_diagnostics,
            log_level=args.log_level,
        )

    try:
        run_pipeline(config=config, full_run=args.stages is None, stages=args.stages)
        return 0
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2
    except Exception:
        logger.exception("Pipeline failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
