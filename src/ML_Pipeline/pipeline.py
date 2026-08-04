"""
End-to-end pipeline orchestrator.

Stages: load -> clean -> business rules -> cluster & aggregate -> train ->
forecast.

The orchestrator now takes a `PipelineConfig` and threads it through every
stage, so settings recorded in the run snapshot are the settings that were
actually used. Previously it took loose path arguments, and the config object
was constructed, logged, saved and then ignored.

Stage 6 also returns its forecasts. Previously `prediction_pipeline` returned
`None`, the orchestrator logged "Predictions output not returned as a DataFrame",
and `run_pipeline` reported success - so a run could complete "successfully"
while producing nothing usable.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import load

from ML_Pipeline.config import PipelineConfig
from ML_Pipeline.data_prep_advanced import data_prep_advanced
from ML_Pipeline.data_prep_basic import data_prep_basic
from ML_Pipeline.data_prep_geospatial import data_prep_geospatial
from ML_Pipeline.model_training import model_training
from ML_Pipeline.prediction_pipeline import prediction_pipeline

logger = logging.getLogger(__name__)


class MLPipeline:
    """Runs the demand-forecast pipeline end to end."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        raw_data_path: str | None = None,
        output_dir: str | None = None,
        test_data_path: str | None = None,
    ):
        """
        Args:
            config: Full pipeline configuration. Preferred.
            raw_data_path, output_dir, test_data_path: Legacy positional-style
                overrides, applied on top of `config` when supplied.
        """
        self.config = config or PipelineConfig()
        if raw_data_path is not None:
            self.config.raw_data_path = raw_data_path
        if output_dir is not None:
            self.config.output_dir = output_dir
        if test_data_path is not None:
            self.config.test_data_path = test_data_path

        self.config.ensure_directories()
        self.output_dir = Path(self.config.output_dir)

        self.df_raw: pd.DataFrame | None = None
        self.df_processed: pd.DataFrame | None = None
        self.models: dict[str, Any] = {}
        self.metrics: dict[str, dict[str, float]] = {}
        self.forecasts: dict[str, pd.DataFrame] = {}

        logger.info("Pipeline initialised. Output directory: %s", self.output_dir)
        self.config.assert_wired()

    # --- paths -----------------------------------------------------------

    @property
    def clean_data_path(self) -> str:
        return self.config.get_data_path("clean")

    @property
    def prepared_data_path(self) -> str:
        return self.config.get_data_path("prepared")

    @property
    def cluster_model_path(self) -> str:
        return self.config.get_model_path("clustering")

    # --- stages ----------------------------------------------------------

    def stage_1_load_data(self) -> pd.DataFrame:
        """Load raw booking data (gzip or plain CSV)."""
        logger.info("=" * 60)
        logger.info("STAGE 1: Loading raw data")
        logger.info("=" * 60)

        path = self.config.raw_data_path
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Raw data not found at {path}. The dataset is git-ignored "
                "because it carries personal data; see docs/DATA_GOVERNANCE.md."
            )
        try:
            self.df_raw = pd.read_csv(path, low_memory=False, compression="gzip")
        except (OSError, EOFError, ValueError):
            self.df_raw = pd.read_csv(path, low_memory=False, compression=None)

        logger.info(
            "Loaded %s rows x %d columns (%.1f MB): %s",
            f"{len(self.df_raw):,}", self.df_raw.shape[1],
            self.df_raw.memory_usage(deep=False).sum() / 1024**2,
            list(self.df_raw.columns),
        )
        return self.df_raw

    def stage_2_basic_preprocessing(self) -> pd.DataFrame:
        """Deduplicate, coerce types, derive per-rider booking gaps."""
        logger.info("=" * 60)
        logger.info("STAGE 2: Basic preprocessing")
        logger.info("=" * 60)

        if self.df_raw is None:
            self.stage_1_load_data()
        self.df_processed = data_prep_basic(self.df_raw)
        logger.info("Shape after basic cleaning: %s", self.df_processed.shape)
        return self.df_processed

    def stage_3_advanced_preprocessing(self) -> pd.DataFrame:
        """Apply business-rule cleaning and persist the cleaned booking data."""
        logger.info("=" * 60)
        logger.info("STAGE 3: Business-rule cleaning")
        logger.info("=" * 60)

        if self.df_processed is None:
            self.stage_2_basic_preprocessing()
        self.df_processed = data_prep_advanced(
            self.df_processed, self.clean_data_path
        )
        logger.info("Shape after business rules: %s", self.df_processed.shape)
        return self.df_processed

    def stage_4_geospatial_clustering(self) -> pd.DataFrame:
        """Cluster pickups and aggregate into the demand grid."""
        logger.info("=" * 60)
        logger.info("STAGE 4: Clustering and aggregation")
        logger.info("=" * 60)

        if self.df_processed is None:
            self.stage_3_advanced_preprocessing()

        self.df_processed = data_prep_geospatial(
            self.df_processed,
            self.cluster_model_path,
            self.prepared_data_path,
            n_clusters=self.config.n_clusters,
            algorithm=self.config.clustering_algorithm,
            freq=self.config.freq,
            interval_minutes=self.config.interval_minutes,
            run_cluster_diagnostics=self.config.run_cluster_diagnostics,
        )
        logger.info(
            "Demand grid: %s rows across %d clusters",
            f"{len(self.df_processed):,}",
            self.df_processed["pickup_cluster"].nunique(),
        )
        return self.df_processed

    def stage_5_model_training(self) -> dict[str, Any]:
        """Train both model variants."""
        logger.info("=" * 60)
        logger.info("STAGE 5: Model training")
        logger.info("=" * 60)

        if self.df_processed is None:
            self.stage_4_geospatial_clustering()

        self.models = model_training(
            self.df_processed,
            self.config.get_model_path("without_lag"),
            self.config.get_model_path("with_lag"),
            config=self.config,
            centroids=self._cluster_centroids(),
        )
        self.metrics = {name: b.metrics for name, b in self.models.items()}
        return self.models

    def stage_6_predictions(self) -> dict[str, pd.DataFrame] | None:
        """Forecast demand for the test window."""
        logger.info("=" * 60)
        logger.info("STAGE 6: Forecasting")
        logger.info("=" * 60)

        if not self.config.test_data_path:
            logger.warning("No test data path configured; skipping forecasting.")
            return None
        if not Path(self.config.test_data_path).exists():
            logger.warning(
                "Test data not found at %s; skipping forecasting.",
                self.config.test_data_path,
            )
            return None

        self.forecasts = prediction_pipeline(
            cleaned_data_path=self.config.test_data_path,
            cluster_model_path=self.cluster_model_path,
            predict_without_lag_path=self.config.get_model_path("without_lag"),
            predict_with_lag_path=self.config.get_model_path("with_lag"),
            data_with_lag_path=self.config.get_data_path("with_lag"),
            data_without_lag_path=self.config.get_data_path("without_lag"),
            horizon_steps=self.config.horizon_steps,
            interval_minutes=self.config.interval_minutes,
            freq=self.config.freq,
        )
        for name, frame in self.forecasts.items():
            logger.info("Forecast %r: %s rows", name, f"{len(frame):,}")
        return self.forecasts

    # --- helpers ---------------------------------------------------------

    def _cluster_centroids(self) -> np.ndarray | None:
        path = Path(self.cluster_model_path)
        if not path.exists():
            return None
        centers = getattr(load(path), "cluster_centers_", None)
        return None if centers is None else np.asarray(centers)

    def run_full_pipeline(self) -> dict[str, Any]:
        """Run every stage in order."""
        logger.info("=" * 60)
        logger.info("BIKE-TAXI DEMAND FORECAST PIPELINE")
        logger.info("Started: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logger.info("=" * 60)
        started = datetime.now()

        self.stage_1_load_data()
        self.stage_2_basic_preprocessing()
        self.stage_3_advanced_preprocessing()
        self.stage_4_geospatial_clustering()
        self.stage_5_model_training()
        forecasts = self.stage_6_predictions()

        elapsed = datetime.now() - started
        logger.info("=" * 60)
        logger.info("PIPELINE SUMMARY")
        logger.info("  Total time: %s", elapsed)
        logger.info("  Models trained: %d", len(self.models))
        for name, metrics in self.metrics.items():
            logger.info(
                "  %s: test RMSE %s, test R2 %s",
                name,
                _fmt(metrics.get("test_rmse")),
                _fmt(metrics.get("test_r2")),
            )
        logger.info("  Output directory: %s", self.output_dir)
        logger.info("=" * 60)

        return {
            "status": "success",
            "total_time": elapsed,
            "models": self.models,
            "metrics": self.metrics,
            "processed_data": self.df_processed,
            "predictions": forecasts,
        }

    def get_pipeline_status(self) -> dict[str, bool]:
        return {
            "data_loaded": self.df_raw is not None,
            "data_processed": self.df_processed is not None,
            "models_trained": len(self.models) > 0,
            "forecasts_generated": len(self.forecasts) > 0,
        }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"
