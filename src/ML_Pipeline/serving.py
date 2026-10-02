"""
Resolving the model that is actually serving.

Split out of `ML_Pipeline.api` so it can be used without a web stack. The
dashboard needs the same three things the API does - the promoted model, the
history its lags read from, and the cluster centroids - and should not have to
install FastAPI to get them. One definition, two consumers.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import load

from ML_Pipeline.config import ModelRegistry, PipelineConfig, latest_artifact
from ML_Pipeline.features import CLUSTER_COL, TARGET_COL, TS_COL, ModelBundle
from ML_Pipeline.utils import read_csv_any

logger = logging.getLogger(__name__)


#: Longest horizon served, in intervals. 96 x 30min = two days, which is where
#: the measured MASE stops clearly beating seasonal-naive. See the horizon table
#: in docs/MODEL_CARD.md.
MAX_HORIZON_STEPS = 96

#: Retrain cadence from the model card. Past this the model is reported stale.
STALE_AFTER_DAYS = 28


class ServingState:
    """
    Everything loaded once at startup: the model, its history, its centroids.

    The demand grid is ~878k rows on the reference dataset, so it is read once
    rather than per request. A recursive forecast needs `max(lag)` intervals of
    contiguous history - 7 days with the current lag set - which the grid has and
    a single day's booking file does not.
    """

    def __init__(self, config: PipelineConfig | None = None):
        if config is None:
            # Honour BIKETAXI_OUTPUT_DIR, which is how the container points the
            # API at a mounted output volume, and the same variable the dashboard
            # already reads.
            output_dir = os.environ.get("BIKETAXI_OUTPUT_DIR", "output")
            config = PipelineConfig(output_dir=output_dir)
        self.config = config
        self.registry = ModelRegistry(
            str(Path(self.config.output_dir) / "model_registry.json")
        )
        self.model_name: str | None = None
        self.info: dict[str, Any] = {}
        self.bundle: ModelBundle | None = None
        self.history: pd.DataFrame | None = None
        self.centroids = None
        self.load()

    def load(self) -> None:
        """Resolve the promoted model and its history. Tolerates absence."""
        promoted = self.registry.production_model()
        if promoted is None:
            logger.warning(
                "No model is promoted to production. Promote one with "
                "ModelRegistry.promote_model(name); /forecast will return 503 "
                "until then."
            )
            return
        self.model_name, self.info = promoted

        model_path = Path(self.info["model_path"])
        if not model_path.exists():
            logger.error(
                "Promoted model %r points at %s, which does not exist.",
                self.model_name, model_path,
            )
            return
        self.bundle = ModelBundle.load_bundle(model_path)

        grid_path = latest_artifact(self.config.output_dir, "prepared")
        if grid_path is None:
            logger.error(
                "No demand grid in %s; recursive forecasting has no history to "
                "seed its lags from.", self.config.output_dir,
            )
            return
        history = read_csv_any(grid_path)
        history[TS_COL] = pd.to_datetime(history[TS_COL])
        self.history = history[[TS_COL, CLUSTER_COL, TARGET_COL]].sort_values(
            [CLUSTER_COL, TS_COL]
        )

        cluster_path = Path(
            self.info.get("metadata", {}).get("cluster_model_path")
            or self.config.get_model_path("clustering")
        )
        if "cluster_lat" in self.bundle.feature_names:
            version = self.info.get("metadata", {}).get("version")
            if version and not cluster_path.exists():
                cluster_path = Path(self.config.output_dir) / (
                    f"pickup_cluster_model_{version}.joblib"
                )
            if cluster_path.exists():
                self.centroids = getattr(load(cluster_path), "cluster_centers_", None)
            else:
                logger.error(
                    "Model uses centroid features but no clustering model was "
                    "found at %s.", cluster_path,
                )

        logger.info(
            "Serving %r, history to %s, %d clusters.",
            self.model_name, self.history[TS_COL].max(),
            self.history[CLUSTER_COL].nunique(),
        )

    # --- derived facts ---------------------------------------------------

    @property
    def trained_at(self) -> datetime | None:
        raw = (self.bundle.trained_at if self.bundle else None) or self.info.get(
            "timestamp"
        )
        try:
            return datetime.fromisoformat(raw) if raw else None
        except (TypeError, ValueError):
            return None

    @property
    def age_days(self) -> float | None:
        trained = self.trained_at
        return None if trained is None else (datetime.now() - trained).days * 1.0

    @property
    def stale(self) -> bool:
        age = self.age_days
        return bool(age is not None and age > STALE_AFTER_DAYS)

    @property
    def history_ends_at(self) -> pd.Timestamp | None:
        return None if self.history is None else self.history[TS_COL].max()

    @property
    def ready(self) -> bool:
        return self.bundle is not None and self.history is not None

    def clusters(self) -> list[int]:
        if self.history is None:
            return []
        return sorted(int(c) for c in self.history[CLUSTER_COL].unique())


@lru_cache(maxsize=1)
def get_state() -> ServingState:
    """Process-wide serving state, built on first use."""
    return ServingState()
