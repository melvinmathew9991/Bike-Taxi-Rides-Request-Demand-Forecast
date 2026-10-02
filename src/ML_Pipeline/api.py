"""
Forecast serving API.

Before this, nothing in the repository could answer "what is demand in cluster 12
at 18:30 tomorrow" except a person running a script by hand. The whole
deployment surface was absent: no API, no container, no schedule.

Three decisions shape this module, and each is this project's own measurement
turned into code rather than a preference:

**It serves a promoted model, not the newest one.** `ModelRegistry.promote_model`
refuses a model that failed its deploy gate, so a forecaster a free
seasonal-naive baseline would beat cannot reach production by being most recent.
Ranking by raw metric would not do: RMSE is not comparable between runs whose
training data differed, as it was not when the cleaning rules were corrected and
mean demand rose.

**It caps the horizon at two days.** Measured by recursive backtest, MASE against
seasonal-naive runs 0.79 at one day, 0.82 at two, 0.89 at four, 0.99 at one week
and 1.06 at two. Past two days the model's advantage decays, and past a week it
is worse than doing nothing. A request for a longer horizon is refused with that
reason rather than served quietly.

**It reports where its history ends, and whether the model is stale.** A
recursive forecast starts from the last observed interval, so its origin is a
fact the caller needs. And the model is anchored to its training era - demand
grew 5.2x across the reference year - so `/model` reports the model's age against
the four-week retraining cadence the model card mandates.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from joblib import load
from pydantic import BaseModel, Field

from ML_Pipeline.config import ModelRegistry, PipelineConfig, latest_artifact
from ML_Pipeline.features import CLUSTER_COL, TARGET_COL, TS_COL, ModelBundle
from ML_Pipeline.forecast import PREDICTION_COL, forecast_recursive
from ML_Pipeline.utils import read_csv_any

logger = logging.getLogger(__name__)

#: Longest horizon served, in intervals. 96 x 30min = two days, which is where
#: the measured MASE stops clearly beating seasonal-naive. See the horizon table
#: in docs/MODEL_CARD.md.
MAX_HORIZON_STEPS = 96

#: Retrain cadence from the model card. Past this the model is reported stale.
STALE_AFTER_DAYS = 28


class ForecastPoint(BaseModel):
    """One cluster's demand for one interval."""

    ts: datetime
    pickup_cluster: int
    predicted_requests: float = Field(ge=0.0)


class ForecastResponse(BaseModel):
    model_name: str
    history_ends_at: datetime = Field(
        description="Last observed interval. The forecast starts after this."
    )
    horizon_steps: int
    model_trained_at: datetime | None = None
    model_age_days: float | None = None
    stale: bool = Field(
        description=f"True when the model is older than {STALE_AFTER_DAYS} days, "
        "past which it has been measured to lose to a seasonal-naive baseline."
    )
    warnings: list[str] = []
    forecast: list[ForecastPoint]


class ModelInfoResponse(BaseModel):
    model_name: str
    stage: str
    trained_at: datetime | None = None
    model_age_days: float | None = None
    stale: bool
    features: list[str]
    lags: list[int]
    beats_seasonal_naive: bool | None = None
    baseline_mase: float | None = None
    test_rmse: float | None = None
    clusters_losing_to_naive: int | None = None
    history_ends_at: datetime | None = None
    max_horizon_steps: int = MAX_HORIZON_STEPS


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


app = FastAPI(
    title="Bike-Taxi Demand Forecast",
    description=(
        "Forecasts ride-request demand per geographic cluster per 30-minute "
        "interval. Serves the model promoted in the registry, which cannot be a "
        "model that failed its deploy gate."
    ),
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness, plus whether a model is actually loaded and serveable."""
    state = get_state()
    return {
        "status": "ok",
        "model_loaded": state.bundle is not None,
        "history_loaded": state.history is not None,
        "ready": state.ready,
    }


@app.get("/model", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """What is serving, how good it was measured to be, and how old it is."""
    state = get_state()
    if state.bundle is None:
        raise HTTPException(
            status_code=503,
            detail="No model is promoted to production. Promote one with "
                   "ModelRegistry.promote_model(name).",
        )
    metrics = state.info.get("metrics", {})
    beats = metrics.get("beats_seasonal_naive")
    return ModelInfoResponse(
        model_name=state.model_name or "unknown",
        stage=state.info.get("stage", "unknown"),
        trained_at=state.trained_at,
        model_age_days=state.age_days,
        stale=state.stale,
        features=list(state.bundle.feature_names),
        lags=[int(x) for x in state.bundle.lags],
        beats_seasonal_naive=None if beats is None else bool(beats),
        baseline_mase=metrics.get("baseline_mase"),
        test_rmse=metrics.get("test_rmse"),
        clusters_losing_to_naive=(
            int(metrics["clusters_losing_to_naive"])
            if "clusters_losing_to_naive" in metrics
            else None
        ),
        history_ends_at=state.history_ends_at,
    )


@app.get("/clusters")
def clusters() -> dict[str, Any]:
    """Clusters this model can forecast for."""
    state = get_state()
    if not state.ready:
        raise HTTPException(status_code=503, detail="Serving state is not ready.")
    return {"clusters": state.clusters(), "count": len(state.clusters())}


@app.get("/forecast", response_model=ForecastResponse)
def forecast(
    steps: int = Query(
        48,
        ge=1,
        le=MAX_HORIZON_STEPS,
        description=(
            f"Intervals to forecast, at most {MAX_HORIZON_STEPS} (two days). "
            "Beyond that the model has been measured to lose its advantage over "
            "a seasonal-naive baseline."
        ),
    ),
    cluster: list[int] | None = Query(
        None, description="Restrict to these clusters. Omit for all."
    ),
) -> ForecastResponse:
    """
    Forecast demand forward from the last observed interval.

    The forecast is recursive: each step consumes the previous step's prediction
    for its short lags, which is why the horizon is capped. The daily and weekly
    lags remain real observations within two days.
    """
    state = get_state()
    if not state.ready:
        raise HTTPException(
            status_code=503,
            detail="Not ready to serve: "
            + ("no model promoted. " if state.bundle is None else "")
            + ("no demand grid found for history. " if state.history is None else ""),
        )

    assert state.bundle is not None and state.history is not None  # for type checkers
    available = state.clusters()
    requested = available if not cluster else sorted(set(cluster))
    unknown = [c for c in requested if c not in available]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown cluster(s) {unknown}. This model was fitted on "
                   f"{len(available)} clusters, 0-{max(available)}.",
        )

    start = state.history_ends_at + pd.tseries.frequencies.to_offset(
        state.bundle.freq
    )
    horizon = pd.date_range(start=start, periods=steps, freq=state.bundle.freq)

    try:
        predicted = forecast_recursive(
            state.bundle,
            state.history,
            horizon,
            clusters=available,
            centroids=state.centroids,
        )
    except ValueError as exc:
        # Most likely too little history for the lag set - a real condition the
        # caller should see rather than a 500.
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    predicted = predicted[predicted[CLUSTER_COL].isin(requested)]

    warnings: list[str] = []
    if state.stale:
        warnings.append(
            f"The serving model is {state.age_days:.0f} days old, past the "
            f"{STALE_AFTER_DAYS}-day retraining cadence. Demand on this dataset "
            "grew 5.2x in a year and trees cannot extrapolate, so a stale model "
            "under-forecasts. Retrain."
        )
    if steps > 48:
        warnings.append(
            f"A {steps}-step horizon reaches past one day, where the daily lag "
            "starts consuming the model's own predictions. Accuracy decays with "
            "horizon; see the horizon table in docs/MODEL_CARD.md."
        )

    return ForecastResponse(
        model_name=state.model_name or "unknown",
        history_ends_at=state.history_ends_at,
        horizon_steps=steps,
        model_trained_at=state.trained_at,
        model_age_days=state.age_days,
        stale=state.stale,
        warnings=warnings,
        forecast=[
            ForecastPoint(
                ts=row[TS_COL],
                pickup_cluster=int(row[CLUSTER_COL]),
                predicted_requests=float(row[PREDICTION_COL]),
            )
            for _, row in predicted.iterrows()
        ],
    )


__all__ = [
    "MAX_HORIZON_STEPS",
    "STALE_AFTER_DAYS",
    "ServingState",
    "app",
    "get_state",
]
