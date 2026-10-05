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

Operation:

* **Authentication is optional.** With `BIKETAXI_API_KEY` set, every endpoint
  but `/health` needs it in an `X-API-Key` header. Unset, the API is open, which
  suits a public demo; it says so in the log at startup. `/health` stays open
  for the container healthcheck and load balancers.
* **`POST /reload` picks up a new model or grid without a restart.** It needs
  the key, and is refused outright when none is configured, so an open demo
  cannot be made to re-read its data on demand.
* **Each request is logged** with its status, duration and the model serving
  it.
"""

from __future__ import annotations

import logging
import os
import secrets
import time
from datetime import datetime
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field

from ML_Pipeline.features import CLUSTER_COL, TS_COL
from ML_Pipeline.forecast import PREDICTION_COL, forecast_recursive
from ML_Pipeline.serving import (
    MAX_HORIZON_STEPS,
    STALE_AFTER_DAYS,
    ServingState,
    get_state,
    reload_state,
)

logger = logging.getLogger(__name__)
access_logger = logging.getLogger("ML_Pipeline.api.access")

# Uvicorn configures its own loggers but not the root, so without this the
# module's INFO lines - the access log included - are dropped.
logging.basicConfig(
    level=os.environ.get("BIKETAXI_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

API_KEY_ENV = "BIKETAXI_API_KEY"


def _configured_key() -> str | None:
    """
    Read per request, so a key set or rotated in the environment applies.

    Surrounding whitespace is dropped. The first key generated for the Cloud Run
    deployment ended in a carriage return - Windows openssl writes CRLF - and a
    header value cannot carry one, so no client could ever have matched it.
    """
    return (os.environ.get(API_KEY_ENV) or "").strip() or None


def require_api_key(x_api_key: str | None = Header(None)) -> None:
    """Reject a request without the configured key. Open when none is set."""
    expected = _configured_key()
    if expected is None:
        return
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key. Send it in the X-API-Key header.",
        )

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
    model_data_through: datetime | None = Field(
        default=None, description="End of the data the model was fitted on."
    )
    data_lag_days: float | None = Field(
        default=None,
        description="Days of observed history the model has not been fitted on.",
    )
    stale: bool = Field(
        description=f"True when the model is older than {STALE_AFTER_DAYS} days, "
        f"or its training data ends more than {STALE_AFTER_DAYS} days before the "
        "history does - past which it has been measured to lose to a "
        "seasonal-naive baseline."
    )
    warnings: list[str] = []
    forecast: list[ForecastPoint]


class ModelInfoResponse(BaseModel):
    model_name: str
    stage: str
    trained_at: datetime | None = None
    model_age_days: float | None = None
    data_through: datetime | None = None
    data_lag_days: float | None = None
    stale: bool
    features: list[str]
    lags: list[int]
    beats_seasonal_naive: bool | None = None
    baseline_mase: float | None = None
    test_rmse: float | None = None
    clusters_losing_to_naive: int | None = None
    history_ends_at: datetime | None = None
    max_horizon_steps: int = MAX_HORIZON_STEPS


app = FastAPI(
    title="Bike-Taxi Demand Forecast",
    description=(
        "Forecasts ride-request demand per geographic cluster per 30-minute "
        "interval. Serves the model promoted in the registry, which cannot be a "
        "model that failed its deploy gate."
    ),
    version="1.0.0",
)

if _configured_key() is None:
    logger.warning(
        "%s is not set: the API is open to anyone who can reach it, and /reload "
        "is disabled. Set it for anything but a public demo.", API_KEY_ENV,
    )


class ReloadResponse(BaseModel):
    reloaded: bool
    model_name: str | None = None
    previous_model_name: str | None = None
    history_ends_at: datetime | None = None
    detail: str


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """One line per request: method, path, status, duration, serving model."""
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    try:
        model = get_state().model_name
    except Exception:  # noqa: BLE001 - logging must never fail a request
        model = None
    access_logger.info(
        "%s %s %d %.1fms model=%s",
        request.method, request.url.path, response.status_code, elapsed_ms,
        model or "-",
    )
    return response


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


@app.post(
    "/reload", response_model=ReloadResponse,
    dependencies=[Depends(require_api_key)],
)
def reload() -> ReloadResponse:
    """
    Load the promoted model and the latest demand grid without a restart.

    Run after promoting a model or refreshing the grid. If the new state cannot
    serve, the running model stays in place and the response says why.
    """
    if _configured_key() is None:
        raise HTTPException(
            status_code=403,
            detail=f"Reload is disabled while the API is open. Set {API_KEY_ENV} "
                   "to enable it.",
        )
    fresh, previous = reload_state()
    previous_name = previous.model_name if previous else None
    if not fresh.ready:
        missing = []
        if fresh.bundle is None:
            missing.append("no promoted model could be loaded")
        if fresh.history is None:
            missing.append("no demand grid was found")
        raise HTTPException(
            status_code=409,
            detail="Reload refused, still serving "
                   f"{previous_name or 'nothing'}: " + "; ".join(missing) + ".",
        )
    logger.info("Reloaded: %s -> %s", previous_name, fresh.model_name)
    return ReloadResponse(
        reloaded=True,
        model_name=fresh.model_name,
        previous_model_name=previous_name,
        history_ends_at=fresh.history_ends_at,
        detail="Serving the newly loaded model and history.",
    )


@app.get(
    "/model", response_model=ModelInfoResponse,
    dependencies=[Depends(require_api_key)],
)
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
        data_through=state.data_through,
        data_lag_days=state.data_lag_days,
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


@app.get("/clusters", dependencies=[Depends(require_api_key)])
def clusters() -> dict[str, Any]:
    """Clusters this model can forecast for."""
    state = get_state()
    if not state.ready:
        raise HTTPException(status_code=503, detail="Serving state is not ready.")
    return {"clusters": state.clusters(), "count": len(state.clusters())}


@app.get(
    "/forecast", response_model=ForecastResponse,
    dependencies=[Depends(require_api_key)],
)
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
    staleness = state.staleness_warning()
    if staleness:
        warnings.append(staleness)
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
        model_data_through=state.data_through,
        data_lag_days=state.data_lag_days,
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
    "reload_state",
]
