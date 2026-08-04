"""
Serving pipeline: booking-level test data -> demand forecasts.

Orchestration only. Every transform is imported from `ML_Pipeline.features` so
the serving path and the training path are provably the same code, and each
model's feature contract travels with it in a `ModelBundle`.

The forecast window is derived from the data. The previous version hardcoded
`datetime(2021, 3, 26)` / `datetime(2021, 3, 27)` and `range(0, 51)` in five
places - carrying the comment "Change this Data based on your data" - so on any
other period the without-lag output came back empty and the with-lag loop
targeted timestamps that did not exist.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import load

from ML_Pipeline.features import (
    CLUSTER_COL,
    TS_COL,
    ModelBundle,
    add_calendar_features,
    build_demand_grid,
    validate_grid,
)
from ML_Pipeline.forecast import (
    forecast_direct,
    forecast_recursive,
)

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_MINUTES = 30


def _read_bookings(path: str | Path) -> pd.DataFrame:
    """Read booking-level CSV, gzip-compressed or not."""
    try:
        return pd.read_csv(path, compression="gzip", low_memory=False)
    except (OSError, EOFError, ValueError):
        return pd.read_csv(path, compression=None, low_memory=False)


def _cluster_centroids(cluster_model: Any) -> np.ndarray | None:
    centers = getattr(cluster_model, "cluster_centers_", None)
    return None if centers is None else np.asarray(centers)


def prediction_pipeline(
    cleaned_data_path: str,
    cluster_model_path: str,
    predict_without_lag_path: str,
    predict_with_lag_path: str,
    data_without_lag_path: str,
    data_with_lag_path: str,
    *,
    horizon_steps: int | None = None,
    horizon_start: str | pd.Timestamp | None = None,
    interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
    freq: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Generate demand forecasts for a booking-level test file.

    Args:
        cleaned_data_path: Booking-level CSV with `ts`, `pick_lat`, `pick_lng`.
        cluster_model_path: Fitted clustering model (joblib).
        predict_without_lag_path: Lag-free model bundle (joblib).
        predict_with_lag_path: Lag-using model bundle (joblib).
        data_without_lag_path: Where to write direct forecasts.
        data_with_lag_path: Where to write recursive forecasts.
        horizon_steps: Intervals to forecast. Defaults to one day's worth.
        horizon_start: First forecast interval. Defaults to the interval right
            after the last observed booking, i.e. forecasting genuinely forward.
        interval_minutes: Grid interval width.
        freq: Pandas offset alias. Defaults to `interval_minutes` minutes.

    Returns:
        `{"without_lag": DataFrame, "with_lag": DataFrame}`. Both carry
        `request_count_pred` and `is_forecast`; the caller no longer has to
        guess which rows are predictions.
    """
    freq = freq or f"{interval_minutes}min"

    logger.info("Loading booking data from %s", cleaned_data_path)
    bookings = _read_bookings(cleaned_data_path)

    required = {TS_COL, "pick_lat", "pick_lng"}
    missing = sorted(required.difference(bookings.columns))
    if missing:
        raise KeyError(
            f"Booking data is missing required column(s): {missing}. "
            f"Found: {list(bookings.columns)}"
        )

    cluster_model = load(cluster_model_path)
    without_lag = ModelBundle.load_bundle(predict_without_lag_path)
    with_lag = ModelBundle.load_bundle(predict_with_lag_path)
    centroids = _cluster_centroids(cluster_model)

    # Assign clusters using the model fitted at training time, so serving and
    # training agree on what "cluster 7" means. `.to_numpy()` matches how the
    # model was fitted; passing a named DataFrame triggers a scikit-learn
    # feature-names warning.
    bookings[CLUSTER_COL] = cluster_model.predict(
        bookings[["pick_lat", "pick_lng"]].to_numpy()
    )
    n_clusters = int(getattr(cluster_model, "n_clusters", bookings[CLUSTER_COL].nunique()))
    labels = list(range(n_clusters))

    panel = build_demand_grid(
        bookings,
        freq=freq,
        interval_minutes=interval_minutes,
        clusters=labels,
    )
    report = validate_grid(panel, freq=freq)
    logger.info("Observed panel: %(rows)d rows, rectangular=%(is_rectangular)s", report)

    panel = add_calendar_features(panel, TS_COL)

    # Horizon derived from the data unless the caller pins it.
    step = pd.tseries.frequencies.to_offset(freq)
    last_observed = pd.to_datetime(panel[TS_COL]).max()
    start = (
        pd.Timestamp(horizon_start)
        if horizon_start is not None
        else last_observed + step
    )
    steps = int(horizon_steps) if horizon_steps else int(pd.Timedelta("1D") / step)
    horizon = pd.date_range(start=start, periods=steps, freq=freq)

    logger.info(
        "Forecast horizon: %d intervals, %s to %s (%d clusters)",
        steps, horizon[0], horizon[-1], len(labels),
    )

    use_centroids = centroids is not None and "cluster_lat" in without_lag.feature_names
    direct = forecast_direct(
        without_lag,
        horizon,
        labels,
        centroids=centroids if use_centroids else None,
    )
    _write(direct, data_without_lag_path)

    use_centroids_lag = centroids is not None and "cluster_lat" in with_lag.feature_names
    recursive = forecast_recursive(
        with_lag,
        panel,
        horizon,
        clusters=labels,
        centroids=centroids if use_centroids_lag else None,
    )
    _write(recursive, data_with_lag_path)

    logger.info(
        "Forecasts complete: %d direct rows, %d recursive rows - all predicted.",
        len(direct), len(recursive),
    )
    return {"without_lag": direct, "with_lag": recursive}


def _write(df: pd.DataFrame, path: str | Path) -> None:
    """Write a forecast frame, gzip-compressed, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip")
    logger.info("Wrote %d forecast rows to %s", len(df), path)
