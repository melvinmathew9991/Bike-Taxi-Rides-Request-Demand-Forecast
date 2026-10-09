"""
Serving pipeline: booking-level test data -> demand forecasts.

Orchestration only. Every transform is imported from `ML_Pipeline.modeling.features` so
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

from ML_Pipeline.modeling.features import (
    CLUSTER_COL,
    TARGET_COL,
    TS_COL,
    ModelBundle,
    add_calendar_features,
    build_demand_grid,
    validate_grid,
)
from ML_Pipeline.modeling.forecast import (
    PREDICTION_COL,
    forecast_direct,
    forecast_recursive,
)
from ML_Pipeline.utils import read_csv_any

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_MINUTES = 30


def _read_bookings(path: str | Path) -> pd.DataFrame:
    """Read booking-level CSV, gzip-compressed or not."""
    return read_csv_any(path)


def _seed_history(
    observed: pd.DataFrame, history_path: str | Path | None, freq: str
) -> pd.DataFrame:
    """
    Build the panel that seeds the recursive forecaster's lags.

    A lag-using model needs `max(lags)` intervals of history immediately before
    the horizon. The test file alone cannot supply that once the model carries a
    weekly lag: the reference test file covers a single day - 48 intervals -
    against the 336 a `lag_336` model requires, so seeding from it raised
    "History has only 48 intervals; this model needs 336" and the run failed at
    this stage.

    The demand grid the pipeline has just written is the right source. It is
    already the `[ts, pickup_cluster, request_count]` panel this function wants,
    it spans the whole training period, and it ends at the interval the horizon
    starts from. Where the two overlap the test file wins, because it is the
    more recent observation of the same intervals.

    Args:
        observed: Panel built from the test bookings.
        history_path: Demand grid to seed from. When None, only `observed` is
            used - which is correct for a model whose lags fit inside it.
        freq: Grid frequency, for the contiguity report.

    Returns:
        Panel sorted by cluster then timestamp, one row per (ts, cluster).
    """
    if history_path is None or not Path(history_path).exists():
        if history_path is not None:
            logger.warning(
                "History grid not found at %s; seeding lags from the test file "
                "alone. A model with lags longer than that file will fail.",
                history_path,
            )
        return observed

    grid = _read_bookings(history_path)
    missing = sorted({TS_COL, CLUSTER_COL, TARGET_COL}.difference(grid.columns))
    if missing:
        raise KeyError(
            f"History grid {history_path} is missing column(s): {missing}. "
            f"Expected the aggregated demand grid, found: {list(grid.columns)}"
        )
    grid[TS_COL] = pd.to_datetime(grid[TS_COL])

    panel_cols = [TS_COL, CLUSTER_COL, TARGET_COL]
    combined = (
        pd.concat([grid[panel_cols], observed[panel_cols]], ignore_index=True)
        .drop_duplicates(subset=[TS_COL, CLUSTER_COL], keep="last")
        .sort_values([CLUSTER_COL, TS_COL])
        .reset_index(drop=True)
    )
    logger.info(
        "Lag history: %s intervals from the grid + %s from the test file "
        "= %s intervals, %s to %s",
        f"{grid[TS_COL].nunique():,}", f"{observed[TS_COL].nunique():,}",
        f"{combined[TS_COL].nunique():,}",
        combined[TS_COL].min(), combined[TS_COL].max(),
    )
    return combined


def cold_start_clusters(
    history: pd.DataFrame,
    clusters: list[int],
    horizon_start: pd.Timestamp,
    *,
    freq: str,
    needed: int,
) -> list[int]:
    """
    Clusters without `needed` observed intervals immediately before the horizon.

    These are the only clusters the lag-free model forecasts. It has no channel
    carrying current demand and loses to seasonal-naive by 75%, so for any
    cluster with enough history the lag model is the forecast; the lag-free one
    is the fallback for a cluster too new to have a week behind it - one added
    by a cluster refit, say.

    When the history as a whole is shorter than `needed`, no cluster is called
    cold: that is a missing or wrong history source, not a new cluster, and the
    recursive forecaster refuses it by name rather than having every cluster
    quietly fall back to the weaker model.
    """
    step = pd.tseries.frequencies.to_offset(freq)
    window_start = horizon_start - needed * step
    stamps = pd.to_datetime(history[TS_COL])
    recent = history[(stamps >= window_start) & (stamps < horizon_start)]
    if recent[TS_COL].nunique() < needed:
        return []
    counts = recent.groupby(CLUSTER_COL)[TS_COL].nunique()
    return [c for c in clusters if counts.get(c, 0) < needed]


def _empty_forecast() -> pd.DataFrame:
    """A forecast file with no rows, so every run writes both files."""
    return pd.DataFrame(columns=[TS_COL, CLUSTER_COL, PREDICTION_COL, "is_forecast"])


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
    history_path: str | Path | None = None,
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
        history_path: Aggregated demand grid used to seed the recursive model's
            lags. Required whenever the model's longest lag exceeds the span of
            `cleaned_data_path` - which it does by default, since `lag_336`
            needs 7 days and a test file is typically one.
        horizon_steps: Intervals to forecast. Defaults to one day's worth.
        horizon_start: First forecast interval. Defaults to the interval right
            after the last observed booking, i.e. forecasting genuinely forward.
        interval_minutes: Grid interval width.
        freq: Pandas offset alias. Defaults to `interval_minutes` minutes.

    Returns:
        `{"without_lag": DataFrame, "with_lag": DataFrame}`. `with_lag` is the
        forecast, for every cluster with a week of history behind the horizon.
        `without_lag` covers only the cold-start clusters that lack it, and is
        usually empty. Both carry `request_count_pred` and `is_forecast`.
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

    # The recursive model reads its lags from history, so it needs depth the
    # test file does not have once the lag set reaches a week back.
    history = _seed_history(panel, history_path, freq)

    needed = max(max(with_lag.lags), int(with_lag.rolling_window))
    cold = cold_start_clusters(history, labels, horizon[0], freq=freq, needed=needed)
    warm = [c for c in labels if c not in cold]
    if cold:
        logger.warning(
            "%d cluster(s) lack %d intervals of history before the horizon and "
            "get the lag-free model, which loses to seasonal-naive by 75%%: %s",
            len(cold), needed, cold,
        )

    use_centroids_lag = centroids is not None and "cluster_lat" in with_lag.feature_names
    recursive = (
        forecast_recursive(
            with_lag,
            history,
            horizon,
            clusters=warm,
            centroids=centroids if use_centroids_lag else None,
        )
        if warm
        else _empty_forecast()
    )
    _write(recursive, data_with_lag_path)

    use_centroids = centroids is not None and "cluster_lat" in without_lag.feature_names
    direct = (
        forecast_direct(
            without_lag,
            horizon,
            cold,
            centroids=centroids if use_centroids else None,
        )
        if cold
        else _empty_forecast()
    )
    _write(direct, data_without_lag_path)

    logger.info(
        "Forecasts complete: %d recursive rows for %d clusters, %d cold-start "
        "rows for %d.", len(recursive), len(warm), len(direct), len(cold),
    )
    return {"without_lag": direct, "with_lag": recursive}


def _write(df: pd.DataFrame, path: str | Path) -> None:
    """Write a forecast frame, gzip-compressed, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip")  # path carries .csv.gz
    logger.info("Wrote %d forecast rows to %s", len(df), path)
