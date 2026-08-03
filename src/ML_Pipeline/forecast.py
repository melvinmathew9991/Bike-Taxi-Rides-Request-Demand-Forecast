"""
Multi-step demand forecasting.

Two regimes, matching the two trained model variants:

* **Direct** (no lag features) - every horizon interval is independent of every
  other, so the whole horizon is predicted in one vectorised call.

* **Recursive** (with lag features) - interval *t* needs the demand at
  *t-1, t-2, t-3*, which for anything past the first step is itself a forecast.
  Each step is predicted, written back into the working series, and used as
  input to the next.

The previous implementation attempted the recursive case but did three things
wrong, and the combination meant almost nothing was actually forecast:

1. It looped a fixed `range(3)`, overwriting one timestamp per iteration, so
   exactly three intervals were predicted regardless of the horizon. Everything
   else written to `data_with_lag.csv` was the *input* count, unchanged and
   indistinguishable from a forecast.
2. Each iteration re-ran the lag builder on its own previous output. That
   builder ends in `dropna()`, and `lag_3` is NaN for the first three intervals
   of every cluster, so each pass silently deleted three more intervals per
   cluster from the front of the series.
3. The function returned `None`, so the caller logged success and passed nothing
   downstream.

Here the horizon is explicit, the history is kept separate from the forecast,
every horizon row is predicted, and the result is returned with an `is_forecast`
flag so a consumer can always tell a prediction from an observation.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ML_Pipeline.features import (
    CLUSTER_COL,
    TARGET_COL,
    TS_COL,
    ModelBundle,
    add_calendar_features,
    attach_cluster_centroids,
)

logger = logging.getLogger(__name__)

PREDICTION_COL = "request_count_pred"


def _horizon_index(
    start: pd.Timestamp, steps: int, freq: str
) -> pd.DatetimeIndex:
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    return pd.date_range(start=start, periods=steps, freq=freq)


def _as_wide(
    history: pd.DataFrame, clusters: Sequence[int], freq: str
) -> pd.DataFrame:
    """Pivot the long panel to timestamps x clusters, on a regular index."""
    wide = history.pivot_table(
        index=TS_COL, columns=CLUSTER_COL, values=TARGET_COL, aggfunc="last"
    )
    full = pd.date_range(wide.index.min(), wide.index.max(), freq=freq)
    wide = wide.reindex(index=full, columns=list(clusters))
    if wide.isna().any().any():
        n = int(wide.isna().sum().sum())
        logger.warning(
            "History has %d missing (timestamp, cluster) cells; filling with 0. "
            "A gappy history makes early lags unreliable.", n,
        )
        wide = wide.fillna(0.0)
    return wide.astype("float64")


def forecast_direct(
    bundle: ModelBundle,
    horizon: pd.DatetimeIndex,
    clusters: Sequence[int],
    *,
    centroids: np.ndarray | None = None,
    clip_min: float | None = 0.0,
) -> pd.DataFrame:
    """
    Forecast a horizon with a model that uses no lag features.

    Every interval is independent, so this is a single vectorised prediction over
    the full (horizon x clusters) grid.
    """
    if bundle.uses_lags:
        raise ValueError(
            "forecast_direct requires a lag-free model; this bundle uses lags. "
            "Use forecast_recursive."
        )

    index = pd.MultiIndex.from_product(
        [horizon, list(clusters)], names=[TS_COL, CLUSTER_COL]
    )
    frame = index.to_frame(index=False)
    frame = add_calendar_features(frame, TS_COL)
    if centroids is not None:
        frame = attach_cluster_centroids(frame, centroids)

    frame[PREDICTION_COL] = bundle.predict(frame, clip_min=clip_min)
    frame["is_forecast"] = True
    logger.info(
        "Direct forecast: %d intervals x %d clusters = %d predictions",
        len(horizon), len(clusters), len(frame),
    )
    return frame.sort_values([CLUSTER_COL, TS_COL]).reset_index(drop=True)


def forecast_recursive(
    bundle: ModelBundle,
    history: pd.DataFrame,
    horizon: pd.DatetimeIndex,
    *,
    clusters: Sequence[int] | None = None,
    centroids: np.ndarray | None = None,
    clip_min: float | None = 0.0,
) -> pd.DataFrame:
    """
    Forecast a horizon recursively with a lag-using model.

    At each step the lag inputs are read from the working series, which holds
    observed demand for historical intervals and previously predicted demand for
    intervals already forecast. Predictions are written back before the next
    step, so error compounds across the horizon exactly as it would in
    production - which is the honest thing to measure.

    Args:
        bundle: Fitted lag-using model plus its feature contract.
        history: Observed panel `[ts, pickup_cluster, request_count]`. Must cover
            at least `max(lags)` intervals immediately before `horizon[0]`.
        horizon: Timestamps to forecast, contiguous at `bundle.freq`.
        clusters: Clusters to forecast. Defaults to those present in `history`.
        centroids: `(n_clusters, 2)` centroid array, if the model uses them.
        clip_min: Floor applied to every prediction. `0.0` by default because
            the target is a non-negative count.

    Returns:
        Long frame of `[ts, pickup_cluster, request_count_pred, is_forecast]`
        plus the calendar/lag features used, one row per (horizon, cluster).
    """
    if not bundle.uses_lags:
        raise ValueError(
            "forecast_recursive requires a lag-using model; this bundle has "
            "none. Use forecast_direct."
        )

    lags = tuple(sorted(int(lag) for lag in bundle.lags))
    max_lag = max(lags)
    window = int(bundle.rolling_window)
    freq = bundle.freq

    hist = history.copy()
    hist[TS_COL] = pd.to_datetime(hist[TS_COL])
    labels = (
        list(clusters)
        if clusters is not None
        else sorted(hist[CLUSTER_COL].unique().tolist())
    )

    hist = hist[hist[TS_COL] < horizon[0]]
    if hist.empty:
        raise ValueError(
            f"History contains no intervals before the horizon start "
            f"({horizon[0]}). Recursive forecasting needs {max_lag} prior "
            "intervals to seed the lags."
        )

    wide = _as_wide(hist, labels, freq)
    needed = max(max_lag, window)
    if len(wide) < needed:
        raise ValueError(
            f"History has only {len(wide)} intervals; this model needs {needed} "
            f"(max lag {max_lag}, rolling window {window}) to seed the lags."
        )

    step = pd.tseries.frequencies.to_offset(freq)
    expected_start = wide.index[-1] + step
    if horizon[0] != expected_start:
        raise ValueError(
            f"Horizon starts at {horizon[0]} but history ends at "
            f"{wide.index[-1]}; expected the horizon to start at "
            f"{expected_start} so the series is contiguous."
        )

    # Working series: history followed by rows to be filled in as we go.
    working = pd.concat(
        [wide, pd.DataFrame(index=horizon, columns=wide.columns, dtype="float64")]
    )
    n_hist = len(wide)
    values = working.to_numpy(dtype="float64")

    base = pd.DataFrame({CLUSTER_COL: labels})
    if centroids is not None:
        base = attach_cluster_centroids(base, centroids)

    records = []
    for offset, stamp in enumerate(horizon):
        row = n_hist + offset

        frame = base.copy()
        frame[TS_COL] = stamp
        frame = add_calendar_features(frame, TS_COL)
        for lag in lags:
            frame[f"lag_{lag}"] = values[row - lag, :]
        frame["rolling_mean"] = values[row - window : row, :].mean(axis=0)

        preds = bundle.predict(frame, clip_min=clip_min)
        values[row, :] = preds

        frame[PREDICTION_COL] = preds
        records.append(frame)

    result = pd.concat(records, ignore_index=True)
    result["is_forecast"] = True
    logger.info(
        "Recursive forecast: %d steps x %d clusters = %d predictions "
        "(%s to %s)",
        len(horizon), len(labels), len(result), horizon[0], horizon[-1],
    )
    return result.sort_values([CLUSTER_COL, TS_COL]).reset_index(drop=True)


def backtest_recursive(
    bundle: ModelBundle,
    panel: pd.DataFrame,
    *,
    horizon_steps: int,
    centroids: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Forecast the final `horizon_steps` of an observed panel and pair the result
    with what actually happened.

    This is the honest evaluation for a recursive forecaster: it compounds its
    own errors across the horizon, whereas scoring one-step-ahead predictions on
    observed lags flatters the model by handing it ground truth at every step.

    Returns:
        Frame with `request_count` (actual) and `request_count_pred`.
    """
    panel = panel.copy()
    panel[TS_COL] = pd.to_datetime(panel[TS_COL])
    stamps = np.sort(panel[TS_COL].unique())
    if len(stamps) <= horizon_steps:
        raise ValueError(
            f"Panel has {len(stamps)} intervals; need more than "
            f"{horizon_steps} to hold out a horizon."
        )

    split = pd.Timestamp(stamps[-horizon_steps])
    history = panel[panel[TS_COL] < split]
    actual = panel[panel[TS_COL] >= split]
    horizon = _horizon_index(split, horizon_steps, bundle.freq)

    predicted = forecast_recursive(
        bundle, history, horizon, centroids=centroids
    )
    merged = actual.merge(
        predicted[[TS_COL, CLUSTER_COL, PREDICTION_COL]],
        on=[TS_COL, CLUSTER_COL],
        how="left",
        validate="one_to_one",
    )
    logger.info(
        "Backtest: %d actual rows over %d steps from %s",
        len(merged), horizon_steps, split,
    )
    return merged
