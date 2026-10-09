#!/usr/bin/env python
"""
Measure prediction-interval coverage: Poisson quantiles against calibrated ones.

The model is fitted with a Poisson objective, so Poisson quantiles around its
prediction are the cheap interval. This checks whether they cover what they
claim, and whether the calibrated intervals in `ML_Pipeline.modeling.intervals`
do better, on recursive forecasts the calibration never saw:

    fit        once, on everything before the pipeline's chronological split
    calibrate  `calibrate_intervals` on origins in the first half of the
               retraining cadence after the split
    score      both intervals on origins in the second half

Both halves sit inside the four-week cadence, so the model is as stale as a
deployed one gets. Coverage is reported overall, by predicted volume and by
horizon, with the share of misses above and below - for dispatch, a miss above
is an under-served interval.

Usage:
    python experiments/measure_intervals.py \\
        --data output/Data_Prepared_<ver>.csv.gz \\
        --cluster-model output/pickup_cluster_model_<ver>.joblib
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from common import BASE_PARAMS, FEATURES, LAGS, ROLLING_WINDOW
from joblib import load
from scipy.stats import poisson

from ML_Pipeline.config import PipelineConfig, max_horizon_steps
from ML_Pipeline.modeling.features import (
    TARGET_COL,
    TS_COL,
    ModelBundle,
    add_calendar_features,
    add_lag_features,
    attach_cluster_centroids,
)
from ML_Pipeline.modeling.intervals import (
    VOLUME_EDGES,
    backtest_errors,
    calibrate_intervals,
    horizon_edges_in_steps,
)
from ML_Pipeline.modeling.splitting import chronological_split
from ML_Pipeline.utils import read_csv_any

logger = logging.getLogger("measure_intervals")


def score(errors: pd.DataFrame, lower: np.ndarray, upper: np.ndarray) -> pd.DataFrame:
    actual = errors["actual"].to_numpy()
    return errors.assign(
        inside=(actual >= lower) & (actual <= upper),
        above=actual > upper,
        below=actual < lower,
        width=upper - lower,
    )


def report(name: str, scored: pd.DataFrame, freq: str) -> None:
    volume = pd.cut(scored["predicted"], [-np.inf, *VOLUME_EDGES, np.inf])
    edges = horizon_edges_in_steps(freq)
    horizon = pd.cut(scored["step"], [0, *edges[:-1], np.inf])
    cols = ["inside", "above", "below", "width"]
    print(f"\n=== {name} ===")
    print(scored[cols].mean().round(3).to_string())
    print("\nby predicted volume:")
    print(scored.groupby(volume, observed=True)[cols].mean().round(3).to_string())
    print("\nby horizon (steps):")
    print(scored.groupby(horizon, observed=True)[cols].mean().round(3).to_string())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Data_Prepared CSV (gzip)")
    parser.add_argument("--cluster-model", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    config = PipelineConfig()
    steps = max_horizon_steps(config.interval_minutes)
    half_days = config.interval_calibration_days // 2
    half_origins = config.interval_origins // 2

    df = read_csv_any(args.data)
    df[TS_COL] = pd.to_datetime(df[TS_COL])
    centroids = np.asarray(load(args.cluster_model).cluster_centers_)
    panel = attach_cluster_centroids(add_calendar_features(df, TS_COL), centroids)
    lagged = add_lag_features(panel, lags=LAGS, rolling_window=ROLLING_WINDOW)
    split = chronological_split(lagged, test_fraction=config.test_fraction)

    model = xgb.XGBRegressor(objective="count:poisson", **BASE_PARAMS)
    model.fit(split.train[FEATURES], split.train[TARGET_COL])
    bundle = ModelBundle(
        model=model, feature_names=FEATURES, uses_lags=True, lags=LAGS,
        rolling_window=ROLLING_WINDOW, freq=config.freq,
    )

    cut = split.test[TS_COL].min()
    calibration = calibrate_intervals(
        bundle, df, start=cut, days=half_days, n_origins=half_origins,
        steps=steps, level=config.interval_level, centroids=centroids,
    )
    if calibration is None:
        raise SystemExit("Too little data after the split to calibrate.")

    later = cut + pd.Timedelta(days=half_days)
    last = later + pd.Timedelta(days=half_days) - steps * pd.Timedelta(config.freq)
    origins = list(pd.date_range(later, last, periods=half_origins).round(config.freq))
    errors = backtest_errors(bundle, df, origins, steps=steps, centroids=centroids)
    logger.info(
        "Fitted before %s; calibrated on %d origins from %s; scored %d errors "
        "from %d origins, %s to %s.",
        cut, calibration.origins, calibration.window_start, len(errors),
        len(origins), origins[0], origins[-1],
    )

    level = config.interval_level
    mu = np.maximum(errors["predicted"].to_numpy(), 1e-9)
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    report(
        f"Poisson {level:.0%}",
        score(errors, poisson.ppf(lo_q, mu), poisson.ppf(hi_q, mu)),
        config.freq,
    )
    lower, upper = calibration.bounds(errors["predicted"].to_numpy(), errors["step"].to_numpy())
    report(f"Calibrated {level:.0%}", score(errors, lower, upper), config.freq)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
