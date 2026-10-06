#!/usr/bin/env python
"""
Measure whether weather and holidays improve the one-step forecast.

An upper-bound experiment, run before any of it goes into the pipeline. Each
variant is scored on the same rolling-origin folds:

    base      the shipped feature set
    past      + rain in the last 1 and 3 hours, temperature, humidity - all
              known at forecast time
    oracle    + rain during the interval being forecast. Not known at forecast
              time; this is the most weather could ever add, as if the model
              had a perfect weather forecast
    holiday   + a Karnataka public-holiday flag

If `oracle` does not beat `base`, weather is not worth adding. If only
`oracle` does, the gain depends on forecast weather, which cannot be measured
on 2020-21 data and would have to be stated as an upper bound.

Folds are spread from June 2020, so the test weeks cover the monsoon: the
default first origin, half-way through the series, would test mostly in the
dry season. Every variant is scored on the same rows, so the comparison is
paired.

Rainy intervals are those whose hour had >= 1 mm of rain. They are a few
percent of the test rows, so they are reported separately - the overall MASE
would hide an effect confined to them.

Usage:
    python experiments/fetch_weather.py
    python experiments/measure_weather.py \\
        --data output/Data_Prepared_<ver>.csv.gz \\
        --cluster-model output/pickup_cluster_model_<ver>.joblib
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import xgboost as xgb
from common import BASE_PARAMS, FEATURES, LAGS, ROLLING_WINDOW
from joblib import load

from ML_Pipeline.modeling.features import (
    TARGET_COL,
    TS_COL,
    add_calendar_features,
    add_lag_features,
    attach_cluster_centroids,
)
from ML_Pipeline.modeling.validation import rolling_origins
from ML_Pipeline.utils import read_csv_any

logger = logging.getLogger("measure_weather")

SEASON = 336
RAINY_MM = 1.0
PAST = ["rain_prev_1h", "rain_prev_3h", "temperature", "humidity"]
VARIANTS = {
    "base": FEATURES,
    "past": [*FEATURES, *PAST],
    "oracle": [*FEATURES, *PAST, "rain_now"],
    "holiday": [*FEATURES, "is_holiday"],
}


def add_weather(panel: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """
    Join hourly weather onto the half-hourly grid.

    Open-Meteo stamps precipitation at the END of the hour it fell in, so for an
    interval in hour H: rain during it is the value stamped H+1h, and the last
    complete hour before it is the value stamped H. Temperature and humidity
    are instantaneous, taken at H.
    """
    w = weather.set_index("ts").sort_index()
    precip = w["precipitation"]
    hour = panel[TS_COL].dt.floor("h")
    out = panel.copy()
    out["rain_now"] = precip.reindex(hour + pd.Timedelta(hours=1)).to_numpy()
    out["rain_prev_1h"] = precip.reindex(hour).to_numpy()
    out["rain_prev_3h"] = (
        precip.rolling(3, min_periods=1).sum().reindex(hour).to_numpy()
    )
    out["temperature"] = w["temperature_2m"].reindex(hour).to_numpy()
    out["humidity"] = w["relative_humidity_2m"].reindex(hour).to_numpy()

    days = panel[TS_COL].dt.normalize()
    calendar = holidays.India(subdiv="KA", years=sorted(days.dt.year.unique()))
    out["is_holiday"] = days.dt.date.map(lambda d: d in calendar).astype("int8")
    return out


def mase(actual, pred, naive) -> float:
    denom = np.mean(np.abs(actual - naive))
    return float(np.mean(np.abs(actual - pred)) / denom) if denom else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--cluster-model", required=True)
    parser.add_argument("--weather", default="data/weather_bengaluru.csv")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--first-origin", default="2020-06-08")
    parser.add_argument("--out", default="output/weather_comparison.csv")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    grid = read_csv_any(args.data)
    grid[TS_COL] = pd.to_datetime(grid[TS_COL])
    weather = pd.read_csv(args.weather, parse_dates=["ts"])
    centroids = np.asarray(load(args.cluster_model).cluster_centers_)

    panel = attach_cluster_centroids(add_calendar_features(grid, TS_COL), centroids)
    panel = add_weather(panel, weather)
    lagged = add_lag_features(panel, lags=LAGS, rolling_window=ROLLING_WINDOW)
    gaps = lagged[["rain_now", "rain_prev_3h", "temperature"]].isna().sum()
    if gaps.any():
        raise SystemExit(f"Weather does not cover the grid:\n{gaps[gaps > 0]}")

    stamps = pd.DatetimeIndex(sorted(lagged[TS_COL].unique()))
    min_train = int(np.searchsorted(stamps, pd.Timestamp(args.first_origin)))
    folds = rolling_origins(
        stamps, n_folds=args.folds, test_size=SEASON, min_train_size=min_train
    )

    rows = []
    for origin, end in folds:
        train = lagged[lagged[TS_COL] < origin]
        test = lagged[(lagged[TS_COL] >= origin) & (lagged[TS_COL] <= end)]
        actual = test[TARGET_COL].to_numpy(dtype="float64")
        naive = test[f"lag_{SEASON}"].to_numpy(dtype="float64")
        rainy = (test["rain_now"] >= RAINY_MM).to_numpy()
        for name, features in VARIANTS.items():
            model = xgb.XGBRegressor(objective="count:poisson", **BASE_PARAMS)
            model.fit(train[features], train[TARGET_COL])
            pred = model.predict(test[features])
            row = {
                "origin": origin.date(), "variant": name,
                "mase": mase(actual, pred, naive),
                "level_ratio": float(pred.mean() / actual.mean()),
                "rainy_share": float(rainy.mean()),
                "rainy_mase": mase(actual[rainy], pred[rainy], naive[rainy])
                if rainy.any() else float("nan"),
                "rainy_level_ratio": float(pred[rainy].mean() / actual[rainy].mean())
                if rainy.any() else float("nan"),
            }
            rows.append(row)
            logger.info(
                "%s %-8s MASE %.4f | rainy %4.1f%% MASE %s",
                origin.date(), name, row["mase"], row["rainy_share"] * 100,
                f"{row['rainy_mase']:.4f}" if rainy.any() else "n/a",
            )

    result = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)

    base = result[result["variant"] == "base"].set_index("origin")
    print("\n" + "=" * 78)
    print(f"ONE STEP AHEAD, {len(folds)} folds from {folds[0][0].date()} "
          f"to {folds[-1][1].date()}  (MASE < 1 beats seasonal-naive)")
    print("=" * 78)
    for name in VARIANTS:
        v = result[result["variant"] == name].set_index("origin")
        diff = v["mase"] - base["mase"]
        rainy_diff = (v["rainy_mase"] - base["rainy_mase"]).dropna()
        print(
            f"{name:8} MASE {v['mase'].mean():.4f} (std {v['mase'].std():.4f}) "
            f"vs base {diff.mean():+.4f}, better in {(diff < 0).sum()}/{len(diff)} | "
            f"rainy MASE {v['rainy_mase'].mean():.4f} vs base "
            f"{rainy_diff.mean():+.4f}, better in "
            f"{(rainy_diff < 0).sum()}/{len(rainy_diff)} | "
            f"rainy level {v['rainy_level_ratio'].mean():.3f}"
        )
    print(f"\nPer-fold detail written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
