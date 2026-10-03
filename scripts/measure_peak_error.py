#!/usr/bin/env python
"""
Measure where a stale model fails: the busiest cluster's peak.

The dashboard found the promoted model losing to seasonal-naive on the busiest
cluster over the last eight weeks (MASE 1.003), while every other cluster beat
it. Two explanations predict that, and they call for different fixes:

* **staleness** - the saved model never saw the test window, the newest fifth of
  the timeline, so it was ~10 weeks behind its data on the day it was trained;
* **a structural ceiling** - a tree ensemble cannot predict above the range its
  leaves were fitted on, so on a growing series new peaks are always clipped.

This separates them by walking forward a week at a time over the last `--weeks`
weeks and scoring, on the same rows:

    frozen     fitted once on everything before the pipeline's chronological
               split and never refitted - what the pipeline used to promote
    level      the shipped target, refitted on all data before each week
    ratio_rm   ratio target over `rolling_mean`, refitted weekly
    ratio_336  ratio target over `lag_336`, refitted weekly

If weekly refitting alone closes the gap, staleness was the cause. If only the
ratio targets close it, the ceiling was. One-step scoring throughout: the lags
are observed values, which is how the deploy gate measures too.

Usage:
    python scripts/measure_peak_error.py \\
        --data output/Data_Prepared_<ver>.csv.gz \\
        --cluster-model output/pickup_cluster_model_<ver>.joblib
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import xgboost as xgb  # noqa: E402
from compare_strategies import BASE_PARAMS, FEATURES, LAGS, ROLLING_WINDOW  # noqa: E402
from joblib import load  # noqa: E402

from ML_Pipeline.config import PipelineConfig  # noqa: E402
from ML_Pipeline.features import (  # noqa: E402
    CLUSTER_COL,
    TARGET_COL,
    TS_COL,
    add_calendar_features,
    add_lag_features,
    attach_cluster_centroids,
)
from ML_Pipeline.splitting import chronological_split  # noqa: E402
from ML_Pipeline.utils import read_csv_any  # noqa: E402

logger = logging.getLogger("measure_peak_error")

WEEK_INTERVALS = 336
STRATEGIES = ("frozen", "level", "ratio_rm", "ratio_336")


def fit_predict(kind: str, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Fit one strategy on `train` and predict `test` one step ahead."""
    if kind in ("frozen", "level"):
        model = xgb.XGBRegressor(objective="count:poisson", **BASE_PARAMS)
        model.fit(train[FEATURES], train[TARGET_COL])
        return model.predict(test[FEATURES])
    base = "rolling_mean" if kind == "ratio_rm" else f"lag_{WEEK_INTERVALS}"
    model = xgb.XGBRegressor(objective="reg:squarederror", **BASE_PARAMS)
    model.fit(train[FEATURES], train[TARGET_COL] / (train[base] + 1.0))
    ratio = np.clip(model.predict(test[FEATURES]), 0, None)
    return ratio * (test[base].to_numpy() + 1.0)


def mase(frame: pd.DataFrame, col: str) -> float:
    actual = frame[TARGET_COL]
    return float(
        (actual - frame[col]).abs().mean() / (actual - frame["naive"]).abs().mean()
    )


def level_ratio(frame: pd.DataFrame, col: str) -> float:
    return float(frame[col].mean() / frame[TARGET_COL].mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Data_Prepared CSV (gzip)")
    parser.add_argument("--cluster-model", required=True)
    parser.add_argument("--weeks", type=int, default=8)
    parser.add_argument(
        "--cluster", type=int, default=None,
        help="Cluster to examine. Defaults to the busiest over the window.",
    )
    parser.add_argument("--out", default="output/peak_error.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    if WEEK_INTERVALS not in LAGS:
        raise SystemExit(f"ratio_336 needs lag_{WEEK_INTERVALS}; lags are {LAGS}")

    df = read_csv_any(args.data)
    df[TS_COL] = pd.to_datetime(df[TS_COL])
    centroids = np.asarray(load(args.cluster_model).cluster_centers_)
    panel = attach_cluster_centroids(add_calendar_features(df, TS_COL), centroids)
    lagged = (
        add_lag_features(panel, lags=LAGS, rolling_window=ROLLING_WINDOW)
        .sort_values([CLUSTER_COL, TS_COL])
        .reset_index(drop=True)
    )
    lagged["naive"] = lagged[f"lag_{WEEK_INTERVALS}"]

    end = lagged[TS_COL].max()
    window_start = end - pd.Timedelta(weeks=args.weeks)
    split_at = chronological_split(
        lagged, test_fraction=PipelineConfig().test_fraction
    ).split_at
    logger.info("Window %s to %s; frozen model fitted before %s", window_start, end, split_at)

    frozen_train = lagged[lagged[TS_COL] < split_at]
    window = lagged[lagged[TS_COL] > window_start].copy()
    window["frozen"] = fit_predict("frozen", frozen_train, window)

    origins = pd.date_range(window_start, end, freq="7D", inclusive="left")
    for origin in origins:
        train = lagged[lagged[TS_COL] <= origin]
        mask = (window[TS_COL] > origin) & (window[TS_COL] <= origin + pd.Timedelta(weeks=1))
        for kind in STRATEGIES[1:]:
            window.loc[mask, kind] = fit_predict(kind, train, window[mask])
        logger.info("Origin %s scored", origin)

    cluster = args.cluster
    if cluster is None:
        cluster = int(window.groupby(CLUSTER_COL)[TARGET_COL].mean().idxmax())
    focus = window[window[CLUSTER_COL] == cluster]
    peak_hours = focus.groupby("hour")[TARGET_COL].mean().nlargest(3).index
    peak = focus[focus["hour"].isin(peak_hours)]
    above = focus[
        focus[TARGET_COL] > frozen_train.loc[frozen_train[CLUSTER_COL] == cluster, TARGET_COL].max()
    ]

    rows = []
    for kind in STRATEGIES:
        per_cluster = window.groupby(CLUSTER_COL).apply(lambda g, k=kind: mase(g, k))
        rows.append({
            "strategy": kind,
            "mase": mase(window, kind),
            "clusters_losing": int((per_cluster >= 1).sum()),
            "worst_cluster_mase": float(per_cluster.max()),
            "focus_mase": mase(focus, kind),
            "focus_level_ratio": level_ratio(focus, kind),
            "focus_peak_level_ratio": level_ratio(peak, kind),
            "above_training_max_level_ratio": level_ratio(above, kind)
            if len(above) else float("nan"),
            "max_prediction": float(window[kind].max()),
        })
    result = pd.DataFrame(rows).set_index("strategy")

    print("\n" + "=" * 78)
    print(
        f"LAST {args.weeks} WEEKS, ONE STEP AHEAD. Focus cluster {cluster} "
        f"(busiest), peak hours {sorted(int(h) for h in peak_hours)}, "
        f"{len(above)} intervals above its frozen-model training max."
    )
    print("=" * 78)
    print(result.round(3).to_string())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out)
    print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
