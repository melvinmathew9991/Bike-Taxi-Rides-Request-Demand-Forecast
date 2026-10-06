#!/usr/bin/env python
"""
Measure how fast a frozen model decays, to set the retraining cadence.

This is the experiment behind the single most operationally consequential number
in the project: how often the model must be retrained. Erring short costs
compute; erring long serves forecasts that a free seasonal-naive baseline beats.

The figures in `docs/MODEL_CARD.md` came from an ad-hoc run that was never
committed, so they could not be reproduced or re-measured when the feature set
changed. This script is that experiment, written down.

Two changes from the original:

**Several origins, not one.** The published curve froze a model at a single date
(2020-12-01) and read the decay off that. On a series where demand grew 5.2x in a
year, one origin measures the quarter you happened to pick as much as it measures
decay - the same objection this project raises against single train/test splits.
Each origin is measured independently and the spread is reported.

**Recursive scoring, because that is how the pipeline serves.** A frozen model in
production still sees observed demand arrive; it simply is not retrained. So the
lags are read from real observations up to the forecast origin, and only the
within-horizon steps consume the model's own predictions. One-step scores are
reported alongside for comparison with the published table.

Usage:
    python experiments/measure_staleness.py \\
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
import xgboost as xgb
from joblib import load

from ML_Pipeline.config import PipelineConfig
from ML_Pipeline.modeling.evaluation import ModelEvaluator
from ML_Pipeline.modeling.features import (
    CLUSTER_COL,
    TARGET_COL,
    TS_COL,
    ModelBundle,
    add_calendar_features,
    add_lag_features,
    attach_cluster_centroids,
    build_feature_names,
)
from ML_Pipeline.modeling.forecast import PREDICTION_COL, forecast_recursive
from ML_Pipeline.utils import read_csv_any

logger = logging.getLogger("measure_staleness")

WEEK_INTERVALS = 336  # 30-minute intervals in a week
DAY_INTERVALS = 48

#: Weeks of staleness to probe. Matches the published table so the two are
#: directly comparable.
DEFAULT_OFFSETS = (1, 2, 3, 4, 5, 6, 8, 10, 13)


def freeze_model(
    train: pd.DataFrame, features: list[str], config: PipelineConfig
) -> ModelBundle:
    """
    Fit the model once and never update it again - the point of the experiment.

    No early stopping: a frozen model has no future validation tail to stop
    against, and holding the tree count fixed keeps every origin comparable.
    """
    params = {**config.xgb_params}
    params.pop("early_stopping_rounds", None)
    model = xgb.XGBRegressor(**params)
    model.fit(train[features], train[TARGET_COL], verbose=False)
    return ModelBundle(
        model=model,
        feature_names=features,
        uses_lags=True,
        lags=tuple(config.lag_features),
        rolling_window=config.rolling_window,
        freq=config.freq,
    )


def score_week(
    bundle: ModelBundle,
    lagged: pd.DataFrame,
    panel: pd.DataFrame,
    week_start: pd.Timestamp,
    centroids: np.ndarray,
) -> dict[str, float] | None:
    """
    Score a frozen model on one week, one step ahead and recursively.

    One step ahead uses the week's true observed lags. Recursive forecasts the
    first 24 hours of the week from observations up to `week_start`, compounding
    its own errors within the horizon - which is how the pipeline serves.
    """
    week_end = week_start + pd.Timedelta(weeks=1)
    window = lagged[(lagged[TS_COL] >= week_start) & (lagged[TS_COL] < week_end)]
    if window.empty:
        return None

    actual = window[TARGET_COL].to_numpy(dtype="float64")
    naive = window["_naive"].to_numpy(dtype="float64")
    if not np.isfinite(naive).any():
        return None

    one_step = np.clip(bundle.predict(window), 0, None)
    result = {
        "one_step_rmse": float(np.sqrt(np.mean((actual - one_step) ** 2))),
        "one_step_mase": ModelEvaluator.mase(actual, one_step, naive),
        "one_step_level_ratio": float(one_step.mean() / actual.mean())
        if actual.mean()
        else float("nan"),
        "mean_actual": float(actual.mean()),
    }

    horizon = pd.date_range(week_start, periods=DAY_INTERVALS, freq=bundle.freq)
    history = panel[panel[TS_COL] < week_start]
    try:
        forecast = forecast_recursive(
            bundle, history, horizon, centroids=centroids
        )
    except ValueError as exc:
        logger.debug("recursive skipped at %s: %s", week_start, exc)
        return result

    merged = (
        panel[(panel[TS_COL] >= horizon[0]) & (panel[TS_COL] <= horizon[-1])]
        .merge(
            forecast[[TS_COL, CLUSTER_COL, PREDICTION_COL]],
            on=[TS_COL, CLUSTER_COL], how="inner",
        )
        .merge(
            lagged[[TS_COL, CLUSTER_COL, "_naive"]],
            on=[TS_COL, CLUSTER_COL], how="left",
        )
    )
    if merged.empty:
        return result

    a = merged[TARGET_COL].to_numpy(dtype="float64")
    p = np.clip(merged[PREDICTION_COL].to_numpy(dtype="float64"), 0, None)
    result.update(
        {
            "recursive_rmse": float(np.sqrt(np.mean((a - p) ** 2))),
            "recursive_mase": ModelEvaluator.mase(
                a, p, merged["_naive"].to_numpy(dtype="float64")
            ),
            "recursive_level_ratio": float(p.mean() / a.mean())
            if a.mean()
            else float("nan"),
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Data_Prepared grid")
    parser.add_argument("--cluster-model", required=True)
    parser.add_argument(
        "--origins", default="2020-09-01,2020-10-15,2020-12-01",
        help="Freeze dates, comma-separated. The last is the one the published "
             "table used.",
    )
    parser.add_argument(
        "--offsets", default=",".join(str(x) for x in DEFAULT_OFFSETS),
        help="Weeks of staleness to probe.",
    )
    parser.add_argument("--out", default="output/staleness.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logging.getLogger("ML_Pipeline.modeling.forecast").setLevel(logging.WARNING)

    config = PipelineConfig()
    offsets = [int(x) for x in args.offsets.split(",")]
    origins = [pd.Timestamp(x) for x in args.origins.split(",")]

    df = read_csv_any(args.data)
    df[TS_COL] = pd.to_datetime(df[TS_COL])
    centroids = np.asarray(load(args.cluster_model).cluster_centers_)

    panel = attach_cluster_centroids(add_calendar_features(df, TS_COL), centroids)
    features = build_feature_names(
        use_lags=True,
        lags=config.lag_features,
        cluster_features=("cluster_lat", "cluster_lng"),
    )
    lagged = add_lag_features(
        panel, lags=config.lag_features, rolling_window=config.rolling_window
    )
    lagged = lagged.sort_values([CLUSTER_COL, TS_COL]).reset_index(drop=True)
    lagged["_naive"] = ModelEvaluator.seasonal_naive_baseline(
        lagged, season_length=WEEK_INTERVALS
    ).to_numpy()

    logger.info(
        "Lags %s | %d features | panel %s | lagged %s",
        tuple(config.lag_features), len(features), panel.shape, lagged.shape,
    )

    rows = []
    for origin in origins:
        train = lagged[lagged[TS_COL] < origin]
        if train.empty:
            logger.warning("Origin %s has no training data; skipping.", origin.date())
            continue
        logger.info("=" * 70)
        logger.info(
            "Freezing a model at %s (%d training rows, mean demand %.2f)",
            origin.date(), len(train), train[TARGET_COL].mean(),
        )
        bundle = freeze_model(train, features, config)

        for weeks in offsets:
            week_start = origin + pd.Timedelta(weeks=weeks - 1)
            scores = score_week(bundle, lagged, panel, week_start, centroids)
            if scores is None:
                logger.info("  %2d weeks stale: no data", weeks)
                continue
            rows.append({"origin": origin.date(), "weeks_stale": weeks, **scores})
            logger.info(
                "  %2d weeks stale | one-step MASE %.3f level %.2f | "
                "recursive MASE %s level %s | mean actual %.2f",
                weeks, scores["one_step_mase"], scores["one_step_level_ratio"],
                f"{scores.get('recursive_mase', float('nan')):.3f}",
                f"{scores.get('recursive_level_ratio', float('nan')):.2f}",
                scores["mean_actual"],
            )

    if not rows:
        logger.error("Nothing measured.")
        return 1

    frame = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    summary = frame.groupby("weeks_stale").agg(
        origins=("origin", "nunique"),
        one_step_mase=("one_step_mase", "mean"),
        one_step_mase_worst=("one_step_mase", "max"),
        recursive_mase=("recursive_mase", "mean"),
        recursive_mase_worst=("recursive_mase", "max"),
        level_ratio=("one_step_level_ratio", "mean"),
    )
    print("\n" + "=" * 78)
    print("MODEL DECAY BY WEEKS SINCE TRAINING  (MASE < 1 beats seasonal-naive)")
    print("=" * 78)
    print(summary.round(3).to_string())

    beaten = summary[summary["one_step_mase"] >= 1.0]
    if len(beaten):
        print(
            f"\nFirst week where the mean one-step MASE reaches 1.0: "
            f"{int(beaten.index[0])}"
        )
    else:
        print(
            f"\nThe model still beats seasonal-naive at {int(summary.index.max())} "
            "weeks stale; it never reaches parity within the window probed."
        )
    worst = summary[summary["one_step_mase_worst"] >= 1.0]
    if len(worst):
        print(
            f"First week where the WORST origin reaches 1.0: {int(worst.index[0])} "
            "- the cadence should follow this, not the mean."
        )
    print(f"\nPer-origin detail written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
