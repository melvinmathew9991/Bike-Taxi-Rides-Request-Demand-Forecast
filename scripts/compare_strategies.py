#!/usr/bin/env python
"""
Compare modelling strategies across rolling origins.

Answers the question a single split cannot: does a strategy beat a
seasonal-naive baseline *consistently*, or did it get one lucky fortnight?

Each strategy is evaluated two ways:

  one-step   the model is handed true observed lags. Optimistic.
  recursive  the model consumes its own predictions over a horizon. This is how
             the pipeline serves, and it is where a trend-anchored model fails.

Usage:
    python scripts/compare_strategies.py --data output/Data_Prepared_<ver>.csv \
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
from joblib import load  # noqa: E402

from ML_Pipeline.features import (  # noqa: E402
    TARGET_COL,
    TS_COL,
    ModelBundle,
    add_calendar_features,
    add_lag_features,
    attach_cluster_centroids,
    build_feature_names,
)
from ML_Pipeline.validation import (  # noqa: E402
    rolling_origin_validate,
    rolling_origin_validate_recursive,
    summarise,
)

logger = logging.getLogger("compare_strategies")

LAGS = (1, 2, 3)
ROLLING_WINDOW = 3
FEATURES = build_feature_names(
    use_lags=True, lags=LAGS, cluster_features=("cluster_lat", "cluster_lng")
)
BASE_PARAMS = dict(
    max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    n_estimators=250, min_child_weight=5, random_state=42, n_jobs=-1,
)
RECENT_DAYS = 56  # 8 weeks


def _recent(train: pd.DataFrame, days: int | None) -> pd.DataFrame:
    if days is None:
        return train
    cutoff = train[TS_COL].max() - pd.Timedelta(days=days)
    return train[train[TS_COL] >= cutoff]


def make_one_step_strategy(*, ratio_target: bool, window_days: int | None):
    """Build a `(train, test) -> predictions` callable."""

    def strategy(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        tr = _recent(train, window_days).dropna(subset=FEATURES + [TARGET_COL])
        if tr.empty:
            return np.full(len(test), train[TARGET_COL].mean())

        if ratio_target:
            base_tr = tr["rolling_mean"].to_numpy() + 1.0
            model = xgb.XGBRegressor(objective="reg:squarederror", **BASE_PARAMS)
            model.fit(tr[FEATURES], tr[TARGET_COL].to_numpy() / base_tr)
            base_te = test["rolling_mean"].to_numpy() + 1.0
            return model.predict(test[FEATURES]) * base_te

        model = xgb.XGBRegressor(objective="count:poisson", **BASE_PARAMS)
        model.fit(tr[FEATURES], tr[TARGET_COL])
        return model.predict(test[FEATURES])

    return strategy


class RatioModel:
    """Wraps a ratio-target estimator so it behaves like a level predictor."""

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        base = np.asarray(X["rolling_mean"], dtype="float64") + 1.0
        return np.asarray(self.model.predict(X), dtype="float64") * base


def make_bundle_factory(*, ratio_target: bool, window_days: int | None):
    """Build a `train -> ModelBundle` callable for recursive evaluation."""

    def fit_bundle(train: pd.DataFrame) -> ModelBundle:
        tr = add_lag_features(
            _recent(train, window_days), lags=LAGS, rolling_window=ROLLING_WINDOW
        )
        if ratio_target:
            base = tr["rolling_mean"].to_numpy() + 1.0
            inner = xgb.XGBRegressor(objective="reg:squarederror", **BASE_PARAMS)
            inner.fit(tr[FEATURES], tr[TARGET_COL].to_numpy() / base)
            model = RatioModel(inner)
        else:
            model = xgb.XGBRegressor(objective="count:poisson", **BASE_PARAMS)
            model.fit(tr[FEATURES], tr[TARGET_COL])

        return ModelBundle(
            model=model, feature_names=FEATURES, uses_lags=True,
            lags=LAGS, rolling_window=ROLLING_WINDOW, freq="30min",
        )

    return fit_bundle


STRATEGIES = {
    "level, full history (current)": dict(ratio_target=False, window_days=None),
    "level, last 8 weeks": dict(ratio_target=False, window_days=RECENT_DAYS),
    "ratio, full history": dict(ratio_target=True, window_days=None),
    "ratio, last 8 weeks": dict(ratio_target=True, window_days=RECENT_DAYS),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Data_Prepared CSV (gzip)")
    parser.add_argument("--cluster-model", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-size", type=int, default=336, help="one-step window")
    parser.add_argument("--horizon", type=int, default=48, help="recursive horizon")
    parser.add_argument("--out", default="output/strategy_comparison.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    df = pd.read_csv(args.data, compression="gzip", low_memory=False)
    df[TS_COL] = pd.to_datetime(df[TS_COL])
    centroids = np.asarray(load(args.cluster_model).cluster_centers_)

    panel = attach_cluster_centroids(add_calendar_features(df, TS_COL), centroids)
    lagged = add_lag_features(panel, lags=LAGS, rolling_window=ROLLING_WINDOW)
    logger.info("Panel %s | lagged %s", panel.shape, lagged.shape)

    one_step, recursive = [], []
    for label, kwargs in STRATEGIES.items():
        logger.info("=" * 70)
        logger.info("ONE-STEP: %s", label)
        one_step.append(
            rolling_origin_validate(
                lagged, make_one_step_strategy(**kwargs),
                n_folds=args.folds, test_size=args.test_size, label=label,
            )
        )
        logger.info("RECURSIVE (%d-interval horizon): %s", args.horizon, label)
        recursive.append(
            rolling_origin_validate_recursive(
                panel, make_bundle_factory(**kwargs),
                n_folds=args.folds, horizon=args.horizon,
                centroids=centroids, label=label,
            )
        )

    print("\n" + "=" * 78)
    print("ONE STEP AHEAD  (MASE < 1 beats seasonal naive)")
    print("=" * 78)
    print(summarise(one_step).round(4).to_string())

    print("\n" + "=" * 78)
    print(f"RECURSIVE, {args.horizon}-INTERVAL HORIZON  (level_ratio 1.0 = right level)")
    print("=" * 78)
    print(summarise(recursive).round(4).to_string())

    combined = pd.concat(
        [pd.concat(one_step).assign(mode="one_step"),
         pd.concat(recursive).assign(mode="recursive")],
        ignore_index=True,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out, index=False)
    print(f"\nPer-fold detail written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
