"""
Training stage: demand grid -> two fitted model bundles.

Two variants are trained, as before:

* **without lag** - calendar and geography only. Usable for any future interval,
  because it needs no recent history.
* **with lag** - adds the previous intervals' demand and a rolling mean. More
  accurate one step out, but must be applied recursively over a horizon, so its
  errors compound.

Notable corrections to the previous version:

* The train/test split is chronological rather than day-of-month (see
  `splitting.py`).
* Lag features are no longer computed and then discarded. The old
  `train_test_data_prep` built `lag_1/2/3` and `rolling_mean`, then excluded them
  all from `feature_cols` - so the work was wasted, and the `dropna()` that
  existed only to clear those lags silently deleted the first three intervals of
  every cluster from the *without-lag* training set for no reason.
* The lag model is additionally scored by recursive backtest, which is the only
  honest measure of it: scoring one-step-ahead predictions against observed lags
  hands the model ground truth at every step and flatters it badly.
* Cluster identity enters as centroid coordinates rather than as a raw integer
  label (see `features.attach_cluster_centroids`).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ML_Pipeline.evaluation import ModelEvaluator
from ML_Pipeline.features import (
    CLUSTER_COL,
    TARGET_COL,
    TS_COL,
    ModelBundle,
    add_calendar_features,
    add_lag_features,
    attach_cluster_centroids,
    build_feature_names,
)
from ML_Pipeline.forecast import PREDICTION_COL, backtest_recursive
from ML_Pipeline.splitting import chronological_split, train_validation_split
from ML_Pipeline.xgb_model import train_xgb

logger = logging.getLogger(__name__)


def _prepare_panel(
    df: pd.DataFrame, centroids: np.ndarray | None, use_centroids: bool
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Ensure calendar features and cluster encoding are present."""
    panel = df.copy()
    panel[TS_COL] = pd.to_datetime(panel[TS_COL])
    panel[TARGET_COL] = pd.to_numeric(panel[TARGET_COL], errors="coerce")
    panel = panel.dropna(subset=[TARGET_COL])

    if not {"hour", "dayofweek"}.issubset(panel.columns):
        panel = add_calendar_features(panel, TS_COL)

    if use_centroids and centroids is not None:
        panel = attach_cluster_centroids(panel, centroids)
        cluster_features = ("cluster_lat", "cluster_lng")
        logger.info("Encoding clusters by centroid coordinates.")
    else:
        cluster_features = (CLUSTER_COL,)
        if use_centroids:
            logger.warning(
                "use_cluster_centroids is set but no centroids were supplied; "
                "falling back to the raw integer label, which a tree model will "
                "split on as if it were ordinal."
            )
    return panel, cluster_features


def model_training(
    df: pd.DataFrame,
    without_lag_model_path: str,
    with_lag_model_path: str,
    *,
    config: Any = None,
    centroids: np.ndarray | None = None,
) -> dict[str, ModelBundle]:
    """
    Train both model variants and persist them.

    Args:
        df: Demand grid `[ts, pickup_cluster, request_count, ...]`.
        without_lag_model_path: Destination for the lag-free bundle.
        with_lag_model_path: Destination for the lag-using bundle.
        config: `PipelineConfig`. Defaults are used when omitted.
        centroids: `(n_clusters, 2)` cluster centroids, for geographic encoding.

    Returns:
        `{"without_lag": ModelBundle, "with_lag": ModelBundle}`.
    """
    from ML_Pipeline.config import PipelineConfig

    config = config or PipelineConfig()
    started = datetime.now()

    panel, cluster_features = _prepare_panel(
        df, centroids, getattr(config, "use_cluster_centroids", True)
    )

    # ---------------- Model 1: no lag features ----------------
    logger.info("=" * 60)
    logger.info("Training model WITHOUT lag features")
    logger.info("=" * 60)

    feats_nolag = build_feature_names(
        use_lags=False, cluster_features=cluster_features
    )
    split = chronological_split(panel, test_fraction=config.test_fraction)
    inner = train_validation_split(
        split.train, validation_fraction=config.validation_fraction
    )

    bundle_nolag = train_xgb(
        inner.train[feats_nolag], inner.train[TARGET_COL],
        X_valid=inner.test[feats_nolag], y_valid=inner.test[TARGET_COL],
        X_test=split.test[feats_nolag], y_test=split.test[TARGET_COL],
        params=config.xgb_params,
        early_stopping_rounds=config.early_stopping_rounds,
        feature_names=feats_nolag,
        uses_lags=False,
        freq=config.freq,
        notes=f"Chronological split at {split.split_at}.",
    )
    bundle_nolag.save(without_lag_model_path)

    # ---------------- Model 2: with lag features ----------------
    logger.info("=" * 60)
    logger.info("Training model WITH lag features")
    logger.info("=" * 60)

    lagged = add_lag_features(
        panel,
        lags=config.lag_features,
        rolling_window=config.rolling_window,
    )
    feats_lag = build_feature_names(
        use_lags=True, lags=config.lag_features, cluster_features=cluster_features
    )
    split_lag = chronological_split(lagged, test_fraction=config.test_fraction)
    inner_lag = train_validation_split(
        split_lag.train, validation_fraction=config.validation_fraction
    )

    bundle_lag = train_xgb(
        inner_lag.train[feats_lag], inner_lag.train[TARGET_COL],
        X_valid=inner_lag.test[feats_lag], y_valid=inner_lag.test[TARGET_COL],
        X_test=split_lag.test[feats_lag], y_test=split_lag.test[TARGET_COL],
        params=config.xgb_params,
        early_stopping_rounds=config.early_stopping_rounds,
        feature_names=feats_lag,
        uses_lags=True,
        lags=tuple(config.lag_features),
        rolling_window=config.rolling_window,
        freq=config.freq,
        notes=f"Chronological split at {split_lag.split_at}.",
    )

    # Honest evaluation of the lag model: let it compound its own errors.
    horizon_steps = _default_horizon(config)
    try:
        backtest = backtest_recursive(
            bundle_lag, panel, horizon_steps=horizon_steps, centroids=centroids
            if "cluster_lat" in feats_lag else None,
        )
        recursive_metrics = ModelEvaluator.calculate_metrics(
            backtest[TARGET_COL], backtest[PREDICTION_COL]
        )
        bundle_lag.metrics.update(
            {f"recursive_{k}": float(v) for k, v in recursive_metrics.items()}
        )
        logger.info(
            "Recursive backtest over %d steps: RMSE %.4f (one-step test RMSE was "
            "%.4f - the gap is the cost of compounding its own errors).",
            horizon_steps,
            recursive_metrics["rmse"],
            bundle_lag.metrics.get("test_rmse", float("nan")),
        )
    except ValueError as exc:
        logger.warning("Recursive backtest skipped: %s", exc)

    bundle_lag.save(with_lag_model_path)

    logger.info("Total training time: %s", datetime.now() - started)
    _log_comparison(bundle_nolag, bundle_lag)
    return {"without_lag": bundle_nolag, "with_lag": bundle_lag}


def _default_horizon(config: Any) -> int:
    """Forecast horizon in intervals; defaults to one day."""
    if getattr(config, "horizon_steps", None):
        return int(config.horizon_steps)
    step = pd.Timedelta(pd.tseries.frequencies.to_offset(config.freq))
    return int(pd.Timedelta("1D") / step)


def _log_comparison(without_lag: ModelBundle, with_lag: ModelBundle) -> None:
    """Report both models side by side so the lag block's value is visible."""
    rows = []
    for name, bundle in (("without_lag", without_lag), ("with_lag", with_lag)):
        rows.append(
            {
                "model": name,
                "test_rmse": bundle.metrics.get("test_rmse"),
                "test_mae": bundle.metrics.get("test_mae"),
                "test_r2": bundle.metrics.get("test_r2"),
                "recursive_rmse": bundle.metrics.get("recursive_rmse"),
            }
        )
    logger.info("Model comparison:\n%s", pd.DataFrame(rows).to_string(index=False))
