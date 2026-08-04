"""
Rolling-origin validation for forecasting models.

A single train/test split gives one number, and on a non-stationary series that
number says as much about which fortnight you happened to hold out as about the
model. On the reference dataset demand grew 5.2x across the year, so a split in
May and a split in February are effectively different problems. Any claim that
one modelling approach beats another has to survive several origins before it is
worth acting on.

This module evaluates a *strategy* - a function that fits a model given training
data - at several successive cut points, always training on the past and scoring
on the future. Two evaluation modes:

* **One step ahead** - the model is given true observed lags at every point.
  This is the optimistic case and is what most published numbers report.
* **Recursive over a horizon** - the model consumes its own predictions. This is
  how the pipeline actually serves, and on a trending series it is far harsher.

Every fold is also scored against a seasonal-naive baseline, because the
decision that matters is not "which model has the lowest RMSE" but "does any of
this beat same-time-last-week".
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ML_Pipeline.evaluation import ModelEvaluator
from ML_Pipeline.features import CLUSTER_COL, TARGET_COL, TS_COL, ModelBundle
from ML_Pipeline.forecast import PREDICTION_COL, forecast_recursive

logger = logging.getLogger(__name__)

#: 30-minute intervals in one week - the default seasonal period for the naive
#: baseline (same weekday, same time of day).
WEEKLY_SEASON_30MIN = 336


@dataclass
class FoldResult:
    """Scores for one origin."""

    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    metrics: dict[str, float] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "n_train": self.n_train,
            "n_test": self.n_test,
            **self.metrics,
        }


def rolling_origins(
    stamps: Sequence[pd.Timestamp],
    *,
    n_folds: int = 5,
    test_size: int,
    min_train_size: int | None = None,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Cut points for rolling-origin validation.

    Each fold trains on everything before its origin and tests on the next
    `test_size` intervals. Folds advance forward in time, so later folds have
    more training data - which mirrors how a deployed model is retrained.

    Args:
        stamps: Sorted distinct timestamps of the panel.
        n_folds: Number of origins.
        test_size: Intervals per test window.
        min_train_size: Minimum intervals before the first origin. Defaults to
            half the series.

    Returns:
        `(origin, test_end)` pairs, oldest first.
    """
    stamps = pd.DatetimeIndex(stamps).sort_values()
    n = len(stamps)
    if test_size < 1:
        raise ValueError(f"test_size must be >= 1, got {test_size}")

    min_train_size = min_train_size if min_train_size is not None else n // 2
    usable = n - min_train_size - test_size
    if usable < 0:
        raise ValueError(
            f"Series has {n} intervals; not enough for {min_train_size} training "
            f"intervals plus a {test_size}-interval test window."
        )

    if n_folds == 1:
        starts = [n - test_size]
    else:
        step = usable // (n_folds - 1) if n_folds > 1 else 0
        if step == 0:
            logger.warning(
                "Series too short for %d distinct origins; folds will overlap.", n_folds
            )
            step = 1
        starts = [min_train_size + i * step for i in range(n_folds)]
        starts = [min(s, n - test_size) for s in starts]

    folds = []
    for s in sorted(set(starts)):
        folds.append((stamps[s], stamps[min(s + test_size, n) - 1]))
    return folds


def _seasonal_naive(
    panel: pd.DataFrame, season_length: int
) -> pd.Series:
    ordered = panel.sort_values([CLUSTER_COL, TS_COL])
    return ordered.groupby(CLUSTER_COL)[TARGET_COL].shift(season_length)


def rolling_origin_validate(
    panel: pd.DataFrame,
    fit_predict: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
    *,
    n_folds: int = 5,
    test_size: int = WEEKLY_SEASON_30MIN,
    min_train_size: int | None = None,
    season_length: int = WEEKLY_SEASON_30MIN,
    label: str = "model",
) -> pd.DataFrame:
    """
    Evaluate a one-step-ahead strategy across several origins.

    Args:
        panel: Full observed panel, with whatever features the strategy needs.
        fit_predict: `(train_df, test_df) -> predictions for test_df`. Receives
            copies; must not rely on seeing the test target.
        n_folds: Number of origins.
        test_size: Intervals per test window (default one week at 30 min).
        season_length: Seasonal period for the naive baseline.
        label: Name recorded in the results.

    Returns:
        One row per fold, with model and baseline scores and MASE.
    """
    working = panel.copy()
    working[TS_COL] = pd.to_datetime(working[TS_COL])
    stamps = np.sort(working[TS_COL].unique())

    folds = rolling_origins(
        stamps, n_folds=n_folds, test_size=test_size, min_train_size=min_train_size
    )
    logger.info("Rolling-origin validation of %r across %d folds", label, len(folds))

    naive_all = _seasonal_naive(working, season_length)
    working = working.sort_values([CLUSTER_COL, TS_COL]).copy()
    working["_naive"] = naive_all.to_numpy()

    results: list[FoldResult] = []
    for i, (origin, test_end) in enumerate(folds, start=1):
        train = working[working[TS_COL] < origin]
        test = working[(working[TS_COL] >= origin) & (working[TS_COL] <= test_end)]
        if train.empty or test.empty:
            logger.warning("Fold %d is empty; skipping.", i)
            continue

        preds = np.clip(np.asarray(fit_predict(train.copy(), test.copy()), dtype=float), 0, None)
        if len(preds) != len(test):
            raise ValueError(
                f"Fold {i}: strategy returned {len(preds)} predictions for "
                f"{len(test)} test rows."
            )

        actual = test[TARGET_COL].to_numpy(dtype=float)
        naive = test["_naive"].to_numpy(dtype=float)

        metrics = {
            "rmse": float(np.sqrt(np.mean((actual - preds) ** 2))),
            "mae": float(np.mean(np.abs(actual - preds))),
            "mase": ModelEvaluator.mase(actual, preds, naive),
            "naive_rmse": float(
                np.sqrt(np.nanmean((actual - naive) ** 2))
            ) if np.isfinite(naive).any() else float("nan"),
            "naive_mae": float(np.nanmean(np.abs(actual - naive)))
            if np.isfinite(naive).any() else float("nan"),
            "mean_actual": float(actual.mean()),
            "mean_pred": float(preds.mean()),
        }
        results.append(
            FoldResult(
                fold=i,
                train_start=train[TS_COL].min(), train_end=train[TS_COL].max(),
                test_start=test[TS_COL].min(), test_end=test[TS_COL].max(),
                n_train=len(train), n_test=len(test), metrics=metrics,
            )
        )
        logger.info(
            "  fold %d | train->%s | test %s..%s | RMSE %.4f | MASE %.3f",
            i, train[TS_COL].max().date(), test[TS_COL].min().date(),
            test[TS_COL].max().date(), metrics["rmse"], metrics["mase"],
        )

    frame = pd.DataFrame([r.as_row() for r in results])
    frame.insert(0, "strategy", label)
    return frame


def rolling_origin_validate_recursive(
    panel: pd.DataFrame,
    fit_bundle: Callable[[pd.DataFrame], ModelBundle],
    *,
    n_folds: int = 5,
    horizon: int = 48,
    min_train_size: int | None = None,
    season_length: int = WEEKLY_SEASON_30MIN,
    centroids: np.ndarray | None = None,
    label: str = "model",
) -> pd.DataFrame:
    """
    Evaluate a strategy in the mode the pipeline actually serves.

    The model forecasts `horizon` intervals from its own predictions, so error
    compounds. This is the number that should drive a deployment decision: a
    model can look competitive one step ahead and still collapse over a day.

    Args:
        panel: Observed panel `[ts, pickup_cluster, request_count, ...]`.
        fit_bundle: `train_df -> ModelBundle` (must set `uses_lags=True`).
        horizon: Intervals to forecast per fold.
        centroids: Cluster centroids, if the model uses them.
    """
    working = panel.copy()
    working[TS_COL] = pd.to_datetime(working[TS_COL])
    stamps = np.sort(working[TS_COL].unique())
    folds = rolling_origins(
        stamps, n_folds=n_folds, test_size=horizon, min_train_size=min_train_size
    )
    logger.info(
        "Recursive rolling-origin validation of %r: %d folds, %d-interval horizon",
        label, len(folds), horizon,
    )

    working = working.sort_values([CLUSTER_COL, TS_COL]).copy()
    working["_naive"] = _seasonal_naive(working, season_length).to_numpy()

    results: list[FoldResult] = []
    for i, (origin, test_end) in enumerate(folds, start=1):
        train = working[working[TS_COL] < origin]
        test = working[(working[TS_COL] >= origin) & (working[TS_COL] <= test_end)]
        if train.empty or test.empty:
            continue

        bundle = fit_bundle(train.copy())
        horizon_index = pd.date_range(origin, periods=horizon, freq=bundle.freq)
        try:
            forecast = forecast_recursive(
                bundle, train, horizon_index, centroids=centroids
            )
        except ValueError as exc:
            logger.warning("Fold %d skipped: %s", i, exc)
            continue

        merged = test.merge(
            forecast[[TS_COL, CLUSTER_COL, PREDICTION_COL]],
            on=[TS_COL, CLUSTER_COL], how="inner",
        )
        actual = merged[TARGET_COL].to_numpy(dtype=float)
        preds = np.clip(merged[PREDICTION_COL].to_numpy(dtype=float), 0, None)
        naive = merged["_naive"].to_numpy(dtype=float)

        metrics = {
            "rmse": float(np.sqrt(np.mean((actual - preds) ** 2))),
            "mae": float(np.mean(np.abs(actual - preds))),
            "mase": ModelEvaluator.mase(actual, preds, naive),
            "naive_rmse": float(np.sqrt(np.nanmean((actual - naive) ** 2))),
            "mean_actual": float(actual.mean()),
            "mean_pred": float(preds.mean()),
            # Ratio of predicted to actual total demand. A recursive forecast
            # that decays toward the training level shows up here long before
            # RMSE makes it obvious.
            "level_ratio": float(preds.mean() / actual.mean()) if actual.mean() else float("nan"),
        }
        results.append(
            FoldResult(
                fold=i,
                train_start=train[TS_COL].min(), train_end=train[TS_COL].max(),
                test_start=merged[TS_COL].min(), test_end=merged[TS_COL].max(),
                n_train=len(train), n_test=len(merged), metrics=metrics,
            )
        )
        logger.info(
            "  fold %d | test %s | RMSE %.4f | MASE %.3f | level ratio %.2f",
            i, merged[TS_COL].min().date(), metrics["rmse"],
            metrics["mase"], metrics["level_ratio"],
        )

    frame = pd.DataFrame([r.as_row() for r in results])
    frame.insert(0, "strategy", label)
    return frame


def summarise(results: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate fold results per strategy.

    Reports the mean and the spread. On a non-stationary series the spread is
    the point: a strategy whose MASE swings either side of 1.0 across folds has
    not been shown to beat the baseline, whatever its average says.
    """
    combined = pd.concat(list(results), ignore_index=True)
    grouped = combined.groupby("strategy")
    summary = pd.DataFrame(
        {
            "folds": grouped.size(),
            "rmse_mean": grouped["rmse"].mean(),
            "rmse_std": grouped["rmse"].std(),
            "mase_mean": grouped["mase"].mean(),
            "mase_std": grouped["mase"].std(),
            "mase_worst": grouped["mase"].max(),
            "folds_beating_naive": grouped["mase"].apply(lambda s: int((s < 1).sum())),
        }
    )
    if "level_ratio" in combined.columns:
        summary["level_ratio_mean"] = grouped["level_ratio"].mean()
    return summary.sort_values("mase_mean")
