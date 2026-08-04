"""
Evaluation metrics and prediction validation.

Metric choice matters more than usual here, because the target is an
*intermittent count*: demand per cluster per half hour has a median of 0 and a
mean below 1. Two consequences drive what this module reports.

Percentage errors are unusable. The previous implementation computed
`mean(|y - yhat| / (y + 1e-8)) * 100`, which on a zero actual divides by 1e-8 and
returns ~1e10. That value was being reported as a percentage and, once the model
registry started recording metrics, persisted alongside the model. It is removed;
`mape` is still computed but only over non-zero actuals, and is reported as
`NaN` with a logged warning when too few remain to be meaningful.

R^2 flatters a sparse target. Predicting the mean everywhere already scores 0,
and a model can post a respectable R^2 while being useless for dispatch. The
honest question is whether the model beats the obvious baselines, so
`compare_to_baselines` scores a seasonal-naive forecast (same interval, previous
week) and a per-cluster historical mean alongside it, and reports MASE - the
error scaled by the naive forecast's error, where < 1 means "better than naive".
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

logger = logging.getLogger(__name__)

#: Below this many non-zero actuals, a percentage error is noise.
MIN_NONZERO_FOR_MAPE = 30


class ModelEvaluator:
    """Metrics, comparison, and diagnostics for demand models."""

    @staticmethod
    def calculate_metrics(y_true, y_pred) -> dict[str, float]:
        """
        Compute regression metrics appropriate to a non-negative count target.

        Returns:
            `mse`, `rmse`, `mae`, `r2`, `poisson_deviance`, `mape`
            (non-zero actuals only), plus residual summaries.
        """
        y_true = np.asarray(y_true, dtype="float64").ravel()
        y_pred = np.asarray(y_pred, dtype="float64").ravel()

        if len(y_true) == 0:
            logger.warning("calculate_metrics received empty arrays.")
            return {}
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
            )

        metrics: dict[str, float] = {}
        metrics["mse"] = float(mean_squared_error(y_true, y_pred))
        metrics["rmse"] = float(np.sqrt(metrics["mse"]))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["r2"] = float(r2_score(y_true, y_pred))
        metrics["poisson_deviance"] = ModelEvaluator.poisson_deviance(y_true, y_pred)

        # MAPE over non-zero actuals only. Undefined where the actual is 0, which
        # is the majority of this target.
        nonzero = y_true != 0
        n_nonzero = int(nonzero.sum())
        if n_nonzero >= MIN_NONZERO_FOR_MAPE:
            metrics["mape"] = float(
                mean_absolute_percentage_error(y_true[nonzero], y_pred[nonzero])
            )
            metrics["mape_coverage"] = float(n_nonzero / len(y_true))
        else:
            logger.debug(
                "MAPE skipped: only %d non-zero actuals (need %d).",
                n_nonzero, MIN_NONZERO_FOR_MAPE,
            )
            metrics["mape"] = float("nan")
            metrics["mape_coverage"] = float(n_nonzero / len(y_true))

        residuals = y_true - y_pred
        metrics["mean_residual"] = float(np.mean(residuals))
        metrics["std_residual"] = float(np.std(residuals))
        metrics["zero_actual_share"] = float(np.mean(y_true == 0))
        metrics["negative_predictions"] = float(np.sum(y_pred < 0))
        return metrics

    @staticmethod
    def poisson_deviance(y_true, y_pred, eps: float = 1e-9) -> float:
        """
        Mean Poisson deviance - the natural loss for count data.

        Lower is better. Unlike squared error it penalises proportionally, so a
        miss of 2 on an expected 1 counts far more than a miss of 2 on an
        expected 50.
        """
        y_true = np.asarray(y_true, dtype="float64").ravel()
        y_pred = np.clip(np.asarray(y_pred, dtype="float64").ravel(), eps, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(y_true > 0, y_true * np.log(y_true / y_pred), 0.0)
        return float(2.0 * np.mean(term - (y_true - y_pred)))

    @staticmethod
    def seasonal_naive_baseline(
        panel: pd.DataFrame,
        *,
        season_length: int,
        ts_col: str = "ts",
        cluster_col: str = "pickup_cluster",
        target: str = "request_count",
    ) -> pd.Series:
        """
        Seasonal-naive forecast: each value predicted by the one `season_length`
        intervals earlier in the same cluster.

        For 30-minute data, `season_length=336` is one week back at the same
        weekday and time of day - a strong and very cheap baseline for demand.
        """
        ordered = panel.sort_values([cluster_col, ts_col])
        return ordered.groupby(cluster_col)[target].shift(season_length)

    @staticmethod
    def mase(y_true, y_pred, naive_pred) -> float:
        """
        Mean Absolute Scaled Error against a supplied naive forecast.

        `< 1` means the model beats the baseline; `>= 1` means it does not, and
        the baseline should be shipped instead.
        """
        y_true = np.asarray(y_true, dtype="float64").ravel()
        y_pred = np.asarray(y_pred, dtype="float64").ravel()
        naive_pred = np.asarray(naive_pred, dtype="float64").ravel()

        valid = ~np.isnan(naive_pred)
        if valid.sum() == 0:
            return float("nan")
        naive_error = np.mean(np.abs(y_true[valid] - naive_pred[valid]))
        if naive_error == 0:
            return float("nan")
        return float(np.mean(np.abs(y_true[valid] - y_pred[valid])) / naive_error)

    @staticmethod
    def compare_to_baselines(
        panel: pd.DataFrame,
        predictions,
        *,
        season_length: int = 336,
        ts_col: str = "ts",
        cluster_col: str = "pickup_cluster",
        target: str = "request_count",
    ) -> pd.DataFrame:
        """
        Score the model against the baselines it must beat to be worth deploying.

        Baselines:
          * **seasonal naive** - same interval one week earlier.
          * **cluster mean** - each cluster's historical average.

        Returns:
            One row per approach, with RMSE, MAE and MASE.
        """
        ordered = panel.sort_values([cluster_col, ts_col]).reset_index(drop=True)
        actual = ordered[target].to_numpy(dtype="float64")
        model_pred = np.asarray(predictions, dtype="float64").ravel()

        naive = ModelEvaluator.seasonal_naive_baseline(
            ordered, season_length=season_length, ts_col=ts_col,
            cluster_col=cluster_col, target=target,
        ).to_numpy(dtype="float64")
        cluster_mean = (
            ordered.groupby(cluster_col)[target].transform("mean").to_numpy(dtype="float64")
        )

        rows = []
        for name, pred in (
            ("model", model_pred),
            ("seasonal_naive", naive),
            ("cluster_mean", cluster_mean),
        ):
            valid = ~np.isnan(pred)
            if valid.sum() == 0:
                continue
            rows.append(
                {
                    "approach": name,
                    "rmse": float(np.sqrt(np.mean((actual[valid] - pred[valid]) ** 2))),
                    "mae": float(np.mean(np.abs(actual[valid] - pred[valid]))),
                    "mase": ModelEvaluator.mase(actual, pred, naive),
                    "n": int(valid.sum()),
                }
            )
        result = pd.DataFrame(rows).set_index("approach")

        if "model" in result.index and "seasonal_naive" in result.index:
            if result.loc["model", "rmse"] >= result.loc["seasonal_naive", "rmse"]:
                logger.warning(
                    "The model does not beat a seasonal-naive forecast "
                    "(RMSE %.4f vs %.4f). Ship the baseline instead until it does.",
                    result.loc["model", "rmse"], result.loc["seasonal_naive", "rmse"],
                )
        return result

    @staticmethod
    def compare_models(models_dict: dict[str, tuple]) -> pd.DataFrame:
        """Compare several models given `{name: (y_true, y_pred)}`."""
        results = []
        for name, (y_true, y_pred) in models_dict.items():
            metrics = ModelEvaluator.calculate_metrics(y_true, y_pred)
            metrics["model"] = name
            results.append(metrics)
        return pd.DataFrame(results).set_index("model")

    @staticmethod
    def get_best_model(comparison_df: pd.DataFrame, metric: str = "rmse") -> str:
        higher_is_better = metric in {"r2", "mape_coverage"}
        return (
            comparison_df[metric].idxmax()
            if higher_is_better
            else comparison_df[metric].idxmin()
        )

    @staticmethod
    def error_analysis(y_true, y_pred) -> dict[str, Any]:
        """Distribution of absolute errors."""
        y_true = np.asarray(y_true, dtype="float64").ravel()
        y_pred = np.asarray(y_pred, dtype="float64").ravel()
        errors = np.abs(y_true - y_pred)
        return {
            "total_samples": int(len(y_true)),
            "mean_error": float(np.mean(errors)),
            "median_error": float(np.median(errors)),
            "std_error": float(np.std(errors)),
            "min_error": float(np.min(errors)),
            "max_error": float(np.max(errors)),
            "percentile_25_error": float(np.percentile(errors, 25)),
            "percentile_75_error": float(np.percentile(errors, 75)),
            "percentile_90_error": float(np.percentile(errors, 90)),
            "percentile_95_error": float(np.percentile(errors, 95)),
        }

    @staticmethod
    def plot_residuals(y_true, y_pred, title: str = "Residual diagnostics"):
        """Four-panel residual diagnostic figure."""
        import matplotlib.pyplot as plt
        from scipy import stats

        y_true = np.asarray(y_true, dtype="float64").ravel()
        y_pred = np.asarray(y_pred, dtype="float64").ravel()
        residuals = y_true - y_pred

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes[0, 0].scatter(y_pred, residuals, alpha=0.3, s=8, color="#2a78d6")
        axes[0, 0].axhline(0, color="#e34948", linestyle="--", linewidth=1.5)
        axes[0, 0].set(xlabel="Predicted", ylabel="Residual", title="Residuals vs predicted")

        axes[0, 1].hist(residuals, bins=40, color="#2a78d6", edgecolor="none")
        axes[0, 1].set(xlabel="Residual", ylabel="Frequency", title="Residual distribution")

        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title("Q-Q plot")

        axes[1, 1].scatter(y_true, y_pred, alpha=0.3, s=8, color="#2a78d6")
        limits = [float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))]
        axes[1, 1].plot(limits, limits, "--", color="#e34948", linewidth=1.5)
        axes[1, 1].set(xlabel="Actual", ylabel="Predicted", title="Actual vs predicted")

        for ax in axes.ravel():
            ax.grid(True, alpha=0.3)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.tight_layout()
        return fig

    @staticmethod
    def cross_validation_analysis(model, X, y, cv=5):
        """
        Rolling-origin cross-validation.

        Uses `TimeSeriesSplit`, not `KFold`: a random fold would train on
        intervals that postdate its validation set, which for a forecasting model
        reports a score that cannot be achieved in production.
        """
        from sklearn.model_selection import TimeSeriesSplit, cross_val_score

        splitter = TimeSeriesSplit(n_splits=cv)
        rmse = np.sqrt(
            -cross_val_score(model, X, y, cv=splitter, scoring="neg_mean_squared_error")
        )
        r2 = cross_val_score(model, X, y, cv=splitter, scoring="r2")
        return {
            "rmse_mean": float(rmse.mean()), "rmse_std": float(rmse.std()),
            "r2_mean": float(r2.mean()), "r2_std": float(r2.std()),
            "rmse_scores": rmse, "r2_scores": r2,
        }


class PredictionValidator:
    """Sanity checks applied to predictions before they are used."""

    @staticmethod
    def check_prediction_bounds(
        y_pred, min_val: float = 0.0, max_val: float | None = None
    ) -> dict[str, Any]:
        """Flag predictions outside the plausible range for a demand count."""
        y_pred = np.asarray(y_pred, dtype="float64").ravel()
        low = int(np.sum(y_pred < min_val))
        high = int(np.sum(y_pred > max_val)) if max_val is not None else 0
        report = {
            "out_of_bounds_low": low,
            "out_of_bounds_high": high,
            "percentage_out_of_bounds": float((low + high) / max(len(y_pred), 1) * 100),
            "min_prediction": float(np.min(y_pred)),
            "max_prediction": float(np.max(y_pred)),
            "mean_prediction": float(np.mean(y_pred)),
        }
        if low:
            logger.warning(
                "%d predictions are below %.1f. Demand cannot be negative; "
                "clip before use.", low, min_val,
            )
        return report

    @staticmethod
    def check_prediction_stability(y_pred, window_size: int = 10) -> dict[str, float]:
        """Coefficient-of-variation stability across a rolling window."""
        y_pred = np.asarray(y_pred, dtype="float64").ravel()
        if len(y_pred) <= window_size:
            return {}
        windows = np.lib.stride_tricks.sliding_window_view(y_pred, window_size)
        stability = 1.0 - (windows.std(axis=1) / (np.abs(windows.mean(axis=1)) + 1e-8))
        return {
            "mean_stability": float(np.mean(stability)),
            "min_stability": float(np.min(stability)),
            "max_stability": float(np.max(stability)),
            "std_stability": float(np.std(stability)),
        }

    @staticmethod
    def validate_predictions(y_true, y_pred) -> dict[str, Any]:
        """Metrics, error analysis, bounds and stability in one call."""
        return {
            "basic_metrics": ModelEvaluator.calculate_metrics(y_true, y_pred),
            "error_analysis": ModelEvaluator.error_analysis(y_true, y_pred),
            "bounds_check": PredictionValidator.check_prediction_bounds(y_pred),
            "stability_check": PredictionValidator.check_prediction_stability(y_pred),
        }


def print_evaluation_report(metrics: dict[str, float]) -> None:
    """Print metrics as an aligned report."""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION REPORT")
    print("=" * 60)
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {name:.<40} {value:>15.4f}")
    print("=" * 60 + "\n")
