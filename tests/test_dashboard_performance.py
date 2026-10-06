"""
Tests for the dashboard's model-performance logic.

The dashboard was the last substantial untested module, and the audit's F1 finding
was that it showed five pages of exploratory analysis and nothing about whether the
forecast is any good.

These cover the parts that compute rather than render: resolving the promoted
model, and rebuilding the feature frame the model needs from a bare demand grid.
Chart drawing is not asserted here - it is exercised by booting the app.

Two behaviours are worth pinning because both were wrong in the first version:

- **The MASE window has to be wider than the backtest.** A seasonal-naive forecast
  is the value 336 intervals earlier, so it cannot be computed over a 48-interval
  horizon. The baseline comparison and per-cluster MASE came out empty, which made
  the two most valuable panels permanently blank.
- **A missing feature must be named, not swallowed.** The first version caught
  `KeyError` and reported "could not score", which hid a genuine train/serve
  mismatch.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from joblib import dump

from ML_Pipeline.config import PipelineConfig
from ML_Pipeline.modeling.evaluation import ModelEvaluator
from ML_Pipeline.modeling.features import (
    CLUSTER_COL,
    TARGET_COL,
    TS_COL,
    ModelBundle,
    add_calendar_features,
    add_lag_features,
)
from ML_Pipeline.registry import ModelRegistry
from ML_Pipeline.serving.state import ServingState

SEASON = 336
N_CLUSTERS = 3


class Flat:
    def __init__(self, value: float = 3.0):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value, dtype="float64")


def build(tmp_path, intervals: int, feature_names: list[str]) -> str:
    """An output directory holding a grid and a promoted bundle."""
    version = "20260102_030405"
    stamps = pd.date_range("2021-01-01", periods=intervals, freq="30min")
    # A daily shape plus a growth trend plus deterministic jitter. The trend
    # matters: a perfectly periodic series makes the seasonal-naive forecast exact,
    # its error zero, and MASE undefined by definition - so a degenerate fixture
    # looks like a broken metric. The real series grew 5.2x in a year.
    rng = np.random.default_rng(20261003)
    grid = pd.DataFrame(
        [
            {
                TS_COL: t,
                CLUSTER_COL: c,
                TARGET_COL: float(
                    max(
                        0.0,
                        2.0
                        + (c + t.hour) % 6
                        + 0.004 * i                      # growth across the series
                        + rng.normal(0, 0.8)             # week-to-week variation
                    )
                ),
            }
            for c in range(N_CLUSTERS)
            for i, t in enumerate(stamps)
        ]
    )
    grid.to_csv(tmp_path / f"Data_Prepared_{version}.csv.gz", index=False,
                compression="gzip")

    bundle = ModelBundle(
        model=Flat(),
        feature_names=feature_names,
        uses_lags=True,
        lags=(1, SEASON),
        rolling_window=3,
        freq="30min",
        trained_at=datetime.now().isoformat(),
        metrics={"test_rmse": 3.0, "baseline_mase": 0.8,
                 "beats_seasonal_naive": 1.0, "clusters_losing_to_naive": 0.0},
    )
    dump(bundle, tmp_path / f"prediction_model_with_lag_{version}.joblib")
    registry = ModelRegistry(str(tmp_path / "model_registry.json"))
    name = f"xgb_with_lag_{version}"
    registry.register_model(
        model_name=name,
        model_path=str(tmp_path / f"prediction_model_with_lag_{version}.joblib"),
        model_type="xgboost", metrics=bundle.metrics,
        metadata={"version": version},
    )
    registry.promote_model(name)
    return name


def state_for(tmp_path) -> ServingState:
    return ServingState(
        PipelineConfig(output_dir=str(tmp_path), logs_dir=str(tmp_path))
    )


def rebuild_window(state: ServingState, weeks: int):
    """
    The feature chain the dashboard performs, mirrored here.

    The serving history is a bare `[ts, cluster, count]` panel, so lags *and*
    calendar features must both be rebuilt. Omitting the calendar half is the bug
    this guards.
    """
    bundle = state.bundle
    frame = add_lag_features(
        state.history, lags=bundle.lags, rolling_window=bundle.rolling_window
    )
    frame = add_calendar_features(frame, TS_COL)
    cutoff = frame[TS_COL].max() - pd.Timedelta(weeks=weeks)
    return frame[frame[TS_COL] >= cutoff].reset_index(drop=True)


CALENDAR_ONLY = ["mins", "hour", "month", "quarter", "dayofweek"]
WITH_LAGS = [*CALENDAR_ONLY, "lag_1", f"lag_{SEASON}", "rolling_mean"]


class TestTheWindowMustBeWiderThanTheBacktest:
    def test_a_short_horizon_cannot_produce_a_baseline(self, tmp_path):
        """
        Why the page needs two measurements.

        Over 48 intervals every seasonal-naive value is NaN, so
        `compare_to_baselines` omits the row entirely and MASE is undefined. That
        is a property of the metric, not a bug - but it means a 48-interval
        backtest cannot answer "does this beat the baseline".
        """
        build(tmp_path, intervals=SEASON + 200, feature_names=WITH_LAGS)
        state = state_for(tmp_path)
        frame = rebuild_window(state, weeks=99)
        short = frame[frame[TS_COL] >= frame[TS_COL].max() - pd.Timedelta(hours=24)]

        result = ModelEvaluator.compare_to_baselines(
            short, state.bundle.predict(short), season_length=SEASON
        )
        assert "seasonal_naive" not in result.index

    def test_a_week_or_more_does_produce_one(self, tmp_path):
        build(tmp_path, intervals=SEASON * 3, feature_names=WITH_LAGS)
        state = state_for(tmp_path)
        window = rebuild_window(state, weeks=2)

        result = ModelEvaluator.compare_to_baselines(
            window, state.bundle.predict(window), season_length=SEASON
        )
        assert "seasonal_naive" in result.index
        assert np.isfinite(result.loc["model", "mase"])

    def test_per_cluster_mase_is_computable_on_the_wide_window(self, tmp_path):
        build(tmp_path, intervals=SEASON * 3, feature_names=WITH_LAGS)
        state = state_for(tmp_path)
        window = rebuild_window(state, weeks=2)

        per_cluster = ModelEvaluator.per_cluster_error(
            window, state.bundle.predict(window), season_length=SEASON
        )
        assert len(per_cluster) == N_CLUSTERS
        assert per_cluster["mase"].notna().all(), (
            "every cluster should score; a NaN here leaves the chart blank"
        )


class TestFeatureRebuild:
    def test_calendar_features_are_rebuilt_from_the_bare_grid(self, tmp_path):
        """
        The serving history carries only ts, cluster and count.

        The first version of the page rebuilt lags but not calendar features, so
        `predict` raised and the page reported a generic failure.
        """
        build(tmp_path, intervals=SEASON * 2, feature_names=WITH_LAGS)
        state = state_for(tmp_path)
        assert set(state.history.columns) == {TS_COL, CLUSTER_COL, TARGET_COL}

        window = rebuild_window(state, weeks=1)
        for feature in state.bundle.feature_names:
            assert feature in window.columns, f"{feature} was not rebuilt"
        assert np.isfinite(state.bundle.predict(window)).all()

    def test_a_feature_the_grid_cannot_supply_is_detectable(self, tmp_path):
        """
        A model needing something the grid lacks is a train/serve mismatch, and the
        page must be able to name it rather than fail opaquely.
        """
        build(tmp_path, intervals=SEASON * 2,
              feature_names=[*WITH_LAGS, "rainfall_mm"])
        state = state_for(tmp_path)
        window = rebuild_window(state, weeks=1)

        missing = [c for c in state.bundle.feature_names if c not in window.columns]
        assert missing == ["rainfall_mm"]
        with pytest.raises(KeyError, match="rainfall_mm"):
            state.bundle.predict(window)


class TestWhatThePageReports:
    def test_the_promoted_model_is_what_gets_scored(self, tmp_path):
        name = build(tmp_path, intervals=SEASON * 2, feature_names=WITH_LAGS)
        state = state_for(tmp_path)
        assert state.model_name == name
        assert state.ready

    def test_the_gate_verdict_is_available_for_the_headline_tiles(self, tmp_path):
        build(tmp_path, intervals=SEASON * 2, feature_names=WITH_LAGS)
        state = state_for(tmp_path)
        metrics = state.info["metrics"]
        assert bool(metrics["beats_seasonal_naive"]) is True
        assert metrics["baseline_mase"] == pytest.approx(0.8)
        assert state.stale is False

    def test_level_ratio_detects_systematic_under_forecasting(self, tmp_path):
        """The metric the model card says degrades earliest."""
        build(tmp_path, intervals=SEASON * 2, feature_names=WITH_LAGS)
        state = state_for(tmp_path)
        window = rebuild_window(state, weeks=1)
        halved = window[TARGET_COL].to_numpy(dtype="float64") * 0.5

        per_cluster = ModelEvaluator.per_cluster_error(
            window, halved, season_length=SEASON
        )
        assert per_cluster["level_ratio"].between(0.49, 0.51).all()
