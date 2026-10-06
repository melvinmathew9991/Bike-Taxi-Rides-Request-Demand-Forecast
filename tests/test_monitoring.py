"""
Tests for the scheduled health check.

The model card's conditions for use included two nobody enforced: retrain every
four weeks, and watch `level_ratio`. These pin that the check fails on each of
the things it exists to catch, and - as importantly - that it does not fail on a
fresh model, or score demand the model was fitted on.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from joblib import dump

from ML_Pipeline.config import PipelineConfig
from ML_Pipeline.modeling.features import CLUSTER_COL, TARGET_COL, TS_COL, ModelBundle
from ML_Pipeline.registry import ModelRegistry
from ML_Pipeline.serving.monitoring import (
    FAIL,
    PASS,
    SKIPPED,
    WARN,
    MissingFeaturesError,
    check_health,
)
from ML_Pipeline.serving.state import ServingState

SEASON = 336
N_CLUSTERS = 3
INTERVALS = SEASON * 6  # six weeks: long enough to go past the 28-day cadence
FEATURES = ["hour", "lag_1", f"lag_{SEASON}", "rolling_mean", CLUSTER_COL]
STAMPS = pd.date_range("2021-01-01", periods=INTERVALS, freq="30min")
END = STAMPS[-1]


class Persistence:
    """Predicts the last observed value, scaled - optionally for one cluster only."""

    def __init__(self, scale: float = 1.0, cluster: int | None = None):
        self.scale, self.cluster = scale, cluster

    def predict(self, X):
        pred = X["lag_1"].to_numpy(dtype="float64").copy()
        hit = (
            np.ones(len(X), dtype=bool) if self.cluster is None
            else X[CLUSTER_COL].to_numpy() == self.cluster
        )
        pred[hit] *= self.scale
        return pred


class ConstantForOne(Persistence):
    """Persistence, except a fixed value for one cluster."""

    def __init__(self, cluster: int, value: float):
        super().__init__()
        self.only, self.value = cluster, value

    def predict(self, X):
        pred = super().predict(X)
        pred[X[CLUSTER_COL].to_numpy() == self.only] = self.value
        return pred


class SeasonalNaive:
    """Is the baseline, so cannot beat it."""

    def predict(self, X):
        return X[f"lag_{SEASON}"].to_numpy(dtype="float64")


def build(tmp_path, model, *, data_through=None, features=FEATURES,
          quiet_cluster: int | None = None) -> pd.Timestamp:
    """
    An output directory with a grid and a promoted bundle. Returns grid end.

    `quiet_cluster` becomes noise around one request per interval, like the
    real grid's quietest clusters.
    """
    version = "20260102_030405"
    stamps = STAMPS
    # Smooth, with a daily shape and growth: persistence beats same-time-last-week
    # comfortably, as a real one-step model does on this series.
    grid = pd.DataFrame(
        [
            {
                TS_COL: t,
                CLUSTER_COL: c,
                TARGET_COL: 5.0 + c + 0.01 * i
                + 2.0 * np.sin(2 * np.pi * (t.hour * 2 + t.minute // 30) / 48),
            }
            for c in range(N_CLUSTERS)
            for i, t in enumerate(stamps)
        ]
    )
    if quiet_cluster is not None:
        rows = grid[CLUSTER_COL] == quiet_cluster
        rng = np.random.default_rng(20261005)
        grid.loc[rows, TARGET_COL] = np.clip(
            1.0 + rng.normal(0, 0.8, rows.sum()), 0, None
        )
    grid.to_csv(tmp_path / f"Data_Prepared_{version}.csv.gz", index=False,
                compression="gzip")

    end = stamps[-1]
    through = end - pd.Timedelta(days=10) if data_through is None else data_through
    bundle = ModelBundle(
        model=model, feature_names=features, uses_lags=True, lags=(1, SEASON),
        rolling_window=3, freq="30min", trained_at=datetime.now().isoformat(),
        data_through=pd.Timestamp(through).isoformat(),
        metrics={"baseline_mase": 0.8, "beats_seasonal_naive": 1.0},
    )
    path = tmp_path / f"prediction_model_with_lag_{version}.joblib"
    dump(bundle, path)
    registry = ModelRegistry(str(tmp_path / "model_registry.json"))
    name = f"xgb_with_lag_{version}"
    registry.register_model(
        model_name=name, model_path=str(path), model_type="xgboost",
        metrics=bundle.metrics, metadata={"version": version},
    )
    registry.promote_model(name)
    return end


def state_for(tmp_path) -> ServingState:
    return ServingState(PipelineConfig(output_dir=str(tmp_path), logs_dir=str(tmp_path)))


def statuses(report) -> dict[str, str]:
    return {c.name: c.status for c in report.checks}


class TestAHealthyModelPasses:
    def test_every_check_passes(self, tmp_path):
        build(tmp_path, Persistence())
        report = check_health(state_for(tmp_path))
        assert statuses(report) == {
            "staleness": PASS, "mase": PASS, "level_ratio": PASS, "clusters": PASS,
        }
        assert report.healthy
        assert report.mase < 1.0
        assert report.level_ratio == pytest.approx(1.0, abs=0.02)

    def test_the_baseline_has_a_value_for_every_scored_row(self, tmp_path):
        """
        The naive forecast comes from the full history. Computed inside a
        one-week window, every value would be NaN and MASE could never be scored.
        """
        build(tmp_path, Persistence())
        report = check_health(state_for(tmp_path))
        assert report.rows_scored == 7 * 48 * N_CLUSTERS
        assert all(row["mase"] is not None for row in report.clusters)


class TestWhatItCatches:
    def test_under_forecasting_fails_the_level_ratio(self, tmp_path):
        build(tmp_path, Persistence(scale=0.8))
        report = check_health(state_for(tmp_path))
        assert statuses(report)["level_ratio"] == FAIL
        assert report.level_ratio == pytest.approx(0.8, abs=0.02)
        assert not report.healthy

    def test_over_forecasting_fails_too(self, tmp_path):
        build(tmp_path, Persistence(scale=1.25))
        report = check_health(state_for(tmp_path))
        assert statuses(report)["level_ratio"] == FAIL

    def test_a_model_no_better_than_the_baseline_fails(self, tmp_path):
        build(tmp_path, SeasonalNaive())
        report = check_health(state_for(tmp_path))
        assert report.mase == pytest.approx(1.0)
        assert statuses(report)["mase"] == FAIL

    def test_a_stale_model_fails(self, tmp_path):
        end = build(tmp_path, Persistence(),
                    data_through=pd.Timestamp("2021-01-01") + pd.Timedelta(days=1))
        report = check_health(state_for(tmp_path))
        assert statuses(report)["staleness"] == FAIL
        assert report.data_lag_days == pytest.approx(
            (end - pd.Timestamp("2021-01-02")) / pd.Timedelta(days=1)
        )

    def test_one_bad_cluster_warns_without_failing(self, tmp_path):
        """One week of one cluster is too noisy to pull the model on, alone."""
        build(tmp_path, Persistence(scale=0.5, cluster=2))
        report = check_health(state_for(tmp_path))
        assert statuses(report)["clusters"] == WARN
        assert "#2" in next(c.detail for c in report.checks if c.name == "clusters")


class TestQuietClusters:
    """
    The replay drill flagged the same quiet clusters every week: all beating the
    baseline, all with level ratios of 1.2-2.1 on under three requests per
    interval. The band is noise there, so it is not applied.
    """

    def test_over_forecasting_a_quiet_cluster_is_not_flagged(self, tmp_path):
        build(tmp_path, ConstantForOne(cluster=0, value=1.3), quiet_cluster=0)
        report = check_health(state_for(tmp_path))
        row = next(r for r in report.clusters if r[CLUSTER_COL] == 0)
        assert row["level_ratio"] > 1.1, "precondition: outside the band"
        assert row["mase"] < 1.0, "precondition: beats the baseline"
        assert statuses(report)["clusters"] == PASS
        assert "1 below 3" in next(
            c.detail for c in report.checks if c.name == "clusters"
        )

    def test_the_floor_is_what_spares_it(self, tmp_path):
        build(tmp_path, ConstantForOne(cluster=0, value=1.3), quiet_cluster=0)
        report = check_health(state_for(tmp_path), min_cluster_volume=0.0)
        assert statuses(report)["clusters"] == WARN

    def test_a_quiet_cluster_losing_to_the_baseline_is_still_flagged(self, tmp_path):
        build(tmp_path, ConstantForOne(cluster=0, value=4.0), quiet_cluster=0)
        report = check_health(state_for(tmp_path))
        row = next(r for r in report.clusters if r[CLUSTER_COL] == 0)
        assert row["mase"] >= 1.0
        assert statuses(report)["clusters"] == WARN


class TestItScoresOnlyUnseenDemand:
    def test_too_little_new_demand_skips_rather_than_scoring_in_sample(self, tmp_path):
        # Badly under-forecasting, so scoring it at all would fail it.
        build(tmp_path, Persistence(scale=0.5), data_through=END - pd.Timedelta(days=2))
        report = check_health(state_for(tmp_path))
        assert statuses(report)["mase"] == SKIPPED
        assert statuses(report)["level_ratio"] == SKIPPED
        assert report.healthy, "a fresh model is not unhealthy for being fresh"

    def test_the_window_is_the_most_recent_week(self, tmp_path):
        end = build(tmp_path, Persistence())
        report = check_health(state_for(tmp_path))
        assert pd.Timestamp(report.window_end) == end
        assert pd.Timestamp(report.window_start) == (
            end - pd.Timedelta(days=7) + pd.Timedelta(minutes=30)
        )


class TestWhenThereIsNothingToCheck:
    def test_no_promoted_model_raises(self, tmp_path):
        state = state_for(tmp_path)
        with pytest.raises(ValueError, match="No model is ready"):
            check_health(state)

    def test_a_feature_the_grid_cannot_supply_is_named(self, tmp_path):
        build(tmp_path, Persistence(), features=[*FEATURES, "rainfall_mm"])
        with pytest.raises(MissingFeaturesError, match="rainfall_mm"):
            check_health(state_for(tmp_path))


class TestTheCommandLine:
    @pytest.fixture
    def cli(self):
        from ML_Pipeline.cli.monitor import main

        return main

    def test_healthy_exits_0_and_writes_the_report(self, tmp_path, cli):
        build(tmp_path, Persistence())
        out = tmp_path / "health.json"
        assert cli(["--output-dir", str(tmp_path), "--json", str(out)]) == 0
        report = json.loads(out.read_text())
        assert report["healthy"] is True
        assert len(report["clusters"]) == N_CLUSTERS

    def test_unhealthy_exits_3(self, tmp_path, cli):
        build(tmp_path, Persistence(scale=0.8))
        assert cli(["--output-dir", str(tmp_path)]) == 3

    def test_nothing_to_check_exits_2(self, tmp_path, cli):
        assert cli(["--output-dir", str(tmp_path)]) == 2
