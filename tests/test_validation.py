"""Tests for rolling-origin validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ML_Pipeline.features import ModelBundle, build_feature_names
from ML_Pipeline.validation import (
    rolling_origin_validate,
    rolling_origin_validate_recursive,
    rolling_origins,
    summarise,
)


class TestRollingOrigins:
    def _stamps(self, n=2000):
        return pd.date_range("2021-01-01", periods=n, freq="30min")

    def test_origins_advance_forward_in_time(self):
        folds = rolling_origins(self._stamps(), n_folds=5, test_size=100)
        origins = [o for o, _ in folds]
        assert origins == sorted(origins)
        assert len(set(origins)) == len(origins)

    def test_every_test_window_is_the_requested_length(self):
        stamps = self._stamps()
        for origin, end in rolling_origins(stamps, n_folds=4, test_size=100):
            window = stamps[(stamps >= origin) & (stamps <= end)]
            assert len(window) == 100

    def test_last_fold_reaches_the_end_of_the_series(self):
        stamps = self._stamps()
        folds = rolling_origins(stamps, n_folds=5, test_size=100)
        assert folds[-1][1] == stamps[-1]

    def test_min_train_size_is_respected(self):
        stamps = self._stamps()
        folds = rolling_origins(stamps, n_folds=3, test_size=100, min_train_size=1200)
        assert folds[0][0] >= stamps[1200]

    def test_single_fold_holds_out_the_tail(self):
        stamps = self._stamps()
        folds = rolling_origins(stamps, n_folds=1, test_size=100)
        assert len(folds) == 1 and folds[0][1] == stamps[-1]

    def test_series_too_short_raises(self):
        with pytest.raises(ValueError, match="not enough"):
            rolling_origins(self._stamps(50), n_folds=3, test_size=100, min_train_size=40)

    def test_invalid_test_size_raises(self):
        with pytest.raises(ValueError, match="test_size"):
            rolling_origins(self._stamps(), n_folds=3, test_size=0)


@pytest.fixture
def trending_panel():
    """Two clusters on a rising trend - the property that broke the model."""
    stamps = pd.date_range("2021-01-01", periods=1500, freq="30min")
    rows = []
    for cluster in (0, 1):
        level = np.linspace(1, 20, len(stamps)) * (1 + cluster)
        rows.append(
            pd.DataFrame(
                {
                    "ts": stamps,
                    "pickup_cluster": cluster,
                    "request_count": np.clip(level, 0, None),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


class TestOneStepValidation:
    def test_reports_one_row_per_fold(self, trending_panel):
        def strategy(train, test):
            return np.full(len(test), train["request_count"].mean())

        out = rolling_origin_validate(
            trending_panel, strategy, n_folds=3, test_size=48,
            season_length=48, label="mean",
        )
        assert len(out) == 3
        assert set(out["fold"]) == {1, 2, 3}
        assert (out["strategy"] == "mean").all()

    def test_never_trains_on_the_future(self, trending_panel):
        seen = []

        def strategy(train, test):
            seen.append((train["ts"].max(), test["ts"].min()))
            return np.zeros(len(test))

        rolling_origin_validate(
            trending_panel, strategy, n_folds=3, test_size=48, season_length=48
        )
        for train_end, test_start in seen:
            assert train_end < test_start

    def test_a_perfect_strategy_scores_zero_error(self, trending_panel):
        def oracle(train, test):
            return test["request_count"].to_numpy()

        out = rolling_origin_validate(
            trending_panel, oracle, n_folds=3, test_size=48, season_length=48
        )
        assert (out["rmse"] < 1e-9).all()
        assert (out["mase"] < 1e-9).all()

    def test_wrong_prediction_count_raises(self, trending_panel):
        with pytest.raises(ValueError, match="returned"):
            rolling_origin_validate(
                trending_panel, lambda tr, te: np.zeros(3),
                n_folds=2, test_size=48, season_length=48,
            )

    def test_records_level_tracking(self, trending_panel):
        """mean_pred vs mean_actual exposes a model stuck at the old level."""
        def stale(train, test):
            return np.full(len(test), train["request_count"].mean())

        out = rolling_origin_validate(
            trending_panel, stale, n_folds=3, test_size=48, season_length=48
        )
        # On a rising series a training-mean forecast must under-predict.
        assert (out["mean_pred"] < out["mean_actual"]).all()


class ConstantLagModel:
    def predict(self, X):
        return np.asarray(X["lag_1"], dtype="float64")


class TestRecursiveValidation:
    def test_reports_level_ratio(self, trending_panel):
        def fit(train):
            return ModelBundle(
                model=ConstantLagModel(),
                feature_names=build_feature_names(
                    use_lags=True, lags=(1, 2, 3),
                ),
                uses_lags=True, lags=(1, 2, 3), rolling_window=3, freq="30min",
            )

        out = rolling_origin_validate_recursive(
            trending_panel, fit, n_folds=2, horizon=24, season_length=48,
        )
        assert "level_ratio" in out.columns
        # A model that just repeats the last observed value cannot follow a
        # rising trend, so it must under-forecast the level.
        assert (out["level_ratio"] <= 1.01).all()

    def test_every_fold_predicts_the_whole_horizon(self, trending_panel):
        def fit(train):
            return ModelBundle(
                model=ConstantLagModel(),
                feature_names=build_feature_names(use_lags=True, lags=(1, 2, 3)),
                uses_lags=True, lags=(1, 2, 3), rolling_window=3, freq="30min",
            )

        out = rolling_origin_validate_recursive(
            trending_panel, fit, n_folds=2, horizon=24, season_length=48,
        )
        assert (out["n_test"] == 24 * 2).all()  # 24 intervals x 2 clusters


class TestSummarise:
    def test_counts_folds_beating_the_baseline(self):
        a = pd.DataFrame({"strategy": ["a"] * 3, "rmse": [1.0, 1.1, 0.9],
                          "mase": [0.8, 0.9, 1.2]})
        b = pd.DataFrame({"strategy": ["b"] * 3, "rmse": [2.0, 2.1, 1.9],
                          "mase": [1.5, 1.6, 1.4]})
        out = summarise([a, b])
        assert out.loc["a", "folds_beating_naive"] == 2
        assert out.loc["b", "folds_beating_naive"] == 0

    def test_reports_spread_not_just_the_mean(self):
        """A strategy straddling MASE 1.0 has not been shown to beat naive."""
        wobbly = pd.DataFrame({"strategy": ["w"] * 3, "rmse": [1, 1, 1],
                               "mase": [0.5, 0.95, 1.5]})
        out = summarise([wobbly])
        assert out.loc["w", "mase_worst"] == 1.5
        assert out.loc["w", "mase_std"] > 0

    def test_orders_by_mean_mase(self):
        good = pd.DataFrame({"strategy": ["good"] * 2, "rmse": [1, 1], "mase": [0.7, 0.8]})
        bad = pd.DataFrame({"strategy": ["bad"] * 2, "rmse": [2, 2], "mase": [1.4, 1.5]})
        assert summarise([bad, good]).index[0] == "good"
