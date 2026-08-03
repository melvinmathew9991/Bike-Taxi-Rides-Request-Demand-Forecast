"""
Tests for multi-step forecasting.

The headline cases here are regression tests for the three defects that made the
old prediction stage produce almost nothing: a fixed three-step loop, row
destruction via repeated `dropna`, and a `None` return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ML_Pipeline.features import ModelBundle, build_feature_names
from ML_Pipeline.forecast import (
    PREDICTION_COL,
    backtest_recursive,
    forecast_direct,
    forecast_recursive,
)

FREQ = "30min"


class ConstantModel:
    """Predicts a fixed value; makes row accounting easy to assert."""

    def __init__(self, value: float = 7.0):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


class EchoLagModel:
    """Returns lag_1 unchanged, so recursion is directly traceable."""

    def predict(self, X):
        return np.asarray(X["lag_1"], dtype="float64")


def _lag_bundle(model, lags=(1, 2, 3), window=3) -> ModelBundle:
    return ModelBundle(
        model=model,
        feature_names=build_feature_names(use_lags=True, lags=lags),
        uses_lags=True,
        lags=lags,
        rolling_window=window,
        freq=FREQ,
    )


def _nolag_bundle(model) -> ModelBundle:
    return ModelBundle(
        model=model,
        feature_names=build_feature_names(use_lags=False),
        uses_lags=False,
        freq=FREQ,
    )


@pytest.fixture
def horizon(panel) -> pd.DatetimeIndex:
    start = pd.to_datetime(panel["ts"]).max() + pd.Timedelta(FREQ)
    return pd.date_range(start=start, periods=48, freq=FREQ)


class TestForecastDirect:
    def test_predicts_every_horizon_row(self, horizon):
        clusters = list(range(4))
        out = forecast_direct(_nolag_bundle(ConstantModel()), horizon, clusters)
        assert len(out) == len(horizon) * len(clusters)
        assert out[PREDICTION_COL].notna().all()
        assert set(out["ts"]) == set(horizon)
        assert sorted(out["pickup_cluster"].unique()) == clusters

    def test_marks_rows_as_forecast(self, horizon):
        out = forecast_direct(_nolag_bundle(ConstantModel()), horizon, [0])
        assert out["is_forecast"].all()

    def test_rejects_a_lag_model(self, horizon):
        with pytest.raises(ValueError, match="lag-free"):
            forecast_direct(_lag_bundle(ConstantModel()), horizon, [0])


class TestForecastRecursive:
    def test_predicts_the_whole_horizon_not_three_steps(self, panel, horizon):
        """
        Regression: the old loop was `for x in range(3)`, so exactly three
        timestamps were predicted no matter how long the horizon was.
        """
        clusters = list(range(4))
        out = forecast_recursive(
            _lag_bundle(ConstantModel()), panel, horizon, clusters=clusters
        )
        assert len(out) == len(horizon) * len(clusters) == 192
        assert out[PREDICTION_COL].notna().all()
        assert set(out["ts"]) == set(horizon)

    def test_does_not_lose_rows_across_steps(self, panel, horizon):
        """
        Regression: each old iteration re-ran a `dropna`-terminated builder on
        its own output, deleting three intervals per cluster per pass.
        """
        out = forecast_recursive(_lag_bundle(ConstantModel()), panel, horizon)
        per_cluster = out.groupby("pickup_cluster").size()
        assert per_cluster.nunique() == 1
        assert per_cluster.iloc[0] == len(horizon)

    def test_returns_a_dataframe(self, panel, horizon):
        """Regression: the old function returned None."""
        out = forecast_recursive(_lag_bundle(ConstantModel()), panel, horizon)
        assert isinstance(out, pd.DataFrame)
        assert not out.empty

    def test_predictions_feed_forward(self, panel, horizon):
        """Step k's lag_1 must be step k-1's prediction, not an observation."""
        out = forecast_recursive(
            _lag_bundle(EchoLagModel()), panel, horizon, clusters=[0]
        )
        out = out.sort_values("ts").reset_index(drop=True)
        assert np.allclose(out["lag_1"].to_numpy()[1:], out[PREDICTION_COL].to_numpy()[:-1])

    def test_echo_model_propagates_last_observed_value(self, small_panel):
        """
        With a model that returns lag_1, every forecast step must equal the last
        observed value - a closed-form check that recursion is wired correctly.
        """
        start = pd.to_datetime(small_panel["ts"]).max() + pd.Timedelta(FREQ)
        horizon = pd.date_range(start=start, periods=5, freq=FREQ)
        out = forecast_recursive(
            _lag_bundle(EchoLagModel()), small_panel, horizon, clusters=[0, 1]
        )
        assert (out.loc[out.pickup_cluster == 0, PREDICTION_COL] == 9.0).all()
        assert (out.loc[out.pickup_cluster == 1, PREDICTION_COL] == 109.0).all()

    def test_negative_predictions_are_clipped(self, panel, horizon):
        out = forecast_recursive(
            _lag_bundle(ConstantModel(-3.0)), panel, horizon, clusters=[0]
        )
        assert (out[PREDICTION_COL] >= 0).all()

    def test_rejects_a_lag_free_model(self, panel, horizon):
        with pytest.raises(ValueError, match="requires a lag-using model"):
            forecast_recursive(_nolag_bundle(ConstantModel()), panel, horizon)

    def test_non_contiguous_horizon_raises(self, panel):
        """A gap between history and horizon would silently corrupt the lags."""
        start = pd.to_datetime(panel["ts"]).max() + pd.Timedelta("6h")
        horizon = pd.date_range(start=start, periods=4, freq=FREQ)
        with pytest.raises(ValueError, match="contiguous"):
            forecast_recursive(_lag_bundle(ConstantModel()), panel, horizon)

    def test_insufficient_history_raises(self):
        stamps = pd.date_range("2021-03-27", periods=2, freq=FREQ)
        thin = pd.DataFrame(
            {"ts": stamps, "pickup_cluster": [0, 0], "request_count": [1.0, 2.0]}
        )
        horizon = pd.date_range(stamps[-1] + pd.Timedelta(FREQ), periods=3, freq=FREQ)
        with pytest.raises(ValueError, match="needs 3"):
            forecast_recursive(_lag_bundle(ConstantModel()), thin, horizon)

    def test_history_entirely_after_horizon_raises(self, panel):
        horizon = pd.date_range("2019-01-01", periods=3, freq=FREQ)
        with pytest.raises(ValueError, match="no intervals before"):
            forecast_recursive(_lag_bundle(ConstantModel()), panel, horizon)


class TestBacktest:
    def test_pairs_every_actual_with_a_prediction(self, panel):
        out = backtest_recursive(_lag_bundle(ConstantModel()), panel, horizon_steps=24)
        assert len(out) == 24 * 4
        assert out[PREDICTION_COL].notna().all()
        assert "request_count" in out.columns

    def test_rejects_horizon_longer_than_panel(self, small_panel):
        with pytest.raises(ValueError, match="need more than"):
            backtest_recursive(
                _lag_bundle(ConstantModel()), small_panel, horizon_steps=50
            )
