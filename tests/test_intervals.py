"""
Tests for prediction intervals.

Poisson quantiles around the point forecast were measured to cover 56% of
intervals at the busiest clusters for a nominal 80%, with misses mostly above.
These specify the calibrated replacement: bounds in whole requests that contain
the point forecast, quantiles looked up by horizon and volume, and a calibration
that reaches its nominal coverage on data it was not fitted on - including for a
model that is biased low, which is the failure that mattered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ML_Pipeline.modeling.features import (
    CLUSTER_COL,
    TARGET_COL,
    TS_COL,
    ModelBundle,
    build_feature_names,
)
from ML_Pipeline.modeling.forecast import PREDICTION_COL, forecast_recursive
from ML_Pipeline.modeling.intervals import (
    LOWER_COL,
    UPPER_COL,
    IntervalCalibration,
    calibrate_intervals,
    horizon_edges_in_steps,
)

FREQ = "30min"
N_CLUSTERS = 3
CLUSTER_RATE = np.array([2.0, 8.0, 30.0])  # quiet, moderate, busy
HOUR_SHAPE = 0.5 + np.sin(np.linspace(0, np.pi, 24))  # 0.5 at night, 1.5 midday
LAGS = (1, 2, 3)


def _mean(cluster: np.ndarray, hour: np.ndarray) -> np.ndarray:
    return CLUSTER_RATE[cluster] * HOUR_SHAPE[hour]


class Oracle:
    """Predicts the true Poisson mean, scaled by `bias`."""

    def __init__(self, bias: float = 1.0):
        self.bias = bias

    def predict(self, X):
        cluster = X[CLUSTER_COL].to_numpy(dtype=int)
        hour = X["hour"].to_numpy(dtype=int)
        return self.bias * _mean(cluster, hour)


def _bundle(model, intervals: IntervalCalibration | None = None) -> ModelBundle:
    return ModelBundle(
        model=model,
        feature_names=build_feature_names(use_lags=True, lags=LAGS),
        uses_lags=True, lags=LAGS, rolling_window=3, freq=FREQ,
        intervals=intervals,
    )


@pytest.fixture
def poisson_panel() -> pd.DataFrame:
    """Twenty days of Poisson demand whose mean the Oracle knows exactly."""
    stamps = pd.date_range("2021-01-01", periods=20 * 48, freq=FREQ)
    ts = np.repeat(stamps, N_CLUSTERS)
    cluster = np.tile(np.arange(N_CLUSTERS), len(stamps))
    counts = np.random.default_rng(7).poisson(_mean(cluster, ts.hour.to_numpy()))
    return pd.DataFrame(
        {TS_COL: ts, CLUSTER_COL: cluster, TARGET_COL: counts.astype("float64")}
    )


def _calibration(lower_z, upper_z, horizon_edges=(1,), volume_edges=()) -> IntervalCalibration:
    return IntervalCalibration(
        level=0.8, horizon_edges=horizon_edges, volume_edges=volume_edges,
        lower_z=lower_z, upper_z=upper_z,
        cell_rows=tuple(tuple(0 for _ in row) for row in lower_z),
        origins=0, window_start="", window_end="",
    )


def _calibrate(panel, model, **kwargs):
    defaults = dict(
        start=pd.Timestamp("2021-01-08"), days=12, n_origins=12, steps=24, level=0.8,
    )
    return calibrate_intervals(_bundle(model), panel, **{**defaults, **kwargs})


class TestBounds:
    def test_they_are_rounded_inward_to_whole_requests(self):
        """[6.4, 12.6] covers exactly the counts 7..12."""
        cal = _calibration(((-1.0,),), ((1.0,),))
        lower, upper = cal.bounds(np.array([9.0, 9.5]), np.array([1, 1]))
        assert lower.tolist() == [6.0, 7.0]  # 9 - 3 = 6; 9.5 - 3.08 = 6.42 -> 7
        assert upper.tolist() == [12.0, 12.0]  # 9 + 3 = 12; 9.5 + 3.08 = 12.58 -> 12

    def test_the_lower_bound_is_never_negative(self):
        cal = _calibration(((-5.0,),), ((1.0,),))
        lower, _ = cal.bounds(np.array([0.5, 2.0]), np.array([1, 1]))
        assert (lower >= 0).all()

    def test_the_rounded_point_forecast_is_always_inside(self):
        """Inward rounding alone would give a forecast of 0.6 an interval of [0, 0]."""
        cal = _calibration(((-0.1,),), ((0.1,),))
        predicted = np.array([0.6, 4.4, 4.6, 20.0])
        lower, upper = cal.bounds(predicted, np.ones(4, dtype=int))
        point = np.round(predicted)
        assert ((lower <= point) & (point <= upper)).all()

    def test_quantiles_are_looked_up_by_horizon_and_volume(self):
        cal = _calibration(
            lower_z=((0.0, 0.0), (0.0, 0.0)),
            upper_z=((1.0, 2.0), (3.0, 4.0)),
            horizon_edges=(2, 4), volume_edges=(10.0,),
        )
        mu = np.array([9.0, 16.0, 9.0, 16.0, 16.0])
        steps = np.array([1, 2, 3, 4, 50])  # 50 is past the last edge
        _, upper = cal.bounds(mu, steps)
        # sqrt(9) = 3 and sqrt(16) = 4 scale the quantiles.
        assert upper.tolist() == [9 + 3, 16 + 8, 9 + 9, 16 + 16, 16 + 16]

    def test_a_volume_on_an_edge_falls_in_the_lower_bucket(self):
        cal = _calibration(
            lower_z=((0.0, 0.0),), upper_z=((1.0, 9.0),), volume_edges=(1.0,),
        )
        _, upper = cal.bounds(np.array([1.0]), np.array([1]))
        assert upper.tolist() == [2.0]

    def test_horizon_edges_are_six_hours_one_day_two_days(self):
        assert horizon_edges_in_steps("30min") == (12, 48, 96)
        assert horizon_edges_in_steps("1h") == (6, 24, 48)


class TestCalibration:
    def test_an_unbiased_model_reaches_its_nominal_coverage_out_of_sample(
        self, poisson_panel
    ):
        cal = _calibrate(poisson_panel, Oracle())
        assert cal is not None and cal.holdout_coverage is not None
        assert cal.holdout_coverage == pytest.approx(0.8, abs=0.05)

    def test_a_model_biased_low_gets_an_interval_shifted_up(self, poisson_panel):
        """
        The case that mattered: a stale model under-forecasting a growing series.
        Poisson quantiles centred on the low forecast miss above; calibrated ones
        learn the bias from the errors and still cover.
        """
        cal = _calibrate(poisson_panel, Oracle(bias=0.7))
        assert cal is not None and cal.holdout_coverage is not None
        assert cal.holdout_coverage == pytest.approx(0.8, abs=0.06)
        assert min(min(row) for row in cal.upper_z) > 0  # every upper quantile above the forecast

    def test_origins_are_spread_over_the_window_after_the_cut(self, poisson_panel):
        start = pd.Timestamp("2021-01-08")
        cal = _calibrate(poisson_panel, Oracle(), start=start, days=10)
        assert cal is not None
        assert cal.origins == 12
        assert pd.Timestamp(cal.window_start) == start
        assert pd.Timestamp(cal.window_end) <= start + pd.Timedelta(days=10)

    def test_too_little_data_past_the_cut_returns_none(self, poisson_panel):
        assert _calibrate(
            poisson_panel, Oracle(), start=poisson_panel[TS_COL].max(),
        ) is None

    def test_its_summary_goes_into_the_metrics(self, poisson_panel):
        cal = _calibrate(poisson_panel, Oracle())
        assert cal is not None
        summary = cal.summary()
        assert summary["interval_level"] == 0.8
        assert summary["interval_origins"] == 12.0
        assert "interval_holdout_coverage" in summary


class TestTheForecastCarriesThem:
    def _forecast(self, panel, bundle, **kwargs):
        start = pd.Timestamp("2021-01-15")
        horizon = pd.date_range(start, periods=6, freq=FREQ)
        return forecast_recursive(bundle, panel[panel[TS_COL] < start], horizon, **kwargs)

    def test_a_calibrated_bundle_adds_lower_and_upper(self, poisson_panel):
        cal = _calibrate(poisson_panel, Oracle())
        out = self._forecast(poisson_panel, _bundle(Oracle(), cal))
        assert {LOWER_COL, UPPER_COL} <= set(out.columns)
        assert (out[LOWER_COL] <= out[PREDICTION_COL].round()).all()
        assert (out[UPPER_COL] >= out[PREDICTION_COL].round()).all()

    def test_a_bundle_without_a_calibration_adds_nothing(self, poisson_panel):
        """Every bundle saved before intervals existed."""
        out = self._forecast(poisson_panel, _bundle(Oracle()))
        assert LOWER_COL not in out.columns

    def test_they_can_be_left_off(self, poisson_panel):
        cal = _calibrate(poisson_panel, Oracle())
        out = self._forecast(poisson_panel, _bundle(Oracle(), cal), with_intervals=False)
        assert LOWER_COL not in out.columns
