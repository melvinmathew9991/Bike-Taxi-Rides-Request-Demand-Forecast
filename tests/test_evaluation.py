"""Tests for evaluation metrics on an intermittent count target."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ML_Pipeline.evaluation import ModelEvaluator, PredictionValidator


class TestCalculateMetrics:
    def test_perfect_predictions(self):
        y = np.array([0.0, 1.0, 2.0, 3.0])
        m = ModelEvaluator.calculate_metrics(y, y)
        assert m["rmse"] == pytest.approx(0.0)
        assert m["mae"] == pytest.approx(0.0)
        assert m["r2"] == pytest.approx(1.0)

    def test_percentage_error_does_not_explode_on_zeros(self):
        """
        Regression: the old `mean(|y-yhat| / (y + 1e-8)) * 100` returned ~1e10
        on a target whose median is 0, and that value was persisted into the
        model registry as a percentage.
        """
        y_true = np.zeros(100)
        y_pred = np.full(100, 0.5)
        m = ModelEvaluator.calculate_metrics(y_true, y_pred)
        assert "mean_absolute_percentage_error" not in m
        assert np.isnan(m["mape"])
        assert m["mape_coverage"] == 0.0
        for key, value in m.items():
            if isinstance(value, float) and not np.isnan(value):
                assert abs(value) < 1e6, f"{key} is implausible: {value}"

    def test_mape_computed_over_nonzero_actuals(self):
        y_true = np.concatenate([np.zeros(50), np.full(50, 10.0)])
        y_pred = np.concatenate([np.zeros(50), np.full(50, 11.0)])
        m = ModelEvaluator.calculate_metrics(y_true, y_pred)
        assert m["mape"] == pytest.approx(0.1, abs=1e-9)
        assert m["mape_coverage"] == pytest.approx(0.5)

    def test_reports_sparsity_and_negative_predictions(self):
        y_true = np.array([0.0, 0.0, 0.0, 4.0])
        y_pred = np.array([-1.0, 0.5, 0.5, 3.0])
        m = ModelEvaluator.calculate_metrics(y_true, y_pred)
        assert m["zero_actual_share"] == pytest.approx(0.75)
        assert m["negative_predictions"] == 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            ModelEvaluator.calculate_metrics([1, 2, 3], [1, 2])

    def test_empty_input_returns_empty(self):
        assert ModelEvaluator.calculate_metrics([], []) == {}


class TestPoissonDeviance:
    def test_zero_for_perfect_predictions(self):
        y = np.array([0.0, 1.0, 5.0, 10.0])
        assert ModelEvaluator.poisson_deviance(y, y) == pytest.approx(0.0, abs=1e-6)

    def test_increases_as_predictions_worsen(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        near = ModelEvaluator.poisson_deviance(y, y + 0.1)
        far = ModelEvaluator.poisson_deviance(y, y + 2.0)
        assert far > near

    def test_penalises_proportionally(self):
        """A miss of 2 on an expected 1 must cost more than on an expected 50."""
        small = ModelEvaluator.poisson_deviance(np.array([1.0]), np.array([3.0]))
        large = ModelEvaluator.poisson_deviance(np.array([50.0]), np.array([52.0]))
        assert small > large


class TestBaselines:
    def _panel(self, weeks: int = 3) -> pd.DataFrame:
        stamps = pd.date_range("2021-01-04", periods=336 * weeks, freq="30min")
        # Perfectly weekly-periodic demand: seasonal naive should be exact.
        values = np.tile(np.arange(336, dtype="float64"), weeks)
        return pd.DataFrame(
            {"ts": stamps, "pickup_cluster": 0, "request_count": values}
        )

    def test_seasonal_naive_is_exact_on_periodic_data(self):
        panel = self._panel()
        naive = ModelEvaluator.seasonal_naive_baseline(panel, season_length=336)
        valid = naive.notna()
        assert np.allclose(naive[valid], panel.loc[valid, "request_count"])

    def test_mase_below_one_when_model_beats_naive(self):
        panel = self._panel()
        naive = ModelEvaluator.seasonal_naive_baseline(panel, season_length=336).to_numpy()
        actual = panel["request_count"].to_numpy()
        # Deliberately degrade the naive comparison by scoring against a noisy one.
        noisy_naive = naive + 5.0
        assert ModelEvaluator.mase(actual, actual, noisy_naive) == pytest.approx(0.0)

    def test_compare_to_baselines_reports_all_approaches(self):
        panel = self._panel()
        preds = panel["request_count"].to_numpy() + 1.0
        result = ModelEvaluator.compare_to_baselines(panel, preds, season_length=336)
        assert set(result.index) >= {"model", "seasonal_naive", "cluster_mean"}
        assert (result["rmse"] >= 0).all()

    def test_warns_when_model_loses_to_naive(self, caplog):
        panel = self._panel()
        terrible = np.full(len(panel), 999.0)
        with caplog.at_level("WARNING"):
            ModelEvaluator.compare_to_baselines(panel, terrible, season_length=336)
        assert "does not beat a seasonal-naive" in caplog.text


class TestPredictionValidator:
    def test_flags_negative_predictions(self):
        report = PredictionValidator.check_prediction_bounds(np.array([-1.0, 0.0, 5.0]))
        assert report["out_of_bounds_low"] == 1
        assert report["min_prediction"] == -1.0

    def test_stability_on_constant_predictions(self):
        report = PredictionValidator.check_prediction_stability(np.ones(50), window_size=10)
        assert report["mean_stability"] == pytest.approx(1.0, abs=1e-6)

    def test_validate_predictions_bundles_all_checks(self):
        y_true = np.array([0.0, 1.0, 2.0] * 20)
        y_pred = y_true + 0.1
        result = PredictionValidator.validate_predictions(y_true, y_pred)
        assert set(result) == {
            "basic_metrics", "error_analysis", "bounds_check", "stability_check"
        }
