"""
Tests for the deploy gate and per-cluster error reporting.

`compare_to_baselines` was written, documented and unit-tested — and called by
nothing outside the test suite. The check that decides whether a model is fit to
ship had to be remembered and run by hand, which during the audit meant writing
a throwaway script to find out that the shipped model was losing to a free
baseline. It now runs on every training pass.

These exercise `_run_deploy_gate` directly rather than through `model_training`,
which would mean fitting real models for each case. The function is the unit of
behaviour being specified, so it is tested as one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ML_Pipeline.config import PipelineConfig
from ML_Pipeline.modeling.evaluation import ModelEvaluator
from ML_Pipeline.modeling.features import CLUSTER_COL, TARGET_COL, ModelBundle
from ML_Pipeline.modeling.training import _run_deploy_gate, _season_length

WEEK_30MIN = 336


class Stub:
    """A model that returns a fixed vector, so the gate's verdict is controllable."""

    def __init__(self, values: np.ndarray):
        self.values = np.asarray(values, dtype="float64")

    def predict(self, X):
        return self.values[: len(X)]


def bundle_for(values: np.ndarray) -> ModelBundle:
    return ModelBundle(
        model=Stub(values),
        feature_names=["hour"],
        uses_lags=True,
        lags=(1, 2, 3),
    )


@pytest.fixture
def gate_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """The 14-day panel, long enough for a same-time-last-week baseline."""
    return panel.sort_values([CLUSTER_COL, "ts"]).reset_index(drop=True)


class TestSeasonLength:
    def test_a_week_of_half_hours(self):
        assert _season_length(PipelineConfig(freq="30min")) == WEEK_30MIN

    def test_tracks_the_configured_frequency(self):
        assert _season_length(PipelineConfig(freq="60min")) == 168
        assert _season_length(PipelineConfig(freq="15min")) == 672


class TestGateVerdict:
    def test_a_perfect_model_passes(self, gate_panel):
        bundle = bundle_for(gate_panel[TARGET_COL].to_numpy(dtype="float64"))
        _run_deploy_gate(bundle, gate_panel, PipelineConfig())
        assert bundle.metrics["beats_seasonal_naive"] == 1.0
        assert bundle.metrics["baseline_mase"] == pytest.approx(0.0, abs=1e-9)

    def test_a_useless_model_fails(self, gate_panel):
        """Predicting zero everywhere must not be allowed to ship."""
        bundle = bundle_for(np.zeros(len(gate_panel)))
        _run_deploy_gate(bundle, gate_panel, PipelineConfig())
        assert bundle.metrics["beats_seasonal_naive"] == 0.0
        assert bundle.metrics["baseline_mase"] > 1.0

    def test_merely_reproducing_the_baseline_does_not_pass(self, gate_panel):
        """
        A model that *is* the baseline earns nothing over it.

        The gate compares RMSE strictly, so a tie fails — which is the right call
        when the alternative costs nothing to run.
        """
        naive = ModelEvaluator.seasonal_naive_baseline(
            gate_panel, season_length=WEEK_30MIN
        ).to_numpy(dtype="float64")
        bundle = bundle_for(np.nan_to_num(naive))
        _run_deploy_gate(bundle, gate_panel, PipelineConfig())
        assert bundle.metrics["beats_seasonal_naive"] == 0.0

    def test_the_verdict_is_recorded_for_the_registry(self, gate_panel):
        bundle = bundle_for(gate_panel[TARGET_COL].to_numpy(dtype="float64"))
        _run_deploy_gate(bundle, gate_panel, PipelineConfig())
        for key in (
            "baseline_mase",
            "baseline_naive_rmse",
            "beats_seasonal_naive",
            "baseline_season_length",
            "clusters_losing_to_naive",
            "worst_cluster_mase",
        ):
            assert key in bundle.metrics, f"{key} missing; it would not reach the registry"


class TestGateCannotBeEvaluated:
    def test_a_window_shorter_than_the_season_records_no_verdict(self, gate_panel):
        """
        "Not measured" must stay distinguishable from "passed".

        With fewer intervals than the seasonal period, every naive value is NaN
        and there is nothing to compare against. Recording a pass there would be
        worse than recording nothing.
        """
        short = gate_panel[gate_panel["ts"] < gate_panel["ts"].min() + pd.Timedelta("2D")]
        bundle = bundle_for(short[TARGET_COL].to_numpy(dtype="float64"))
        _run_deploy_gate(bundle, short, PipelineConfig())
        assert "beats_seasonal_naive" not in bundle.metrics

    def test_a_feature_mismatch_is_survived(self, gate_panel):
        """The gate must not take the training run down with it."""
        bundle = ModelBundle(
            model=Stub(np.zeros(len(gate_panel))),
            feature_names=["a_column_that_does_not_exist"],
            uses_lags=True,
        )
        _run_deploy_gate(bundle, gate_panel, PipelineConfig())
        assert "beats_seasonal_naive" not in bundle.metrics


class TestPerClusterError:
    def test_one_row_per_cluster(self, gate_panel):
        out = ModelEvaluator.per_cluster_error(
            gate_panel, gate_panel[TARGET_COL].to_numpy(dtype="float64"),
            season_length=WEEK_30MIN,
        )
        assert len(out) == gate_panel[CLUSTER_COL].nunique()
        assert set(out[CLUSTER_COL]) == set(gate_panel[CLUSTER_COL])

    def test_sorted_worst_first(self, gate_panel):
        rng = np.random.default_rng(0)
        preds = gate_panel[TARGET_COL].to_numpy(dtype="float64") + rng.normal(
            0, 3, len(gate_panel)
        )
        out = ModelEvaluator.per_cluster_error(
            gate_panel, preds, season_length=WEEK_30MIN
        )
        mase = out["mase"].dropna().to_numpy()
        assert (np.diff(mase) <= 1e-12).all(), "worst cluster must come first"

    def test_it_isolates_a_single_bad_cluster(self, gate_panel):
        """
        The reason this exists: a global RMSE hides which areas get under-served.
        """
        preds = gate_panel[TARGET_COL].to_numpy(dtype="float64").copy()
        victim = gate_panel[CLUSTER_COL].unique()[1]
        preds[(gate_panel[CLUSTER_COL] == victim).to_numpy()] = 0.0

        out = ModelEvaluator.per_cluster_error(
            gate_panel, preds, season_length=WEEK_30MIN
        ).set_index(CLUSTER_COL)
        assert out.loc[victim, "mase"] > 1.0
        others = out.drop(index=victim)["mase"]
        assert (others < 1e-6).all(), "only the sabotaged cluster should look bad"

    def test_level_ratio_reports_systematic_under_forecasting(self, gate_panel):
        preds = gate_panel[TARGET_COL].to_numpy(dtype="float64") * 0.5
        out = ModelEvaluator.per_cluster_error(
            gate_panel, preds, season_length=WEEK_30MIN
        )
        assert out["level_ratio"].between(0.49, 0.51).all()
