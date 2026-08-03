"""
End-to-end integration tests.

These run the real pipeline over synthetic booking data, so the stages are
exercised together rather than only in isolation. They are marked `slow` because
they fit models; deselect with `-m "not slow"`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ML_Pipeline.config import PipelineConfig
from ML_Pipeline.features import ModelBundle
from ML_Pipeline.model_training import model_training
from ML_Pipeline.pipeline import MLPipeline
from ML_Pipeline.prediction_pipeline import prediction_pipeline

pytestmark = pytest.mark.slow

N_CLUSTERS = 4
CENTRES = np.array([[12.90, 77.55], [12.98, 77.62], [12.93, 77.70], [13.02, 77.58]])
HOUR_WEIGHTS = np.array(
    [0.2, 0.1, 0.05, 0.05, 0.1, 0.3, 0.8, 1.6, 2.2, 1.8, 1.3, 1.2,
     1.4, 1.3, 1.2, 1.4, 1.8, 2.4, 2.6, 2.1, 1.5, 1.0, 0.6, 0.35]
)


def _bookings(n: int, start: str, days: int, seed: int) -> pd.DataFrame:
    """Synthetic booking-level rows with a realistic daily demand shape."""
    rng = np.random.default_rng(seed)
    hours = rng.choice(24, n, p=HOUR_WEIGHTS / HOUR_WEIGHTS.sum())
    offsets = rng.integers(0, days, n) * 24 * 60 + hours * 60 + rng.integers(0, 60, n)
    which = rng.integers(0, len(CENTRES), n)
    pick = CENTRES[which] + rng.normal(0, 0.010, (n, 2))
    drop = CENTRES[rng.integers(0, len(CENTRES), n)] + rng.normal(0, 0.010, (n, 2))
    return pd.DataFrame(
        {
            "ts": (pd.Timestamp(start) + pd.to_timedelta(offsets, unit="m")).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "number": rng.integers(1, 5000, n).astype(str),
            "pick_lat": pick[:, 0], "pick_lng": pick[:, 1],
            "drop_lat": drop[:, 0], "drop_lng": drop[:, 1],
        }
    )


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """A full pipeline run over synthetic data, shared by the tests below."""
    root = tmp_path_factory.mktemp("e2e")
    raw = root / "raw_data.csv"
    test = root / "test_bookings.csv"
    _bookings(60_000, "2020-03-26", 40, seed=11).to_csv(raw, index=False, compression="gzip")
    _bookings(8_000, "2020-05-06", 6, seed=12).to_csv(test, index=False, compression="gzip")

    config = PipelineConfig(
        raw_data_path=str(raw),
        test_data_path=str(test),
        output_dir=str(root / "output"),
        logs_dir=str(root / "logs"),
        n_clusters=N_CLUSTERS,
        horizon_steps=24,
        xgb_params={
            "objective": "count:poisson", "max_depth": 4, "learning_rate": 0.1,
            "n_estimators": 60, "random_state": 42, "n_jobs": 2,
        },
        early_stopping_rounds=10,
    )
    pipeline = MLPipeline(config=config)
    results = pipeline.run_full_pipeline()
    return {"config": config, "pipeline": pipeline, "results": results, "root": root}


class TestFullPipeline:
    def test_completes_successfully(self, workspace):
        assert workspace["results"]["status"] == "success"

    def test_trains_both_model_variants(self, workspace):
        models = workspace["results"]["models"]
        assert set(models) == {"without_lag", "with_lag"}
        assert all(isinstance(b, ModelBundle) for b in models.values())

    def test_returns_predictions_rather_than_none(self, workspace):
        """Regression: the prediction stage used to return None."""
        predictions = workspace["results"]["predictions"]
        assert predictions is not None
        assert set(predictions) == {"without_lag", "with_lag"}
        for frame in predictions.values():
            assert isinstance(frame, pd.DataFrame)
            assert not frame.empty

    def test_every_horizon_row_is_forecast(self, workspace):
        """Regression: only 3 timestamps used to be predicted."""
        for frame in workspace["results"]["predictions"].values():
            assert len(frame) == 24 * N_CLUSTERS
            assert frame["request_count_pred"].notna().all()
            assert frame["is_forecast"].all()

    def test_no_negative_demand_is_forecast(self, workspace):
        for frame in workspace["results"]["predictions"].values():
            assert (frame["request_count_pred"] >= 0).all()

    def test_metrics_are_recorded(self, workspace):
        for name, metrics in workspace["results"]["metrics"].items():
            assert "test_rmse" in metrics, f"{name} has no test RMSE"
            assert np.isfinite(metrics["test_rmse"])

    def test_metrics_are_plausible_magnitudes(self, workspace):
        """Guards against the exploding percentage-error metric returning."""
        for metrics in workspace["results"]["metrics"].values():
            for key, value in metrics.items():
                if isinstance(value, float) and np.isfinite(value):
                    assert abs(value) < 1e6, f"{key} is implausible: {value}"

    def test_artifacts_exist_where_the_registry_says(self, workspace):
        config = workspace["config"]
        for kind in ("without_lag", "with_lag", "clustering"):
            assert pd.io.common.file_exists(config.get_model_path(kind)), kind

    def test_config_snapshot_is_loadable(self, workspace):
        path = workspace["config"].save_config()
        assert PipelineConfig.load_config(path).n_clusters == N_CLUSTERS


class TestConfigIsHonoured:
    def test_n_clusters_reaches_the_clustering_stage(self, workspace):
        """
        Regression: `n_clusters` was ignored while the stage hardcoded 50, and
        the `--n-clusters` flag documented as an out-of-memory fix did nothing.
        """
        grid = workspace["pipeline"].df_processed
        assert grid["pickup_cluster"].nunique() == N_CLUSTERS

    def test_xgb_params_reach_the_trainer(self, workspace):
        """Regression: xgb_params was ignored while the trainer hardcoded its own."""
        bundle = workspace["results"]["models"]["without_lag"]
        assert bundle.params["max_depth"] == 4
        assert bundle.params["objective"] == "count:poisson"

    def test_horizon_steps_reaches_the_forecaster(self, workspace):
        for frame in workspace["results"]["predictions"].values():
            assert frame["ts"].nunique() == 24

    def test_centroid_features_are_used_by_default(self, workspace):
        bundle = workspace["results"]["models"]["without_lag"]
        assert "cluster_lat" in bundle.feature_names
        assert "pickup_cluster" not in bundle.feature_names

    def test_lag_settings_are_recorded_on_the_bundle(self, workspace):
        bundle = workspace["results"]["models"]["with_lag"]
        assert bundle.lags == (1, 2, 3)
        assert bundle.rolling_window == 3


class TestServingContract:
    def test_serving_uses_the_persisted_feature_order(self, workspace):
        """A model must never be handed columns in a different order than it saw."""
        config = workspace["config"]
        bundle = ModelBundle.load_bundle(config.get_model_path("with_lag"))
        assert bundle.feature_names[0] in {"cluster_lat", "pickup_cluster"}
        assert bundle.feature_names[-1] == "rolling_mean"

    def test_rerunning_prediction_is_reproducible(self, workspace):
        config = workspace["config"]
        out = prediction_pipeline(
            cleaned_data_path=config.test_data_path,
            cluster_model_path=config.get_model_path("clustering"),
            predict_without_lag_path=config.get_model_path("without_lag"),
            predict_with_lag_path=config.get_model_path("with_lag"),
            data_without_lag_path=str(workspace["root"] / "rerun_nolag.csv"),
            data_with_lag_path=str(workspace["root"] / "rerun_lag.csv"),
            horizon_steps=24,
        )
        first = workspace["results"]["predictions"]["with_lag"]["request_count_pred"]
        again = out["with_lag"]["request_count_pred"]
        assert np.allclose(first.to_numpy(), again.to_numpy())

    def test_missing_columns_are_reported_clearly(self, workspace, tmp_path):
        config = workspace["config"]
        bad = tmp_path / "bad.csv"
        pd.DataFrame({"ts": ["2020-05-06 00:00:00"]}).to_csv(
            bad, index=False, compression="gzip"
        )
        with pytest.raises(KeyError, match="pick_lat"):
            prediction_pipeline(
                cleaned_data_path=str(bad),
                cluster_model_path=config.get_model_path("clustering"),
                predict_without_lag_path=config.get_model_path("without_lag"),
                predict_with_lag_path=config.get_model_path("with_lag"),
                data_without_lag_path=str(tmp_path / "a.csv"),
                data_with_lag_path=str(tmp_path / "b.csv"),
            )


class TestTrainingDirectly:
    def test_lag_features_are_actually_used(self, panel):
        """
        Regression: the old trainer computed lag_1/2/3 and rolling_mean and then
        excluded every one of them from the feature list.
        """
        config = PipelineConfig(
            output_dir=".", n_clusters=4, use_cluster_centroids=False,
            xgb_params={"n_estimators": 20, "max_depth": 3, "n_jobs": 2},
            early_stopping_rounds=5,
        )
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            bundles = model_training(
                panel,
                str(Path(tmp) / "nolag.joblib"),
                str(Path(tmp) / "lag.joblib"),
                config=config,
            )
        lag_features = bundles["with_lag"].feature_names
        assert {"lag_1", "lag_2", "lag_3", "rolling_mean"}.issubset(lag_features)
        assert not {"lag_1", "rolling_mean"} & set(bundles["without_lag"].feature_names)


class TestEarlyStoppingRefit:
    """
    The validation tail used for early stopping is, by construction, the most
    recent data. Leaving it out of the final fit is costly on a trending series.
    """

    def _data(self):
        from ML_Pipeline.features import add_calendar_features
        stamps = pd.date_range("2021-01-01", periods=2000, freq="30min")
        df = add_calendar_features(pd.DataFrame({"ts": stamps}), "ts")
        # Strongly trending target, as in the real dataset. Clipped at 0 because
        # a count target cannot be negative (and Poisson rejects it).
        noise = np.random.default_rng(0).normal(0, 1, len(df))
        df["request_count"] = np.clip(np.linspace(1, 50, len(df)) + noise, 0, None)
        return df

    def test_refit_uses_every_training_row(self):
        from ML_Pipeline.xgb_model import train_xgb
        df = self._data()
        feats = ["hour", "dayofweek", "mins", "month", "quarter"]
        tr, va = df.iloc[:1500], df.iloc[1500:]
        params = {"n_estimators": 40, "max_depth": 3, "n_jobs": 2}

        refit = train_xgb(
            tr[feats], tr["request_count"], X_valid=va[feats], y_valid=va["request_count"],
            params=params, early_stopping_rounds=10, feature_names=feats,
        )
        assert refit.training_rows == len(tr) + len(va)

    def test_no_refit_option_trains_on_less(self):
        from ML_Pipeline.xgb_model import train_xgb
        df = self._data()
        feats = ["hour", "dayofweek", "mins", "month", "quarter"]
        tr, va = df.iloc[:1500], df.iloc[1500:]
        params = {"n_estimators": 40, "max_depth": 3, "n_jobs": 2}

        plain = train_xgb(
            tr[feats], tr["request_count"], X_valid=va[feats], y_valid=va["request_count"],
            params=params, early_stopping_rounds=10, feature_names=feats,
            refit_on_full_training=False,
        )
        assert plain.training_rows == len(tr)

    def test_validation_metrics_are_pre_refit(self):
        """After refitting, validation rows are in-sample; the recorded numbers
        must come from the model that had not yet seen them."""
        from ML_Pipeline.xgb_model import train_xgb
        df = self._data()
        feats = ["hour", "dayofweek", "mins", "month", "quarter"]
        tr, va = df.iloc[:1500], df.iloc[1500:]
        bundle = train_xgb(
            tr[feats], tr["request_count"], X_valid=va[feats], y_valid=va["request_count"],
            params={"n_estimators": 40, "max_depth": 3, "n_jobs": 2},
            early_stopping_rounds=10, feature_names=feats,
        )
        assert "valid_rmse" in bundle.metrics
        assert "selected_n_estimators" in bundle.metrics
        # An extrapolating trend the trees cannot follow => clearly non-trivial error.
        assert bundle.metrics["valid_rmse"] > 0
