"""
Tests for the final refit, and for staleness measured from the training data.

The promoted model used to be the one the deploy gate scored, which by
construction had never seen the test window - the newest fifth of the timeline.
On the reference dataset that made it ten weeks stale on the day it was trained,
and it lost to seasonal-naive on the busiest cluster while the dashboard reported
its age as zero days. These specify both halves of the fix: the saved model is
refitted on everything, and staleness is measured from where its data ends.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from joblib import dump, load
from test_serving_api import build_output_dir

from ML_Pipeline.config import PipelineConfig
from ML_Pipeline.modeling.features import TS_COL
from ML_Pipeline.modeling.training import model_training
from ML_Pipeline.serving.state import STALE_AFTER_DAYS, ServingState

#: Absolute, because AppTest resolves a relative path against the calling file
#: in some streamlit versions and against the working directory in others.
APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def _train(panel: pd.DataFrame, tmp_path, *, refit: bool):
    # The fixture panel is 14 days; a 60% test window is the shortest that
    # still spans the week the deploy gate's seasonal-naive baseline needs.
    config = PipelineConfig(
        output_dir=str(tmp_path), n_clusters=4, use_cluster_centroids=False,
        lag_features=(1, 2, 3), refit_on_all_data=refit, test_fraction=0.6,
        xgb_params={"n_estimators": 20, "max_depth": 3, "n_jobs": 2},
        early_stopping_rounds=5,
        # Interval calibration is tested in test_intervals.py; the default 24
        # backtests would add ~15 seconds to every training run here.
        interval_origins=4,
    )
    return model_training(
        panel, str(tmp_path / "nolag.joblib"), str(tmp_path / "lag.joblib"),
        config=config,
    )


class TestFinalRefit:
    def test_the_saved_model_has_seen_the_newest_data(self, panel, tmp_path):
        bundles = _train(panel, tmp_path, refit=True)
        newest = panel[TS_COL].max()
        for name, bundle in bundles.items():
            assert pd.Timestamp(bundle.data_through) == newest, name
        assert load(tmp_path / "lag.joblib").data_through == newest.isoformat()

    def test_it_trains_on_every_row(self, panel, tmp_path):
        bundles = _train(panel, tmp_path, refit=True)
        assert bundles["without_lag"].training_rows == len(panel)
        assert (
            bundles["with_lag"].metrics["final_fit_rows"]
            == bundles["with_lag"].training_rows
        )

    def test_evaluation_still_comes_from_the_held_out_fit(self, panel, tmp_path):
        """The gate verdict must survive the refit, or promotion would refuse it."""
        refit = _train(panel, tmp_path / "a", refit=True)["with_lag"]
        held = _train(panel, tmp_path / "b", refit=False)["with_lag"]
        for key in ("test_rmse", "baseline_mase", "beats_seasonal_naive"):
            assert refit.metrics[key] == pytest.approx(held.metrics[key]), key

    def test_the_tree_count_is_the_one_early_stopping_chose(self, panel, tmp_path):
        bundle = _train(panel, tmp_path, refit=True)["with_lag"]
        assert bundle.model.n_estimators == int(bundle.metrics["selected_n_estimators"])

    def test_without_it_the_model_stops_at_the_split(self, panel, tmp_path):
        bundle = _train(panel, tmp_path, refit=False)["with_lag"]
        assert pd.Timestamp(bundle.data_through) < panel[TS_COL].max()


def _state(tmp_path, *, data_through: str = "", notes: str = "") -> ServingState:
    build_output_dir(tmp_path)
    path = next(tmp_path.glob("prediction_model_with_lag_*.joblib"))
    bundle = load(path)
    bundle.data_through, bundle.notes = data_through, notes
    dump(bundle, path)
    return ServingState(PipelineConfig(output_dir=str(tmp_path), logs_dir=str(tmp_path)))


class TestDataLag:
    def test_measured_from_where_the_training_data_ends(self, tmp_path):
        state = _state(tmp_path, data_through="2021-01-03T00:00:00")
        expected = (state.history_ends_at - pd.Timestamp("2021-01-03")).days
        assert state.data_lag_days == pytest.approx(expected, abs=1)

    def test_a_model_trained_today_on_old_data_is_stale(self, tmp_path):
        """The case the wall-clock age missed."""
        state = _state(tmp_path, data_through="2020-10-01T00:00:00")
        assert state.age_days == 0
        assert state.data_lag_days > STALE_AFTER_DAYS
        assert state.stale is True
        assert "training data ends" in state.staleness_warning()

    def test_a_current_model_is_not_stale(self, tmp_path):
        state = _state(tmp_path, data_through="2021-01-09T00:00:00")
        assert state.stale is False
        assert state.staleness_warning() is None

    def test_older_bundles_recover_it_from_their_split_note(self, tmp_path):
        state = _state(tmp_path, notes="Chronological split at 2020-11-01 05:00:00.")
        assert state.data_through == pd.Timestamp("2020-11-01 05:00:00")
        assert state.stale is True

    def test_unknown_when_nothing_records_it(self, tmp_path):
        state = _state(tmp_path)
        assert state.data_through is None
        assert state.data_lag_days is None
        assert state.stale is False

    def test_the_api_reports_it(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient

        from ML_Pipeline.serving.api import app

        state = _state(tmp_path, data_through="2020-10-01T00:00:00")
        monkeypatch.setattr("ML_Pipeline.serving.api.get_state", lambda: state)
        client = TestClient(app)

        info = client.get("/model").json()
        assert info["stale"] is True
        assert info["data_lag_days"] > STALE_AFTER_DAYS
        assert datetime.fromisoformat(info["data_through"]) == datetime(2020, 10, 1)

        body = client.get("/forecast", params={"steps": 4}).json()
        assert body["data_lag_days"] == pytest.approx(info["data_lag_days"])
        assert any("training data ends" in w for w in body["warnings"])


class TestTheDashboardStaysOutOfSample:
    """
    After the refit, the promoted model has been fitted on the whole history, so
    the page's backtest and scoring window would be in-sample - flattering numbers
    presented as evidence. The page must score only demand the model has not seen.
    """

    def _page(self, tmp_path, monkeypatch, *, data_through):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        from test_dashboard_performance import SEASON, WITH_LAGS, build

        # The serving state is a process-wide resource cache; without this a test
        # scores whichever output directory an earlier test loaded.
        st.cache_resource.clear()
        st.cache_data.clear()

        build(tmp_path, intervals=SEASON * 3, feature_names=WITH_LAGS)
        path = next(tmp_path.glob("prediction_model_with_lag_*.joblib"))
        bundle = load(path)
        history_end = pd.read_csv(next(tmp_path.glob("Data_Prepared_*.csv.gz")))[
            TS_COL
        ].max()
        bundle.data_through = data_through(pd.Timestamp(history_end)).isoformat()
        dump(bundle, path)

        monkeypatch.setenv("BIKETAXI_OUTPUT_DIR", str(tmp_path))
        at = AppTest.from_file(str(APP), default_timeout=120)
        at.run()
        at.sidebar.radio[0].set_value("Model performance").run()
        assert not at.exception
        return at

    def test_a_freshly_refit_model_is_not_scored_on_its_training_data(
        self, tmp_path, monkeypatch
    ):
        at = self._page(tmp_path, monkeypatch, data_through=lambda end: end)
        assert any("has not seen" in i.value for i in at.info)
        labels = [m.label for m in at.metric]
        assert "Level ratio" not in labels  # the backtest did not run
        verdicts = [e.value for e in [*at.success, *at.error]]
        assert not any("baseline" in v for v in verdicts)  # nor did the comparison

    def test_new_demand_is_scored_and_only_that(self, tmp_path, monkeypatch):
        at = self._page(
            tmp_path, monkeypatch, data_through=lambda end: end - pd.Timedelta(days=9)
        )
        assert any("9 days since the model's training data ends" in s.value
                   for s in [*at.success, *at.error])
