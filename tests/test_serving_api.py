"""
Tests for model promotion and the forecast serving API.

Nothing in this repository could serve a forecast before: no API, no container,
no schedule. These specify the two behaviours that make serving trustworthy
rather than merely present — that a model which failed its deploy gate cannot
reach production, and that the horizon is capped where the model's measured
advantage over a free baseline runs out.

The API is exercised through `TestClient`, so routing, validation and
serialisation are all covered rather than just the functions behind them.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from joblib import dump

from ML_Pipeline.api import MAX_HORIZON_STEPS, STALE_AFTER_DAYS, ServingState, app
from ML_Pipeline.config import ModelRegistry, PipelineConfig
from ML_Pipeline.features import CLUSTER_COL, TARGET_COL, TS_COL, ModelBundle

N_CLUSTERS = 3
INTERVALS = 400  # > 336, so a weekly lag has history to read


class Flat:
    """A model returning a constant, so forecast values are predictable."""

    def __init__(self, value: float = 4.0):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value, dtype="float64")


def build_output_dir(tmp_path, *, gate_passed=True, trained_days_ago=0, promote=True):
    """A realistic output directory: a grid, a bundle, and a registry entry."""
    version = "20260102_030405"
    stamps = pd.date_range("2021-01-01", periods=INTERVALS, freq="30min")
    grid = pd.DataFrame(
        [
            {TS_COL: t, CLUSTER_COL: c, TARGET_COL: float(2 + (c + t.hour) % 7)}
            for c in range(N_CLUSTERS)
            for t in stamps
        ]
    )
    grid.to_csv(tmp_path / f"Data_Prepared_{version}.csv.gz", index=False,
                compression="gzip")

    bundle = ModelBundle(
        model=Flat(),
        feature_names=["lag_1", "lag_336"],
        uses_lags=True,
        lags=(1, 336),
        rolling_window=3,
        freq="30min",
        trained_at=(datetime.now() - timedelta(days=trained_days_ago)).isoformat(),
        metrics={
            "test_rmse": 3.5,
            "baseline_mase": 0.8 if gate_passed else 1.2,
            "beats_seasonal_naive": 1.0 if gate_passed else 0.0,
            "clusters_losing_to_naive": 0.0 if gate_passed else 3.0,
        },
    )
    model_path = tmp_path / f"prediction_model_with_lag_{version}.joblib"
    dump(bundle, model_path)

    registry = ModelRegistry(str(tmp_path / "model_registry.json"))
    name = f"xgb_with_lag_{version}"
    registry.register_model(
        model_name=name,
        model_path=str(model_path),
        model_type="xgboost",
        metrics=bundle.metrics,
        metadata={"version": version},
    )
    if promote:
        registry.promote_model(name, force=not gate_passed)
    return name, version


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient wired to a freshly built output directory."""
    build_output_dir(tmp_path)
    state = ServingState(PipelineConfig(output_dir=str(tmp_path), logs_dir=str(tmp_path)))
    monkeypatch.setattr("ML_Pipeline.api.get_state", lambda: state)
    return TestClient(app)


class TestPromotion:
    def test_a_promoted_model_is_what_serving_resolves(self, tmp_path):
        name, _ = build_output_dir(tmp_path)
        registry = ModelRegistry(str(tmp_path / "model_registry.json"))
        promoted = registry.production_model()
        assert promoted is not None
        assert promoted[0] == name
        assert promoted[1]["stage"] == "production"

    def test_a_model_that_failed_its_gate_cannot_be_promoted(self, tmp_path):
        """
        The point of promotion. Serving the newest model would serve one that a
        free seasonal-naive baseline beats.
        """
        name, _ = build_output_dir(tmp_path, gate_passed=False, promote=False)
        registry = ModelRegistry(str(tmp_path / "model_registry.json"))
        with pytest.raises(ValueError, match="did not beat its seasonal-naive"):
            registry.promote_model(name)
        assert registry.production_model() is None

    def test_a_failed_gate_can_be_forced_with_a_reason(self, tmp_path):
        name, _ = build_output_dir(tmp_path, gate_passed=False, promote=False)
        registry = ModelRegistry(str(tmp_path / "model_registry.json"))
        registry.promote_model(name, force=True)
        assert registry.production_model()[0] == name

    def test_a_model_with_no_gate_verdict_cannot_be_promoted(self, tmp_path):
        """
        Unverified is not the same as passing.

        This was caught against the real registry: the pre-gate August model
        carries no verdict, and an earlier version of `promote_model` let it
        through on a warning - putting into production the very model the audit
        found losing to the baseline at MASE 0.999.
        """
        build_output_dir(tmp_path, promote=False)
        registry = ModelRegistry(str(tmp_path / "model_registry.json"))
        registry.register_model(
            model_name="legacy_no_verdict",
            model_path=str(tmp_path / "legacy.joblib"),
            model_type="xgboost",
            metrics={"test_rmse": 3.0},  # no beats_seasonal_naive
        )
        with pytest.raises(ValueError, match="no deploy-gate verdict"):
            registry.promote_model("legacy_no_verdict")
        assert registry.production_model() is None

        registry.promote_model("legacy_no_verdict", force=True)
        assert registry.production_model()[0] == "legacy_no_verdict"

    def test_promoting_an_unregistered_model_is_an_error(self, tmp_path):
        registry = ModelRegistry(str(tmp_path / "model_registry.json"))
        with pytest.raises(KeyError, match="not registered"):
            registry.promote_model("nothing_like_this")

    def test_only_one_model_holds_production(self, tmp_path):
        build_output_dir(tmp_path)
        registry = ModelRegistry(str(tmp_path / "model_registry.json"))
        registry.register_model(
            model_name="xgb_with_lag_20260202_030405",
            model_path=str(tmp_path / "x.joblib"),
            model_type="xgboost",
            metrics={"test_rmse": 3.0, "beats_seasonal_naive": 1.0},
        )
        registry.promote_model("xgb_with_lag_20260202_030405")
        stages = [e.get("stage") for e in registry.registry.values()]
        assert stages.count("production") == 1
        assert stages.count("archived") == 1

    def test_rollback_restores_the_previous_model(self, tmp_path):
        first, _ = build_output_dir(tmp_path)
        registry = ModelRegistry(str(tmp_path / "model_registry.json"))
        registry.register_model(
            model_name="xgb_with_lag_20260202_030405",
            model_path=str(tmp_path / "x.joblib"),
            model_type="xgboost",
            metrics={"test_rmse": 3.0, "beats_seasonal_naive": 1.0},
        )
        registry.promote_model("xgb_with_lag_20260202_030405")
        rolled = registry.rollback()
        assert rolled is not None
        assert rolled[0] == first

    def test_rollback_with_nothing_archived_returns_none(self, tmp_path):
        build_output_dir(tmp_path)
        registry = ModelRegistry(str(tmp_path / "model_registry.json"))
        assert registry.rollback() is None

    def test_promotion_survives_a_reload(self, tmp_path):
        name, _ = build_output_dir(tmp_path)
        again = ModelRegistry(str(tmp_path / "model_registry.json"))
        assert again.production_model()[0] == name
        stored = json.loads((tmp_path / "model_registry.json").read_text())
        assert stored[name]["stage"] == "production"


class TestHealthAndModel:
    def test_health_reports_readiness(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["ready"] is True

    def test_model_endpoint_describes_what_is_serving(self, client):
        body = client.get("/model").json()
        assert body["stage"] == "production"
        assert body["beats_seasonal_naive"] is True
        assert body["baseline_mase"] == pytest.approx(0.8)
        assert body["lags"] == [1, 336]
        assert body["max_horizon_steps"] == MAX_HORIZON_STEPS

    def test_clusters_are_listed(self, client):
        body = client.get("/clusters").json()
        assert body["count"] == N_CLUSTERS


class TestForecast:
    def test_a_default_request_returns_one_day_per_cluster(self, client):
        body = client.get("/forecast").json()
        assert body["horizon_steps"] == 48
        assert len(body["forecast"]) == 48 * N_CLUSTERS

    def test_the_forecast_starts_after_the_last_observation(self, client):
        body = client.get("/forecast?steps=2").json()
        ends = pd.Timestamp(body["history_ends_at"])
        first = pd.Timestamp(body["forecast"][0]["ts"])
        assert first == ends + pd.Timedelta("30min")

    def test_no_negative_demand_is_served(self, client):
        body = client.get("/forecast?steps=4").json()
        assert all(p["predicted_requests"] >= 0 for p in body["forecast"])

    def test_a_cluster_filter_is_honoured(self, client):
        body = client.get("/forecast?steps=4&cluster=1").json()
        assert {p["pickup_cluster"] for p in body["forecast"]} == {1}
        assert len(body["forecast"]) == 4

    def test_an_unknown_cluster_is_rejected_with_the_valid_range(self, client):
        response = client.get("/forecast?steps=2&cluster=99")
        assert response.status_code == 400
        assert "99" in response.json()["detail"]

    def test_the_horizon_is_capped_where_the_measured_advantage_runs_out(self, client):
        """
        Past two days the model loses to seasonal-naive. Serving a longer horizon
        quietly would hand out forecasts worse than doing nothing.
        """
        response = client.get(f"/forecast?steps={MAX_HORIZON_STEPS + 1}")
        assert response.status_code == 422

    def test_a_horizon_past_one_day_is_served_but_warned_about(self, client):
        body = client.get("/forecast?steps=60").json()
        assert body["forecast"]
        assert any("horizon" in w for w in body["warnings"])

    def test_a_one_day_horizon_carries_no_warning(self, client):
        assert client.get("/forecast?steps=48").json()["warnings"] == []


class TestStaleness:
    def test_an_old_model_is_flagged_and_warned_about(self, tmp_path, monkeypatch):
        build_output_dir(tmp_path, trained_days_ago=STALE_AFTER_DAYS + 10)
        state = ServingState(
            PipelineConfig(output_dir=str(tmp_path), logs_dir=str(tmp_path))
        )
        monkeypatch.setattr("ML_Pipeline.api.get_state", lambda: state)
        client = TestClient(app)

        assert client.get("/model").json()["stale"] is True
        body = client.get("/forecast?steps=4").json()
        assert body["stale"] is True
        assert any("retraining cadence" in w for w in body["warnings"])

    def test_a_fresh_model_is_not_flagged(self, client):
        assert client.get("/model").json()["stale"] is False


KEY = "test-key-123"


class TestAuthentication:
    """Optional: required when BIKETAXI_API_KEY is set, open otherwise."""

    def test_without_a_configured_key_the_api_is_open(self, client, monkeypatch):
        monkeypatch.delenv("BIKETAXI_API_KEY", raising=False)
        assert client.get("/forecast?steps=2").status_code == 200

    @pytest.mark.parametrize("path", ["/forecast?steps=2", "/model", "/clusters"])
    def test_with_a_key_configured_a_request_without_it_is_refused(
        self, client, monkeypatch, path
    ):
        monkeypatch.setenv("BIKETAXI_API_KEY", KEY)
        assert client.get(path).status_code == 401
        assert client.get(path, headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get(path, headers={"X-API-Key": KEY}).status_code == 200

    def test_health_stays_open_for_the_container_healthcheck(self, client, monkeypatch):
        monkeypatch.setenv("BIKETAXI_API_KEY", KEY)
        assert client.get("/health").status_code == 200


def add_promoted_model(tmp_path, name="xgb_with_lag_20260202_030405"):
    """Register and promote a second model, as a retrain would."""
    source = next(tmp_path.glob("prediction_model_with_lag_*.joblib"))
    target = tmp_path / f"prediction_model_with_lag_{name[-15:]}.joblib"
    target.write_bytes(source.read_bytes())
    registry = ModelRegistry(str(tmp_path / "model_registry.json"))
    registry.register_model(
        model_name=name, model_path=str(target), model_type="xgboost",
        metrics={"test_rmse": 3.0, "beats_seasonal_naive": 1.0},
    )
    registry.promote_model(name)
    return name


@pytest.fixture
def live(tmp_path, monkeypatch):
    """The real process-wide state, pointed at a temporary output directory."""
    monkeypatch.setenv("BIKETAXI_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr("ML_Pipeline.serving._state", None)
    return TestClient(app)


class TestReload:
    def test_a_newly_promoted_model_is_served_without_a_restart(
        self, tmp_path, live, monkeypatch
    ):
        first, _ = build_output_dir(tmp_path)
        monkeypatch.setenv("BIKETAXI_API_KEY", KEY)
        headers = {"X-API-Key": KEY}
        assert live.get("/model", headers=headers).json()["model_name"] == first

        second = add_promoted_model(tmp_path)
        assert live.get("/model", headers=headers).json()["model_name"] == first, (
            "precondition: without a reload the old model keeps serving"
        )
        body = live.post("/reload", headers=headers).json()
        assert body["reloaded"] is True
        assert body["model_name"] == second
        assert body["previous_model_name"] == first
        assert live.get("/model", headers=headers).json()["model_name"] == second

    def test_a_reload_that_cannot_serve_keeps_the_running_model(
        self, tmp_path, live, monkeypatch
    ):
        first, _ = build_output_dir(tmp_path)
        monkeypatch.setenv("BIKETAXI_API_KEY", KEY)
        headers = {"X-API-Key": KEY}
        live.get("/model", headers=headers)
        for grid in tmp_path.glob("Data_Prepared_*"):
            grid.unlink()

        response = live.post("/reload", headers=headers)
        assert response.status_code == 409
        assert first in response.json()["detail"]
        assert live.get("/model", headers=headers).json()["model_name"] == first
        assert live.get("/forecast?steps=2", headers=headers).status_code == 200

    def test_reload_is_disabled_on_an_open_api(self, tmp_path, live, monkeypatch):
        """An open demo must not re-read its data for anyone who asks."""
        build_output_dir(tmp_path)
        monkeypatch.delenv("BIKETAXI_API_KEY", raising=False)
        assert live.post("/reload").status_code == 403

    def test_reload_needs_the_key(self, tmp_path, live, monkeypatch):
        build_output_dir(tmp_path)
        monkeypatch.setenv("BIKETAXI_API_KEY", KEY)
        assert live.post("/reload").status_code == 401


class TestAccessLog:
    def test_each_request_is_logged_with_status_and_model(self, client, caplog):
        with caplog.at_level("INFO", logger="ML_Pipeline.api.access"):
            client.get("/forecast?steps=2")
        line = next(r.getMessage() for r in caplog.records
                    if r.name == "ML_Pipeline.api.access")
        assert line.startswith("GET /forecast 200 ")
        assert "model=xgb_with_lag_20260102_030405" in line


class TestRegistryPortability:
    def test_a_registry_written_on_windows_loads_elsewhere(self, tmp_path):
        """
        The real registry held `output\\prediction_model_...joblib`. In a Linux
        container that is not a path to anything, so the API served nothing.
        """
        name, version = build_output_dir(tmp_path)
        registry_file = tmp_path / "model_registry.json"
        stored = json.loads(registry_file.read_text())
        stored[name]["model_path"] = (
            f"some\\other\\machine\\output\\prediction_model_with_lag_{version}.joblib"
        )
        registry_file.write_text(json.dumps(stored))

        state = ServingState(
            PipelineConfig(output_dir=str(tmp_path), logs_dir=str(tmp_path))
        )
        assert state.ready
        assert state.model_name == name


class TestNotReady:
    def test_without_a_promoted_model_serving_refuses_clearly(self, tmp_path, monkeypatch):
        """503, not 500, and it says what is missing."""
        build_output_dir(tmp_path, promote=False)
        state = ServingState(
            PipelineConfig(output_dir=str(tmp_path), logs_dir=str(tmp_path))
        )
        monkeypatch.setattr("ML_Pipeline.api.get_state", lambda: state)
        client = TestClient(app)

        assert client.get("/health").json()["ready"] is False
        assert client.get("/model").status_code == 503
        response = client.get("/forecast")
        assert response.status_code == 503
        assert "no model promoted" in response.json()["detail"]
