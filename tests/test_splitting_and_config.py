"""Tests for chronological splitting, configuration wiring, and the registry."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from ML_Pipeline.config import ModelRegistry, PipelineConfig
from ML_Pipeline.splitting import chronological_split, train_validation_split


class TestChronologicalSplit:
    def test_no_train_row_postdates_a_test_row(self, panel):
        """The whole point: fit on the past, score on the future."""
        split = chronological_split(panel, test_fraction=0.2)
        assert split.train["ts"].max() < split.test["ts"].min()

    def test_split_is_on_the_timeline_not_row_count(self, panel):
        """Each cluster must contribute the same period to each side."""
        split = chronological_split(panel, test_fraction=0.25)
        per_cluster = split.test.groupby("pickup_cluster")["ts"].nunique()
        assert per_cluster.nunique() == 1

    def test_every_row_is_allocated(self, panel):
        split = chronological_split(panel, test_fraction=0.2)
        assert len(split.train) + len(split.test) == len(panel)

    def test_test_fraction_is_approximately_honoured(self, panel):
        split = chronological_split(panel, test_fraction=0.25)
        share = split.test["ts"].nunique() / panel["ts"].nunique()
        assert share == pytest.approx(0.25, abs=0.02)

    def test_does_not_split_on_day_of_month(self, panel):
        """
        Regression for the old `day <= 23` / `day > 23` split, which put test
        weeks between training weeks in every month.
        """
        split = chronological_split(panel, test_fraction=0.2)
        test_days = set(pd.to_datetime(split.test["ts"]).dt.day)
        train_days = set(pd.to_datetime(split.train["ts"]).dt.day)
        # A pure day-of-month split would make these disjoint.
        assert test_days & train_days

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_fraction_raises(self, panel, bad):
        with pytest.raises(ValueError, match="test_fraction"):
            chronological_split(panel, test_fraction=bad)

    def test_single_timestamp_raises(self):
        df = pd.DataFrame({"ts": ["2021-03-27"], "request_count": [1.0]})
        with pytest.raises(ValueError, match="at least 2 distinct"):
            chronological_split(df)


class TestTrainValidationSplit:
    def test_validation_tail_is_chronological(self, panel):
        inner = train_validation_split(panel, validation_fraction=0.1)
        assert inner.train["ts"].max() < inner.test["ts"].min()

    def test_zero_fraction_yields_empty_validation(self, panel):
        inner = train_validation_split(panel, validation_fraction=0.0)
        assert len(inner.test) == 0
        assert len(inner.train) == len(panel)


class TestPipelineConfig:
    def test_defaults_are_valid(self, tmp_path):
        config = PipelineConfig(output_dir=str(tmp_path), logs_dir=str(tmp_path))
        config.validate()
        assert config.n_clusters >= 1

    def test_objective_suits_a_count_target(self, tmp_path):
        config = PipelineConfig(output_dir=str(tmp_path), logs_dir=str(tmp_path))
        assert config.xgb_params["objective"] == "count:poisson"

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"n_clusters": 0}, "n_clusters"),
            ({"test_fraction": 0}, "test_fraction"),
            ({"test_fraction": 1.0}, "test_fraction"),
            ({"validation_fraction": 1.0}, "validation_fraction"),
            ({"interval_minutes": 0}, "interval_minutes"),
            ({"lag_features": (0, 1)}, "lag_features"),
            ({"rolling_window": 0}, "rolling_window"),
            ({"clustering_algorithm": "dbscan"}, "clustering_algorithm"),
        ],
    )
    def test_invalid_settings_fail_fast(self, tmp_path, kwargs, message):
        with pytest.raises(ValueError, match=message):
            PipelineConfig(output_dir=str(tmp_path), logs_dir=str(tmp_path), **kwargs)

    def test_model_and_registry_paths_agree(self, tmp_path):
        """
        Regression: `get_model_path` returned a versioned filename while the
        pipeline wrote an unversioned one, so every registry entry pointed at a
        file that did not exist.
        """
        config = PipelineConfig(output_dir=str(tmp_path), logs_dir=str(tmp_path))
        for kind in ("without_lag", "with_lag", "clustering"):
            path = config.get_model_path(kind)
            assert config.model_version in path
            assert path.endswith(".joblib")

    def test_roundtrip_through_json(self, tmp_path):
        config = PipelineConfig(
            output_dir=str(tmp_path), logs_dir=str(tmp_path), n_clusters=17
        )
        loaded = PipelineConfig.load_config(config.save_config())
        assert loaded.n_clusters == 17
        assert loaded.model_version == config.model_version

    def test_unknown_keys_are_ignored_on_load(self, tmp_path):
        path = tmp_path / "cfg.json"
        path.write_text(
            json.dumps({"n_clusters": 9, "retired_setting": True, "output_dir": str(tmp_path)}),
            encoding="utf-8",
        )
        assert PipelineConfig.load_config(str(path)).n_clusters == 9

    def test_snapshot_contains_every_field(self, tmp_path):
        config = PipelineConfig(output_dir=str(tmp_path), logs_dir=str(tmp_path))
        snapshot = config.to_dict()
        for field_name in ("lag_features", "rolling_window", "test_fraction",
                           "use_cluster_centroids", "early_stopping_rounds", "freq"):
            assert field_name in snapshot, f"{field_name} missing from the snapshot"


class TestModelRegistry:
    def test_best_model_selection_works_with_metrics(self, tmp_path):
        """
        Regression: the pipeline never passed metrics, so every entry stored `{}`
        and `get_best_model` always returned None.
        """
        registry = ModelRegistry(str(tmp_path / "registry.json"))
        for name, rmse in (("a", 2.0), ("b", 1.0)):
            registry.register_model(
                name, str(tmp_path / f"{name}.joblib"), "xgboost",
                metrics={"test_rmse": rmse},
            )

        best = registry.get_best_model("xgboost", metric="test_rmse")
        assert best is not None and best[0] == "b"

    def test_higher_is_better_metrics(self, tmp_path):
        registry = ModelRegistry(str(tmp_path / "registry.json"))
        for name, r2 in (("a", 0.4), ("b", 0.8)):
            registry.register_model(
                name, str(tmp_path / f"{name}.joblib"), "xgboost", metrics={"test_r2": r2}
            )
        best = registry.get_best_model("xgboost", metric="test_r2", higher_is_better=True)
        assert best[0] == "b"

    def test_returns_none_when_no_metrics_recorded(self, tmp_path):
        registry = ModelRegistry(str(tmp_path / "registry.json"))
        registry.register_model("a", str(tmp_path / "a.joblib"), "xgboost")
        assert registry.get_best_model("xgboost", metric="test_rmse") is None

    def test_corrupt_registry_does_not_crash(self, tmp_path):
        path = tmp_path / "registry.json"
        path.write_text("{not json", encoding="utf-8")
        assert ModelRegistry(str(path)).registry == {}

    def test_persists_across_instances(self, tmp_path):
        path = str(tmp_path / "registry.json")
        ModelRegistry(path).register_model(
            "a", str(tmp_path / "a.joblib"), "xgboost", metrics={"test_rmse": 1.0}
        )
        assert "a" in ModelRegistry(path).list_models()
