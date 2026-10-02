"""
Tests for the command-line wiring in `run_pipeline.py`.

Nothing here fits a model or reads real data: these exercise how `--config` and
the flags combine into a `PipelineConfig`. That path had no coverage, which is
how two defects survived a full audit of the pipeline - naming `--config` sent
every other flag down a branch that never ran, and the default test-data path
pointed at a directory that does not exist, so the forecasting stage warned and
skipped on every default run while the process still exited 0.
"""

from __future__ import annotations

import pytest

from ML_Pipeline.config import PipelineConfig
from run_pipeline import build_parser, config_from_args


def parse(*argv: str):
    return build_parser().parse_args(list(argv))


@pytest.fixture
def snapshot(tmp_path):
    """A saved config snapshot with distinctive, non-default values."""
    config = PipelineConfig(
        output_dir=str(tmp_path),
        logs_dir=str(tmp_path),
        n_clusters=7,
        test_fraction=0.3,
        test_data_path="data/from_the_snapshot.csv",
    )
    return config.save_config()


class TestFlagsWithoutAConfig:
    def test_flags_reach_the_config(self):
        config = config_from_args(
            parse("--n-clusters", "123", "--test-fraction", "0.35", "--horizon-steps", "96")
        )
        assert config.n_clusters == 123
        assert config.test_fraction == 0.35
        assert config.horizon_steps == 96

    def test_omitted_flags_fall_back_to_the_dataclass_defaults(self):
        config = config_from_args(parse())
        defaults = PipelineConfig()
        assert config.n_clusters == defaults.n_clusters
        assert config.test_fraction == defaults.test_fraction
        assert config.raw_data_path == defaults.raw_data_path
        assert config.test_data_path == defaults.test_data_path

    def test_no_centroids_disables_centroid_features(self):
        assert config_from_args(parse()).use_cluster_centroids is True
        assert config_from_args(parse("--no-centroids")).use_cluster_centroids is False

    def test_cluster_diagnostics_is_off_unless_asked_for(self):
        assert config_from_args(parse()).run_cluster_diagnostics is False
        assert config_from_args(parse("--cluster-diagnostics")).run_cluster_diagnostics is True


class TestConfigAndFlagsCompose:
    def test_config_alone_is_honoured(self, snapshot):
        config = config_from_args(parse("--config", snapshot))
        assert config.n_clusters == 7
        assert config.test_fraction == 0.3
        assert config.test_data_path == "data/from_the_snapshot.csv"

    def test_a_flag_overrides_the_snapshot(self, snapshot):
        """The regression: --config used to discard every other flag."""
        config = config_from_args(parse("--config", snapshot, "--n-clusters", "100"))
        assert config.n_clusters == 100, "the flag must win over the loaded snapshot"
        assert config.test_fraction == 0.3, "untouched snapshot values must survive"

    def test_several_flags_override_together(self, snapshot):
        config = config_from_args(
            parse(
                "--config", snapshot,
                "--n-clusters", "11",
                "--test-data", "data/from_the_flag.csv",
                "--horizon-steps", "48",
            )
        )
        assert config.n_clusters == 11
        assert config.test_data_path == "data/from_the_flag.csv"
        assert config.horizon_steps == 48
        assert config.test_fraction == 0.3

    def test_absent_flags_do_not_clobber_the_snapshot_with_none(self, snapshot):
        config = config_from_args(parse("--config", snapshot))
        assert config.test_data_path is not None
        assert config.raw_data_path is not None
        assert config.log_level == "INFO"

    def test_store_true_flag_overrides_a_snapshot(self, snapshot):
        assert config_from_args(parse("--config", snapshot)).use_cluster_centroids is True
        overridden = config_from_args(parse("--config", snapshot, "--no-centroids"))
        assert overridden.use_cluster_centroids is False

    def test_the_snapshot_model_version_is_preserved(self, snapshot):
        """Overriding a flag must not re-stamp the version and orphan artifacts."""
        loaded = PipelineConfig.load_config(snapshot)
        config = config_from_args(parse("--config", snapshot, "--n-clusters", "100"))
        assert config.model_version == loaded.model_version


class TestOverridesAreValidated:
    @pytest.mark.parametrize("flag,value", [("--test-fraction", "1.5"), ("--n-clusters", "0")])
    def test_an_invalid_override_fails_fast(self, snapshot, flag, value):
        with pytest.raises(ValueError):
            config_from_args(parse("--config", snapshot, flag, value))


class TestDefaultPaths:
    def test_the_test_data_default_is_where_the_file_actually_lives(self):
        """
        Guards the second defect: this default was
        `data/test_dataset/cleaned_test_booking_data.csv`, a directory that does
        not exist in the repository layout, so `stage_6_predictions` logged
        "Test data not found" and skipped on every default run.
        """
        assert PipelineConfig().test_data_path == "data/cleaned_test_booking_data.csv"

    def test_the_cli_does_not_carry_its_own_copy_of_the_path_defaults(self):
        """One source of truth, so the two cannot drift apart again."""
        defaults = {action.dest: action.default for action in build_parser()._actions}
        assert defaults["test_data"] is None
        assert defaults["raw_data"] is None
