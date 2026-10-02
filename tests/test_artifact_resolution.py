"""
Tests for resolving the newest pipeline artefacts.

This is the coverage whose absence let three defects of one shape ship: a
hardcoded path that no test exercised. The dashboard looked for
`output/Data_Prepared.csv` while the pipeline wrote
`Data_Prepared_<version>.csv`, so a completely successful run rendered "No
prepared demand data found"; the CLI defaulted `test_data_path` to a directory
that does not exist, so the forecasting stage skipped on every run.

The resolver and the path builders now share `DATA_STEMS`, and these tests pin
both halves of that agreement.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from ML_Pipeline.config import (
    DATA_STEMS,
    PipelineConfig,
    latest_artifact,
    latest_version,
)


def touch(directory, name: str) -> None:
    (directory / name).write_text("ts,pickup_cluster,request_count\n", encoding="utf-8")


class TestLatestArtifact:
    def test_the_newest_version_wins(self, tmp_path):
        for version in ("20250101_010101", "20260102_030405", "20250630_120000"):
            touch(tmp_path, f"Data_Prepared_{version}.csv")
        found = latest_artifact(tmp_path, "prepared")
        assert found is not None
        assert found.name == "Data_Prepared_20260102_030405.csv"

    def test_version_strings_sort_chronologically_as_text(self, tmp_path):
        """`%Y%m%d_%H%M%S` sorts lexicographically, so no date parsing is needed."""
        touch(tmp_path, "Data_Prepared_20260102_030405.csv")
        touch(tmp_path, "Data_Prepared_20260102_030406.csv")
        assert latest_artifact(tmp_path, "prepared").name.endswith("030406.csv")

    def test_does_not_depend_on_filesystem_timestamps(self, tmp_path):
        """Copying a directory rewrites mtimes; the version in the name survives."""
        touch(tmp_path, "Data_Prepared_20260102_030405.csv")
        touch(tmp_path, "Data_Prepared_20250101_010101.csv")
        # Make the OLD run the most recently modified file.
        stale = tmp_path / "Data_Prepared_20250101_010101.csv"
        os.utime(stale, (time.time() + 1000, time.time() + 1000))
        assert latest_artifact(tmp_path, "prepared").name.endswith("20260102_030405.csv")

    @pytest.mark.parametrize("data_type", sorted(DATA_STEMS))
    def test_every_artefact_type_resolves(self, tmp_path, data_type):
        touch(tmp_path, f"{DATA_STEMS[data_type]}_20260102_030405.csv")
        found = latest_artifact(tmp_path, data_type)
        assert found is not None, f"{data_type} did not resolve"
        assert found.name == f"{DATA_STEMS[data_type]}_20260102_030405.csv"

    def test_with_lag_does_not_match_without_lag(self, tmp_path):
        """The two stems share a prefix; a sloppy glob would confuse them."""
        touch(tmp_path, "data_without_lag_20260102_030405.csv")
        assert latest_artifact(tmp_path, "with_lag") is None
        touch(tmp_path, "data_with_lag_20260102_030405.csv")
        assert latest_artifact(tmp_path, "with_lag").name.startswith("data_with_lag_")

    def test_unversioned_legacy_file_is_still_found(self, tmp_path):
        touch(tmp_path, "Data_Prepared.csv")
        assert latest_artifact(tmp_path, "prepared").name == "Data_Prepared.csv"

    def test_versioned_beats_legacy(self, tmp_path):
        touch(tmp_path, "Data_Prepared.csv")
        touch(tmp_path, "Data_Prepared_20260102_030405.csv")
        assert latest_artifact(tmp_path, "prepared").name.endswith("030405.csv")

    def test_empty_directory_returns_none(self, tmp_path):
        """So the dashboard can show its empty state rather than invent numbers."""
        assert latest_artifact(tmp_path, "prepared") is None

    def test_missing_directory_returns_none(self, tmp_path):
        assert latest_artifact(tmp_path / "nope", "prepared") is None


class TestLatestVersion:
    def test_reads_the_newest_config_snapshot(self, tmp_path):
        for version in ("20250101_010101", "20260102_030405"):
            (tmp_path / f"pipeline_config_{version}.json").write_text("{}", encoding="utf-8")
        assert latest_version(tmp_path) == "20260102_030405"

    def test_none_when_no_run_has_completed(self, tmp_path):
        assert latest_version(tmp_path) is None


class TestResolverAgreesWithTheWriter:
    def test_what_the_config_writes_is_what_the_resolver_finds(self, tmp_path):
        """
        The regression guard: writer and reader must agree on the filename.

        Builds each path the way the pipeline does, creates that exact file, and
        asserts the resolver returns it. This is the assertion whose absence let
        the dashboard and the pipeline disagree indefinitely.
        """
        config = PipelineConfig(output_dir=str(tmp_path), logs_dir=str(tmp_path))
        for data_type in DATA_STEMS:
            written = Path(config.get_data_path(data_type))
            touch(tmp_path, written.name)
            found = latest_artifact(tmp_path, data_type)
            assert found is not None, f"resolver missed {data_type}"
            assert str(found) == str(written), (
                f"{data_type}: writer wrote {written}, resolver found {found}"
            )
