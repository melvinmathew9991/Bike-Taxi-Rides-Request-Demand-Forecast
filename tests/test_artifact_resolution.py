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

import gzip
import os
import time
from pathlib import Path

import pandas as pd
import pytest

from ML_Pipeline.artifacts import DATA_STEMS, latest_artifact, latest_version
from ML_Pipeline.config import PipelineConfig
from ML_Pipeline.utils import read_csv_any

CSV = "ts,pickup_cluster,request_count\n0,0,0\n"


def touch(directory, name: str) -> None:
    """Create an artefact, gzip-compressed when the name says it is."""
    path = directory / name
    if name.endswith(".gz"):
        path.write_bytes(gzip.compress(CSV.encode()))
    else:
        path.write_text(CSV, encoding="utf-8")


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


class TestCompressedNaming:
    """
    Outputs are `.csv.gz`; earlier runs wrote gzip under a bare `.csv`.

    Both must resolve, and the correctly-named one must win when both exist.
    """

    def test_a_csv_gz_artefact_resolves(self, tmp_path):
        touch(tmp_path, "Data_Prepared_20260102_030405.csv.gz")
        found = latest_artifact(tmp_path, "prepared")
        assert found is not None
        assert found.name.endswith(".csv.gz")

    def test_csv_gz_wins_over_a_legacy_csv_of_the_same_run(self, tmp_path):
        touch(tmp_path, "Data_Prepared_20260102_030405.csv")
        touch(tmp_path, "Data_Prepared_20260102_030405.csv.gz")
        assert latest_artifact(tmp_path, "prepared").name.endswith(".csv.gz")

    def test_a_legacy_csv_output_directory_still_works(self, tmp_path):
        """Output directories from before the rename must not go dark."""
        touch(tmp_path, "Data_Prepared_20250101_010101.csv")
        assert latest_artifact(tmp_path, "prepared").name.endswith("010101.csv")

    def test_the_newest_csv_gz_wins_among_several(self, tmp_path):
        for v in ("20250101_010101", "20260102_030405", "20250630_120000"):
            touch(tmp_path, f"Data_Prepared_{v}.csv.gz")
        assert latest_artifact(tmp_path, "prepared").name == (
            "Data_Prepared_20260102_030405.csv.gz"
        )

    def test_what_the_config_writes_carries_the_compressed_extension(self, tmp_path):
        config = PipelineConfig(output_dir=str(tmp_path), logs_dir=str(tmp_path))
        for data_type in DATA_STEMS:
            assert config.get_data_path(data_type).endswith(".csv.gz")

    def test_pandas_can_read_an_output_with_no_special_handling(self, tmp_path):
        """
        The point of the rename. A bare `.csv` holding gzip bytes failed a plain
        `pd.read_csv` with `UnicodeDecodeError: invalid start byte`, so every
        consumer needed a sniffing fallback.
        """
        touch(tmp_path, "Data_Prepared_20260102_030405.csv.gz")
        path = latest_artifact(tmp_path, "prepared")
        frame = pd.read_csv(path)
        assert list(frame.columns) == ["ts", "pickup_cluster", "request_count"]


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


class TestReadCsvAny:
    """
    One reader, replacing four near-identical copies.

    Each copy was a `try` on gzip with a fallback to plain, needed because
    pipeline outputs were gzip under a bare `.csv`. Detection is now by content -
    a gzip member starts with `1f 8b` - rather than by catching an exception from
    a full parse.
    """

    def test_reads_gzip_hiding_under_a_csv_name(self, tmp_path):
        """The reference dataset's own shape: `data/raw_data.csv` is gzip."""
        path = tmp_path / "raw_data.csv"
        path.write_bytes(gzip.compress(CSV.encode()))
        frame = read_csv_any(path)
        assert list(frame.columns) == ["ts", "pickup_cluster", "request_count"]

    def test_reads_a_plain_csv(self, tmp_path):
        path = tmp_path / "plain.csv"
        path.write_text(CSV, encoding="utf-8")
        assert len(read_csv_any(path)) == 1

    def test_reads_a_correctly_named_csv_gz(self, tmp_path):
        path = tmp_path / "out.csv.gz"
        path.write_bytes(gzip.compress(CSV.encode()))
        assert len(read_csv_any(path)) == 1

    @pytest.mark.parametrize("compressed", [True, False], ids=["gzip", "plain"])
    def test_reads_an_uploaded_file_object(self, compressed):
        """The dashboard's upload path hands over a file object, not a path."""
        import io as _io

        payload = gzip.compress(CSV.encode()) if compressed else CSV.encode()
        handle = _io.BytesIO(payload)
        frame = read_csv_any(handle)
        assert list(frame.columns) == ["ts", "pickup_cluster", "request_count"]

    def test_a_file_object_is_left_rewound_for_the_parser(self):
        import io as _io

        handle = _io.BytesIO(CSV.encode())
        read_csv_any(handle)
        # Sniffing must not consume the stream the parser needs.
        handle.seek(0)
        assert handle.read(2) == b"ts"

    def test_a_malformed_csv_reports_itself_rather_than_as_not_gzip(self, tmp_path):
        """
        The old `except (OSError, EOFError, ValueError)` wrapped a full parse, so
        a genuine CSV error was swallowed and retried as plain text, surfacing as
        something unrelated. A parse error should now reach the caller.
        """
        path = tmp_path / "bad.csv"
        path.write_text('a,b\n"unterminated,1\n2,3,4,5\n', encoding="utf-8")
        with pytest.raises(pd.errors.ParserError):
            read_csv_any(path, engine="c", on_bad_lines="error")
