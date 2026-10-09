"""
Tests for the booking input contract.

Nothing used to check the input. A file that broke the documented schema was
cleaned into something smaller, and the run trained on what was left. These
pin both halves of the check: a broken file stops the run and says why, and a
few bad rows, which real data has, do not.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from ML_Pipeline.config import PipelineConfig
from ML_Pipeline.data.contract import (
    FORECAST_COLUMNS,
    MAX_BAD_SHARE,
    InputContractError,
    MissingColumnsError,
    check_bookings,
)
from ML_Pipeline.pipeline import MLPipeline

N = 1_000


def bookings(n: int = N, seed: int = 3) -> pd.DataFrame:
    """A file that meets the contract: Bengaluru pickups, schema-format stamps."""
    rng = np.random.default_rng(seed)
    stamps = pd.Timestamp("2021-01-01") + pd.to_timedelta(rng.integers(0, 86_400 * 7, n), unit="s")
    return pd.DataFrame({
        "ts": stamps.strftime("%Y-%m-%d %H:%M:%S"),
        "number": rng.integers(1, 500, n).astype(str),
        "pick_lat": rng.uniform(12.85, 13.10, n),
        "pick_lng": rng.uniform(77.45, 77.75, n),
        "drop_lat": rng.uniform(12.85, 13.10, n),
        "drop_lng": rng.uniform(77.45, 77.75, n),
    })


def spoil(df: pd.DataFrame, column: str, value, rows: int) -> pd.DataFrame:
    out = df.copy()
    out[column] = out[column].astype(object)
    out.loc[: rows - 1, column] = value
    return out


class TestAFileThatMeetsTheContract:
    def test_passes_and_reports_its_rows(self):
        report = check_bookings(bookings(), source="ok.csv")
        assert report.rows == N
        assert not any(report.bad_rows.values())
        assert report.outside_service_area == 0.0

    def test_a_few_bad_rows_are_logged_not_fatal(self, caplog):
        """The reference dataset has 121 unparseable rider ids in 8.4 million."""
        few = int(N * MAX_BAD_SHARE)  # exactly at the threshold
        with caplog.at_level(logging.WARNING):
            report = check_bookings(spoil(bookings(), "number", "abc", few), source="f.csv")
        assert report.bad_rows["number not numeric"] == few
        assert "number not numeric: 10" in caplog.text

    def test_a_forecast_file_needs_no_rider_column(self):
        df = bookings().drop(columns=["number", "drop_lat", "drop_lng"])
        check_bookings(df, source="test.csv", required=FORECAST_COLUMNS)

    def test_a_forecast_file_is_not_judged_on_columns_it_does_not_use(self):
        df = spoil(bookings(), "number", "garbage", N)
        check_bookings(df, source="test.csv", required=FORECAST_COLUMNS)


class TestABrokenFileStopsTheRun:
    def test_a_missing_column_is_named(self):
        with pytest.raises(MissingColumnsError, match="pick_lng") as caught:
            check_bookings(bookings().drop(columns=["pick_lng"]), source="m.csv")
        assert isinstance(caught.value, KeyError)  # what the old checks raised
        assert not str(caught.value).startswith("'")

    def test_an_empty_file(self):
        with pytest.raises(InputContractError, match="no rows"):
            check_bookings(bookings().iloc[:0], source="e.csv")

    def test_timestamps_in_another_format(self):
        """These used to become NaT and be dropped one row at a time."""
        df = bookings()
        df["ts"] = pd.to_datetime(df["ts"]).dt.strftime("%d/%m/%Y %H:%M")
        with pytest.raises(InputContractError, match="ts not in"):
            check_bookings(df, source="dates.csv")

    def test_more_than_the_threshold_of_bad_rows(self):
        many = int(N * MAX_BAD_SHARE) + 1
        with pytest.raises(InputContractError, match=r"number not numeric: 11 of 1,000"):
            check_bookings(spoil(bookings(), "number", "abc", many), source="n.csv")

    @pytest.mark.parametrize("value", [np.nan, 95.0, -91.0])
    def test_missing_or_impossible_latitudes(self, value):
        with pytest.raises(InputContractError, match="drop_lat"):
            check_bookings(spoil(bookings(), "drop_lat", value, 50), source="c.csv")

    def test_swapped_latitude_and_longitude(self):
        """Every pickup lands outside Bengaluru, and Rule 6 would drop them all."""
        df = bookings().rename(columns={"pick_lat": "pick_lng", "pick_lng": "pick_lat"})
        with pytest.raises(InputContractError, match="swapped"):
            check_bookings(df, source="swapped.csv")

    def test_the_service_area_check_can_be_turned_off(self):
        df = bookings().rename(columns={"pick_lat": "pick_lng", "pick_lng": "pick_lat"})
        report = check_bookings(df, source="elsewhere.csv", service_area=None)
        assert report.outside_service_area is None


class TestThePipelineChecksAtLoad:
    def test_stage_1_refuses_a_broken_raw_file(self, tmp_path):
        raw = tmp_path / "raw.csv.gz"
        df = bookings()
        df["ts"] = pd.to_datetime(df["ts"]).dt.strftime("%d/%m/%Y %H:%M")
        df.to_csv(raw, index=False, compression="gzip")
        pipeline = MLPipeline(config=PipelineConfig(
            raw_data_path=str(raw), output_dir=str(tmp_path), logs_dir=str(tmp_path),
        ))
        with pytest.raises(InputContractError, match="ts not in"):
            pipeline.stage_1_load_data()

    def test_the_run_command_exits_2_with_the_reason(self, tmp_path):
        """Bad input, like a missing file: a reason, not exit 1 and a trace."""
        from ML_Pipeline.cli.run import main

        raw = tmp_path / "raw.csv.gz"
        df = bookings().rename(columns={"pick_lat": "pick_lng", "pick_lng": "pick_lat"})
        df.to_csv(raw, index=False, compression="gzip")
        code = main([
            "--raw-data", str(raw), "--output", str(tmp_path / "out"),
            "--log-file", str(tmp_path / "run.log"),
        ])
        assert code == 2
        assert "swapped" in (tmp_path / "run.log").read_text(encoding="utf-8")
