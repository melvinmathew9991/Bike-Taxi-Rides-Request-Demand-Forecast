"""
Tests for `biketaxi stage-demo`, which decides what goes to the cloud.

The output directory mixes what serving needs with booking-level personal data.
These pin that staging copies the allow-list and nothing else.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from joblib import dump, load
from test_serving_api import build_output_dir

from ML_Pipeline.cli.stage_demo import main as stage_main
from ML_Pipeline.config import PipelineConfig
from ML_Pipeline.serving.state import ServingState


@pytest.fixture
def stage():
    return stage_main


@pytest.fixture
def output_dir(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    _, version = build_output_dir(out)
    # A clusterer as the pipeline saves it: centres plus a label per booking.
    dump(
        SimpleNamespace(cluster_centers_=np.zeros((3, 2)), labels_=np.arange(1000)),
        out / f"pickup_cluster_model_{version}.joblib",
    )
    # Booking-level personal data, which must never be staged.
    pd.DataFrame({"number": [1], "pick_lat": [12.9], "pick_lng": [77.6]}).to_csv(
        out / f"clean_data_{version}.csv.gz", index=False, compression="gzip"
    )
    return out


def test_only_the_allow_list_is_staged(output_dir, tmp_path, stage):
    staged = tmp_path / "staged"
    assert stage(["--output-dir", str(output_dir), "--out", str(staged)]) == 0
    names = sorted(p.name for p in staged.iterdir())
    assert names == sorted([
        "Data_Prepared_20260102_030405.csv.gz",
        "model_registry.json",
        "pickup_cluster_model_20260102_030405.joblib",
        "prediction_model_with_lag_20260102_030405.joblib",
    ])
    assert not any(n.startswith("clean_data") for n in names)


def test_the_clusterer_keeps_its_centres_and_loses_the_per_booking_labels(
    output_dir, tmp_path, stage
):
    staged = tmp_path / "staged"
    stage(["--output-dir", str(output_dir), "--out", str(staged)])
    clusterer = load(staged / "pickup_cluster_model_20260102_030405.joblib")
    assert clusterer.cluster_centers_.shape == (3, 2)
    assert not hasattr(clusterer, "labels_")


def test_history_is_trimmed(output_dir, tmp_path, stage):
    staged = tmp_path / "staged"
    stage(["--output-dir", str(output_dir), "--out", str(staged), "--history-days", "7"])
    grid = pd.read_csv(staged / "Data_Prepared_20260102_030405.csv.gz",
                       parse_dates=["ts"])
    assert grid["ts"].max() - grid["ts"].min() < pd.Timedelta(days=7)


def test_a_grid_with_a_personal_column_is_refused(output_dir, tmp_path, stage):
    grid_path = output_dir / "Data_Prepared_20260102_030405.csv.gz"
    grid = pd.read_csv(grid_path)
    grid["pick_lat"] = 12.9
    grid.to_csv(grid_path, index=False, compression="gzip")
    with pytest.raises(SystemExit, match="pick_lat"):
        stage(["--output-dir", str(output_dir), "--out", str(tmp_path / "staged")])


def test_the_staged_directory_serves(output_dir, tmp_path, stage):
    staged = tmp_path / "staged"
    stage(["--output-dir", str(output_dir), "--out", str(staged),
           "--history-days", "8"])
    registry = json.loads((staged / "model_registry.json").read_text())
    assert len(registry) == 1
    state = ServingState(PipelineConfig(output_dir=str(staged), logs_dir=str(tmp_path)))
    assert state.ready
