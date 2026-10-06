"""The experiments' shared settings must follow PipelineConfig, not a copy of it."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from ML_Pipeline.config import PipelineConfig

COMMON = Path(__file__).resolve().parents[1] / "experiments" / "common.py"


def _load_common():
    spec = importlib.util.spec_from_file_location("experiments_common", COMMON)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lags_and_window_come_from_config():
    common = _load_common()
    config = PipelineConfig()
    assert common.LAGS == tuple(config.lag_features)
    assert common.ROLLING_WINDOW == config.rolling_window
    assert [f"lag_{lag}" for lag in common.LAGS] == [
        f for f in common.FEATURES if f.startswith("lag_")
    ]


def test_params_match_config_except_tree_count_and_objective():
    common = _load_common()
    expected = {k: v for k, v in PipelineConfig().xgb_params.items() if k != "objective"}
    expected["n_estimators"] = common.SWEEP_N_ESTIMATORS
    assert common.BASE_PARAMS == expected
    assert "objective" not in common.BASE_PARAMS
