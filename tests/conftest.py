"""
Shared fixtures.

All fixtures build data in-memory. The repository ships no data file - the
booking-level source carries personal data and is deliberately git-ignored - so
the test suite must be able to run on a fresh clone with nothing downloaded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ML_Pipeline.features import (
    add_calendar_features,
    add_lag_features,
    build_demand_grid,
    build_feature_names,
)

N_CLUSTERS = 4
FREQ = "30min"

#: Demand shape by hour, used to give the synthetic data a realistic profile
#: (a morning and an evening peak) rather than uniform noise.
HOUR_PROFILE = np.array(
    [0.2, 0.1, 0.05, 0.05, 0.1, 0.3, 0.8, 1.6, 2.2, 1.8, 1.3, 1.2,
     1.4, 1.3, 1.2, 1.4, 1.8, 2.4, 2.6, 2.1, 1.5, 1.0, 0.6, 0.35]
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20240803)


@pytest.fixture
def bookings(rng: np.random.Generator) -> pd.DataFrame:
    """Booking-level rows: timestamp plus pickup coordinates."""
    n = 20_000
    minutes = rng.integers(0, 14 * 24 * 60, n)
    centres = np.array([[12.90, 77.55], [12.98, 77.62], [12.93, 77.70], [13.02, 77.58]])
    which = rng.integers(0, len(centres), n)
    coords = centres[which] + rng.normal(0, 0.012, (n, 2))
    return pd.DataFrame(
        {
            "ts": pd.Timestamp("2021-01-01") + pd.to_timedelta(minutes, unit="m"),
            "pick_lat": coords[:, 0],
            "pick_lng": coords[:, 1],
            "pickup_cluster": which,
        }
    )


@pytest.fixture
def panel(bookings: pd.DataFrame) -> pd.DataFrame:
    """Rectangular demand grid with calendar features."""
    grid = build_demand_grid(bookings, freq=FREQ, clusters=list(range(N_CLUSTERS)))
    return add_calendar_features(grid, "ts")


@pytest.fixture
def lagged_panel(panel: pd.DataFrame) -> pd.DataFrame:
    return add_lag_features(panel, lags=(1, 2, 3), rolling_window=3)


@pytest.fixture
def feature_names_nolag() -> list[str]:
    return build_feature_names(use_lags=False)


@pytest.fixture
def feature_names_lag() -> list[str]:
    return build_feature_names(use_lags=True, lags=(1, 2, 3))


@pytest.fixture
def small_panel() -> pd.DataFrame:
    """
    A tiny, fully deterministic panel: 2 clusters x 10 intervals.

    Values are chosen so lag expectations can be asserted by hand.
    """
    stamps = pd.date_range("2021-03-27", periods=10, freq=FREQ)
    return pd.DataFrame(
        {
            "ts": list(stamps) * 2,
            "pickup_cluster": [0] * 10 + [1] * 10,
            "request_count": [float(v) for v in range(10)]
            + [float(100 + v) for v in range(10)],
        }
    )
