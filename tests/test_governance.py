"""Tests for `ML_Pipeline.governance`, the one definition of where personal data stops."""

from __future__ import annotations

import pandas as pd
import pytest

from ML_Pipeline.governance import (
    GRID_COLUMNS,
    PERSONAL_DATA_COLUMNS,
    DataGovernanceError,
    assert_no_personal_data,
    columns_outside_grid,
)


def grid() -> pd.DataFrame:
    return pd.DataFrame(
        {"ts": pd.date_range("2021-01-01", periods=3, freq="30min"),
         "pickup_cluster": [0, 1, 2], "request_count": [1, 0, 4]}
    )


def test_the_aggregated_grid_passes():
    assert_no_personal_data(grid())


@pytest.mark.parametrize("column", sorted(PERSONAL_DATA_COLUMNS))
def test_any_personal_column_is_refused_and_named(column):
    df = grid().assign(**{column: 1})
    with pytest.raises(DataGovernanceError, match=column):
        assert_no_personal_data(df)


def test_personal_data_is_never_allowed_in_the_grid():
    assert not PERSONAL_DATA_COLUMNS & GRID_COLUMNS


def test_columns_outside_grid_names_only_the_extras():
    assert columns_outside_grid([*grid().columns, "hour"]) == []
    assert columns_outside_grid([*grid().columns, "number", "zz"]) == ["number", "zz"]


def test_the_dashboard_and_staging_use_these_definitions():
    """The lists used to be written out three times; nothing may redefine them."""
    from ML_Pipeline.cli import stage_demo
    from ML_Pipeline.dashboard import common
    from ML_Pipeline.data import prep_advanced

    assert prep_advanced.PERSONAL_DATA_COLUMNS is PERSONAL_DATA_COLUMNS
    assert common.assert_no_personal_data is assert_no_personal_data
    assert stage_demo.columns_outside_grid is columns_outside_grid
