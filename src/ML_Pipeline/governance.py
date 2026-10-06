"""
Where personal data stops, as code.

The booking-level tables carry a pseudonymous rider identifier (`number`)
joined to pickup and drop coordinates at ~0.1 m precision, from which home and
workplace are inferable. The demand grid - counts per cluster per interval -
carries none of it. docs/DATA_GOVERNANCE.md describes that boundary; this
module is the one place the code defines it. Three controls read it:

- the cleaning stage warns when it writes personal fields to disk
  (`ML_Pipeline.data.prep_advanced`);
- the dashboard refuses to render a file that carries them
  (`assert_no_personal_data`);
- demo staging refuses to upload a grid with any column outside the aggregated
  set (`ML_Pipeline.cli.stage_demo`).

These lists used to be written out separately in each of the three places.
"""

from __future__ import annotations

import pandas as pd

#: Booking-level columns that are personal data. A frame carrying any of them
#: is not an aggregated grid.
PERSONAL_DATA_COLUMNS: frozenset[str] = frozenset(
    {"number", "pick_lat", "pick_lng", "drop_lat", "drop_lng"}
)

#: Every column the aggregated demand grid may carry. Used as an allow-list
#: wherever data leaves the machine.
GRID_COLUMNS: frozenset[str] = frozenset(
    {"ts", "pickup_cluster", "request_count", "mins", "hour", "month", "quarter",
     "dayofweek"}
)


class DataGovernanceError(RuntimeError):
    """Raised when a file would expose personal data where it must not go."""


def assert_no_personal_data(df: pd.DataFrame) -> None:
    """
    Refuse to proceed if the frame carries booking-level personal data.

    Raises:
        DataGovernanceError: if any personal-data column is present.
    """
    present = sorted(PERSONAL_DATA_COLUMNS.intersection(df.columns))
    if present:
        raise DataGovernanceError(
            "Refusing to display this file: it contains booking-level personal "
            f"data ({', '.join(present)}). This dashboard renders only the "
            "aggregated demand grid. Point it at output/Data_Prepared_<version>.csv.gz."
        )


def columns_outside_grid(columns) -> list[str]:
    """Columns that are not in the aggregated grid's allow-list, sorted."""
    return sorted(set(columns) - GRID_COLUMNS)
