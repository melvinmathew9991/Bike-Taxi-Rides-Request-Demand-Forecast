"""
The input contract for booking files, checked rather than described.

`docs/DATA_SCHEMA.md` describes the booking input, but nothing checked it. A
file that broke the description did not fail. It was cleaned into something
smaller and the run carried on. A `ts` column in another date format became
`NaT` and was dropped row by row. Swapped `pick_lat` and `pick_lng` put every
pickup outside Bengaluru, and Rule 6 dropped them all. Either way the run went
on to train on whatever was left.

This module checks the file when it is loaded, before any cleaning. It tells
two kinds of problem apart:

* **a broken file** - a column missing, no rows, more than `MAX_BAD_SHARE` of
  rows failing a rule, or most pickups outside the service area. The run stops
  and names the problem.
* **a few bad rows** - which real data has (121 unparseable rider ids and 252
  unknown-rider sentinels in 8.4 million on the reference dataset). These are
  logged, and cleaning drops them as it always has.

The checks are written by hand rather than with pandera. The rules here are
shares of rows against a threshold, which pandera expresses awkwardly, and six
columns did not justify the extra dependencies.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd

from ML_Pipeline.data.cleaning_rules import BENGALURU_BBOX

logger = logging.getLogger(__name__)

#: Format of `ts` in the source data, e.g. `2020-03-26 07:07:17`.
TS_FORMAT = "%Y-%m-%d %H:%M:%S"

#: Above this share of rows failing any one rule, the file is treated as broken
#: rather than as containing a few bad rows. The reference dataset's worst rule
#: fails 0.0014% of rows, so 1% leaves a wide margin.
MAX_BAD_SHARE = 0.01

#: Above this share of pickups outside the service area, the file is treated as
#: the wrong area or as having lat/lng swapped. 2.8% of the reference raw
#: file's pickups lie outside Bengaluru, and Rule 6 drops them.
MAX_OUTSIDE_SERVICE_AREA = 0.5

COORDINATE_COLUMNS = ("pick_lat", "pick_lng", "drop_lat", "drop_lng")

#: Columns every booking file must have, and those training also needs.
FORECAST_COLUMNS = ("ts", "pick_lat", "pick_lng")
TRAINING_COLUMNS = ("ts", "number", "pick_lat", "pick_lng", "drop_lat", "drop_lng")


class InputContractError(ValueError):
    """A booking file breaks the input contract badly enough to stop the run."""


class MissingColumnsError(InputContractError, KeyError):
    """Required columns are absent. Also a KeyError, as the old checks raised."""

    def __str__(self) -> str:  # KeyError would otherwise quote the message
        return str(self.args[0]) if self.args else ""


@dataclass
class ContractReport:
    """What was checked, and how many rows broke each rule."""

    source: str
    rows: int
    bad_rows: dict[str, int] = field(default_factory=dict)
    outside_service_area: float | None = None

    def share(self, rule: str) -> float:
        return self.bad_rows.get(rule, 0) / self.rows if self.rows else 0.0


def _bad_rows(df: pd.DataFrame, required: tuple[str, ...]) -> dict[str, int]:
    """Rows failing each rule, for the required columns only."""
    bad: dict[str, int] = {}
    if "ts" in required:
        parsed = pd.to_datetime(df["ts"], format=TS_FORMAT, errors="coerce")
        bad[f"ts not in {TS_FORMAT} format"] = int(parsed.isna().sum())
    if "number" in required:
        numeric = pd.to_numeric(df["number"], errors="coerce")
        bad["number not numeric"] = int(numeric.isna().sum())
    for col in COORDINATE_COLUMNS:
        if col not in required:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        limit = 90.0 if col.endswith("_lat") else 180.0
        bad[f"{col} missing or outside +/-{limit:g}"] = int(
            (values.isna() | (values.abs() > limit)).sum()
        )
    return bad


def _outside_share(
    df: pd.DataFrame, area: tuple[float, float, float, float]
) -> float:
    min_lat, max_lat, min_lng, max_lng = area
    lat = pd.to_numeric(df["pick_lat"], errors="coerce")
    lng = pd.to_numeric(df["pick_lng"], errors="coerce")
    inside = lat.between(min_lat, max_lat) & lng.between(min_lng, max_lng)
    return float(1.0 - inside.mean())


def check_bookings(
    df: pd.DataFrame,
    *,
    source: str,
    required: Iterable[str] = TRAINING_COLUMNS,
    service_area: tuple[float, float, float, float] | None = BENGALURU_BBOX,
) -> ContractReport:
    """
    Check a booking file against the input contract.

    Args:
        df: The file as loaded, before any cleaning.
        source: Its path or name, for messages.
        required: Columns it must have. `TRAINING_COLUMNS` for raw training
            data; `FORECAST_COLUMNS` for a file that is only forecast from.
        service_area: `(min_lat, max_lat, min_lng, max_lng)` most pickups must
            fall in. None skips the check.

    Returns:
        The report, for logging. A few bad rows are logged as a warning.

    Raises:
        MissingColumnsError: a required column is absent.
        InputContractError: no rows, too many bad rows on any rule, or most
            pickups outside the service area.
    """
    required = tuple(required)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise MissingColumnsError(
            f"{source} is missing required column(s): {missing}. Found: "
            f"{list(df.columns)}. See docs/DATA_SCHEMA.md."
        )
    if len(df) == 0:
        raise InputContractError(f"{source} has no rows.")

    report = ContractReport(source=source, rows=len(df), bad_rows=_bad_rows(df, required))

    broken = {
        rule: n for rule, n in report.bad_rows.items() if report.share(rule) > MAX_BAD_SHARE
    }
    if broken:
        detail = "; ".join(
            f"{rule}: {n:,} of {report.rows:,} rows ({n / report.rows:.1%})"
            for rule, n in broken.items()
        )
        raise InputContractError(
            f"{source} breaks the input contract - more than "
            f"{MAX_BAD_SHARE:.0%} of rows fail a rule, which is a malformed "
            f"file rather than a few bad rows. {detail}. See docs/DATA_SCHEMA.md."
        )

    if service_area is not None:
        report.outside_service_area = _outside_share(df, service_area)
        if report.outside_service_area > MAX_OUTSIDE_SERVICE_AREA:
            raise InputContractError(
                f"{report.outside_service_area:.0%} of pickups in {source} fall "
                f"outside the service area {service_area} (min_lat, max_lat, "
                "min_lng, max_lng). Either the file is for another area, or "
                "pick_lat and pick_lng are swapped. Cleaning would drop them "
                "and train on what is left."
            )

    flagged = {rule: n for rule, n in report.bad_rows.items() if n}
    if flagged:
        logger.warning(
            "%s: a few rows break the input contract; cleaning drops them. %s",
            source, ", ".join(f"{rule}: {n:,}" for rule, n in flagged.items()),
        )
    logger.info(
        "%s meets the input contract: %s rows%s.", source, f"{report.rows:,}",
        "" if report.outside_service_area is None
        else f", {report.outside_service_area:.1%} of pickups outside the service area",
    )
    return report
