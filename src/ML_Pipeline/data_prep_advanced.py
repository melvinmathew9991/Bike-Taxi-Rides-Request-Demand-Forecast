"""
Advanced cleaning stage: apply business rules and persist the cleaned data.

Data governance
---------------
The frame written here is still **booking-level personal data**: a pseudonymous
rider identifier joined to pickup and drop coordinates at roughly 0.1 m
precision. Under India's DPDP Act 2023 that is personal data, and precise
location traces are among its most sensitive forms - home and workplace are
directly inferable from a rider's repeated pickups.

`columns` is therefore an explicit allow-list rather than "everything we
happen to have", and `drop_rider_id` lets a caller write the cleaned dataset
without the identifier when downstream stages do not need it. The aggregation in
`data_prep_geospatial` is what actually removes the personal data: from that
point on the pipeline handles only counts per region per half hour.

See docs/DATA_GOVERNANCE.md.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from ML_Pipeline.advanced_cleanup import advanced_cleanup

logger = logging.getLogger(__name__)

#: Columns carried forward from cleaning. Explicit, so a new upstream column
#: cannot silently start being written to disk.
CLEANED_COLUMNS: tuple[str, ...] = (
    "ts", "number", "pick_lat", "pick_lng", "drop_lat", "drop_lng",
    "geodesic_distance", "hour", "mins", "month", "quarter", "dayofweek",
    "booking_timestamp", "booking_time_diff_hr", "booking_time_diff_min",
)

#: Personal-data columns, called out so the governance boundary is visible here
#: rather than only in a document.
PERSONAL_DATA_COLUMNS: frozenset[str] = frozenset(
    {"number", "pick_lat", "pick_lng", "drop_lat", "drop_lng"}
)


def data_prep_advanced(
    df: pd.DataFrame,
    path: str | Path,
    *,
    columns: tuple[str, ...] = CLEANED_COLUMNS,
    drop_rider_id: bool = False,
) -> pd.DataFrame:
    """
    Apply business-rule cleaning and write the result.

    Args:
        df: Output of `data_prep_basic`.
        path: Destination for the cleaned data (gzip CSV).
        columns: Allow-list of columns to retain.
        drop_rider_id: Omit the rider identifier from the written file. The
            identifier is needed by the cleaning rules but not by anything
            downstream, so dropping it narrows what is persisted.

    Returns:
        The cleaned frame.
    """
    started = datetime.now()
    cleaned = advanced_cleanup(df)

    keep = [c for c in columns if c in cleaned.columns]
    absent = sorted(set(columns) - set(keep))
    if absent:
        logger.warning("Requested column(s) not present after cleaning: %s", absent)
    if drop_rider_id:
        keep = [c for c in keep if c != "number"]

    dataset = cleaned.loc[:, keep].copy()

    retained_personal = sorted(PERSONAL_DATA_COLUMNS.intersection(dataset.columns))
    if retained_personal:
        logger.warning(
            "Writing booking-level personal data to %s (columns: %s). This file "
            "is git-ignored and must be handled per docs/DATA_GOVERNANCE.md.",
            path, ", ".join(retained_personal),
        )

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(path, index=False, compression="gzip")
    logger.info(
        "Advanced preprocessing: %d rows, %d columns written to %s in %s",
        len(dataset), len(dataset.columns), path, datetime.now() - started,
    )
    return dataset
