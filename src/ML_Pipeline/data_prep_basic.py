"""Basic cleaning: deduplicate, coerce types, derive time fields."""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from ML_Pipeline.features import add_calendar_features
from ML_Pipeline.shift_time import shift_time
from ML_Pipeline.utils import convert_into_datetime, convert_into_numeric, remove_duplicates

logger = logging.getLogger(__name__)

#: Sentinel used in the source data for an unidentified rider.
UNKNOWN_RIDER = -1


def data_prep_basic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean booking-level data and derive per-rider booking gaps.

    Steps:
      1. Drop duplicate `(ts, number)` pairs - one rider cannot have two
         distinct requests at the same second.
      2. Coerce `number` and `ts`, dropping rows that fail to parse.
      3. Drop the `-1` rider sentinel, which aggregates many unidentified
         riders into one pseudo-rider and would corrupt the per-rider
         deduplication rules downstream.
      4. Add calendar features and the epoch-second booking timestamp.
      5. Add per-rider gaps between consecutive bookings.

    Returns:
        Cleaned copy, sorted by rider and timestamp.
    """
    started = datetime.now()
    required = {"ts", "number"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"data_prep_basic missing required column(s): {missing}")

    initial = len(df)
    out = remove_duplicates(df, ["ts", "number"])
    logger.info("Dropped %d duplicate (ts, number) rows", initial - len(out))

    out = convert_into_numeric(out, "number")
    out = out.dropna(subset=["number"]).reset_index(drop=True)
    out["number"] = out["number"].astype("int64")

    before = len(out)
    out = out.loc[out["number"] != UNKNOWN_RIDER].reset_index(drop=True)
    logger.info("Dropped %d rows with the unknown-rider sentinel", before - len(out))

    out = convert_into_datetime(out, "ts")
    before = len(out)
    out = out.dropna(subset=["ts"]).reset_index(drop=True)
    logger.info("Dropped %d rows with unparseable timestamps", before - len(out))

    out = add_calendar_features(out, "ts")
    out = out.sort_values(["number", "ts"]).reset_index(drop=True)
    out["booking_timestamp"] = out["ts"].astype("int64") // 10**9
    out = shift_time(out)

    logger.info(
        "Basic preprocessing: %d -> %d rows in %s",
        initial, len(out), datetime.now() - started,
    )
    return out
