"""
Business-rule cleaning of booking-level data.

The rules encode a domain assumption: a logged booking request is not always a
distinct unit of demand. A rider who rebooks after a long wait, a cancelled
driver, or a mistyped drop pin generates several rows for one real trip
intention, and counting them all inflates demand exactly where it is already
highest.

Each rule below is stated with the threshold it uses, because the previous
version's comments and code disagreed - one said "within 4mins" over a filter of
`>= 8` minutes, another said "remove ... > 500kms" for a rule that removes rides
outside Karnataka *and* over 500 km.

All filters are applied to explicit copies rather than to chained boolean masks,
which previously produced `SettingWithCopyWarning` territory: `advanced_cleanup`
assigned a new column onto an unmarked slice and called `reset_index(inplace=True)`
on it. That works today and is one pandas release from not working.
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from ML_Pipeline.utils import haversine_km

logger = logging.getLogger(__name__)

# Geographic bounding boxes, (min_lat, max_lat, min_lng, max_lng).
INDIA_BBOX = (6.2325274, 35.6745457, 68.1113787, 97.395561)
KARNATAKA_BBOX = (11.5945587, 18.4767308, 74.0543908, 78.588083)

MIN_TRIP_DISTANCE_KM = 0.05      # 50 m: pickup and drop effectively identical
MAX_PLAUSIBLE_TRIP_KM = 500.0    # beyond this a bike-taxi trip is not credible
REBOOK_SAME_LOCATION_HOURS = 1   # same rider, same pickup pin, within an hour
MIN_MINUTES_BETWEEN_BOOKINGS = 8 # shorter gaps read as retries, not new demand


def _outside(df: pd.DataFrame, bbox: tuple[float, float, float, float]) -> pd.Series:
    """Boolean mask: True where pickup or drop falls outside `bbox`."""
    min_lat, max_lat, min_lng, max_lng = bbox
    return (
        (df.pick_lat <= min_lat) | (df.pick_lat >= max_lat)
        | (df.pick_lng <= min_lng) | (df.pick_lng >= max_lng)
        | (df.drop_lat <= min_lat) | (df.drop_lat >= max_lat)
        | (df.drop_lng <= min_lng) | (df.drop_lng >= max_lng)
    )


def advanced_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply business-rule filters to booking-level data.

    Requires `booking_time_diff_hr` / `booking_time_diff_min` from
    `shift_time`, and pickup/drop coordinates.

    Returns:
        Cleaned copy with a `geodesic_distance` column (km).
    """
    started = datetime.now()
    required = {
        "number", "pick_lat", "pick_lng", "drop_lat", "drop_lng",
        "booking_time_diff_hr", "booking_time_diff_min",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"advanced_cleanup missing required column(s): {missing}")

    out = df.copy()
    initial = len(out)

    # Rule 1: same rider rebooking the same pickup pin within an hour.
    repeat = out.duplicated(subset=["number", "pick_lat", "pick_lng"], keep=False) & (
        out.booking_time_diff_hr <= REBOOK_SAME_LOCATION_HOURS
    )
    out = out.loc[~repeat].copy()
    logger.info("Rule 1 (rebooking same pin within %dh): dropped %d rows",
                REBOOK_SAME_LOCATION_HOURS, initial - len(out))

    # Rule 2: retries. A rider's *first* booking has no previous timestamp; the
    # upstream fill of 0 makes its diff enormous, so it survives this filter.
    # That is intended, but it is an accidental sentinel rather than a designed
    # one - see shift_time, which now marks first bookings explicitly.
    before = len(out)
    if "is_first_booking" in out.columns:
        keep = out.is_first_booking | (out.booking_time_diff_min >= MIN_MINUTES_BETWEEN_BOOKINGS)
    else:
        keep = out.booking_time_diff_min >= MIN_MINUTES_BETWEEN_BOOKINGS
    out = out.loc[keep].copy()
    logger.info("Rule 2 (bookings <%d min apart): dropped %d rows",
                MIN_MINUTES_BETWEEN_BOOKINGS, before - len(out))

    # Distance, vectorised over the whole frame.
    out["geodesic_distance"] = np.round(
        haversine_km(out.pick_lat, out.pick_lng, out.drop_lat, out.drop_lng), 2
    )

    # Rule 3: pickup and drop effectively the same place.
    before = len(out)
    out = out.loc[out.geodesic_distance > MIN_TRIP_DISTANCE_KM].copy()
    logger.info("Rule 3 (trip shorter than %.0f m): dropped %d rows",
                MIN_TRIP_DISTANCE_KM * 1000, before - len(out))

    # Rule 4: coordinates outside India - data errors, not trips.
    before = len(out)
    out = out.loc[~_outside(out, INDIA_BBOX)].copy()
    logger.info("Rule 4 (outside India bounding box): dropped %d rows", before - len(out))

    # Rule 5: outside Karnataka AND implausibly long. Either alone is allowed:
    # a legitimate trip may cross the state line, and a long trip within the
    # state may be genuine. Only the combination is treated as bad data.
    before = len(out)
    suspect = _outside(out, KARNATAKA_BBOX) & (out.geodesic_distance > MAX_PLAUSIBLE_TRIP_KM)
    out = out.loc[~suspect].copy()
    logger.info("Rule 5 (outside Karnataka and >%.0f km): dropped %d rows",
                MAX_PLAUSIBLE_TRIP_KM, before - len(out))

    out = out.reset_index(drop=True)
    logger.info(
        "advanced_cleanup: %d -> %d rows (%.2f%% removed) in %s",
        initial, len(out), (1 - len(out) / max(initial, 1)) * 100,
        datetime.now() - started,
    )
    return out
