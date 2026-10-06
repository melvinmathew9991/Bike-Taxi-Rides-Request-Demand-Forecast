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

#: The area the model is for. The model card has always scoped it to Bangalore
#: and called other areas out of scope, but nothing enforced it: 4.05% of cleaned
#: pickups (156,689) lay hundreds of km away - Hyderabad, Mysuru, Chennai,
#: Odisha, Rajasthan - and took 8 of the 50 clusters, each spanning cities. The
#: edges are generous and sit in near-empty ground: 51 pickups fall within ~30 km
#: outside the box and 5-8 in each 0.05-degree strip inside its edges, so no
#: suburb is cut. Kempegowda airport (13.20, 77.71) is inside.
BENGALURU_BBOX = (12.70, 13.30, 77.30, 77.90)

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


def advanced_cleanup(
    df: pd.DataFrame,
    *,
    service_area: tuple[float, float, float, float] | None = BENGALURU_BBOX,
) -> pd.DataFrame:
    """
    Apply business-rule filters to booking-level data.

    Requires the gap columns from `shift_time` - `booking_time_diff_min` for
    Rule 2 and `pin_time_diff_min` for Rule 1 - and pickup/drop coordinates.

    Args:
        service_area: Bounding box pickups must fall in (Rule 6). None keeps
            every pickup that passes Rules 1-5.

    Returns:
        Cleaned copy with a `geodesic_distance` column (km).
    """
    started = datetime.now()
    required = {
        "number", "pick_lat", "pick_lng", "drop_lat", "drop_lng",
        "booking_time_diff_hr", "booking_time_diff_min",
        # Rule 1 needs the per-pin gap, not the per-rider one. See below.
        "pin_time_diff_min",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"advanced_cleanup missing required column(s): {missing}")

    out = df.copy()
    initial = len(out)

    # Rule 1: same rider rebooking the same pickup pin within an hour.
    #
    # This is the gap to that rider's previous booking FROM THAT PIN, which is
    # what the rule describes. The previous implementation used
    #
    #     duplicated(subset=["number", "pick_lat", "pick_lng"], keep=False)
    #         & (booking_time_diff_hr <= 1)
    #
    # which is a different rule. `duplicated(keep=False)` is true for every row
    # whose (rider, pin) recurs anywhere in the dataset - 61.3% of rows on the
    # reference data - and `booking_time_diff_hr` is the gap to the previous
    # booking from ANY pin. So a row was deleted whenever a commuter had ever
    # used that pin twice and happened to book something else in the preceding
    # hour, and `keep=False` deleted the first of each group too.
    #
    # Measured on the reference dataset: the old mask dropped 3,979,406 rows
    # (47.9%) against 3,430,896 (41.3%) for the rule as described - 548,510 rows
    # of real demand, 11.7% of which were a rider's first-ever booking at that
    # pin and so could not have been rebookings.
    #
    # A first booking at a pin has no predecessor there, so its gap is NaN and
    # the comparison is False: it is kept, by definition rather than by sentinel.
    #
    # The comparison is in MINUTES. `pin_time_diff_hr` is floored, so a
    # threshold of `<= 1` hour there actually spans anything under two hours -
    # 61 minutes floors to 1. The old rule had the same flaw on
    # `booking_time_diff_hr`, which is a further reason it over-dropped: "within
    # an hour" was really "within two".
    if "pin_time_diff_min" not in out.columns:
        raise KeyError(
            "advanced_cleanup needs 'pin_time_diff_min' (minutes since the "
            "rider's previous booking from the same pickup pin), produced by "
            "shift_time(). Without it Rule 1 cannot be evaluated as specified."
        )
    repeat = out["pin_time_diff_min"] <= REBOOK_SAME_LOCATION_HOURS * 60
    out = out.loc[~repeat.fillna(False)].copy()
    logger.info("Rule 1 (rebooking same pin within %d min): dropped %d rows",
                REBOOK_SAME_LOCATION_HOURS * 60, initial - len(out))

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

    # Rule 6: pickup outside the service area. Unlike Rules 4 and 5 these are
    # real bookings, not bad data - they are demand somewhere this model does not
    # cover. Pickup only: demand is counted where the ride starts, so a trip
    # from Bengaluru to anywhere is Bengaluru demand.
    if service_area is not None:
        before = len(out)
        min_lat, max_lat, min_lng, max_lng = service_area
        inside = out.pick_lat.between(min_lat, max_lat) & out.pick_lng.between(
            min_lng, max_lng
        )
        out = out.loc[inside].copy()
        logger.info("Rule 6 (pickup outside the service area %s): dropped %d rows",
                    service_area, before - len(out))

    out = out.reset_index(drop=True)
    logger.info(
        "advanced_cleanup: %d -> %d rows (%.2f%% removed) in %s",
        initial, len(out), (1 - len(out) / max(initial, 1)) * 100,
        datetime.now() - started,
    )
    return out
